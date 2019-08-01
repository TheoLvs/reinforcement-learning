import numpy as np
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection


def _is_in_box(x, y, box):
    # Get box coordinates
    x_left, x_right, y_bottom, y_top = box
    return x_left <= x <= x_right and y_bottom <= y <= y_top


class DeliveryEnvironment(object):
    def __init__(self, n_stops=10, max_box=10, method="distance", **kwargs):

        print(f"Initialized Delivery Environment with {n_stops} random stops")
        print(f"Target metric for optimization is {method}")

        # Initialization
        self.n_stops = n_stops
        self.action_space = self.n_stops
        self.observation_space = self.n_stops
        self.max_box = max_box
        self.stops = []
        self.method = method

        # Generate stops
        self._generate_constraints(**kwargs)
        self._generate_stops()
        self._generate_q_values()
        # self.render()

        # Initialize first point
        self.reset()

    def _generate_constraints(self, box_size=0.2, traffic_intensity=5):

        if self.method == "traffic_box":

            x_left = np.random.rand() * self.max_box * (1 - box_size)
            y_bottom = np.random.rand() * self.max_box * (1 - box_size)

            x_right = x_left + np.random.rand() * box_size * self.max_box
            y_top = y_bottom + np.random.rand() * box_size * self.max_box

            self.box = (x_left, x_right, y_bottom, y_top)
            self.traffic_intensity = traffic_intensity

    def _generate_stops(self):

        if self.method == "traffic_box":

            points = []
            while len(points) < self.n_stops:
                x, y = np.random.rand(2) * self.max_box
                if not _is_in_box(x, y, self.box):
                    points.append((x, y))

            xy = np.array(points)

        else:
            # Generate geographical coordinates
            xy = np.random.rand(self.n_stops, 2) * self.max_box

        self.x = xy[:, 0]
        self.y = xy[:, 1]

    def _generate_q_values(self):

        # Generate actual Q Values corresponding to time elapsed between two points
        if self.method in ["distance", "traffic_box"]:
            xy = np.column_stack([self.x, self.y])
            self.q_stops = cdist(xy, xy)
        elif self.method == "time":
            self.q_stops = np.random.rand(self.n_stops, self.n_stops) * self.max_box
            np.fill_diagonal(self.q_stops, 0)
        else:
            raise Exception("Method not recognized")

    def render(self, return_img=False):

        fig = plt.figure(figsize=(7, 7))
        ax = fig.add_subplot(111)
        ax.set_title("Delivery Stops")

        # Show stops
        ax.scatter(self.x, self.y, c="red", s=50)

        # Show START
        if len(self.stops) > 0:
            xy = self._get_xy(initial=True)
            xytext = xy[0] + 0.1, xy[1] - 0.05
            ax.annotate("START", xy=xy, xytext=xytext, weight="bold")

        # Show itinerary
        if len(self.stops) > 1:
            ax.plot(self.x[self.stops], self.y[self.stops], c="blue", linewidth=1, linestyle="--")

            # Annotate END
            xy = self._get_xy(initial=False)
            xytext = xy[0] + 0.1, xy[1] - 0.05
            ax.annotate("END", xy=xy, xytext=xytext, weight="bold")

        if hasattr(self, "box"):
            left, bottom = self.box[0], self.box[2]
            width = self.box[1] - self.box[0]
            height = self.box[3] - self.box[2]
            rect = Rectangle((left, bottom), width, height)
            collection = PatchCollection([rect], facecolor="red", alpha=0.2)
            ax.add_collection(collection)

        plt.xticks([])
        plt.yticks([])

        if return_img:
            fig.canvas.draw_idle()
            image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
            image = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
            plt.close()
            return image
        else:
            plt.show()

    def reset(self):

        # Stops placeholder
        self.stops = []

        # Random first stop
        first_stop = np.random.randint(self.n_stops)
        self.stops.append(first_stop)

        return first_stop

    def step(self, destination):

        # Get current state
        state = self._get_state()
        new_state = destination

        # Get reward for such a move
        reward = self._get_reward(state, new_state)

        # Append new_state to stops
        self.stops.append(destination)
        done = len(self.stops) == self.n_stops

        return new_state, reward, done

    def _get_state(self):
        return self.stops[-1]

    def _get_xy(self, initial=False):
        state = self.stops[0] if initial else self._get_state()
        x = self.x[state]
        y = self.y[state]
        return x, y

    def _get_reward(self, state, new_state):
        base_reward = self.q_stops[state, new_state]

        if self.method == "distance":
            return base_reward
        elif self.method == "time":
            return base_reward + np.random.randn()
        elif self.method == "traffic_box":

            # Additional reward correspond to slowing down in traffic
            xs, ys = self.x[state], self.y[state]
            xe, ye = self.x[new_state], self.y[new_state]
            intersections = DeliveryEnvironment._calculate_box_intersection(xs, xe, ys, ye, self.box)
            if len(intersections) > 0:
                i1, i2 = intersections
                distance_traffic = np.sqrt((i2[1] - i1[1])**2 + (i2[0] - i1[0])**2)
                additional_reward = distance_traffic * self.traffic_intensity * np.random.rand()
            else:
                additional_reward = np.random.rand()

            return base_reward + additional_reward

    @staticmethod
    def _calculate_point(x1, x2, y1, y2, x=None, y=None):

        if y1 == y2:
            return y1
        elif x1 == x2:
            return x1
        else:
            a = (y2 - y1) / (x2 - x1)
            b = y2 - a * x2

            if x is None:
                x = (y - b) / a
                return x
            elif y is None:
                y = a * x + b
                return y
            else:
                raise Exception("Provide x or y")

    @staticmethod
    def _calculate_box_intersection(x1, x2, y1, y2, box):

        # Get box coordinates
        x_left, x_right, y_bottom, y_top = box

        # Intersections
        intersections = []

        # Top intersection
        i_top = DeliveryEnvironment._calculate_point(x1, x2, y1, y2, y=y_top)
        if x_left < i_top < x_right:
            intersections.append((i_top, y_top))

        # Bottom intersection
        i_bottom = DeliveryEnvironment._calculate_point(x1, x2, y1, y2, y=y_bottom)
        if x_left < i_bottom < x_right:
            intersections.append((i_bottom, y_bottom))

        # Left intersection
        i_left = DeliveryEnvironment._calculate_point(x1, x2, y1, y2, x=x_left)
        if y_bottom < i_left < y_top:
            intersections.append((x_left, i_left))

        # Right intersection
        i_right = DeliveryEnvironment._calculate_point(x1, x2, y1, y2, x=x_right)
        if y_bottom < i_right < y_top:
            intersections.append((x_right, i_right))

        return intersections
