
import numpy as np
import pygame

from .rectangle import Rectangle


class GridAgent(Rectangle):
    def __init__(self,x,y,width = 1,height = 1,box_size = None,color = (255,0,0)):
        super().__init__(x,y,width,height,box_size,color)

        self.change_direction()


    @property
    def static(self):
        return False

    def __repr__(self):
        return f"Agent(x={self.x},y={self.y})"


    #=================================================================================
    # MOVEMENTS
    #=================================================================================

    def change_direction(self):
        self.direction_angle = np.random.uniform(0,2*np.pi)

    def follow_direction(self,dr = 1,env = None):
        return self.move(angle = self.direction_angle,dr = dr,env = env)


    def move(self,dx = 0,dy = 0,angle = None,dr = None,env = None):

        # Store default value for collisions
        is_collision = False

        # Move using radial movement (with angle and radius)
        if angle is not None:

            box_size = 1 if env is None else env.box_size

            # Compute delta directions with basic trigonometry
            # In a grid environment, movement is rounded to the integer to fit in the grid
            dx = int(dr * box_size * np.cos(angle))
            dy = int(dr * box_size * np.sin(angle))

            return self.move(dx = dx,dy = dy,env = env)

        # Move using euclidean movement (with dx and dy)
        else:

            if env is None: 
                
                # If movement is not bounded by environment
                # We update position without constraints
                self.x += dx
                self.y += dy

            else:
                
                # Store movements
                x = self.x + dx
                y = self.y + dy

                # Correct moves going offscreen
                x,y = env.correct_offscreen_move(x,y)

                # Compute collisions
                collider = self.get_collider(x,y)
                is_collision,_ = self.collides_with(env.objects,collider = collider)

                if not is_collision:

                    # Store new positions as attributes
                    self.x = x
                    self.y = y

                    


        return is_collision




    def random_walk(self,env = None):

        # Sample a random move between -1, 0 or + 1
        # ie 8 moves around agent's position
        dx,dy = np.random.randint(0,3,2) - 1
        self.move(dx,dy,env = env)



