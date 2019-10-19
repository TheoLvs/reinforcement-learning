
# Base libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import uuid
import imageio
from tqdm import tqdm_notebook
from collections import defaultdict

# Interaction & animation
from ipywidgets import widgets
from IPython.display import display
from ipywidgets import interact,IntSlider,Text


def action(duration,delay = 0,periods = 1,loop = False):
    t = 0

    while True:
        if t%(duration+delay) < delay:
            start,end = False,False
            yield start,end
        else:
            start,end = True,False
            yield start,end
        t += 1

        if not loop:
            if t//(duration+delay) >= periods:
                break

    start,end = True,True
    yield start,end



#======================================================================================
# ENVIRONMENT CLASSES
#======================================================================================


class Environment:
    def __init__(self):

        SCHEMA = ["agent_id","agent_type","agent"]
        self._data = pd.DataFrame(columns = SCHEMA).set_index("agent_id")


    @property
    def agents(self):
        return self._data["agent"].tolist()

    @property
    def data(self):
        return self._data.drop(columns = "agent")
    
    def __getitem__(self,key):
        if isinstance(key,int):
            return self._data.iloc[key].loc["agent"]
        else:
            return self.get_agent(key)

    def __iter__(self):
        return iter(self.agents)


    def _repr_html_(self):
        return self.data.head(20)._repr_html_()

    def add_agent(self,agent,agent_data):
        agent_data["agent"] = agent
        agent_data = pd.DataFrame([agent_data]).set_index("agent_id")
        self._data = self._data.append(agent_data,verify_integrity = True,sort = False)

    def remove_agent(self,agent_id):
        self._data.drop(agent_id,inplace = True)

    def get_agent(self,agent_id):
        return self._data.loc[agent_id,"agent"]

    def step(self,fast = False):
        """Discrete step function 
        """

        if not fast:

            # Initialize reward at 0
            reward = 0

            # Loop over each agent
            for agent in self.agents:
                reward_agent = agent.step()
                reward += reward_agent

            # Compute if episode is finished
            done = len(self.agents) == 0

            return reward,done

        else:
            reward = self._data["agent"].map(lambda x : x.step()).sum()


    def run(self,n,render = True):
        """Run episode function
        """

        experiment = Experiment()

        # Loop each step in the episode
        for i in tqdm_notebook(range(n)):

            # Compute reward and if episode is finished
            reward,done = self.step()

            # If episode is finished stop
            if done:
                break

            # Otherwise append to reward and save image
            else:

                data = {"reward":reward}

                if render:
                    fig = self.render(return_fig = True)
                    data["fig"] = fig

                experiment.log(data)


        return experiment


def fig_to_img(fig):
    fig.canvas.draw_idle()
    image = np.frombuffer(fig.canvas.tostring_rgb(), dtype='uint8')
    image  = image.reshape(fig.canvas.get_width_height()[::-1] + (3,))
    plt.close()
    return image


class Environment2D(Environment):
    def __init__(self,bounds = None):
        """2D Environment
        bounds = (xmin,xmax,ymin,ymax)
        Add something like an occlusion zone, or at least somewhere where agents can't go
        """
        super().__init__()
        self.bounds = bounds


    @property
    def xmin(self):
        if self.bounds is not None:
            return self.bounds[0]

    @property
    def xmax(self):
        if self.bounds is not None:
            return self.bounds[1]
    
    @property
    def ymin(self):
        if self.bounds is not None:
            return self.bounds[2]

    @property
    def ymax(self):
        if self.bounds is not None:
            return self.bounds[3]



    def render(self,return_img = False,return_fig = False):

        # Create figure
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(111)

        # Set bounds if needed
        if self.bounds is not None:
            ax.set_xlim([self.xmin,self.xmax])
            ax.set_ylim([self.ymin,self.ymax])

        # Plot scatter plot
        self.data[["x","y"]].plot(kind = "scatter",x="x",y="y",ax = ax)

        # Return image for animation
        if return_img:
            # From https://ndres.me/post/matplotlib-animated-gifs-easily/
            img = fig_to_img(fig)
            return img

        # Return figure
        if return_fig:
            plt.close()
            return fig


        

#======================================================================================
# AGENT CLASSES
#======================================================================================




class Agent:
    def __init__(self,env,agent_data = {}):

        # Base parameters
        self.agent_id = str(uuid.uuid1())
        self.agent_type = self.__class__.__name__
        self.internal_clock = 0

        # Base agent data
        agent_data = {
            "agent_id":self.agent_id,
            "agent_type":self.agent_type,
            **agent_data
        }

        # Prepare data as argument        
        self.env = env
        self.env.add_agent(self,agent_data)


    def step(self):
        self.internal_clock += 1


    def __getitem__(self,key):
        return self.get(key)

    def __repr__(self):
        return f"{self.agent_type}(id={self.agent_id})"

    def get_data(self):
        return self.env[self.agent_id].to_dict()

    def get(self,key):
        return self.env._data.loc[self.agent_id,key]

    def set(self,key,value):
        self.env._data.loc[self.agent_id,key] = value

    def add(self,key,value):
        self.env._data.loc[self.agent_id,key] += value

    def multiply(self,key,value):
        self.env._data.loc[self.agent_id,key] *= value

    def sub(self,key,value):
        self.env._data.loc[self.agent_id,key] -= value

    def divide(self,key,value):
        self.env._data.loc[self.agent_id,key] /= value



class Agent2D(Agent):


    def move(self,x = None,y = None,dx = 0,dy = 0,angle = None):
        """Move function for base agent

        TODO:
        - find a way to do allowed moves programmatically and dynamically
        - Subclass this function into a Agent2D
        """

        # Move if x and y are given
        if x is not None and y is not None:
            pass

        # Move if angle and velocity are given
        elif angle is not None:

            # Compute delta directions with basic trigonometry
            dx = self.velocity * np.cos(angle)
            dy = self.velocity * np.sin(angle)

            # Move towards other point
            self.move(dx = dx,dy = dy)

        # Move with delta directions
        else:
            self.add("x",dx)
            self.add("y",dy)


    def move_towards(self,x_target,y_target):

        # Find coords
        x,y = self.get("x"),self.get("y")

        # Compute direction with basic trigonometry
        angle = np.arctan2(y_target - y,x_target - x)

        # Move towards target
        self.move(angle = angle)


    def wander(self,pivot_frequency):

        t = 0
        angle = np.random.uniform(0,2*np.pi)          

        while True:

            dx = n

        pass





class StaticAgent(Agent):
    pass







#======================================================================================
# SUGARSCAPE CLASSES
#======================================================================================



class Rabbit(Agent2D):
    def __init__(self,env,**kwargs):

        # Prepare agent parameters
        agent_data = {
            "life":np.random.randint(100,200),
            "x":np.random.randint(0,10),
            "y":np.random.randint(0,10),
            "velocity":0.5,
        }
        agent_data["life_left"] = agent_data["life"]

        # Init
        super().__init__(env,agent_data)


    @property
    def velocity(self):
        return self["velocity"]
    

    def step(self):
        super().step()

        self.sub("life_left",1)

        # try:
        #     start,end = next(self.actions)
        #     if end:
        #         print(f"Recharge {self.agent_id}!")
        #         self.add("life_left",10)
        # except:
        #     pass

        # Aleatory move
        # dx,dy = np.random.randn() / 5,np.random.randn() / 5
        # self.move(dx = dx,dy = dy)

        self.move_towards(10,10)

        if self["life_left"] == 0:
            self.env.remove_agent(self.agent_id)
            return -1
        else:
            return 0



class Food(StaticAgent):
    def __init__(self):
        pass




class Sugarscape(Environment2D):
    pass



class Experiment:

    def __init__(self):

        self.data = defaultdict(list)


    @property
    def fig(self):
        return self.data["fig"]
    

    def log(self,data):
        for k,v in data.items():
            self.data[k].append(v)

    def save_as_gif(self,path):
        pass


        #         if save is not None:
        #             assert isinstance(save,str)
        #             img = self.show(return_img = True)
        #             imgs.append(img)

        # if save is not None:
        #     imageio.mimsave(save,imgs)

    def replay(self):

        play = widgets.Play(
            value=0,
            min=0,
            max=len(self.fig)-1,
            step=1,
            description="Press play",
            disabled=False
        )


        @interact(
            i = play,
            # path = Text(value='test.gif',placeholder='Type something',description='Path for gif:'),
        )
        def show(i):
            return self.fig[i]