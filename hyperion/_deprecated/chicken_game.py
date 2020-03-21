
import sys
sys.path.append("C:/git/reinforcement-learning")


from hyperion.agents import *
from hyperion.environment import *

import random
import numpy as np
import uuid
import attr


STATUSES = ["EGG","CHICKEN","COW","FARMER","SUPERMAN"]
SIZE = 100


@attr.s(slots = True)
class Player(Agent):

    # # Agent id
    # id = attr.ib()
    # id.default
    # def _init_id(self):
    #     return str(uuid.uuid1())

    # Status
    status = attr.ib(default = 0,init=False)

    # Position
    x = attr.ib(init = False)
    @x.default
    def _init_x(self):
        return random.randint(0,SIZE)


    def step(self,env):

        # Movement
        new_x = self.x + random.choice([-1,1])
        new_x = np.clip(new_x,0,SIZE-1)
        self.x = new_x

        # Others
        others = env.inverse_loc(self.id)
        for other in others:
            if other.x == self.x:
                if other.status == self.status:
                    other.status = 0
                    self.status += 1

    def interacts_with(self,other):
        return self.x == other.x,1


class ChickenGame(Environment):

    def render(self):
        env = [" "]*SIZE
        for agent in self.agents:
            env[agent.x] = str(agent.status)
        return "|"+"".join(env)+"|"



    def interactions(self):
        pass



