
import numpy as np
import pygame

from ..agent import Agent


class GridAgent(Rectangle):
    def __init__(self,x,y,width = 1,height = 1):
        super().__init__(x,y,width,height)

    #=================================================================================
    # MOVEMENTS
    #=================================================================================

    def move(self,dx = 0,dy = 0,angle = None,dl = None,env = None):

        if angle is not None:
            pass
        else:
            self.x += dx
            self.y += dy


    def random_walk(self,env = None):

        # Sample a random move between -1, 0 or + 1
        # ie 8 moves around agent's position
        dx,dy = np.random.randint(0,3,2) - 1
        self.move(dx,dy,env = env)



