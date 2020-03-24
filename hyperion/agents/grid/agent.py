
import numpy as np
import pygame

from ..agent import Agent


class GridAgent(Agent):
    def __init__(self,x,y,width = 1,height = 1):

        super().__init__()

        self.x = x
        self.y = y
        self.width = width
        self.height = height


    @property
    def pos(self):
        return self.x,self.y


    def get_pos_px(self,box_size):
        return (
            self.x * box_size,
            self.y * box_size,
            self.width * box_size,
            self.height * box_size
        )


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




    #=================================================================================
    # RENDERERS
    #=================================================================================


    def render(self,env,color = (180,20,150)):

        # Get x,y,width,height in pixels from box size in the grid
        pos = self.get_pos_px(env.box_size)

        # Draw a rectangle on the grid using pygame
        pygame.draw.rect(env.screen,color,pos)

        # TODO add circle representation (+ pictures)
        # pygame.draw.circle(self.screen,color,(self.fig.x,self.fig.y),10)
        # pass

