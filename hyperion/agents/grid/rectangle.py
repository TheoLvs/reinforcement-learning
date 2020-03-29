

import numpy as np
import pygame

from ..agent import Agent



class Rectangle(Agent):

    def __init__(self,x,y,width,height,color):
        
        super().__init__()

        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.color = color


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
    # MOVEMENT
    #=================================================================================

    def move(self,*args,**kwargs):
        pass


    def is_static(self):
        return True



    #=================================================================================
    # RENDERERS
    #=================================================================================


    def render(self,env):

        # Get x,y,width,height in pixels from box size in the grid
        pos = self.get_pos_px(env.box_size)

        # Draw a rectangle on the grid using pygame
        pygame.draw.rect(env.screen,self.color,pos)

        # TODO add circle representation (+ pictures)
        # pygame.draw.circle(self.screen,color,(self.fig.x,self.fig.y),10)
        # pass

