

import numpy as np
import pygame

from ..base_object import BaseObject



class Rectangle(BaseObject):

    def __init__(self,x,y,width,height,box_size,color):
        
        super().__init__()

        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.box_size = box_size
        self.color = color


    def __repr__(self):
        return f"Rectangle(x={self.x},y={self.y},w={self.width},h={self.height})"


    @property
    def pos(self):
        return self.x,self.y


    @property
    def static(self):
        return True


    @property
    def dimensions(self):
        return (
            self.x * self.box_size,
            self.y * self.box_size,
            self.width * self.box_size,
            self.height * self.box_size
        )

    @property
    def collider(self):
        return pygame.rect.Rect(self.dimensions)


    def get_collider(self,x,y):
        dimensions = (
            x * self.box_size,
            y * self.box_size,
            self.width * self.box_size,
            self.height * self.box_size
        )
        return pygame.rect.Rect(dimensions)




    #=================================================================================
    # MOVEMENT
    #=================================================================================

    def move(self,*args,**kwargs):
        pass

    def step(self,*args,**kwargs):
        pass


    def collides_with(self,others,collider = None):

        # In the absence of external colliders
        # We take the internal collider
        if collider is None:
            collider = self.collider
        
        # If no other colliders
        if len(others) == 0:
            collisions = []

        # Compute collisions using PyGame
        else:
            other_colliders = [other.collider for other in others if other.id != self.id]
            collisions = collider.collidelistall(other_colliders)

        # Return signal of collision and colliders touched
        if len(collisions) > 0:
            return True,collisions
        else:
            return False,collisions





    #=================================================================================
    # RENDERERS
    #=================================================================================


    def render(self,env):

        # Draw a rectangle on the grid using pygame
        pygame.draw.rect(env.screen,self.color,self.dimensions)

        # TODO add circle representation (+ pictures)
        # pygame.draw.circle(self.screen,color,(self.fig.x,self.fig.y),10)
        # pass

