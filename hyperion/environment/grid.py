
import numpy as np
import pygame

# Custom libraries
from .spatial import SpatialEnvironment

BACKGROUND_COLOR = (0, 0, 0)


class GridEnvironment(SpatialEnvironment):
    def __init__(self,box_size = 10,width = 100,height = 60,agents = None,static = None):

        self.width = width
        self.height = height
        self.box_size = box_size

        self.agents = [] if agents is None else agents
        self.static = [] if static is None else static

        self.setup_screen()

        self.grid = np.zeros((width,height))


    @property
    def elements(self):
        return self.agents + self.static


    #=================================================================================
    # COLLIDERS
    #=================================================================================

    pass



    #=================================================================================
    # RENDERERS
    #=================================================================================

    def setup_screen(self):
        self.screen = pygame.display.set_mode((
            self.width * self.box_size,
            self.height * self.box_size
            ))

        self.reset_screen()


    def reset_screen(self):
        self.screen.fill(BACKGROUND_COLOR)



    def render(self):

        self.reset_screen()

        for el in self.elements:
            el.render(self)

        pygame.display.update()


        

        





        


