
import numpy as np
import pygame

# Custom libraries
from .spatial import SpatialEnvironment

BACKGROUND_COLOR = (0, 0, 0)


class GridEnvironment(SpatialEnvironment):
    def __init__(self,box_size = 10,width = 100,height = 60,objects = None):

        self.width = width
        self.height = height
        self.box_size = box_size

        self.agents = []
        self.static = []
        self.add_object(objects)

        self.setup_screen()

        self.grid = np.zeros((width,height))


    @property
    def objects(self):
        return self.agents + self.static


    def add_object(self,obj):

        # If we add a list of objects, using recursive function to add each object
        if isinstance(obj,list):
            for o in obj:
                self.add_object(o)
            
        # Add object to either the static or agent list
        else:
            if obj.static:
                self.static.append(obj)
            else:
                self.agents.append(obj)


    #=================================================================================
    # COLLIDERS
    #=================================================================================


    def correct_offscreen_move(self,x,y):

        env_width = self.width
        env_height = self.height

        # Check with x
        if x > env_width:
            new_x = 0
        elif x < 0:
            new_x = env_width
        else:
            new_x = x

        # Check with y
        if y > env_height:
            new_y = 0
        elif y < 0:
            new_y = env_height
        else:
            new_y = y

        return new_x,new_y


    def is_object_colliding(self,obj):
        is_collision,_ = obj.collides_with(self.objects)
        return is_collision


    #=================================================================================
    # SPAWNERS
    #=================================================================================



    def spawn(self,spawner,n,overlap = False,**kwargs):

        # Spawn n elements (works also with n = 1)
        for i in range(n):

            spawned = False

            while not spawned:

                # Generate random position
                x = np.random.randint(0,self.width)
                y = np.random.randint(0,self.height)

                # Spawn new object using spawner
                # Pass also kwargs to spawner
                obj = spawner(x,y,**kwargs)

                # If we don't care about overlapping when spawning
                # Just add new object to the queue 
                if overlap:
                    spawned = True
                    self.add_object(obj)

                # If we don't want overlapping, we use collisions to spawn efficiently new objects
                else:
                    if not self.is_object_colliding(obj):
                        spawned = True
                        self.add_object(obj)






    #=================================================================================
    # ENV LIFECYCLE
    #=================================================================================


    def step(self):

        for agent in self.agents:
            agent.step(self)



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

        for el in self.objects:
            el.render(self)

        pygame.display.update()




        

        





        


