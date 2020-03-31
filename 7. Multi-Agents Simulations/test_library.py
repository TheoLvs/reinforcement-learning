

import sys
sys.path.append("C:/git/reinforcement-learning")

import time
import pygame
import numpy as np

from hyperion.environment.grid import GridEnvironment
from hyperion.agents.grid import GridAgent,Rectangle

BOX_SIZE = 5


class TestAgent(GridAgent):

    def step(self,env):

        while self.follow_direction(dr = 1,env = env):
            self.change_direction()



agent_spawner = lambda x,y : TestAgent(x,y,2,2,BOX_SIZE)




obstacles = [
    Rectangle(20,30,100,10,BOX_SIZE,(0,200,100)),
    Rectangle(40,30,10,100,BOX_SIZE,(0,200,100)),
    Rectangle(70,70,100,10,BOX_SIZE,(0,200,100)),
    Rectangle(170,10,10,10,BOX_SIZE,(0,200,100)),
]

obstacle_spawner = lambda x,y : Rectangle(x,y,np.random.randint(1,10)*10,np.random.randint(1,10)*10,BOX_SIZE,(0,100,200))


# Setup grid
env = GridEnvironment(BOX_SIZE,220,120,objects = None)
env.spawn(obstacle_spawner,5)
env.spawn(agent_spawner,20)

n_steps = 1000
i = 0
simulation_on = True
step_duration = 0.1


while simulation_on:


    env.step()
    env.render()
    time.sleep(step_duration)

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            # agent.change_direction()
            simulation_on = False
    
        if event.type == pygame.QUIT:
            simulation_on = False

    if i >= n_steps:
        simulation_on = False
    else:
        i += 1
        

pygame.quit()
