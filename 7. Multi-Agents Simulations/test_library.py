

import sys
sys.path.append("C:/git/reinforcement-learning")

import time
import pygame

from hyperion.environment.grid import GridEnvironment
from hyperion.agents.grid import GridAgent,StaticObject


agent = GridAgent(10,20)


# Setup grid
env = GridEnvironment(10,100,60,agents = [agent])
n_steps = 100
step_duration = 0.05

for i in range(n_steps):

    agent.random_walk(env)
    env.render()
    time.sleep(step_duration)

    for event in pygame.event.get():
        if event.type == pygame.KEYDOWN:
            break


pygame.quit()
