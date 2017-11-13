#!/usr/bin/env python
# -*- coding: utf-8 -*- 


"""--------------------------------------------------------------------
REINFORCEMENT LEARNING

Started on the 25/08/2017

theo.alves.da.costa@gmail.com
https://github.com/theolvs
------------------------------------------------------------------------
"""





# Usual libraries
import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import sys
import random
import time
from tqdm import tqdm
import random
import gym
import numpy as np


# Keras (Deep Learning)
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


# Custom RL library
import sys
sys.path.insert(0,'..')

from rl import utils
from rl.agents.dqn_agent import DQNAgent

import pybullet_envs.bullet.minitaur_gym_env as e





#----------------------------------------------------------------
# CONSTANTS


N_EPISODES = 1000
MAX_STEPS = 1000
RENDER = True
RENDER_EVERY = 50



#----------------------------------------------------------------
# MAIN LOOP


if __name__ == "__main__":

    # Define the gym environment
    env = e.MinitaurBulletEnv(render=True)

    # Get the environement action and observation space
    state_size = env.observation_space.shape[0]
    action_size = env.action_space.shape[0]

    # Create the RL Agent
    agent = DQNAgent(state_size,action_size)

    # Initialize a list to store the rewards
    rewards = []





    #---------------------------------------------
    # ITERATION OVER EPISODES
    for i_episode in range(N_EPISODES):



        # Reset the environment
        s = env.reset()


        #-----------------------------------------
        # EPISODE RUN
        for i_step in range(MAX_STEPS):
        
            # Render the environement
            if RENDER : env.render() #and (i_step % RENDER_EVERY == 0)

            # The agent chose the action considering the given current state
            a = agent.act(s)
            
            # Take the action, get the reward from environment and go to the next state
            s_next,r,done,info = env.step(a)

            # Tweaking the reward to make it negative when we lose
            r = r if not done else -10

            # Remember the important variables
            agent.remember(s,a,r,s_next,done)
                
            # Go to the next state
            s = s_next
            
            # If the episode is terminated
            if done:
                print("Episode {}/{} finished after {} timesteps - epsilon : {:.2}".format(i_episode+1,N_EPISODES,i_step,agent.epsilon))
                break


        #-----------------------------------------

        # Store the rewards
        rewards.append(i_step)


        # Training
        agent.train()





    # Plot the average running rewards
    utils.plot_average_running_rewards(rewards)
