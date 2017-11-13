#!/usr/bin/env python
# -*- coding: utf-8 -*- 


"""--------------------------------------------------------------------
REINFORCEMENT LEARNING

Started on the 13/11/2017

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
import tensorflow as tf
import keras.backend as K

# Custom RL library
import sys
sys.path.insert(0,'..')

from rl import utils
from rl.agents.actor_critic_agent import ActorCriticAgent







#----------------------------------------------------------------
# CONSTANTS


N_EPISODES = 10000
MAX_STEPS = 500
RENDER = True
RENDER_EVERY = 50



#----------------------------------------------------------------
# MAIN LOOP


if __name__ == "__main__":

    # Define the gym environment
    sess = tf.Session()
    K.set_session(sess)
    env = gym.make('Pendulum-v0')

    # Define the agent
    agent = ActorCriticAgent(env, sess)

    # Initialize a list to store the rewards
    rewards = []





    #---------------------------------------------
    # ITERATION OVER EPISODES
    for i_episode in range(N_EPISODES):



        # Reset the environment
        s = env.reset()

        reward = 0


        #-----------------------------------------
        # EPISODE RUN
        for i_step in range(MAX_STEPS):

            # Render the environement
            if RENDER : env.render() #and (i_step % RENDER_EVERY == 0)

            # The agent chose the action considering the given current state
            s = s.reshape((1, env.observation_space.shape[0]))
            a = agent.act(s)
            a = a.reshape((1, env.action_space.shape[0]))
            
            # Take the action, get the reward from environment and go to the next state
            s_next,r,done,_ = env.step(a)
            s_next = s_next.reshape((1, env.observation_space.shape[0]))
            reward += r

            # Tweaking the reward to make it negative when we lose

            # Remember the important variables
            agent.remember(s,a,r,s_next,done)
                
            # Go to the next state
            s = s_next
            
            # If the episode is terminated
            if done:
                print("Episode {}/{} finished after {} timesteps - epsilon : {:.2} - reward : {}".format(i_episode+1,N_EPISODES,i_step,agent.epsilon,reward))
                break


        #-----------------------------------------

        # Store the rewards
        rewards.append(i_step)


        # Training
        agent.train()





    # Plot the average running rewards
    utils.plot_average_running_rewards(rewards)
