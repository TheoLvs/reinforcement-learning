#!/usr/bin/env python
# -*- coding: utf-8 -*- 


"""--------------------------------------------------------------------
REINFORCEMENT LEARNING

Started on the 25/08/2017

Inspiration from https://keon.io/deep-q-learning/

theo.alves.da.costa@gmail.com
https://github.com/theolvs
------------------------------------------------------------------------
"""



import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import sys
import random
import time
import random
import numpy as np

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


from rl import utils
from rl.memory import Memory
from rl.agents.base_agent import Agent



class DQNAgent(Agent):
    def __init__(self,states_size,actions_size,epsilon = 1.0,epsilon_min = 0.01,epsilon_decay = 0.995,gamma = 0.95,lr = 0.001):
        self.states_size = states_size
        self.actions_size = actions_size
        self.memory = Memory()
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.lr = lr
        self.model = self.build_model(states_size,actions_size)





    def build_model(self,states_size,actions_size):
        model = Sequential()
        model.add(Dense(24,input_dim = states_size,activation = "relu"))
        model.add(Dense(24,activation = "relu"))
        model.add(Dense(actions_size,activation = "linear"))
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.lr))
        return model






    def train_on_batch(self,batch_size = 32):
        if len(self.memory.cache) > batch_size:
            batch = random.sample(self.memory.cache, batch_size)
        else:
            batch = self.memory.cache

        for state,action,reward,next_state,done in batch:
            state = self.expand_state_vector(state)
            next_state = self.expand_state_vector(next_state)


            targets = self.model.predict(state)

            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            else:
                target = reward

            targets[0][action] = target

            self.model.fit(state,targets,epochs = 1,verbose = 0)


        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay





    def act(self,state):
        state = self.expand_state_vector(state)

        q = self.model.predict(state)

        if np.random.rand() > self.epsilon:
            a = np.argmax(q[0])
        else:
            a = np.random.randint(self.actions_size)

        return a 


