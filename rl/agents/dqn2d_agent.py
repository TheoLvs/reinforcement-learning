#!/usr/bin/env python
# -*- coding: utf-8 -*- 


"""--------------------------------------------------------------------
REINFORCEMENT LEARNING

Started on the 19/10/2018

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

from keras.layers import Input, LSTM, Dense, Conv2D, MaxPooling2D, Dropout, Flatten
from keras.layers import concatenate
from keras.models import Model
from keras.utils import plot_model,to_categorical

from rl import utils
from rl.memory import Memory
from rl.agents.base_agent import Agent
from rl.agents.dqn_agent import DQNAgent





def create_vision_model(input_shape):
    input_image = Input(shape=input_shape)
    conv1 = Conv2D(32,(3,3),padding="same",activation="relu")(input_image)
    pool1 = MaxPooling2D(pool_size=(2,2))(conv1)
    drop1 = Dropout(0.25)(pool1)

    conv2 = Conv2D(64,(3,3),padding="same",activation="relu")(drop1)
    pool2 = MaxPooling2D(pool_size=(2,2))(conv2)
    drop2 = Dropout(0.25)(pool2)

    out = Flatten()(drop2)

    vision_model = Model(inputs=input_image, outputs=out)
    return vision_model


def create_model(input_shape,output_dim):

    input1 = Input(shape=input_shape)
    input2 = Input(shape=input_shape)

    vision_model = create_vision_model(input_shape)

    out1 = vision_model(input1)
    out2 = vision_model(input2)

    concatenated = concatenate([out1,out2])

    hidden = Dense(128, activation='relu')(concatenated)
    output = Dense(output_dim, activation='softmax')(hidden)

    model = Model([input1, input2], output)

    return model





class DQN2DAgent(DQNAgent):



    def build_model(self,states_size,actions_size):
        model = create_model(states_size,actions_size)
        model.compile(loss='categorical_crossentropy',
                      metrics=['accuracy'],
                      optimizer="adam")
        return model



    def train(self,batch_size = 32):
        if len(self.memory.cache) > batch_size:
            batch = random.sample(self.memory.cache, batch_size)
        else:
            batch = self.memory.cache

        # Unzip batch
        states,actions,rewards,next_states,before_states,dones = zip(*batch)

        # Concat states
        states = np.vstack(states)
        next_states = np.vstack(next_states)
        before_states = np.vstack(before_states)

        # Compute targets
        targets = self.model.predict([before_states,states])

        # Compute new targets
        rewards = np.array(rewards).reshape(-1,1)
        dones = 1-np.array(dones,dtype=np.int32).reshape(-1,1)
        predictions = (self.gamma * np.max(self.model.predict([before_states,states]),axis = 1)).reshape(-1,1)
        new_targets = rewards + dones * predictions
        new_targets = new_targets.astype("float32")

        # Correct targets
        actions = to_categorical(np.array(actions).reshape(-1,1),self.actions_size)
        np.place(targets,actions,new_targets)

        # Training
        self.model.fit([states,next_states],targets,epochs = 1,verbose = 0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay





    def act(self,before_state,state):
        before_state = self.expand_state_vector(before_state)
        state = self.expand_state_vector(state)


        if np.random.rand() > self.epsilon:
            q = self.model.predict([before_state,state])

            if self.observation_type == "discrete":
                a = np.argmax(q[0])
            elif self.observation_type == "continuous":
                a = np.squeeze(np.clip(q,self.low,self.high))

        else:
            if self.observation_type == "discrete":
                a = np.random.randint(self.actions_size)
            elif self.observation_type == "continuous":
                a = np.random.uniform(self.low,self.high,self.actions_size)
        return a 


