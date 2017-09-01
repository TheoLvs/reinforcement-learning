#!/usr/bin/env python
# -*- coding: utf-8 -*- 


"""--------------------------------------------------------------------
REINFORCEMENT LEARNING

Started on the 25/08/2017

theo.alves.da.costa@gmail.com
https://github.com/theolvs
------------------------------------------------------------------------
"""

import numpy as np
import matplotlib.pyplot as plt
import gym

# Deep Learning (Keras, Tensorflow)
import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import SGD,RMSprop, Adam
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D,ZeroPadding2D,Conv2D
from keras.utils.np_utils import to_categorical

env = gym.make('CartPole-v0')
N_EPISODES = 1000
EPSILON = 1
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.995
BATCH_SIZE = 32
MAX_STEPS = 100
GAMMA = 0.95
lr = 0.001


class Memory(object):
    def __init__(self):
        self.inputs = None
        self.targets = None
        self.length = 0
    
    def cache(self,X,y):
        self.inputs = np.vstack([self.inputs,X]) if self.inputs is not None else X
        self.targets = np.vstack([self.targets,y]) if self.targets is not None else y
        self.length = len(self.inputs)
        
        
    def sample(self,batch_size = 32):
        if self.length > batch_size:
            selection = np.random.choice(range(self.length),batch_size,replace = False)
            inputs = self.inputs[selection,:]
            targets = self.targets[selection,:]
            return inputs,targets
        else:
            return self.inputs,self.targets
        
        
    def empty_cache(self):
        self.__init__()




def initialize_Q_model(states,actions):
    model = Sequential()
    model.add(Dense(32,input_dim = states,activation = "relu"))
    model.add(Dense(32,activation = "relu"))
    model.add(Dense(actions))
    model.add(Activation("linear"))

    model.compile(loss='mse',
                  optimizer=Adam(lr=lr),
                  metrics=['accuracy'])
    return model


model = initialize_Q_model(len(env.observation_space.high),env.action_space.n)
memory = Memory()





epsilon = EPSILON



if __name__ == "__main__":


    rewards = []

    # ITERATION OVER EPISODES
    for i_episode in range(N_EPISODES):

        s = env.reset()


        
        episode_reward = 0
        i = 0

        # EPISODE LOOP
        while i < MAX_STEPS:


            
            env.render()

            # Convert the state to a state vector
            s_vector = np.expand_dims(s,axis = 0)
            
            # Choose an action with a decayed epsilon greedy exploration
            q = model.predict(s_vector)
            if np.random.rand() > epsilon:
                a = np.argmax(q)
            else:
                a = np.random.randint(env.action_space.n)
                
            
            # Take the action, and get the reward from environment
            s_new,r,done,u = env.step(a)

            
            # Convert the new state to a state vector
            s_new_vector = np.expand_dims(s_new,axis = 0)
            q_new = model.predict(s_new_vector)
            
            
            # print("-","reward ",r," action ",a,"!!!" if done else "")
            
            # Update our knowledge in the Q-table
            X = s_vector
            y = q
            
            if not done:
                y[0][a] = r + GAMMA * np.max(q_new)
            else:
                y[0][a] = r

            
            # Caching to train later
            if memory is not None:
                memory.cache(X,y)
                
                
            
            
            
            # Update the caches
            episode_reward += r
            s = s_new
            
            # If the episode is terminated
            i += 1
            if done:
                print("Episode {} finished after {} timesteps".format(i_episode+1,i+1))
                break


        rewards.append(episode_reward)

        # Training
        inputs,targets = memory.sample(BATCH_SIZE)
        if epsilon >= EPSILON_MIN : epsilon *= EPSILON_DECAY
        print(epsilon)
        model.fit(inputs,targets,epochs = 1,verbose = 0)

    average_running_rewards = np.cumsum(rewards)/np.array(range(1,len(rewards)+1))
    plt.figure(figsize = (15,4))
    plt.plot(average_running_rewards)
    plt.show()





