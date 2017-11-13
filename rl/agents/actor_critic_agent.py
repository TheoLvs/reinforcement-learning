#!/usr/bin/env python
# -*- coding: utf-8 -*- 


"""--------------------------------------------------------------------
REINFORCEMENT LEARNING

Started on the 25/08/2017

Inspiration from https://keon.io/deep-q-learning/
https://towardsdatascience.com/reinforcement-learning-w-keras-openai-actor-critic-models-f084612cfd69

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

from keras.models import Sequential, Model
from keras.layers import Dense, Dropout, Input
from keras.layers.merge import Add, Multiply
from keras.optimizers import Adam
import keras.backend as K
import tensorflow as tf

from rl import utils
from rl.memory import Memory
from rl.agents.base_agent import Agent



class ActorCriticAgent(Agent):
    def __init__(self,env,sess,epsilon = 1.0,epsilon_min = 0.01,epsilon_decay = 0.995,gamma = 0.95,lr = 0.001,tau = 0.125,actor_activation = "linear"):

        # Main parameters
        self.env = env
        self.sess = sess

        # Other parameters
        self.memory = Memory()
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.gamma = gamma
        self.tau = tau
        self.lr = lr

        # Models
        self.initialize_actor_model(actor_activation)
        self.initialize_critic_model()


    def initialize_actor_model(self,actor_activation):
        self.actor_state_input, self.actor_model = self.build_actor_model(actor_activation)
        _, self.target_actor_model = self.build_actor_model(actor_activation)

        self.actor_critic_grad = tf.placeholder(tf.float32, [None, self.env.action_space.shape[0]]) # where we will feed de/dC (from critic)
        
        actor_model_weights = self.actor_model.trainable_weights
        self.actor_grads = tf.gradients(self.actor_model.output, actor_model_weights, -self.actor_critic_grad) # dC/dA (from actor)
        grads = zip(self.actor_grads, actor_model_weights)
        self.optimize = tf.train.AdamOptimizer(self.lr).apply_gradients(grads)



    def build_actor_model(self,activation = ""):
        # Define the layers of the network
        state_input = Input(shape=self.env.observation_space.shape)
        h1 = Dense(24, activation='relu')(state_input)
        h2 = Dense(48, activation='relu')(h1)
        h3 = Dense(24, activation='relu')(h2)
        output = Dense(self.env.action_space.shape[0],activation='relu')(h3)
        
        # Compute the model
        model = Model(input=state_input, output=output)
        model.compile(loss="mse", optimizer=Adam(lr=self.lr))
        return state_input, model


    def initialize_critic_model(self):
        self.critic_state_input, self.critic_action_input, self.critic_model = self.build_critic_model()
        _, _, self.target_critic_model = self.build_critic_model()

        self.critic_grads = tf.gradients(self.critic_model.output,self.critic_action_input) # where we calcaulte de/dC for feeding above
        
        # Initialize for later gradient calculations
        self.sess.run(tf.initialize_all_variables())




    def build_critic_model(self):
        state_input = Input(shape=self.env.observation_space.shape)
        state_h1 = Dense(24, activation='relu')(state_input)
        state_h2 = Dense(48)(state_h1)
        
        action_input = Input(shape=self.env.action_space.shape)
        action_h1    = Dense(48)(action_input)
        
        merged    = Add()([state_h2, action_h1])
        merged_h1 = Dense(24, activation='relu')(merged)
        output = Dense(1, activation='relu')(merged_h1)
        model  = Model(input=[state_input,action_input], output=output)
        
        model.compile(loss="mse", optimizer=Adam(lr=self.lr))
        return state_input, action_input, model






    def train(self,batch_size = 32):
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        if len(self.memory.cache) > batch_size:
            batch = random.sample(self.memory.cache, batch_size)
        else:
            batch = self.memory.cache

        self._train_actor(batch)
        self._train_critic(batch)





    def _train_actor(self,batch):
        for state,action,reward,next_state,_ in batch:
            predicted_action = self.actor_model.predict(state)
            grads = self.sess.run(self.critic_grads, feed_dict={
                self.critic_state_input:  state,
                self.critic_action_input: predicted_action
            })[0]

            self.sess.run(self.optimize, feed_dict={
                self.actor_state_input: state,
                self.actor_critic_grad: grads
            })
            


    def _train_critic(self,batch):
        for state,action,reward,next_state,done in batch:
            if not done:
                target_action = self.target_actor_model.predict(next_state)
                future_reward = self.target_critic_model.predict([next_state, target_action])[0][0]
                reward += self.gamma * future_reward
            self.critic_model.fit([state, action], reward, verbose=0)
        


    def _update_actor_target(self):
        actor_model_weights  = self.actor_model.get_weights()
        actor_target_weights = self.target_critic_model.get_weights()
        
        for i in range(len(actor_target_weights)):
            actor_target_weights[i] = actor_model_weights[i]
        self.target_critic_model.set_weights(actor_target_weights)


    def _update_critic_target(self):
        critic_model_weights  = self.critic_model.get_weights()
        critic_target_weights = self.critic_target_model.get_weights()
        
        for i in range(len(critic_target_weights)):
            critic_target_weights[i] = critic_model_weights[i]
        self.critic_target_model.set_weights(critic_target_weights)     


    def update_target(self):
        self._update_actor_target()
        self._update_critic_target()




    def act(self, state):



            
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return self.actor_model.predict(state)