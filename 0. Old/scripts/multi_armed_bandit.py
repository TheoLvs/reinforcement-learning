#!/usr/bin/env python
# -*- coding: utf-8 -*- 


"""--------------------------------------------------------------------
REINFORCEMENT LEARNING
Multi Armed Bandit Problem

Started on the 14/04/2017


theo.alves.da.costa@gmail.com
https://github.com/theolvs
------------------------------------------------------------------------
"""


import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys


# Deep Learning (Keras, Tensorflow)
import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import SGD,RMSprop, Adam
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D,ZeroPadding2D,Conv2D
from keras.utils.np_utils import to_categorical




#===========================================================================================================
# BANDIT DEFINITION
#===========================================================================================================



class Bandit(object):
    def __init__(self,p = None):
        '''Simple bandit initialization'''
        self.p = p if p is not None else np.random.random()
        
    def pull(self):
        '''Simulate a pull from the bandit
           
        '''
        if np.random.random() < self.p:
            return 1
        else:
            return -1
        


def create_list_bandits(n = 4,p = None):
    if p is None: p = [None]*n
    bandits = [Bandit(p = p[i]) for i in range(n)]
    return bandits





#===========================================================================================================
# NEURAL NETWORK
#===========================================================================================================



def build_fcc_model(H = 100,lr = 0.1,dim = 4):
    model = Sequential()
    model.add(Dense(H, input_dim=dim))
    model.add(Activation('relu'))
    model.add(Dense(H))
    model.add(Activation('relu'))

    sgd = SGD(lr=lr, decay=1e-6, momentum=0.9, nesterov=True)


    model.add(Dense(dim))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    return model


model = build_fcc_model()





#===========================================================================================================
# SAMPLING ACTION
#===========================================================================================================


def sample_action(probas,epsilon = 0.2):
    probas = probas[0]
    if np.random.rand() < epsilon:
        choice = np.random.randint(0,len(probas))
    else:
        choice = np.random.choice(range(len(probas)),p = probas)
    return choice









#===========================================================================================================
# EPISODE
#===========================================================================================================




def run_episode(bandits,model,probas = None,train = True,epsilon = 0.2):
    
    if probas is None:
        probas = np.ones((1,len(bandits)))/len(bandits)
    
    # sampling action
    bandit_to_pull = sample_action(probas,epsilon = epsilon)
    action = to_categorical(bandit_to_pull,num_classes=probas.shape[1])
    
    # reward
    reward = bandits[bandit_to_pull].pull()
    
    # feed vectors
    X = action
    y = (action - probas)*reward
        
    if train:
        model.train_on_batch(X,y)
        
    # update probabilities
    probas = model.predict(X)
    
    return reward,probas







#===========================================================================================================
# GAME
#===========================================================================================================


def run_game(n_episodes = 100,lr = 0.1,n_bandits = 4,p = None,epsilon = 0.2):

    # DEFINE THE BANDITS
    bandits = create_list_bandits(n = n_bandits,p = p)
    probabilities_to_win = [x.p for x in bandits]
    best_bandit = np.argmax(probabilities_to_win)
    print(">> Probabilities to win : {} -> Best bandit : {}".format(probabilities_to_win,best_bandit))

    # INITIALIZE THE NEURAL NETWORK
    model = build_fcc_model(lr = lr,dim = n_bandits)
    
    # INITIALIZE BUFFERS
    rewards = []
    avg_rewards = []
    all_probas = np.array([])
    
    # EPISODES LOOP
    for i in range(n_episodes):
        print("\r[{}/{}] episodes completed".format(i+1,n_episodes),end = "")

        # Random choice at the first episode
        if i == 0:
            reward,probas = run_episode(bandits = bandits,model = model,epsilon = epsilon)
            
        # Updated probabilities at the following episodes
        else:
            reward,probas = run_episode(bandits = bandits,model = model,probas = probas)

            
        # Store the rewards and the probas
        rewards.append(reward)
        avg_rewards.append(np.mean(rewards))
        all_probas = np.append(all_probas,probas)
        
    print("")
    
    
    # GET THE BEST PREDICTED BANDIT
    predicted_bandit = np.argmax(probas)
    print(">> Predicted bandit : {} - {}".format(predicted_bandit,"CORRECT !!!" if predicted_bandit == best_bandit else "INCORRECT"))

    
    # PLOT THE EVOLUTION OF PROBABILITIES OVER TRAINING
    all_probas = all_probas.reshape((n_episodes,n_bandits)).transpose()
    plt.figure(figsize = (12,5))
    plt.title("Probabilities on Bandit choice - {} episodes - learning rate {}".format(n_episodes,lr))
    for i,p in enumerate(list(all_probas)):
        plt.plot(p,label = "Bandit {}".format(i),lw = 1)
        
    plt.plot(avg_rewards,linestyle="-", dashes=(5, 4),color = "black",lw = 0.5,label = "average running reward")
    plt.legend()
    plt.ylim([-0.2,1])
    
    plt.show()





