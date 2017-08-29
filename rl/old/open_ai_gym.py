#!/usr/bin/env python
# -*- coding: utf-8 -*- 


"""--------------------------------------------------------------------
REINFORCEMENT LEARNING
First RL script done using Keras and policy gradients

- Inspired by @steinbrecher script on https://gym.openai.com/evaluations/eval_usjJ7onVTTwrn43wrbBiAv
- Still inspired by Karpathy's work too

Started on the 30/12/2016


https://github.com/rybskej/atari-py
https://sourceforge.net/projects/vcxsrv/


Environment which works with script:  
- CartPole-v0
- MountainCar-v0
- Taxi-v1


theo.alves.da.costa@gmail.com
https://github.com/theolvs
------------------------------------------------------------------------
"""


import numpy as np
import gym
import os
from keras.models import load_model, Sequential
from keras.layers import Dense, Activation
from keras.optimizers import SGD, RMSprop



#-------------------------------------------------------------------------------




    
# def main(n_episodes = 20):
#     for i_episode in range(n_episodes):
#         observation = env.reset()
#         print(observation)
#         break
#         for t in range(1000):
#             if render: env.render
#             print(observation)
#             action = env.action_space.sample()
#             observation, reward, done, info = env.step(action)
#             if done:
#                 print("Episode finished after {} timesteps".format(t+1))
#                 break






#-------------------------------------------------------------------------------








class Brain():
    def __init__(self,env,env_name = "default",H = 500,reload = False):

        self.env_name = env_name
        self.base_path = "C:/Users/talvesdacosta/Documents/Perso/Data Science/15. Reinforcement Learning/3. Open AI Gym/models/"
        file = [x for x in os.listdir(self.base_path) if self.env_name in x]

        self.H = H
        self.gamma = 0.975
        self.batch_size = 10

        try:
            self.observation_space = env.observation_space.n
            self.observation_to_vectorize = True
        except Exception as e:
            self.observation_space = env.observation_space.shape[0]
            self.observation_to_vectorize = False

        self.action_space = env.action_space.n


        if len(file) == 0 or reload:
            print('>> Building a fully connected neural network')
            self.episode_number = 0
            self.model = self.build_fcc_model(H,input_dim = self.observation_space,output_dim = self.action_space)
        else:
            print('>> Loading the previously trained model')
            self.episode_number = int(file[0][file[0].find("(")+1:file[0].find(")")])
            self.model = load_model(self.base_path + file[0])



        self.inputs,self.actions,self.probas,self.rewards,self.step_rewards = [],[],[],[],[]
        self.episode_rewards,self.episode_running_rewards = [],[]
        self.reward_sum = 0
        self.running_reward = 0




    def build_fcc_model(self,H = 500,input_dim = 4,output_dim = 2):
        model = Sequential()
        model.add(Dense(H, input_dim=input_dim))
        model.add(Activation('relu'))
        model.add(Dense(H))
        model.add(Activation('relu'))

        sgd = SGD(lr=0.05, decay=1e-6, momentum=0.9, nesterov=True)

        if output_dim <= 2:
            model.add(Dense(1))
            model.add(Activation('sigmoid'))
            model.compile(loss='mse',
                          optimizer=sgd,
                          metrics=['accuracy'])
        else:
            model.add(Dense(output_dim))
            model.add(Activation('softmax'))
            model.compile(loss='categorical_crossentropy',
                          optimizer=sgd,
                          metrics=['accuracy'])

        return model



    def to_input(self,observation):
        if self.observation_to_vectorize:
            observation = self.vectorize_observation(observation,self.observation_space)
        return np.reshape(observation,(1,self.observation_space))


    def predict(self,observation):

        x = self.to_input(observation)

        # getting the probability of action
        probas = self.model.predict(x)[0]

        # sampling the correct action
        action= self.sample_action(probas)

        return x,action,probas


    def sample_action(self,probabilities):
        if len(probabilities)<=2:
            action = 1 if np.random.uniform() < probabilities[0] else 0
        else:
            action = np.random.choice(len(probabilities),p = np.array(probabilities))

        return action

    def vectorize_action(self,action):
        if self.action_space <= 2:
            return action
        else:
            onehot_vector = np.zeros(self.action_space)
            onehot_vector[action] = 1
            return onehot_vector

    def vectorize_observation(self,value,size):
        onehot_vector = np.zeros(size)
        onehot_vector[value] = 1
        return onehot_vector



    def record(self,input = None,action = None,proba = None,reward = None):
        if type(input) != type(None):
            self.inputs.append(input)

        if type(action) != type(None):
            self.actions.append(action)

        if type(proba) != type(None):
            self.probas.append(proba)

        if type(reward) != type(None):
            self.rewards.append(reward)
            self.reward_sum += reward




    def discounting_rewards(self,r,normalization = True):
        discounted_r = np.zeros_like(r)
        running_add = 0
        for t in reversed(range(0, r.size)):
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add

        if normalization:
            discounted_r = np.subtract(discounted_r,np.mean(discounted_r),casting = "unsafe")
            discounted_r = np.divide(discounted_r,np.std(discounted_r),casting = "unsafe")

        return discounted_r


    def discount_rewards(self,normalization = True):
        rewards = np.vstack(self.rewards)
        return self.discounting_rewards(rewards,normalization)


    def record_episode(self):
        self.step_rewards.extend(self.discount_rewards(normalization = True))
        self.episode_rewards.append(self.reward_sum)
        self.running_reward = np.mean(self.episode_rewards)
        self.episode_number += 1

    def reset_episode(self):
        self.rewards = []
        self.reward_sum = 0

    def update_on_batch(self):
        print('... Training on batch of size %s'%self.batch_size)
        self.actions = np.vstack(self.actions)
        self.probas = np.vstack(self.probas)
        self.step_rewards = np.vstack(self.step_rewards)
        self.inputs = np.vstack(self.inputs)

        self.targets = self.step_rewards * (self.actions - self.probas) + self.probas

        #ajouter la protection de la max rewards

        self.model.train_on_batch(self.inputs,self.targets)

        self.inputs,self.actions,self.probas,self.step_rewards = [],[],[],[]

    def save_model(self):
        file = [x for x in os.listdir(self.base_path) if self.env_name in x]
        self.model.save(self.base_path+"%s(%s).h5"%(self.env_name,self.episode_number))
        if len(file)>0:
            os.remove(self.base_path+file[0])
        # self.model.save(self.base_path+"%s.h5"%(self.env_name))











def main(env_name = 'CartPole-v0',n_episodes = 20,render = False,reload = False,n_by_episode = 1000):
    env = gym.make(env_name)
    brain = Brain(env,env_name = env_name,reload = reload)
    # env.monitor.start(brain.base_path+'monitor/%s'%env_name)


    for i_episode in range(1,n_episodes+1):
        observation = env.reset()   
        for t in range(n_by_episode):
            if render: env.render()

            x,action,proba = brain.predict(observation)

            observation, reward, done, info = env.step(action)
            action = brain.vectorize_action(action)
            brain.record(input = x,action = action,proba = proba,reward = reward)

            if done or t == n_by_episode - 1:
                brain.record_episode()
                print("Episode {} : total reward was {:0.03f} and running mean {:0.03f}".format(brain.episode_number, brain.reward_sum, brain.running_reward))


                if i_episode % brain.batch_size == 0:
                    brain.update_on_batch()

                if i_episode % 100 == 0:
                    brain.save_model()


                brain.reset_episode()

                break

    # env.monitor.close()