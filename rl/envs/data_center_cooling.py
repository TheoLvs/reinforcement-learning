#!/usr/bin/env python
# -*- coding: utf-8 -*- 


"""--------------------------------------------------------------------
REINFORCEMENT LEARNING
DATA CENTER COOLING

Started on the 25/08/2017


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
from tqdm import tqdm
from collections import Counter
from scipy import stats

# Deep Learning (Keras, Tensorflow)
import tensorflow as tf
from keras.models import Sequential
from keras.optimizers import SGD,RMSprop, Adam
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import MaxPooling2D,ZeroPadding2D,Conv2D
from keras.utils.np_utils import to_categorical


# Plotly
import plotly.graph_objs as go
from plotly import tools

np.random.seed(1)


#===========================================================================================================
# COOLING CENTER ENVIRONMENT
#===========================================================================================================



class DataCenterCooling(object):
    def __init__(self,levels_activity = 20,levels_cooling = 10,cost_factor = 5,risk_factor = 1.6):

        self.hour = 0
        self.cost_factor = cost_factor
        self.risk_factor = risk_factor
        self.levels_activity = levels_activity
        self.levels_cooling = levels_cooling
        self.define_activity(levels_activity)
        self.define_cooling(levels_cooling)

        
    def define_activity(self,levels_activity):
        # Define the peaks of activity
        peak_morning = np.random.randint(7,10)
        peak_evening = np.random.randint(17,22)

        # Build the distribution
        x1 = np.array(stats.poisson.pmf(range(24),peak_morning))
        x2 = np.array(stats.poisson.pmf(range(24),peak_evening))
        x = x1 + x2
        x *= (100/0.14)

        # Discretize the distribution
        take_closest = lambda j,vector:min(vector,key=lambda x:abs(x-j))
        percentiles = np.percentile(x,range(0,100,int(100/levels_activity)))
        assert len(percentiles) == levels_activity
        x_disc = np.array([take_closest(y,percentiles) for y in x])

        # Store the variable
        self.observation_space = percentiles
        self.activity = np.expand_dims(x_disc,axis = 0)



    def define_cooling(self,levels_cooling):
        self.action_space = list([int(100/levels_cooling*i) for i in range(levels_cooling)])
        assert len(self.action_space) == levels_cooling

        initial_value = random.choice(self.action_space)
        self.cooling = np.full((1,24),initial_value)



    def reset(self):
        self.__init__(self.levels_activity,self.levels_cooling,self.cost_factor)
        return self.reset_state()

    def reset_state(self):
        activity = self.activity[0][0]
        activity_state = self.convert_activity_to_state(activity)
        return activity_state


    def convert_activity_to_state(self,activity):
        state = int(np.where(self.observation_space == activity)[0][0])
        return state



    def render(self,with_plotly = False):

        rewards,winnings,losses,failures = self.compute_daily_rewards()

        if not with_plotly:
            # Show the activity and cooling
            plt.figure(figsize = (14,5))
            plt.plot(np.squeeze(self.activity),c ="red",label = "activity")
            plt.plot(np.squeeze(self.cooling),c = "blue",label = "cooling")
            plt.legend()
            plt.show()

            # Show the rewards
            plt.figure(figsize = (14,5))
            plt.title("Total reward : {}".format(int(np.sum(rewards))))
            plt.plot(rewards,c = "blue",label = "profits")
            plt.plot(losses*(-1),c = "red",label = "costs")
            plt.plot(winnings,c = "green",label = "revenues")
            plt.legend()
            plt.show()
        else:
            data_states = self.render_states_plotly()["data"]
            data_rewards = self.render_rewards_plotly()["data"]
            data_states
            fig = tools.make_subplots(rows=2, cols=1, specs=[[{}], [{}]],
                          shared_xaxes=True, shared_yaxes=False,
                          vertical_spacing=0.1)

            for i,trace in enumerate(data_rewards):
                fig.append_trace(trace, 2, 1)

            for i,trace in enumerate(data_states):
                fig.append_trace(trace, 1, 1)

            # print(len(failures))
            # print(len(rewards))

            # shapes = [{"type":"line","x0":hour+1,"y0":0,"x1":hour+1,"y1":failure} for hour,failure in enumerate(failures) if failure > 0]
            fig['layout'].update(title="Total reward : {}".format(int(np.sum(rewards))))
            fig['layout']['xaxis'].update(dtick = 1)
            # fig['layout'].update(shapes=shapes)
            return fig


    def render_states_plotly(self):
        # Create a trace
        x = list(range(24))
        trace_activity = go.Scatter(x = x,y = np.squeeze(self.activity),name = "activity",line = dict(color = "red",width = 2),ysrc = "activity")
        trace_cooling = go.Scatter(x = x,y = np.squeeze(self.cooling),name = "cooling",line = dict(color = "#34aac1",width = 2))

        data = [trace_activity,trace_cooling]
        fig = {"data":data}
        return fig


    def render_rewards_plotly(self):
        rewards,winnings,losses,failures = self.compute_daily_rewards()
        # Create a trace
        x = list(range(24))
        trace_rewards = go.Scatter(x = x,y = np.squeeze(rewards),name = "rewards",line = dict(color = "#34aac1",width = 2),ysrc = "rewards")
        trace_winnings = go.Scatter(x = x,y = np.squeeze(winnings),name = "revenues",line = dict(color = "#10c576",width = 1),mode = "lines+markers")
        trace_losses = go.Scatter(x = x,y = np.squeeze(losses),name = "costs",line = dict(color = "red",width = 1),mode = "lines+markers")

        data = [trace_rewards,trace_winnings,trace_losses]
        fig = {"data":data}
        return fig


        


    def compute_reward(self,activity,cooling):

        # CALCULATING THE WINNINGS
        win = activity

        # CALCULATING THE LOSSES
        if cooling >= activity:
            cost = (0 if self.cost_factor < 1.0 else 1)*(cooling)**np.sqrt(self.cost_factor)
            failure = 0
        else:
            difference = (activity-cooling)/(cooling+1)
            default_probability = np.tanh(difference)
            if np.random.rand() > default_probability or self.risk_factor <= 1.0:
                cost = 0
            else:
                cost = np.random.normal(loc = self.risk_factor,scale = 0.4) * 150

            # cost += (cooling * min(1,self.cost_factor))**2
            cost += (0 if self.cost_factor < 1.0 else 1)*(cooling)

            failure = cost

        return win,cost,failure






    def compute_daily_rewards(self):
        winnings = []
        losses = []
        rewards = []
        failures = []
        for i in range(24):
            activity = self.activity[0][i]
            cooling = self.cooling[0][i]
            win,loss,failure = self.compute_reward(activity,cooling)
            winnings.append(win)
            losses.append(loss)
            rewards.append(win-loss)
            failures.append(failure)

        return np.array(rewards),np.array(winnings),np.array(losses),np.array(failures)







    def step(self,cooling_action):

        # Convert cooling_action to cooling_value
        cooling = self.action_space[cooling_action]

        # Update the cooling
        self.cooling[0][self.hour] = cooling

        activity = self.activity[0][self.hour]
        win,loss,failure = self.compute_reward(activity,cooling)
        reward = win-loss

        self.hour += 1

        if int(self.hour) == 24:
            new_state = self.reset_state()
            done = True
        else:
            new_activity = self.activity[0][self.hour]
            new_state = self.convert_activity_to_state(new_activity)
            done = False


        return new_state,reward,done


    
    


