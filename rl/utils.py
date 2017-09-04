#!/usr/bin/env python
# -*- coding: utf-8 -*- 


"""--------------------------------------------------------------------
REINFORCEMENT LEARNING

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
import random
import numpy as np
import pylab



def plot_average_running_rewards(rewards,save = None):
    average_running_rewards = np.cumsum(rewards)/np.array(range(1,len(rewards)+1))
    figure = plt.figure(figsize = (15,4))
    plt.plot(average_running_rewards)

    if save is None:
        plt.show()
    else:
        plt.savefig(save)




