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

from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam






class Agent(object):
    def __init__(self):
        pass


    def expand_state_vector(self,state):
        if len(state.shape) == 1 or len(state.shape)==3:
            return np.expand_dims(state,axis = 0)
        else:
            return state



    def remember(self,*args):
        self.memory.save(args)





