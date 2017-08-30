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
from tqdm import tqdm





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
        
        