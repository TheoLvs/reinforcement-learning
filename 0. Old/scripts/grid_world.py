#!/usr/bin/env python
# -*- coding: utf-8 -*- 


"""--------------------------------------------------------------------
REINFORCEMENT LEARNING
Grid World

Started on the 08/08/2017


References : 
- https://www.youtube.com/watch?v=A5eihauRQvo&t=5s
- https://github.com/llSourcell/q_learning_demo
- http://firsttimeprogrammer.blogspot.fr/2016/09/getting-ai-smarter-with-q-learning.html


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





#===========================================================================================================
# CELLS DEFINITION
#===========================================================================================================


class Cell(object):
    def __init__(self,reward = 0,is_terminal = False,is_occupied = False,is_wall = False,is_start = False):
        self.reward = reward
        self.is_terminal = is_terminal
        self.is_occupied = is_occupied
        self.is_wall = is_wall
        self.is_start = is_start

    def __repr__(self):
        if self.is_occupied:
            return "x"
        else:
            return " "


    def __str__(self):
        return self.__str__()




class Start(Cell):
    def __init__(self):
        super().__init__(is_occupied = True,is_start = True)




class End(Cell):
    def __init__(self,reward = 10):
        super().__init__(reward = reward,is_terminal = True)

    def __repr__(self):
        return "O"



class Hole(Cell):
    def __init__(self,reward = -10):
        super().__init__(reward = reward,is_terminal = True)

    def __repr__(self):
        return "X"



class Wall(Cell):
    def __init__(self):
        super().__init__(is_wall = True)

    def __repr__(self):
        return "#"




#===========================================================================================================
# GRIDS DEFINITION
#===========================================================================================================




class Grid(object):
    def __init__(self,cells):
        self.grid = cells


    def __repr__(self):
        pass


    def __str__(self):
        pass

