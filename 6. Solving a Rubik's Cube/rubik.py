# Base Data Science snippet
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import time
from tqdm import tqdm_notebook
from matplotlib.patches import Rectangle
from matplotlib.collections import PatchCollection

plt.style.use("seaborn-dark")

import sys
sys.path.append("../")
from rl.agents.q_agent import QAgent



COLORS = ["red","white","orange","yellow","green","blue"]
WIDTH_SQUARE = 0.05


class RubiksCube(object):
    def __init__(self):

        print(f"Initialized RubiksCube")
        self.data = np.array(*[list(range(6))*9])
        np.random.shuffle(self.data)


    def render(self):
        
        fig = plt.figure(figsize=(7,7))
        ax = fig.add_subplot(111)

        for i in range(4):
            face_data = self.data[i*9:(i+1)*9]
            face = RubiksFace(face_data)
            face.render(ax = ax,init_height = 0.4,init_width = 0.15 + i*3*(WIDTH_SQUARE+0.005))


        for i in range(4,6):
            face_data = self.data[i*9:(i+1)*9]
            face = RubiksFace(face_data)
            init_height = 0.4 + 3*(WIDTH_SQUARE+0.005) if i == 4 else 0.4 - 3*(WIDTH_SQUARE+0.005)
            face.render(ax = ax,init_height = init_height,init_width = 0.15 + 3*(WIDTH_SQUARE+0.005))

        plt.xticks([])
        plt.yticks([])
        plt.show()





class RubiksFace(object):
    def __init__(self,array):
        assert len(array) == 9
        self.array = array.reshape(3,3)
        
    def render(self,ax = None,init_height = 0,init_width = 0):

        if ax is None:
            fig = plt.figure(figsize=(7,7))
            ax = fig.add_subplot(111)


        
        for i in range(3):
            for j in range(3):

                square = self.array[i,j]
                color = COLORS[square]

                rect = Rectangle((init_width + i*WIDTH_SQUARE,init_height + j*WIDTH_SQUARE), WIDTH_SQUARE, WIDTH_SQUARE)
                collection = PatchCollection([rect],facecolor = color,alpha = 0.8,edgecolor="black")
                ax.add_collection(collection)



