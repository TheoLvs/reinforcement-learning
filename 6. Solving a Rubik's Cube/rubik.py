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
FACES = ["FRONT","RIGHT","BACK","LEFT","TOP","BOTTOM"]


class RubiksCube(object):
    def __init__(self,shuffle = True):

        print(f"Initialized RubiksCube")
        self.data = np.array([[i]*9 for i in range(6)])
        self.data = self._to_1D(self.data)

        if shuffle:
            np.random.shuffle(self.data)

    @staticmethod
    def _to_1D(array):
        return np.squeeze(array.reshape(1,-1))

    @staticmethod
    def _to_2D(array):
        return array.reshape(6,9)

    @staticmethod
    def _to_square(face):
        return face.reshape(3,3)


    def get_face(self,face,as_square = True):
        if isinstance(face,str):
            assert face in FACES
            face = FACES.index(face)
        face = self.data[face*9:(face+1)*9]
        if as_square:
            face = self._to_square(face)
        return face


    def rotate(self,face,clockwise = True):
        pass


    def render3D(self):
        pass   


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



