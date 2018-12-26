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

# import sys
# sys.path.append("../")
# from rl.agents.q_agent import QAgent

#----------------------------------------------------------------------------------------------------------------------------
# CONSTANTS
#----------------------------------------------------------------------------------------------------------------------------

COLORS = ["red","white","orange","yellow","green","blue"]
WIDTH_SQUARE = 0.05
FACES = ["LEFT","FRONT","RIGHT","BACK","TOP","BOTTOM"]

LEFT_SLICE = np.s_[0,:]
RIGHT_SLICE = np.s_[-1,:]
TOP_SLICE = np.s_[:,0]
BOTTOM_SLICE = np.s_[:,-1]

FACES_LINK = {
    "LEFT":[
        ("BACK",RIGHT_SLICE),
        ("BOTTOM",LEFT_SLICE),
        ("FRONT",LEFT_SLICE),
        ("TOP",LEFT_SLICE),
    ],
    "FRONT":[
        ("LEFT",RIGHT_SLICE),
        ("BOTTOM",BOTTOM_SLICE),
        ("RIGHT",LEFT_SLICE),
        ("TOP",TOP_SLICE),
    ],
    "RIGHT":[
        ("TOP",RIGHT_SLICE),
        ("FRONT",RIGHT_SLICE),
        ("BOTTOM",RIGHT_SLICE),
        ("BACK",LEFT_SLICE),
    ],
    "BACK":[
        ("TOP",BOTTOM_SLICE),
        ("RIGHT",RIGHT_SLICE),
        ("BOTTOM",TOP_SLICE),
        ("LEFT",LEFT_SLICE),
    ],
    "TOP":[
        ("LEFT",BOTTOM_SLICE),
        ("FRONT",BOTTOM_SLICE),
        ("RIGHT",BOTTOM_SLICE),
        ("BACK",BOTTOM_SLICE),
    ],
    "BOTTOM":[
        ("BACK",TOP_SLICE),
        ("RIGHT",TOP_SLICE),
        ("FRONT",TOP_SLICE),
        ("LEFT",TOP_SLICE),
    ],
}




#----------------------------------------------------------------------------------------------------------------------------
# RUBIKS CUBE ENVIRONMENT CLASS
#----------------------------------------------------------------------------------------------------------------------------

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

    @staticmethod
    def _to_array(face):
        return face.reshape(9)


    @staticmethod
    def _facestr_to_faceid(face):
        """Convert face as string to face ID (between 0 and 5)
        """
        if isinstance(face,str):
            assert face in FACES
            face = FACES.index(face)
        return face


    @staticmethod
    def _rotate_array(array,clockwise = True):
        if clockwise:
            return array[1:] + [array[0]]
        else:
            return [array[-1]] + array[:-1]


    def get_face(self,face,as_square = True):
        """Function to get one face of the Rubik's cube
        """

        # Convert face as string to face ID (between 0 and 5)
        face = self._facestr_to_faceid(face)

        # Select matching face in the data array
        face = self.data[face*9:(face+1)*9]

        # Reshape face data to a square 
        if as_square:
            face = self._to_square(face)

        # Return face data
        return face




    def set_face(self,face,array):

        # Convert face as string to face ID (between 0 and 5)
        face = self._facestr_to_faceid(face)

        # Reshape array
        if array.shape == (3,3):
            array = self._to_array(array)

        # Set face
        self.data[face*9:(face+1)*9] = array





    def rotate(self,face,clockwise = True):
        """Rotate one face of the Rubik's cube
        """
        # Convert face as string to face ID (between 0 and 5)
        face_id = self._facestr_to_faceid(face)

        # Get face
        face_data = self.get_face(face_id)

        # Rotate selected face
        sense = -1 if clockwise else 1
        face_data = np.rot90(face_data,k=sense)
        self.set_face(face,face_data)

        # Get other faces
        linked_faces,slices = zip(*FACES_LINK[face])
        slices_data = [np.copy(self.get_face(linked_faces[i])[slices[i]]) for i in range(4)]

        # Rotate arrays
        slices_data = self._rotate_array(slices_data,clockwise = clockwise)

        # Set new rotated arrays
        for i in range(4):
            face = linked_faces[i]
            face_data = self.get_face(face)
            face_data[slices[i]] = slices_data[i]
            self.set_face(face,face_data)
        


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
        if array.shape == (3,3):
            self.array = array
        else:
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



