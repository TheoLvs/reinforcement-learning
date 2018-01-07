#!/usr/bin/env python
# -*- coding: utf-8 -*-


"""--------------------------------------------------------------------
CHROME DINO
Started on the 2018/01/03
theo.alves.da.costa@gmail.com
https://github.com/theolvs
------------------------------------------------------------------------
"""

import os
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import sys
import time
from tqdm import tqdm
import pyautogui
import random
import cv2
from PIL import Image,ImageGrab

import torch





#=================================================================================================================================
# GAME
#=================================================================================================================================



class DinoGame(object):
    def __init__(self):
        pass

    #--------------------------------------------------------------------
    # HELPER FUNCTIONS

    def click_screen(self):
        pyautogui.click(x=1000,y = 500)


    def refresh_page(self):
        self.click_screen()
        pyautogui.press("f5")


    def move(self,action = "up"):
        if action is not None:
            pyautogui.press(action)


    #--------------------------------------------------------------------
    # GRAB GAME FUNCTIONS


    def grab_roi(self):
        return ImageGrab.grab(bbox=(1020,350,1850,600))


    def grab_game(self,how = "all"):
        img = self.grab_roi()
        img_array = np.array(img)

        # Edges
        img_edges = cv2.Canny(img_array,100,200)

        # Mask
        img_mask = self._extract_game(img_edges)

        # Morphological
        img_eroded = cv2.erode(img_mask,np.ones((4,1),np.uint8))
        img_dilated = cv2.dilate(img_eroded,np.ones((10,10),np.uint8))

        # Contours
        img_contours,contours,_ = cv2.findContours(img_dilated.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        img_contours = cv2.cvtColor(img_contours,cv2.COLOR_GRAY2BGR)
        cv2.drawContours(img_contours, contours, -1, (0,0,255), 3)
        img_lines = img_contours.copy()
        xs = list(sorted([y.min(axis = 0)[0][0] for y in contours]))
        for x in xs:
            cv2.line(img_lines,(x,0),(x,img_lines.shape[0]),(0,0,255),1)

        # Output
        imgs = {
            "raw":img_array,
            "edges":img_edges,
            "mask":img_mask,
            "morph":img_dilated,
            "contours":img_lines,
        }
        if how != "all" and how in imgs.edges():
            imgs = imgs[how]

        return imgs,xs


    def _extract_game(self,img):
        mask = np.zeros_like(img)
        mask[80:220,130:800] = 255
        masked = cv2.bitwise_and(img,mask)
        return masked



    #--------------------------------------------------------------------
    # RUN THE GAME


    def run_episode(self,render = None,policy = None,**kwargs):

        # Episode initialization
        roi_array = np.zeros_like(np.array(self.grab_roi()))
        self.click_screen()
        time.sleep(0.1)
        pyautogui.press("up")
        t = time.time()

        # Episode main loop
        while True:

            # Data acquisition
            imgs,xs = self.grab_game()

            # Score
            score = (time.time() - t)*10

            # Action
            self.act(imgs,xs,score,policy,**kwargs)

            # Rendering
            if render is not None and render in imgs:
                cv2.imshow(render,imgs[render])
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

            # Condition of stop
            if score > 20 and (imgs["raw"] == roi_array).all():
                break
            roi_array = imgs["raw"]

        # Score
        score = (time.time() - t)*10
        return score



    def run_generation(self):
        pass



    def run_game(self):
        pass



    #--------------------------------------------------------------------
    # ACTIONS


    def act(self,imgs,xs,score,policy = None,**kwargs):
        

        if policy == "random":
            action = random.choice(["up","down",None])
            self.move(action)

        elif policy == "rules":
            th = 300 if "th" not in kwargs else kwargs["th"]
            th = max(th - score/100,150)
            if len(xs) > 0 and xs[0] < th:
                self.move("up")

        else:
            pass






#=================================================================================================================================
# ELEMENT
#=================================================================================================================================


class Dino(object):
    def __init__(self):
        self.score = None

    def __add__(self):
        pass


    def mutate(self):
        pass


    def evaluate(self):
        return self.score


    def set_score(self,score):
        self.score = score








#=================================================================================================================================
# POPULATION
#=================================================================================================================================


class Population(object):
    def __init__(self,dinos = None,n = 20):

        if dinos is None:
            self.dinos = [Dino() for i in range(n)]
        else:
            self.dinos = dinos


    def __getitem__(self,key):
        if type(key) == tuple or type(key) == list:
            d = []
            for i in key:
                d.append(self.dinos[i])
            return d
        else:
            return self.dinos[key]
    
    def __iter__(self):
        return iter(self.dinos)
    
    def __len__(self):
        return len(self.dinos)



    def evaluate(self):
        fitnesses = [(i,dist.evaluate()) for i,dist in enumerate(self)]
        indices,fitnesses = zip(*sorted(fitnesses,key = lambda x : x[1],reverse = True))
        return indices,fitnesses



    def selection(self,top = 0.5):
        indices,fitnesses = self.evaluate()
        n = int(top*len(fitnesses))
        return indices[:n]


    def crossover(self):
        pass


    def mutate(self):
        for d in self:
            d.mutate()


    def evolve(self):
        pass










#=================================================================================================================================
# RUN
#=================================================================================================================================




if __name__ == "__main__":

    game = DinoGame()
    game.run_episode(render = "contours",policy  = "rules",th = 280)