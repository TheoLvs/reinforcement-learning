#!/usr/bin/env python
# -*- coding: utf-8 -*- 



"""--------------------------------------------------------------------
PREPROCESSING
Started on the 30/10/2017


https://medium.com/@galen.ballew/opencv-lanedetection-419361364fc0
theo.alves.da.costa@gmail.com
https://github.com/theolvs
------------------------------------------------------------------------
"""


import numpy as np
import cv2
from PIL import Image

#=============================================================================================================================
# HELPER FUNCTIONS
#=============================================================================================================================


def to_black_and_white(img):
    return cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)


def detect_edges(img,threshold1 = 200,threshold2 = 300):
    return cv2.Canny(img, threshold1 = threshold1, threshold2=threshold2)


def gaussian_smooth(img):
    return cv2.GaussianBlur(img,(5,5),0)


def select_part_from_mask(img,vertices):
    vertices = np.array([vertices],dtype = np.int32)
    
    mask = np.zeros_like(img)
    
    # fill the mask
    cv2.fillPoly(mask, vertices, 255)
    
    # now only show the area that is the mask
    masked = cv2.bitwise_and(img, mask)
    return masked


def draw_lines(img,lines):
    # img = np.copy(img)
    img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
    for line in lines:
        coords = line[0]
        cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [255,0,0], 1)
    return img


def detect_hough_lines(img,alpha = 100,min_length = 200,max_gap = 15,main_lines = True):
    lines = cv2.HoughLinesP(img, 1, np.pi/180, alpha,np.array([]),min_length,max_gap)
    if lines is not None:

        if main_lines:
            left,right = detect_main_lines(lines)
            lines = np.vstack([x for x in [left,right] if len(x) > 0])
            lines = np.int32(np.expand_dims(np.vstack(lines),axis = 1))
        else:
            left,right = categorize_lines(lines)

        img = draw_lines(img,lines)
        return img,lines,left,right
    else:
        return img,[],[],[]


def detect_main_lines(lines):
    """
    line is x_top,y_top,x_bottom,y_bottom
    """

    left,right = categorize_lines(lines)

    if len(left) > 0:
        left = left.mean(axis = 1).astype('int32')

    if len(right) > 0:
        right = right.mean(axis = 1).astype('int32')

    return left,right


def categorize_lines(lines):

    left = []
    right = []
    for line in lines:
        y_left,y_right = line[0][1],line[0][3]
        if y_left > y_right:
            left.append(line[0])
        else:
            right.append(line[0])

    if len(left) > 0: left = np.expand_dims(np.vstack(left),axis = 0)
    if len(right)> 0: right = np.expand_dims(np.vstack(right),axis = 0)
    return left,right



def intersection_lines(left,right):
    get_coord = lambda v : (v[0],v[2],v[1],v[3])
    get_a = lambda x1,x2,y1,y2 : (y2-y1)/(x2-x1)
    get_b = lambda x1,x2,y1,y2 : (y1*x2 - y2*x1)/(x2-x1)
    get_x = lambda a1,a2,b1,b2 : (b2-b1)/(a1-a2)

    left = get_coord(left[0])
    right = get_coord(right[0])

    a_left,a_right = get_a(*left),get_a(*right)
    b_left,b_right = get_b(*left),get_b(*right)

    return get_x(a_left,a_right,b_left,b_right)




#=============================================================================================================================
# CLASS REPRESENTATION
#=============================================================================================================================




class CameraImage(object):
    def __init__(self,image):
        self.img = image
        self.array = self.to_array()
        
    def to_array(self):
        return np.array(self.img)
    
    def set_array(self,array):
        self.array = array
        self.img = Image.fromarray(array)
        
    def _repr_png_(self):
        return self.img._repr_png_()


    def preprocess(self,threshold = 50,min_length = 50,main_lines = True):

        # Basic preprocessing
        img = to_black_and_white(self.array)
        img = detect_edges(img,300,600)
        img = gaussian_smooth(img)

        # Select from mask
        vertices = [[0,140],[0,60],[320,60],[320,140]]
        img = select_part_from_mask(img,vertices)
        
        # Detect lanes
        img,lines,left,right = detect_hough_lines(img,threshold,min_length = min_length,max_gap = 1,main_lines = main_lines)

        self.set_array(img)

        return left,right


    def act(self):
        pass
        
