#!/usr/bin/env python
# -*- coding: utf-8 -*- 



"""--------------------------------------------------------------------
PREPROCESSING
Started on the 30/10/2017

theo.alves.da.costa@gmail.com
https://github.com/theolvs
------------------------------------------------------------------------
"""



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
    img = np.copy(img)
    for line in lines:
        coords = line[0]
        cv2.line(img, (coords[0], coords[1]), (coords[2], coords[3]), [255,255,255], 2)
    return img


def detect_hough_lines(img,alpha = 100,min_length = 200,max_gap = 15):
    lines = cv2.HoughLinesP(img, 1, np.pi/180, alpha,min_length,max_gap)
    img = draw_lines(img,lines)
    return img