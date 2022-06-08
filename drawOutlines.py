# -*- coding: utf-8 -*-
"""
Created on Wed Mar 30 09:56:39 2022

@author: vinkjo
"""
import cv2
import numpy as np

def drawOutlines(impath,outpath,leafimage,necrosisimage,petriimage,plugimage):
    im = cv2.imread(impath)
    cv2.imshow('image',im)
    contours,_ = cv2.findContours(np.uint8(leafimage),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for i in contours:
        cv2.drawContours(im,i,-1,(0,255,0),thickness=5)
    contours,_ = cv2.findContours(np.uint8(necrosisimage),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for i in contours:
        cv2.drawContours(im,i,-1,(0,0,0),thickness=5)
    contours,_ = cv2.findContours(np.uint8(petriimage),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for i in contours:
        cv2.drawContours(im,i,-1,(0,0,255),thickness=5)
    contours,_ = cv2.findContours(np.uint8(plugimage),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for i in contours:
        cv2.drawContours(im,i,-1,(255,0,0),thickness=5)        
    cv2.imwrite(outpath,im)