# -*- coding: utf-8 -*-
"""
Created on Thu Jun  2 15:55:29 2022

@author: vinkjo
"""
import cv2
import numpy as np
import h5py
from skimage import measure

trial = np.load('C://Users//vinkjo//OneDrive - Victoria University of Wellington - STAFF//Desktop//Machine learning Leaves//Raw data//3770//030322 1 D8_h5//Objects_leaf//IMG_0777.JPG_Object Identities.npy')
trialpixel = h5py.File('C://Users//vinkjo//OneDrive - Victoria University of Wellington - STAFF//Desktop//Machine learning Leaves//Raw data//3770//030322 1 D8_h5//Pixelprobabilities_necrosis//IMG_0777.JPG_Probabilities.h5')
trialpixel = np.squeeze(np.array(trialpixel['exported_data']))[:,:,1]

def plugfinder():
    radius = 50
    blur = cv2.GaussianBlur(trialpixel,(radius*2+1,radius*2+1),radius)
    plugmask = np.zeros(np.shape(trial))
    for i in range(1,np.max(trial)+1):
        maxvalue = np.max(blur[trial==i])
        centre = np.where(blur==maxvalue)
        cv2.circle(plugmask,(centre[1][0],centre[0][0]),radius,i,-1)
    
        
def necrosisfinder(necrosisimage,necrosisprobabilityimage):
    threshold = 0.5
    necrosismask = np.zeros(np.shape(trial))
    for i in range(1,np.max(trial)+1):
        contours = cv2.findContours(np.uint8(trial==i),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)    
        ellips= cv2.fitEllipse(contours[0][0])
        image_center = ellips[0]
        angle = ellips[2]
        rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
        rot_matinv = cv2.getRotationMatrix2D(image_center, -angle, 1.0)
        result = cv2.warpAffine(np.uint8(trial==i), rot_mat, trial.shape[1::-1], flags=cv2.INTER_LINEAR)
        minvalue = np.min(np.where(result==1)[0])
        maxvalue = np.max(np.where(result==1)[0])
        segments = range(minvalue,maxvalue,int((maxvalue-minvalue)/50))
        segmentmask = np.zeros(np.shape(trial))
        coords = np.where(result==1)
        for i in range(1,len(segments)):
           segmentcoords = (coords[0][coords[0]>=(segments[i-1]-1)],coords[1][coords[0]>=(segments[i-1]-1)])
           segmentmask[segmentcoords]=i
        segmentmaskrot = cv2.warpAffine(segmentmask,rot_matinv, trial.shape[1::-1], flags=cv2.WARP_FILL_OUTLIERS)
        #segmentmaskrot = np.round(segmentmaskrot)
        necrosislist = list()
        for i in range(1,len(segments)):
            rotcoords = (segmentmaskrot==i)
            avnecrosis = np.mean(trialpixel[rotcoords])
            if avnecrosis > threshold:
                necrosismask[rotcoords]=1
                necrosislist.append(i)
        for i in range(1,len(segments)):
            if i not in necrosislist:
                if (i+1 in necrosislist and i-1 in necrosislist) or (i+2 in necrosislist and i-2 in necrosislist):
                    necrosislist.append(i)
                    rotcoords = (segmentmaskrot==i)
                    necrosismask[rotcoords]=1
    necrosismask = measure.label(necrosismask)