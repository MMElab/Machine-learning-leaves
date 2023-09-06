# -*- coding: utf-8 -*-
"""
Created on Tue Aug  1 16:02:29 2023

@author: vinkjo
"""
import cv2
import numpy as np
from parameters import plugradius

def filter_contours_by_size(contours, min_contour_area=100):
    filtered_contours = []
    for contour in contours:
        contour_area = cv2.contourArea(contour)
        if contour_area >= min_contour_area:
            filtered_contours.append(contour)
    return filtered_contours

def petrifinder(petriprobabilityimage):
    petriuint8 = cv2.blur(petriprobabilityimage, (50, 50))
    petriuint8 = np.uint8(petriuint8*255)
    detected_circles = cv2.HoughCircles(petriuint8,cv2.HOUGH_GRADIENT, 1.5, 3000, param1 = 50,param2 = 30, minRadius = 1000, maxRadius = 2000)
    petriimage = np.zeros(np.shape(petriuint8))
    a = int(detected_circles[0][0][0])
    b = int(detected_circles[0][0][1])
    r = int(detected_circles[0][0][2])
    petriimage = cv2.circle(petriimage,(a,b),r,1,-1)
    petriarea = np.pi*r**2
    return petriimage,[a,b],r,petriarea

def leaffinder(image):
    # Convert the image to grayscale
    gray = np.uint8(255*image)

    # Apply GaussianBlur to reduce noise and improve contour detection
    blurred = cv2.GaussianBlur(gray, (51, 51), 0)

    # Threshold the image to create a binary image (black and white)
    _, thresholded = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)

    # Find contours in the binary image
    contours, _ = cv2.findContours(thresholded, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = filter_contours_by_size(contours, min_contour_area=5000)
    # Draw the contours on a copy of the original image
    image_with_contours = np.zeros_like(image)
    leafareas = list()
    for i in range(0,len(contours)):
        cv2.drawContours(image_with_contours, contours, i, i+1, thickness=cv2.FILLED)
        leafareas.append(cv2.contourArea(contours[i]))
    return contours, image_with_contours, leafareas

def plugfinder(leafimage,plugprobabilityimage,leafindexes=[]):
    plugcentredict = dict()
    plugleafdict = dict()
    blur = cv2.GaussianBlur(plugprobabilityimage,(plugradius*2+1,plugradius*2+1),plugradius)
    plugimage = np.zeros(np.shape(leafimage))
    if not leafindexes:
        leafindexes = np.unique(leafimage[leafimage!=0])
    for i in leafindexes:
        minvalue = np.min(blur[leafimage==i])
        centre = np.where(blur==minvalue)
        plugcentredict[int(i)]=[int(centre[1][0]),int(centre[0][0])]
        plugleafdict[int(i)]=int(i)
        plugimage = cv2.circle(plugimage,(centre[1][0],centre[0][0]),plugradius,int(i),-1)
    return plugimage,plugcentredict,plugleafdict

def find_necrosis_pixels(leafimage, leafnumber,center,radius):
    circlemask = cv2.circle(np.zeros_like(leafimage),center,radius,1,-1)
    necrosis_pixels = (leafimage==leafnumber) & (circlemask>0)
    return necrosis_pixels

def necrosisfinder(leafimage, necrosisprobabilityimage,plugcentredict,leafindexes=[]):
   minimumradius =50
   filterradius = 250
   maximumradius = 1000
   interval = 30
   necrosisfilterseed = 0.2
   necrosisfilter = 0.5   
   necrosisradiusdict = dict()
   necrosisareadict = dict()
   blurrednecrosisimage = cv2.blur(necrosisprobabilityimage,(100,100))
   
   if not leafindexes:
       leafindexes = np.unique(leafimage[leafimage!=0])

   oldnecrosis_pixels = np.zeros_like(necrosisprobabilityimage)
   for i in leafindexes:
       necrosisradiusdict[i]=0
       necrosisareadict[i]=0
       # Filter 
       necrosis_pixels = find_necrosis_pixels(leafimage,i,plugcentredict[i],filterradius)
       if necrosis_pixels.any():
           minprobnecrosis = np.min(blurrednecrosisimage[necrosis_pixels])
           if minprobnecrosis<necrosisfilterseed:
               for radius in range(minimumradius, maximumradius,interval):
                   necrosis_pixels = find_necrosis_pixels(leafimage,i,plugcentredict[i],radius)
                   thistimenecrosis_pixels = (necrosis_pixels & ~ (oldnecrosis_pixels==i))
                   average_necrosis_pixels = np.average(blurrednecrosisimage[thistimenecrosis_pixels])
                   if average_necrosis_pixels>necrosisfilter:
                       break
                   oldnecrosis_pixels[necrosis_pixels!=0] = i
                   necrosisradiusdict[i]=radius
                   necrosisareadict[i]=np.sum(oldnecrosis_pixels==i)
               
   return oldnecrosis_pixels, necrosisradiusdict, necrosisareadict