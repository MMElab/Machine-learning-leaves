# -*- coding: utf-8 -*-
# """
# Created on Mon Apr  4 15:54:48 2022

# @author: vinkjo
# """
## This script allows you to fix mistakes in the automated classification manually
# There are several options 

# Import functions
import cv2
import numpy as np
import pandas as pd
import os
from pathlib import Path
#import skimage.draw
import matplotlib.pyplot as plt
import h5py
from tkinter import *
from tkinter.filedialog import askopenfilename
from tkinter import filedialog
import sys
import imutils
sys.path.append('C:\\Users\\vinkjo\\OneDrive - Victoria University of Wellington - STAFF\\Desktop\\Machine learning Leaves')
from Polygon import PolygonDrawer
from parameters import plugradius
import featurefinderfunctions as fff

# Number of leaves possible in image
amountofleaves = [3,5]

# Functions
# Draws plug around centre that you click on in the image with radius size specified (Default 50)
def find_smallest_missing_integer(lst):
    # Convert the list to a set for faster lookup
    num_set = set(lst)
    # Start checking from 1
    smallest_integer = 1

    while True:
        if smallest_integer not in num_set:
            return smallest_integer
        smallest_integer += 1

def draw_circle(event,x,y,flags,param):
    coordlist = param
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y),plugradius,(255,0,0),5)
        cv2.imshow('image',img)
        print(f"Clicked at x={x}, y={y}")
        mouseX,mouseY = x,y
        coordlist.append([x,y])
    return coordlist

def draw_plug(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDBLCLK:
        img = cv2.imread(impath)
        x,y = y,plugimage.shape[0]-x
        leaf=leafimage[y,x]
        if leaf == 0:
            print("Click where there is a leaf")
        else:
            print(f"Clicked at x={x}, y={y}")
            plugcentredict[leaf] = [x,y] 
            plugimage[plugimage==leaf]=0
            plugpixels = cv2.circle(plugimage,(x,y),50,100,-1)
            plugimage[plugpixels==100]=leaf
            contours,_ = cv2.findContours(np.flip(np.uint8(plugimage.T),axis=1),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img,contours,-1,(255,0,0),thickness=5)
            cv2.imshow('image',img) 
        
def draw_necrosis(event,x,y,flags,param):   
    if event == cv2.EVENT_LBUTTONDBLCLK:
        img = cv2.imread(impath)
        x,y = y,necrosisimage.shape[0]-x
        leaf=leafimage[y,x]
        if leaf == 0:
            print("Click where there is a leaf")
        else:
            print(f"Clicked at x={x}, y={y}")
            plugcentre = plugcentredict[leaf]
            radius = np.sqrt((x-plugcentre[0])**2+(y-plugcentre[1])**2)
            radiusdict[leaf]= radius
            necrosisimage[necrosisimage==leaf]=0
            necrosispixels = fff.find_necrosis_pixels(leafimage, leaf, plugcentre, int(radius))
            necrosisimage[necrosispixels==1]=leaf
            contours,_ = cv2.findContours(np.flip(np.uint8(necrosisimage.T),axis=1),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img,contours,-1,(0,0,0),thickness=5)
        cv2.imshow('image',img)       
    if event == cv2.EVENT_RBUTTONDBLCLK:
        img = cv2.imread(impath)
        x,y = y,necrosisimage.shape[0]-x
        leaf=leafimage[y,x]
        if leaf == 0:
            print("Click where there is a leaf")
        else:
            print(f"Removed necrosis from leaf #{leaf}")
            print(f"Clicked at x={x}, y={y}")
            radiusdict[leaf]= 0
            necrosisimage[necrosisimage==leaf]=0
            contours,_ = cv2.findContours(np.flip(np.uint8(necrosisimage.T),axis=1),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img,contours,-1,(0,0,0),thickness=5)
            cv2.imshow('image',img)       

def leaf_remover(event,x,y,flags,param):
    global leafimage
    if event == cv2.EVENT_RBUTTONDBLCLK:
        img = cv2.imread(impath)
        x,y = y,necrosisimage.shape[0]-x
        leaf=leafimage[y,x]
        if leaf == 0:
            print("Click where there is a leaf")
        else:
            print(f"Removed leaf #{leaf}")
            print(f"Clicked at x={x}, y={y}")
            leafimage[leafimage==leaf]=0
            contours,_ = cv2.findContours(np.flip(np.uint8(leafimage.T),axis=1),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(img,contours,-1,(0,255,0),thickness=5)
            cv2.imshow('image',img)       


def evaluate_dict_of_strings(dictionary):
    evaluated_dict = {}
    for key, value in dictionary.items():
        # Split the string value by commas and convert each part to an integer
        tuple_values = eval(value)
        evaluated_dict[key] = tuple_values
    return evaluated_dict
        
def find_necrosis_pixels_on_click(leafimage,plugcentredict):
    global necrosisimage
    global img
    global radiusdict
    necrosisimageoriginal = necrosisimage.copy()
    k = ord('n')
    while k == ord('n'):
        radiusdict = dict()
        necrosisimage = necrosisimageoriginal.copy()
        img = cv2.imread(impath)
        leafcontours,_ = cv2.findContours(np.flip(np.uint8(leafimage.T),axis=1),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img,leafcontours,-1,(0,255,0),thickness=5)
        cv2.namedWindow('image',cv2.WINDOW_NORMAL)
        cv2.setMouseCallback('image',draw_necrosis)
        contours,_ = cv2.findContours(np.flip(np.uint8(necrosisimage.T),axis=1),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img,contours,-1,(0,0,0),thickness=5)
        cv2.imshow('image',img)
        k = cv2.waitKey(0)
        if k == ord('q'):
            necrosisimage = necrosisimageoriginal.copy()
            radiusdict = dict()
        cv2.destroyAllWindows()
    return necrosisimage, radiusdict

def find_plug_pixels_on_click(leafimage):
    global plugimage
    global plugcentredict
    global img
    plugcentredictoriginal = plugcentredict.copy()
    plugimageoriginal = plugimage.copy()
    k = ord('n')
    while k == ord('n'):
        plugcentredict = plugcentredictoriginal.copy()
        plugimage = plugimageoriginal.copy()
        img = cv2.imread(impath)
        cv2.namedWindow('image',cv2.WINDOW_NORMAL)  
        cv2.setMouseCallback('image',draw_plug)
        contours,_ = cv2.findContours(np.flip(np.uint8(plugimage.T),axis=1),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(img,contours,-1,(255,0,0),thickness=5)
        cv2.imshow('image',img)
        k = cv2.waitKey(0)
        if k == ord('q'):
            plugimage = plugimageoriginal.copy()
            plugcentredict = plugcentredictoriginal.copy()
        cv2.destroyAllWindows()
    return plugimage, plugcentredict
# def necrosisfinder(leafimage,necrosisprobabilityimage,leafnumber,necrosismask=0,manual=[]):

#     threshold = 0.6
#     contours = cv2.findContours(np.uint8(leafimage==leafnumber),cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)    
#     ellips= cv2.fitEllipse(contours[0][0])
#     image_center = ellips[0]
#     angle = ellips[2]
#     global rot_mat, rot_matinv
#     rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
#     rot_matinv = cv2.getRotationMatrix2D(image_center, -angle, 1.0)
#     result = cv2.warpAffine(np.uint8(leafimage==leafnumber), rot_mat, leafimage.shape[1::-1], flags=cv2.INTER_LINEAR)
#     minvalue = np.min(np.where(result==1)[0])
#     maxvalue = np.max(np.where(result==1)[0])
#     if len(manual)>0:
#         startpoint = manual[0]
#         endpoint = manual[1]
#         startcoords = np.dot(rot_mat[0:2,0:2],startpoint)+rot_mat[:,2]
#         endcoords = np.dot(rot_mat[0:2,0:2],endpoint)+rot_mat[:,2]
#         start = int(np.max([0,np.min([startcoords[1],endcoords[1]])]))
#         end = int(np.max([startcoords[1],endcoords[1]]))
#         result[start:end]=2
#         resultrot = cv2.warpAffine(result,rot_matinv, leafimage.shape[1::-1], flags=cv2.WARP_FILL_OUTLIERS)   
#         resultrot[leafimage!=leafnumber]=0
#         resultrot[resultrot==1]=0
#         resultrot[resultrot==2]=1
#         necrosismask = resultrot
#     else:
#         segments = range(minvalue,maxvalue,int((maxvalue-minvalue)/50))
#         segmentmask = np.zeros(np.shape(leafimage))
#         coords = np.where(result==1)
#         for i in range(1,len(segments)):
#            segmentcoords = (coords[0][coords[0]>=segments[i-1]],coords[1][coords[0]>=segments[i-1]])
#            segmentmask[segmentcoords]=i
#         segmentmaskrot = cv2.warpAffine(segmentmask,rot_matinv, leafimage.shape[1::-1], flags=cv2.WARP_FILL_OUTLIERS)
#         necrosislist = list()
#         for i in range(1,len(segments)):
#             rotcoords = (segmentmaskrot==i)
#             avnecrosis = np.mean(necrosisprobabilityimage[rotcoords])
#             if avnecrosis > threshold:
#                 necrosismask[rotcoords]=1
#                 necrosislist.append(i)
#         for i in range(1,len(segments)):
#             if i not in necrosislist:
#                 if (i+1 in necrosislist and i-1 in necrosislist) or (i+2 in necrosislist and i-2 in necrosislist):
#                     necrosislist.append(i)
#                     rotcoords = (segmentmaskrot==i)
#                     necrosismask[rotcoords]=1
#         necrosismask = measure.label(necrosismask)
#     return necrosismask

def resizeimage(img, screen_width, screen_height):
    
    img_height, img_width, _ = img.shape
    aspect_ratio = img_width / img_height
# Determine the new dimensions to fit the screen
    if img_width > screen_width or img_height > screen_height:
        if img_width > screen_width:
            new_width = screen_width
            new_height = int(new_width / aspect_ratio)
        else:
            new_height = screen_height
            new_width = int(new_height * aspect_ratio)
    else:
        new_width = img_width
        new_height = img_height

    # Resize the image using OpenCV
    resized_img = cv2.resize(img, (new_width, new_height))
    return resized_img


# def draw_plug(impath): 
#     global img
#     #img = h5py.File(impath)
#     #img = np.squeeze(img['data'][0])
#     #img = img[:,:,::-1]
#     k = ord('n')
#     while k == ord('n'):
#         coordlist = []
#         img = cv2.imread(impath)
#         cv2.namedWindow('image',cv2.WINDOW_NORMAL) 
#         cv2.setMouseCallback('image',lambda event, x, y, flags, param: draw_circle(event, x, y, flags, (coordlist)))
#         cv2.imshow('image',img)
#         k = cv2.waitKey(0)
#         cv2.destroyAllWindows()
#     # while(1):
#     #     cv2.imshow('image',img)
#     #     k = cv2.waitKey(20) & 0xFF
#     #     if k == ord('q'):
#     #         cv2.destroyAllWindows()
#     #         break
#     #     elif k == ord('n'):
#     #         cv2.destroyAllWindows()
#     #         pluglist = []
#     #         img = h5py.File(impath)
#     #         img = np.squeeze(img['data'][0])
#     #         img = img[:,:,::-1]
#     #         #img = cv2.imread(impath)
#     #         cv2.namedWindow('image',cv2.WINDOW_NORMAL)
#     #         cv2.setMouseCallback('image',draw_circle)
#     return coordlist

def drawOutlines(impath,outpath,leafimage,necrosisimage,petriimage,plugimage):
    # im = h5py.File(impath)
    # im = np.squeeze(im['data'][0])
    # im = im[:,:,::-1]
    im = cv2.imread(impath)
    if len(im) == 3024:
        im = cv2.rotate(im, cv2.cv2.ROTATE_90_CLOCKWISE)
    contours,_ = cv2.findContours(np.flip(np.uint8(leafimage.T),axis=1),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for i in contours:
        cv2.drawContours(im,i,-1,(0,255,0),thickness=5)
    contours,_ = cv2.findContours(np.flip(np.uint8(necrosisimage.T),axis=1),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for i in contours:
        cv2.drawContours(im,i,-1,(0,0,0),thickness=5)
    contours,_ = cv2.findContours(np.flip(np.uint8(petriimage.T),axis=1),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for i in contours:
        cv2.drawContours(im,i,-1,(0,0,255),thickness=5)
    contours,_ = cv2.findContours(np.flip(np.uint8(plugimage.T),axis=1),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for i in contours:
        cv2.drawContours(im,i,-1,(255,0,0),thickness=5)        
    cv2.imwrite(outpath,im)
    return 

def drawOutlines_display(impath,leafimage,necrosisimage,petriimage,plugimage):
    #im = h5py.File(impath)
    #im = np.squeeze(im['data'][0])
   # im = im[:,:,::-1]
    im = cv2.imread(impath)
    if len(im) == 3024:
        im = cv2.rotate(im, cv2.cv2.ROTATE_90_CLOCKWISE)
    contours,_ = cv2.findContours(np.flip(np.uint8(leafimage.T),axis=1),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for i in contours:
        cv2.drawContours(im,i,-1,(0,255,0),thickness=5)
    contours,_ = cv2.findContours(np.flip(np.uint8(necrosisimage.T),axis=1),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for i in contours:
        cv2.drawContours(im,i,-1,(0,0,0),thickness=5)
    contours,_ = cv2.findContours(np.flip(np.uint8(petriimage.T),axis=1),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for i in contours:
        cv2.drawContours(im,i,-1,(0,0,255),thickness=5)
    contours,_ = cv2.findContours(np.flip(np.uint8(plugimage.T),axis=1),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    for i in contours:
        cv2.drawContours(im,i,-1,(255,0,0),thickness=5) 
    return im

def addorremoveobject(filein,filename,folder):
    global leafimage,petriimage,necrosisimage,plugimage, impath, plugcentredict
    impath = filein
    processimage = 'y'
    while processimage =='y':
        h5folder = os.path.dirname(filein) + '_h5'+os.path.sep
        # Petri data
        petrifilename = h5folder+filename+'_petri_classification.npy'
        petriimage = np.squeeze(np.load(petrifilename))
        
        #Leaf data
        leaffilename = h5folder+filename+'_leaf_classification.npy'
        leafimage = np.squeeze(np.load(leaffilename))
        
        # Plug data
        plugfilename = h5folder+filename+'_plug_classification.npy'
        plugimage = np.squeeze(np.load(plugfilename))
        
        # Necrosis data
        necrosisfilename = h5folder+filename+'_necrosis_classification.npy'
        necrosisimage = np.squeeze(np.load(necrosisfilename))
        
        
        # petrifilename = folder +'Objects_petri/'+filename+'.JPG_table.csv'
        # petricsv = pd.read_csv(petrifilename)
        # petriimagefilename = folder +'Objects_petri/'+filename+'.JPG_Object Identities.npy'
        # petriimage = np.squeeze(np.load(petriimagefilename))
        # petriimage[petriimage!=int(petricsv['labelimage_oid'])]=0
        
        # Leaf data
        # leafcsvfilename = folder +'Objects_leaf/'+filename+'.JPG_table.csv'
        # leafcsv = pd.read_csv(leafcsvfilename)
        # leafimagefilename = folder +'Objects_leaf/'+filename+'.JPG_Object Identities.npy'
        # leafimage = np.squeeze(np.load(leafimagefilename))
        # leafprobabilityimage = h5py.File(folder +'Pixelprobabilities_leaf/'+filename+'.JPG_Probabilities.h5')
        # leafprobabilityimage = np.squeeze(leafprobabilityimage['exported_data'])[:,:,0]

        # Plug data
        # plugcsvfilename = folder +'Objects_plug/'+filename+'.JPG_table.csv'
        # plugcsv = pd.read_csv(plugcsvfilename)
        # plugimagefilename = folder +'Objects_plug/'+filename+'.JPG_Object Identities.npy'
        # plugimage = np.squeeze(np.load(plugimagefilename))
        # plugprobabilityimage = h5py.File(h5folder +'Pixelprobabilities_plug/'+filename+'.JPG_Probabilities.h5')
        # plugprobabilityimage = np.squeeze(plugprobabilityimage['exported_data'])[:,:,1]
        # necrosisprobabilityimage = h5py.File(h5folder +'Pixelprobabilities_necrosis/'+filename+'.JPG_Probabilities.h5')
        # necrosisprobabilityimage = np.squeeze(necrosisprobabilityimage['exported_data'])[:,:,1]
        #Necrosis data
        # necrosiscsvfilename = folder +'Objects_necrosis/'+filename+'.JPG_table.csv'
        # necrosiscsv = pd.read_csv(necrosiscsvfilename)
        # necrosisimagefilename = folder +'Objects_necrosis/'+filename+'.JPG_Object Identities.npy'
        # necrosisimage = np.squeeze(np.load(necrosisimagefilename))
        
        #Summaryoutput
        summaryoutputfilename = h5folder+filename+'_Summaryoutput.csv'
        summaryoutput = pd.read_csv(summaryoutputfilename)
        plugcentredict = dict(zip(summaryoutput.leafnumber, summaryoutput.plugcentre))
        plugcentredict = evaluate_dict_of_strings(plugcentredict)
        necrosisradiusdict = dict(zip(summaryoutput.leafnumber, summaryoutput.necrosisradius))                 
        value2=input('Which object do you want to modify(p(lug)/d(ish)/l(eaf)/n(ecrosis))?')                      
        
        # Fix necrosis
        if value2 == 'n':
            necrosisareadict = dict()
            print("Double click on edge of necrosis to add (y to save, q to exit, n to try again)")
            necrosisimage, radiusdict = find_necrosis_pixels_on_click(leafimage,plugcentredict)        
            for i in np.unique(necrosisimage):
                necrosisareadict[i] = np.sum(necrosisimage==i)
            
            necrosisradiusdict.update(radiusdict)
            summaryoutput['necrosisarea'] = summaryoutput['leafnumber'].map(necrosisareadict)
            summaryoutput['necrosisradius'] = summaryoutput['leafnumber'].map(necrosisradiusdict)
            np.save(necrosisfilename,np.uint8(necrosisimage))
            summaryoutput.to_csv(summaryoutputfilename,index=False)
        
        # Fix plug
        elif value2 == 'p':    
            plugimage, plugcentredict = find_plug_pixels_on_click(leafimage)
            summaryoutput['plugcentre'] = summaryoutput['leafnumber'].map(plugcentredict)
            print("Double click on middle of plug to add (y to save, q to exit, n to try again)")
            np.save(plugfilename,np.uint8(plugimage))
            summaryoutput.to_csv(summaryoutputfilename,index=False)
        
        # Fix leaf (also adds plug and potential necrosis at same time)
        elif value2 == 'l':
             
            # Remove leaves
             originalleafimage = leafimage.copy() 
             k = ord('n')
             exitcall = 0
             adding = 0
             while k == (ord('n')):
                 leafimage = originalleafimage.copy()
                 img = cv2.imread(impath)
                 cv2.namedWindow('image',cv2.WINDOW_NORMAL)
                 contours,_ = cv2.findContours(np.flip(np.uint8(leafimage.T),axis=1),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                 cv2.drawContours(img,contours,-1,(0,255,0),thickness=5)
                 cv2.setMouseCallback("image", leaf_remover)
                 cv2.imshow('image',img) 
                 print('Double click with right mouse button to remove leaves')
                 print("Press a to move towards adding leaves")
                 print("Press y if done")
                 k = cv2.waitKey(0)
                 if k == ord('q'):
                     cv2.destroyAllWindows()
                     exitcall = 1
                     break
                 elif k == ord('n'):
                     continue
                 elif k == ord('a'):
                     adding = 1
             if exitcall == 1:
                return
                
            # Remove from summaryoutput leaves no longer in leafimage
             summaryoutputold = summaryoutput.copy()
             summaryoutput = summaryoutput[summaryoutput['leafnumber'].isin(np.unique(leafimage))]
             plugimage[leafimage==0]=0
             necrosisimage[leafimage==0]=0
             cv2.destroyAllWindows()
             
             # Draw in leaves
             if adding == 1:
                 img = cv2.imread(impath)
                 contours,_ = cv2.findContours(np.flip(np.uint8(leafimage.T),axis=1),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                 cv2.drawContours(img,contours,-1,(0,255,0),thickness=5)
                 k = ord('n')
                 pointscollection = []
                 leafimage2 = leafimage.copy()
                 while k == (ord('n')):
                     cv2.namedWindow('Drawingimage',cv2.WINDOW_NORMAL)
                     print("draw Outline in Figure by double clicking with left mouse button at points along the leafs edge ending \n with a double right mouse button click")
                     pdrun = PolygonDrawer('Drawingimage',img)
                     points = pdrun.run()
                     pointscollection.append(points)
                     k = cv2.waitKey(0)
                     cv2.destroyAllWindows()
                     if k == ord('q'):
                         pointscollection = []
                         leafimage = originalleafimage.copy()
                         cv2.destroyAllWindows()
                         break
                     elif k == ord('n'):
                         leafimage = leafimage2.copy()
                         img = cv2.imread(impath)
                         cv2.namedWindow('image',cv2.WINDOW_NORMAL)
                         contours,_ = cv2.findContours(np.flip(np.uint8(leafimage.T),axis=1),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
                         cv2.drawContours(img,contours,-1,(0,255,0),thickness=5)
                         pointscollection = []
                         continue
                     elif k == ord('y'):
                         k = ord('n')
                         continue
                 for points in pointscollection:
                     points = [np.flip(x) for x in np.array([points])[0]]
                     points = [[x[0], leafimage.shape[0]-x[1]] for x in points]
                     newnumber = find_smallest_missing_integer(np.unique(leafimage))
                     leafimage = cv2.fillPoly(leafimage.copy(), np.array([points]), newnumber)
                     leafarea = np.sum(leafimage==newnumber)
                 
                    # Calculate location of plug and necrosis for new leaf
                     plugprobabilityimage = h5py.File(h5folder +'Pixelprobabilities_plug/'+filename+'.JPG_Probabilities.h5')
                     plugprobabilityimage = np.squeeze(plugprobabilityimage['exported_data'])[:,:,1]
                     necrosisprobabilityimage = h5py.File(h5folder +'Pixelprobabilities_necrosis/'+filename+'.JPG_Probabilities.h5')
                     necrosisprobabilityimage = np.squeeze(necrosisprobabilityimage['exported_data'])[:,:,0]
                     plugimagenew,plugcentredictnew,plugleafdictnew = fff.plugfinder(leafimage,plugprobabilityimage,leafindexes=[newnumber])
                     plugimage[plugimagenew>0] = plugimagenew[plugimagenew>0]
                     plugcentredict.update(plugcentredictnew)
                     necrosisimagenew, necrosisradiusdict, necrosisareadict = fff.necrosisfinder(leafimage, necrosisprobabilityimage,plugcentredict,leafindexes=[newnumber])
                     necrosisimage[necrosisimagenew>0] = necrosisimagenew[necrosisimagenew>0]
                     
                     # Add new entry to summaryoutput
                     newentry= pd.DataFrame({"leafnumber": newnumber,'petriradius':summaryoutputold.petriradius[0],'petriarea':summaryoutputold.petriarea[0],'petricenter':summaryoutputold.petricenter[0],\
                    'leafarea':leafarea, 'plugcentre':str(plugcentredict[newnumber]), 'necrosisradius': necrosisradiusdict[newnumber], 'necrosisarea': necrosisareadict[newnumber]},index=[0])
                     summaryoutput = summaryoutput.append(newentry, ignore_index=True)
                     
             
                 # Save everything
             np.save(leaffilename,np.uint8(leafimage))        
             np.save(necrosisfilename,np.uint8(necrosisimage))    
             np.save(plugfilename,np.uint8(plugimage))    
             summaryoutput.to_csv(summaryoutputfilename,index=False)
        outpath = folder[:-1] + '_outlined_images\\' + filename + '.JPG'
        drawOutlines(impath, outpath, leafimage, necrosisimage, petriimage, plugimage)
        im = drawOutlines_display(impath, leafimage, necrosisimage, petriimage, plugimage)
        cv2.namedWindow('image',cv2.WINDOW_NORMAL)
        cv2.imshow('image',im)
        value = input('Do you want to fix more on the same image? (y/n)')
        if value == 'y':
            continue
        else:
            break

# ## Select whether you want to fix any other images that you select manually              
processmultiple = 1

while processmultiple == 1:
    print('Select the file you want to fix')
    filein = askopenfilename()
    filename = os.path.basename(filein).split('.')[0]
    folder = os.path.dirname(filein)
    folder = folder.rstrip('_outlined_images') + os.path.sep
    filein = folder + filename + '.JPG'
    addorremoveobject(filein,filename,folder)
    value = input("Do you want to fix any other image? (y/n)")
    if value == 'y':
        continue
    else:
        break

#         im = drawOutlines_display(filein,leafimage,necrosisimage,petriimage,plugimage)
#         cv2.namedWindow('image',cv2.WINDOW_NORMAL)
#         cv2.imshow('image',im)
#         value=input('Do you want to add or remove an object?(a(dd)/r(emove)/s(kip))')
#         k = cv2.waitKey(0)
#         cv2.destroyAllWindows()
#         if auto == 0:
#             value2=input('Which object do you want to modify(p(lug)/d(ish)/l(eaf)/n(ecrosis))?')
#         else:
#             value2 = auto
#         l=0
#         while l==0:
#             l=1
#             if value2 == 'p':
#                 objectcsv = plugcsv
#                 objectimage = plugimage
#                 objectname = 'plug'
#             elif value2 == 'd':
#                 objectcsv = petricsv
#                 objectimage = petriimage
#                 objectname = 'petri'                
#             elif value2 == 'l':
#                 objectcsv = leafcsv
#                 objectimage = leafimage
#                 objectname = 'leaf' 
#             elif value2 == 'n':
#                 objectcsv = necrosiscsv
#                 objectimage = necrosisimage
#                 objectname = 'necrosis' 
#             else:
#                 l = 0
#                 value2=input('Choose again, you can only choose between p/d/l/n')
#         print ('You chose to modify object '+ objectname)   
        
#         if value == 'r':
#             if len(objectcsv)==0:
#                 print('there is nothing to remove')
#                 break
#             for number in np.unique(objectimage[objectimage!=0]):
#                 i = objectcsv.loc[objectcsv['labelimage_oid']==number].index[0]
#                 if np.isnan(objectcsv['Center of the object_1'][i]):
#                     centre = [np.where(necrosisimage==1)[0][0],np.where(necrosisimage==1)[1][0]]
#                 else:
#                     centre = (int(im.shape[1]-objectcsv['Center of the object_1'][i]),int(objectcsv['Center of the object_0'][i]))
#                 im = cv2.putText(im,str(number),centre,cv2.FONT_HERSHEY_PLAIN,10,(255,255,255),20)
#             cv2.namedWindow('image',cv2.WINDOW_NORMAL)
#             cv2.imshow('image',im)
#             k = cv2.waitKey(1000)
#             cv2.destroyAllWindows()
#             plt.imshow(im[:,:,::-1])
#             plt.show()
#             removeindices=input('Which numbers do you want to remove?')
#             removeindices = eval(removeindices)
#             if not isinstance(removeindices,int):
#                 removeindices= [int(x) for x in list(removeindices)]
#             else:
#                 removeindices=[removeindices]
#             for i in removeindices:
#                 objectimage[objectimage==i]=0
#             objectcsv = objectcsv.drop(objectcsv.loc[objectcsv['labelimage_oid'].isin(removeindices)].index)
#             im = drawOutlines_display(filein,leafimage,necrosisimage,petriimage,plugimage)
#             cv2.namedWindow('image',cv2.WINDOW_NORMAL)
#             cv2.imshow('image',im)
#             k = cv2.waitKey(3000)
#             cv2.destroyAllWindows()
#             plt.imshow(im[:,:,::-1])
#             plt.show()
#             check=input('Are you sure you wanted these objects removed?')
#             if check == 'y':
#                 outpath = folder[:-4] + '_outlined_images\\' + filename + '.JPG'
#                 objectcsvfilename = folder +'Objects_'+objectname+'/'+filename+'.JPG_table.csv'
#                 objectimagefilename = folder +'Objects_'+objectname+'/'+filename+'.JPG_Object Identities.npy'
#                 drawOutlines(filein,outpath,leafimage,necrosisimage,petriimage,plugimage)
#                 objectimage=np.save(objectimagefilename,objectimage)
#                 objectcsv.to_csv(objectcsvfilename,index=False)
#             processimage=input('Do you want to continue modifying this image (y/n)?')
        
#         elif value == 'a':
#             drawing = 1
#             while drawing == 1:
#                 if 'labelimage_oid' in objectcsv.columns:
#                     object_id = next(i for i, e in enumerate(sorted(objectcsv['object_id']) + [ None ], 0) if i != e)                              
#                 else:
#                     object_id = 0
#                 newnumber = int(np.max(objectimage)+1)
                
                
#                 if value2 == 'n':
#                     print('draw start and end of necrosis')
#                     pdrun = PolygonDrawer('Drawingimage',im,1)
#                     points = pdrun.run()
#                     points = [np.flip(x) for x in np.array([points])[0]]
#                     points = [np.array([x[0], 3024-x[1]]) for x in points]
#                     leafnumber = np.max([leafimage[points[0][1],points[0][0]], leafimage[points[1][1],points[1][0]]])
#                     previousnecrosis = np.unique(necrosisimage[leafimage == leafnumber])
#                     newnecrosis = necrosisfinder(leafimage,0,leafnumber,0,points)
#                     globals()[objectname+'image'] = objectimage.copy()
#                     globals()[objectname+'image'][newnecrosis==1]=newnumber
#                     globals()[objectname+'image'][leafimage==0]=0
                
#                 elif value2 == 'p':
#                     print('Double click on the plug and press y to save')
#                     print('Press n if you made a mistake and want to start over')
#                     print('If you are finished with an image press q')
                    
#                     impath = str(folder)[:-4]+'_outlined_images\\'+filename+'.JPG'
#                     pluglist = draw_plug(impath)
                    
#                     for i in pluglist:
#                         coordinates = i
#                         coordinates[0] = np.size(plugimage,axis=0)-coordinates[0]
#                         leaf = leafimage[coordinates[0],coordinates[1]]
#                         plugs = np.unique(plugimage[leafimage==leaf])
#                         oldplugs = plugs[plugs>0]
#                         globals()[objectname+'image'][skimage.draw.disk((coordinates[0],coordinates[1]), radius=radius)]=newnumber
#                 else:
#                     print('draw Outline in Figure')
#                     pdrun = PolygonDrawer('Drawingimage',im)
#                     points = pdrun.run()
#                     points = [np.flip(x) for x in np.array([points])[0]]
#                     points = [np.array([x[0], 3024-x[1]]) for x in points]
#                     globals()[objectname+'image'] = cv2.fillPoly(objectimage.copy(), np.array([points]), newnumber)
                
#                 if value2 == 'n':
#                     for i in previousnecrosis:
#                         globals()[objectname+'image'][globals()[objectname+'image']==i]=0
#                 if value2 == 'p':
#                     for i in oldplugs:
#                         globals()[objectname+'image'][globals()[objectname+'image']==i]=0

                            
#                 im = drawOutlines_display(filein,leafimage,necrosisimage,petriimage,plugimage)
#                 cv2.namedWindow('Converted_image',cv2.WINDOW_NORMAL)
#                 cv2.imshow('Converted_image',im)
#                 k = cv2.waitKey(4000)
#                 cv2.destroyAllWindows()
#                 plt.imshow(im[:,:,::-1])
#                 plt.show()
#                 check = input('are you happy with the selected region(y/n)?')
#                 cv2.destroyAllWindows()
#                 if check == 'y':
#                     objectimage = globals()[objectname+'image'].copy()
#                     if value2 == 'n':
#                         for i in previousnecrosis:
#                             if 'labelimage_oid' in objectcsv.columns:
#                                  objectcsv.drop(objectcsv[objectcsv.labelimage_oid ==i].index)
#                     if value2 == 'p':
#                         for i in oldplugs:
#                             if 'labelimage_oid' in objectcsv.columns:
#                                 objectcsv.drop(objectcsv[objectcsv.labelimage_oid ==i].index)
                                
#                     outpath = folder[:-4] + '_outlined_images\\' + filename + '.JPG'
#                     drawOutlines(filein,outpath,leafimage,necrosisimage,petriimage,plugimage)
#                     findcentre = objectimage.copy()
#                     findcentre[findcentre!=newnumber]=0
#                     cnts = cv2.findContours(np.uint8(findcentre), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
#                     cnts = imutils.grab_contours(cnts)
#                     for i in cnts:
#                         M = cv2.moments(i)
#                         cX = int(M["m10"] / M["m00"])
#                         cY = int(M["m01"] / M["m00"])
#                         c, R = cv2.minEnclosingCircle(i)
#                     objectcsvfilename = folder +'Objects_'+objectname+'/'+filename+'.JPG_table.csv'
#                     objectimagefilename = folder +'Objects_'+objectname+'/'+filename+'.JPG_Object Identities.npy'
#                     np.save(objectimagefilename,objectimage)
#                     newentry = {'Predicted Class': 'Manual','object_id':object_id,'labelimage_oid': newnumber,'Size in pixels': np.sum(objectimage==newnumber), 'Object Area': np.sum(objectimage==newnumber), 'Center of the object_0': cX, 'Center of the object_1': cY,'Radii of the object_1': R, 'Radii of the object_0': R}
#                     objectcsv = objectcsv.append(newentry,ignore_index=True)
#                     objectcsv.dropna(how='all',inplace=True)
#                     objectcsv.to_csv(objectcsvfilename,index=False)
                    
#                 elif check == 'n':
#                     globals()[objectname+'image'] = objectimage
#                     im = drawOutlines_display(filein,leafimage,necrosisimage,petriimage,plugimage)
#                 check = input('Do you want to select another region(y/n)?')
#                 if check == 'n':
#                     drawing = 0
#                     processimage = 'n'
                
#         else:
#             processimage=input('Do you want to continue modifying this image (y/n)?')
#     return

# # Folder to correct (folder should contain subfolders for each day of the experiment) 
# root = Tk()
# root.attributes('-topmost',True)
# root.update()
# root.withdraw()
# multifolder = filedialog.askdirectory('What folder is ')
# multifolderpath = Path(multifolder)




#         break 
        
            # ## Select whether you want to automatically detect images with incorrect number of leaves and fix
            # value = input("Do you want to fix leaves? (y/n)")
            # if value != 'y' and value != 'n': 
            #     print(f'You entered {value}, which is not y or n')
            # elif value == 'y':
            #     print(f'You entered {value}')
            #     leaves = 1                    
            # elif value == 'n':
            #     print(f'You entered {value}')
            #     leaves = 0 
                
            # if leaves == 1: 
            #     for folderpath in multifolderpath.glob("*_h5"):
            #         for file in folderpath.glob("*Summaryoutput.csv"):
            #             folder = str(folderpath)+'\\'
            #             datafile = pd.read_csv(file)
            #             filename = os.path.basename(datafile['filename'][0])
            #             numberofleaves = len(datafile)
            #             filein = str(folderpath)[:-3]+'\\'+filename+'.JPG'
            #             if numberofleaves not in amountofleaves:
            #                 addorremoveobject(filein,filename,folder,'l')                 
                                
            # # Automatically detects whether plugs are missing and 
            # #Draws plug around centre that you click on in the image with radius size specified (Default 40)
            # plug = 0
            # value = input("Do you want to fix plugs? (y/n)")
              
            # if value != 'y' and value != 'n':
            #     print(f'You entered {value}, which is not y/n')
            # elif value == 'n':
            #     print(f'You entered {value}')
            #     plug = 0
            # elif value == 'y':
            #     print(f'You entered {value}')
            #     plug = 1

            # if plug == 1:
            #     print('Double click on the plug and press y to save')
            #     print('Press n if you made a mistake and want to start over')
            #     print('If you are finished with an image press q')
            #     for folderpath in multifolderpath.glob("*_h5"):
            #             for filein in folderpath.glob("*Summaryoutput.csv"):
            #                 datafile = pd.read_csv(filein)
            #                 filename = os.path.basename(datafile['filename'][0])
            #                 numberofleaves = len(datafile)
            #                 plugcsvfilename = str(folderpath) +'/Objects_plug/'+filename+'.JPG_table.csv'
            #                 plugcsv = pd.read_csv(plugcsvfilename)
            #                 if 'object_id' in plugcsv.keys():
            #                     numberofplugs=np.sum(plugcsv['object_id'].notna())
            #                     if numberofplugs<numberofleaves and numberofplugs>0:
            #                         impath = str(folderpath)[:-3]+'_outlined_images\\'+filename+'.JPG'
            #                         pluglist = draw_plug(impath)
            #                         plugimagefilename = str(folderpath) +'/Objects_plug/'+filename+'.JPG_Object Identities.npy'
            #                         plugimage = np.squeeze(np.load(plugimagefilename))
            #                         leafimagefilename = str(folderpath) +'/Objects_leaf/'+filename+'.JPG_Object Identities.npy'
            #                         leafimage = np.squeeze(np.load(leafimagefilename))
                                    
            #                         for i in pluglist:
            #                             coordinates = i
            #                             coordinates[0] = np.size(plugimage,axis=0)-coordinates[0]
            #                             plugleafdict[i]=leafimage[coordinates[0],coordinates[1]]                            
            #                             #i[1] = np.size(plugimage,axis=1)-i[1]
            #                             newplugid = max(plugcsv['object_id'])+1
            #                             plugcsv = plugcsv.append(pd.Series(),ignore_index=True)
            #                             plugcsv['object_id'].iloc[max(plugcsv.index)]=newplugid
            #                             plugcsv['labelimage_oid'] = newplugid+1
            #                             plugcsv['Predicted Class'].iloc[max(plugcsv.index)]='Manual'
            #                             plugcsv['Center of the object_0'].iloc[max(plugcsv.index)]=coordinates[1]
            #                             plugcsv['Center of the object_1'].iloc[max(plugcsv.index)]=coordinates[0]
            #                             plugimage[skimage.draw.disk((coordinates[0],coordinates[1]), radius=radius)]=newplugid+1 
            #                             plugcsv['Size in pixels'].iloc[max(plugcsv.index)]=np.sum(plugimage==newplugid+1)
            #                         plugcsv.to_csv(plugcsvfilename,index=False)
            # #                        plugimage = np.save(plugimagefilename,plugimage)
            #     print('Plugs fixed')