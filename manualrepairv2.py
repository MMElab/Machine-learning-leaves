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
import skimage.draw
import matplotlib.pyplot as plt
import h5py
from tkinter import *
from tkinter.filedialog import askopenfilename
import sys
import imutils
sys.path.append('C:\\Users\\vinkjo\\OneDrive - Victoria University of Wellington - STAFF\\Desktop\\Machine learning Leaves')
from Polygon import PolygonDrawer
# Folder to correct (folder should contain subfolders for each day of the experiment) 
multifolder = 'C:\\Users\\vinkjo\\OneDrive - Victoria University of Wellington - STAFF\\Desktop\\Machine learning Leaves\\Raw data\\3770'
multifolderpath = Path(multifolder)

# Number of leaves possible in image
amountofleaves = [3,5]

# Functions
# Draws plug around centre that you click on in the image with radius size specified (Default 40)
radius = 40
def draw_circle(event,x,y,flags,param):
    global mouseX,mouseY
    if event == cv2.EVENT_LBUTTONDBLCLK:
        cv2.circle(img,(x,y),40,(255,0,0),5)
        mouseX,mouseY = x,y
    return

def draw_plug(impath): 
    global img
    img = cv2.imread(impath)
    cv2.namedWindow('image',cv2.WINDOW_NORMAL) 
    cv2.setMouseCallback('image',draw_circle)
    pluglist = []
    while(1):
        cv2.imshow('image',img)
        k = cv2.waitKey(20) & 0xFF
        if k == ord('q'):
            cv2.destroyAllWindows()
            break
        elif k == ord('n'):
            cv2.destroyAllWindows()
            pluglist = []
            img = cv2.imread(impath)
            cv2.namedWindow('image',cv2.WINDOW_NORMAL)
            cv2.setMouseCallback('image',draw_circle)
        elif k == ord('y'):
            pluglist.append([mouseX,mouseY])
    return pluglist

def drawOutlines(impath,outpath,leafimage,necrosisimage,petriimage,plugimage):
    im = cv2.imread(impath)
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
    im = cv2.imread(impath)
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

def addorremoveobject(filein,filename,folder,auto):
    processimage = 'y'
    global leafimage,petriimage,necrosisimage,plugimage
    while processimage =='y':
        # Petri data
        petrifilename = folder +'Objects_petri/'+filename+'.JPG_table.csv'
        petricsv = pd.read_csv(petrifilename)
        petriimagefilename = folder +'Objects_petri/'+filename+'.JPG_Object Identities.npy'
        petriimage = np.squeeze(np.load(petriimagefilename))
        petriimage[petriimage!=int(petricsv['labelimage_oid'])]=0
        
        # Leaf data
        leafcsvfilename = folder +'Objects_leaf/'+filename+'.JPG_table.csv'
        leafcsv = pd.read_csv(leafcsvfilename)
        leafimagefilename = folder +'Objects_leaf/'+filename+'.JPG_Object Identities.npy'
        leafimage = np.squeeze(np.load(leafimagefilename))
        leafprobabilityimage = h5py.File(folder +'Pixelprobabilities_leaf/'+filename+'.JPG_Probabilities.h5')
        leafprobabilityimage = np.squeeze(leafprobabilityimage['exported_data'])[:,:,0]

        # Plug data
        plugcsvfilename = folder +'Objects_plug/'+filename+'.JPG_table.csv'
        plugcsv = pd.read_csv(plugcsvfilename)
        plugimagefilename = folder +'Objects_plug/'+filename+'.JPG_Object Identities.npy'
        plugimage = np.squeeze(np.load(plugimagefilename))
        plugimage[leafimage==0]=0
        plugprobabilityimage = h5py.File(folder +'Pixelprobabilities_plug/'+filename+'.JPG_Probabilities.h5')
        plugprobabilityimage = np.squeeze(plugprobabilityimage['exported_data'])[:,:,1]
        
        #Necrosis data
        necrosiscsvfilename = folder +'Objects_necrosis/'+filename+'.JPG_table.csv'
        necrosiscsv = pd.read_csv(necrosiscsvfilename)
        necrosisimagefilename = folder +'Objects_necrosis/'+filename+'.JPG_Object Identities.npy'
        necrosisimage = np.squeeze(np.load(necrosisimagefilename))
        necrosisimage[leafimage==0]=0
        
        im = drawOutlines_display(filein,leafimage,necrosisimage,petriimage,plugimage)
        cv2.namedWindow('image',cv2.WINDOW_NORMAL)
        cv2.imshow('image',im)
        k = cv2.waitKey(20)
        value=input('Do you want to add or remove an object?(a(dd)/r(emove)/s(kip))')
        cv2.destroyAllWindows()
        if auto == 0:
            value2=input('Which object do you want to modify(p(lug)/d(ish)/l(eaf)/n(ecrosis))?')
        else:
            value2 = auto
        l=0
        while l==0:
            l=1
            if value2 == 'p':
                objectcsv = plugcsv
                objectimage = plugimage
                objectname = 'plug'
            elif value2 == 'd':
                objectcsv = petricsv
                objectimage = petriimage
                objectname = 'petri'                
            elif value2 == 'l':
                objectcsv = leafcsv
                objectimage = leafimage
                objectname = 'leaf' 
            elif value2 == 'n':
                objectcsv = necrosiscsv
                objectimage = necrosisimage
                objectname = 'necrosis' 
            else:
                l = 0
                value2=input('Choose again, you can only choose between p/d/l/n')
        print ('You chose to modify object '+ objectname)   
        
        if value == 'r':
            if len(objectcsv)==0:
                print('there is nothing to remove')
                break
            
            for i in objectcsv.index:
                number = objectcsv['labelimage_oid'][i]
                centre = (int(im.shape[1]-objectcsv['Center of the object_1'][i]),int(objectcsv['Center of the object_0'][i]))
                im = cv2.putText(im,str(number),centre,cv2.FONT_HERSHEY_PLAIN,10,(255,255,255),20)
            cv2.namedWindow('image',cv2.WINDOW_NORMAL)
            cv2.imshow('image',im)
            k = cv2.waitKey(1000)
            cv2.destroyAllWindows()
            plt.imshow(im)
            plt.show()
            removeindices=input('Which numbers do you want to remove?')
            removeindices = eval(removeindices)
            if not isinstance(removeindices,int):
                removeindices= [int(x) for x in list(removeindices)]
            else:
                removeindices=[removeindices]
            for i in removeindices:
                objectimage[objectimage==i]=0
            objectcsv = objectcsv.drop(objectcsv.loc[objectcsv['labelimage_oid'].isin(removeindices)].index)
            im = drawOutlines_display(filein,leafimage,necrosisimage,petriimage,plugimage)
            cv2.namedWindow('image',cv2.WINDOW_NORMAL)
            cv2.imshow('image',im)
            k = cv2.waitKey(3000)
            cv2.destroyAllWindows()
            plt.imshow(im)
            plt.show()
            check=input('Are you sure you wanted these objects removed?')
            if check == 'y':
                outpath = folder[:-4] + '_outlined_images\\' + filename + '.JPG'
                objectcsvfilename = folder +'Objects_'+objectname+'/'+filename+'.JPG_table.csv'
                objectimagefilename = folder +'Objects_'+objectname+'/'+filename+'.JPG_Object Identities.npy'
                drawOutlines(filein,outpath,leafimage,necrosisimage,petriimage,plugimage)
                objectimage=np.save(objectimagefilename,objectimage)
                objectcsv.to_csv(objectcsvfilename,index=False)
            processimage=input('Do you want to continue modifying this image (y/n)?')
        
        elif value == 'a':
            drawing = 1
            while drawing == 1:
                print('draw Outline in Figure')
                pdrun = PolygonDrawer('Drawingimage',im)
                points = pdrun.run()
                points = [np.flip(x) for x in np.array([points])[0]]
                points = [np.array([x[0], 3024-x[1]]) for x in points]
                
                if 'labelimage_oid' in objectcsv.columns:
                    newnumber = next(i for i, e in enumerate(sorted(objectcsv['labelimage_oid']) + [ None ], 1) if i != e) 
                    object_id = next(i for i, e in enumerate(sorted(objectcsv['object_id']) + [ None ], 0) if i != e)                              
                else:
                    newnumber = 1
                    object_id = 0
                
                globals()[objectname+'image'] = cv2.fillPoly(objectimage.copy(), np.array([points]), newnumber)
                
                if value2 == 'n' or value2 == 'p':
                    globals()[objectname+'image'][leafimage==0]=0
                    
                im = drawOutlines_display(filein,leafimage,necrosisimage,petriimage,plugimage)
                cv2.namedWindow('Converted_image',cv2.WINDOW_NORMAL)
                cv2.imshow('Converted_image',im)
                k = cv2.waitKey(3000)
                cv2.destroyAllWindows()
                plt.imshow(im)
                plt.show()
                check = input('are you happy with the selected region(y/n)?')
                cv2.destroyAllWindows()
                if check == 'y':
                    objectimage = globals()[objectname+'image'].copy()
                    outpath = folder[:-4] + '_outlined_images\\' + filename + '.JPG'
                    drawOutlines(filein,outpath,leafimage,necrosisimage,petriimage,plugimage)
                    findcentre = objectimage.copy()
                    findcentre[findcentre!=newnumber]=0
                    cnts = cv2.findContours(findcentre, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
                    cnts = imutils.grab_contours(cnts)
                    for i in cnts:
                        M = cv2.moments(i)
                        cX = int(M["m10"] / M["m00"])
                        cY = int(M["m01"] / M["m00"])
                    
                    objectcsvfilename = folder +'Objects_'+objectname+'/'+filename+'.JPG_table.csv'
                    objectimagefilename = folder +'Objects_'+objectname+'/'+filename+'.JPG_Object Identities.npy'
                    np.save(objectimagefilename,objectimage)
                    newentry = {'Predicted Class': 'Manual','object_id':object_id,'labelimage_oid': newnumber,'Size in pixels': np.sum(objectimage==newnumber), 'Object Area': np.sum(objectimage==newnumber), 'Center of the object_0': cX, 'Center of the object_1': cY}
                    objectcsv = objectcsv.append(newentry,ignore_index=True)
                    objectcsv.to_csv(objectcsvfilename,index=False)
                elif check == 'n':
                    globals()[objectname+'image'] = objectimage
                    im = drawOutlines_display(filein,leafimage,necrosisimage,petriimage,plugimage)
                check = input('Do you want to select another region(y/n)?')
                if check == 'n':
                    drawing = 0
                
        else:
            processimage=input('Do you want to continue modifying this image (y/n)?')
    return



## Select whether you want to automatically detect images with incorrect number of leaves and fix
value = input("Do you want to fix leaves? (y/n)")
if value != 'y' and value != 'n': 
    print(f'You entered {value}, which is not y or n')
elif value == 'y':
    print(f'You entered {value}')
    leaves = 1                    
elif value == 'n':
    print(f'You entered {value}')
    leaves = 0 
    
if leaves == 1: 
    for folderpath in multifolderpath.glob("*_h5"):
        for file in folderpath.glob("*Summaryoutput.csv"):
            folder = str(folderpath)+'\\'
            datafile = pd.read_csv(file)
            filename = os.path.basename(datafile['filename'][0])
            numberofleaves = len(datafile)
            filein = str(folderpath)[:-3]+'\\'+filename+'.JPG'
            if numberofleaves not in amountofleaves:
                addorremoveobject(filein,filename,folder,'l')
                   
                    
# Automatically detects whether plugs are missing and 
#Draws plug around centre that you click on in the image with radius size specified (Default 40)
plug = 0
value = input("Do you want to fix plugs? (y/n)")
  
if value != 'y' and value != 'n':
    print(f'You entered {value}, which is not y/n')
elif value == 'n':
    print(f'You entered {value}')
    plug = 0
elif value == 'y':
    print(f'You entered {value}')
    plug = 1
if plug == 1:
    print('Double click on the plug and press y to save')
    print('Press n if you made a mistake and want to start over')
    print('If you are finished with an image press q')
    for folderpath in multifolderpath.glob("*_h5"):
            for filein in folderpath.glob("*Summaryoutput.csv"):
                datafile = pd.read_csv(filein)
                filename = os.path.basename(datafile['filename'][0])
                numberofleaves = len(datafile)
                plugcsvfilename = str(folderpath) +'/Objects_plug/'+filename+'.JPG_table.csv'
                oldplugcsvfilename = str(folderpath) +'/Objects_plug_old/'+filename+'.JPG_table.csv'
                plugcsv = pd.read_csv(plugcsvfilename)
                if 'object_id' in plugcsv.keys():
                    numberofplugs=np.sum(plugcsv['object_id'].notna())
                    if numberofplugs<numberofleaves and numberofplugs>0:
                        impath = str(folderpath)[:-3]+'_outlined_images\\'+filename+'.JPG'
                        pluglist = draw_plug(impath)
                        plugimagefilename = str(folderpath) +'/Objects_plug/'+filename+'.JPG_Object Identities.npy'
                        plugimage = np.squeeze(np.load(plugimagefilename))
                        for i in pluglist:
                            coordinates = i
                            coordinates[0] = np.size(plugimage,axis=0)-coordinates[0]
                            #i[1] = np.size(plugimage,axis=1)-i[1]
                            newplugid = max(plugcsv['object_id'])+1
                            plugcsv = plugcsv.append(pd.Series(),ignore_index=True)
                            plugcsv['object_id'].iloc[max(plugcsv.index)]=newplugid
                            plugcsv['labelimage_oid'] = plugcsv['object_id']+1
                            plugcsv['Predicted Class'].iloc[max(plugcsv.index)]='Manual'
                            plugcsv['Center of the object_0'].iloc[max(plugcsv.index)]=coordinates[1]
                            plugcsv['Center of the object_1'].iloc[max(plugcsv.index)]=coordinates[0]
                            plugimage[skimage.draw.disk((coordinates[0],coordinates[1]), radius=radius)]=newplugid+1 
                            plugcsv['Size in pixels'].iloc[max(plugcsv.index)]=np.sum(plugimage==newplugid+1)
                        plugcsv.to_csv(plugcsvfilename,index=False)
                        plugimage = np.save(plugimagefilename,plugimage)
    print('Plugs fixed')

## Select whether you want to fix any other images that you select manually              
processmultiple = 1
while processmultiple == 1:
    value = input("Do you want to fix any other image? (y/n)")
    if value == 'y': 
        #print(f'You entered {value}, which is not y or n')
        print('Select the file you want to fix')
        root = Tk()
        root.withdraw()
        root.update()
        filein = askopenfilename()
        root.destroy()
        filename = os.path.basename(filein).split('.')[0]
        folder = os.path.dirname(filein)
        folder = folder.rstrip('_outlined_images')+'_h5/'
        auto=0
        addorremoveobject(filein,filename,folder,auto)
    else:
        break 
        
            