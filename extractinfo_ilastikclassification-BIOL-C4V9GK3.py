"""
Created on Wed Mar 23 13:40:24 2022

@author: vinkjo
This script is run after the ilastikbatch script in order to extract all the relevant info related to the classification 
such as the size of the petri dish, the size of leaf and the location and size of the plug and necrosis area
"""


import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math 
from skimage.morphology import skeletonize
import subprocess
import os
from pathlib import Path
import collections
import h5py


## Filters
filterplugabsence = 2                   # Filters out all plugs in a dataset if x are missing (Default=2)
filterminimumnecrosis = 15000           # Minimum area that necrosis needs to be to be considered (Area minus plug) Default = 10000
filterminimumdistancenecrosis = -100    # Minimum distance of necrosis to plug (So filters out far away areas) Default = -100
maxleaves =5                            # Filters out lowest probability leaves if more leaves are present than expected
## Functions
# Finds the closest and furthest point based on a skeleton
def closestandfurthestpoints(skeleton,point):
    skeletoncoordinates = np.where(skeleton==True)
    skeletoncoordinates = np.asarray(skeletoncoordinates).T
    point = point[::-1]
    Distances = np.sum((skeletoncoordinates-point)**2,axis=1)
    closestpoint = np.argmin(Distances)
    closestpoint = skeletoncoordinates[closestpoint,:]
    closestdistance = math.sqrt(np.min(Distances))
    furthestpoint = np.argmax(Distances)
    furthestpoint = skeletoncoordinates[furthestpoint,:]
    furthestdistance = math.sqrt(np.max(Distances))
    
    return closestpoint,furthestpoint,closestdistance,furthestdistance

# Draws outlines of the plug,petridish,leaf and necrosis in the original image
def drawOutlines(impath,outpath,leafimage,necrosisimage,petriimage,plugimage):
    im = cv2.imread(impath)
    cv2.imshow('image',im)
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


multifolder = 'C:\\Users\\vinkjo\\OneDrive - Victoria University of Wellington - STAFF\\Desktop\\Machine learning Leaves\\Raw data\\3770'
multifolderpath = Path(multifolder)
for folderpath in multifolderpath.glob("*_h5"):
    folder = str(folderpath)+'\\'
    for filein in folderpath.glob("*.h5"):
        # Finding filename in folder
        filename = filein
        filename = os.path.basename(filein)
        filename = filename[:-7]
        # Loading all the ilastik generated datafiles:
        
        # Petri data
        petrifilename = folder +'Objects_petri/'+filename+'.JPG_table.csv'
        petricsv = pd.read_csv(petrifilename)
        if len(petricsv)>1:
            petricsv = petricsv.loc[petricsv['Size in pixels']==petricsv['Size in pixels'].max()]
            
        petriradius = (petricsv['Radii of the object_1'][0]+petricsv['Radii of the object_0'][0])/2
        petrisize =petricsv['Size in pixels'][0]
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
        if len(leafcsv)>maxleaves:
            sumprob = 10000000
            for i in leafcsv['labelimage_oid']:
                tempsumprob = np.sum(leafprobabilityimage[leafimage==i])
                if tempsumprob<sumprob:
                    sumprob=tempsumprob
                    removeindex = i
            leafcsv.drop(leafcsv.loc[leafcsv['labelimage_oid']==removeindex].index)
            leafimage[leafimage==removeindex]=0
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
        numberofleaves=leafimage.max()
    
    
    # Intiliaze     
        Summaryoutput = leafcsv.copy()
        Summaryoutput['filename']=folder+filename
        Summaryoutput['petriradius'] = petriradius
        Summaryoutput['petrisize'] = petrisize
        Summaryoutput['plug_centre'] = np.nan
        Summaryoutput['plug_id'] = np.nan
        Summaryoutput['necrosis_id'] = np.nan
        Summaryoutput['necrosis_area'] = np.nan
        Summaryoutput['necrosis_distance'] = np.nan
        Summaryoutput['necrosis_leaffraction'] = np.nan
        # if len(leafcsv) == len(plugcsv):
        #     matchingleaveplug = 1
        #     noplugs = 0
        # elif len(plugcsv) == 0:
        #     noplugs = 1
        #     matchingleaveplug = 0
        # else: 
        #     matchingleaveplug = 0
        #     noplugs = 0
    
    # Defines which number of plug/necrosis belongs to which leaf        
        necrosisleafdict = dict() 
        plugleafdict = dict()
        plugcentredict = dict()
        
        if 'labelimage_oid' in plugcsv:
            plugcsv['labelimage_oid'] = plugcsv['object_id']+1
            ## Filter plugs and assign to leaf
            for i in plugcsv.labelimage_oid:
                plugcentre = plugcsv[['Center of the object_0','Center of the object_1']].loc[plugcsv['labelimage_oid']==i]
                plugcentre = [plugcentre.iloc[0][0],plugcentre.iloc[0][1]]
                plugcentredict[i] = plugcentre
                if np.sum(plugimage==i)==0:
                    plugcsv = plugcsv[plugcsv.labelimage_oid != i]
                    continue
                leafid = np.unique(leafimage[plugimage==i])[0]
                if leafid in plugleafdict.keys():
                    competingplug = plugleafdict[leafid]
                    # If multiple potential plugs in one leaf, the one with highest probability is picked
                    competingprob = np.sum(plugprobabilityimage[plugimage==competingplug])
                    prob = np.sum(plugprobabilityimage[plugimage==i])
                    
                    if prob>competingprob:
                        plugleafdict[np.unique(leafimage[plugimage==i])[0]] = i
                        print("In "+filename+" , plug "+str(competingplug)+" was removed")
                        plugcsv = plugcsv[plugcsv.labelimage_oid != competingplug]
                        plugimage[plugimage==competingplug]=0
                    else:
                        print("In "+filename+", plug "+str(i)+" was removed")
                        plugcsv = plugcsv[plugcsv.labelimage_oid != i]
                        plugimage[plugimage==i]=0
                else:    
                    plugleafdict[np.unique(leafimage[plugimage==i])[0]] = i
            
            Summaryoutput['plug_id']=Summaryoutput['labelimage_oid'].map(plugleafdict)
            Summaryoutput['plug_centre']=Summaryoutput['plug_id'].map(plugcentredict)           
            
            if 'labelimage_oid' in necrosiscsv:
                for i in necrosiscsv.labelimage_oid:
                    if np.sum(necrosisimage==i)==0:
                        necrosiscsv = necrosiscsv[necrosiscsv.labelimage_oid != i]
                        continue
                    leafid = np.unique(leafimage[necrosisimage==i])[0]
                    if not leafid in plugleafdict:
                        print("In "+filename+", necrosis "+str(i)+" was removed, no plug in leaf")
                        necrosisimage[necrosisimage==i]=0
                        continue
                    plugid = plugleafdict[leafid]
                    plugcentre = plugcentredict[plugid]
                    necrosismask = np.isin(necrosisimage,i)
                    contours, hierarchy = cv2.findContours(np.uint8(necrosismask), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
                    closestDist = cv2.pointPolygonTest(contours[0], plugcentre, True)
                    necrosismaskminplug = necrosismask
                    necrosismaskminplug[plugimage==plugid]=False
                    if closestDist<filterminimumdistancenecrosis:
                        print("In "+filename+", necrosis "+str(i)+" was removed, too far from plug")
                        necrosisimage[necrosisimage==i]=0
                    elif np.sum(necrosismaskminplug)<filterminimumnecrosis:
                        print("In "+filename+", necrosis "+str(i)+" was removed, too small area")
                        necrosisimage[necrosisimage==i]=0
                    else: 
                        closest,furthest,closestdist,furthestdist = closestandfurthestpoints(necrosismask,plugcentre)
                        necrosisleafdict[leafid]=i
                        necrosismask[plugimage==plugid]=True
                        Summaryoutput['necrosis_area'].loc[Summaryoutput['labelimage_oid']==leafid]=np.sum(necrosismask)
                        Summaryoutput['necrosis_distance'].loc[Summaryoutput['labelimage_oid']==leafid]=furthestdist
                        print("In "+filename+", necrosis "+str(i)+" was at distance " +str(furthestdist))
            Summaryoutput['necrosis_leaffraction']=Summaryoutput['necrosis_area']/Summaryoutput['Object Area']
            Summaryoutput['necrosis_id']=Summaryoutput['labelimage_oid'].map(necrosisleafdict)
        else:
            print(filename+' has no plugs')
            necrosisimage[necrosisimage>0] = 0
            plugimage[plugimage>0] = 0

        impath = folder[:-4] + '\\' + filename + '.JPG'
        outpath = folder[:-4] + '_outlined_images\\' + filename + '.JPG'
        if not os.path.isdir(folder[:-4] + '_outlined_images\\'):
            os.mkdir(folder[:-4] + '_outlined_images\\')
        drawOutlines(impath,outpath,leafimage,necrosisimage,petriimage,plugimage)
        outputfilename = os.path.join(folder,filename+'_Summaryoutput.csv')
        Summaryoutput.to_csv(outputfilename,index=False)
        plugcsv.to_csv(plugcsvfilename,index=False)
        leafcsv.to_csv(leafcsvfilename,index=False)
        petricsv.to_csv(petrifilename,index=False)
        necrosiscsv.to_csv(necrosiscsvfilename,index=False)
        petriimage=np.save(petriimagefilename,petriimage)
        plugimage=np.save(plugimagefilename,plugimage)
        necrosisimage=np.save(necrosisimagefilename,necrosisimage)
        leafimage=np.save(leafimagefilename,leafimage)
    
 #     if matchingleaveplug == 1:
    #         for i in range(len(plugcsv)):
    #             mask = np.isin(leafimage,i+1)
    #             mask = np.uint8(mask)
    #             mask = np.squeeze(mask)
    
    #             # contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    
    #             # for j in range(numberofleaves):
    #             #     plugcentre = plugcsv[['Center of the object_0','Center of the object_1']].iloc[j]
    #             #     plugcentre = [plugcentre[0],plugcentre[1]]
    #             #     closestDist = cv2.pointPolygonTest(contours[0], plugcentre, True)
    #             #     if closestDist>0:
    #             #         plugleafdict[i+1] = j
    #             #         break
    #             # furthestcontour = 0;
    #             # for i in range(len(contours[0])):
    #             #     dist = math.hypot(contours[0][i][0][0]-plugcentre.iloc[0][0], contours[0][i][0][1]-plugcentre.iloc[0][1])
    #             #     if dist > furthestcontour:
    #             #         furthestcontour = dist
    
    
    
    
    
    # for i in leafcsv.object_id:
    #     mask = np.isin(leafimage,i)
    #     mask = np.squeeze(mask)
    #     plugmask = np.isin(plugimage,plugleafdict[i])
    #     plugmask = np.squeeze(plugmask)        
    #     skeleton = skeletonize(mask)
    #     plugcentre = plugcsv[['Center of the object_0','Center of the object_1']].iloc[i-1]
    #     plugcentre = [plugcentre[1],plugcentre[0]]
    #     leafcsv['plugcentre'].iloc[i-1] = plugcentre
    #     closest,furthest,a,b = closestandfurthest(skeleton,plugcentre)
    #     imageshape = np.shape(mask)
    #     A1,B1 =np.polyfit([plugcentre[1],closest[1]],[plugcentre[0],closest[0]],1)
    #     x = np.linspace(0,imageshape[1])
    #     fig,ax = plt.subplots()
    #     ax.plot(x,A1*x+B1)
#     ax.set_xlim((0,4000))
#     ax.set_ylim((0,3000))
#     plt.imshow(mask, cmap='gray')
#     plt.show()
#     plt.figure()
#     plt.imshow(mask, cmap='gray')   
#     plt.imshow(skeleton, cmap='jet', alpha=0.8)
#     plt.imshow(plugmask,cmap='Reds_r',alpha = 0.2)
#     if i in necrosisleafdict:
#         necrosismask = np.isin(necrosisimage,necrosisleafdict[i])
#         necrosismask = np.squeeze(necrosismask)   
#         plt.imshow(necrosismask, cmap='Reds_r',alpha = 0.2)             
#         closest,furthest,a,b = closestandfurthest(necrosismask,plugcentre)
#         print(b)


# for i in range(len(Summaryoutput)):
#     if i+1 in plugleafdict:
#         Summaryoutput['plugcentre'].iloc[i] = plugcsv[['Center of the object_0','Center of the object_1']].iloc[plugleafdict[i+1]-1]
#         Summaryoutput['plugid'].iloc[i] = plugleafdict[i+1]
#     if i+1 in necrosisleafdict:
#         Summaryoutput['necrosisarea'].iloc[i] = necrosisleafdict[i+1]
#         Summaryoutput['necrosisdistance'].iloc[i] = necrosisleafdict[i+1]


# testimage = np.load('C:/Users/vinkjo/Downloads/OneDrive_2022-03-17/230222 1 D0_h5/Objects_Petri/IMG_9796.JPG_Object Identities.npy')
# mask = np.isin(leafimage,1)
# mask = np.uint8(mask)
# mask = np.squeeze(mask)
# contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
# redPoint = [40,500]
# closestDist = cv2.pointPolygonTest(contours[0], redPoint, True)