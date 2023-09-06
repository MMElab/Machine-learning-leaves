# -*- coding: utf-8 -*-
"""
Created on Fri Mar 18 14:20:03 2022

@author: vinkjo
"""

import numpy as np
import pandas as pd
import os
from pathlib import Path
import re
from scipy.optimize import linear_sum_assignment
from tkinter import filedialog
import cv2
import seaborn as sns
# Parameters
petriradius = 75 # Size of petridish in mm
minnumberofleaves = 5 # Minimum number of leaves. Files with less leaves will not be taken into account

# This script links leaves from different images together by calculating the Hu moments of images and of leaves 
# These Hu moments are shape dependent but independent of rotation, scaling etc.
# 

# Specify folder path (usual structure is multifolderpath/Experiment_DX_h5)
multifolder = filedialog.askdirectory()
multifolderpath = Path(multifolder)
# Load datafiles
originalfolderpath = list(multifolderpath.glob("*D0_h5"))[0]
combineddatafile = pd.DataFrame()

# Calculate properties of leaves in original (Day 0) folder
Sumtot_original = pd.DataFrame()
originalleaffolderpath = originalfolderpath / 'Objects_leaf'
for filein in originalfolderpath.glob("*Summaryoutput.csv"):
    datafile_original = pd.read_csv(filein)
    filename = os.path.basename(filein).rstrip('_Summaryoutput.csv')
    filenameleafimage = originalfolderpath + '/' + (filename + '_leaf_classification.npy')
    leafimage = np.load(filenameleafimage)
    Humomentsdict = dict()
    
    # Calculates Humoments for each leaf
    for leaf in datafile_original['leafnumber']:
        a = np.uint8(leafimage==leaf)   
        moments = cv2.moments(a)
        Humoments = cv2.HuMoments(moments)
        Humoments = Humoments[0:6]
        Humomentsdict[leaf]=Humoments
    
    # Calculates Humoments for entire image
    leafimage[leafimage>1]=1
    moments = cv2.moments(leafimage)
    Humoments = cv2.HuMoments(moments)
    Humoments = Humoments[0:6]
    numberofleaves = len(datafile_original)
    
    if numberofleaves<minnumberofleaves:
        continue
    normalizedscale = petriradius/datafile_original['petriradius'][0]
    datafile_original['normleafarea']=datafile_original['leafarea']*(normalizedscale**2)
    Sumtot_original = Sumtot_original.append([[filename,numberofleaves,normalizedscale,Humoments,Humomentsdict,dict(zip(datafile_original['leafnumber'],datafile_original['normleafarea'])),list(datafile_original['leafnumber'])]])
Sumtot_original.index = np.arange(0,len(Sumtot_original))

# Calculate properties of leaves in other (>Day 0) folders
for folderpath in multifolderpath.glob("*_h5"):
    if folderpath == originalfolderpath:
        continue
    else:
        Sumtot = pd.DataFrame()
        
        # Calculate Hu moments for each image
        for filein in folderpath.glob("*Summaryoutput.csv"):
            datafile = pd.read_csv(filein)
            filename = os.path.basename(filein).rstrip('_Summaryoutput.csv')
            filenameleafimage = folderpath + '/' + (filename + '_leaf_classification.npy')
            leafimage = np.load(filenameleafimage)
            Humomentsdict = dict()
            
            # Calculates Humoments for each leaf
            for leaf in datafile['leafnumber']:
                a = np.uint8(leafimage==leaf)
                moments = cv2.moments(a)
                Humoments = cv2.HuMoments(moments)
                Humoments = Humoments[0:6]
                Humomentsdict[leaf]=Humoments
            
            # Calculates Humoments for entire image
            leafimage[leafimage>1]=1
            moments = cv2.moments(leafimage)
            Humoments = cv2.HuMoments(moments)
            Humoments = Humoments[0:6]
            numberofleaves = len(datafile)
            if numberofleaves<minnumberofleaves:
                continue
            normalizedscale = petriradius/list(datafile['petriradius'])[0]
            datafile['normleavearea']=datafile['Size in pixels']*(normalizedscale**2)
            Sumtot = Sumtot.append([[filename,numberofleaves,normalizedscale,Humoments,Humomentsdict,dict(zip(datafile['labelimage_oid'],datafile['normleavearea'])),list(datafile['object_id'])]])
        Sumtot.index = np.arange(0,len(Sumtot))
        
        
        # Match images based on Hu moments
        bestpartnerdict = dict()
        numberofimages = len(Sumtot)
        # Calculate distance of Hu moments for imageset in DX to D0 (more distant > more dissimilar images)
        distanceimageHu=np.full((numberofimages,numberofimages),10000.5)
        for j,k in enumerate(Sumtot[3]):
            for l,m in enumerate(Sumtot_original[3]):
                ka=np.sign(k)*np.log(abs(k))
                ma=np.sign(m)*np.log(abs(m))
                distanceimageHu[j,l] = np.sum(abs(1/ka-1/ma))
                Sumsize = np.sum(list(Sumtot[5].loc[j].values()))
                Sumsize_original = np.sum(list(Sumtot_original[5].loc[l].values()))
                # Add an extra check whether the total leaf area is roughly equal between the D0 and DX image
                if abs(Sumsize-Sumsize_original)/Sumsize > 0.1:
                    distanceimageHu[j,l] = distanceimageHu[j,l]*100
        bestmatch1,bestmatch2 = linear_sum_assignment(distanceimageHu)
        # Best partner image in this frame
        bestpartnerdict = dict(zip(bestmatch1,bestmatch2))        
        
        # Calculate distance of Hu moments for each leaf within the best partner images
        bestpartnerdictleaves = dict()
        for i in Sumtot.index:
            originalHumomdict = Sumtot_original.iloc[bestpartnerdict[i]][4]
            Humomentsdict = Sumtot.iloc[i][4]
            sizedict = Sumtot.iloc[i][5]
            total = sum(sizedict.values())
            sizedict = {key: value / total for key, value in sizedict.items()}
            originalsizedict = Sumtot_original.iloc[bestpartnerdict[i]][5]
            total = sum(originalsizedict.values())
            originalsizedict = {key: value / total for key, value in originalsizedict.items()}
            
            distanceleafsizes=np.full((max(Humomentsdict.keys())+1,max(originalHumomdict.keys())+1),10000.5)
            for j in Humomentsdict:
                for l in originalHumomdict:
                    k = Humomentsdict[j]
                    m = originalHumomdict[l]
                    ka=np.sign(k)*np.log(abs(k))
                    ma=np.sign(m)*np.log(abs(m))
                    distanceleafsizes[j,l] = np.sum(abs(1/ka-1/ma))
                    # Add an extra check whether the relative leaf area is roughly similar between the two leaves
                    if abs(sizedict[j]-originalsizedict[l])>0.06:
                        distanceleafsizes[j,l]=100*distanceleafsizes[j,l]
                
            bestmatch1,bestmatch2 = linear_sum_assignment(distanceleafsizes)
            # Best partner leaf in this frame
            bestpartnerdictleaves[i] = dict(zip(bestmatch1,bestmatch2))
            
        if len(np.unique(bestpartnerdict.values())[0])==len(np.unique(list(bestpartnerdict.values()))):
            print('Linking has ended. Datasets have been linked to unique images.')
        else:
            print('Linking has ended. Some datasets are linked erronously. Check the problem')              
        
        ## Make combined datafile
        combineddatafileentry = pd.DataFrame()
        for filein in folderpath.glob("*Summaryoutput.csv"):
            datafile = pd.read_csv(filein)
            filename = os.path.basename(datafile['filename'][0])
            #location = np.where(Sumtot[0].str.contains(filename))[0][0]
            if np.sum(Sumtot[0]==filename)==0:
                print('Skipping '+filename)
                continue
            location = np.where(Sumtot[0]==filename)[0][0]
            datafile['filename_original']=Sumtot_original.iloc[bestpartnerdict[location]][0]
            day = int(re.search("D(.{1,2})_h5", str(filein))[0][1:-3])
            leafdict = bestpartnerdictleaves[location]
            if len(leafdict)>0:
                if len(datafile)==3 or len(datafile)==5:
                    datafile['labelimage_oid_original'] = datafile['labelimage_oid'].map(leafdict).astype(int)
                    combineddatafileentry['objectid'] = datafile['filename_original'].astype(str) +'_'+datafile['labelimage_oid_original'].astype(str)
                    combineddatafileentry['day']=day
                    combineddatafileentry[['Size in pixels','petriradius','necrosis_area','necrosis_distance','necrosis_leaffraction','plug_id','filename']]=datafile[['Size in pixels','petriradius','necrosis_area','necrosis_distance','necrosis_leaffraction','plug_id','filename']]
                    combineddatafile=combineddatafile.append(combineddatafileentry)
                    combineddatafileentry = pd.DataFrame()
            datafile.to_csv(filein,index=False)

# Post processing
combineddatafile['normleavearea']=combineddatafile['Size in pixels']*(75/combineddatafile['petriradius'])**2   
combineddatafile[['number','condition','BR','D0','leaf']]=combineddatafile['objectid'].str.rsplit('_',n=4,expand=True)
combineddatafile = combineddatafile.fillna(0)
combineddatafile.index = np.arange(0,len(combineddatafile))
#combineddatafile = combineddatafile.loc[combineddatafile['objectid'].str.contains('Ctrl')==False]
# trial= combineddatafile.groupby(['condition','day']).mean().reset_index()
# trial.set_index('day',inplace=True)
# trial.groupby('condition')['necrosis_leaffraction'].plot(legend=True)

# Plotting
f=sns.lineplot(data=combineddatafile,x='day',y='necrosis_leaffraction',hue='condition')
f.set_title('Necrosis over time')
f.set_ylabel("Necrosis (fraction leaf)")
for i in np.unique(combineddatafile['condition']):
    subsetfile = combineddatafile.loc[combineddatafile['condition']==i]    
    subsetfile = subsetfile.drop_duplicates(subset=['day','objectid'])
    subsetfile.index = np.arange(0, len(subsetfile))
    g = sns.relplot(data = subsetfile, x = "day", y = "necrosis_leaffraction",
                col = "objectid", hue = "objectid",
                kind = "line", palette = "Spectral",   
                linewidth = 4, zorder = 5,
                col_wrap = 5, height = 3, aspect = 1.5, legend = False
               )

    #add text and silhouettes
    for time, ax in g.axes_dict.items():
        ax.text(.1, .85, time,
                transform = ax.transAxes, fontweight="bold"
                )
        sns.lineplot(data = subsetfile, x = "day", y = "necrosis_leaffraction", units="objectid",
                     estimator = None, color= ".7", linewidth=1, ax=ax
                     )

    g.set_titles("")
    g.set_axis_labels("Time (days)", "Necrosis (fraction leaf)")
    g.tight_layout()
       

    
# trialimage = np.load('C:/Users/vinkjo/Downloads/OneDrive_2022-03-17/230222 1 D0_h5/IMG_9796.JPG_Object Identities.npy')
# trialimage = np.squeeze(trialimage)
# numberofobjects = np.amax(trialimage)
# for i in range(1,numberofobjects+1):
#     print(i)
#     mask = np.isin(trialimage,i)
#     print(mask.sum())
                           