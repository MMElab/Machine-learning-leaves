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

# Load datafiles
maxleafdifference = 200
petriradius = 75

# Specify folder path (usual structure is multifolderpath/Experiment_DX_h5)
# multifolder = 'C:\\Users\\vinkjo\\OneDrive - Victoria University of Wellington - STAFF\\Desktop\\Machine learning Leaves\\Raw data\\3770'
# multifolderpath = Path(multifolder)

multifolder = filedialog.askdirectory()
multifolderpath = Path(multifolder)
originalfolderpath = list(multifolderpath.glob("*D0_h5"))[0]
combineddatafile = pd.DataFrame()
# Calculate properties of leaves in original (Day 0) folder
Sumtot_original = pd.DataFrame()
for filein in originalfolderpath.glob("*Summaryoutput.csv"):
    datafile_original = pd.read_csv(filein)
    filename = os.path.basename(datafile_original['filename'][0])
    numberofleaves = len(datafile_original)
    normalizedscale = petriradius/datafile_original['petriradius'][0]
    datafile_original['normleavearea']=datafile_original['Size in pixels']*(normalizedscale**2)
    Sumtot_original = Sumtot_original.append([[filename,numberofleaves,normalizedscale,list(datafile_original['normleavearea']),list(datafile_original['object_id'])]])

# Calculate properties of leaves in other (>Day 0) folders
for folderpath in multifolderpath.glob("*_h5"):
    if folderpath == originalfolderpath:
        continue
    else:
        Sumtot = pd.DataFrame()
        for filein in folderpath.glob("*Summaryoutput.csv"):
            datafile = pd.read_csv(filein)
            filename = os.path.basename(datafile['filename'][0])
            numberofleaves = len(datafile)
            normalizedscale = petriradius/datafile['petriradius'][0]
            datafile['normleavearea']=datafile['Size in pixels']*(normalizedscale**2)
            Sumtot = Sumtot.append([[filename,numberofleaves,normalizedscale,list(datafile['normleavearea']),list(datafile['object_id'])]])
        
# Match leaves from different frames based on size       
        bestpartnerdict = dict()
        leaflinkdict = dict()
        numberofimages = len(Sumtot)+1
        distanceleafsizes=np.full((numberofimages,numberofimages),10000)
        leaflinkarray=np.full((numberofimages,numberofimages),dict())
        for j,k in enumerate(Sumtot[3]):
            sortedsizes= np.sort(k)
            sortedindices = np.argsort(k)
            bestmatch=10000
            for l,m in enumerate(Sumtot_original[3]):
                if len(m)!=len(k):
                    # print(j)
                    # print(k)
                    continue
                sortedsizes1= np.sort(m)
                sortedindices1 = np.argsort(m)
                distanceleafsizes[j,l]=np.sum(abs(sortedsizes-sortedsizes1))
                if np.sum(abs(np.array(k)-np.array(m)))/len(m)<maxleafdifference:
                    leaflinkarray[j,l] = dict(zip(range(1,len(m)+1),range(1,len(m)+1)))
                else:
                    leaflinkarray[j,l]= dict(zip(sortedindices+1,sortedindices1+1))
                # if np.sum(trial)<bestmatch:
                #     bestmatch=np.sum(trial)
                #     bestpartner = l
                #     if np.sum(abs(np.array(k)-np.array(m)))/len(m)<maxleafdifference:
                #         leafdict = dict(zip(range(1,len(m)+1),range(1,len(m)+1)))
                #     else:
                #         leafdict = dict(zip(sortedindices,sortedindices1))   
        bestmatch1,bestmatch2 = linear_sum_assignment(distanceleafsizes)
        bestpartnerdict = dict(zip(bestmatch1,bestmatch2))
            #bestpartnerdict[j]=bestpartner
        for j in range(0,len(leaflinkarray)):
            leaflinkdict[j]=leaflinkarray[j,bestpartnerdict[j]]
        if len(np.unique(bestpartnerdict.values())[0])==len(np.unique(list(bestpartnerdict.values()))):
            print('Linking has ended. Datasets have been linked to unique images.')
        else:
            print('Linking has ended. Some datasets are linked erronously. Check the problem')              
        
        ## Make combined datafile
        combineddatafileentry = pd.DataFrame()
        for filein in folderpath.glob("*Summaryoutput.csv"):
            datafile = pd.read_csv(filein)
            filename = os.path.basename(datafile['filename'][0])
            location = np.where(Sumtot[0].str.contains(filename))[0][0]
            datafile['filename_original']=Sumtot_original.iloc[bestpartnerdict[location]][0]
            day = int(re.search("D(.{1,2})_h5", str(filein))[0][1:-3])
            leafdict = leaflinkdict[location]
            if len(leafdict)>0:
                if len(datafile)==3 or len(datafile)==5:
                    datafile['labelimage_oid_original'] = (datafile['object_id']+1).map(leafdict).astype(int)
                    combineddatafileentry['objectid'] = datafile['filename_original'].astype(str) +'_'+datafile['labelimage_oid_original'].astype(str)
                    combineddatafileentry['day']=day
                    combineddatafileentry[['Size in pixels','petriradius','necrosis_area','necrosis_distance','necrosis_leaffraction','plug_id']]=datafile[['Size in pixels','petriradius','necrosis_area','necrosis_distance','necrosis_leaffraction','plug_id']]
                    combineddatafile=combineddatafile.append(combineddatafileentry)
                    combineddatafileentry = pd.DataFrame()
            datafile.to_csv(filein,index=False)
combineddatafile['normleavearea']=combineddatafile['Size in pixels']*(75/combineddatafile['petriradius'])**2
        
combineddatafile[['number','condition','BR','D0','leaf']]=combineddatafile['objectid'].str.rsplit('_',n=4,expand=True)
#combineddatafile = combineddatafile.loc[combineddatafile['objectid'].str.contains('Ctrl')==False]

combineddatafile = combineddatafile.fillna(0)
trial= combineddatafile.groupby(['condition','day']).mean().reset_index()
trial.set_index('day',inplace=True)
trial.groupby('condition')['necrosis_leaffraction'].plot(legend=True)

       

    
# trialimage = np.load('C:/Users/vinkjo/Downloads/OneDrive_2022-03-17/230222 1 D0_h5/IMG_9796.JPG_Object Identities.npy')
# trialimage = np.squeeze(trialimage)
# numberofobjects = np.amax(trialimage)
# for i in range(1,numberofobjects+1):
#     print(i)
#     mask = np.isin(trialimage,i)
#     print(mask.sum())
                           