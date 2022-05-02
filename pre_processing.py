# -*- coding: utf-8 -*-
"""
Created on Mon Mar 28 13:23:59 2022

@author: scabini
"""

import os
import glob
# import scipy.io
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2 as cv
from PIL import Image
from PIL.TiffTags import TAGS
import imagej
import numpy as np
import csv
from matplotlib.patches import Rectangle

dataset = '' #path of the data (root folder contains our data)

def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles

def pre_process(img):

        
    # sample = images[0]
    # image = mpimg.imread(sample) 
    # img = cv.imread(sample, -1)
    img = img[300:630, 800:1060]
    # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    
    
    # cv.imwrite(dataset + 'cropped_images/' + sample.split('/')[-1], cv.cvtColor(img, cv.COLOR_BGR2RGB))
    img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
    
    mask = cv.inRange(img, (85, 0, 0), (135, 255,255))
    
    ## slice the blue
    imask = mask>0
    blues = np.zeros_like(img, np.uint8)
    blues[imask] = img[imask]
    blues = cv.cvtColor(blues, cv.COLOR_HSV2BGR)
    # cv.imwrite(dataset + 'new_masks/greens_' + sample.split('/')[-1], blues)
    

    thresh1 = cv.cvtColor(blues, cv.COLOR_RGB2GRAY)
    
    _,thresh1 = cv.threshold(thresh1,50,255,cv.THRESH_BINARY)
    
    kernel = np.ones((7,7), np.uint8)  
    thresh1 = cv.erode(thresh1, kernel, iterations=1)  
    
    kernel = np.ones((3,3), np.uint8)  
    thresh1 = cv.dilate(thresh1, kernel, iterations=1)  
    
    kernel = np.ones((3,3), np.uint8)  
    thresh1 = cv.erode(thresh1, kernel, iterations=1)  
    
    kernel = np.ones((3,3), np.uint8)  
    thresh1 = cv.dilate(thresh1, kernel, iterations=1)  
    
    return cv.cvtColor(img, cv.COLOR_HSV2BGR), thresh1

def pre_processing(images):
    
    # images = getListOfFiles(images_path)    
    
    # for sample in masks:
    #     data = scipy.io.loadmat(sample)  
    #     mask = data['mask']
    #     plt.imshow(mask)
    #     plt.show()
    
    for sample in images:
        
        # sample = images[0]
        # image = mpimg.imread(sample) 
        # img = cv.imread(sample, -1)
        img = mpimg.imread(sample)
        img = img[1200:2350, 2400:3180]
        # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        
        
        cv.imwrite(dataset + 'cropped_images/' + sample.split('/')[-1], cv.cvtColor(img, cv.COLOR_BGR2RGB))
        img = cv.cvtColor(img, cv.COLOR_RGB2HSV)
        
        # _,threshR = cv.threshold(img[:,:,0],50,255,cv.THRESH_BINARY)
        # _,threshG = cv.threshold(img[:,:,1],50,255,cv.THRESH_BINARY)
        # _,threshB = cv.threshold(img[:,:,1:2],100,255,cv.THRESH_BINARY)
        mask = cv.inRange(img, (85, 10, 10), (135, 255,255))
        
        ## slice the green
        imask = mask>0
        blues = np.zeros_like(img, np.uint8)
        blues[imask] = img[imask]
        blues = cv.cvtColor(blues, cv.COLOR_HSV2BGR)
        # cv.imwrite(dataset + 'new_masks/greens_' + sample.split('/')[-1], blues)
        

        thresh1 = cv.cvtColor(blues, cv.COLOR_RGB2GRAY)
        
        _,thresh1 = cv.threshold(thresh1,50,255,cv.THRESH_BINARY)
        
        kernel = np.ones((15,15), np.uint8)  
        thresh1 = cv.erode(thresh1, kernel, iterations=1)  
        
        kernel = np.ones((3,3), np.uint8)  
        thresh1 = cv.dilate(thresh1, kernel, iterations=1)  
        
        kernel = np.ones((7,7), np.uint8)  
        thresh1 = cv.erode(thresh1, kernel, iterations=1)  
        
        kernel = np.ones((3,3), np.uint8)  
        thresh1 = cv.dilate(thresh1, kernel, iterations=1)  
        
        cv.imwrite(dataset + 'new_masks/' + sample.split('/')[-1], thresh1)
        
        
        
# images_old = getListOfFiles(dataset + 'old_samples/')

# new_names = [i.split('/')[-1] for i in images_old]


# for i in range(len(new_names)):
#     new_names[i]='S' + str(int(new_names[i][1])+5) + new_names[i][2:]

# images_old_new = [dataset + 'all_samples/'+new_names[i] for i in range(len(new_names))]


# for file in range(len(images_old_new)):
#     os.rename(images_old[file], images_old_new[file])





# # images_old = getListOfFiles(dataset + 'old_samples/')
# images_new = getListOfFiles(dataset + 'new_samples/')
# # images_all = getListOfFiles(dataset + 'all_samples/')

# files = glob.glob(dataset + 'cropped_images/*')
# for f in files:
#     os.remove(f)
    
# files = glob.glob(dataset + 'new_masks/*')
# for f in files:
#     os.remove(f)

# pre_processing(images_new)

























    
    
    
    
    
