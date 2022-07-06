# -*- coding: utf-8 -*-
"""
Created on Thu Mar 10 10:03:52 2022

@author: scabini
"""


import os
# import scipy.io
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import cv2 as cv
import numpy as np
import pandas as pd
from sklearn import linear_model
from sklearn.neighbors import KNeighborsClassifier
from sklearn import svm
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import LeaveOneOut
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from matplotlib.patches import Patch
from sklearn import preprocessing
from sklearn.metrics import accuracy_score

dataset = '' #path of the data (root folder contains our data)

images_path = dataset + 'cropped_images/'
masks_path = dataset + 'new_masks/'

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
    

def get_feature(img, mask, colorspace, disc_factor=1.0):
    factor = disc_factor * 256
    step= int(np.round(256/factor))
    # print(step)
    bins = [i for i in range(0,256,step)]
    if colorspace == 'RGB':
        binsH = [i for i in range(0,256,step)]
    elif colorspace =='HSV':
        factor = disc_factor * 181
        step= int(np.round(181/factor))
        binsH = [i for i in range(0,181,step)]        
        img = cv.cvtColor(img, cv.COLOR_RGB2HSV)
     
    ret,mask = cv.threshold(mask,200,255,cv.THRESH_BINARY)
    mask= np.where(mask  == 255.)

    imgR = img[:,:,0]
    hist_R,_ = np.histogram(imgR[mask], bins=binsH)

    imgG = img[:,:,1]
    hist_G,_ = np.histogram(imgG[mask], bins=bins)
    # hist_G[0]=0
    
    imgB = img[:,:,2]
    # print(np.max(imgR), np.max(imgG), np.max(imgB))
    hist_B,_ = np.histogram(imgB[mask], bins=bins)
    # hist_B[0]=0
    
    return np.hstack([hist_R/sum(hist_R), hist_G/sum(hist_G), hist_B/sum(hist_B)])
 
    
def get_features(images, masks, colorspace, disc_factor=1.0):
    factor = disc_factor * 256
    step= int(np.round(256/factor))
    # print(step)
    bins = [i for i in range(0,256,step)]
    if colorspace == 'RGB':
        binsH = [i for i in range(0,256,step)]
    elif colorspace =='HSV':
        factor = disc_factor * 181
        step= int(np.round(181/factor))
        binsH = [i for i in range(0,181,step)]        

    feature_matrix = np.zeros((len(images),len(binsH)-1+ (len(bins)-1)*2))
    classes = np.zeros((len(images)))
    for i in range(len(images)):
        
        # img = cv.cvtColor(img, cv.COLOR_RGB2YCrCb)
        # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        classe = images[i].split('/')[-1]
        classe = classe.split('.')[0].split('_')[-1]
        classes[i] = classe
        
        img = mpimg.imread(images[i])
        
        #simple filtering
        # img = cv.GaussianBlur(img,(7,7),cv.BORDER_DEFAULT)
    
        if colorspace =='HSV':
            img = cv.cvtColor(img, cv.COLOR_BGR2HSV)
        
        # mask = np.round(mpimg.imread(masks[i])/255.0)
        mask = mpimg.imread(masks[i])
        ret,mask = cv.threshold(mask,200,255,cv.THRESH_BINARY)
        mask= np.where(mask  == 255.)

        imgR = img[:,:,0]
        hist_R,_ = np.histogram(imgR[mask], bins=binsH)

        imgG = img[:,:,1]
        hist_G,_ = np.histogram(imgG[mask], bins=bins)
        # hist_G[0]=0
        
        imgB = img[:,:,2]
        # print(np.max(imgR), np.max(imgG), np.max(imgB))
        hist_B,_ = np.histogram(imgB[mask], bins=bins)
        # hist_B[0]=0
        
        feature_matrix[i] = np.hstack([hist_R/sum(hist_R), hist_G/sum(hist_G), hist_B/sum(hist_B)])
        # return img, hist_R/sum(hist_R), hist_G/sum(hist_G), hist_B/sum(hist_B)
    return feature_matrix, classes

def get_manualFolds(images):
    foldindex=[]
    for i in range(len(images)):        
        # img = cv.cvtColor(img, cv.COLOR_RGB2YCrCb)
        # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        classe = images[i].split('/')[-1]
        classe = int(classe.split('.')[0].split('_')[-1])
        
        sample = images[i].split('/')[-1]
        sample = int(sample[1])
        
        replica = images[i].split('/')[-1]
        replica = int(replica[4])
        
        foldindex.append(sample) #sensor-wise validation
        # foldindex.append(replica)
        # foldindex.append(classe)
        
    return foldindex
    
    
    
images = getListOfFiles(images_path) 
masks  = getListOfFiles(masks_path) 

foldindex = get_manualFolds(images)

repetitions=1   
colorspace = 'HSV'

X, y = get_features(images, masks, colorspace=colorspace, disc_factor=1.0)

# le = preprocessing.LabelEncoder()
# le.fit(y)
# y=le.transform(y)

accs=[]
regs=[]
reg_preds=[]
fig1, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(3,3)) 
# ax1 = ax1.flat
df = []
for it in range(repetitions):
    # crossval = LeaveOneOut()
    # crossval = StratifiedKFold(n_splits=5, shuffle=True, random_state=it*666)
    # crossval.get_n_splits(X)
    # for train_index, test_index in crossval.split(X, foldindex):
    #     X_train, X_test = X[train_index], X[test_index]
    #     y_train, y_test = y[train_index], y[test_index]
    for fold in np.unique(foldindex):        
        X_train, X_test = X[foldindex!=fold], X[foldindex==fold]
        y_train, y_test = y[foldindex!=fold], y[foldindex==fold]
        
        ##### select between different classification models
        classifier = KNeighborsClassifier(n_neighbors=1)
        # classifier = svm.SVC(kernel='rbf', C=3)
        # classifier = LinearDiscriminantAnalysis(solver='svd')
    
               
        classifier.fit(X_train, y_train)
        
        preds=classifier.predict(X_test)
        acc=accuracy_score(y_test, preds)
        accs.append(acc)
       
        
        ##### select between different regression models
        # regressor = linear_model.LinearRegression().fit(X_train, y_train) 
        
        # regressor = linear_model.SGDRegressor(verbose=False).fit(X_train, y_train) 
        # regressor = linear_model.RidgeCV(cv=5).fit(X_train, y_train)
        regressor = linear_model.ElasticNetCV(cv=5).fit(X_train, y_train)
        
        reg_pred = regressor.predict(X_test)
        # reg_preds.append(reg_pred)
            
        ax1.scatter(y_test, reg_pred, color='black', s=10)
        df.append([y_test, reg_pred])    
        reg= regressor.score(X_test, y_test)
        
        regs.append(reg)
        # print(acc)

acc=np.mean(accs)
std= np.std(accs)
print('classification accuracy='+ str(np.round(acc, decimals=2)) + ' ($\pm$' + str(np.round(std, decimals=2)) + ')')  

      
ax1.plot([x for x in np.unique(y)],[x for x in np.unique(y)])   
ax1.set_xticks([x for x in np.unique(y)]) 
ax1.set_yticks([x for x in np.unique(y)]) 
acc=np.mean(regs)
std= np.std(regs)
# ax1.set_title('least squares')
print('ElasticNet, $R^2$='+ str(np.round(acc, decimals=2)) + ' ($\pm$' + str(np.round(std, decimals=2)) + ')')
ax1.set_title('ElasticNet, $R^2=$'+ str(np.round(acc, decimals=2)) + ' ($\pm$' + str(np.round(std, decimals=2)) + ')')  
ax1.set_xlabel('ground truth $\epsilon$ ($\%$)', fontsize=13)
ax1.set_ylabel('predicted $\epsilon$ ($\%$)', fontsize=13)
fig1.tight_layout()
# fig1.savefig('plots/' + colorspace + "_regression.jpg", dpi=350)   


# df.append([x for x in np.unique(y)])
# df.append([x for x in np.unique(y)])

df = pd.DataFrame(np.asarray(df).reshape((2,33*5)))
df.to_excel('plots/' + colorspace + '_regression.xlsx', index=False, header=False)

############ VIZUALIZATIONS ###############
def get_hists(images, masks, colorspace):
    if colorspace == 'RGB':
        binsH = [i for i in range(0,256)]
    elif colorspace =='HSV':
        binsH = [i for i in range(0,180)]
        
    bins = [i for i in range(0,256)]
    for i in range(len(images)):
        img = mpimg.imread(images[i])
        # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        
        if colorspace =='HSV':
            img = cv.cvtColor(img, cv.COLOR_RGB2HSV)
        
        # mask = np.round(mpimg.imread(masks[i])/255.0)
        mask = mpimg.imread(masks[i])
        ret,mask = cv.threshold(mask,200,255,cv.THRESH_BINARY)
        mask= np.where(mask  == 255.)

        imgR = img[:,:,0]
        hist_R,_ = np.histogram(imgR[mask], bins=binsH)
        
        imgG = img[:,:,1]
        hist_G,_ = np.histogram(imgG[mask], bins=bins)
        # hist_G[0]=0
        
        imgB = img[:,:,2]
        hist_B,_ = np.histogram(imgB[mask], bins=bins)
        
        return img, hist_R/sum(hist_R), hist_G/sum(hist_G), hist_B/sum(hist_B)


fig2, ax1 = plt.subplots(nrows=1, ncols=2, figsize=(3, 3)) 
ax1 = ax1.flat

# a=0
a=0

imgA, hist_R, hist_G, hist_B= get_hists(images[a:a+1], masks[a:a+1], colorspace=colorspace)
# img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

plt.subplot(2, 1, 1)
print(images[a])
# plt.title(colorspace, fontsize=14)
# ax1[0].set_title(images[a].split('/')[-1])]
# x_axis = np.argwhere(hist_R!=0)
ra=plt.plot(np.argwhere(hist_R!=0),hist_R[hist_R!=0], color='red', label='hue')
rg=plt.plot(np.argwhere(hist_G!=0),hist_G[hist_G!=0], color='green', label='saturation')
rb=plt.plot(np.argwhere(hist_B!=0),hist_B[hist_B!=0], color='blue', label='value')

b=7
# # b= 6
# # b= 11
# b=22

print(images[b])
imgB, hist_R, hist_G, hist_B= get_hists(images[b:b+1], masks[b:b+1], colorspace=colorspace)
# # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)

# plt.plot(hist_R, color='red', linestyle='--')
# plt.plot(hist_G, color='green', linestyle='--')
# plt.plot(hist_B, color='blue', linestyle='--')
# plt.xlabel('intensity', fontsize=13)
# plt.ylabel('frequency (%)', fontsize=13)



pa1 = Patch(edgecolor='red', facecolor='white')
pa2 = Patch(edgecolor='green', facecolor='white')
pa3 = Patch(edgecolor='blue', facecolor='white')
#
# pb1 = Patch(edgecolor='red', linestyle='--', facecolor='white')
# pb2 = Patch(edgecolor='green', linestyle='--', facecolor='white')
# pb3 = Patch(edgecolor='blue', linestyle='--', facecolor='white')

# plt.legend(handles=[pa1, pb1, pa2, pb2, pa3, pb3],
#           labels=['', '', '', '', images[a].split('/')[-1], images[b].split('/')[-1]],
#           ncol=3, handletextpad=0.5, handlelength=1.0, columnspacing=-0.5,
#           fontsize=13)

plt.legend()
plt.tight_layout()

# plt.subplot(2, 2, 3)
# plt.title(images[a].split('/')[-1])
plt.xticks([]), plt.yticks([])

# df = []
# df.append(np.argwhere(hist_R!=0)[:,0])
# df.append(hist_R[hist_R!=0])

# df.append(np.argwhere(hist_G!=0)[:,0])
# df.append(hist_G[hist_G!=0])

# df.append(np.argwhere(hist_B!=0)[:,0])
# df.append(hist_B[hist_B!=0])

# df = pd.DataFrame(df)
# df.to_excel('plots/HSV_distribution.xlsx', index=False, header=False)

# if colorspace == 'HSV':
#     imgA = cv.cvtColor(imgA, cv.COLOR_HSV2RGB)
# plt.imshow(imgA)

# plt.subplot(2, 2, 4)
# plt.title(images[b].split('/')[-1])
# plt.xticks([]), plt.yticks([])
# if colorspace == 'HSV':
#     imgB = cv.cvtColor(imgB, cv.COLOR_HSV2RGB)
# plt.imshow(imgB)
# plt.savefig('plots/' + images[a].split('/')[-1] + images[b].split('/')[-1] + colorspace+ ".jpg", dpi=350) 

# plt.savefig('plots/FOR_PAPER' + images[a].split('/')[-1] + images[b].split('/')[-1] + colorspace+ ".jpg", dpi=350) 












