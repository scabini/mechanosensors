# -*- coding: utf-8 -*-
"""
Created on Tue Mar 29 20:15:28 2022

@author: scabini
"""
import pandas as pd
import cv2 as cv
from sklearn import linear_model
from pre_processing import pre_process
from machine_learning import getListOfFiles,get_feature,get_features
import numpy as np
import matplotlib.pyplot as plt

dataset = 'C:/Users/Svartox/Documents/datasets/Mecanosensors/' #path of the data (root folder contains our data)

images_path = dataset + 'cropped_images/'
masks_path = dataset + 'new_masks/'

images = getListOfFiles(images_path) 
masks  = getListOfFiles(masks_path) 
  
with open(dataset+'curves.txt') as f:
    lines = f.readlines()

equipment_curve = [float(i.split('\n')[0]) for i in lines]
equipment_curve = np.asarray(equipment_curve)

X, y = get_features(images, masks, colorspace='HSV', disc_factor=1.0)


regressor = linear_model.ElasticNetCV(cv=5).fit(X, y)

sample = 'S1'

video = cv.VideoCapture(dataset+ 'videos/' + sample + '.MOV')
success,frame = video.read()

fourcc = cv.VideoWriter_fourcc('m', 'p', '4', 't')
out = cv.VideoWriter(dataset+ 'videos/' + sample +  '_predictions.mp4',fourcc,30,(1920,1080))

i=0
j=0

if sample == 'S1':
    starting_frame = 140 #S1
elif sample == 'S2':
    starting_frame = 265 #S2
elif sample == 'S3':
    starting_frame = 165 #S3
elif sample == 'S4':
    starting_frame = 380 #S4
elif sample == 'S5':
    starting_frame = 335 #S5

preds=[]
gtruth=[]
while success:    
    img, mask = pre_process(frame)
    features = get_feature(img, mask, colorspace='HSV')
                           
    reg_pred = regressor.predict(features.reshape(-1, 690))
    preds.append(reg_pred[0])
    gtruth.append(equipment_curve[j])
    # cv.putText(frame, ' predicted: ' +  str(np.round(reg_pred[0], decimals=2)), (250, 850), cv.FONT_HERSHEY_SIMPLEX, 4, (0, 0, 255), 7)
    # cv.putText(frame, 'equipment: ' + str(equipment_curve[j]), (250, 1000), cv.FONT_HERSHEY_SIMPLEX, 4 , (255, 0, 0), 7) 
    
    print("g.truth=", equipment_curve[j], '--- ElasticNet=', reg_pred)
    
    if i > starting_frame and (i-starting_frame)%3==0  and j < 191:        
        j = min(j +1, 190)


    # out.write(frame)    
    i=i+1
    success,frame = video.read()


    
video.release()

# out.release()

cv.destroyAllWindows()


fig, ax1 = plt.subplots(nrows=1, ncols=1, figsize=(3.4, 2.4))

ax1.plot([i/30 for i in range(len(gtruth))], gtruth, color='blue', label='equipment')
ax1.plot([i/30 for i in range(len(gtruth))], preds, color='red', linestyle='--', label='prediction')



# plt.title(sample)
plt.xlabel('video time (seconds)', fontsize=13)
plt.ylabel('tension $\epsilon$ ($\%$)', fontsize=13)
plt.legend(loc='lower center', bbox_to_anchor=(0.5, 0., 0.11, 0.5))
plt.tight_layout()
plt.savefig('plots/' + sample +'_videoValidation.jpg', dpi=350) 

df = []
df.append([i/30 for i in range(len(gtruth))])
df.append(gtruth)
df.append(preds)
df = pd.DataFrame(df)
df.to_excel('plots/' + sample +'_videoValidation.xlsx', index=False, header=False)






