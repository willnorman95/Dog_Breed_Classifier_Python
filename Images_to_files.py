# -*- coding: utf-8 -*-
"""
Created on Mon Mar 25 12:02:28 2019

@author: Will's PC
"""
import numpy as np
import scipy.io as sio
from PIL import Image
from keras.preprocessing import image



file_mat = sio.loadmat('file_list.mat')
train_mat = sio.loadmat('train_list.mat')
test_mat = sio.loadmat('test_list.mat')
annotations_mat = sio.loadmat('test_list.mat')

path = 'Re-sized_Images/'

img_width = 200
img_height = 200

train_files = []
train_labels = []

for x in range (1,len(test_mat['file_list'])-1,2):
    
    label = test_mat['labels'][x][0]-1
    
    img_path = test_mat['file_list'][x][0][0]
    
    img = image.load_img(path+img_path,target_size=(img_width,img_height))
    
    img.save('annotation/Validation/'+str(img_path))     #+str(label)+'_'+str(x)+'.png')