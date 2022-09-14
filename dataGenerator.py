# custom data generator
# without data sugmentation

import os
import numpy as np
from utils import *

# load images
def load_img(img_dir, img_list):
    
    images = []
    
    for i, image_name in enumerate(img_list):
        
        # could add pre-process here 
        
        if (image_name.split('.')[1] == 'npy'):
            
            image = np.load(img_dir + image_name)  # load npy if data type is correct
            images.append(image)
        else:
            print('illegal data format')
            
    images = np.array(images)  # convert into array
    
    return images

def imageLoader(img_dir, img_list, mask_dir, mask_list, batch_size):
    L = len(img_list)
    
    # keras require generator to be infinite, so we use while true
    while True:
        
        batch_start = 0
        batch_end = batch_size
        
        while batch_start < L:
            
            limit = min(batch_end, L) # 考虑最后一个batch分割不完整的情况
            
            # X = load_img(img_list[batch_start:limit])
            # Y = load_img(mask_list[batch_start:limit])
            
            X = load_img(img_dir, img_list[batch_start:limit])
            Y = load_img(mask_dir, mask_list[batch_start:limit])
            
            yield(X,Y) # output the X and Y in batch size
            
            batch_start += batch_size # 都往后挪一个batch
            batch_end += batch_size

            
# the deep 2D UNet requires different data size

def imageLoaderDeep(img_dir, img_list, mask_dir, mask_list, batch_size):
    L = len(img_list)
    
    # keras require generator to be infinite, so we use while true
    while True:
        
        batch_start = 0
        batch_end = batch_size
        
        while batch_start < L:
            
            limit = min(batch_end, L) # 考虑最后一个batch分割不完整的情况
            
            # X = load_img(img_list[batch_start:limit])
            # Y = load_img(mask_list[batch_start:limit])
            
            X = load_img(img_dir, img_list[batch_start:limit])
            Y = load_img(mask_dir, mask_list[batch_start:limit])
            
            # remove the last channel
            X = np.squeeze(X)
            Y = np.argmax(Y, axis=-1).astype('float64')
            
            yield(X,Y) # output the X and Y in batch size
            
            batch_start += batch_size # 都往后挪一个batch
            batch_end += batch_size
            

# generator for 3D U-Net
def imageLoader3D(img_dir, img_list, mask_dir, mask_list, batch_size):
    L = len(img_list)
    
    # keras require generator to be infinite, so we use while true
    while True:
        
        batch_start = 0
        batch_end = batch_size
        
        while batch_start < L:
            
            limit = min(batch_end, L) # 考虑最后一个batch分割不完整的情况
            
            # X = load_img(img_list[batch_start:limit])
            # Y = load_img(mask_list[batch_start:limit])
            
            X = load_img(img_dir, img_list[batch_start:limit])
            Y = load_img(mask_dir, mask_list[batch_start:limit])
            
            # remove the last channel
            Y = np.argmax(Y, axis=-1).astype('float64') 
            Y = np.expand_dims(Y, axis=4) # keep the shape [None, W, H, D, CH=1]
            
            yield(X,Y) # output the X and Y in batch size
            
            batch_start += batch_size # 都往后挪一个batch
            batch_end += batch_size
            

# generator for 3D U-Net. pad slices to stack
def imageLoader3DSlice(img_dir, img_list, mask_dir, mask_list, batch_size):
    L = len(img_list)
    
    # keras require generator to be infinite, so we use while true
    while True:
        
        batch_start = 0
        batch_end = batch_size
        
        while batch_start < L:
            
            limit = min(batch_end, L) # 考虑最后一个batch分割不完整的情况
            
            # X = load_img(img_list[batch_start:limit])
            # Y = load_img(mask_list[batch_start:limit])
            
            X = load_img(img_dir, img_list[batch_start:limit])
            Y = load_img(mask_dir, mask_list[batch_start:limit])
            
            # remove the last channel
            Y = np.argmax(Y, axis=-1).astype('float64') 
            Y = np.expand_dims(Y, axis=4) # keep the shape [None, W, H, D, CH=1]
            
            # image to slice to stack
            X = slice2stack(X[:,0:63:4,...], [1,1,0.25]) 
            
            yield(X,Y) # output the X and Y in batch size
            
            batch_start += batch_size # 都往后挪一个batch
            batch_end += batch_size
            

# image generator for cycleGAN
def imageLoader3DIMG(img_dir, img_list, mask_dir, mask_list, batch_size):
    L = len(img_list)
    
    # keras require generator to be infinite, so we use while true
    while True:
        
        batch_start = 0
        batch_end = batch_size
        
        while batch_start < L:
            
            limit = min(batch_end, L) # 考虑最后一个batch分割不完整的情况
            
            X = load_img(img_dir, img_list[batch_start:limit])
            X = (X-0.5) / 0.5 # [0,1] -> [-1, 1]

            yield(X)
            
            batch_start += batch_size # 都往后挪一个batch
            batch_end += batch_size
            
def imageLoader3DMSK(img_dir, img_list, mask_dir, mask_list, batch_size):
    L = len(img_list)
    
    # keras require generator to be infinite, so we use while true
    while True:
        
        batch_start = 0
        batch_end = batch_size
        
        while batch_start < L:
            
            limit = min(batch_end, L) # 考虑最后一个batch分割不完整的情况
            
            # X = load_img(img_dir, img_list[batch_start:limit])
            Y = load_img(mask_dir, mask_list[batch_start:limit])
            
            # remove the last channel
            Y = np.argmax(Y, axis=-1).astype('float64') 
            Y = np.expand_dims(Y, axis=4) # keep the shape [None, W, H, D, CH=1]
            Y = (Y-0.5) / 0.5  # [0,1] -> [-1, 1]
            
            # yield(X,Y) # output the X and Y in batch size
            yield(Y)
            
            batch_start += batch_size # 都往后挪一个batch
            batch_end += batch_size