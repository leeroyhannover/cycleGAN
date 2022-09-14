# data preparation for raw mrc file. data reading, scaling, croping, adjusting data format and split into train/val/test
# data from 07.2022. single molecule

import os
import mrcfile
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
from tqdm import tqdm  # ! this might result into problem with 'object'
import os
import pandas as pd
import random
import argparse
from tensorflow.keras.utils import to_categorical
import glob

import splitfolders  # use to segment the data

from natsort import natsorted


# read into the data
def readMRC(path):
    with mrcfile.open(path, mode='r+', permissive=True) as mrc:
        mrc.header.map = mrcfile.constants.MAP_ID # for synthetic data, need to generate ID
        data = mrc.data
    return data

# visualize
def visusalizeIMG(n_slice, test_img, test_msk):
    # n_slice = random.randint(0, test_img.shape[2])
    plt.figure(figsize=(8, 8))

    plt.subplot(121)
    plt.imshow(test_img[n_slice,:,:], cmap='gray')
    plt.title('focal slice')
    plt.subplot(122)
    plt.imshow(test_msk[n_slice,:,:], cmap='gray')
    plt.title('GT slice')
    plt.show()

# normalize into [0, 1]
def normStack(imageStack):
    Range = np.max(imageStack) - np.min(imageStack)
    normTemp = ((imageStack - np.min(imageStack))/Range - 0.5) * 2 
    
    return (normTemp +1)/2 
    

if __name__ == "__main__":
    
    PATH = 'H:/My Drive/rawData/MDC_HZDR/simulation_20220818/'
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', default= PATH + 'focal/', help='path to the images')
    parser.add_argument('--mask_dir', default= PATH + 'GT/', help='path to the labels')
    parser.add_argument('--crop_size', default=(18, 82), help='crop the image to enhance SNR')
    parser.add_argument('--threshold', type=float, default=0.825, help='threshold the normalized masks')
    
    # parser.add_argument('--split_val', default=20, help='number of images for validation')
    # parser.add_argument('--split_test', default=19, help='number of images for testing')
    # parser.add_argument('--resolution', default=[1, 1, 1], help='New Resolution to resample the data to same spacing')
    # parser.add_argument('--smooth', default=False, help='Set True if you want to smooth a bit the binary mask')
    
    args = parser.parse_args()
    
    if not os.path.isdir('./preData_08_2022'):
        os.mkdir('./preData_08_2022')
        
    SAVED_PATH = 'E:/EM/demo/3D_Unet_keras/preData_08_2022/'
        
    img_list = natsorted(glob.glob(args.img_dir + '/*focal.mrc'))
    msk_list = natsorted(glob.glob(args.mask_dir + '/*gt.mrc'))
    
    
    num_images = len(img_list)
    print('images number:', num_images)

    for num in range(num_images):
        
        temp_img = readMRC(img_list[num])
        temp_msk = readMRC(msk_list[num])
        
        # # rescale into [0,1]
        # temp_img = np.interp(temp_img, (temp_img.min(), temp_img.max()), (0, 1))
        # temp_msk = np.interp(temp_msk, (temp_msk.min(), temp_msk.max()), (0, 1))
        
        # normalize the data [0, 1]
        temp_img = normStack(temp_img)
        temp_msk = normStack(temp_msk)
        
        # binary the mask
        # temp_msk = (temp_msk < args.threshold).astype(int)  # for the 07.2022 data < 0.825
        temp_msk = (temp_msk > args.threshold).astype(int)  # for the 08.2022 data > 0.45
        
        # crop the stack 
        temp_msk = temp_msk[args.crop_size[0]:args.crop_size[1], args.crop_size[0]:args.crop_size[1], args.crop_size[0]:args.crop_size[1]]
        temp_img = temp_img[args.crop_size[0]:args.crop_size[1], args.crop_size[0]:args.crop_size[1], args.crop_size[0]:args.crop_size[1]]
        
        # adjust the data format
        # temp_img = np.stack([temp_img, temp_img, temp_img], axis = 3) # use for transfer learning in 3 channels
        temp_img = np.stack([temp_img], axis = 3)
        temp_msk = to_categorical(temp_msk, num_classes=2) 
        
        # save file
        val, counts = np.unique(temp_msk, return_counts=True)
        
        if (1 - (counts[0]/counts.sum())) > 0.01:  #At least 1% useful volume with labels that are not 0
            # print("Save Me:", num)
            np.save(SAVED_PATH + 'images/image_'+str(num)+'.npy', temp_img)
            np.save(SAVED_PATH + 'masks/mask_'+str(num)+'.npy', temp_msk)

        else:
            print("too low SNR:", num)
    
    # split data into training, validation and testing
    OUTPUT_PATH = 'E:/EM/demo/3D_Unet_keras/data/2022_08/inputData/'
    # splitfolders.ratio(SAVED_PATH, output=OUTPUT_PATH, seed=42, ratio=(.8, .1, .1), group_prefix=None) # default values
    splitfolders.ratio(SAVED_PATH, output=OUTPUT_PATH, seed=42, ratio=(.8, .2), group_prefix=None)  # test will be manually assigned

        