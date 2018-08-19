#! /usr/bin/python3

'''
A runnable version of distracted_driver_detection, without any GUI displayed.
Only get the data and train the model on gpu, then generate the test results.
Author: nick
Date: 2018-08-19
'''

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
from torchvision import datasets, models, transforms
import os
import time
import copy
from sklearn.model_selection import KFold
import warnings
warnings.filterwarnings("ignore")

# Global Variable
csv_path = 'driver_imgs_list.csv'
train_img_path = os.path.join('imgs','train')
test_img_path = os.path.join('imgs','testroot')
cache_path = os.path.join('cache')
result_path = os.path.join('result')
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
use_cuda = torch.cuda.is_available()
    
# Hyperparameters
N_EPOCHS = 20
BATCH_SIZE = 16
LR = 0.001
PIN_MEMORY = use_cuda

# get data info from csv
def driver_image_list(csv_path):
    driver_image_list = pd.read_csv(csv_path)
    return driver_image_list

# transform function for valid and test data
def transform_data_normal(target_size=224):
    return transforms.Compose([
        transforms.Resize(256)
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
    ])

# transform function for train data with augmentation
def transform_data_aug(target_size=224):
    return transforms.Compose([
        transforms.RandomCrop(224),
        transforms.ToTensor(),
        transforms.NOrmalize((0.4914,0.4822,0.4465),(0.2023,0.1994,0.2010))
    ])


if __name__ == "__main__":
    print(device)
