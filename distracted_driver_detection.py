#! /usr/bin/python3

'''
A runnable version of distracted_driver_detection, without any GUI displayed.
Only get the data and train the model on gpu, then generate the test results.
Author: nick
Date: 2018-08-19
'''

import numpy as np
import pandas as pd
from PIL import Image
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
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
BATCH_SIZE = 32
WORKERS = 8
LR = 0.0001
DROP_RATE = 0.5
PIN_MEMORY = use_cuda
criterion = nn.CrossEntropyLoss()

# get unique_drivers and driver_img_dict from csv file
def get_driver_img_map(csv_path):
    driver_img_list = pd.read_csv(csv_path)
    unique_drivers = list(set(driver_img_list['subject']))
    driver_img_dict = {}
    for driver_id in unique_drivers:
        img_list = driver_img_list[driver_img_list['subject']==driver_id]['img'].values.tolist()
        if driver_id in driver_img_dict:
            driver_img_dict[driver_id].extend(img_list)
        else:
            driver_img_dict[driver_id] = img_list

    return unique_drivers, driver_img_dict, driver_img_list


# transform function for valid and test data
def transform_data_normal(target_size=224):
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

# transform function for train data with augmentation
def transform_data_aug(target_size=224):
    return transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

# cache data
def cache_data(data, path):
    if os.path.isdir(os.path.dirname(path)):
        file = open(path, 'wb')
        pickle.dump(data, file)
        file.close()
    else:
        print('Directory does not exists')

# restore data
def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        file = open(path, 'wb')
        data = pickle.load(file)
    return data

# predict mean for esemble
def predict_mean(data, nfolds):
    first_fold = np.array(data[0])
    for i in range(1, nfolds):
        frist_fold += np.array(data[i])
    first_fold /= nfolds
    return first_fold.tolist()

def append_chunk(main, part):
    for p in part:
        main.append(p)
    return main

# create result csv
def create_submission(predictions, testid_list, filename,result_path):
    predictions = predictions.clip(min=1e-15, max=1-1e-15)
    df = pd.DataFrame(np.array(predictions), columns=['c'+str(i) for i in range(10)])
    df.insert(0,'img',testid_list)
    path = os.path.join(result_path, filename)
    df.to_csv(path, index=None)
    print('{} generated'.format(path))



# customize dataset for cross_validation
class SplitTrainingDataset(torch.utils.data.Dataset):
    def __init__(self, driver_img_list, img_path, split_img_list, tSet=False):
        self.img_path = img_path
        self.driver_img_list = driver_img_list
        classesnames = list(set(driver_img_list['classname']))
        classesnames.sort()
        self.class_names = classesnames
        self.split_img_list = split_img_list
        self.tSet = tSet
        super(SplitTrainingDataset, self).__init__()

    def __len__(self):
        return len(self.split_img_list)

    def __getitem__(self,i):
        img_name = self.split_img_list[i]
        class_name = self.driver_img_list[self.driver_img_list['img']==img_name]['classname'].values[0]
        path = os.path.join(self.img_path, class_name, img_name)
        img = Image.open(path).convert('RGB')
        if self.tSet:
            self.transform = transform_data_aug()
        else:
            self.transform = transform_data_normal()
        img = self.transform(img)
        label = self.class_names.index(class_name)
        return img,label

# Customize pretrained model 
class Myvgg(nn.Module):
    def __init__(self, basemodel, modelname, num_classes=10):
        super(Myvgg, self).__init__()
        self.features = basemodel.features
        self.modelname = modelname
        for param in basemodel.parameters():
            param.requires_grad = False
        # Parameters of newly contructed modules have requires_grad=True by default
        self.classifier = nn.Sequential(
            nn.Linear(512*7*7,4096),
            nn.ReLU(True),
            nn.Dropout(DROP_RATE),
            nn.Linear(4096,4096),
            nn.ReLU(True),
            nn.Dropout(DROP_RATE),
            nn.Linear(4096,num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0),-1)
        x = self.classifier(x)
        return x

# train single model with data,criterion,optimizer,sheduler
def train_model(model, datasize_t, datasize_v, loader_t, loader_v, criterion, optimizer, scheduler, num_fold, num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = float("inf")
    early_stop_count = 0

    for epoch in range(num_epochs):
        if early_stop_count == 5:
            print('model did not make progress in last 5 epochs, early stop...')
            break

        print('Epoch {}/{}'.format(epoch, num_epochs-1))
        print('-'*20)
        
        # Each epoch has a trainning and validation phase
        for phase in ['train','val']:
            is_train = True
            if phase == 'train':
                scheduler.step()
                model.train()    # Set model to trainning mode
            else:
                is_train = False
                model.eval()     # Set model to evaluate mode
            
            # count loss and accuracy of phase
            running_loss = 0.0
            running_corrects = 0
            
            if is_train:
                # train phase
                for inputs, labels in loader_t:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # zero the parameter gradients because pytorch accumulate grad
                    optimizer.zero_grad()
                    with torch.set_grad_enabled(True):
                        outputs = model(inputs)
                        _,preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)
                        
                        # backworad + optimize only if in training phase
                        loss.backward()
                        optimizer.step()

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                # calculate avarage loss and accuracy of trainning dataset
                epoch_loss = running_loss / datasize_t
                epoch_acc = running_corrects.double() / datasize_t
                print('Train Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
            else:
                # valid phase
                for inputs, labels in loader_v:
                    inputs = inputs.to(device)
                    labels = labels.to(device)
                    # zero the parameter gradients
                    with torch.set_grad_enabled(False):
                        outputs = model(inputs)
                        _,preds = torch.max(outputs, 1)
                        loss = criterion(outputs, labels)

                    # statistics
                    running_loss += loss.item() * inputs.size(0)
                    running_corrects += torch.sum(preds == labels.data)
                # calculate avarage loss and accuracy of valid dataset
                epoch_loss = running_loss / datasize_v
                epoch_acc = running_corrects.double() / datasize_v
                print('Valid Loss: {:.4f} Acc: {:.4f}'.format(epoch_loss, epoch_acc))
                
                # save best model
                if epoch_acc > best_acc:
                    print('this epoch make progess')
                    best_acc = epoch_acc
                    best_model_wts = copy.deepcopy(model.state_dict())
                    best_loss = epoch_loss
                    # Reset early_stop
                    early_stop_count = 0
                else:
                    print('this epoch does not make progress')
                    early_stop_count += 1

    time_elapsed = time.time() - since
    print('Fold{}: Trainning complete in {:.0f}m {:.0f}s'.format(num_fold, time_elapsed // 60, time_elapsed % 60))
    print('Fold{}: Best val Acc: {:.4f}'.format(num_fold, best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    # save model for esemble
    torch.save(model.state_dict(), os.path.join(cache_path, model.modelname+'_'+str(num_fold)+'.pth'))
    
    return model,best_loss

# esemble model with kfold data split
def kfold_split_and_train(mymodel, csv_path, img_path, k=10, num_epochs = N_EPOCHS):    
    loss_scores = []
    unique_driver_ids, driver_img_dict, driver_img_list = get_driver_img_map(csv_path)
    kf = KFold(n_splits=k, shuffle=True, random_state=51)
    num_fold = 0
    for train_driver_ids, valid_driver_ids in kf.split(unique_driver_ids):
        num_fold += 1
        print('{} data split:'.format(num_fold))
        print(train_driver_ids)
        print(valid_driver_ids)
        # get image_name_list by unique driver id
        train_img_name_list = []
        valid_img_name_list = []

        for index in train_driver_ids:
            train_img_name_list.extend(driver_img_dict[unique_driver_ids[index]])
        for index in valid_driver_ids:
            valid_img_name_list.extend(driver_img_dict[unique_driver_ids[index]])

        train_dataset = SplitTrainingDataset(driver_img_list, img_path, train_img_name_list, True)
        valid_dataset = SplitTrainingDataset(driver_img_list, img_path, valid_img_name_list)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=PIN_MEMORY)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=PIN_MEMORY)
        
        # transfer learning using pretrained model with conv layer parameters locked
        model = copy.deepcopy(mymodel)
        model = model.to(device)
        optimizer = optim.Adam(model.classifier.parameters(), lr=LR)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)    


        # train model and save result
        train_model(model,
                    len(train_img_name_list),
                    len(valid_img_name_list),
                    train_loader,
                    valid_loader,
                    criterion,
                    optimizer,
                    scheduler,
                    num_fold,
                    num_epochs
                   )

        #if num_fold == 2:
        #    break

if  __name__ == "__main__":
    vgg16bn = models.vgg16_bn(pretrained=True)
    my_vgg16bn = Myvgg(vgg16bn, 'vgg16bn')
    kfold_split_and_train(my_vgg16bn, csv_path, train_img_path)

