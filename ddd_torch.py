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
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import os
import pickle
import time
import sys
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
N_EPOCHS = 10
BATCH_SIZE = 64
WORKERS = 0
LR = 1e-5
DROP_RATE = 0.5
PIN_MEMORY = use_cuda
criterion = nn.CrossEntropyLoss()

# get unique_drivers and driver_img_dict from csv file
def get_driver_img_map(csv_path):
    driver_img_list = pd.read_csv(csv_path)
    unique_drivers = list(set(driver_img_list['subject']))
    unique_drivers.sort()
    driver_img_dict = {}
    for driver_id in unique_drivers:
        img_list = driver_img_list[driver_img_list['subject']==driver_id]['img'].values.tolist()
        if driver_id in driver_img_dict:
            driver_img_dict[driver_id].extend(img_list)
        else:
            driver_img_dict[driver_id] = img_list

    return unique_drivers, driver_img_dict, driver_img_list


# transform function for valid and test data
def transform_data_normal(target_size=299):
    return transforms.Compose([
        transforms.Resize(target_size),
        transforms.CenterCrop(target_size),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])
    ])

# transform function for train data with augmentation, in this case keep aug same as normal
def transform_data_aug(target_size=299):
    return transforms.Compose([
        #transforms.RandomResizedCrop(target_size),
        transforms.Resize(target_size),
        transforms.CenterCrop(target_size),
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
        file = open(path, 'rb')
        data = pickle.load(file)
    return data

# predict mean for esemble
def predict_mean(data, nfolds):
    first_fold = np.array(data[0])
    for i in range(1, nfolds):
        first_fold += np.array(data[i])
    first_fold /= nfolds
    return first_fold.tolist()

def append_chunk(main, part):
    for p in part:
        main.append(p.tolist())
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
        classnames = list(set(driver_img_list['classname']))
        classnames.sort()
        self.class_names = classnames
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
        #label = np.zeros(len(self.class_names))
        class_index = self.class_names.index(class_name)
        #label[class_index] = 1
        label = class_index
        return img,label

# Customize pretrained model 
class Myvgg(nn.Module):
    def __init__(self, basemodel, modelname, num_classes=10):
        super(Myvgg, self).__init__()
        self.features = basemodel.features
        self.modelname = modelname
        #for param in basemodel.parameters():
        #    param.requires_grad = False
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
def train_model(model, modelname, datasize_t, datasize_v, loader_t, loader_v, criterion, optimizer, scheduler, num_fold, num_epochs):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    best_loss = float("inf")
    early_stop_count = 0

    for epoch in range(num_epochs):
        #if early_stop_count == 7:
        #    print('model did not make progress in last 7 epochs, early stop...')
        #    break

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
                        outputs, aux = model(inputs)
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
                if epoch_loss < best_loss:
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
    print('Fold{}: Best val loss : {:.4f}'.format(num_fold, best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    # save model for esemble
    torch.save(model.state_dict(), os.path.join(cache_path, modelname+'_'+str(num_fold)+'.pth'))
    
    return model,best_loss

# esemble model with kfold data split
def kfold_split_and_train(mymodel, modelname, csv_path, img_path, k=10, num_epochs = N_EPOCHS):    
    loss_scores = []
    unique_driver_ids, driver_img_dict, driver_img_list = get_driver_img_map(csv_path)
    kf = KFold(n_splits=k, shuffle=True, random_state=51)
    num_fold = 0
    for train_driver_ids, valid_driver_ids in kf.split(unique_driver_ids):
        num_fold += 1
        if num_fold > 1:
            break
        print('Fold{} data split:'.format(num_fold))
        print(train_driver_ids)
        print(valid_driver_ids)
        # get image_name_list by unique driver id
        train_img_name_list = []
        valid_img_name_list = []

        for index in train_driver_ids:
            train_img_name_list.extend(driver_img_dict[unique_driver_ids[index]])
        for index in valid_driver_ids:
            valid_img_name_list.extend(driver_img_dict[unique_driver_ids[index]])

        train_img_size = len(train_img_name_list)
        valid_img_size = len(valid_img_name_list)
        print('Fold{} train data size:{},valid data size:{}'.format(num_fold,train_img_size,valid_img_size))
        train_dataset = SplitTrainingDataset(driver_img_list, img_path, train_img_name_list, True)
        valid_dataset = SplitTrainingDataset(driver_img_list, img_path, valid_img_name_list)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=PIN_MEMORY)
        valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=WORKERS, pin_memory=PIN_MEMORY)
        
        # transfer learning using pretrained model with conv layer parameters locked
        model = copy.deepcopy(mymodel)
        model = model.to(device)
        #optimizer = optim.Adam(model.classifier.parameters(), lr=LR)
        optimizer = optim.Adam(model.parameters(), lr=LR)
        scheduler = lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)    

        # train model and save result
        train_model(model,
                    modelname,
                    train_img_size,
                    valid_img_size,
                    train_loader,
                    valid_loader,
                    criterion,
                    optimizer,
                    scheduler,
                    num_fold,
                    num_epochs
                   )

# kfold predict
def kfold_predict_data(mymodel, modelname, test_img_dir, weight_path, k=10):
    test_dataset = datasets.ImageFolder(test_img_dir, transform_data_normal())
    test_img_name_list = [os.path.basename(img) for img,_ in test_dataset.imgs]
    test_img_length = len(test_img_name_list)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=WORKERS, pin_memory=PIN_MEMORY)
    num_fold = 0
    prediction_list = []
    for i in range(k):
        num_fold += 1
        start_time = time.time()
        print('Strat Fold{} test prediction'.format(num_fold))
        weight_file_path = os.path.join(weight_path, modelname+'_'+str(num_fold)+'.pth')
        predictions = []
        cache_file = 'test_prediction_'+modelname+'_'+str(num_fold)+'.h5'
        test_prediction_cache_path = os.path.join('cache', cache_file)
        if not os.path.isfile(test_prediction_cache_path):
            mymodel.load_state_dict(torch.load(weight_file_path))
            mymodel = mymodel.to(device)
            with torch.no_grad():
                for inputs,_ in test_loader:
                    inputs = inputs.to(device)
                    outputs = F.softmax(mymodel(inputs),dim=1)
                    outputs = outputs.cpu().numpy()
                    predictions = append_chunk(predictions, outputs)
            cache_data(predictions, test_prediction_cache_path)
            print('{} generated'.format(cache_file))
        else:
            print('{} already exists'.format(cache_file))
            predictions = restore_data(test_prediction_cache_path)
        elapse_time = time.time() - start_time
        print('Finish Fold {} test prediction, time cost:{}m {}s'.format(num_fold, elapse_time//60, elapse_time%60))
        prediction_list.append(predictions)
        # mean result
    avg_result = predict_mean(prediction_list, k)
    avg_result = np.array(avg_result)

    return prediction_list, test_img_name_list, avg_result

if  __name__ == "__main__":
    if len(sys.argv) < 2:
        print('lack of argv')
        sys.exit()
    if sys.argv[1] == 'train':
        print('train process')

        # Train process
        #vgg16bn = models.vgg16_bn(pretrained=True)
        #my_vgg16bn = Myvgg(vgg16bn, 'vgg16bn')
        inception_v3 = models.inception_v3(pretrained=True)
        inception_v3.fc = nn.Linear(2048, 10)
        kfold_split_and_train(inception_v3, 'inception_v3', csv_path, train_img_path)
    
    # Test process
    elif sys.argv[1] == 'test':
        print('test process')
        inception_v3 = models.inception_v3(pretrained=False)
        inception_v3.fc = nn.Linear(2048,10)
        inception_v3.eval()
        prediction_list, test_img_name_list, avg_result = kfold_predict_data(inception_v3, 'inception_v3', test_img_path, cache_path,1)
        create_submission(avg_result, test_img_name_list, 'inception_v3_full-net_1e-5-1fold-part.csv', result_path)
    else:
        print('only train and test work')
