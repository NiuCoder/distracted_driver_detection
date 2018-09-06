#! /usr/bin/python3
'''
distracted_driver_detection task with keras implement, without generator because cross_validation
'''
import numpy as np
import os
import glob
import cv2
import pickle
import time
import pandas as pd
import warnings
import sys
import keras
warnings.filterwarnings("ignore")
np.random.seed(2018)
from sklearn.model_selection import KFold
from keras.models import Model
from keras.layers import Input, Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.optimizers import SGD, Adam
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.utils import np_utils
import keras.applications as models
from keras.preprocessing import image
from sklearn.metrics import log_loss

# Global Variables
csv_path = os.path.join('driver_imgs_list.csv')
train_img_path = os.path.join('imgs', 'train')
test_img_path = os.path.join('imgs', 'testroot', 'test')
# Hyperparameters
LR = 1e-5
EPOCHS = 20
BATCHSIZE = 16

# get image data, return a 224x224x3 numpy array, element type uint8 for memory saving
def get_img(path, size=(224,224)):
    img = image.load_img(path, target_size=size)
    img = image.img_to_array(img)
    img = img.astype(dtype=np.uint8)
    return img

# get driver data
def get_driver_data(csv_path=csv_path):
    drivers = dict()
    classes = dict()
    f = open(csv_path,'r')
    line = f.readline()
    while 1:
        line = f.readline()
        if line == '':
            break
        arr = line.strip().split(',')
        drivers[arr[2]] = arr[0]
        if arr[0] not in classes.keys():
            classes[arr[0]] = [(arr[1], arr[2])]
    f.close()
    return drivers, classes

# load train data to array
def load_train_data(size=(224,224)):
    since = time.time()
    print('Start load train data...')
    X_train = []
    y_train = []
    driver_id = []
    driver_data, dr_class = get_driver_data()
    for i in range(10):
        filepath = os.path.join(train_img_path, 'c'+str(i), '*.jpg')
        files = glob.glob(filepath)
        for fl in files:
            flbase = os.path.basename(fl)
            img = get_img(fl, size)
            X_train.append(img)
            y_train.append(i)
            driver_id.append(driver_data[flbase])
    time_elapsed = time.time()-since
    print('load train data finished, total time:{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    unique_drivers = sorted(list(set(driver_id)))
    X_train = np.array(X_train, dtype=np.uint8)
    y_train = np.array(y_train, dtype=np.uint8)
    y_train = np_utils.to_categorical(y_train, 10)

    return X_train, y_train, driver_id, unique_drivers

# load test data and split file
def split_test_files():
    path = os.path.join(test_img_path, '*.jpg')
    files = sorted(glob.glob(path))
    length = len(files)
    split_file_list = [files[i*length // 10 : (i+1)*length // 10] for i in range(10)]
    
    return split_file_list

# load test data to array by part
def load_test_data_part(filelist, part, size=(224, 224)):
    X_test_part = []
    X_test_part_id = []
    
    for fl in filelist[part]:
        flbase = os.path.basename(fl)
        img = get_img(fl, size)
        X_test_part.append(img)
        X_test_part_id.append(flbase)

    return X_test_part, X_test_part_id

def read_test_data(filelist,part,size=(224,224)):
    start_time = time.time()
    test_data,test_id = load_test_data_part(filelist,part,size)
    test_data = np.array(test_data,dtype=np.uint8)
    return test_data,test_id

# cache data
def cache_data(data, path):
    if os.path.isdir(os.path.dirname(path)):
        fl = open(path, 'wb')
        pickle.dump(data, fl)
        fl.close()
    else:
        print('Directory do not exist')

# restore data
def restore_data(path):
    data = dict()
    if os.path.isfile(path):
        fl = open(path, 'rb')
        data = pickle.load(fl)
    return data

#copy split driver data
def copy_selected_drivers(train_data, train_target, driver_id, driver_list):
    index_list = []
    train_data = np.array(train_data, dtype='uint8')
    train_target = np.array(train_target, dtype='uint8')
    for i in range(len(driver_id)):
        if driver_id[i] in driver_list:
            index_list.append(i)
    data = train_data[index_list]
    target = train_target[index_list]
    return data, target

# customize model
def my_model(MODEL, lr=LR):
    input_tensor = Input(shape=(224,224,3))
    base_model = MODEL(input_tensor=input_tensor,weights='imagenet',include_top=False)
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    if base_model.name != 'resnet50':
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.5)(x)
        x = Dense(4096, activation='relu')(x)
        x = Dropout(0.5)(x)
    predictions = Dense(10, activation='softmax')(x)
    model = Model(inputs=base_model.input, outputs=predictions)
    adam = Adam(lr=lr)
    model.compile(optimizer=adam, loss='categorical_crossentropy', metrics=['accuracy'])
    modelname = base_model.name
    return model, modelname

# train model
def train_model(model, modelname, foldnum, X_train, y_train, X_valid, y_valid, epochs, batchsize):
    start_time = time.time()
    weights_path = os.path.join('cache', 'weights_'+modelname+'_'+str(foldnum)+'.h5')
    if not os.path.isfile(weights_path):
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=5, verbose=0),
            ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True, verbose=0),
        ]
        model.fit(X_train, y_train, batch_size=batchsize, nb_epoch=epochs, shuffle=True,
                  verbose=1, validation_data=(X_valid, y_valid),callbacks=callbacks)

    if os.path.isfile(weights_path):
        model.load_weights(weights_path)
    predictions_valid = model.predict(X_valid, batch_size=batchsize, verbose=0)
    loss = round(log_loss(y_valid, predictions_valid),4)
    end_time = time.time()
    train_time = end_time - start_time
    print('Fold:{} train time:{:.2f}m {:.2f}s'.format(foldnum,train_time // 60, train_time % 60))
    return loss

# kfold train
def kfold_train_data(epochs, batchsize, MODEL, preprocess_func,learning_rate,n_folds=10):
    logloss_scores = []
    train_data, train_target, driver_id, unique_drivers = load_train_data()
    kf = KFold(n_splits=n_folds, shuffle=True,random_state=51)
    num_fold = 0
    for train_drivers, valid_drivers in kf.split(unique_drivers):
        num_fold += 1
        print('Start Fold{} split train'.format(num_fold))
        unique_train_drivers = [unique_drivers[i] for i in train_drivers]
        X_train, y_train = copy_selected_drivers(train_data, train_target, driver_id, unique_train_drivers)
        unique_valid_drivers = [unique_drivers[i] for i in valid_drivers]
        X_valid, y_valid = copy_selected_drivers(train_data, train_target, driver_id, unique_valid_drivers)
        X_train = X_train.astype('float16')
        X_valid = X_valid.astype('float16')
        model, modelname = my_model(MODEL, learning_rate)
        X_train = preprocess_func(X_train)
        X_valid = preprocess_func(X_valid)

        score = train_model(model, modelname, num_fold, X_train, y_train, X_valid, y_valid, epochs, batchsize)
        logloss_scores.append(score)

    return logloss_scores

def predict_mean(data,nfolds):
    first_fold = np.array(data[0])
    for i in range(1,nfolds):
        first_fold += np.array(data[i])
    first_fold /= nfolds
    result = np.array(first_fold.tolist())
    return result

def append_chunk(main,part):
    for p in part:
        main.append(p)
    return main

def predict_test_data(MODEL,preprocess_func,nfolds=10):
    model,modelname = my_model(MODEL)
    num_fold = 0
    prediction_list = []
    testid_list = []
    testid_cache_path = os.path.join('cache','testid_list.h5')
    testfile_list = split_test_files()
    for i in range(nfolds):
        num_fold += 1
        weights_path = os.path.join('cache','weights_'+modelname+'_'+str(num_fold)+'.h5')
        if not os.path.isfile(weights_path):
            print(weights_path+' File not exists')
            return []
        start = time.time()
        model.load_weights(weights_path)
        test_prediction_cache_path = os.path.join('cache','test_prediction_'+modelname+'_'+str(num_fold)+'.h5')
        if not os.path.isfile(test_prediction_cache_path):
            prediction = []
            for part in range(10):
                test_data_chunk,test_id_chunk = read_test_data(testfile_list,part)
                test_data_chunk = test_data_chunk.astype('float16')
                test_data_chunk = preprocess_func(test_data_chunk)
                prediction_part = model.predict(test_data_chunk)
                prediction = append_chunk(prediction,prediction_part)
                if i == 0:
                    testid_list = append_chunk(testid_list,test_id_chunk)
                    cache_data(testid_list,testid_cache_path)
            cache_data(prediction,test_prediction_cache_path)
            prediction_list.append(prediction)
        elif not os.path.isfile(testid_cache_path):
            for part in range(100):
                test_data_chunk, test_id_chunk = read_test_data(testfile_list,part)
                if i == 0:
                    testid_list = append_chunk(testid_list,test_id_chunk)
                    cache_data(testid_list,testid_cache_path)
            prediction = restore_data(test_prediction_cache_path)
            prediction_list.append(prediction)
        else:
            prediction = restore_data(test_prediction_cache_path)
            prediction_list.append(prediction)
            if i == 0:
                testid_list = restore_data(testid_cache_path)
        end_time = time.time()
        predict_time = round(end_time-start,2)
    return prediction_list,testid_list

def create_submission(predictions,testid_list,filename,submit_file_path='subm'):
    predictions = predictions.clip(min=1e-15,max=1-1e-15)
    df = pd.DataFrame(np.array(predictions),columns=['c'+str(i) for i in range(10)])
    df.insert(0,'img',testid_list)
    path = os.path.join(submit_file_path,filename)
    df.to_csv(path,index=None)
    print('Create submit file finished')

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('lack of argv')
        sys.exit()
    elif sys.argv[1] == 'train':
        print('Train Start')
        vgg16 = models.vgg16.VGG16 
        scores = kfold_train_data(EPOCHS, BATCHSIZE, vgg16, models.vgg16.preprocess_input, LR)
        print(scores)
    elif sys.argv[1] == 'test':
        print('Test Start')
        vgg16 = models.vgg16.VGG16(weights='imagenet', include_top=False)
        prediction_list,testid_list = predict_test_data(vgg16,models.vgg16.preprocess_input,3)
        create_submission(predict_mean(prediction_list,3),testid_list,'vgg16-3fold-avg.csv')
    else:
        print('Please input argv with either train or test') 
