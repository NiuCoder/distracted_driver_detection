#! /usr/bin/python3
#coding:utf-8
from sklearn.model_selection import KFold
import numpy as np
import pandas as pd
import math
import sys
import time
import os
from utils import generate_train_data, batch_generator, batch_generator_triplet, test_batch_generator, triplet_loss, identity_loss, generate_train_data_triplet
from keras.applications.inception_v3 import InceptionV3
from keras.applications.vgg16 import VGG16
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
from keras.layers import Input, Dense, Lambda, Dropout, Conv2D, BatchNormalization, AveragePooling2D, Flatten, Add, Activation

from keras.models import Model
from keras.models import load_model
from keras.optimizers import SGD, Adam
from keras.utils import multi_gpu_model
import glob
from keras import regularizers
np.random.seed(2019)


class DriverDetection:

    def __init__(self, csv_path='driver_imgs_list.csv', img_root_path='imgs'):
        self.csv_path = csv_path
        self.img_root_path = img_root_path
        self.driver_img_list = pd.read_csv(csv_path)
        self.learning_rate = 0.0001
        self.nbr_gpus = 1
        self.batch_size = 64
        self.nbr_classes = 10
        self.nbr_epoch = 20

    def random_split_driver_ids(self, percent=0.2):
        driver_ids = self.driver_img_list['subject'].unique().tolist()
        val_num = math.floor(len(driver_ids) * percent)
        valid_driver_ids = np.random.choice(driver_ids, val_num, replace=False)
        train_driver_ids = np.setdiff1d(driver_ids, valid_driver_ids)
        return train_driver_ids, valid_driver_ids

    def kfold_split_driver_ids(self, n_folds=10):
        driver_ids = self.driver_img_list['subject'].unique().tolist()
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=51)
        train_driver_ids = []
        valid_driver_ids = []
        for train_driver_index, valid_driver_index in kf.split(driver_ids):
            train_drivers = [driver_ids[i] for i in train_driver_index]
            valid_drivers = [driver_ids[i] for i in valid_driver_index]
            train_driver_ids.append(train_drivers)
            valid_driver_ids.append(valid_drivers)
        return train_driver_ids, valid_driver_ids

    def get_dataframe_by_driver_ids(self, driver_ids):
        dataframe = pd.DataFrame({}, columns=['subject', 'classname', 'img'])
        for driver_id in driver_ids:
            frame = self.driver_img_list[self.driver_img_list['subject'] == driver_id]
            dataframe = dataframe.append(frame)
            dataframe = dataframe.reset_index(drop=True)
        return dataframe

    def train_with_vgg16_no_triplet(self, train_frame, valid_frame, fold_num=None, fine_tune=False, model_path=None):
        start_time = time.time()
        if fine_tune and model_path:
            print('Finetune and Loading vgg16 weight ...')
            model = load_model(model_path)
        else:
            # transfer learning with vgg16
            input_tensor = Input(shape=(224, 224, 3))
            vgg16 = VGG16(include_top=False, input_tensor=input_tensor,
                          weights='imagenet', pooling='avg')
            vgg16.get_layer(index=-1).name = 'globalavg'
            x = vgg16.output
            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.5)(x)
            x = Dense(1024, activation='relu', name='features')(x)
            x = Dropout(0.5)(x)
            predictions = Dense(10, activation='softmax',
                                name='predictions')(x)
            model = Model(inputs=vgg16.input, outputs=predictions)

        if self.nbr_gpus > 1:
            print('Using multiple GPUS: {}\n'.format(self.nbr_gpus))
            model = multi_gpu_model(model, gpus=self.nbr_gpus)
            self.batch_size *= self.nbr_gpus
        else:
             print('Using a single GPU.\n')

        optimizer = Adam(lr=self.learning_rate)
#         optimizer = SGD(lr = self.learning_rate, momentum = 0.9, decay = 0.0, nesterov = True)
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy', metrics=['accuracy'])

        if not fold_num:
            best_model_path = './models/vgg16_no_triplet_random.h5'
        else:
            best_model_path = './models/vgg16_no_triplet_kfold_'+fold_num+'.h5'
        check_point = ModelCheckpoint(
            best_model_path, monitor='val_loss', verbose=1, save_best_only=True)
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.1, patience=3, verbose=1, min_lr=0.0000001)
        early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

        print('# Train Images: {}.'.format(len(train_frame)))
        steps_per_epoch = int(
            math.ceil(len(train_frame) * 1. / self.batch_size))
        print('# Val Images: {}.'.format(len(valid_frame)))
        validation_steps = int(
            math.ceil(len(valid_frame) * 1. / self.batch_size))

        history = model.fit_generator(batch_generator(train_frame, self.nbr_classes, self.img_root_path, self.batch_size),
                                      steps_per_epoch=steps_per_epoch, epochs=self.nbr_epoch, verbose=1,
                                      validation_data=batch_generator(
            valid_frame, self.nbr_classes, self.img_root_path, self.batch_size, shuffle=False),
            validation_steps=validation_steps, callbacks=[
                check_point, reduce_lr, early_stop],
            max_queue_size=10, workers=8, use_multiprocessing=True)
        best_val_loss = min(history.history['val_loss'])

        end_time = time.time()
        train_time = end_time - start_time
        if not fold_num:
            print('Best val loss:{}'.format(best_val_loss))
            print('random split Train time:{:.2f}m {:.2f}s'.format(
                train_time // 60, train_time % 60))
        else:
            print('Fold:{} Best val loss:{}'.format(fold_num, best_val_loss))
            print('Fold:{} train time:{:.2f}m {:.2f}s'.format(
                fold_num, train_time // 60, train_time % 60))

    def train_with_inception_no_triplet(self, train_frame, valid_frame, fold_num=None, fine_tune=False, model_path=None):
        start_time = time.time()
        if fine_tune and model_path:
            print('Finetune and Loading vgg16 weight ...')
            model = load_model(model_path)
        else:
            # transfer learning with vgg16
            input_tensor = Input(shape=(224, 224, 3))
            basemodel = InceptionV3(include_top=False, input_tensor=input_tensor,
                          weights='imagenet', pooling='avg')
            basemodel.get_layer(index=-1).name = 'globalavg'
            x = basemodel.output
            # x = Dense(1024, activation='relu')(x)
            # x = Dropout(0.5)(x)
            x = Dense(1024, activation='relu', name='features')(x)
            # x = Dropout(0.5)(x)
            predictions = Dense(10, activation='softmax',
                                name='predictions')(x)
            model = Model(inputs=basemodel.input, outputs=predictions)

        if self.nbr_gpus > 1:
            print('Using multiple GPUS: {}\n'.format(self.nbr_gpus))
            model = multi_gpu_model(model, gpus=self.nbr_gpus)
            self.batch_size *= self.nbr_gpus
        else:
             print('Using a single GPU.\n')

        optimizer = Adam(lr=self.learning_rate)
#         optimizer = SGD(lr = self.learning_rate, momentum = 0.9, decay = 0.0, nesterov = True)
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy', metrics=['accuracy'])

        if not fold_num:
            best_model_path = './models/inceptv3_no_triplet_random.h5'
        else:
            best_model_path = './models/inceptv3_no_triplet_kfold_'+fold_num+'.h5'
        check_point = ModelCheckpoint(
            best_model_path, monitor='val_loss', verbose=1, save_best_only=True)
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.1, patience=3, verbose=1, min_lr=0.0000001)
        early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

        print('# Train Images: {}.'.format(len(train_frame)))
        steps_per_epoch = int(
            math.ceil(len(train_frame) * 1. / self.batch_size))
        print('# Val Images: {}.'.format(len(valid_frame)))
        validation_steps = int(
            math.ceil(len(valid_frame) * 1. / self.batch_size))

        history = model.fit_generator(batch_generator(train_frame, self.nbr_classes, self.img_root_path, self.batch_size, False, True, 224, 224, 'inceptv3'),
                                      steps_per_epoch=steps_per_epoch, epochs=self.nbr_epoch, verbose=1,
                                      validation_data=batch_generator(
            valid_frame, self.nbr_classes, self.img_root_path, self.batch_size, False, False, True, 224, 224, 'inceptv3'),
            validation_steps=validation_steps, callbacks=[
                check_point, reduce_lr, early_stop],
            max_queue_size=10, workers=8, use_multiprocessing=True)
        best_val_loss = min(history.history['val_loss'])

        end_time = time.time()
        train_time = end_time - start_time
        if not fold_num:
            print('Best val loss:{}'.format(best_val_loss))
            print('random split Train time:{:.2f}m {:.2f}s'.format(
                train_time // 60, train_time % 60))
        else:
            print('Fold:{} Best val loss:{}'.format(fold_num, best_val_loss))
            print('Fold:{} train time:{:.2f}m {:.2f}s'.format(
                fold_num, train_time // 60, train_time % 60))

    def train_with_vgg16_no_triplet_no_generator(self, train_frame, valid_frame, fold_num=None, fine_tune=False, model_path=None):
        start_time = time.time()
        if fine_tune and model_path:
            print('Finetune and Loading vgg16 weight ...')
            model = load_model(model_path)
        else:
            # transfer learning with vgg16
            input_tensor = Input(shape=(224, 224, 3))
            vgg16 = VGG16(include_top=False, input_tensor=input_tensor,
                          weights='imagenet', pooling='avg')
            vgg16.get_layer(index=-1).name = 'globalavg'
            x = vgg16.output
            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.5)(x)
            x = Dense(1024, activation='relu', name='features')(x)
            x = Dropout(0.5)(x)
            # x = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(0.01))(x)
            predictions = Dense(10, activation='softmax',
                                name='predictions')(x)
            model = Model(inputs=vgg16.input, outputs=predictions)

        if self.nbr_gpus > 1:
            print('Using multiple GPUS: {}\n'.format(self.nbr_gpus))
            model = multi_gpu_model(model, gpus=self.nbr_gpus)
            self.batch_size *= self.nbr_gpus
        else:
             print('Using a single GPU.\n')

        optimizer = Adam(lr=self.learning_rate)
        # optimizer = SGD(lr=self.learning_rate, momentum=0.9,
        #                 decay=0.0, nesterov=True)
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy', metrics=['accuracy'])

        if not fold_num:
            best_model_path = './models/vgg16_no_triplet_random.h5'
        else:
            best_model_path = './models/vgg16_no_triplet_kfold_'+fold_num+'.h5'
        check_point = ModelCheckpoint(
            best_model_path, monitor='val_loss', verbose=1, save_best_only=True)
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.1, patience=3, verbose=1, min_lr=0.0000001)
        early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

        print('# Train Images: {}.'.format(len(train_frame)))
        print('# Val Images: {}.'.format(len(valid_frame)))
        print('Start load train and valid image into numpy array and preprocess ...')
        X_train, Y_train = generate_train_data(train_frame, 10, self.img_root_path, shuffle=False,
                                               augment=False, img_width=224, img_height=224, model='vgg16')
        X_valid, Y_valid = generate_train_data(valid_frame, 10, self.img_root_path, shuffle=False,
                                               augment=False, img_width=224, img_height=224, model='vgg16')

        history = model.fit(X_train, Y_train, batch_size=self.batch_size, nb_epoch=self.nbr_epoch, shuffle=True,
                            verbose=1, validation_data=(X_valid, Y_valid), callbacks=[check_point, reduce_lr, early_stop])
        best_val_loss = min(history.history['val_loss'])

        end_time = time.time()
        train_time = end_time - start_time
        if not fold_num:
            print('Best val loss:{}'.format(best_val_loss))
            print('random split Train time:{:.2f}m {:.2f}s'.format(
                train_time // 60, train_time % 60))
        else:
            print('Fold:{} Best val loss:{}'.format(fold_num, best_val_loss))
            print('Fold:{} train time:{:.2f}m {:.2f}s'.format(
                fold_num, train_time // 60, train_time % 60))

    def train_with_vgg16_with_triplet(self, train_frame, valid_frame, fold_num=None, fine_tune=False, fine_tune_on_attr=False, model_path=None):
        start_time = time.time()
        if fine_tune and model_path:
            print('Finetune on the triplet model ...')
            model = load_model(model_path)
        elif fine_tune_on_attr and model_path:
            print('Finetune on the attrbute model ...')
            basemodel = load_model(model_path)
            f_prediction = basemodel.get_layer(name='predictions').output
            f_feature = basemodel.get_layer(name='features').output

            anchor = basemodel.input
            positive = Input(shape=(224, 224, 3), name='positive')
            negative = Input(shape=(224, 224, 3), name='negative')
            f_branch = Model(inputs=basemodel.input, outputs=f_feature)
            f_sls_anchor = f_branch(anchor)
            f_sls_positive = f_branch(positive)
            f_sls_negative = f_branch(negative)

            loss = Lambda(triplet_loss, output_shape=(1,))(
                [f_sls_anchor, f_sls_positive, f_sls_negative])
            model = Model(inputs=[anchor, positive, negative], outputs=[
                          f_prediction, loss])
        else:
            print('Loading vgg16 weights from ImageNet Pretrained ...')
            input_tensor = Input(shape=(224, 224, 3))
            vgg16 = VGG16(include_top=False, input_tensor=input_tensor,
                          weights='imagenet', pooling='avg')
            f_base = vgg16.output
            anchor = vgg16.input
            positive = Input(shape=(224, 224, 3), name='positive')
            negative = Input(shape=(224, 224, 3), name='negative')
            f_feature = Dense(1024, activation='relu', kernel_regularizer=regularizers.l2(
                0.01), name='features')(f_base)
            f_prediction = Dense(10, activation='softmax',
                                 name='predictions')(f_feature)
            f_branch = Model(inputs=vgg16.input, outputs=f_feature)
            f_sls_anchor = f_branch(anchor)
            f_sls_positive = f_branch(positive)
            f_sls_negative = f_branch(negative)

            loss = Lambda(triplet_loss, output_shape=(1,))(
                [f_sls_anchor, f_sls_positive, f_sls_negative])
            model = Model(inputs=[anchor, positive, negative], outputs=[
                          f_prediction, loss])

        if self.nbr_gpus > 1:
            print('Using multiple GPUS: {}\n'.format(self.nbr_gpus))
            model = multi_gpu_model(model, gpus=self.nbr_gpus)
            self.batch_size *= self.nbr_gpus
        else:
             print('Using a single GPU.\n')

        optimizer = Adam(lr=self.learning_rate)
        # optimizer = SGD(lr=self.learning_rate, momentum=0.9,
        #                 decay=0.0, nesterov=True)
        model.compile(optimizer=optimizer,
                      loss=['categorical_crossentropy', identity_loss], metrics=['accuracy'])

        if not fold_num:
            best_model_path = './models/vgg16_with_triplet_random.h5'
        else:
            best_model_path = './models/vgg16_with_triplet_kfold_'+fold_num+'.h5'
        check_point = ModelCheckpoint(
            best_model_path, monitor='val_loss', verbose=1, save_best_only=True)
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.1, patience=3, verbose=1, min_lr=0.00001)
        early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

        # print('# Train Images: {}.'.format(len(train_frame)))
        # steps_per_epoch = int(
        #     math.ceil(len(train_frame) * 1. / self.batch_size))
        # print('# Val Images: {}.'.format(len(valid_frame)))
        # validation_steps = int(
        #     math.ceil(len(valid_frame) * 1. / self.batch_size))

        # history = model.fit_generator(batch_generator_triplet(train_frame, self.nbr_classes, self.img_root_path, self.batch_size, 'train'),
        #                               steps_per_epoch=steps_per_epoch, epochs=self.nbr_epoch, verbose=1,
        #                               validation_data=batch_generator_triplet(
        #     valid_frame, self.nbr_classes, self.img_root_path, self.batch_size, 'val'),
        #     validation_steps=validation_steps, callbacks=[
        #         check_point, reduce_lr, early_stop],
        #     max_queue_size=10, workers=8, use_multiprocessing=True)
        print('# Train Images triples: {}.'.format(len(train_frame)))
        print('# Val Images triples: {}.'.format(len(valid_frame)))
        print('Start load train and valid image into numpy array and preprocess ...')
        X_train, Y_train = generate_train_data_triplet(train_frame, 10, self.img_root_path, shuffle=False,mode='train',
                                               augment=False, img_width=224, img_height=224, model='vgg16')
        X_valid, Y_valid = generate_train_data_triplet(valid_frame, 10, self.img_root_path, shuffle=False,mode='valid',
                                               augment=False, img_width=224, img_height=224, model='vgg16')

        history = model.fit(X_train, Y_train, batch_size=self.batch_size, nb_epoch=self.nbr_epoch, shuffle=True,
                            verbose=1, validation_data=(X_valid, Y_valid), callbacks=[check_point, reduce_lr, early_stop])
        best_val_loss = min(history.history['val_loss'])

        end_time = time.time()
        train_time = end_time - start_time
        if not fold_num:
            print('Best val loss:{}'.format(best_val_loss))
            print('random split Train time:{:.2f}m {:.2f}s'.format(
                train_time // 60, train_time % 60))
        else:
            print('Fold:{} Best val loss:{}'.format(fold_num, best_val_loss))
            print('Fold:{} train time:{:.2f}m {:.2f}s'.format(
                fold_num, train_time // 60, train_time % 60))

    def main_block(self, x, filters, n, strides, dropout):
        # Normal part
        x_res = Conv2D(filters, (3, 3), strides=strides, padding="same")(
            x)  # , kernel_regularizer=l2(5e-4)
        x_res = BatchNormalization()(x_res)
        x_res = Activation('relu')(x_res)
        x_res = Conv2D(filters, (3, 3), padding="same")(x_res)
        # Alternative branch
        x = Conv2D(filters, (1, 1), strides=strides)(x)
        # Merge Branches
        x = Add()([x_res, x])

        for i in range(n-1):
            # Residual conection
            x_res = BatchNormalization()(x)
            x_res = Activation('relu')(x_res)
            x_res = Conv2D(filters, (3, 3), padding="same")(x_res)
            # Apply dropout if given
            if dropout:
                x_res = Dropout(dropout)(x)
            # Second part
            x_res = BatchNormalization()(x_res)
            x_res = Activation('relu')(x_res)
            x_res = Conv2D(filters, (3, 3), padding="same")(x_res)
            # Merge branches
            x = Add()([x, x_res])

        # Inter block part
        x = BatchNormalization()(x)
        x = Activation('relu')(x)
        return x


    def build_model(self, input_dims, output_dim, n, k, act="relu", dropout=None):
        """ Builds the model. Params:
                - n: number of layers. WRNs are of the form WRN-N-K
                    It must satisfy that (N-4)%6 = 0
                - k: Widening factor. WRNs are of the form WRN-N-K
                    It must satisfy that K%2 = 0
                - input_dims: input dimensions for the model
                - output_dim: output dimensions for the model
                - dropout: dropout rate - default=0 (not recomended >0.3)
                - act: activation function - default=relu. Build your custom
                    one with keras.backend (ex: swish, e-swish)
        """
        # Ensure n & k are correct
        assert (n-4) % 6 == 0
        assert k % 2 == 0
        n = (n-4)//6
        # This returns a tensor input to the model
        inputs = Input(shape=(input_dims))

        # Head of the model
        x = Conv2D(16, (3, 3), padding="same")(inputs)
        x = BatchNormalization()(x)
        x = Activation('relu')(x)

        # 3 Blocks (normal-residual)
        x = self.main_block(x, 16*k, n, (1, 1), dropout)  # 0
        x = self.main_block(x, 32*k, n, (2, 2), dropout)  # 1
        x = self.main_block(x, 64*k, n, (2, 2), dropout)  # 2

        # Final part of the model
        x = AveragePooling2D((8, 8))(x)
        x = Flatten()(x)
        outputs = Dense(output_dim, activation="softmax")(x)

        model = Model(inputs=inputs, outputs=outputs)
        return model

    def train_with_customermized_model(self, train_frame, valid_frame, fold_num=None, fine_tune=False, model_path=None):
        start_time = time.time()
        if fine_tune and model_path:
            print('Finetune and Loading vgg16 weight ...')
            model = load_model(model_path)
        else:
            model = self.build_model((224,224,3), 10, 16, 4)

        if self.nbr_gpus > 1:
            print('Using multiple GPUS: {}\n'.format(self.nbr_gpus))
            model = multi_gpu_model(model, gpus=self.nbr_gpus)
            self.batch_size *= self.nbr_gpus
        else:
             print('Using a single GPU.\n')

        optimizer = Adam(lr=self.learning_rate)
#         optimizer = SGD(lr = self.learning_rate, momentum = 0.9, decay = 0.0, nesterov = True)
        model.compile(optimizer=optimizer,
                      loss='categorical_crossentropy', metrics=['accuracy'])

        if not fold_num:
            best_model_path = './models/vgg16_no_triplet_random.h5'
        else:
            best_model_path = './models/vgg16_no_triplet_kfold_'+fold_num+'.h5'
        check_point = ModelCheckpoint(
            best_model_path, monitor='val_loss', verbose=1, save_best_only=True)
        reduce_lr = ReduceLROnPlateau(
            monitor='val_loss', factor=0.1, patience=3, verbose=1, min_lr=0.0000001)
        early_stop = EarlyStopping(monitor='val_loss', patience=5, verbose=1)

        print('# Train Images: {}.'.format(len(train_frame)))
        steps_per_epoch = int(
            math.ceil(len(train_frame) * 1. / self.batch_size))
        print('# Val Images: {}.'.format(len(valid_frame)))
        validation_steps = int(
            math.ceil(len(valid_frame) * 1. / self.batch_size))

        history = model.fit_generator(batch_generator(train_frame, self.nbr_classes, self.img_root_path, self.batch_size),
                                      steps_per_epoch=steps_per_epoch, epochs=self.nbr_epoch, verbose=1,
                                      validation_data=batch_generator(
            valid_frame, self.nbr_classes, self.img_root_path, self.batch_size, shuffle=False),
            validation_steps=validation_steps, callbacks=[
                check_point, reduce_lr, early_stop],
            max_queue_size=10, workers=8, use_multiprocessing=True)
        best_val_loss = min(history.history['val_loss'])

        end_time = time.time()
        train_time = end_time - start_time
        if not fold_num:
            print('Best val loss:{}'.format(best_val_loss))
            print('random split Train time:{:.2f}m {:.2f}s'.format(
                train_time // 60, train_time % 60))
        else:
            print('Fold:{} Best val loss:{}'.format(fold_num, best_val_loss))
            print('Fold:{} train time:{:.2f}m {:.2f}s'.format(
                fold_num, train_time // 60, train_time % 60))

    def predict_with_vgg16(self, test_img_path, model_path):
        test_files = sorted(glob.glob(test_img_path))
        steps = int(math.ceil(len(test_files) * 1. / self.batch_size))
        model = load_model(model_path)
        predictions = model.predict_generator(test_batch_generator(test_files, self.batch_size), steps,
                                              max_queue_size=80, workers=8, use_multiprocessing=True)
        test_ids = [os.path.split(ele)[-1] for ele in test_files]
        return predictions, test_ids

    def create_submission(self, predictions, testid_list, filename, submit_file_path='subm'):
        predictions = predictions.clip(min=1e-15, max=1-1e-15)
        df = pd.DataFrame(np.array(predictions), columns=[
                          'c'+str(i) for i in range(10)])
        df.insert(0, 'img', testid_list)
        path = os.path.join(submit_file_path, filename)
        df.to_csv(path, index=None)
        print('Create submit file finished')


if __name__ == "__main__":
    s = DriverDetection()
    print(s.csv_path)
    # random split driver id
    if sys.argv[1] == 'random':
        print('Start train with random driver id split ...')
        train_driver_ids, valid_driver_ids = s.random_split_driver_ids()
        train_frame = s.get_dataframe_by_driver_ids(train_driver_ids)
        valid_frame = s.get_dataframe_by_driver_ids(valid_driver_ids)
        s.train_with_vgg16_no_triplet(train_frame, valid_frame)
    # kfold split driver id
    elif sys.argv[1] == 'kfold':
        print('Start train with kfold driver id split ...')
        train_driver_ids, valid_driver_ids = s.kfold_split_driver_ids()
        for i in range(len(train_driver_ids)):
            print('Start train foldnum:{}'.format(i))
            train_frame = s.get_dataframe_by_driver_ids(train_driver_ids[i])
            valid_frame = s.get_dataframe_by_driver_ids(valid_driver_ids[i])
            s.train_with_vgg16_no_triplet(train_frame, valid_frame, i)
