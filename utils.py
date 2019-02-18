
from sklearn.utils import shuffle as sk_shuffle
import numpy as np
import cv2
# from imgaug import augmenters as iaa
# import imgaug as ia
from keras.applications.vgg16 import preprocess_input as vgg16_preprocess_input
from keras.applications.inception_v3 import preprocess_input as incept3_preprocess_input
import random
import os
import keras.backend as K


# Data Agumentation: https://github.com/aleju/imgaug

"""
# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
st = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.
seq = iaa.Sequential([
        iaa.Fliplr(0.5), # horizontally flip 50% of all images
        #iaa.Flipud(0.5), # vertically flip 50% of all images
        st(iaa.Crop(percent=(0, 0.15))), # crop images by 0-15% of their height/width
        #st(iaa.GaussianBlur((0, 2.0))), # blur images with a sigma between 0 and 3.0
        st(iaa.Add((-10, 10), per_channel=0.5)), # change brightness of images (by -10 to 10 of original value)
        st(iaa.Multiply((0.85, 1.15), per_channel=0.5)), # change brightness of images (75-125% of original value)
        st(iaa.ContrastNormalization((0.9, 1.1), per_channel=0.5)), # improve or worsen the contrast
        st(iaa.Affine(
            scale={"x": (0.9, 1.1), "y": (0.9, 1.1)}, # scale images to 80-120% of their size, individually per axis
            #translate_px={"x": (-10, 10), "y": (-10, 10)}, # translate by -16 to +16 pixels (per axis)
            rotate=(-15, 15), # rotate by -10 to +10 degrees
            #shear=(-5, 5), # shear by -16 to +16 degrees
            order=ia.ALL, # use any of scikit-image's interpolation methods
            mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
        ))
    ],
    random_order=True # do all of the above in random order
)
"""

# Sometimes(0.5, ...) applies the given augmenter in 50% of all cases,
# e.g. Sometimes(0.5, GaussianBlur(0.3)) would blur roughly every second image.
# sometimes = lambda aug: iaa.Sometimes(0.5, aug)

# Define our sequence of augmentation steps that will be applied to every image
# All augmenters with per_channel=0.5 will sample one value _per image_
# in 50% of all cases. In all other cases they will sample new values
# _per channel_.
# seq = iaa.Sequential(
#     [
#         # apply the following augmenters to most images
#         iaa.Fliplr(0.5), # horizontally flip 50% of all images
#         # crop images by -5% to 10% of their height/width
#         sometimes(iaa.CropAndPad(
#             percent=(-0.05, 0.1),
#             pad_mode=ia.ALL,
#             pad_cval=(0, 255)
#         )),
#         sometimes(iaa.Affine(
#             scale={"x": (0.85, 1.15), "y": (0.85, 1.5)}, # scale images to 80-120% of their size, individually per axis
#             translate_percent={"x": (-0.15, 0.15), "y": (-0.15, 0.15)}, # translate by -20 to +20 percent (per axis)
#             rotate=(-15, 15), # rotate by -45 to +45 degrees
#             shear=(-5, 5), # shear by -16 to +16 degrees
#             order=[0, 1], # use nearest neighbour or bilinear interpolation (fast)
#             cval=(0, 255), # if mode is constant, use a cval between 0 and 255
#             mode=ia.ALL # use any of scikit-image's warping modes (see 2nd image from the top for examples)
#         )),
#         # execute 0 to 5 of the following (less important) augmenters per image
#         # don't execute all of them, as that would often be way too strong
#         iaa.SomeOf((0, 3),
#             [
#                 iaa.OneOf([
#                     iaa.GaussianBlur((0, 2.0)), # blur images with a sigma between 0 and 3.0
#                     iaa.AverageBlur(k=(1, 5)), # blur image using local means with kernel sizes between 2 and 7
#                     iaa.MedianBlur(k=(1, 5)), # blur image using local medians with kernel sizes between 2 and 7
#                 ]),

#                 iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.03*255), per_channel=0.5), # add gaussian noise to images
#                 iaa.OneOf([
#                     iaa.Dropout((0.01, 0.1), per_channel=0.5), # randomly remove up to 10% of the pixels
#                     iaa.CoarseDropout((0.03, 0.15), size_percent=(0.01, 0.03), per_channel=0.2),
#                 ]),
#                 iaa.Add((-10, 10), per_channel=0.5), # change brightness of images (by -10 to 10 of original value)

#                 iaa.ContrastNormalization((0.3, 1.0), per_channel=0.5), # improve or worsen the contrast
#                 iaa.Grayscale(alpha=(0.0, 1.0)),
#                 sometimes(iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25)), # move pixels locally around (with random strengths)
#                 sometimes(iaa.PiecewiseAffine(scale=(0.01, 0.05))), # sometimes move parts of the image around
#             ],
#             random_order=True
#         )
#     ],
#     random_order=True
# )

def center_crop(x, center_crop_size):
        centerw, centerh = x.shape[0] // 2, x.shape[1] // 2
        halfw, halfh = center_crop_size[0] // 2, center_crop_size[1] // 2
        cropped = x[centerw - halfw: centerw + halfw,
                centerh - halfh: centerh + halfh, :]

        return cropped

def load_img(img_path, return_width=224, crop_method=center_crop):
    # Given an image path, return a scaled array
    img = cv2.imread(img_path)
    h = img.shape[0]
    w = img.shape[1]
    shorter = min(w, h)
    longer = max(w, h)
    img_cropped = crop_method(img, (shorter, shorter))
    img_resized = cv2.resize(
        img_cropped, (return_width, return_width), interpolation=cv2.INTER_CUBIC)
    img_rgb = img_resized
    img_rgb[:, :, [0, 1, 2]] = img_resized[:, :, [2, 1, 0]]

    return img_rgb

def generate_train_data(dataframe, nbr_classes, img_root_path, shuffle=True, 
                        augment=False, img_width=224, img_height=224, model='vgg16'):
    N = dataframe.shape[0]
    if shuffle:
        dataframe = sk_shuffle(dataframe)
    X_train = np.zeros((N, img_width, img_height, 3))
    Y_train = np.zeros((N, nbr_classes))
    for index, row in dataframe.iterrows():
        driver_id = row['subject']
        classname = row['classname']
        label = int(classname[-1])
        img_name = row['img']
        img_path = os.path.join(img_root_path, 'train', classname, img_name)

        img = load_img(img_path, img_width)
        X_train[index] = img
        Y_train[index, label] = 1
    
    X_train = X_train.astype(np.float16)
    if model == 'vgg16':
        X_train = vgg16_preprocess_input(X_train)
    elif model == 'inceptv3':
        X_train = incept3_preprocess_input(X_train)

    return X_train, Y_train


def generate_train_data_triplet(dataframe, nbr_classes, img_root_path, shuffle=True, mode='train',
                        augment=False, img_width=224, img_height=224, model='vgg16'):
    N = dataframe.shape[0]
    if shuffle:
        dataframe = sk_shuffle(dataframe)
    X_anchor = np.zeros((N, img_width, img_height, 3))
    X_positive = np.zeros((N, img_width, img_height, 3))
    X_negative = np.zeros((N, img_width, img_height, 3))
    Y_train = np.zeros((N, nbr_classes))
    Y_pseudo_label = np.zeros((N, 1))
    for index, row in dataframe.iterrows():
        driver_id = row['subject']
        classname = row['classname']
        label = int(classname[-1])
        img_name = row['img']
        img_path = os.path.join(img_root_path, 'train', classname, img_name)

        img = load_img(img_path, img_width)

        if mode == 'train':
            same_driver_df = dataframe[dataframe['subject']==driver_id]
            classname_list = same_driver_df['classname'].unique().tolist()
            # find positive sample which has the same subject and the same classname but different img name
            same_driver_same_class_df = same_driver_df[same_driver_df['classname']==classname]
            same_driver_same_class_diff_img_df = same_driver_same_class_df[same_driver_same_class_df['img']!=img_name]
            assert len(same_driver_same_class_diff_img_df) != 0, 'driver:{},classname:{},only has one img:{}'.format(driver_id,classname,img_name)
            positive_row = same_driver_same_class_diff_img_df.sample(1)
            same_driver_same_class_df = []
            same_driver_same_class_diff_img_df = []
            # find negative sample which has the same subject and the different classname also different img name (Hard negative)
            classname_list.remove(classname)
            assert classname_list != [], 'driver: {},only has one class: {}'.format(driver_id,classname)
            other_classname = random.choice(classname_list)
            hard_negative_df = same_driver_df[same_driver_df['classname']==other_classname]
            negative_row = hard_negative_df.sample(1)
            hard_negative_df = []
            same_driver_df = []

            positive_img_path = os.path.join(img_root_path, 'train', positive_row['classname'].values[0], positive_row['img'].values[0])
            negative_img_path = os.path.join(img_root_path, 'train', negative_row['classname'].values[0], negative_row['img'].values[0])
            positive_img = load_img(positive_img_path, img_width)
            negative_img = load_img(negative_img_path, img_width)
        elif mode == 'valid':
            positive_img = img
            negative_img = img
        
        X_anchor[index] = img
        X_positive[index] = positive_img
        X_negative[index] = negative_img
        Y_train[index, label] = 1

    X_anchor = X_anchor.astype(np.float16)
    X_positive = X_positive.astype(np.float16)
    X_negative = X_negative.astype(np.float16)
    if model == 'vgg16':
        X_anchor = vgg16_preprocess_input(X_anchor)
        X_positive = vgg16_preprocess_input(X_positive)
        X_negative = vgg16_preprocess_input(X_negative)
    elif model == 'inceptv3':
        X_anchor = incept3_preprocess_input(X_anchor)
        X_positive = incept3_preprocess_input(X_positive)
        X_negative = incept3_preprocess_input(X_negative)

    return ([X_anchor,X_positive,X_positive], [Y_train,Y_pseudo_label])


def batch_generator(dataframe, nbr_classes, img_root_path, batch_size, shuffle=True, augment=False,
                     return_label=True, img_width=224, img_height=224, model='vgg16'):
    N = dataframe.shape[0]
    if shuffle:
        dataframe = sk_shuffle(dataframe)
    batch_index = 0
    while True:
        current_index = (batch_index * batch_size) % N
        if N >= (current_index + batch_size):
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = N - current_index
            batch_index = 0
            if shuffle:
                dataframe = sk_shuffle(dataframe)
        
        X_batch = np.zeros((current_batch_size, img_width, img_height, 3))
        Y_batch = np.zeros((current_batch_size, nbr_classes))
        
        for i in range(current_index, current_index + current_batch_size):
            row = dataframe.loc[i,:]
            driver_id = row['subject']
            classname = row['classname']
            label = int(classname[-1])
            img_name = row['img']
            img_path = os.path.join(img_root_path, 'train', classname, img_name)

            img = load_img(img_path, img_width)
            X_batch[i - current_index] = img
            if return_label:
                Y_batch[i - current_index, label] = 1

        if augment:
            X_batch = X_batch.astype(np.uint8)
            X_batch = seq.augment_images(X_batch)

        X_batch = X_batch.astype(np.float16)
        if model == 'vgg16':
            X_batch = vgg16_preprocess_input(X_batch)
        elif model == 'inceptv3':
            X_batch = incept3_preprocess_input(X_batch)

        if return_label:
            yield (X_batch, Y_batch)
        else:
            yield X_batch


def batch_generator_triplet(dataframe, nbr_classes, img_root_path, batch_size=16, mode='train',
                            return_label=True, img_width=224, img_height=224, shuffle=True, augment=False, model='vgg16'):
    N = dataframe.shape[0]
    if shuffle:
        dataframe = sk_shuffle(dataframe)
    batch_index = 0
    while True:
        current_index = (batch_index * batch_size) % N
        if N >= (current_index + batch_size):
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = N - current_index
            batch_index = 0
            if shuffle:
                dataframe = sk_shuffle(dataframe)

        X_anchor = np.zeros((current_batch_size, img_width, img_height, 3))
        X_positive = np.zeros((current_batch_size, img_width, img_height, 3))
        X_negative = np.zeros((current_batch_size, img_width, img_height, 3))
        Y_class = np.zeros((current_batch_size, nbr_classes))
        Y_pseudo_label = np.zeros((current_batch_size, 1))

        for i in range(current_index, current_index + current_batch_size):
            row = dataframe.loc[i, :]
            driver_id = row['subject']
            classname = row['classname']
            label = int(classname[-1])
            img_name = row['img']
            anchor_img_path = os.path.join(img_root_path, 'train', classname, img_name)
            anchor_img = load_img(anchor_img_path, img_width)

            if mode == 'train':
                same_driver_df = dataframe[dataframe['subject']==driver_id]
                classname_list = same_driver_df['classname'].unique().tolist()
                # find positive sample which has the same subject and the same classname but different img name
                same_driver_same_class_df = same_driver_df[same_driver_df['classname']==classname]
                same_driver_same_class_diff_img_df = same_driver_same_class_df[same_driver_same_class_df['img']!=img_name]
                assert len(same_driver_same_class_diff_img_df) != 0, 'driver:{},classname:{},only has one img:{}'.format(driver_id,classname,img_name)
                positive_row = same_driver_same_class_diff_img_df.sample(1)
                same_driver_same_class_df = []
                same_driver_same_class_diff_img_df = []
                # find negative sample which has the same subject and the different classname also different img name (Hard negative)
                classname_list.remove(classname)
                assert classname_list != [], 'driver: {},only has one class: {}'.format(driver_id,classname)
                other_classname = random.choice(classname_list)
                hard_negative_df = same_driver_df[same_driver_df['classname']==other_classname]
                negative_row = hard_negative_df.sample(1)
                hard_negative_df = []
                same_driver_df = []

                positive_img_path = os.path.join(img_root_path, 'train', positive_row['classname'].values[0], positive_row['img'].values[0])
                negative_img_path = os.path.join(img_root_path, 'train', negative_row['classname'].values[0], negative_row['img'].values[0])
                positive_img = load_img(positive_img_path, img_width)
                negative_img = load_img(negative_img_path, img_width)

                X_anchor[i - current_index] = anchor_img
                X_positive[i - current_index] = positive_img
                X_negative[i - current_index] = negative_img
            elif mode == 'val':
                X_anchor[i - current_index] = anchor_img
                X_positive[i - current_index] = anchor_img
                X_negative[i - current_index] = anchor_img
            
            if return_label:
                Y_class[i - current_index, label] = 1

        if augment:
            X_anchor = X_anchor.astype(np.uint8)
            X_anchor = seq.augment_images(X_anchor)
            X_positive = X_positive.astype(np.uint8)
            X_positive = seq.augment_images(X_positive)
            X_negative = X_negative.astype(np.uint8)
            X_negative = seq.augment_images(X_negative)

        X_anchor = X_anchor.astype(np.float16)
        X_positive = X_positive.astype(np.float16)
        X_negative = X_negative.astype(np.float16)

        if model == 'vgg16':
            X_anchor = vgg16_preprocess_input(X_anchor)
            X_positive = vgg16_preprocess_input(X_positive)
            X_negative = vgg16_preprocess_input(X_negative)
        elif model == 'inceptv3':
            X_anchor = incept3_preprocess_input(X_anchor)
            X_positive = incept3_preprocess_input(X_positive)
            X_negative = incept3_preprocess_input(X_negative)

        if return_label:
            yield ([X_anchor, X_positive, X_negative], [Y_class, Y_pseudo_label])
        else:
            if mode == 'feature_extraction':
                yield X_anchor
            else:
                yield [X_anchor, X_positive, X_negative]

def test_batch_generator(test_files, batch_size, img_width=224, img_height=224, model='vgg16'):
    N = len(test_files)
    batch_index = 0
    while True:
        current_index = (batch_index * batch_size) % N
        if N >= (current_index + batch_size):
            current_batch_size = batch_size
            batch_index += 1
        else:
            current_batch_size = N - current_index
            batch_index = 0

        X_batch = np.zeros((current_batch_size, img_width, img_height, 3))
        for i in range(current_index, current_index + current_batch_size):
            img_path = test_files[i]
            img = load_img(img_path, img_width)
            X_batch[i - current_index] = img

        X_batch = X_batch.astype(np.float16)
        if model == 'vgg16':
            X_batch = vgg16_preprocess_input(X_batch)
        elif model == 'inceptv3':
            X_batch = incept3_preprocess_input(X_batch)

        yield X_batch

def identity_loss(y_true, y_pred):
    return K.mean(y_pred - 0 * y_true)

def triplet_loss(vects):
    # f_anchor.shape = (batch_size, 256)
    f_anchor, f_positive, f_negative = vects
    # L2 normalize anchor, positive and negative, otherwise,
    # the loss will result in ''nan''!
    f_anchor = K.l2_normalize(f_anchor, axis=-1)
    f_positive = K.l2_normalize(f_positive, axis=-1)
    f_negative = K.l2_normalize(f_negative, axis=-1)

    dis_anchor_positive = K.sum(K.square(K.abs(f_anchor - f_positive)),
                                axis=-1, keepdims=True)

    dis_anchor_negative = K.sum(K.square(K.abs(f_anchor - f_negative)),
                                axis=-1, keepdims=True)
    loss = dis_anchor_positive + 1 - dis_anchor_negative
    return loss

