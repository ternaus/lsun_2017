from __future__ import division

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, Cropping2D
from keras.layers.merge import concatenate
from keras.layers import BatchNormalization

from keras import backend as K
from keras.layers import Activation
from sklearn.utils import shuffle

import cv2

from keras.optimizers import Nadam, SGD
import pandas as pd

from keras.callbacks import ModelCheckpoint, History

from keras.models import model_from_json
import datetime
import tensorflow as tf


__author__ = 'Vladimir Iglovikov'

img_rows = 512
img_cols = 768


smooth = K.epsilon()

num_channels = 3
num_mask_channels = 66


def pixel_softmax(y_true, y_pred):
    """
    Thanks to Alexander Movchan
     
    :param y_true: y_true shape: (batch_size, h, w)
    :param y_pred: y_pred shape: (batch_size, h, w, num_classes)
    :return: 
    """

    y_true = tf.cast(y_true[:, :, :, 0], tf.int32)
    return K.mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y_true, logits=y_pred))


def ConvBN2(x, num_filter, stride_size=3):
    x = Conv2D(num_filter, (stride_size, stride_size), padding='same', kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('selu')(x)
    x = Conv2D(num_filter, (stride_size, stride_size), padding='same', kernel_initializer='he_uniform')(x)
    x = BatchNormalization()(x)
    x = Activation('selu')(x)
    return x


def get_unet0(num_start_filters=32):
    inputs = Input((img_rows, img_cols, num_channels))
    conv1 = ConvBN2(inputs, num_start_filters)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = ConvBN2(pool1, 2 * num_start_filters)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = ConvBN2(pool2, 4 * num_start_filters)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = ConvBN2(pool3, 8 * num_start_filters)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = ConvBN2(pool4, 16 * num_start_filters)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4])
    conv6 = ConvBN2(up6, 8 * num_start_filters)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3])
    conv7 = ConvBN2(up7, 4 * num_start_filters)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2])
    conv8 = ConvBN2(up8, 2 * num_start_filters)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1])
    conv9 = ConvBN2(up9, num_start_filters)

    conv10 = Conv2D(num_mask_channels, (1, 1))(conv9)
    model = Model(inputs=inputs, outputs=conv10)

    return model


# def form_batch(X_path, y_path, batch_size, horizontal_flip=False, vertical_flip=False):
#     image_names = os.listdir(X_path)
#
#     num_images = len(image_names)
#
#     X_batch = np.zeros((batch_size, img_rows, img_cols, num_channels))
#     y_batch = np.zeros((batch_size, img_rows, img_cols))
#
#     for i in range(batch_size):
#         random_image = np.random.randint(0, num_images - 1)
#
#         file_name = image_names[random_image]
#
#         img = cv2.imread(os.path.join(X_path, file_name))
#         img_mask = cv2.imread(os.path.join(y_path, file_name.replace('.jpg', '.png')), 0)
#
#         yb = img_mask
#         xb = img
#
#         if horizontal_flip:
#             if np.random.random() < 0.5:
#                 xb = flip_axis(xb, 0)
#                 yb = flip_axis(yb, 0)
#
#         if vertical_flip:
#             if np.random.random() < 0.5:
#                 xb = flip_axis(xb, 1)
#                 yb = flip_axis(yb, 1)
#
#         y_batch[i] = yb
#
#         X_batch[i] = xb
#
#     return X_batch, y_batch


def normalize(x):
    x[:, :, :, 0] -= 103.939
    x[:, :, :, 1] -= 116.779
    x[:, :, :, 2] -= 123.68

    return x


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def batch_generator(X_path, y_path, batch_size, horizontal_flip=False, vertical_flip=False):
    X_file_list = os.listdir(X_path)

    num_batches = int(len(X_file_list) / batch_size)
    while True:
        for batch_index in range(num_batches):
            X_batch = np.zeros((batch_size, img_rows, img_cols, num_channels))
            y_batch = np.zeros((batch_size, img_rows, img_cols))

            for i in range(batch_size):
                img_path = X_file_list[batch_size * batch_index + i]
                mask_path = X_file_list[batch_size * batch_index + i][:-3] + 'png'

                img = cv2.imread(os.path.join(X_path, img_path))
                img_mask = cv2.imread(os.path.join(y_path, mask_path), 0)
                yb = img_mask
                xb = img

                if horizontal_flip:
                    if np.random.random() < 0.5:
                        xb = flip_axis(xb, 0)
                        yb = flip_axis(yb, 0)

                if vertical_flip:
                    if np.random.random() < 0.5:
                        xb = flip_axis(xb, 1)
                        yb = flip_axis(yb, 1)

                y_batch[i] = yb

                X_batch[i] = xb

            # X_batch, y_batch = form_batch(X_path, y_path, batch_size, horizontal_flip, vertical_flip)

            # Add augmentations here

            X_batch = normalize(X_batch)

            yield X_batch, np.expand_dims(y_batch.astype(np.uint8), 3)

        X_file_list = shuffle(X_file_list)


def save_model(model, cross):
    json_string = model.to_json()
    if not os.path.isdir('cache'):
        os.mkdir('cache')
    json_name = 'architecture_' + cross + '.json'
    weight_name = 'model_weights_' + cross + '.h5'
    open(os.path.join('cache', json_name), 'w').write(json_string)
    model.save_weights(os.path.join('cache', weight_name), overwrite=True)


def save_history(history, suffix):
    filename = 'history/history_' + suffix + '.csv'
    pd.DataFrame(history.history).to_csv(filename, index=False)


def read_model(cross=''):
    json_name = 'architecture_' + cross + '.json'
    weight_name = 'model_weights_' + cross + '.h5'
    model = model_from_json(open(os.path.join('../src/cache', json_name)).read())
    model.load_weights(os.path.join('../src/cache', weight_name))
    return model


if __name__ == '__main__':
    now = datetime.datetime.now()
    suffix = str(now.strftime("%Y-%m-%d-%H-%M"))

    data_path = '../data'

    train_path = os.path.join(data_path, 'training_new')
    X_train_path = os.path.join(train_path, 'images')
    y_train_path = os.path.join(train_path, 'instances')

    validation_path = os.path.join(data_path, 'validation_new')
    X_val_path = os.path.join(validation_path, 'images')
    y_val_path = os.path.join(validation_path, 'instances')

    print('[{}] Creating and compiling model...'.format(str(datetime.datetime.now())))

    model = get_unet0()

    print('[{}] Reading train...'.format(str(datetime.datetime.now())))

    batch_size = 4
    nb_epoch = 500

    now = datetime.datetime.now()
    suffix = str(now.strftime("%Y-%m-%d-%H-%M"))

    history = History()

    callbacks = [
        ModelCheckpoint('cache/resnet_full_' + suffix + '.hdf5', save_best_only=False, verbose=1),
        # EarlyStopping(patience=10, monitor='val_loss'),
        history
    ]

    model.load_weights('cache/resnet_full_2017-07-07-08-40.hdf5')

    model.compile(optimizer=Nadam(lr=1e-5), loss=pixel_softmax)

    model.fit_generator(batch_generator(X_train_path, y_train_path, batch_size, horizontal_flip=False, vertical_flip=True),
                        steps_per_epoch=500,
                        epochs=nb_epoch,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=batch_generator(X_val_path, y_val_path, batch_size),
                        validation_steps=100,
                        # workers=8,
                        # max_queue_size=8,
                        # use_multiprocessing=True
                        )

    save_model(model, "{batch}_{epoch}_{suffix}".format(batch=batch_size, epoch=nb_epoch, suffix=suffix))
    save_history(history, suffix)
