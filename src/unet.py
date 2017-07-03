from __future__ import division

import numpy as np
from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D, MaxPooling2D, UpSampling2D, Cropping2D
from keras.layers.merge import concatenate

from keras import backend as K
from keras.layers import Activation

import cv2

from keras.layers.normalization import BatchNormalization

from keras.optimizers import Nadam
import pandas as pd
from keras.backend import binary_crossentropy
import threading
from keras.callbacks import ModelCheckpoint, History, EarlyStopping
import os

import random

from keras.models import model_from_json
import datetime


img_rows = 224
img_cols = 224


smooth = K.epsilon()

num_channels = 3
num_mask_channels = 66


def jaccard_coef(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_int(y_true, y_pred):
    y_pred_pos = K.round(K.clip(y_pred, 0, 1))

    intersection = K.sum(y_true * y_pred_pos, axis=[0, -1, -2])
    sum_ = K.sum(y_true + y_pred_pos, axis=[0, -1, -2])

    jac = (intersection + smooth) / (sum_ - intersection + smooth)

    return K.mean(jac)


def jaccard_coef_loss(y_true, y_pred):
    return -K.log(jaccard_coef(y_true, y_pred)) + binary_crossentropy(y_pred, y_true)


# def ConvBN(x, n_filter, sz=3):
#     x = Conv2D(n_filter, (sz, sz), padding='same', kernel_initializer='he_uniform')(x)
#     x = BatchNormalization()(x)
#     x = selu()(x)
#     # x = Dropout(0.5)(x)
#     return x
#
#
# def ConvBN2(x, n_filter, sz=3):
#     x = Conv2D(n_filter, (sz, sz), padding='same', kernel_initializer='he_uniform')(x)
#     x = BatchNormalization()(x)
#     x = selu(x)
#     x = Conv2D(n_filter, (sz, sz), padding='same', kernel_initializer='he_uniform')(x)
#     x = BatchNormalization()(x)
#     x = selu(x)
#     # x = Dropout(0.5)(x)
#     return x
#

# def get_unet1(isz=ISZ, N_band=num_channels, N_Cls=num_mask_channels, lr=1e-3, ngpus=1):
#     nflt = 32
#     inputs = Input((isz, isz, N_band))
#     stem = BatchNormalization()(inputs)
#     conv1 = ConvBN2(stem, nflt, 3)
#     pool1 = MaxPooling2D()(conv1)
#     conv2 = ConvBN2(pool1, nflt * 2, 3)
#     pool2 = MaxPooling2D()(conv2)
#     conv3 = ConvBN2(pool2, nflt * 4, 3)
#     pool3 = MaxPooling2D()(conv3)
#     conv4 = ConvBN2(pool3, nflt * 8, 3)
#     pool4 = MaxPooling2D()(conv4)
#     conv5 = ConvBN2(pool4, nflt * 8, 3)
#     up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4])
#     conv6 = ConvBN2(up6, nflt * 8, 3)
#     up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3])
#     conv7 = ConvBN2(up7, nflt * 4, 3)
#     up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2])
#     conv8 = ConvBN2(up8, nflt * 2, 3)
#     up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1])
#     conv9 = ConvBN2(up9, nflt, 3)
#     conv10 = Conv2D(N_Cls, (1, 1), activation="sigmoid")(conv9)
#     model = Model(inputs=inputs, outputs=conv10)
#     # model.summary()
#     if ngpus > 1:
#         model = make_parallel(model, ngpus)
#     model.compile(loss='binary_crossentropy',
#                   optimizer='adam',
#                   metrics=[dice_coef])
#     return model


class threadsafe_iter:
    """Takes an iterator/generator and makes it thread-safe by
    serializing call to the `next` method of given iterator/generator.
    """
    def __init__(self, it):
        self.it = it
        self.lock = threading.Lock()

    def __iter__(self):
        return self

    def next(self):
        with self.lock:
            return self.it.next()


def threadsafe_generator(f):
    """A decorator that takes a generator function and makes it thread-safe.
    """
    def g(*a, **kw):
        return threadsafe_iter(f(*a, **kw))
    return g


def pixel_softmax(y_true, y_pred):
   y_pred = K.reshape(y_pred, (-1, num_mask_channels))
   y_pred = K.softmax(y_pred)
   y_true = K.reshape(y_true, (-1, num_mask_channels))
   return K.mean(K.categorical_crossentropy(y_true, y_pred))


def get_unet0():
    inputs = Input((img_rows, img_cols, num_channels))
    conv1 = Conv2D(32, (3, 3), padding="same", kernel_initializer="he_uniform")(inputs)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('selu')(conv1)
    conv1 = Conv2D(32, (3, 3), padding="same", kernel_initializer="he_uniform")(conv1)
    conv1 = BatchNormalization()(conv1)
    conv1 = Activation('selu')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Conv2D(64, (3, 3), padding="same", kernel_initializer="he_uniform")(pool1)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('selu')(conv2)
    conv2 = Conv2D(64, (3, 3), padding="same", kernel_initializer="he_uniform")(conv2)
    conv2 = BatchNormalization()(conv2)
    conv2 = Activation('selu')(conv2)

    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Conv2D(128, (3, 3), padding="same", kernel_initializer="he_uniform")(pool2)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('selu')(conv3)
    conv3 = Conv2D(128, (3, 3), padding="same", kernel_initializer="he_uniform")(conv3)
    conv3 = BatchNormalization()(conv3)
    conv3 = Activation('selu')(conv3)

    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Conv2D(256, (3, 3), padding="same", kernel_initializer="he_uniform")(pool3)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('selu')(conv4)
    conv4 = Conv2D(256, (3, 3), padding="same", kernel_initializer="he_uniform")(conv4)
    conv4 = BatchNormalization()(conv4)
    conv4 = Activation('selu')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Conv2D(512, (3, 3), padding="same", kernel_initializer="he_uniform")(pool4)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('selu')(conv5)
    conv5 = Conv2D(512, (3, 3), padding="same", kernel_initializer="he_uniform")(conv5)
    conv5 = BatchNormalization()(conv5)
    conv5 = Activation('selu')(conv5)

    up6 = concatenate([UpSampling2D(size=(2, 2))(conv5), conv4])
    conv6 = Conv2D(256, (3, 3), padding="same", kernel_initializer="he_uniform")(up6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('selu')(conv6)
    conv6 = Conv2D(256, (3, 3), padding="same", kernel_initializer="he_uniform")(conv6)
    conv6 = BatchNormalization()(conv6)
    conv6 = Activation('selu')(conv6)

    up7 = concatenate([UpSampling2D(size=(2, 2))(conv6), conv3])
    conv7 = Conv2D(128, (3, 3), padding="same", kernel_initializer="he_uniform")(up7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('selu')(conv7)
    conv7 = Conv2D(128, (3, 3), padding="same", kernel_initializer="he_uniform")(conv7)
    conv7 = BatchNormalization()(conv7)
    conv7 = Activation('selu')(conv7)

    up8 = concatenate([UpSampling2D(size=(2, 2))(conv7), conv2])
    conv8 = Conv2D(64, (3, 3), padding="same", kernel_initializer="he_uniform")(up8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('selu')(conv8)
    conv8 = Conv2D(64, (3, 3), padding="same", kernel_initializer="he_uniform")(conv8)
    conv8 = BatchNormalization()(conv8)
    conv8 = Activation('selu')(conv8)

    up9 = concatenate([UpSampling2D(size=(2, 2))(conv8), conv1])
    conv9 = Conv2D(32, (3, 3), padding="same", kernel_initializer="he_uniform")(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('selu')(conv9)
    conv9 = Conv2D(32, (3, 3), padding="same", kernel_initializer="he_uniform")(conv9)
    crop9 = Cropping2D(cropping=((16, 16), (16, 16)))(conv9)
    conv9 = BatchNormalization()(crop9)
    conv9 = Activation('selu')(conv9)

    conv10 = Conv2D(num_mask_channels, (1, 1))(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    return model


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def grayscale_to_mask(img):
    mask = np.zeros((img_rows, img_cols, num_mask_channels))
    for i in range(num_mask_channels):
        mask[:, :, i] = (img == i)
    return mask.astype(np.uint8)


def form_batch(X_path, y_path, batch_size):
    image_names = os.listdir(X_path)

    num_images = len(image_names)

    X_batch = np.zeros((batch_size, img_rows, img_cols, num_channels))
    y_batch = np.zeros((batch_size, img_rows, img_cols, num_mask_channels))

    for i in range(batch_size):
        random_image = np.random.randint(0, num_images - 1)

        file_name = image_names[random_image]

        img = cv2.imread(os.path.join(X_path, file_name))
        img_mask = cv2.imread(os.path.join(y_path, file_name.replace('.jpg', '.png')), 0)

        X_height = img.shape[0]
        X_width = img.shape[1]

        random_width = np.random.randint(0, X_width - img_cols - 1)
        random_height = np.random.randint(0, X_height - img_rows - 1)

        y_batch[i] = grayscale_to_mask(img_mask[random_height: random_height + img_rows,
                                       random_width: random_width + img_cols])

        X_batch[i] = img[random_height: random_height + img_rows, random_width: random_width + img_cols, :]
    return X_batch, y_batch


# @threadsafe_generator
def batch_generator(X_path, y_path, batch_size, horizontal_flip=False, vertical_flip=False, swap_axis=False):
    while True:
        X_batch, y_batch = form_batch(X_path, y_path, batch_size)

        # for i in range(X_batch.shape[0]):
        #     xb = X_batch[i]
        #     yb = y_batch[i]
        #
        #     if horizontal_flip:
        #         if np.random.random() < 0.5:
        #             xb = flip_axis(xb, 0)
        #             yb = flip_axis(yb, 0)
        #
        #     if vertical_flip:
        #         if np.random.random() < 0.5:
        #             xb = flip_axis(xb, 1)
        #             yb = flip_axis(yb, 1)
        #
        #     if swap_axis:
        #         if np.random.random() < 0.5:
        #             xb = xb.swapaxes(0, 1)
        #             yb = yb.swapaxes(0, 1)
        #
        #     X_batch[i] = xb
        #     y_batch[i] = yb

        yield X_batch, y_batch[:, 16:16 + img_rows - 32, 16:16 + img_cols - 32, :]


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

    train_path = os.path.join(data_path, 'training')
    X_train_path = os.path.join(train_path, 'images')
    y_train_path = os.path.join(train_path, 'instances')

    validation_path = os.path.join(data_path, 'validation')
    X_val_path = os.path.join(validation_path, 'images')
    y_val_path = os.path.join(validation_path, 'instances')

    print('[{}] Creating and compiling model...'.format(str(datetime.datetime.now())))

    model = get_unet0()

    print('[{}] Reading train...'.format(str(datetime.datetime.now())))

    batch_size = 24
    nb_epoch = 10

    now = datetime.datetime.now()
    suffix = str(now.strftime("%Y-%m-%d-%H-%M"))

    history = History()

    callbacks = [
        ModelCheckpoint('cache/resnet_full_' + suffix + '.hdf5', save_best_only=False, verbose=1),
        # EarlyStopping(patience=10, monitor='val_loss'),
        history
    ]

    model.compile(optimizer=Nadam(lr=1e-3), loss=pixel_softmax)
    # model.compile(optimizer=Nadam(lr=1e-3), loss=jaccard_coef_loss,
    #               metrics=['categorical_crossentropy', jaccard_coef_int])
    model.fit_generator(batch_generator(X_train_path, y_train_path, batch_size,
                                        horizontal_flip=False, vertical_flip=False, swap_axis=False),
                        steps_per_epoch=500,
                        epochs=nb_epoch,
                        verbose=1,
                        callbacks=callbacks,
                        workers=8,
                        max_queue_size=8,
                        use_multiprocessing=True
                        )

    save_model(model, "{batch}_{epoch}_{suffix}".format(batch=batch_size, epoch=nb_epoch, suffix=suffix))
    save_history(history, suffix)
