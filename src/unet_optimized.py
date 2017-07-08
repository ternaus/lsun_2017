from __future__ import division

import os
os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

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

from keras.callbacks import ModelCheckpoint, History

from keras.models import model_from_json
import datetime
import tensorflow as tf


__author__ = 'Vladimir Iglovikov'

img_rows = 224
img_cols = 224


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
    conv9 = Conv2D(num_start_filters, (3, 3), padding="same", kernel_initializer="he_uniform")(up9)
    conv9 = BatchNormalization()(conv9)
    conv9 = Activation('selu')(conv9)
    conv9 = Conv2D(num_start_filters, (3, 3), padding="same", kernel_initializer="he_uniform")(conv9)
    crop9 = Cropping2D(cropping=((16, 16), (16, 16)))(conv9)
    conv9 = BatchNormalization()(crop9)
    conv9 = Activation('selu')(conv9)

    conv10 = Conv2D(num_mask_channels, (1, 1))(conv9)

    model = Model(inputs=inputs, outputs=conv10)

    return model


def form_batch(X_path, y_path, batch_size):
    image_names = os.listdir(X_path)

    num_images = len(image_names)

    X_batch = np.zeros((batch_size, img_rows, img_cols, num_channels))
    y_batch = np.zeros((batch_size, img_rows, img_cols))

    for i in range(batch_size):
        random_image = np.random.randint(0, num_images - 1)

        file_name = image_names[random_image]

        img = cv2.imread(os.path.join(X_path, file_name))
        img_mask = cv2.imread(os.path.join(y_path, file_name.replace('.jpg', '.png')), 0)

        X_height = img.shape[0]
        X_width = img.shape[1]

        random_width = np.random.randint(0, X_width - img_cols - 1)
        random_height = np.random.randint(0, X_height - img_rows - 1)

        # y_batch[i] = grayscale_to_mask(img_mask[random_height: random_height + img_rows,
        #                                random_width: random_width + img_cols])

        y_batch[i] = img_mask[random_height: random_height + img_rows,
                                       random_width: random_width + img_cols]

        X_batch[i] = img[random_height: random_height + img_rows, random_width: random_width + img_cols, :]
    return X_batch, y_batch


def normalize(x):
    x[:, :, :, 0] -= 103.939
    x[:, :, :, 1] -= 116.779
    x[:, :, :, 2] -= 123.68

    return x


def batch_generator(X_path, y_path, batch_size):
    while True:
        X_batch, y_batch = form_batch(X_path, y_path, batch_size)

        # Add augmentations here

        X_batch = normalize(X_batch)

        yield X_batch, np.expand_dims(y_batch[:, 16:16 + img_rows - 32, 16:16 + img_cols - 32].astype(np.uint8), 3)


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
    nb_epoch = 50

    now = datetime.datetime.now()
    suffix = str(now.strftime("%Y-%m-%d-%H-%M"))

    history = History()

    callbacks = [
        ModelCheckpoint('cache/resnet_full_' + suffix + '.hdf5', save_best_only=False, verbose=1),
        # EarlyStopping(patience=10, monitor='val_loss'),
        history
    ]

    model.load_weights('cache/resnet_full_2017-07-04-11-57.hdf5')

    model.compile(optimizer=Nadam(lr=1e-4), loss=pixel_softmax)
    model.fit_generator(batch_generator(X_train_path, y_train_path, batch_size),
                        steps_per_epoch=500,
                        epochs=nb_epoch,
                        verbose=1,
                        callbacks=callbacks,
                        validation_data=batch_generator(X_val_path, y_val_path, batch_size),
                        validation_steps=100,
                        workers=8,
                        max_queue_size=8,
                        use_multiprocessing=True
                        )

    save_model(model, "{batch}_{epoch}_{suffix}".format(batch=batch_size, epoch=nb_epoch, suffix=suffix))
    save_history(history, suffix)
