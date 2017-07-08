"""
script generated png files that are consumed as a prediction
"""

import os
import cv2
import pandas as pd
from keras.models import model_from_json
from tqdm import tqdm
import numpy as np

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

data_path = '../data'
validation_path = os.path.join(data_path, 'validation_new')
testing_path = os.path.join(data_path, 'testing_new')


try:
    os.mkdir(os.path.join(data_path, 'val_pred'))
except:
    pass

try:
    os.mkdir(os.path.join(data_path, 'test_pred'))
except:
    pass


def read_model(cross=''):
    json_name = '../src/cache/architecture_4_50_2017-07-06-07-46.json'
    weight_name = '../src/cache/resnet_full_2017-07-07-08-40.hdf5'
    model = model_from_json(open(json_name).read())
    model.load_weights(weight_name)
    return model


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def normalize(x):
    x[:, :, 0] -= 103.939
    x[:, :, 1] -= 116.779
    x[:, :, 2] -= 123.68

    return x

if __name__ == '__main__':
    val_shapes = pd.read_csv(os.path.join(data_path, 'validation.csv'))
    test_shapes = pd.read_csv(os.path.join(data_path, 'testing.csv'))

    val_shapes = val_shapes.set_index('file_name')

    model = read_model()

    for file_name in tqdm(val_shapes.index):
        img = cv2.imread(file_name.replace('validation', 'validation_new')).astype(np.float32)
        img = normalize(img)

        temp = np.zeros((2, img.shape[0], img.shape[1], 3))
        temp[0] = img
        temp[1] = flip_axis(img, 1)

        prediction = model.predict(temp)

        mask = np.argmax(prediction[0], axis=2).astype(np.uint8)

        original_height = val_shapes.loc[file_name, 'height']
        original_width = val_shapes.loc[file_name, 'width']

        resized_mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_LINEAR)

        final_path = file_name.replace('validation/images', 'val_pred')[:-3] + 'png'
        cv2.imwrite(final_path, resized_mask)

