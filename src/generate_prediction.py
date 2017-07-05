"""
script generated png files that are consumed as a prediction
"""

import os
import cv2
import pandas as pd
from keras.models import model_from_json
from tqdm import tqdm
import numpy as np


data_path = '../data'
validation_path = os.path.join(data_path, 'validation_320')
testing_path = os.path.join(data_path, 'testing_320')


try:
    os.mkdir(os.path.join(data_path, 'val_pred'))
except:
    pass

try:
    os.mkdir(os.path.join(data_path, 'test_pred'))
except:
    pass


def read_model(cross=''):
    json_name = '../src/cache/architecture_24_50_2017-07-03-14-07.json'
    weight_name = '../src/cache/resnet_full_2017-07-04-11-57.hdf5'
    model = model_from_json(open(json_name).read())
    model.load_weights(weight_name)
    return model


def pad_image(img, initial_size=(224, 224), final_size=(192, 192)):
    shift = int((initial_size[0] - final_size[0]) / 2)

    height, width, num_channels = img.shape

    if height % final_size[1] == 0:
        num_h_tiles = int(height / final_size[1])
    else:
        num_h_tiles = int(height / final_size[1]) + 1

    if width % final_size[1] == 0:
        num_w_tiles = int(width / final_size[1])
    else:
        num_w_tiles = int(width / final_size[1]) + 1

    rounded_height = num_h_tiles * final_size[0]
    rounded_width = num_w_tiles * final_size[0]

    padded_height = rounded_height + 2 * shift
    padded_width = rounded_width + 2 * shift

    padded = np.zeros((padded_height, padded_width, num_channels)).astype(np.float32)

    padded[shift:shift + height, shift: shift + width, :] = img

    # add mirror reflections to the padded areas
    up = padded[shift:2 * shift, shift:-shift, :][::-1, :]
    padded[:shift, shift:-shift, :] = up

    lag = padded.shape[0] - height - shift
    bottom = padded[height + shift - lag:shift + height, shift:-shift, :][::-1, :]
    padded[height + shift:, shift:-shift, :] = bottom

    left = padded[:, shift:2 * shift, :][:, ::-1, :]
    padded[:, :shift, :] = left

    lag = padded.shape[1] - width - shift
    right = padded[:, width + shift - lag:shift + width, :][:, ::-1, :]
    padded[:, width + shift:, :] = right
    return padded


def make_prediction(model, img, initial_size=(224, 224), final_size=(192, 192), num_masks=66):
    shift = int((initial_size[0] - final_size[0]) / 2)

    height, width, num_channels = img.shape

    if height % final_size[1] == 0:
        num_h_tiles = int(height / final_size[1])
    else:
        num_h_tiles = int(height / final_size[1]) + 1

    if width % final_size[1] == 0:
        num_w_tiles = int(width / final_size[1])
    else:
        num_w_tiles = int(width / final_size[1]) + 1

    rounded_height = num_h_tiles * final_size[0]
    rounded_width = num_w_tiles * final_size[0]

    padded_height = rounded_height + 2 * shift
    padded_width = rounded_width + 2 * shift

    padded = pad_image(img, initial_size, final_size)

    h_start = range(0, padded_height, final_size[0])[:-1]
    assert len(h_start) == num_h_tiles

    w_start = range(0, padded_width, final_size[0])[:-1]
    assert len(w_start) == num_w_tiles

    temp = []
    for h in h_start:
        for w in w_start:
            temp += [padded[h:h + initial_size[0], w:w + initial_size[0], :]]

    prediction = model.predict(np.array(temp))

    predicted_mask = np.zeros((rounded_height, rounded_width, num_masks))

    for j_h, h in enumerate(h_start):
         for j_w, w in enumerate(w_start):
             i = len(w_start) * j_h + j_w
             predicted_mask[h: h + final_size[0], w: w + final_size[0], :] = prediction[i]

    return predicted_mask[:height, :width]


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
        img = cv2.imread(file_name.replace('validation', 'validation_320')).astype(np.float32)
        img = normalize(img)

        prediction = make_prediction(model, img, initial_size=(224, 224), final_size=(192, 192), num_masks=66)

        mask = np.argmax(prediction, axis=2).astype(np.uint8)

        original_height = val_shapes.loc[file_name, 'height']
        original_width = val_shapes.loc[file_name, 'width']

        resized_mask = cv2.resize(mask, (original_width, original_height), interpolation=cv2.INTER_LINEAR)

        final_path = file_name.replace('validation/images', 'val_pred')[:-3] + 'png'
        cv2.imwrite(final_path, resized_mask)

