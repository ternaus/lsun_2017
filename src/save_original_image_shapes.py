"""
Initial images are decently big. Currently we downscale images. But at the prediction time we need to have png
 images exactly of the initial size.
 
 => we need to cache what we have in the beginning
"""

import pandas as pd
import os
import cv2
data_path = '../data'
from joblib import Parallel, delayed


def get_shape(file_name):
    img = cv2.imread(file_name)
    height, width, _ = img.shape
    return file_name, height, width


for image_set in ['training', 'validation', 'testing']:
    target_path = os.path.join(data_path, image_set, 'images')

    file_list = [os.path.join(target_path, x) for x in os.listdir(target_path)]

    name_height_width = Parallel(n_jobs=16)(delayed(get_shape)(r) for r in file_list)

    df = pd.DataFrame(name_height_width, columns=['file_name', 'height', 'width'])

    df.to_csv(os.path.join(data_path, image_set + '.csv'), index=False)
