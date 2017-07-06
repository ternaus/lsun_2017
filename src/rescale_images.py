"""
No time, no resources => Let's downscale images
"""
from __future__ import division

import os
from joblib import Parallel, delayed
import cv2

new_width = 768
new_height = 512


def downscale(old_file_name):
    img = cv2.imread(os.path.join(old_file_name))

    new_file_name = (old_file_name
                     .replace('training', 'training_new')
                     .replace('validation', 'validation_new')
                     .replace('testing', 'testing_new')
                     )

    img_new = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(new_file_name, img_new)

if __name__ == '__main__':
    data_path = '../data'

    for image_set in ['training', 'validation', 'testing']:
        try:
            os.mkdir(os.path.join(data_path, image_set + '_new'))
        except:
            pass

        for subset in ['images', 'instances', 'labels']:
            try:
                os.mkdir(os.path.join(data_path, image_set + '_new', subset))
            except:
                pass

    for image_set in ['training', 'validation']:
        for subset in ['images', 'instances', 'labels']:
            target_path = os.path.join(data_path, image_set, subset)
            file_names = os.listdir(target_path)
            file_names = [os.path.join(target_path, file_name) for file_name in file_names]

            result = Parallel(n_jobs=8)(delayed(downscale)(r) for r in file_names)

    for image_set in ['testing']:
        for subset in ['images']:
            target_path = os.path.join(data_path, image_set, subset)
            file_names = os.listdir(target_path)
            file_names = [os.path.join(target_path, file_name) for file_name in file_names]

            result = Parallel(n_jobs=8)(delayed(downscale)(r) for r in file_names)
