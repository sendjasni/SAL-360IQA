import pandas as pd
from scipy.signal.signaltools import convolve2d
import numpy as np
import tensorflow as tf


class Dataset:
    def __init__(self):
        pass

    def Local_normalization(self, img, P=3, Q=3, C=1):
        kernel = np.ones((P, Q)) / (P * Q)
        img_mean = convolve2d(img, kernel, boundary='symm', mode='same')
        img_sm = convolve2d(np.square(img), kernel,
                            boundary='symm', mode='same')
        img_std = np.sqrt(np.maximum(img_sm - np.square(img_mean), 0)) + C
        img_ln = (img - img_mean) / img_std
        return img_ln

    def get_input(self, data_file, patches_path, nbr_img, nbr_patches, norm=True):

        df = pd.read_csv(data_file)
        moss = df['MOS'].tolist()

        patches = []
        mos_per_patch = []
        for i, f in enumerate(df['PATCH'][:nbr_img * nbr_patches]):
            patch = tf.keras.preprocessing.image.load_img(patches_path + f)
            patch = tf.keras.preprocessing.image.img_to_array(patch)
            if norm:
                patch = patch.convert('L')
                patch = self.Local_normalization(patch)
            patches.append(patch)

            mos_per_patch.append(moss[i])

        return patches, mos_per_patch
