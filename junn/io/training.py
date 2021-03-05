from functools import reduce

import tensorflow as tf
from tunable import Selectable

from ..datasets import dataset_from_tiff, dataset_from_filenames


class TrainingInputMode:
    pass


class ModeTile(TrainingInputMode):
    pass


class ModeBBox(TrainingInputMode):
    pass


class TrainingInput(Selectable):
    def get(self, *args, mode=None) -> tf.data.Dataset:
        pass


class ImageJTiffInput(TrainingInput, TrainingInput.Default):
    def get(self, *args, mode=ModeTile) -> tf.data.Dataset:
        if mode == ModeTile:
            return reduce(lambda a, b: a.concatenate(b), [dataset_from_tiff(file_name) for file_name in args])
        elif mode == ModeBBox:
            raise NotImplementedError
        else:
            raise RuntimeError('Invalid mode passed')


class ImageDirectoryInput(TrainingInput):
    def __init__(self, images='', labels='', filename_list='',
                 normalize_labels=False, binarize_labels=False, remove_alpha=True):
        self.images, self.labels = images, labels
        self.filename_list = filename_list
        self.normalize_labels = normalize_labels
        self.binarize_labels = binarize_labels
        self.remove_alpha = remove_alpha

    def get(self, *args, mode=ModeTile) -> tf.data.Dataset:
        if mode == ModeTile:
            dataset = dataset_from_filenames([self.images], [self.labels], filename_list=self.filename_list)

            @tf.function
            def _remove_alpha(x, y):
                if tf.shape(y)[-1] == 4:
                    y = y[:, :, :3]
                return x, y

            if self.remove_alpha:
                dataset = dataset.map(_remove_alpha)

            @tf.function
            def _normalize_labels(x, y):

                y = tf.reduce_mean(y, axis=-1)
                y = y[..., tf.newaxis]
                y = y/tf.reduce_max(y)

                return x, y

            if self.normalize_labels:
                dataset = dataset.map(_normalize_labels)

            @tf.function
            def _binarize_labels(x, y):
                if tf.shape(y)[-1] > 1:
                    y = tf.reduce_max(y, axis=-1, keepdims=True)
                    y = tf.where(tf.greater(y, 0), tf.cast(1.0, tf.float32), tf.cast(0.0, tf.float32))
                return x, y

            if self.binarize_labels:
                dataset = dataset.map(_binarize_labels)

            return dataset
        elif mode == ModeBBox:
            raise NotImplementedError
        else:
            raise RuntimeError('Invalid mode passed')
