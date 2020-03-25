from functools import reduce

import tensorflow as tf
from tunable import Selectable

from ..datasets import dataset_from_tiff, dataset_from_filenames


class TrainingInput(Selectable):
    def get(self, *args) -> tf.data.Dataset:
        pass


class ImageJTiffInput(TrainingInput, TrainingInput.Default):
    def get(self, *args) -> tf.data.Dataset:
        return reduce(lambda a, b: a.concatenate(b), [dataset_from_tiff(file_name) for file_name in args])


class ImageDirectoryInput(TrainingInput):
    def __init__(self, images='', labels='', normalize_labels=False):
        self.images, self.labels = images, labels
        self.normalize_labels = normalize_labels

    def get(self, *args) -> tf.data.Dataset:
        dataset = dataset_from_filenames([self.images], [self.labels])

        @tf.function
        def _normalize_labels(x, y):

            y = tf.reduce_mean(y, axis=-1)
            y = y[..., tf.newaxis]
            y = y/tf.reduce_max(y)

            return x, y

        if self.normalize_labels:
            dataset = dataset.map(_normalize_labels)

        return dataset
