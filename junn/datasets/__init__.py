import glob
import os.path

import numpy as np
import tensorflow as tf

from ..io.file_lists import generate_glob_and_replacer, prepare_for_regex
from ..io.tiffmasks import tiff_peek, tiff_masks


def dataset_from_tiff(file_name, mode='dynamic'):
    """
    Loads a TIFF Stack with ImageJ ROIs as a TensorFlow Dataset (image, label).
    :param file_name:
    :param mode:
    :return:
    """
    info = tiff_peek(file_name)

    dtype = tf.as_dtype(info.dtype)
    shape = (info.h, info.w)

    if mode == 'preload':
        buffer = np.zeros((info.pages, 2,) + shape, dtype=info.dtype)
        for n, (x, y) in enumerate(tiff_masks(file_name)):
            buffer[n, 0], buffer[n, 1] = x, y
        dataset = tf.data.Dataset.from_tensor_slices((buffer[:, 0], buffer[:, 1]))
    elif mode == 'dynamic':
        # somehow this does not work,
        # even after .cache() 'ing it later, it seems to be rerun every step
        dataset = tf.data.Dataset.from_generator(tiff_masks,
                                                 output_shapes=(shape, shape),
                                                 output_types=(dtype, dtype),
                                                 args=(file_name,))
    else:
        raise RuntimeError('Invalid mode')

    @tf.function
    def _force_three_dim(image, labels):
        image = tf.expand_dims(image, -1)
        labels = tf.expand_dims(labels, -1)

        return image, labels

    @tf.function
    def _cast(image, labels):
        return tf.cast(image, tf.float32), tf.cast(labels, tf.float32)

    if info.dtype not in (np.uint8, np.float32):
        # TensorFlow sadly is quite picky for what dtypes it supports all operations
        dataset = dataset.map(_cast)

    dataset = dataset.map(_force_three_dim)

    return dataset


def dataset_from_filenames(file_names, label_names):
    """
    Load a list of filenames as TensorFlow dataset (image, label).
    :param file_names:
    :param label_names:
    :return:
    """
    file_list = []

    # TODO: this function still is kinda old python glory

    image_pattern, labels_pattern = None, None

    for image_pattern, labels_pattern in zip(file_names, label_names):
        glob_pattern, replace = generate_glob_and_replacer(image_pattern, labels_pattern)

        file_list += list(sorted([(file, replace(file)) for file in glob.glob(glob_pattern)]))

    for image_file, labels_file in file_list:
        if not os.path.isfile(image_file):
            raise RuntimeError("Image file missing:" + str(image_file))
        if not os.path.isfile(labels_file):
            raise RuntimeError("Labels file missing:" + str(labels_file))

    if not file_list:
        raise RuntimeError("Empty file list. This is probably not what you wanted.")

    image_pattern = prepare_for_regex(image_pattern, task='glob')
    labels_pattern = prepare_for_regex(labels_pattern, task='glob')

    dataset = tf.data.Dataset.zip((
        tf.data.Dataset.list_files(image_pattern, shuffle=False),
        tf.data.Dataset.list_files(labels_pattern, shuffle=False)
    ))

    @tf.function
    def _load(image_file_, labels_file_):
        image_data_ = tf.io.decode_image(tf.io.read_file(image_file_), expand_animations=False)
        labels_data_ = tf.io.decode_image(tf.io.read_file(labels_file_), expand_animations=False)

        return image_data_, labels_data_

    dataset = dataset.map(_load)

    return dataset
