"""Dataset helper module."""
import glob
import os.path

import numpy as np
import tensorflow as tf

from ..io.file_lists import generate_glob_and_replacer, prepare_for_regex
from ..io.tiffmasks import tiff_masks, tiff_peek


def dataset_from_tiff(file_name, mode='dynamic'):
    """
    Load a TIFF Stack with ImageJ ROIs as a TensorFlow Dataset (image, label).

    :param file_name:
    :param mode:
    :return:
    """
    info = tiff_peek(file_name)

    dtype = tf.as_dtype(info.dtype)
    shape = (info.h, info.w)

    if mode == 'preload':
        buffer = np.zeros(
            (
                info.pages,
                2,
            )
            + shape,
            dtype=info.dtype,
        )
        for n, (x, y) in enumerate(tiff_masks(file_name)):
            buffer[n, 0], buffer[n, 1] = x, y
        dataset = tf.data.Dataset.from_tensor_slices((buffer[:, 0], buffer[:, 1]))
    elif mode == 'dynamic':
        # somehow this does not work,
        # even after .cache() 'ing it later, it seems to be rerun every step
        dataset = tf.data.Dataset.from_generator(
            tiff_masks,
            output_shapes=(shape, shape),
            output_types=(dtype, dtype),
            args=(file_name,),
        )
    else:
        raise RuntimeError("Invalid mode")

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


def dataset_from_filenames(file_names, label_names, filename_list=''):
    """
    Load a list of filenames as TensorFlow dataset (image, label).

    :param file_names:
    :param label_names:
    :param filename_list:
    :return:
    """
    file_list = []

    # TODO: this function still is kinda old python glory

    image_pattern, labels_pattern = None, None

    for image_pattern, labels_pattern in zip(file_names, label_names):
        glob_pattern, replace = generate_glob_and_replacer(
            image_pattern, labels_pattern
        )

        file_list += list(
            sorted([(file, replace(file)) for file in glob.glob(glob_pattern)])
        )

    for image_file, labels_file in file_list:
        if not os.path.isfile(image_file):
            raise RuntimeError("Image file missing:" + str(image_file))
        if not os.path.isfile(labels_file):
            raise RuntimeError("Labels file missing:" + str(labels_file))

    if not file_list:
        raise RuntimeError("Empty file list. This is probably not what you wanted.")

    image_pattern = prepare_for_regex(image_pattern, task='glob')
    labels_pattern = prepare_for_regex(labels_pattern, task='glob')

    if filename_list:
        filename_list = [
            single_file.strip() for single_file in open(filename_list).readlines()
        ]

        # this is more heuristically

        if filename_list[0].endswith(image_pattern[-4:]):
            filename_list = [single_file[:-4] for single_file in filename_list]
        elif filename_list[0].endswith(labels_pattern[-4:]):
            filename_list = [single_file[:-4] for single_file in filename_list]

        image_list = [
            image_pattern.replace('*', single_file) for single_file in filename_list
        ]

        label_list = [
            labels_pattern.replace('*', single_file) for single_file in filename_list
        ]

        dataset = tf.data.Dataset.zip(
            (
                tf.data.Dataset.from_tensor_slices(image_list),
                tf.data.Dataset.from_tensor_slices(label_list),
            )
        )
    else:
        dataset = tf.data.Dataset.zip(
            (
                tf.data.Dataset.list_files(image_pattern, shuffle=False),
                tf.data.Dataset.list_files(labels_pattern, shuffle=False),
            )
        )

    @tf.function
    def _ends_tiff(name):
        name = tf.strings.lower(name)
        return tf.math.logical_or(
            tf.equal(tf.strings.substr(name, -4, -1), ".tif"),
            tf.equal(tf.strings.substr(name, -5, -1), ".tiff"),
        )

    def _decode_tiff(name):
        from tifffile import TiffFile

        name = name.numpy().decode()
        with TiffFile(name) as tiff:
            image = tiff.asarray().astype(np.float32)
            if image.ndim == 2:
                image = image[:, :, np.newaxis]
            return image

    @tf.function
    def _flexible_decode_image(name):
        if _ends_tiff(name):
            return tf.py_function(func=_decode_tiff, inp=[name], Tout=tf.float32)
        else:
            return tf.cast(
                tf.io.decode_image(tf.io.read_file(name), expand_animations=False),
                tf.float32,
            )

    @tf.function
    def _load(image_file_, labels_file_):
        tf.print("Loading", image_file_, labels_file_)
        image_data_ = _flexible_decode_image(image_file_)
        labels_data_ = _flexible_decode_image(labels_file_)
        return image_data_, labels_data_

    dataset = dataset.map(_load)

    return dataset
