# from junn.common.configure_tensorflow import configure_tensorflow
# configure_tensorflow(0)  # very important to set allow_growth to True

import pytest

import numpy as np
from tifffile import TiffWriter
from roifile import ImagejRoi

from PIL import Image


def quadratic_test_image(size=512, dtype=np.uint8):
    image = np.zeros((size, size), dtype=dtype)

    image[128:128 + 256, 128:128 + 256] = 2 ** (dtype().nbytes * 8) - 1

    return image


def some_imagejroi():
    return ImagejRoi.frompoints(
        [[128.0 + 000.0, 128.0 + 000.0],
         [128.0 + 256.0, 128.0 + 000.0],
         [128.0 + 256.0, 128.0 + 256.0],
         [128.0 + 000.0, 128.0 + 256.0],
         [128.0 + 000.0, 128.0 + 000.0]])


def trackmate_roi():
    roi = some_imagejroi()
    roi.name = 'TrackMate ROI'
    return roi


def dualpoint_roi():
    return ImagejRoi.frompoints(
        some_imagejroi().coordinates()[:2]
    )


@pytest.fixture(scope='function')
def empty_training_data(tmpdir_factory):
    name = str(tmpdir_factory.mktemp('training').join('train.tif'))

    with TiffWriter(name, imagej=True) as tiff:
        image = quadratic_test_image(size=512, dtype=np.uint16)

        tiff.save(
            image,
            resolution=(1.0/0.065, 1.0/0.065), metadata=dict(unit='um'),
            ijmetadata=dict(Overlays=[roi.tobytes() for roi in [some_imagejroi()]])
        )

    return name


@pytest.fixture(scope='function')
def funny_tiff_file(tmpdir_factory):
    name = str(tmpdir_factory.mktemp('training').join('train.tif'))

    with TiffWriter(name, imagej=True) as tiff:
        image = quadratic_test_image(size=512, dtype=np.uint16)

        tiff.save(
            image,
            resolution=(1.0/0.065, 1.0/0.065), metadata=dict(unit='um'),
            ijmetadata=dict(Overlays=[roi.tobytes() for roi in [
                some_imagejroi(),
                trackmate_roi(),
                dualpoint_roi()
            ]])
        )

    return name


@pytest.fixture(scope='function')
def empty_training_image_directory(tmpdir_factory):
    base = tmpdir_factory.mktemp('training_directory')
    name = str(base)

    images_dir, labels_dir = base.mkdir('images'), base.mkdir('labels')

    image = quadratic_test_image()

    Image.fromarray(image, 'L').save(str(images_dir.join('image-000.png')))
    Image.fromarray(image, 'L').save(str(labels_dir.join('label-000.png')))

    return name


@pytest.fixture(scope='function')
def disable_cuda():
    import os
    from contextlib import contextmanager

    @contextmanager
    def _disable_cuda():
        old_value = os.environ.get('CUDA_VISIBLE_DEVICES')
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        yield

        if old_value:
            os.environ['CUDA_VISIBLE_DEVICES'] = old_value

    def giver():
        return _disable_cuda()

    return giver


@pytest.fixture(scope='function')
def tf_eager():
    from contextlib import contextmanager
    import tensorflow as tf

    @contextmanager
    def _eager_change_context(eager):
        old_value = tf.config.experimental_functions_run_eagerly()
        tf.config.experimental_run_functions_eagerly(eager)
        yield
        tf.config.experimental_run_functions_eagerly(old_value)

    def giver(eager):
        return _eager_change_context(eager)

    return giver
