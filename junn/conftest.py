import os
import sys

if os.path.isdir('junn-predict'):
    sys.path.insert(0, 'junn-predict')

from junn_predict.common.configure_tensorflow import configure_tensorflow

configure_tensorflow(0)  # very important to set allow_growth to True

import warnings

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    import tensorflow.python.autograph.impl.api

from contextlib import contextmanager

import numpy as np
import pytest
from PIL import Image
from roifile import ImagejRoi
from tifffile import TiffWriter
from tunable import Selectable, TunableManager


def quadratic_test_image(size=512, dtype=np.uint8):
    image = np.zeros((size, size), dtype=dtype)

    image[128 : 128 + 256, 128 : 128 + 256] = 2 ** (dtype().nbytes * 8) - 1

    return image


def some_imagejroi():
    return ImagejRoi.frompoints(
        [
            [128.0 + 000.0, 128.0 + 000.0],
            [128.0 + 256.0, 128.0 + 000.0],
            [128.0 + 256.0, 128.0 + 256.0],
            [128.0 + 000.0, 128.0 + 256.0],
            [128.0 + 000.0, 128.0 + 000.0],
        ]
    )


def trackmate_roi():
    roi = some_imagejroi()
    roi.name = 'TrackMate ROI'
    return roi


def dualpoint_roi():
    return ImagejRoi.frompoints(some_imagejroi().coordinates()[:2])


@pytest.fixture(autouse=True)
def reset_state():
    TunableManager.init()

    Selectable.SelectableChoice.overrides.clear()
    Selectable.SelectableChoice.parameters.clear()

    yield


@pytest.fixture
def empty_training_data(tmpdir_factory):
    name = str(tmpdir_factory.mktemp('training').join('train.tif'))

    with TiffWriter(name, imagej=True) as tiff:
        image = quadratic_test_image(size=512, dtype=np.uint16)

        tiff.save(
            image,
            resolution=(1.0 / 0.065, 1.0 / 0.065),
            metadata=dict(
                unit='um', Overlays=[roi.tobytes() for roi in [some_imagejroi()]]
            ),
        )

    return name


@pytest.fixture
def funny_tiff_file(tmpdir_factory):
    name = str(tmpdir_factory.mktemp('training').join('train.tif'))

    with TiffWriter(name, imagej=True) as tiff:
        image = quadratic_test_image(size=512, dtype=np.uint16)

        tiff.save(
            image,
            resolution=(1.0 / 0.065, 1.0 / 0.065),
            metadata=dict(
                unit='um',
                Overlays=[
                    roi.tobytes()
                    for roi in [some_imagejroi(), trackmate_roi(), dualpoint_roi()]
                ],
            ),
        )

    return name


@pytest.fixture
def empty_training_image_directory(tmpdir_factory):
    base = tmpdir_factory.mktemp('training_directory')
    name = str(base)

    images_dir, labels_dir = base.mkdir('images'), base.mkdir('labels')

    image = quadratic_test_image()

    Image.fromarray(image, 'L').save(str(images_dir.join('image-000.png')))
    Image.fromarray(image, 'L').save(str(labels_dir.join('label-000.png')))

    return name


@pytest.fixture
def disable_cuda():
    @contextmanager
    def _disable_cuda(new_value=''):
        old_value = os.environ.get('CUDA_VISIBLE_DEVICES')
        os.environ['CUDA_VISIBLE_DEVICES'] = new_value
        yield

        if old_value:
            os.environ['CUDA_VISIBLE_DEVICES'] = old_value
        elif old_value is None:
            del os.environ['CUDA_VISIBLE_DEVICES']

    return _disable_cuda


@pytest.fixture
def tf_eager():
    import tensorflow as tf

    @contextmanager
    def _eager_change_context(eager):
        old_value = tf.config.experimental_functions_run_eagerly()
        tf.config.experimental_run_functions_eagerly(eager)
        yield
        tf.config.experimental_run_functions_eagerly(old_value)

    return _eager_change_context
