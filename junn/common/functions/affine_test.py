import math

import pytest
import tensorflow as tf

from junn.common.functions.affine import *

# @pytest.fixture(autouse=True)
# def run_eagerly():
#     print("run it!")
#     #before = tf.config.experimental_functions_run_eagerly()
#     #tf.config.experimental_run_functions_eagerly(True)
#     yield
#     #tf.config.experimental_run_functions_eagerly(before)
#     print("ran it.")


def test_radians():
    assert 2 * math.pi == radians(360.0)

    for n in range(0, 360):
        assert radians(n) == math.radians(n)


def test_shape_to_h_w(tf_eager):
    with tf_eager(True):
        assert tuple(shape_to_h_w([800, 600])) == (800, 600)

        assert tuple(shape_to_h_w([800, 600, 3])) == (800, 600)

        assert tuple(shape_to_h_w([15, 800, 600, 3])) == (800, 600)

        from tensorflow.python.framework.errors_impl import InvalidArgumentError

        with pytest.raises(InvalidArgumentError):
            assert tuple(shape_to_h_w([999, 15, 800, 600, 3])) == (800, 600)


def test_tfm_identity():
    assert (
        ([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] == tfm_identity())
        .numpy()
        .all()
    )


def test_tfm_shift():
    assert (
        ([[1.0, 0.0, -5.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] == tfm_shift(y=5))
        .numpy()
        .all()
    )

    assert (
        ([[1.0, 0.0, 0.0], [0.0, 1.0, -5.0], [0.0, 0.0, 1.0]] == tfm_shift(x=5))
        .numpy()
        .all()
    )

    assert (tfm_shift(x=5.0) @ tfm_shift(x=-5.0) == tfm_identity()).numpy().all()
    assert (tfm_shift(y=5.0) @ tfm_shift(y=-5.0) == tfm_identity()).numpy().all()
    assert (
        (tfm_shift(x=5.0, y=5.0) @ tfm_shift(x=-5.0, y=-5) == tfm_identity())
        .numpy()
        .all()
    )


# noinspection DuplicatedCode
def test_tfm_reflect(tf_eager):
    with tf_eager(True):
        assert (tfm_reflect(x=5.0) @ tfm_reflect(x=5.0) == tfm_identity()).numpy().all()
        assert (tfm_reflect(x=5) @ tfm_reflect(x=5) == tfm_identity()).numpy().all()
        assert (tfm_reflect(x=1.0) @ tfm_reflect(x=1.0) == tfm_identity()).numpy().all()

        assert (tfm_reflect(y=5.0) @ tfm_reflect(y=5.0) == tfm_identity()).numpy().all()
        assert (tfm_reflect(y=5) @ tfm_reflect(y=5) == tfm_identity()).numpy().all()
        assert (tfm_reflect(y=1.0) @ tfm_reflect(y=1.0) == tfm_identity()).numpy().all()


def test_tfm_rotate(tf_eager):
    with tf_eager(True):
        result = tfm_rotate(45.0)

    assert True  # TODO


def test_tfm_scale(tf_eager):
    with tf_eager(True):
        result = tfm_scale(1.5)

    assert True  # TODO


def test_tfm_shear(tf_eager):
    with tf_eager(True):
        result = tfm_shear(x_angle=10.0, y_angle=10.0)

    assert True  # TODO


def test_tfm_to_tf_transform():
    assert (
        (
            [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]
            == tfm_to_tf_transform(tfm_identity())
        )
        .numpy()
        .all()
    )
