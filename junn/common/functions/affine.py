import tensorflow as tf
from math import pi

const_pi_180 = pi / 180.0


@tf.function
def radians(degrees):
    """
    Converts degrees to radians.

    :param degrees:
    :return:
    """
    return degrees * const_pi_180


@tf.function
def shape_to_h_w(shape):
    """
    Extracts height and width from a shape tuple which can have between 1 and 4 dimensions.
    :param shape:
    :return:
    """

    # TensorFlow does not like a 1 <= tf.rank(shape) <= 4 comparison!
    # noinspection PyChainedComparisons
    with tf.control_dependencies(
            [tf.Assert(tf.size(shape) >= 1 and tf.size(shape) <= 4, ["Invalid shape passed to shape_to_h_w"])]):
        if tf.size(shape) == 4:
            return shape[1], shape[2]
        else:
            return shape[0], shape[1]


@tf.function
# noinspection PyUnusedLocal
def tfm_identity(shape=(0, 0)):
    """
    Generates an identity matrix for use with the affine transformation matrices.
    :param shape:
    :return:
    """
    return tf.eye(3, dtype=tf.float32)


@tf.function
# noinspection PyUnusedLocal
def tfm_shift(x=0.0, y=0.0, shape=None):
    """
    Generates a shift affine transformation matrix.
    :param x:
    :param y:
    :param shape:
    :return:
    """
    return tf.convert_to_tensor([
        [1.0, 0.0, -y],
        [0.0, 1.0, -x],
        [0.0, 0.0, 1.0]
    ], dtype=tf.float32)


@tf.function
def tfm_reflect(x=0.0, y=0.0, shape=(0, 0)):
    """
    Generates a reflection affine transformation matrix.
    :param x:
    :param y:
    :param shape:
    :return:
    """
    w, h = shape_to_h_w(shape)
    x = 1.0 if x < 1.0 else -1.0
    y = 1.0 if y < 1.0 else -1.0

    if x > 0.0:
        w = 0.0
    else:
        w = tf.cast(w, tf.float32)

    if y > 0.0:
        h = 0.0
    else:
        h = tf.cast(h, tf.float32)

    return tf.convert_to_tensor([
        [y, 0.0, 0.0],
        [0.0, x, 0.0],
        [0.0, 0.0, 1.0]
    ], dtype=tf.float32) @ tfm_shift(w, h)


@tf.function
def tfm_rotate(angle=0.0, shape=(0, 0)):
    """
    Generates a rotation affine transformation matrix.
    :param angle:
    :param shape:
    :return:
    """
    w, h = shape_to_h_w(shape)

    rad = radians(angle)
    c_theta = tf.cos(rad)
    s_theta = tf.sin(rad)

    return (
            tfm_shift(x=-w / 2, y=-h / 2) @
            tf.convert_to_tensor([
                [c_theta, s_theta, 0.0],  # tf.cast(h, tf.float32) * 0.5],
                [-s_theta, c_theta, 0.0],  # -tf.cast(w, tf.float32) * 0.5],
                [0.0, 0.0, 1.0]
            ], dtype=tf.float32) @
            tfm_shift(x=w / 2, y=h / 2)
    )


@tf.function
def tfm_scale(xs=1.0, ys=None, shape=(0, 0)):
    """
    Generates a scaling affine transformation matrix.
    :param xs:
    :param ys:
    :param shape:
    :return:
    """
    if ys is None:
        ys = xs

    w, h = shape_to_h_w(shape)
    return (
            tfm_shift(x=-w / 2, y=-h / 2) @
            tf.convert_to_tensor([
                [1.0 / ys, 0.0, 0.0],
                [0.0, 1.0 / xs, 0.0],
                [0.0, 0.0, 1.0]
            ], dtype=tf.float32) @
            tfm_shift(x=w / 2, y=h / 2)
    )


@tf.function
def tfm_shear(x_angle=0.0, y_angle=0.0, shape=(0, 0)):
    """
    Generates a shear transformation matrix.
    :param x_angle:
    :param y_angle:
    :param shape:
    :return:
    """
    w, h = shape_to_h_w(shape)
    phi, psi = tf.tan(radians(x_angle)), tf.tan(radians(y_angle))

    return (
            tfm_shift(x=-w / 2, y=-h / 2) @
            tf.convert_to_tensor([
                [1.0, phi, 0.0],
                [psi, 1.0, 0.0],
                [0.0, 0.0, 1.0]
            ], dtype=tf.float32) @
            tfm_shift(x=w / 2, y=h / 2)
    )


@tf.function
def tfm_to_tf_transform(mat):
    """
    Truncates a affine transformation matrix to the parameters used by tf transform.
    :param mat:
    :return:
    """
    return tf.reshape(mat, (9,))[:8]
