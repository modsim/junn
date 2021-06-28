"""Additional helper functions mostly on tensors, written as TensorFlow-tf.functions."""
import tensorflow as tf


@tf.function
def tf_function_nop(*input_):
    """
    No operation, yield the input parameters.

    :param input_: Parameter(s) to pass thru
    :return: input_
    """
    return input_


@tf.function
def tf_function_one_arg_nop(input_):
    """
    No operation, take and yield exactly one parameter.

    :param input_: Parameter to pass thru
    :return: input_
    """
    return input_


@tf.function
def gaussian2d(size=(32, 32), sigma=0.5):
    """
    Generate a Gaussian kernel (not normalized).

    :param size: k x m size of the returned kernel
    :param sigma: standard deviation of the returned Gaussian
    :return: A tensor with the Gaussian kernel
    """
    x, y = tf.meshgrid(tf.linspace(-1.0, 1.0, size[0]), tf.linspace(-1.0, 1.0, size[1]))
    d_squared = x * x + y * y
    two_times_sigma_squared = 2.0 * (sigma ** 2.0)
    return tf.exp(-d_squared / two_times_sigma_squared)


@tf.function
def sigma_to_k(sigma):
    """
    Calculate a suitable kernel size for a given standard deviation.

    :param sigma: Standard deviation
    :return: Size
    """
    return int(sigma * 2 + 1)


@tf.function
def get_gaussian_kernel(sigma=0.5, k=None):
    """
    Generate a Gaussian kernel (normalized).

    :param sigma: Standard deviation
    :param k: Desired kernel size, if None it will be calculated from sigma
    :return: The Gaussian kernel, normalized
    """
    if k is None:
        k = sigma_to_k(sigma)

    kernel = gaussian2d(size=(k, k), sigma=sigma)[..., tf.newaxis, tf.newaxis]
    kernel = kernel / tf.math.reduce_sum(kernel)
    return kernel


# @tf.function
def get_gaussian_kernels(lower_sigma=0.8, upper_sigma=2.5, steps=25):
    """
    Generate a stack of Gaussian kernels (from lower_sigma to upper_sigma in steps).

    :param lower_sigma: Lower boundary of standard deviations
    :param upper_sigma: Upper boundary of standard deviations
    :param steps: Number of steps
    :return: A three dimensional tensor with the stacked Gaussian kernels
    """
    return tf.stack(
        [
            get_gaussian_kernel(s, k=sigma_to_k(upper_sigma))
            for s in tf.linspace(lower_sigma, upper_sigma, steps)
        ]
    )


@tf.function
def convolve(image, kernel):
    """
    Perform a 2D convolution using the tf.nn.conv2d TF op.

    :param image: Image to convolve
    :param kernel: Kernel to convolve
    :return: Convolved image
    """
    return tf.nn.conv2d(
        image[tf.newaxis], kernel, strides=[1, 1, 1, 1], padding='SAME'
    )[0]


@tf.function
def pad_to(input_, target_block_size, mode='REFLECT'):
    """
    Pad a tensor to match target_block_size.

    :param input_: Input tensor
    :param target_block_size: Desired target block size
    :param mode: Mode, passed to tf.pad
    :return: The padded input_
    """
    image_shape = tf.shape(input_)

    # paddings = tf.convert_to_tensor(
    #     [[0,
    #       tf.cond(target_block_size[0] >= image_shape[0],
    #       lambda: target_block_size[0] - image_shape[0], lambda: 0)],
    #      [0,
    #       tf.cond(target_block_size[1] >= image_shape[1],
    #       lambda: target_block_size[1] - image_shape[1], lambda: 0)],
    #      [0,0]
    # ])

    paddings = tf.convert_to_tensor(
        [
            [
                0,
                target_block_size[0] - image_shape[0]
                if target_block_size[0] >= image_shape[0]
                else 0,
            ],
            [
                0,
                target_block_size[1] - image_shape[1]
                if target_block_size[1] >= image_shape[1]
                else 0,
            ],
            [0, 0],
        ]
    )

    padded_input = tf.pad(input_, paddings=paddings, mode=mode)

    return padded_input
