import os
import logging

from .seeding import set_seed


def configure_tensorflow(seed=None):
    """
    Configures TensorFlow. It is important that this function is called BEFORE first importing TF into a Python process.

    - Reduces TF's log-level (before) loading.
    - Sets all devices to dynamic memory allocation (growing instead of complete)
    - Removes the tensorflow log adapter from the global logger
    - Sets Keras to use TensorFlow (and sets USE_TENSORFLOW_KERAS to 1, custom environment variable)

    :param seed: Optional. If set, will be passed to set_seed()
    :return:
    """
    tf_log = 'TF_CPP_MIN_LOG_LEVEL'
    tf_log_level = '2'
    if tf_log not in os.environ:
        # pass
        os.environ[tf_log] = tf_log_level
        # I tried catching TF messages by redirecting stderr, but it is a mess, since any
        # important error messages (stderr!) get swallowed as well.
        # TF needs better logging handling, see https://github.com/tensorflow/tensorflow/issues/37390

    import tensorflow as tf

    devices = tf.config.list_physical_devices('GPU')

    for device in devices:
        tf.config.experimental.set_memory_growth(device, True)

    tf_logger = logging.getLogger('tensorflow')
    for handler in tf_logger.handlers[:]:
        tf_logger.removeHandler(handler)

    os.environ['KERAS_BACKEND'] = 'tensorflow'
    os.environ['USE_TENSORFLOW_KERAS'] = '1'

    if seed is not None:
        set_seed(seed)

    # do we need this?
    # https://www.tensorflow.org/api_docs/python/tf/config/threading
