import os
import sys
import logging


def get_gpu_memory_usages_megabytes():
    try:
        import py3nvml.py3nvml as nvml
    except ImportError:
        return [0.0]

    # noinspection PyUnresolvedReferences
    try:
        try:
            nvml.nvmlInit()
        except nvml.NVMLError_LibraryNotFound:
            if nvml.sys.platform[:3] == 'win':
                nvml.nvmlLib = nvml.CDLL(nvml.os.path.join(nvml.os.getenv('SystemRoot'), 'System32', 'nvml.dll'))
                nvml.nvmlInit()
            else:
                raise
    except nvml.NVMLError_LibraryNotFound:
        return [(0, 0)]

    device_count = nvml.nvmlDeviceGetCount()

    devices = list(range(device_count))
    device_handles = [nvml.nvmlDeviceGetHandleByIndex(device) for device in devices]

    memory_infos = [nvml.nvmlDeviceGetMemoryInfo(handle) for device, handle in zip(devices, device_handles)]
    megabytes = 1024**2
    memory_usages = [(int(memory_info.used) // megabytes, int(memory_info.total) // megabytes) for memory_info in memory_infos]

    nvml.nvmlShutdown()

    return memory_usages


def set_seed(seed):
    """
    Sets various seeds for random number generators, so their behavior becomes reproducible.
    (numpy, python random, TensorFlow)
    :param seed:
    :return:
    """

    import numpy as np
    import tensorflow as tf
    import random as rn

    # this might be cargo-cult ... it does not seem to affect an already running Python process
    os.environ['PYTHONHASHSEED'] = str(seed)

    np.random.seed(seed)

    rn.seed(seed)

    tf.random.set_seed(seed)


def configure_tensorflow(seed=None, windows_maximum_gpu_memory=0.75):
    """
    Configures TensorFlow. It is important that this function is called BEFORE first importing TF into a Python process.

    - Reduces TF's log-level (before) loading.
    - Sets all devices to dynamic memory allocation (growing instead of complete)
    - (On Windows) Sets overall maximum TensorFlow memory to windows_maximum_gpu_memory
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

    if sys.platform[:3] == 'win':
        memory_usages = get_gpu_memory_usages_megabytes()

        for device, (memory_used, memory_total) in zip(devices, memory_usages):
            tf.config.experimental.set_virtual_device_configuration(device, [
                tf.config.experimental.VirtualDeviceConfiguration(
                    memory_limit=int(memory_total * windows_maximum_gpu_memory)
                )
            ])

    tf_logger = logging.getLogger('tensorflow')
    for handler in tf_logger.handlers[:]:
        tf_logger.removeHandler(handler)

    os.environ['KERAS_BACKEND'] = 'tensorflow'
    os.environ['USE_TENSORFLOW_KERAS'] = '1'

    if seed is not None:
        set_seed(seed)

    # do we need this?
    # https://www.tensorflow.org/api_docs/python/tf/config/threading
