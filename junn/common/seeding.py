import os


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
