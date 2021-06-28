"""Helper functions for distributed training."""
import logging
import os

import tensorflow as tf

try:
    import horovod.tensorflow.keras as hvd
except ImportError:
    hvd = None


log = logging.getLogger(__name__)

_running_distributed = False
_initialized = False


def is_running_distributed():
    """
    Return True if JUNN is running distributed and Horovod is initialized.

    :return:
    """
    return _running_distributed


def _init_horovod():
    global _running_distributed

    log.info("Enabling Horovod ...")
    hvd.init()
    log.info(
        "Horovod: global rank: %r, global size: %r, local rank: %r, local size: %r",
        hvd.rank(),
        hvd.size(),
        hvd.local_rank(),
        hvd.local_size(),
    )

    if hvd.size() > 1:
        _running_distributed = True


def pin_devices():
    """
    Pin each GPU to a worker.

    :return:
    """
    devices = tf.config.get_visible_devices('GPU')

    if not devices:
        log.warning(
            "Horovod: No GPUs available. Will not pin anything. "
            "(Is this really what you wanted?)"
        )
    else:
        target_device = devices[hvd.local_rank()]

        log.info(
            "Horovod: Device pinning enabled, setting this worker "
            "(local/global %d/%d) to use %r",
            hvd.local_rank(),
            hvd.rank(),
            target_device,
        )
        tf.config.set_visible_devices([target_device], 'GPU')


def init(device_pinning=True):
    """
    Initialize Horovod if present.

    :param device_pinning: Whether GPUs should be pinned.
    :return:
    """
    global _running_distributed, _initialized

    if _initialized:
        return

    if hvd:
        _init_horovod()

        if device_pinning:
            pin_devices()
    else:
        if 'OMPI_COMM_WORLD_SIZE' in os.environ:  # OMPI_COMM_WORLD_RANK
            log.warning(
                "You have apparently started JUNN with mpirun, "
                "but Horovod is not available. Check your configuration!"
            )

    _initialized = True


def local_rank():
    """
    Return the local rank (on the physical machine).

    :return:
    """
    if hvd:
        return hvd.local_rank()
    else:
        return 0


def rank():
    """
    Return the global rank.

    This is the rank within all running instances on possibly multiple machines.

    :return:
    """
    if hvd:
        return hvd.rank()
    else:
        return 0


def size():
    """
    Return the count of running instances.

    :return:
    """
    if hvd:
        return hvd.size()
    else:
        return 1


def is_rank_zero():
    """
    Return whether the current process is rank zero, i.e. the main process.

    :return:
    """
    return rank() == 0


def get_callbacks():
    """
    Get Keras callbacks.

    If Horovod is present, this will be the BroadCastGlobalVariables callback.

    :return:
    """
    callbacks = []
    if hvd:
        callbacks.append(hvd.callbacks.BroadcastGlobalVariablesCallback(0))
    return callbacks


def wrap_optimizer(optimizer):
    """
    Wrap the Keras optimizer.

    If Horovod is present, with DistributedOptimizer.

    :param optimizer:
    :return:
    """
    if hvd:
        return hvd.DistributedOptimizer(optimizer)
    else:
        return optimizer


def barrier(name):
    """
    Create a barrier which will synchronize all processes.

    Can be used if the worker processes need to wait for the main process.

    :param name:
    :return:
    """
    if hvd:
        # apparently this returns a tensor!
        # I guess, in eager mode, it's fine as is, in graph execution mode, it might be
        # necessary to attach it to some session?

        return hvd.allreduce([0.0], name=name)
    else:
        return tf.convert_to_tensor([0.0], dtype=tf.float64)
