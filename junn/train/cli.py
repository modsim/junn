import tqdm
# tqdm.tqdm()
# from ..common.stderr_redirection import StdErrLogRedirector
# StdErrLogRedirector.start_redirect()

from ..common.configure_tensorflow import configure_tensorflow
configure_tensorflow(seed=0)

from ..common.seeding import set_seed
set_seed(0)

import os
import sys
import logging

import tqdm
import jsonpickle
import numpy as np
import tensorflow as tf

from tifffile import memmap as tifffile_memmap

from tunable import TunableManager
from tunable.tunablemanager import LoadTunablesAction

from . import BatchSize, ValidationSteps

from ..common import distributed

from junn_predict.common.cli import get_common_argparser_and_setup
from junn_predict.common.logging import DelayedFileLog

from ..datasets.tfrecord import create_example, read_junn_tfrecord
from ..io.training import TrainingInput

from ..networks import NeuralNetwork
from ..networks.all import *

TRAINING_DATA_DIR = 'training_data'
# TRAINING_DATA_METADATA = 'metadata.json'
TRAINING_DATA_TFRECORD = 'data.tfrecord'

LOG_DATA_DIR = 'logs'
LOG_FILE_PATTERN = 'junn.{pid}.log'

log = logging.getLogger(__name__)


def pad_to(input_, target_shape):
    pad = [[0, ts-is_] for is_, ts in zip(input_.shape, target_shape)]
    return np.pad(input_, pad)


def pad_all_arrays_to_largest(*args):
    sizes = np.array([arg.shape for arg in args])
    sizes = sizes.max(axis=0)
    return [pad_to(arg, sizes) for arg in args]


def output_training_data_and_benchmark(output_dataset, output_dataset_count, nn, dataset):
    if not distributed.is_rank_zero():
        return

    log.info("Not training, just writing output examples to %s", output_dataset)
    prepared_dataset = dataset
    batch_size = BatchSize.value
    # batch_size = None
    prepared_dataset = nn.prepare_input(dataset, training=True, validation=False, batch=batch_size, skip_raw=True)

    if batch_size is None:
        batch_size = 1
    benchmark_sample_count = 1000
    log.info("Benchmark for %d samples", benchmark_sample_count)
    for image, labels in prepared_dataset.take(1):
        intermediate = np.concatenate(pad_all_arrays_to_largest(image[0][np.newaxis, ...], labels[0][np.newaxis, ...]), axis=0)
        intermediate = intermediate[:, np.newaxis, :, :, :]
        intermediate = np.swapaxes(intermediate, 1, 4)

    for image, labels in tqdm.tqdm(prepared_dataset.take(benchmark_sample_count), unit=' samples',
                                   unit_scale=batch_size):
        pass

    # ImageJ TIFF Files have TZCYXS order
    tiff_output = tifffile_memmap(output_dataset,
                                  shape=(output_dataset_count * batch_size,) + intermediate.shape,
                                  dtype=intermediate.dtype, imagej=True)
    n = 0
    for image_batch, labels_batch in prepared_dataset.take(output_dataset_count):
        for image, labels in zip(image_batch, labels_batch):
            output = np.concatenate(pad_all_arrays_to_largest(image[np.newaxis, ...], labels[np.newaxis, ...]), axis=0)
            output = output[:, np.newaxis, :, :, :]
            output = np.swapaxes(output, 1, 4)

            tiff_output[n] = output
            n += 1


def set_delayed_logger_filename(model_directory):
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, DelayedFileLog):
            log_dir_name = os.path.join(model_directory, LOG_DATA_DIR)
            if not os.path.isdir(log_dir_name):
                os.mkdir(log_dir_name)
            log_file_name = os.path.join(log_dir_name, LOG_FILE_PATTERN.format(pid=os.getpid()))
            handler.setFilename(log_file_name)
            break


def main(args=None):
    if args is None:
        args = sys.argv[1:]

    # StdErrLogRedirector.stop_redirect()
    args, parser = get_common_argparser_and_setup(args=args)

    parser.add_argument('--deterministic', dest='deterministic', default=False, action='store_true')
    # parser.add_argument('--validation', dest='validation', type=str, help="validation datasets", action='append')
    parser.add_argument('--resume', dest='resume',
                        help="resumes training process from model", default=False, action='store_true')
    parser.add_argument('--load', dest='load',
                        type=str, help="load weights from other model", default=None)
    parser.add_argument('--embed', dest='embed',
                        help="embed raw training data into model directory", default=False, action='store_true')
    parser.add_argument('--output-dataset', dest='output_dataset',
                        type=str, help="output dataset", default=None)
    parser.add_argument('--output-dataset-count', dest='output_dataset_count',
                        type=int, help="output dataset count", default=1024)
    parser.add_argument('--shell', dest='shell',
                        type=str, action='append', default=[])
    parser.add_argument('--disable-device-pinning', dest='device_pinning',
                        action='store_false', default=True)

    args = parser.parse_args(args=args)

    # StdErrLogRedirector.start_redirect()
    # StdErrLogRedirector.start_processing()

    if args.deterministic:
        log.info("Deterministic training is currently not working. Setting up as much determinism as possible, tho.")
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['HOROVOD_FUSION_THRESHOLD'] = '0'

    distributed.init(device_pinning=args.device_pinning)

    if not args.resume and not args.input:
        raise RuntimeError("No input specified.")

    if args.resume:
        log.info("Resuming from %s", args.model)
        with open(os.path.join(args.model, tf.saved_model.ASSETS_DIRECTORY, NeuralNetwork.ASSET_ARGUMENTS)) as fp:
            args = jsonpickle.loads(fp.read())

        # TODO nicer, but needs upgrade in tunable
        LoadTunablesAction(None, None)(None, None, os.path.join(args.model, tf.saved_model.ASSETS_DIRECTORY, NeuralNetwork.ASSET_TUNABLES))

    # experimental things would go here

    nn = NeuralNetwork()

    ti = TrainingInput()
    log.info("Loading dataset %s(%r)", type(ti).__name__, args.input)
    dynamic_dataset = ti.get(*args.input, mode=nn.input_mode)

    if not args.embed:
        dataset = dynamic_dataset
    else:
        if embedded_dataset_exists(args.model):
            log.info("Restoring training dataset from %s", get_embedded_dataset_filename(args.model))
            dataset = load_embedded_dataset(args.model)
        else:
            log.info("writing or waiting for data")

            if distributed.is_rank_zero():
                write_embedded_dataset(args.model, dynamic_dataset)
            distributed.barrier("write_embedded_samples")

            dataset = load_embedded_dataset(args.model)

            log.info("done")

    # much faster to do this here, before cache
    dataset = nn.apply_raw_fn(dataset)

    # cache the dataset in RAM
    dataset = dataset.cache()

    # ... and repeat indefinitely
    dataset = dataset.repeat()

    dataset = dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    log.info("TensorFlow eager status: %r", tf.executing_eagerly())

    if args.output_dataset:
        return output_training_data_and_benchmark(args.output_dataset, args.output_dataset_count, nn, dataset)

    if 'before' in args.shell:
        assert not distributed.is_running_distributed()
        import IPython
        IPython.embed()

    nn.update_asset(nn.ASSET_TUNABLES, TunableManager.get_serialization('json'))
    nn.update_asset(nn.ASSET_ARGUMENTS, jsonpickle.dumps(args))

    nn.easy_setup(model_path=args.model, load_from_path=args.load)

    dataset = nn.prepare_input(dataset,
                               training=True,
                               validation=False,
                               batch=BatchSize.value,
                               skip_raw=True)

    validation = None
    if validation:
        validation = nn.prepare_input(validation,
                                      training=True,
                                      validation=True,
                                      batch=ValidationSteps.value,
                                      skip_raw=True)

    # warm up cudnn
    tf.nn.conv2d(tf.zeros((1, 32, 32, 1)), tf.zeros((2, 2, 1, 1)), 1, 'SAME')

    log.info("Starting training ...")
    if distributed.is_rank_zero():
        log.info("You can investigate metrics using tensorboard, run:\npython -m tensorboard.main --logdir \"%s\"",
                 os.path.abspath(args.model))

    set_delayed_logger_filename(args.model)

    # StdErrLogRedirector.stop_redirect()
    # StdErrLogRedirector.stop_processing()

    try:
        nn.train(dataset, validation=validation)
    except KeyboardInterrupt:
        log.info("Ctrl-C pressed, saving the model ... ")
    finally:
        nn.save_model()
        log.info("Saving done.")

    if 'after' in args.shell:
        assert not distributed.is_running_distributed()
        import IPython
        IPython.embed()


def get_embedded_dataset_filename(model_name):
    return os.path.join(model_name, TRAINING_DATA_DIR, TRAINING_DATA_TFRECORD)


def embedded_dataset_exists(model_name):
    return os.path.isfile(get_embedded_dataset_filename(model_name))


def load_embedded_dataset(model_name):
    return read_junn_tfrecord(get_embedded_dataset_filename(model_name))


def write_embedded_dataset(model_name, dataset):
    training_data_directory, metadata_file = (
        os.path.join(model_name, TRAINING_DATA_DIR),
        None,  # os.path.join(args.model, TRAINING_DATA_DIR, TRAINING_DATA_METADATA),
    )

    tfrecord_file = get_embedded_dataset_filename(model_name)
    log.info("This is rank zero ... creating embedded dataset.")

    if not os.path.isdir(model_name) or not os.path.isdir(training_data_directory):
        os.mkdir(model_name)
        os.mkdir(training_data_directory)

    tfr_options = tf.io.TFRecordOptions(
        compression_type=''
    )

    if not os.path.isfile(tfrecord_file):
        with tf.io.TFRecordWriter(tfrecord_file, tfr_options) as writer:
            for x, y in dataset:
                writer.write(create_example(x, y))


experimental_things = """
# profiler needs either root access or GPU driver set to allow regular users to access performance counters
# https://developer.nvidia.com/nvidia-development-tools-solutions-ERR_NVGPUCTRPERM-permission-issue-performance-counters
# TODO add an argument to allow profiling!
# from tensorflow.python.eager.profiler import start_profiler_server
# start_profiler_server(6009)

# # tf debugging looks interesting, but appears removed in TF2?
# if True:
#     from tensorflow.python import debug as tf_debug
#     from tensorflow.keras import backend as K
#     K.set_session(tf_debug.TensorBoardDebugWrapperSession(tf.Session(), 'localhost:7777'))

# so this is actually making it slower
# noinspection PyUnreachableCode
if False:
    tf.config.optimizer.set_jit(True)  # its lacking some ops where I don't really know at which point they are added?

# lets try float16
# noinspection PyUnreachableCode
if False:
    # this currently doesn't work. although it should.
    loss_scale = 'dynamic'
    policy = tf.keras.mixed_precision.experimental.Policy('mixed_float16', loss_scale=loss_scale)
    tf.keras.mixed_precision.experimental.set_policy(policy)
"""
