"""The training cli module."""
import logging
import os
import sys
from argparse import ArgumentParser
from typing import List, Optional, Tuple

import jsonpickle
import numpy as np

# isort: off
from junn_predict.common import autoconfigure_tensorflow
import tensorflow as tf

# isort: on
import tqdm
from junn_predict.common.cli import get_common_argparser_and_setup
from junn_predict.common.logging import DelayedFileLog
from tifffile import memmap as tifffile_memmap
from tunable import TunableManager
from tunable.tunablemanager import LoadTunablesAction

from ..common import distributed
from ..datasets.tfrecord import create_example, read_junn_tfrecord
from ..io.training import TrainingInput
from ..networks import NeuralNetwork
from ..networks.all import __networks__
from . import BatchSize, DatasetGenerationBenchmarkCount, ValidationSteps

# tqdm.tqdm()
# from ..common.stderr_redirection import StdErrLogRedirector
# StdErrLogRedirector.start_redirect()


autoconfigure_tensorflow = autoconfigure_tensorflow
__networks__ = __networks__

TRAINING_DATA_DIR = 'training_data'
# TRAINING_DATA_METADATA = 'metadata.json'
TRAINING_DATA_TFRECORD = 'data.tfrecord'

LOG_DATA_DIR = 'logs'
LOG_FILE_PATTERN = 'junn.{pid}.log'

log = logging.getLogger(__name__)


def pad_to(input_: np.ndarray, target_shape: Tuple[int, ...]) -> np.ndarray:
    """
    Pad the argument to have target_shape.

    :param input_:
    :param target_shape:
    :return:
    """
    pad = [[0, ts - is_] for is_, ts in zip(input_.shape, target_shape)]
    return np.pad(input_, pad)


def pad_all_arrays_to_largest(*args: np.ndarray) -> List[np.ndarray]:
    """
    Pad all arguments to equal the size of the largest argument.

    :param args:
    :return:
    """
    sizes = np.array([arg.shape for arg in args])
    sizes = sizes.max(axis=0, initial=0)
    return [pad_to(arg, sizes) for arg in args]


def output_training_data_and_benchmark(
    output_dataset: str,
    output_dataset_count: int,
    nn: NeuralNetwork,
    dataset: tf.data.Dataset,
) -> None:
    """
    Benchmark and write samples of prepared training data to a TIFF file.

    :param output_dataset: TIFF file to write to
    :param output_dataset_count: Count to write
    :param nn: NeuralNetwork instance with the input data pipeline
    :param dataset: Dataset providing training data
    :return:
    """
    if not distributed.is_rank_zero():
        return

    log.info("Not training, just writing output examples to %s", output_dataset)
    # prepared_dataset = dataset
    batch_size = BatchSize.value
    # batch_size = None
    prepared_dataset = nn.prepare_input(
        dataset, training=True, validation=False, batch=batch_size, skip_raw=True
    )

    if batch_size is None:
        batch_size = 1
    benchmark_sample_count = DatasetGenerationBenchmarkCount.value
    log.info("Benchmark for %d samples", benchmark_sample_count)

    intermediate = None

    for image, labels in prepared_dataset.take(1):
        intermediate = np.concatenate(
            pad_all_arrays_to_largest(
                image[0][np.newaxis, ...], labels[0][np.newaxis, ...]
            ),
            axis=0,
        )
        intermediate = intermediate[:, np.newaxis, :, :, :]
        intermediate = np.swapaxes(intermediate, 1, 4)

    for _, _ in tqdm.tqdm(
        prepared_dataset.take(benchmark_sample_count),
        unit=' samples',
        unit_scale=batch_size,
    ):
        pass

    # ImageJ TIFF Files have TZCYXS order
    tiff_output = tifffile_memmap(
        output_dataset,
        shape=(output_dataset_count * batch_size,) + intermediate.shape,
        dtype=intermediate.dtype,
        imagej=True,
    )
    n = 0
    for image_batch, labels_batch in prepared_dataset.take(output_dataset_count):
        for image, labels in zip(image_batch, labels_batch):
            output = np.concatenate(
                pad_all_arrays_to_largest(
                    image[np.newaxis, ...], labels[np.newaxis, ...]
                ),
                axis=0,
            )
            output = output[:, np.newaxis, :, :, :]
            output = np.swapaxes(output, 1, 4)

            tiff_output[n] = output
            n += 1


def set_delayed_logger_filename(model_directory: str) -> None:
    """
    Add the delayed logger with a model as target for log files.

    :param model_directory: Model path
    :return:
    """
    root_logger = logging.getLogger()
    for handler in root_logger.handlers:
        if isinstance(handler, DelayedFileLog):
            log_dir_name = os.path.join(model_directory, LOG_DATA_DIR)
            if not os.path.isdir(log_dir_name):
                os.mkdir(log_dir_name)
            log_file_name = os.path.join(
                log_dir_name, LOG_FILE_PATTERN.format(pid=os.getpid())
            )
            handler.setFilename(log_file_name)
            break


def add_argparser_arguments(parser: ArgumentParser) -> ArgumentParser:
    """
    Add the training-specific arguments to the argparser instance.

    :param parser:
    :return:
    """
    parser.add_argument(
        '--deterministic', dest='deterministic', default=False, action='store_true'
    )
    # parser.add_argument('--validation', dest='validation', type=str,
    # help="validation datasets", action='append')
    parser.add_argument(
        '--resume',
        dest='resume',
        help="resumes training process from model",
        default=False,
        action='store_true',
    )
    parser.add_argument(
        '--load',
        dest='load',
        type=str,
        help="load weights from other model",
        default=None,
    )
    parser.add_argument(
        '--embed',
        dest='embed',
        help="embed raw training data into model directory",
        default=False,
        action='store_true',
    )
    parser.add_argument(
        '--output-dataset',
        dest='output_dataset',
        type=str,
        help="output dataset",
        default=None,
    )
    parser.add_argument(
        '--output-dataset-count',
        dest='output_dataset_count',
        type=int,
        help="output dataset count",
        default=1024,
    )
    parser.add_argument('--shell', dest='shell', type=str, action='append', default=[])
    parser.add_argument(
        '--disable-device-pinning',
        dest='device_pinning',
        action='store_false',
        default=True,
    )
    return parser


def main(args: Optional[List[str]] = None) -> None:
    """
    JUNN training main entrypoint.

    :param args: Optional command line arguments, if ``None``, ``sys.argv`` will be used.
    :return:
    """
    if args is None:
        args = sys.argv[1:]

    # StdErrLogRedirector.stop_redirect()
    args, parser = get_common_argparser_and_setup(args=args)

    add_argparser_arguments(parser)

    args = parser.parse_args(args=args)

    # StdErrLogRedirector.start_redirect()
    # StdErrLogRedirector.start_processing()

    if args.deterministic:
        log.info(
            "Deterministic training is currently not working. "
            "Setting up as much determinism as possible, tho."
        )
        os.environ['TF_DETERMINISTIC_OPS'] = '1'
        os.environ['HOROVOD_FUSION_THRESHOLD'] = '0'

    distributed.init(device_pinning=args.device_pinning)

    if not args.resume and not args.input:
        raise RuntimeError("No input specified.")

    if args.resume:
        log.info("Resuming from %s", args.model)
        with open(
            os.path.join(
                args.model,
                tf.saved_model.ASSETS_DIRECTORY,
                NeuralNetwork.ASSET_ARGUMENTS,
            )
        ) as fp:
            args = jsonpickle.loads(fp.read())

        # TODO nicer, but needs upgrade in tunable
        tunable_asset_file = os.path.join(
            args.model, tf.saved_model.ASSETS_DIRECTORY, NeuralNetwork.ASSET_TUNABLES
        )
        LoadTunablesAction(None, None)(None, None, tunable_asset_file)

    # experimental things would go here

    nn = NeuralNetwork()

    ti = TrainingInput()
    log.info("Loading dataset %s(%r)", type(ti).__name__, args.input)
    dynamic_dataset = ti.get(*args.input, mode=nn.input_mode)

    if not args.embed:
        dataset = dynamic_dataset
    else:
        if embedded_dataset_exists(args.model):
            log.info(
                "Restoring training dataset from %s",
                get_embedded_dataset_filename(args.model),
            )
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
        return output_training_data_and_benchmark(
            args.output_dataset, args.output_dataset_count, nn, dataset
        )

    if 'before' in args.shell:
        assert not distributed.is_running_distributed()
        import IPython

        IPython.embed()

    nn.update_asset(nn.ASSET_TUNABLES, TunableManager.get_serialization('json'))
    nn.update_asset(nn.ASSET_ARGUMENTS, jsonpickle.dumps(args))

    nn.easy_setup(model_path=args.model, load_from_path=args.load)

    dataset = nn.prepare_input(
        dataset, training=True, validation=False, batch=BatchSize.value, skip_raw=True
    )

    # TODO: Re-enable the validation code
    validation = None
    if validation:
        validation = nn.prepare_input(
            validation,
            training=True,
            validation=True,
            batch=ValidationSteps.value,
            skip_raw=True,
        )

    # warm up cudnn
    tf.nn.conv2d(tf.zeros((1, 32, 32, 1)), tf.zeros((2, 2, 1, 1)), 1, 'SAME')

    log.info("Starting training ...")
    if distributed.is_rank_zero():
        log.info(
            "You can investigate metrics using tensorboard, run:\n"
            "python -m tensorboard.main --logdir \"%s\"",
            os.path.abspath(args.model),
        )

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


def get_embedded_dataset_filename(model_name: str) -> str:
    """
    Return the path of an embedded dataset in a model.

    :param model_name: Model path
    :return: path to the embedded dataset as string
    """
    return os.path.join(model_name, TRAINING_DATA_DIR, TRAINING_DATA_TFRECORD)


def embedded_dataset_exists(model_name: str) -> bool:
    """
    Check whether a model contains an embedded dataset.

    :param model_name: Model path
    :return: True or False
    """
    return os.path.isfile(get_embedded_dataset_filename(model_name))


def load_embedded_dataset(model_name: str) -> tf.data.Dataset:
    """
    Load an embedded dataset from a model.

    :param model_name: Model path
    :return: the dataset
    """
    return read_junn_tfrecord(get_embedded_dataset_filename(model_name))


def write_embedded_dataset(model_name: str, dataset: tf.data.Dataset) -> None:
    """
    Embed a dataset into a model.

    :param model_name: Model path
    :param dataset: Dataset
    :return: None
    """
    training_data_directory, metadata_file = (  # noqa: F841
        os.path.join(model_name, TRAINING_DATA_DIR),
        None,  # os.path.join(args.model, TRAINING_DATA_DIR, TRAINING_DATA_METADATA),
    )

    tfrecord_file = get_embedded_dataset_filename(model_name)
    log.info("This is rank zero ... creating embedded dataset.")

    if not os.path.isdir(model_name) or not os.path.isdir(training_data_directory):
        os.mkdir(model_name)
        os.mkdir(training_data_directory)

    tfr_options = tf.io.TFRecordOptions(compression_type='')

    if not os.path.isfile(tfrecord_file):
        with tf.io.TFRecordWriter(tfrecord_file, tfr_options) as writer:
            for x, y in dataset:
                writer.write(create_example(x, y))
