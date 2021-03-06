"""The network package, containing general infrastructure and helpers for networks."""
import logging
import os
import tempfile
import time
from typing import Any, Dict, Optional

import jsonpickle
import jsonpickle.ext.numpy as jsonpickle_numpy
import tensorflow as tf
import tensorflow.python.util.deprecation as deprecation
from junn_predict.common.tensorflow_addons import try_load_tensorflow_addons

try_load_tensorflow_addons()
from keras_nvidia_statistics import NvidiaDeviceStatistics

# noinspection PyPep8Naming
from tensorflow.keras import backend as K
from tensorflow.keras import callbacks, optimizers
from tensorflow.keras.callbacks import LambdaCallback, TensorBoard
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow_addons.callbacks import TQDMProgressBar
from tunable import Selectable

from ..common import distributed
from ..common.callbacks import TensorBoardSegmentationCallback
from ..common.functions import tf_function_nop, tf_function_one_arg_nop
from ..train import (
    Decay,
    Epochs,
    LearningRate,
    Metrics,
    Momentum,
    Optimizer,
    PreprocessingMapParallel,
    Profile,
    StepsPerEpoch,
    TensorBoardHistogramFrequency,
    TensorBoardSegmentationDataset,
    TensorBoardSegmentationEpochs,
    TensorBoardWriteGradients,
    ValidationSteps,
)
from .util import (
    format_size,
    get_default_signature,
    get_weight_counts,
    numpy_to_scalar,
    warn_unused_arguments,
)

jsonpickle_numpy.register_handlers()

__all__ = 'NeuralNetwork', 'Input', 'Model', 'warn_unused_arguments'


# noinspection PyMethodMayBeStatic
class NeuralNetwork(Selectable):
    """The NeuralNetwork selectable base class."""

    PREDICTION_SIGNATURE = 'predict'

    ASSET_PREFIX = '_junn_'

    ASSET_HISTORY = ASSET_PREFIX + 'keras_history.json'
    ASSET_TUNABLES = ASSET_PREFIX + 'tunables.json'
    ASSET_ARGUMENTS = ASSET_PREFIX + 'arguments.json'

    input_mode = None

    def __init__(self, **kwargs):
        """
        Initialize a NeuralNetwork class.

        :param kwargs:
        """
        self.log = logging.getLogger(__name__)
        self.model: Optional[tf.keras.models.Model] = None

        self.assets: Dict[str, Any] = {}
        self._asset_temp_dir = None

        self.prediction_fn: Any = None
        self.signatures: Dict[str, Any] = {}
        self.signature_default_key: Optional[str] = None
        self.model_path: Optional[str] = None

        self.callbacks: Optional[list] = None

        self.current_epoch: int = 0

        self._statistics_about_weights: Dict[str, Any] = {}

        self.history = None

        self.parameters = kwargs.copy()

        self.init(**kwargs)

    def init(self, **kwargs):
        """
        Inner init function to be overridden by networks.

        :param kwargs:
        :return:
        """
        pass

    def easy_setup(self, model_path=None, load_from_path=None):
        """
        Perform an easy setup of the NeuralNetwork class.

        Loads a model from the specified path.

        :param model_path:
        :param load_from_path:
        :return:
        """
        self.setup_model()

        self.try_load(model_path)

        if load_from_path:
            self.try_load(load_from_path)

        self.compile()
        self.set_model_path(model_path)
        self.save_model()

    def setup_model(self, print_statistics=True):
        """
        Create and setup the model.

        :param print_statistics:
        :return:
        """
        self.model = self.get_model()
        self.model.junn_assets = self.assets
        self.prediction_fn = self.get_prediction_fn()

        self.signatures = self.get_signatures()
        self.signatures[self.PREDICTION_SIGNATURE] = self.prediction_fn

        default_key, default_signature = get_default_signature(self.model)
        self.signature_default_key = default_key
        self.signatures[default_key] = default_signature

        self.calculate_statistics()
        if print_statistics:
            self.print_model_statistics()

    def calculate_statistics(self):
        """
        Calculate model statistics and store them in the model instance.

        :return:
        """
        stats = get_weight_counts(self.model)

        stats['float'] = K.floatx()
        stats['float_size'] = dict(float16=2, float32=4, float64=8)[K.floatx()]

        self._statistics_about_weights = stats

    def print_model_statistics(self):
        """
        Print the statistics stored in the model instance.

        :return:
        """
        stats = self._statistics_about_weights

        self.log.info(
            "Network parameter count, trainable:     % 12d", stats['trainable']
        )
        self.log.info(
            "Network parameter count, non-trainable: % 12d", stats['non_trainable']
        )
        self.log.info(
            "Network parameter size:                 % 19s (type %s)",
            format_size(
                (stats['trainable'] + stats['non_trainable']) * stats['float_size']
            ),
            stats['float'],
        )

    def try_load(self, path):
        """
        Try loading the model specified.

        :param path:
        :return:
        """
        if not distributed.is_rank_zero():  # only rank zero may load ...
            return

        checkpoint = os.path.join(path, 'cp.ckpt')
        checkpoint_index = os.path.join(path, 'cp.ckpt.index')
        asset_path = os.path.join(path, 'assets')

        # model = tf.saved_model.load(path)

        if (
            tf.saved_model.contains_saved_model(path)
            and os.path.isdir(path)
            and os.path.isfile(checkpoint_index)
        ):
            # expect partial is necessary to prevent warnings due to assets
            # not being used by the vanilla keras model
            self.model.load_weights(checkpoint).expect_partial()
            for asset in tf.io.gfile.listdir(asset_path):
                self.update_asset(
                    asset, tf.io.read_file(os.path.join(asset_path, asset))
                )

            # restore history
            history = jsonpickle.loads(self.get_asset(self.ASSET_HISTORY))

            try:
                self.current_epoch = (
                    int(max(history['epoch'])) + 1
                )  # the current one will be the /next/
            except ValueError:
                self.log.warning(
                    "Tried to load existing saved model, "
                    "but history data was apparently corrupted."
                )
                self.current_epoch = 1
            self.history = history
        else:
            pass

    def get_asset(self, name):
        """
        Get an asset by name.

        :param name:
        :return:
        """
        assert name in self.assets
        return tf.io.read_file(self.assets[name]).numpy()

    def update_asset(self, name, contents=b''):
        """
        Update an asset by name.

        :param name:
        :param contents:
        :return:
        """
        if not distributed.is_rank_zero():  # only rank zero may save
            return

        if self._asset_temp_dir is None:
            self._asset_temp_dir = tempfile.mkdtemp()

        complete_path = os.path.join(self._asset_temp_dir, name)

        asset = tf.saved_model.Asset(complete_path)

        self.assets[name] = asset

        tf.io.write_file(asset, contents)

    def save_model(self):
        """
        Save the model.

        :return:
        """
        if not distributed.is_rank_zero():  # only rank zero may save
            return

        history = getattr(self.model, 'history', None)

        if history:
            # a history object contains a lot of unpicklable things,
            # let's copy over some interesting parts

            if self.history:
                mini_history = dict(
                    epoch=self.history['epoch'] + history.epoch,
                    history={
                        k: self.history['history'][k] + list(map(numpy_to_scalar, v))
                        for k, v in history.history.items()
                    },
                )
            else:
                mini_history = dict(
                    epoch=history.epoch,
                    history={
                        k: list(map(numpy_to_scalar, v))
                        for k, v in history.history.items()
                    },
                )

            self.update_asset(self.ASSET_HISTORY, jsonpickle.dumps(mini_history))

        # re-assure connection from local asset variable to model
        # (i.e. soon-to-be serialized) asset variable
        self.model.junn_assets = self.assets

        with deprecation.silence():  # prevents
            # resource_variable_ops.py:1781: calling BaseResourceVariable.__init__
            # (from tensorflow.python.ops.resource_variable_ops) with constraint
            # is deprecated and will be removed in a future version.
            tf.saved_model.save(self.model, self.model_path, signatures=self.signatures)

        export_tflite = False

        if export_tflite:
            # does not work (yet)
            functions = [
                fun.get_concrete_function() for fun in self.signatures.values()
            ]
            # so, only the default
            functions = [
                self.signatures[self.signature_default_key].get_concrete_function()
            ]

            tflite_converter = tf.lite.TFLiteConverter.from_concrete_functions(
                functions
            )

            tflite_converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]
            # tf.lite.Optimize.OPTIMIZE_FOR_SIZE, tf.lite.Optimize.OPTIMIZE_FOR_LATENCY

            tflite_model = tflite_converter.convert()

            with open(os.path.join(self.model_path, 'model.tflite'), 'wb+') as fp:
                fp.write(tflite_model)

    def set_model_path(self, path):
        """
        Set the model path.

        :param path:
        :return:
        """
        self.model_path = path

    def get_loss(self):
        """
        Get the loss function.

        To be overridden by a network.

        :return:
        """
        raise RuntimeError("Not implemented.")

    # noinspection PyMethodMayBeStatic
    def get_optimizer(self):
        """
        Get the optimizer.

        To be overridden by a network.

        :return:
        """
        lr = LearningRate.value * distributed.size()
        return distributed.wrap_optimizer(
            dict(
                adam=lambda: optimizers.Adam(lr=lr, decay=Decay.value),
                sgd=lambda: optimizers.SGD(
                    lr=lr, momentum=Momentum.value, decay=Decay.value
                ),
                rmsprop=lambda: optimizers.RMSprop(lr=lr, decay=Decay.value),
            )[Optimizer.value.lower()]()
        )

    def get_metrics(self):
        """
        Get the metrics to track.

        :return:
        """
        return Metrics.get_list()

    def compile(self):
        """
        Compile the model.

        :return:
        """
        self.model.compile(
            optimizer=self.get_optimizer(),
            loss=self.get_loss(),
            metrics=self.get_metrics(),
            experimental_run_tf_function=False,  # ?!
        )

    def get_model(self):
        """
        Get the model.

        To be overridden by the network.

        :return:
        """
        raise RuntimeError("Not implemented.")

    def get_callbacks(self):
        """
        Get the callbacks to be used as Keras callbacks.

        :return:
        """
        the_callbacks = distributed.get_callbacks()

        # all callbacks which are for _all_ workers must go here

        if not distributed.is_rank_zero():
            return the_callbacks

        # now add callbacks which are only for rank zero

        the_callbacks.append(NvidiaDeviceStatistics(output=self.log.info))

        the_callbacks.append(
            LambdaCallback(
                on_epoch_end=lambda epoch, logs: logs.update(
                    dict(wallclock=float(time.time()))
                )
            )
        )

        stats = self._statistics_about_weights

        the_callbacks.append(
            LambdaCallback(
                on_epoch_end=lambda epoch, logs: logs.update(
                    dict(
                        parameter_count_trainable=stats['trainable'],
                        parameter_count_non_trainable=stats['non_trainable'],
                    )
                )
            )
        )

        the_callbacks.append(TQDMProgressBar())

        the_callbacks.append(
            callbacks.ModelCheckpoint(
                self.model_path + '/cp.ckpt', save_weights_only=True, monitor='loss'
            ),
        )

        tensorboard_callback = TensorBoard(
            log_dir=self.model_path,
            profile_batch=Profile.value,
            write_grads=TensorBoardWriteGradients.value,
            histogram_freq=TensorBoardHistogramFrequency.value,
            write_images=False,
        )

        if TensorBoardSegmentationDataset.value:
            tbsc = TensorBoardSegmentationCallback(
                tensorboard_callback,
                prediction_callback=lambda image: self.predict(image),
                input_file_name=TensorBoardSegmentationDataset.value,
                every_epoch=TensorBoardSegmentationEpochs.value,
                metrics=Metrics.get_list(),
            )

            the_callbacks.append(tbsc)

        # must be last to have access to all log items
        the_callbacks.append(tensorboard_callback)

        return the_callbacks

    def train(self, dataset, validation=None):
        """
        Run the training.

        :param dataset:
        :param validation:
        :return:
        """
        if self.callbacks is None:
            self.callbacks = self.get_callbacks()

        self.log.info("starting current training from epoch %d", self.current_epoch)

        self.model.fit(
            x=dataset,
            epochs=Epochs.value // distributed.size(),
            steps_per_epoch=StepsPerEpoch.value,
            validation_data=validation,
            validation_steps=ValidationSteps.value if validation else None,
            initial_epoch=self.current_epoch,
            callbacks=self.callbacks,
            verbose=False,
        )

    def prepare_input(
        self,
        dataset,
        training: bool = True,
        validation: bool = False,
        skip_raw: bool = False,
        batch: int = None,
    ):
        """
        Prepare the input.

        :param dataset:
        :param training:
        :param validation:
        :param skip_raw:
        :param batch:
        :return:
        """
        if not skip_raw:
            dataset = self.apply_raw_fn(dataset)

        if training:
            dataset = self.apply_training_fn(dataset, validation=validation)

        if batch:
            dataset = dataset.batch(batch)

        if training:
            dataset = self.apply_training_batch_fn(dataset, validation=validation)

        return dataset

    def get_raw_fn(self):
        """
        Return the raw data function.

        :return:
        """
        return tf_function_one_arg_nop

    def apply_raw_fn(self, dataset: tf.data.Dataset):
        """
        Apply the raw data function to the dataset.

        :param dataset:
        :return:
        """
        fun = self.get_raw_fn()

        @tf.function
        def only_x(*args):
            result = (fun(args[0]),) + args[1:]
            return result

        return dataset.map(
            only_x
        )  # , num_parallel_calls=PreprocessingMapParallel.prepared())

    # noinspection PyUnusedLocal
    def get_training_fn(self, validation: bool = False):
        """
        Get the training data function.

        :param validation:
        :return:
        """
        return tf_function_nop

    def apply_training_fn(self, dataset: tf.data.Dataset, validation: bool = False):
        """
        Apply the training function to a dataset.

        :param dataset:
        :param validation:
        :return:
        """
        return dataset.map(
            self.get_training_fn(validation=validation),
            num_parallel_calls=PreprocessingMapParallel.prepared(),
        )

    # noinspection PyUnusedLocal
    def get_training_batch_fn(self, validation: bool = False):
        """
        Get the training batch function.

        :param validation:
        :return:
        """
        return tf_function_nop

    def apply_training_batch_fn(
        self, dataset: tf.data.Dataset, validation: bool = False
    ):
        """
        Apply the training batch function.

        :param dataset:
        :param validation:
        :return:
        """
        return dataset.map(
            self.get_training_batch_fn(validation=validation),
            num_parallel_calls=PreprocessingMapParallel.prepared(),
        )

    def get_signatures(self):
        """
        Get the signatures of the model.

        :return:
        """
        return {}

    def get_prediction_fn(self):
        """
        Get the prediction function of the model.

        :return:
        """

        @tf.function
        def _predict(input_):
            return self.model(input_)

        return _predict

    def predict(self, image):
        """
        Predict an image using the prediction function.

        :param image:
        :return:
        """
        return self.prediction_fn(image)
