from tensorflow.keras.callbacks import Callback
import numpy as np
from tifffile import imread as tifffile_imread

import tensorflow as tf


class TensorBoardSegmentationCallback(Callback):
    """
    The callback will run the model on a set of test images to produce test predictions (e.g. segmentations) observable
    via TensorBoard.
    """
    def __init__(self, tensorboard_callback, prediction_callback, input_file_name=None, every_epoch=1):
        """
        Constructor.
        :param tensorboard_callback:
        :param prediction_callback:
        :param input_file_name:
        :param every_epoch:
        """
        super(self.__class__, self).__init__()  # py23 compatible
        # we don't check if backend is tensorflow, because we assume that
        # if we get a tensorboard callback passed, backend must be tensorflow
        self.tb = tensorboard_callback
        self.prediction_callback = prediction_callback

        self.every_epoch = every_epoch

        self.input_images = self._prepare_input_data(input_file_name)

    @staticmethod
    def _prepare_input_data(input_file_name):
        tiff_data = tifffile_imread(input_file_name)
        if tiff_data.ndim == 3:
            if tiff_data.shape[2] == 3:
                # maybe RGB
                return [tiff_data]
            else:
                return [plane[..., np.newaxis] for plane in tiff_data]
        elif tiff_data.ndim == 2:
            return [tiff_data[..., np.newaxis]]

    @staticmethod
    def _prepare_result(result):
        if result.ndim == 2:
            result = result[np.newaxis, ..., np.newaxis]
        elif result.ndim == 3:
            result = result[np.newaxis, ...]
        return result

    # noinspection PyUnusedLocal
    def on_epoch_end(self, epoch, logs=None):
        """
        Callback run on every epoch end.
        :param epoch:
        :param logs:
        :return:
        """
        if self.input_images is None:
            return

        if (epoch % self.every_epoch) != 0:
            return

        # noinspection PyProtectedMember
        with self.tb._get_writer(self.tb._train_run_name).as_default():
            segmentation_str = "segmentation_%%0%dd" % len(str(len(self.input_images)))
            for n, image in enumerate(self.input_images):
                result = self.prediction_callback(image.astype(np.float32))

                result = self._prepare_result(result)

                tf.summary.image(segmentation_str % n, result, step=epoch)
