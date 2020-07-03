from tensorflow.keras.callbacks import Callback
import time
import numpy as np

import tensorflow as tf

from junn_predict.common.timed import Timed
from ..io.tiffmasks import tiff_masks


class TensorBoardSegmentationCallback(Callback):
    """
    The callback will run the model on a set of test images to produce test predictions (e.g. segmentations) observable
    via TensorBoard.
    """
    def __init__(self, tensorboard_callback, prediction_callback, input_file_name=None, every_epoch=1, metrics=None):
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

        self.input_images_masks = list(tiff_masks(input_file_name,
                                                  background=0.0,
                                                  foreground=1.0,
                                                  border=0.0,
                                                  ))

        self.metrics = metrics if metrics else []

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
        if not self.input_images_masks:
            return

        if (epoch % self.every_epoch) != 0:
            return

        # noinspection PyProtectedMember
        with self.tb._get_writer(self.tb._train_run_name).as_default():
            segmentation_str = "segmentation_%%0%dd" % len(str(len(self.input_images_masks)))
            for n, (image, mask) in enumerate(self.input_images_masks):

                segmentation_name = segmentation_str % n

                image = image[..., np.newaxis].astype(np.float32)
                mask = mask[..., np.newaxis].astype(np.float32)

                with Timed() as runtime:
                    raw_result = np.array(self.prediction_callback(image))

                pixels_per_second = image.size / float(runtime)

                result = self._prepare_result(raw_result)

                tf.summary.image(segmentation_name, result, step=epoch)

                logs['%s_%s' % (segmentation_name, 'pixels_per_second')] = pixels_per_second

                for metric in self.metrics:
                    logs['%s_%s' % (segmentation_name, metric.__name__)] = float(metric(mask, raw_result))
