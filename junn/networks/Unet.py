import tensorflow as tf
from .functional.unet_layer import unet
from . import NeuralNetwork, Input, Model
from .mixins import DeepImageJCompatibilityMixin
from .mixins.preprocessing import PerImageStandardizationPreprocessingMixin
from .mixins.tile_based_training import TilebasedTrainingMixin
from .mixins.prediction import PredictionSignatureMixin
from .mixins.tile_based_prediction import TilebasedPredictionMixin
from ..common.losses import *


class TilebasedNetwork(DeepImageJCompatibilityMixin, TilebasedTrainingMixin, TilebasedPredictionMixin,
                       PredictionSignatureMixin, NeuralNetwork, NeuralNetwork.Virtual):
    def init(self, **kwargs):
        # noinspection PyAttributeOutsideInit
        self.tile_size = (128, 128, 1,)

        if 'tile_size' in kwargs:
            # noinspection PyAttributeOutsideInit
            self.tile_size = (kwargs['tile_size'], kwargs['tile_size'], 1)


class DiceLoss:
    # noinspection PyMethodMayBeStatic
    def get_loss(self):
        return dice_loss


class Unet(DiceLoss, PerImageStandardizationPreprocessingMixin, TilebasedNetwork):
    def get_model(self):
        parameters = dict(
            # defaults
        )
        parameters.update(self.parameters)

        self.log.info("Building a %s using parameters %r" % (self.__class__.__name__, parameters,))

        input_ = Input(self.tile_size)
        output_ = unet(input_, **parameters)
        # output_ = tf.clip_by_value(output_, 0.0, 1.0)

        return Model(inputs=[input_], outputs=[output_])
