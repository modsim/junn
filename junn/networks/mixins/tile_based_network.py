from ...io.training import ModeTile
from .. import NeuralNetwork
from . import (
    DeepImageJCompatibilityMixin,
    PredictionSignatureMixin,
    TilebasedPredictionMixin,
    TilebasedTrainingMixin,
)


class TilebasedNetwork(
    DeepImageJCompatibilityMixin,
    TilebasedTrainingMixin,
    TilebasedPredictionMixin,
    PredictionSignatureMixin,
    NeuralNetwork,
    NeuralNetwork.Virtual,
):
    input_mode = ModeTile

    def init(self, **kwargs):
        # noinspection PyAttributeOutsideInit
        self.tile_size = (
            128,
            128,
            1,
        )

        if 'tile_size' in kwargs:
            # noinspection PyAttributeOutsideInit
            self.tile_size = (kwargs['tile_size'], kwargs['tile_size'], 1)
