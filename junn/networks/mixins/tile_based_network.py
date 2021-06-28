"""Tile-based network helper."""
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
    """Mixin to run a model in a tile-based manner over arbitrary sized images."""

    input_mode = ModeTile

    def init(self, **kwargs):  # noqa: D102
        # noinspection PyAttributeOutsideInit
        self.tile_size = (
            128,
            128,
            1,
        )

        if 'tile_size' in kwargs:
            # noinspection PyAttributeOutsideInit
            self.tile_size = (kwargs['tile_size'], kwargs['tile_size'], 1)
