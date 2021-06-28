"""MSD module containing the MSD network class for use within JUNN."""
import tensorflow as tf

from . import Input, Model, warn_unused_arguments
from .functional.msd import msd_net
from .mixins.losses import DiceLoss
from .mixins.preprocessing import PerImageStandardizationPreprocessingMixin
from .mixins.tile_based_network import TilebasedNetwork


class MSD(DiceLoss, PerImageStandardizationPreprocessingMixin, TilebasedNetwork):
    """MSD Network class."""

    def get_model(self):  # noqa: D102
        parameters = dict(
            # defaults
        )
        parameters.update(self.parameters)

        warn_unused_arguments(parameters, msd_net, self.log)

        self.log.info(
            "Building a %s using parameters %r"
            % (
                self.__class__.__name__,
                parameters,
            )
        )

        inputs = Input(self.tile_size)

        if 'tile_size' in parameters:
            del parameters['tile_size']

        outputs = msd_net(inputs, tile_size=self.tile_size, **parameters)

        outputs = tf.clip_by_value(outputs, 0.0, 1.0)

        return Model(inputs=[inputs], outputs=[outputs])
