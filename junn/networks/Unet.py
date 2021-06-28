"""Unet module containing the U-Net network class for use within JUNN."""
from . import Input, Model, warn_unused_arguments
from .functional.unet_layer import unet
from .mixins.losses import DiceLoss
from .mixins.multichannel import MultichannelHandling
from .mixins.preprocessing import PerImageStandardizationPreprocessingMixin
from .mixins.tile_based_network import TilebasedNetwork
from .mixins.weighted_loss import WeightedLoss


class Unet(
    WeightedLoss,
    MultichannelHandling,
    DiceLoss,
    PerImageStandardizationPreprocessingMixin,
    TilebasedNetwork,
):
    """The U-Net Network class."""

    def get_model(self):  # noqa: D102
        parameters = dict(
            # defaults
        )
        parameters.update(self.parameters)

        warn_unused_arguments(parameters, unet, self.log)

        self.log.info(
            "Building a %s using parameters %r"
            % (
                self.__class__.__name__,
                parameters,
            )
        )

        input_ = Input(self.tile_size)
        output_ = unet(input_, **parameters)
        # output_ = tf.clip_by_value(output_, 0.0, 1.0)

        return Model(inputs=[input_], outputs=[output_])
