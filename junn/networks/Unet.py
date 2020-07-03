from . import Input, Model, warn_unused_arguments
from .mixins.multichannel import MultichannelHandling
from .mixins.preprocessing import PerImageStandardizationPreprocessingMixin
from .mixins.tile_based_network import TilebasedNetwork
from .mixins.losses import DiceLoss

from .functional.unet_layer import unet


class Unet(MultichannelHandling, DiceLoss, PerImageStandardizationPreprocessingMixin, TilebasedNetwork):
    def get_model(self):
        parameters = dict(
            # defaults
        )
        parameters.update(self.parameters)

        warn_unused_arguments(parameters, unet, self.log)

        self.log.info("Building a %s using parameters %r" % (self.__class__.__name__, parameters,))

        input_ = Input(self.tile_size)
        output_ = unet(input_, **parameters)
        # output_ = tf.clip_by_value(output_, 0.0, 1.0)

        return Model(inputs=[input_], outputs=[output_])
