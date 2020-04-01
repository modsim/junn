import tensorflow as tf

from . import Input, Model, warn_unused_arguments
from .mixins.preprocessing import PerImageStandardizationPreprocessingMixin
from .mixins.tile_based_network import TilebasedNetwork
from .mixins.losses import DiceLoss

from .functional.link_net import link_net


class LinkNet(DiceLoss, PerImageStandardizationPreprocessingMixin, TilebasedNetwork):
    def get_model(self):
        parameters = dict(
            # defaults
        )
        parameters.update(self.parameters)

        warn_unused_arguments(parameters, link_net, self.log)

        self.log.info("Building a %s using parameters %r" % (self.__class__.__name__, parameters,))

        inputs = Input(self.tile_size)

        outputs = link_net(inputs, **parameters)
        outputs = tf.clip_by_value(outputs, 0.0, 1.0)

        return Model(inputs=[inputs], outputs=[outputs])
