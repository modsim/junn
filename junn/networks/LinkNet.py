import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Lambda

from .Unet import TilebasedNetwork, DiceLoss
from .mixins.preprocessing import PerImageStandardizationPreprocessingMixin

from .functional.link_net import link_net


class LinkNet(DiceLoss, PerImageStandardizationPreprocessingMixin, TilebasedNetwork):
    def get_model(self):
        parameters = dict(
            # defaults
        )
        parameters.update(self.parameters)

        self.log.info("Building a %s using parameters %r" % (self.__class__.__name__, parameters,))

        inputs = Input(self.tile_size)

        outputs = link_net(inputs, **parameters)
        outputs = tf.clip_by_value(outputs, 0.0, 1.0)

        return Model(inputs=[inputs], outputs=[outputs])
