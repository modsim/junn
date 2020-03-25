import tensorflow as tf

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Lambda

from .Unet import TilebasedNetwork, DiceLoss
from .mixins.preprocessing import PerImageStandardizationPreprocessingMixin

from .functional.msd import msd_net


class MSD(DiceLoss, PerImageStandardizationPreprocessingMixin, TilebasedNetwork):
    def get_model(self):
        kwargs = dict()
        kwargs.update(self.kwargs)

        print("Building using %r" % (kwargs,))

        inputs = Input(self.tile_size)

        outputs = msd_net(inputs, **kwargs)

        outputs = tf.clip_by_value(outputs, 0.0, 1.0)

        return Model(inputs=[inputs], outputs=[outputs])
