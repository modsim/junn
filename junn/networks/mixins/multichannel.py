"""Mixin to handle multichannel images."""
import tensorflow as tf

from ...io import REGION_FOREGROUND
from .tile_based_network import TilebasedNetwork


class MultichannelHandling(TilebasedNetwork, TilebasedNetwork.Virtual):
    """Mixin to handle multichannel images."""

    def get_training_fn(self, validation: bool = False):  # noqa: D102
        parent_fn = super().get_training_fn(validation=validation)

        @tf.function
        def _remap_labels(image, labels):
            labels = tf.where(
                labels == REGION_FOREGROUND,
                tf.cast(1, labels.dtype),
                tf.cast(0, labels.dtype),
            )
            return image, labels

        output_channels = (
            self.parameters['output_channels']
            if 'output_channels' in self.parameters
            else 1
        )

        if output_channels > 1:

            @tf.function
            def _inner(image, labels):
                image, labels = parent_fn(image, labels)
                labels = tf.cast(
                    tf.one_hot(
                        tf.cast(labels[:, :, 0], tf.int32), depth=output_channels
                    ),
                    tf.float32,
                )
                return image, labels

        else:

            @tf.function
            def _inner(image, labels):
                image, labels = _remap_labels(image, labels)
                image, labels = parent_fn(image, labels)
                return image, labels

        return _inner

    def get_prediction_fn(self):  # noqa: D102
        parent_fn = super().get_prediction_fn()

        output_channels = (
            self.parameters['output_channels']
            if 'output_channels' in self.parameters
            else 1
        )
        rebinarize_output = (
            self.parameters['rebinarize_output']
            if 'rebinarize_output' in self.parameters
            else True
        )

        if output_channels > 1:

            @tf.function(
                input_signature=[tf.TensorSpec([None, None, None], dtype=tf.float32)]
            )
            def _inner(image):
                prediction = parent_fn(image)
                prediction = tf.argmax(prediction, axis=-1)
                if rebinarize_output:
                    prediction = tf.where(
                        tf.equal(prediction, REGION_FOREGROUND), 1.0, 0.0
                    )

                prediction = tf.cast(prediction[..., tf.newaxis], tf.float32)

                return prediction

            return _inner
        else:
            return parent_fn

    def get_loss(self):  # noqa: D102
        output_channels = (
            self.parameters['output_channels']
            if 'output_channels' in self.parameters
            else 1
        )

        if output_channels > 1:
            from ...common.losses import dice_loss  # , dice_loss_unclipped

            return dice_loss  # dice_loss_unclipped
        else:
            return super().get_loss()
