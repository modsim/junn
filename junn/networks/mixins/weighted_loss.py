import tensorflow as tf

from ...common.functions import convolve, get_gaussian_kernel
from .tile_based_network import TilebasedNetwork


@tf.function
def calculate_weightmap(image, sigma=3.5, overlap_ratio=2.5, inner_ratio=0.75, empty=0.25):
    blurred = convolve(image, get_gaussian_kernel(sigma))
    weightmap = (1 - image) * overlap_ratio * blurred + inner_ratio * image + (empty * ((image - 1) / -1))
    return weightmap


@tf.function
def split_weight_off(raw_y_true, y_pred):
    size = tf.shape(raw_y_true)[1]

    y_true = raw_y_true[:, :size // 2, :, :]
    y_weight = raw_y_true[:, size // 2:, :, :]

    return y_true, y_pred, y_weight


class WeightedLoss(TilebasedNetwork):
    def get_training_fn(self, validation: bool = False):
        parent_fn = super().get_training_fn(validation=validation)

        weighted_loss = self.parameters['weighted_loss'] if 'weighted_loss' in self.parameters else False

        if weighted_loss:

            @tf.function
            def _inner(image, labels):
                image, labels = parent_fn(image, labels)
                weights = calculate_weightmap(labels)

                labels_and_weights = tf.concat([labels, weights], axis=0)

                return image, labels_and_weights

            return _inner
        else:
            return parent_fn

    def get_loss(self):
        weighted_loss = self.parameters['weighted_loss'] if 'weighted_loss' in self.parameters else False

        if weighted_loss:
            from ...common.losses import dice_index_weighted

            @tf.function
            def _inner(raw_y_true, y_pred):
                y_true, y_pred, y_weight = split_weight_off(raw_y_true, y_pred)
                return -dice_index_weighted(y_true, y_pred, y_weight)

            return _inner
        else:
            return super().get_loss()

    def get_metrics(self):
        weighted_loss = self.parameters['weighted_loss'] if 'weighted_loss' in self.parameters else False

        metrics = super().get_metrics()

        def _process(fun):
            @tf.function
            def _inner(raw_y_true, y_pred):
                y_true, y_pred, _ = split_weight_off(raw_y_true, y_pred)
                return fun(y_true, y_pred)

            _inner.__name__ = fun.__name__

            return _inner

        if weighted_loss:
            return [_process(metric) for metric in metrics]
        else:
            return metrics
