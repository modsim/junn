"""Additional losses suitable for segmentation tasks."""
import tensorflow as tf
from tensorflow.keras.losses import *


@tf.function
def epsilon():
    """
    Epsilon of the current compute hardware.

    :return: epsilon as a floating point type
    """
    from tensorflow.keras.backend import epsilon as _epsilon

    return _epsilon()


@tf.function
def dice_index_direct(y_true, y_pred):
    """
    Directly calculate the Dice index.

    :param y_true: ground truth
    :param y_pred: prediction
    :return: Dice index
    """
    y_true, y_pred = flatten_and_clip(y_true), flatten_and_clip(y_pred)
    intersection = tf.reduce_sum(y_true * y_pred)
    return (2.0 * intersection) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + epsilon()
    )


@tf.function
def dice_loss(y_true, y_pred):
    """
    Calculate the Dice loss, i.e. the negative Dice index.

    :param y_true: ground truth
    :param y_pred: prediction
    :return: Dice loss
    """
    return -dice_index_direct(y_true, y_pred)


@tf.function
def dice_loss_unclipped(y_true, y_pred):
    """
    Calculate the Dice loss, i.e. the negative Dice index without clipping.

    :param y_true: ground truth
    :param y_pred: prediction
    :return: Dice loss
    """
    y_true, y_pred = tf.reshape(y_true, [-1]), tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true * y_pred)
    return -(
        (2.0 * intersection)
        / (tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + epsilon())
    )


@tf.function
def dice_index_weighted(y_true, y_pred, y_weight):
    """
    Calculate the Dice index with weighting applied.

    :param y_true: ground truth
    :param y_pred: prediction
    :param y_weight: weights
    :return: weighted Dice index
    """
    y_true, y_pred = flatten_and_clip(y_true), flatten_and_clip(y_pred)
    y_weight = tf.reshape(y_weight, [-1])
    intersection = tf.reduce_sum(y_true * y_pred * y_weight) / 1.0
    return (2.0 * intersection) / (
        tf.reduce_sum(y_true) + tf.reduce_sum(y_pred) + epsilon()
    )


@tf.function
def dice_loss_weighted(y_true, y_pred, y_weight):
    """
    Calculate the weighted Dice loss, i.e. negative Dice index.

    :param y_true: ground truth
    :param y_pred: prediction
    :param y_weight: weights
    :return: weighted Dice loss
    """
    return -dice_index_weighted(y_true, y_pred, y_weight)


@tf.function
def weighted_loss(y_true, y_pred, y_weight):
    """
    Weighted loss function.

    :param y_true: ground truth
    :param y_pred: prediction
    :param y_weight: weights
    :return: weighted loss
    """
    return -(
        tf.reduce_sum(
            (y_true * y_pred * y_weight) + ((1 - y_true) * (1 - y_pred) * y_weight)
        )
        / tf.reduce_sum(y_weight)
    )


@tf.function
def flatten_and_clip(values):
    """
    Flatten and clip a tensor.

    :param values: values
    :return: flattened and clipped to 0-1 values
    """
    return tf.clip_by_value(tf.reshape(values, [-1]), 0, 1)


def mixin_flatten_and_clip(what):
    """
    Mixin to assure a function gets a flattened and clipped tensor.

    :param what: function to be modified
    :return: modified function
    """

    def _inner(y_true, y_pred):
        return what(*flatten_and_clip(y_true), flatten_and_clip(y_pred))

    # peculiar behavior
    if isinstance(what.__name__, str):
        _inner.__name__ = what.__name__

    if _inner.__name__[0] == '_':
        _inner.__name__ = _inner.__name__[1:]

    return tf.function()(_inner)


@tf.function
def _true_positive(y_true, y_pred):
    return tf.reduce_sum(y_true * y_pred)


@tf.function
def _true_negative(y_true, y_pred):
    return tf.reduce_sum((1 - y_true) * (1 - y_pred))


@tf.function
def _false_positive(y_true, y_pred):
    return tf.reduce_sum((1 - y_true) * y_pred)


@tf.function
def _false_negative(y_true, y_pred):
    return tf.reduce_sum(y_true * (1 - y_pred))


true_positive = mixin_flatten_and_clip(_true_positive)
true_negative = mixin_flatten_and_clip(_true_negative)
false_positive = mixin_flatten_and_clip(_false_positive)
false_negative = mixin_flatten_and_clip(_false_negative)


@tf.function
def tp_tn_fp_fn_precision_recall(y_true, y_pred):
    """
    Calculate a set of simple metrics in one go.

    True positives, true negatives, false positives,
    false negatives, precision as well as recall.

    :param y_true: ground truth
    :param y_pred: prediction
    :return: tuple of metrics (tp, tn, fp, fn, precision, recall)
    """
    y_true, y_pred = flatten_and_clip(y_true), flatten_and_clip(y_pred)
    tp, tn = _true_positive(y_true, y_pred), _true_negative(y_true, y_pred)
    fp, fn = _false_positive(y_true, y_pred), _false_negative(y_true, y_pred)

    precision_ = tp / (tp + fp + epsilon())
    recall_ = tp / (tp + fn + epsilon())

    return tp, tn, fp, fn, precision_, recall_


@tf.function
def accuracy(y_true, y_pred):
    """
    Calculate the accuracy.

    :param y_true: ground truth
    :param y_pred: prediction
    :return: accuracy
    """
    tp, tn, fp, fn, precision_, recall_ = tp_tn_fp_fn_precision_recall(y_true, y_pred)

    return (tp + tn) / (tp + tn + fp + fn + epsilon())


@tf.function
def precision(y_true, y_pred):
    """
    Calculate the precision.

    :param y_true: ground truth
    :param y_pred: prediction
    :return: precision
    """
    tp, tn, fp, fn, precision_, recall_ = tp_tn_fp_fn_precision_recall(y_true, y_pred)
    return precision_


@tf.function
def recall(y_true, y_pred):
    """
    Calculate the recall.

    :param y_true: ground truth
    :param y_pred: prediction
    :return: recall
    """
    tp, tn, fp, fn, precision_, recall_ = tp_tn_fp_fn_precision_recall(y_true, y_pred)
    return recall_


@tf.function
def f_score(y_true, y_pred):
    """
    Calculate the F score.

    :param y_true: ground truth
    :param y_pred: prediction
    :return: F-score
    """
    tp, tn, fp, fn, precision_, recall_ = tp_tn_fp_fn_precision_recall(y_true, y_pred)
    return 2 * (precision_ * recall_) / (precision_ + recall_ + epsilon())


@tf.function
def dice_index(y_true, y_pred):
    """
    Calculate the Dice index.

    :param y_true: ground truth
    :param y_pred: prediction
    :return: the Dice index
    """
    tp, tn, fp, fn, precision_, recall_ = tp_tn_fp_fn_precision_recall(y_true, y_pred)
    return (2 * tp) / (tp + fp + tp + fn + epsilon())


@tf.function
def jaccard_index(y_true, y_pred):
    """
    Calculate the Jaccard index.

    :param y_true: ground truth
    :param y_pred: prediction
    :return: the Jaccard index
    """
    tp, tn, fp, fn, precision_, recall_ = tp_tn_fp_fn_precision_recall(y_true, y_pred)
    return tp / (tp + fp + epsilon())


def generate_tversky_loss(alpha=0.5, beta=0.5):
    """
    Generate a Tversky loss function.

    :param alpha: alpha
    :param beta: beta
    :return: a Tversky loss function with fixed parameters
    """

    def _inner_tversky_loss(y_true, y_pred):
        return -tversky_index(y_true, y_pred, alpha=alpha, beta=beta)

    _inner_tversky_loss.__name__ = (
        'tversky_loss_%.2f_%.2f'
        % (
            alpha,
            beta,
        )
    ).replace('.', '_')

    return tf.function()(_inner_tversky_loss)


@tf.function
def tversky_index(y_true, y_pred, alpha=0.5, beta=0.5):
    """
    Calculate the Tversky index.

    :param y_true: ground truth
    :param y_pred: prediction
    :param alpha: alpha
    :param beta: beta
    :return: the Tversky index
    """
    tp, tn, fp, fn, precision_, recall_ = tp_tn_fp_fn_precision_recall(y_true, y_pred)
    return tp / (tp + alpha * fp + beta * fn + epsilon())


@tf.function
def nan_loss(y_true, y_pred):
    """
    Return always NaN for debugging and testing purposes.

    :param y_true: ground truth
    :param y_pred: prediction
    :return: all NaN
    """
    tp, tn, fp, fn, precision_, recall_ = tp_tn_fp_fn_precision_recall(y_true, y_pred)
    return tp * tn * fp * fn * precision_ * recall_ * tf.sqrt(-1.0)
