from tensorflow.keras.losses import *
from tensorflow.keras import backend as K


def dice_index_direct(y_true, y_pred):
    y_true, y_pred = flatten_and_clip(y_true, y_pred)
    intersection = K.sum(y_true * y_pred)
    return (2.0 * intersection) / (K.sum(y_true) + K.sum(y_pred) + K.epsilon())


def dice_loss(y_true, y_pred):
    return -dice_index_direct(y_true, y_pred)


def dice_loss_unclipped(y_true, y_pred):
    y_true, y_pred = K.flatten(y_true), K.flatten(y_pred)
    intersection = K.sum(y_true * y_pred)
    return -((2.0 * intersection) / (K.sum(y_true) + K.sum(y_pred) + K.epsilon()))


def dice_index_weighted(y_true, y_pred, y_weight):
    y_true, y_pred = flatten_and_clip(y_true, y_pred)
    y_weight = K.flatten(y_weight)
    intersection = K.sum(y_true * y_pred * y_weight) / 1.0
    return (2.0 * intersection) / (K.sum(y_true) + K.sum(y_pred) + K.epsilon())


def weight_remover(fun):
    def _inner(y_true, y_pred):
        y_true = y_true[:, 0:1]
        y_pred = y_pred[:, 0:1]
        return fun(y_true, y_pred)
    _inner.__name__ = fun.__name__  # cheating
    return _inner


def dice_loss_weighted(y_true, y_pred):
    y_weight = y_true[:, 1:2]

    y_true = y_true[:, 0:1]
    y_pred = y_pred[:, 0:1]

    return -dice_index_weighted(y_true, y_pred, y_weight)


def weighted_loss(y_true, y_pred):
    y_weight = K.flatten(y_true[:, 1:2])
    y_true, y_pred = flatten_and_clip(y_true[:, 0:1], y_pred[:, 0:1])

    return -(
            K.sum((y_true * y_pred * y_weight) +
                  ((1 - y_true) * (1 - y_pred) * y_weight)) / K.sum(y_weight)
    )


def flatten_and_clip(y_true, y_pred):
    y_true = K.clip(K.flatten(y_true), 0, 1)
    y_pred = K.clip(K.flatten(y_pred), 0, 1)
    return y_true, y_pred


def mixin_flatten_and_clip(what):
    def _inner(y_true, y_pred):
        return what(*flatten_and_clip(y_true, y_pred))
    _inner.__name__ = what.__name__  # cheating

    if _inner.__name__[0] == '_':
        _inner.__name__ = _inner.__name__[1:]

    return _inner


def _true_positive(y_true, y_pred):
    return K.sum(y_true * y_pred)


def _true_negative(y_true, y_pred):
    return K.sum((1 - y_true) * (1 - y_pred))


def _false_positive(y_true, y_pred):
    return K.sum((1 - y_true) * y_pred)


def _false_negative(y_true, y_pred):
    return K.sum(y_true * (1 - y_pred))


true_positive = mixin_flatten_and_clip(_true_positive)
true_negative = mixin_flatten_and_clip(_true_negative)
false_positive = mixin_flatten_and_clip(_false_positive)
false_negative = mixin_flatten_and_clip(_false_negative)


def tp_tn_fp_fn_precision_recall(y_true, y_pred):
    y_true, y_pred = flatten_and_clip(y_true, y_pred)
    tp, tn = _true_positive(y_true, y_pred), _true_negative(y_true, y_pred)
    fp, fn = _false_positive(y_true, y_pred), _false_negative(y_true, y_pred)

    precision_ = tp / (tp + fp + K.epsilon())
    recall_ = tp / (tp + fn + K.epsilon())

    return tp, tn, fp, fn, precision_, recall_


def accuracy(y_true, y_pred):
    tp, tn, fp, fn, precision_, recall_ = tp_tn_fp_fn_precision_recall(y_true, y_pred)

    return (tp+tn) / (tp+tn+fp+fn + K.epsilon())


def precision(y_true, y_pred):
    tp, tn, fp, fn, precision_, recall_ = tp_tn_fp_fn_precision_recall(y_true, y_pred)
    return precision_


def recall(y_true, y_pred):
    tp, tn, fp, fn, precision_, recall_ = tp_tn_fp_fn_precision_recall(y_true, y_pred)
    return recall_


def f_score(y_true, y_pred):
    tp, tn, fp, fn, precision_, recall_ = tp_tn_fp_fn_precision_recall(y_true, y_pred)
    return 2 * (precision_*recall_) / (precision_+recall_ + K.epsilon())


def dice_index(y_true, y_pred):
    tp, tn, fp, fn, precision_, recall_ = tp_tn_fp_fn_precision_recall(y_true, y_pred)
    return (2 * tp) / (tp + fp + tp + fn + K.epsilon())


def jaccard_index(y_true, y_pred):
    tp, tn, fp, fn, precision_, recall_ = tp_tn_fp_fn_precision_recall(y_true, y_pred)
    return tp / (tp + fp + K.epsilon())


def generate_tversky_loss(alpha=0.5, beta=0.5):
    def _inner_tversky_loss(y_true, y_pred):
        return -tversky_index(y_true, y_pred, alpha=alpha, beta=beta)

    _inner_tversky_loss.__name__ = ('tversky_loss_%.2f_%.2f' % (alpha, beta,)).replace('.', '_')

    return _inner_tversky_loss


def tversky_index(y_true, y_pred, alpha=0.5, beta=0.5):
    tp, tn, fp, fn, precision_, recall_ = tp_tn_fp_fn_precision_recall(y_true, y_pred)
    return tp / (tp + alpha * fp + beta * fn + K.epsilon())
