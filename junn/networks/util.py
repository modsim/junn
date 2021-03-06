import inspect

from tensorflow.keras.backend import count_params
from tensorflow.python.saved_model import signature_serialization

# noinspection PyProtectedMember
from tensorflow.python.saved_model.save import _AugmentedGraphView
from tensorflow.python.saved_model.signature_constants import (
    DEFAULT_SERVING_SIGNATURE_DEF_KEY,
)


def get_weight_counts(model):
    def sum_weights(layer_list):
        return sum(count_params(layer) for layer in layer_list)

    try:
        # noinspection PyProtectedMember
        trainable = sum_weights(model._collected_trainable_weights)
    except AttributeError:
        trainable = sum_weights(model.trainable_weights)

    non_trainable = sum_weights(model.non_trainable_weights)

    return dict(trainable=trainable, non_trainable=non_trainable)


def get_default_signature(model):
    return (
        DEFAULT_SERVING_SIGNATURE_DEF_KEY,
        signature_serialization.find_function_to_export(_AugmentedGraphView(model)),
    )


def format_size(bytes_):
    suffixes = ['B', 'KiB', 'MiB', 'GiB', 'TiB', 'PiB', 'EiB']

    index = 0
    start = 1
    while bytes_ >= start * 1024:
        start *= 1024
        index += 1

    return "%.2f %s" % (bytes_ / start, suffixes[index])


def numpy_to_scalar(val):
    try:
        return val.item()
    except AttributeError:
        return val


def get_keyword_arguments(func):
    # could also check for .kind == inspect._ParameterKind.POSITIONAL_OR_KEYWORD, but how would new Python kw only react
    # noinspection PyProtectedMember
    return {
        key
        for key, value in inspect.signature(func).parameters.items()
        if value.default is not inspect._empty
    }


def warn_unused_arguments(parameters, func, log):
    unused_arguments = set(parameters.keys()) - set(get_keyword_arguments(func))
    if unused_arguments != set():
        log.warn(
            "Unused arguments for call %s: %s",
            func.__name__,
            ', '.join(sorted(unused_arguments)),
        )
