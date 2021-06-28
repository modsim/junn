"""Helper functionality to read and write ``.tfrecord`` files."""
import tensorflow as tf

"""
Mapping of type identifiers (numeric) to TensorFlow types.
"""
parse_tensor_type_lookup = {
    1: tf.float32,
    2: tf.float64,
    3: tf.int32,
    4: tf.uint8,
    5: tf.int16,
    6: tf.int8,
    8: tf.complex64,
    9: tf.int64,
    10: tf.bool,
    14: tf.bfloat16,
    17: tf.uint16,
    18: tf.complex128,
    19: tf.float16,
}


@tf.function
def get_tensor_magic_number(tensor):
    """
    Return the type identifier of a serialized TensorFlow tensor proto.

    :param tensor:
    :return:
    """
    magic = tf.reshape(tf.io.decode_raw(tf.strings.substr(tensor, 1, 1), tf.uint8), [])
    return magic


@tf.function
def parse_tensor(tensor, coerce):
    """
    Parse a binary tensor containing a serialized tensor proto to a Tensor.

    Coercing it into a specific type.

    :param tensor:
    :param coerce:
    :return:
    """
    magic = get_tensor_magic_number(tensor)
    arguments = []
    for n in range(0, max(parse_tensor_type_lookup.keys()) + 1):
        if n in parse_tensor_type_lookup:
            dtype = parse_tensor_type_lookup[n]
            arguments.append(
                (n, lambda: tf.cast(tf.io.parse_tensor(tensor, dtype), coerce))
            )
        else:
            arguments.append((n, lambda: tf.zeros(0, dtype=coerce)))

    return tf.switch_case(tf.cast(magic, tf.int32), arguments)


def create_example(x, y):
    """
    Create an ``tf.train.Example`` with JUNN-convention contents.

    :param x:
    :param y:
    :return:
    """

    def _bytes_feature(value):
        if isinstance(value, type(tf.constant(0))):
            value = value.numpy()
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))

    return tf.train.Example(
        features=tf.train.Features(
            feature=dict(
                x=_bytes_feature(tf.io.serialize_tensor(x)),
                y=_bytes_feature(tf.io.serialize_tensor(y)),
            )
        )
    ).SerializeToString()


def read_junn_tfrecord(file_name):
    """
    Read a ``.tfrecord`` file following JUNN convention and yields Dataset.

    :param file_name:
    :return:
    """
    feature_description = dict(
        x=tf.io.FixedLenFeature([], tf.string), y=tf.io.FixedLenFeature([], tf.string)
    )

    @tf.function
    def _parse(example_proto):
        return tf.io.parse_single_example(example_proto, feature_description)

    # sad but true: we need to peek into the dataset to get the dtype
    first = next(iter(tf.data.TFRecordDataset(file_name).map(_parse)))

    x_magic_number, y_magic_number = (
        get_tensor_magic_number(first['x']),
        get_tensor_magic_number(first['y']),
    )
    x_type, y_type = (
        parse_tensor_type_lookup[x_magic_number.numpy()],
        parse_tensor_type_lookup[y_magic_number.numpy()],
    )

    @tf.function
    def _unserialize(example):
        return tf.io.parse_tensor(example['x'], x_type), tf.io.parse_tensor(
            example['y'], y_type
        )

    raw_dataset = tf.data.TFRecordDataset(file_name)
    dataset = raw_dataset
    dataset = dataset.map(_parse)
    dataset = dataset.map(_unserialize)

    return dataset
