from tensorflow.keras.layers import (
    Activation,
    Add,
    BatchNormalization,
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    Reshape,
    concatenate,
)


def link_net(
    input_tensor,
    activation='selu',
    last_activation='sigmoid',
    batch_normalization=False,
    categorical=False,
    output_channels=1,
    **kwargs
):
    """
    A LinkNet according to

    LinkNet: Exploiting Encoder Representations for Efficient Semantic Segmentation
    by A. Chaurasia and E. Culurciello. arXiv 1707.03718

    :param input_tensor:
    :param activation: Activation used for each convolution layer
    :param last_activation: Final activation used
    :param batch_normalization:
    :param categorical: Whether the output should be categorical (multiple channels) or a single channel
    :param output_channels: Count of output channels (i.e. classes)
    :param kwargs:
    :return:
    """

    if categorical:
        raise NotImplementedError('categorical prediction currently not implemented.')

    # having a up-sample as a very late step leads to 2x2 blocked images, which is too imprecise for our usage

    depths = {
        1: dict(m=64, n=64),
        2: dict(m=64, n=128),
        3: dict(m=128, n=256),
        4: dict(m=256, n=512),
    }

    min_depth = 1
    max_depth = 4

    def encoder_m_n(num):
        return depths[num]['m'], depths[num]['n']

    def decoder_m_n(num):
        n, m = encoder_m_n(num)
        return m, n

    # noinspection PyShadowingNames
    def inner_pair(tensor, depth):
        if depth == max_depth:
            return tensor

        m, n = encoder_m_n(depth)

        # Encoder part

        # #

        outer_residual_connection = tensor

        # residual_connection = tensor

        # THIS DOES NOT WORK?!

        tensor = Conv2D(
            filters=n, kernel_size=3, padding='same', strides=2, activation=activation
        )(tensor)

        if batch_normalization:
            tensor = BatchNormalization()(tensor)

        # fix
        residual_connection = tensor
        # /
        tensor = Conv2D(
            filters=n, kernel_size=3, padding='same', activation=activation
        )(tensor)
        tensor = Add()([tensor, residual_connection])

        # #

        residual_connection = tensor
        tensor = Conv2D(
            filters=n, kernel_size=3, padding='same', activation=activation
        )(tensor)

        if batch_normalization:
            tensor = BatchNormalization()(tensor)

        tensor = Conv2D(
            filters=n, kernel_size=3, padding='same', activation=activation
        )(tensor)
        tensor = Add()([tensor, residual_connection])

        # deeper layers
        tensor = inner_pair(tensor, depth=depth + 1)

        m, n = decoder_m_n(depth)

        # Decoder part

        tensor = Conv2D(
            filters=m // 4, kernel_size=1, padding='same', activation=activation
        )(tensor)

        if batch_normalization:
            tensor = BatchNormalization()(tensor)

        tensor = Conv2DTranspose(
            filters=m // 4,
            kernel_size=3,
            padding='same',
            strides=2,
            activation=activation,
        )(tensor)

        if batch_normalization:
            tensor = BatchNormalization()(tensor)

        tensor = Conv2D(
            filters=n, kernel_size=1, padding='same', activation=activation
        )(tensor)

        if depth > 1:
            tensor = Add()([tensor, outer_residual_connection])

        return tensor

    tensor = input_tensor

    tensor = Conv2D(
        filters=64, kernel_size=7, padding='same', strides=2, activation=activation
    )(tensor)

    if batch_normalization:
        tensor = BatchNormalization()(tensor)

    tensor = MaxPooling2D(pool_size=(3, 3), strides=2)(tensor)

    tensor = inner_pair(tensor, depth=min_depth)

    # they differentiate between full conv. (apparently, input=output size) and convolution (apparently mode=valid)
    # we only do full convolutions in the code here

    tensor = Conv2DTranspose(
        filters=32, kernel_size=3, padding='same', strides=2, activation=activation
    )(tensor)
    if batch_normalization:
        tensor = BatchNormalization()(tensor)
    tensor = Conv2D(filters=32, kernel_size=3, padding='same', activation=activation)(
        tensor
    )
    if batch_normalization:
        tensor = BatchNormalization()(tensor)
    tensor = Conv2DTranspose(
        filters=output_channels,
        kernel_size=2,
        padding='same',
        strides=2,
        activation=last_activation,
    )(tensor)
    result = tensor

    #                concatenate(last_tensors) if len(last_tensors) > 1 else last_tensors[0]

    # if not categorical:
    #     result = Conv2D(filters=output_channels, kernel_size=1, activation=last_activation)(combined)
    # else:
    #     result = Conv2D(filters=output_channels, kernel_size=1, activation='linear')(combined)
    #     result = Reshape((tile_size[0] * tile_size[1], output_channels))(result)
    #     result = Activation(last_activation)(result)
    #     result = Reshape(tile_size + (output_channels,))(result)

    return result
