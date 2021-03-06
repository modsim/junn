from tensorflow.keras.layers import Activation, Conv2D, Reshape, concatenate


def msd_net(
    input_tensor,
    d=3,
    w=2,
    tile_size=(
        32,
        32,
    ),
    activation='selu',
    last_activation='sigmoid',
    categorical=False,
    output_channels=1,
    kernel_size=3,
    max_dilation=10,
    filters_per_convolution=1,
    **kwargs
):
    """
    A MS-D net according to

    A mixed-scale dense convolutional neural network for image analysis
    by D. M. Pelt and J. A. Sethian. PNAS. 10.1073/pnas.1715832114

    :param input_tensor:
    :param d: The depth of the network
    :param w: The width of the network
    :param tile_size: Tile size
    :param activation: Activation used for each convolution layer
    :param last_activation: Final activation used
    :param categorical: Whether the output should be categorical (multiple channels) or a single channel
    :param output_channels: Count of output channels (i.e. classes)
    :param kernel_size: The kernel size used for the convolutions
    :param max_dilation: Maximum dilation used
    :param filters_per_convolution: Filter count per convolution
    :param kwargs:
    :return:
    """

    last_tensors = [input_tensor]

    for i in range(d):
        add = []
        for j in range(w):
            dilation = (i * w + j) % max_dilation + 1
            node = Conv2D(
                filters=filters_per_convolution,
                kernel_size=kernel_size,
                dilation_rate=dilation,
                padding='same',
                activation=activation,
                name='conv2d_d_%d_w_%d'
                % (
                    i,
                    j,
                ),
            )(
                # we must COPY the list here because to tf issue #32023
                concatenate(last_tensors[:])
                if len(last_tensors) > 1
                else last_tensors[0]
            )

            add += [node]

        last_tensors += add

    combined = concatenate(last_tensors[:])

    if not categorical:
        result = Conv2D(
            filters=output_channels, kernel_size=1, activation=last_activation
        )(combined)
    else:
        result = Conv2D(filters=output_channels, kernel_size=1, activation='linear')(
            combined
        )
        result = Reshape((tile_size[0] * tile_size[1], output_channels))(result)
        result = Activation(last_activation)(result)
        result = Reshape(tile_size + (output_channels,))(result)

    return result
