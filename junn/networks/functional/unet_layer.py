from tensorflow.keras.layers import (
    Layer, Conv2D, Conv2DTranspose, Dropout, BatchNormalization,
    concatenate,
    MaxPooling2D, UpSampling2D,
    Add, Activation, Reshape
)

from functools import partial


class RecurrentConvolution2D(Layer):
    def __init__(self, recurrent_num=1, **kwargs):
        super(RecurrentConvolution2D, self).__init__()

        self.recurrent_num = recurrent_num
        self.initial_conv = Conv2D(**kwargs)
        self.recurrent_conv = Conv2D(**kwargs)

    def get_config(self):
        config = super().get_config().copy()

        config['recurrent_num'] = self.recurrent_num

        return config

    def call(self, inputs, **kwargs):
        output = self.initial_conv(inputs)

        for _ in range(self.recurrent_num):
            output = self.recurrent_conv(concatenate([output, inputs]))

        return output


def unet_unit(tensor,
              filters=64, kernel_size=3, activation='relu', batch_normalization=True, name='unet_unit',
              recurrent_num=0,
              residual_connections=False, dropout=0.0):

    parameters = dict(filters=filters, kernel_size=kernel_size, padding='same', activation=activation)

    convolution = partial(Conv2D, **parameters)

    # recurrent_pre_tensor = None
    pre_tensor = None

    first_conv = convolution(name='%s_conv_first' % (name,))
    first_batch_norm = BatchNormalization(name='%s_batch_normalization_first' % (name,))

    second_conv = convolution(name='%s_conv_second' % (name,))
    second_batch_norm = BatchNormalization(name='%s_batch_normalization_second' % (name,))

    # if recurrent_pre_tensor is not None:
    #     tensor = concatenate([tensor, recurrent_pre_tensor])
    # else:
    #     recurrent_pre_tensor = tensor

    tensor = first_conv(tensor)

    if pre_tensor is None:
        pre_tensor = tensor

    if batch_normalization:
        tensor = first_batch_norm(tensor)

    if dropout:
        tensor = Dropout(rate=dropout)(tensor)

    if recurrent_num:
        tensor = RecurrentConvolution2D(recurrent_num=recurrent_num, **parameters)(tensor)
    else:
        tensor = second_conv(tensor)

    # for all or just non-recurrent ?!
    if batch_normalization:
        tensor = second_batch_norm(tensor)

    if residual_connections:
        tensor = Add()([pre_tensor, tensor])

    return tensor


def unet_level(tensor,
               filters=64, level=0, level_factor=2.0, total_levels=0, dropout_up=0.0, dropout_down=0.0,
               just_convolutions=False,
               activation='relu', batch_normalization=True, kernel_size=3,
               residual_connections=False, recurrent_num=0,
               name='unet'):
    configured_unet_unit = partial(unet_unit,
                                   filters=filters,
                                   activation=activation,
                                   batch_normalization=batch_normalization,
                                   kernel_size=kernel_size,
                                   recurrent_num=recurrent_num,
                                   residual_connections=residual_connections)

    tensor = configured_unet_unit(tensor, dropout=dropout_down, name='%s_level_%d_down' % (name, total_levels - level,))

    horizontal_connection = tensor

    if level > 0:
        if not just_convolutions:
            tensor = MaxPooling2D()(tensor)
        else:
            tensor = Conv2D(filters, 3, strides=2, padding='same', activation=activation)(tensor)

        tensor = unet_level(tensor, filters=int(filters * level_factor), level=level - 1, total_levels=total_levels,
                            dropout_up=dropout_up, dropout_down=dropout_down,
                            just_convolutions=just_convolutions,
                            activation=activation, batch_normalization=batch_normalization, kernel_size=kernel_size,
                            residual_connections=residual_connections, recurrent_num=recurrent_num,
                            name=name)

        if not just_convolutions:
            tensor = UpSampling2D()(tensor)
        else:
            tensor = Conv2DTranspose(filters, 3, strides=2, padding='same', activation=activation)(tensor)

        horizontal_connection = Conv2D(filters, 1)(horizontal_connection)

        tensor = concatenate([tensor, horizontal_connection], axis=3)

        tensor = configured_unet_unit(tensor, dropout=dropout_up, name='%s_level_%d_up' % (name, total_levels - level,))
    else:
        tensor = tensor

    return tensor


def unet(input_tensor, levels=4, filters=64, activation='relu', output_channels=1, dropout_down=0.0, dropout_up=0.0,
         just_convolutions=False, batch_normalization=True, kernel_size=3, residual_connections=False,
         level_factor=2.0, last_activation=None, recurrent_num=0,
         name='unet', categorical=False, **kwargs):

    if last_activation is None:  # set defaults, only last_activation is dependent on categorical
        if not categorical:
            last_activation = 'sigmoid'
        else:
            last_activation = 'softmax'

    tensor = unet_level(input_tensor,
                        filters=filters, level=levels, total_levels=levels, name=name,
                        dropout_down=dropout_down, dropout_up=dropout_up, just_convolutions=just_convolutions,
                        activation=activation, batch_normalization=batch_normalization, kernel_size=kernel_size,
                        level_factor=level_factor,
                        recurrent_num=recurrent_num,
                        residual_connections=residual_connections,
                        )
    if not categorical:
        tensor = Conv2DTranspose(filters=output_channels, kernel_size=1, activation=last_activation)(tensor)
    else:
        tensor = Conv2D(filters=output_channels, kernel_size=1, activation='linear')(tensor)
        tile_size = tuple([int(dim) for dim in input_tensor.shape[1:3]])
        tensor = Reshape((tile_size[0] * tile_size[1], output_channels))(tensor)
        tensor = Activation(last_activation)(tensor)
        tensor = Reshape(tile_size + (output_channels,))(tensor)

    return tensor
