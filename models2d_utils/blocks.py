from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, UpSampling2D, Conv2DTranspose, BatchNormalization
from tensorflow.keras.layers import Activation, Add
from .depth_to_space2d import *
from ..decoder_methods import *


# Notes:
# I don't use biases in point-wise convolutions
def conv2d_block(
        inputs, n_filters,
        num_layers=2,
        kernel_size=3, use_conv_bias=False,
        separable_conv=False, res_block=False,
        bn_before_act=False,
        init_seed=7,
):
    general_conv_kwargs = {
        "kernel_size": (kernel_size, kernel_size),
        "strides": (1, 1),
        "padding": 'same',
        "use_bias": use_conv_bias,
        "bias_initializer": 'zeros',
        "activation": None,
    }

    general_batchnorm_kwargs = {
        "axis": -1,
        "momentum": 0.99,
        "epsilon": 0.001,
        "center": True,
        "scale": True,
        "beta_initializer": 'zeros',
        "gamma_initializer": 'ones',
        "moving_mean_initializer": 'zeros',
        "moving_variance_initializer": 'ones',
    }

    # first layer in block
    if separable_conv:
        inputs = Conv2D(
            filters=n_filters,
            kernel_size=(1, 1),
            kernel_initializer=tf.keras.initializers.HeNormal(seed=init_seed),
            strides=(1, 1),
            padding='same',
            use_bias=False,
            activation=None,
        )(inputs)
        x = DepthwiseConv2D(
            depth_multiplier=1,
            depthwise_initializer=tf.keras.initializers.HeNormal(seed=init_seed),
            **general_conv_kwargs
        )(inputs)
    else:
        x = Conv2D(
            filters=n_filters,
            kernel_initializer=tf.keras.initializers.HeNormal(seed=init_seed),
            **general_conv_kwargs
        )(inputs)

    if bn_before_act:
        x = BatchNormalization(**general_batchnorm_kwargs)(x)

    x = Activation('relu')(x)

    for i_layer in range(num_layers - 1):
        # i - layer in block
        if separable_conv:
            x = DepthwiseConv2D(
                depth_multiplier=1,
                depthwise_initializer=tf.keras.initializers.HeNormal(seed=init_seed),
                **general_conv_kwargs
            )(x)
        else:
            x = Conv2D(
                filters=n_filters,
                kernel_initializer=tf.keras.initializers.HeNormal(seed=init_seed),
                **general_conv_kwargs
            )(x)

        if bn_before_act:
            x = BatchNormalization(**general_batchnorm_kwargs)(x)

        x = Activation('relu')(x)

        # ToDo - Для блоков с большим количеством свёрток имеет смысл добавить Dense-связь
        if res_block:
            if not separable_conv:
                # Слои имеют разную размерность, поэтому приводим к одной
                inputs = Conv2D(
                    filters=n_filters,
                    kernel_size=(1, 1),
                    kernel_initializer=tf.keras.initializers.HeNormal(seed=init_seed),
                    strides=(1, 1),
                    padding='same',
                    use_bias=False,
                    activation=None,
                )(inputs)
            x = Add()([x, inputs])

    return x


def decode2d_block(
        inputs, input_n_filters,
        dec_conv_bias=False,
        decoder_method='nearest',
        init_seed=7
):
    '''
        Повышение разрешения реализуется посредством:
           1) интерполяционного подхода;
           2) операции reshape (depth_to_space)
           3) обучаемой транспонированной свёртки
        Для 1) и 2) способа возможно задать дополнительные свёрточные фильтры
    '''
    # ToDo:
    #  scale_factor
    #  kernel_size for additional conv

    if decoder_method not in DECODER_METHODS:
        raise Exception(
            "Error in decode2d_block: unknown decoder method", decoder_method
        )

    if decoder_method.find('nearest') != -1 or decoder_method.find('bilinear') != -1:
        if decoder_method.find('nearest') != -1:
            out = UpSampling2D(size=(2, 2), interpolation='nearest')(inputs)
        else:
            out = UpSampling2D(size=(2, 2), interpolation='bilinear')(inputs)
        out_filters = input_n_filters
    elif decoder_method.find('subpixel') != -1:
        out = inputs
        if decoder_method.find('subpixel4') == -1:
            # DepthToSpace2D при scale=2 уменьшает кол-во каналов в 4 раза
            # Чтобы сохранить кол-во каналов, используем дополнительную point-wise conv
            if decoder_method.find('subpixel2') != -1:
                out_filters = input_n_filters * 2
            else:
                out_filters = input_n_filters * 4

            out = Conv2D(
                filters=out_filters,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding='same',
                use_bias=False,
                kernel_initializer=tf.keras.initializers.HeNormal(seed=init_seed),
                bias_initializer='zeros',
            )(out)
        else:
            out_filters = input_n_filters

        out = DepthToSpace2D(
            scale=2
        )(out)
        out_filters = out_filters // 4
    elif decoder_method.find('convTranspose') != -1:
        if decoder_method.find('convTranspose4') != -1:
            out_filters = input_n_filters // 4
        elif decoder_method.find('convTranspose2') != -1:
            out_filters = input_n_filters // 2
        else:
            out_filters = input_n_filters

        # new_dim = ((dim - 1) * strides[0] + kernel_size[0] - 2 * padding[0] + output_padding[0])
        # new_dim = ((dim - 1) * 2 + 2 =  2 * dim
        out = Conv2DTranspose(
            filters=out_filters,
            kernel_size=(2, 2),
            strides=(2, 2),
            padding='valid',
            use_bias=dec_conv_bias,
            kernel_initializer=tf.keras.initializers.HeNormal(seed=init_seed),
            bias_initializer='zeros',
        )(inputs)

    # Для convTranspose нет смысла вводить дополнительные преобразования обычными свёртками
    if decoder_method.find('conv') != -1 and decoder_method.find('convTranspose') == -1:
        if decoder_method.find('conv4') != -1:
            out_filters = out_filters // 4
        elif decoder_method.find('conv2') != -1:
            out_filters = out_filters // 2
        else:
            out_filters = out_filters

        if decoder_method.find('sepconv') != -1:
            out = Conv2D(
                filters=out_filters,
                kernel_size=(1, 1),
                kernel_initializer=tf.keras.initializers.HeNormal(seed=init_seed),
                strides=(1, 1),
                padding='same',
                use_bias=False,
                activation=None,
            )(out)

            out = DepthwiseConv2D(
                kernel_size=(2, 2),
                strides=(1, 1),
                padding='same',
                use_bias=dec_conv_bias,
                bias_initializer='zeros',
                activation=None,
                depth_multiplier=1,
                depthwise_initializer=tf.keras.initializers.HeNormal(seed=init_seed),
            )(out)
        else:
            out = Conv2D(
                filters=out_filters,
                kernel_size=(2, 2),
                strides=(1, 1),
                padding='same',
                use_bias=dec_conv_bias,
                kernel_initializer=tf.keras.initializers.HeNormal(seed=init_seed),
                bias_initializer='zeros',
                activation=None,
            )(out)

    return out
