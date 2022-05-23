from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, DepthwiseConv2D, UpSampling2D, Conv2DTranspose, \
    MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Input, Activation, Concatenate, Add
from .depth_to_space2d import *
from ..decoder_methods import *
from ..models_utils.metrics import *


def get_model_name(
        n_filters, separable_conv, shortcut_connection,
        conv_bias, bn_before_act, decoder_method, upconv_bias,
        global_random_seed, init_random_seed
):
    return f'UNet2D__{n_filters}nf_' \
           f'{"sepConv_" if separable_conv else ""}' \
           f'{"shortCon_" if shortcut_connection else ""}' \
           "relu_" \
           f'{"convBias_" if conv_bias else "bnBeforeAct_" if bn_before_act else ""}' \
           f'{decoder_method}_' \
           f'{"upconv_bias" if upconv_bias else ""}' \
           f'__BCE_loss__{global_random_seed}grs_{init_random_seed}irs'


# Notes:
# I don't use biases in point-wise convolutions
def conv2d_block(
        inputs, n_filters,
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

    # first layer
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

    # second layer
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

    x = Activation('relu')(x)
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


# retrain_decoder - если True, тогда pretrained_weights - веса базовой модели с pretrained_decoder_method
def UNet2D(
        optimizer, loss='binary_crossentropy', metrics=[BinaryMeanIOU(2, name=None, dtype=None)],
        retrain_decoder=False,
        pretrained_weights=None,
        init_seed=7,
        input_size=(None, None, 1), n_filters=64,
        separable_conv=False, shortcut_connection=False,
        bn_before_act=False, use_conv_bias=False,
        decoder_method='nearest', use_upconv_bias=False,
        pretrained_decoder_method='nearest', pretrained_use_upconv_bias=False,
):
    if retrain_decoder:
        assert retrain_decoder, \
            "You must specify the pre-trained model weights for transfer learning."
    inputs = Input(input_size)

    general_conv2d_block_kwargs = {
        "kernel_size": 3,
        "use_conv_bias": use_conv_bias,
        "separable_conv": separable_conv,
        "res_block": shortcut_connection,
        "bn_before_act": bn_before_act,
        "init_seed": init_seed,
    }

    general_decode2d_block_kwargs = {
        "decoder_method": decoder_method,
        "dec_conv_bias": use_upconv_bias,
        "init_seed": init_seed
    }
    general_pretrained_decode2d_block_kwargs = {
        "decoder_method": pretrained_decoder_method,
        "dec_conv_bias": pretrained_use_upconv_bias,
        "init_seed": init_seed
    }

    conv1 = conv2d_block(
        inputs,
        n_filters=n_filters * 1,
        **general_conv2d_block_kwargs
    )
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = conv2d_block(
        pool1,
        n_filters=n_filters * 2,
        **general_conv2d_block_kwargs
    )
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = conv2d_block(
        pool2,
        n_filters=n_filters * 4,
        **general_conv2d_block_kwargs
    )
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = conv2d_block(
        pool3,
        n_filters=n_filters * 8,
        **general_conv2d_block_kwargs
    )
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = conv2d_block(
        pool4,
        n_filters=n_filters * 16,
        **general_conv2d_block_kwargs
    )

    if retrain_decoder:  # and pretrained_weights:
        up6_ = decode2d_block(
            conv5,
            input_n_filters=n_filters * 16,
            **general_pretrained_decode2d_block_kwargs
        )

        merge6_ = Concatenate(axis=3)([conv4, up6_])
        conv6_ = conv2d_block(
            merge6_,
            n_filters=n_filters * 8,
            **general_conv2d_block_kwargs
        )
        up7_ = decode2d_block(
            conv6_,
            input_n_filters=n_filters * 8,
            **general_pretrained_decode2d_block_kwargs
        )

        merge7_ = Concatenate(axis=3)([conv3, up7_])
        conv7_ = conv2d_block(
            merge7_,
            n_filters=n_filters * 4,
            **general_conv2d_block_kwargs
        )
        up8_ = decode2d_block(
            conv7_,
            input_n_filters=n_filters * 4,
            **general_pretrained_decode2d_block_kwargs
        )

        merge8_ = Concatenate(axis=3)([conv2, up8_])
        conv8_ = conv2d_block(
            merge8_,
            n_filters=n_filters * 2,
            **general_conv2d_block_kwargs
        )
        up9_ = decode2d_block(
            conv8_,
            input_n_filters=n_filters * 2,
            **general_pretrained_decode2d_block_kwargs
        )

        merge9_ = Concatenate(axis=3)([conv1, up9_])
        conv9_ = conv2d_block(
            merge9_, n_filters * 1,
            **general_conv2d_block_kwargs
        )
        conv10_ = Conv2D(
            filters=1,
            kernel_size=(1, 1),
            strides=(1, 1),
            padding='same',
            activation='sigmoid',
            use_bias=True, # ToDo
            kernel_initializer=tf.keras.initializers.HeNormal(seed=init_seed),
            bias_initializer='zeros'
        )(conv9_)

        model_ = Model(inputs=inputs, outputs=conv10_)
        model_.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        # print(list(zip(model_.layers, reversed(list(range(1, len(model_.layers) + 1))))))
        # print(model_.layers[1].get_weights())

        # print("----------------------")
        model_.load_weights(pretrained_weights)
        print("Model was loaded to retrain part of the decoder")
        # print(model_.layers[1].get_weights())

        step = 6 if bn_before_act else 4  # 2 BatchNormalization objects
        step += 2 if shortcut_connection else 0  # PointWiseConv and core.TFOpLambda (Add?) objects
        step += 1 if separable_conv and not shortcut_connection else 0  # additional PointWiseConv
        # Отрезаем декодировщик и переучиваем его с новой реализацией блока декодировщика
        conv1 = model_.layers[step].output
        conv2 = model_.layers[step * 2 + 1].output
        conv3 = model_.layers[step * 3 + 2].output
        conv4 = model_.layers[step * 4 + 3].output
        conv5 = model_.layers[step * 5 + 4].output
        del model_
    up6 = decode2d_block(
        conv5,
        input_n_filters=n_filters * 16,
        **general_decode2d_block_kwargs
    )

    merge6 = Concatenate(axis=3)([conv4, up6])
    conv6 = conv2d_block(
        merge6,
        n_filters=n_filters * 8,
        **general_conv2d_block_kwargs
    )
    up7 = decode2d_block(
        conv6,
        input_n_filters=n_filters * 8,
        **general_decode2d_block_kwargs
    )

    merge7 = Concatenate(axis=3)([conv3, up7])
    conv7 = conv2d_block(
        merge7,
        n_filters=n_filters * 4,
        **general_conv2d_block_kwargs
    )
    up8 = decode2d_block(
        conv7,
        input_n_filters=n_filters * 4,
        **general_decode2d_block_kwargs
    )

    merge8 = Concatenate(axis=3)([conv2, up8])
    conv8 = conv2d_block(
        merge8,
        n_filters=n_filters * 2,
        **general_conv2d_block_kwargs
    )
    up9 = decode2d_block(
        conv8,
        input_n_filters=n_filters * 2,
        **general_decode2d_block_kwargs
    )

    merge9 = Concatenate(axis=3)([conv1, up9])
    conv9 = conv2d_block(
        merge9,
        n_filters=n_filters * 1,
        **general_conv2d_block_kwargs
    )
    conv10 = Conv2D(
        filters=1,
        kernel_size=(1, 1),
        strides=(1, 1),
        padding='same',
        activation='sigmoid',
        use_bias=True, # ToDo
        kernel_initializer=tf.keras.initializers.HeNormal(seed=init_seed),
        bias_initializer='zeros'
    )(conv9)

    model = Model(inputs=inputs, outputs=conv10)
    model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
    # print("----------------------")
    # print(model.layers[1].get_weights())
    if (not retrain_decoder and pretrained_weights):
        model.load_weights(pretrained_weights)
        print("Model was loaded for additional training")
    return model
