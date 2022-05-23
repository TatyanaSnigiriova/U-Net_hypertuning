from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv3D, UpSampling3D, Conv3DTranspose, MaxPooling3D, BatchNormalization
from tensorflow.keras.layers import Input, Activation, Concatenate, Add

from ..decoder_methods import *
from ..models_utils.metrics import *


# Notes:
# I don't use biases in point-wise convolutions
def conv3d_block(
        inputs, n_filters,
        kernel_size=3, use_conv_bias=False,
        res_block=False,
        bn_before_act=False,
        init_seed=7,
):
    general_conv_kwargs = {
        "kernel_size": (kernel_size, kernel_size, kernel_size),
        "strides": (1, 1, 1),
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
    x = Conv3D(
        filters=n_filters,
        kernel_initializer=tf.keras.initializers.HeNormal(seed=init_seed),
        **general_conv_kwargs
    )(inputs)

    if bn_before_act:
        x = BatchNormalization(**general_batchnorm_kwargs)(x)

    x = Activation('relu')(x)

    # second layer
    x = Conv3D(
        filters=n_filters,
        kernel_initializer=tf.keras.initializers.HeNormal(seed=init_seed),
        **general_conv_kwargs
    )(x)

    if bn_before_act:
        x = BatchNormalization(**general_batchnorm_kwargs)(x)

    if res_block:
        # Слои имеют разную размерность, поэтому приводим к одной
        inputs = Conv3D(
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


def decode3d_block(
        inputs, input_n_filters,
        dec_conv_bias=False,
        decoder_method='nearest',
        init_seed=7
):
    '''
        Возможности повышения разрешения 3D представления являются ограниченными следующими методами:
           1) интерполяционным подходом ближайшего соседа;
           3) обучаемой транспонированной свёрткой
        Для 1) способа возможно задать дополнительные свёрточные фильтры
    '''
    # ToDo:
    #  scale_factor
    #  kernel_size for additional conv

    if decoder_method not in DECODER_METHODS:
        raise Exception(
            "Error in decode3d_block: unknown decoder method", decoder_method
        )

    if decoder_method.find('nearest') != -1:
        out = UpSampling3D(size=(2, 2, 2))(inputs)
        out_filters = input_n_filters
    elif decoder_method.find('convTranspose') != -1:
        if decoder_method.find('convTranspose4') != -1:
            out_filters = input_n_filters // 4
        elif decoder_method.find('convTranspose2') != -1:
            out_filters = input_n_filters // 2
        else:
            out_filters = input_n_filters

        out = Conv3DTranspose(
            filters=out_filters,
            kernel_size=(2, 2, 2),
            strides=(2, 2, 2),
            padding='same',
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

        out = Conv3D(
            filters=out_filters,
            kernel_size=(2, 2, 2),
            strides=(1, 1, 1),
            padding='same',
            use_bias=dec_conv_bias,
            kernel_initializer=tf.keras.initializers.HeNormal(seed=init_seed),
            bias_initializer='zeros',
            activation=None,
        )(out)

    return out


# retrain_decoder - если True, тогда pretrained_weights - веса базовой модели с pretrained_decoder_method
def UNet3D(
        optimizer, loss='binary_crossentropy', metrics=[BinaryMeanIOU(2, name=None, dtype=None)],
        retrain_decoder=False,
        pretrained_weights=None,
        init_seed=7,
        input_size=(None, None, 1), n_filters=64,
        shortcut_connection=False,
        bn_before_act=False, use_conv_bias=False,
        decoder_method='nearest', use_upconv_bias=False,
        pretrained_decoder_method='nearest', pretrained_use_upconv_bias=False,
):
    if retrain_decoder:
        assert retrain_decoder, \
            "You must specify the pre-trained model weights for transfer learning."
    inputs = Input(input_size)

    general_conv3d_block_kwargs = {
        "kernel_size": 3,
        "use_conv_bias": use_conv_bias,
        "res_block": shortcut_connection,
        "bn_before_act": bn_before_act,
        "init_seed": init_seed,
    }

    general_decode3d_block_kwargs = {
        "decoder_method": decoder_method,
        "dec_conv_bias": use_upconv_bias,
        "init_seed": init_seed
    }
    general_pretrained_decode3d_block_kwargs = {
        "decoder_method": pretrained_decoder_method,
        "dec_conv_bias": pretrained_use_upconv_bias,
        "init_seed": init_seed
    }

    conv1 = conv3d_block(
        inputs,
        n_filters=n_filters * 1,
        **general_conv3d_block_kwargs
    )
    pool1 = MaxPooling3D(pool_size=(2, 2))(conv1)

    conv2 = conv3d_block(
        pool1,
        n_filters=n_filters * 2,
        **general_conv3d_block_kwargs
    )
    pool2 = MaxPooling3D(pool_size=(2, 2))(conv2)

    conv3 = conv3d_block(
        pool2,
        n_filters=n_filters * 4,
        **general_conv3d_block_kwargs
    )
    pool3 = MaxPooling3D(pool_size=(2, 2))(conv3)

    conv4 = conv3d_block(
        pool3,
        n_filters=n_filters * 8,
        **general_conv3d_block_kwargs
    )
    pool4 = MaxPooling3D(pool_size=(2, 2))(conv4)

    conv5 = conv3d_block(
        pool4,
        n_filters=n_filters * 16,
        **general_conv3d_block_kwargs
    )

    if retrain_decoder:  # and pretrained_weights:
        up6_ = decode3d_block(
            conv5,
            input_n_filters=n_filters * 16,
            **general_pretrained_decode3d_block_kwargs
        )

        merge6_ = Concatenate(axis=3)([conv4, up6_])
        conv6_ = conv3d_block(
            merge6_,
            n_filters=n_filters * 8,
            **general_conv3d_block_kwargs
        )
        up7_ = decode3d_block(
            conv6_,
            input_n_filters=n_filters * 8,
            **general_pretrained_decode3d_block_kwargs
        )

        merge7_ = Concatenate(axis=3)([conv3, up7_])
        conv7_ = conv3d_block(
            merge7_,
            n_filters=n_filters * 4,
            **general_conv3d_block_kwargs
        )
        up8_ = decode3d_block(
            conv7_,
            input_n_filters=n_filters * 4,
            **general_pretrained_decode3d_block_kwargs
        )

        merge8_ = Concatenate(axis=3)([conv2, up8_])
        conv8_ = conv3d_block(
            merge8_,
            n_filters=n_filters * 2,
            **general_conv3d_block_kwargs
        )
        up9_ = decode3d_block(
            conv8_,
            input_n_filters=n_filters * 2,
            **general_pretrained_decode3d_block_kwargs
        )

        merge9_ = Concatenate(axis=3)([conv1, up9_])
        conv9_ = conv3d_block(
            merge9_, n_filters * 1,
            **general_conv3d_block_kwargs
        )
        conv10_ = Conv3D(
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
        # Отрезаем декодировщик и переучиваем его с новой реализацией блока декодировщика
        conv1 = model_.layers[step].output
        conv2 = model_.layers[step * 2 + 1].output
        conv3 = model_.layers[step * 3 + 2].output
        conv4 = model_.layers[step * 4 + 3].output
        conv5 = model_.layers[step * 5 + 4].output

    up6 = decode3d_block(
        conv5,
        input_n_filters=n_filters * 16,
        **general_decode3d_block_kwargs
    )

    merge6 = Concatenate(axis=3)([conv4, up6])
    conv6 = conv3d_block(
        merge6,
        n_filters=n_filters * 8,
        **general_conv3d_block_kwargs
    )
    up7 = decode3d_block(
        conv6,
        input_n_filters=n_filters * 8,
        **general_decode3d_block_kwargs
    )

    merge7 = Concatenate(axis=3)([conv3, up7])
    conv7 = conv3d_block(
        merge7,
        n_filters=n_filters * 4,
        **general_conv3d_block_kwargs
    )
    up8 = decode3d_block(
        conv7,
        input_n_filters=n_filters * 4,
        **general_decode3d_block_kwargs
    )

    merge8 = Concatenate(axis=3)([conv2, up8])
    conv8 = conv3d_block(
        merge8,
        n_filters=n_filters * 2,
        **general_conv3d_block_kwargs
    )
    up9 = decode3d_block(
        conv8,
        input_n_filters=n_filters * 2,
        **general_decode3d_block_kwargs
    )

    merge9 = Concatenate(axis=3)([conv1, up9])
    conv9 = conv3d_block(
        merge9,
        n_filters=n_filters * 1,
        **general_conv3d_block_kwargs
    )
    conv10 = Conv3D(
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
