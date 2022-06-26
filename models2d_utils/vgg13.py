from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Concatenate
from .blocks import *


class VGG13:
    class Encoder:
        def __init__(
                self, n_layers=5,
                n_filters=64, init_seed=7,
                separable_conv=False, shortcut_connection=False,
                bn_before_act=False, use_conv_bias=False,
                last_max_pooling=False, ret_conv_stack=False,
        ):
            # ToDo
            assert n_layers > 0, ""

            self.n_layers = n_layers
            self.n_filters = n_filters
            self.init_seed = init_seed
            self.separable_conv = separable_conv
            self.shortcut_connection = shortcut_connection
            self.bn_before_act = bn_before_act
            self.use_conv_bias = use_conv_bias
            self.last_max_pooling = last_max_pooling
            self.ret_conv_stack = ret_conv_stack

            self.general_conv2d_block_kwargs = {
                "num_layers": 2,
                "kernel_size": 3,
                "use_conv_bias": self.use_conv_bias,
                "separable_conv": self.separable_conv,
                "res_block": self.shortcut_connection,
                "bn_before_act": self.bn_before_act,
                "init_seed": self.init_seed,
            }

            self.step = 6 if self.bn_before_act else 4  # 2 BatchNormalization objects
            self.step += 2 if self.shortcut_connection else 0  # PointWiseConv and core.TFOpLambda (Add?) objects
            self.step += 1 if self.separable_conv and not self.shortcut_connection else 0  # additional PointWiseConv

        def __call__(self, inputs):
            if self.ret_conv_stack:
                conv_stack = []
            outputs = inputs
            del inputs

            for i_layer in range(self.n_layers - 1):
                conv = conv2d_block(
                    outputs,
                    n_filters=self.n_filters * 2 ** i_layer,
                    **self.general_conv2d_block_kwargs
                )
                if self.ret_conv_stack:
                    conv_stack.append(conv)
                outputs = MaxPooling2D(pool_size=(2, 2))(conv)

            # n-layer
            embedding = conv2d_block(
                outputs,
                n_filters=self.n_filters * 2 ** (self.n_layers - 1),
                **self.general_conv2d_block_kwargs
            )

            if self.last_max_pooling:
                if self.ret_conv_stack:
                    conv_stack.append(embedding)
                embedding = MaxPooling2D(pool_size=(2, 2))(embedding)
            if self.ret_conv_stack:
                return embedding, conv_stack
            else:
                return embedding

        def idxs_conv_stack_generator(self):
            # Берем activation-layer перед pooling
            for i_layer in range(1, self.n_layers):
                print(i_layer, self.step * i_layer + (i_layer - 1))
                yield self.step * i_layer + (i_layer - 1)

        def get_last_layer_idx(self):
            idx = self.step * self.n_layers + (self.n_layers - 1)
            print("embedding idx is", idx)
            if self.last_max_pooling:
                return idx + 1
            else:
                return idx

    class Decoder:
        def __init__(
                self, n_layers=5,
                n_filters=64, init_seed=7,
                separable_conv=False, shortcut_connection=False,
                bn_before_act=False, use_conv_bias=False,
                decoder_method='nearest', use_upconv_bias=False,
        ):
            # ToDo
            assert n_layers > 0, ""

            self.n_layers = n_layers
            self.n_filters = n_filters
            self.init_seed = init_seed
            self.separable_conv = separable_conv
            self.shortcut_connection = shortcut_connection
            self.bn_before_act = bn_before_act
            self.use_conv_bias = use_conv_bias
            self.decoder_method = decoder_method
            self.use_upconv_bias = use_upconv_bias

            self.general_conv2d_block_kwargs = {
                "num_layers": 2,
                "kernel_size": 3,
                "use_conv_bias": self.use_conv_bias,
                "separable_conv": self.separable_conv,
                "res_block": self.shortcut_connection,
                "bn_before_act": self.bn_before_act,
                "init_seed": self.init_seed,
            }

            self.general_decode2d_block_kwargs = {
                "decoder_method": self.decoder_method,
                "dec_conv_bias": self.use_upconv_bias,
                "init_seed": self.init_seed
            }

        def __call__(self, embedding, conv_stack):
            outputs = embedding
            for i_layer in range(self.n_layers, 0, -1):
                up = decode2d_block(
                    outputs,
                    input_n_filters=self.n_filters * 2 ** i_layer,
                    **self.general_decode2d_block_kwargs
                )
                if conv_stack:
                    up = Concatenate(axis=3)([conv_stack.pop(), up])

                outputs = conv2d_block(
                    up,
                    n_filters=self.n_filters * 2 ** (i_layer - 1),
                    **self.general_conv2d_block_kwargs
                )

            outputs = Conv2D(
                filters=1,
                kernel_size=(1, 1),
                strides=(1, 1),
                padding='same',
                activation='sigmoid',
                use_bias=True,  # ToDo
                kernel_initializer=tf.keras.initializers.HeNormal(seed=self.init_seed),
                bias_initializer='zeros'
            )(outputs)

            return outputs
