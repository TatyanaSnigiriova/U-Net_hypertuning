import tensorflow as tf

'''
    В tensorflow нет реализации для слоя SubpixelConv2d из tensorlayers, 
    но так как все слои tensorlayers реализованы на операциях tensorflow, 
    мы можем использовать их определение слоя SubpixelConv2d и немного исправить его под свои нужды.
    Source:
    https://tensorlayer.readthedocs.io/en/latest/_modules/tensorlayer/layers/convolution/super_resolution.html#SubpixelConv2d
    Я не стала называть слой SubpixelConv2D, т.к. слой не имеет обучаемых параметров
'''


class DepthToSpace2D(tf.keras.layers.Layer):
    def __init__(
            self,
            scale=2,
            n_out_channels=None,
            in_channels=None,
            name=None  # 'subpixel_conv2d'
    ):
        super(DepthToSpace2D, self).__init__(name=name)
        self.scale = scale
        self.n_out_channels = n_out_channels
        self.in_channels = in_channels

        if self.in_channels is not None:
            self.build(None)
            self._built = True

    def build(self, inputs_shape):
        if inputs_shape is not None:
            self.in_channels = inputs_shape[-1]

        if self.in_channels / (self.scale ** 2) % 1 != 0:
            raise Exception(
                "SubpixelConv2D: The number of input channels == (scale x scale) x The number of output channels"
            )
        self.n_out_channels = int(self.in_channels / (self.scale ** 2))

    def call(self, inputs):
        outputs = self._PS(X=inputs, r=self.scale, n_out_channels=self.n_out_channels)
        return outputs

    def _PS(self, X, r, n_out_channels):

        _err_log = "SubpixelConv2D: The number of input channels == (scale x scale) x The number of output channels"

        if n_out_channels >= 1:
            if int(X.get_shape()[-1]) != (r ** 2) * n_out_channels:
                raise Exception(_err_log)

            X = tf.nn.depth_to_space(input=X, block_size=r)
        else:
            raise RuntimeError(_err_log)
        return X

    def get_config(self):
        config = super().get_config().copy()
        config.update({
            'scale': self.scale,
            'n_out_channels': self.n_out_channels,
            'in_channels': self.in_channels,
        })
        return config

    @classmethod
    def from_config(cls, config):
        return cls(**config)
