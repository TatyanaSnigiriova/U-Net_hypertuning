from tensorflow.keras.optimizers import Adam

from ..models_utils.metrics import *
from ..models2d_utils.vgg16 import *
from ..models2d_utils.base_model import *


class SegNet2D(BaseCoderDecoderModel):
    model_type_name = "SegNet2D"

    def __init__(
            self,
            input_size=(None, None, 1),
            deep=5,
            n_filters=64, init_seed=7,
            separable_conv=False, shortcut_connection=False,
            bn_before_act=False, use_conv_bias=False,
            decoder_method='nearest', use_upconv_bias=False,
            retrain_decoder=False,
            pretrained_weights=None,
            pretrained_decoder_method='nearest', pretrained_use_upconv_bias=False,
            optimizer=Adam(lr=1e-4, beta_1=0.9, beta_2=0.999, epsilon=1e-07),
            loss='binary_crossentropy', metrics=[BinaryMeanIOU(2, name=None, dtype=None)],
            custom_obj_dict={},
    ):
        self.input_size = input_size
        self.deep = deep
        self.ret_conv_stack = False
        if retrain_decoder:
            assert pretrained_weights, \
                "You must specify the pre-trained model weights for transfer learning."

        # ToDo - UnPooling operation
        self.encoder_part = VGG16.Encoder(
            n_layers=self.deep,
            n_filters=n_filters, init_seed=init_seed,
            separable_conv=separable_conv, shortcut_connection=shortcut_connection,
            bn_before_act=bn_before_act, use_conv_bias=use_conv_bias,
            last_max_pooling=True, ret_conv_stack=self.ret_conv_stack
        )

        self.decoder_part = VGG16.Decoder(
            n_layers=self.deep,
            n_filters=n_filters, init_seed=init_seed,
            separable_conv=separable_conv, shortcut_connection=shortcut_connection,
            bn_before_act=bn_before_act, use_conv_bias=use_conv_bias,
            decoder_method=decoder_method, use_upconv_bias=use_upconv_bias,
        )

        if retrain_decoder:  # and pretrained_weights:
            self.pretrained_decoder_part = VGG16.Decoder(
                n_layers=self.deep,
                n_filters=n_filters, init_seed=init_seed,
                separable_conv=separable_conv, shortcut_connection=shortcut_connection,
                bn_before_act=bn_before_act, use_conv_bias=use_conv_bias,
                decoder_method=pretrained_decoder_method, use_upconv_bias=pretrained_use_upconv_bias,
            )
        else:
            self.pretrained_decoder_part = None

        super().__init__(
            input_size=self.input_size, encoder_part=self.encoder_part,
            pretrained_decoder_part=self.pretrained_decoder_part, decoder_part=self.decoder_part,
            coder_decoder_connection=self.ret_conv_stack,
            optimizer=optimizer, loss=loss, metrics=metrics,
            pretrained_weights=pretrained_weights,
            custom_obj_dict=custom_obj_dict,
        )
