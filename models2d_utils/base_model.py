from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from ..models_utils.metrics import *


class BaseCoderDecoderModel:
    # retrain_decoder - если True, тогда pretrained_weights - веса базовой модели с pretrained_decoder_method
    def __init__(
            self,
            input_size, encoder_part,
            pretrained_decoder_part, decoder_part,
            coder_decoder_connection,
            optimizer, loss='binary_crossentropy', metrics=[BinaryMeanIOU(2, name=None, dtype=None)],
            pretrained_weights=None,
            custom_obj_dict={}
    ):
        inputs = Input(input_size)

        if coder_decoder_connection:
            embedding, conv_stack = encoder_part(inputs)
        else:
            embedding = encoder_part(inputs)
        self.last_encoder_layer_idx = encoder_part.get_last_layer_idx()
        if pretrained_decoder_part:
            if coder_decoder_connection:
                outputs_ = pretrained_decoder_part(
                    embedding=embedding, conv_stack=conv_stack,
                )
            else:
                outputs_ = pretrained_decoder_part(
                    embedding=embedding, conv_stack=None,
                )

            model_ = Model(inputs=inputs, outputs=outputs_)
            model_.compile(optimizer=optimizer, loss=loss, metrics=metrics)

            layers_list = list(zip(model_.layers, reversed(list(range(1, len(model_.layers) + 1)))))
            for i_layer in range(len(layers_list)):
                print(i_layer, layers_list[i_layer])

            # print("----------------------")
            # ToDo - нужно ли обновить состояние оптимизатора
            model_.load_weights(pretrained_weights)
            # model_.load_model(pretrained_weights)
            # model_ = tf.keras.models.load_model(pretrained_weights, custom_obj_dict)
            print("Model was loaded to retrain part of the decoder")
            # print(model_.layers[1].get_weights())

            # Отрезаем декодировщик и переучиваем его с новой реализацией блока декодировщика
            if coder_decoder_connection:
                conv_stack = []
                for idx in encoder_part.idxs_conv_stack_generator():
                    conv_stack.append(model_.layers[idx].output)

            embedding = model_.layers[self.last_encoder_layer_idx].output
            del model_, outputs_

        if coder_decoder_connection:
            outputs = decoder_part(
                embedding=embedding, conv_stack=conv_stack,
            )
        else:
            outputs = decoder_part(
                embedding=embedding, conv_stack=None,
            )
        self.model = Model(inputs=inputs, outputs=outputs)
        self.model.compile(optimizer=optimizer, loss=loss, metrics=metrics)
        # print("----------------------")
        # print(model.layers[1].get_weights())
        if (not pretrained_decoder_part and pretrained_weights):
            self.model.load_weights(pretrained_weights)
            # self.model.load_model(pretrained_weights)
            # self.model = tf.keras.models.load_model(pretrained_weights, custom_objects=custom_obj_dict)
            print("Model was loaded for additional training")

    def freeze_encoder(
            self,
    ):
        for i in range(self.last_encoder_layer_idx + 1):
            self.model.layers[i].trainable = False
            print(i, self.model.layers[i])

        # model = Model(inputs=model_.layers[0].input, outputs=model_.layers[-1].output)
        # print([layer.trainable for layer in self.model.layers])
        # return model

    def unfreeze_encoder(
            self,
    ):
        for i in range(self.last_encoder_layer_idx + 1):
            self.model.layers[i].trainable = True
        # model = Model(inputs=model_.layers[0].input, outputs=model_.layers[-1].output)
        # print([layer.trainable for layer in self.model.layers])
        # return model

    def summary(self, **kwargs):
        return self.model.summary(**kwargs)

    def count_params(self, **kwargs):
        return self.model.count_params(**kwargs)

    def fit(self, **kwargs):
        return self.model.fit(**kwargs)

    def get_optimizer(self):
        return self.model.optimizer
