from tensorflow.keras.callbacks import Callback
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)


def get_model_name(
        model_name,
        n_filters, separable_conv, shortcut_connection,
        conv_bias, bn_before_act, decoder_method, upconv_bias,
        global_random_seed, init_random_seed
):
    return f'{model_name}__{n_filters}nf_' \
           f'{"sepConv_" if separable_conv else ""}' \
           f'{"shortCon_" if shortcut_connection else ""}' \
           "relu_" \
           f'{"convBias_" if conv_bias else "bnBeforeAct_" if bn_before_act else ""}' \
           f'{decoder_method}_' \
           f'{"upconv_bias" if upconv_bias else ""}' \
           f'__BCE_loss__{global_random_seed}grs_{init_random_seed}irs'


class EarlyStoppingByLossVal(Callback):
    def __init__(self, monitor='val_loss', value=0.00001, verbose=0):
        super(Callback, self).__init__()
        self.monitor = monitor
        self.value = value
        self.verbose = verbose

    def on_epoch_end(self, epoch, logs={}):
        current = logs.get(self.monitor)
        if current is None:
            warnings.warn("Early stopping requires %s available!" % self.monitor, RuntimeWarning)

        if current < self.value:
            if self.verbose > 0:
                print("Epoch %05d: early stopping THR" % epoch)
            self.model.stop_training = True
