from tensorflow.keras.callbacks import Callback
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

def freezeEncoder(
        model_,
        bn_before_act=False,
):
    step = 6 if bn_before_act else 4
    for i in range(step * 5 + 5):
        model_.layers[i].trainable = False

    # model = Model(inputs=model_.layers[0].input, outputs=model_.layers[-1].output)
    print([layer.trainable for layer in model_.layers])
    # return model


def unfreezeEncoder(
        model_,
        bn_before_act=False,
):
    step = 6 if bn_before_act else 4
    for i in range(step * 5 + 5):
        model_.layers[i].trainable = True
    # model = Model(inputs=model_.layers[0].input, outputs=model_.layers[-1].output)
    print([layer.trainable for layer in model_.layers])
    # return model


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