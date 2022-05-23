import numpy as np
import tensorflow as tf
from keras.metrics import MeanIoU, BinaryCrossentropy


class BinaryMeanIOU(MeanIoU):
    def update_state(self, y_true, y_pred, sample_weight=None, threshold=0.5):
        y_pred = tf.where(condition=tf.math.greater(y_pred, threshold), x=1.0, y=0.0)
        return super().update_state(y_true, y_pred, sample_weight)


class BinaryMeanIOU_(MeanIoU):  # Только для версии 2.6, в версии 2.8 metrics притерпела заметные изменения
    def __init__(self, num_classes, name=None, dtype=None, threshold=0.5):
        self.threshold = threshold

        super(BinaryMeanIOU_, self).__init__(num_classes, name, dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.where(condition=tf.math.greater(y_pred, self.threshold), x=1.0, y=0.0)
        return super().update_state(y_true, y_pred, sample_weight)


class TryBinaryMeanIOU2D(MeanIoU):
    def __init__(self, num_classes, name=None, dtype=None, threshold=0.5):
        self.threshold = threshold

        super(TryBinaryMeanIOU2D, self).__init__(num_classes, name, dtype)

    # ToDo - проверить на большем размере батча
    def update_state(self, y_true, y_pred, sample_weight=None):
        format = tf.keras.backend.image_data_format()
        if format == "channels_first":
            depth_dim = 4
        else:
            depth_dim = 3

        y_pred = tf.where(condition=tf.math.greater(y_pred, self.threshold), x=1.0, y=0.0)

        vals_by_batch_volumes = []
        vals_by_batch_volumes_ = []
        vals_by_batch_volumes__ = []
        # ----------------------------------------------
        print("Посчитаем без сброса по кажлому объему в батче")
        for ivolume in range(y_true.shape[0]):
            valls_by_img = []

            for iimg in range(y_true.shape[depth_dim]):
                if format == "channels_first":
                    super().update_state(y_true[ivolume, :, :, :, iimg], y_pred[ivolume, :, :, :, iimg], sample_weight)
                else:
                    super().update_state(y_true[ivolume, :, :, iimg, :], y_pred[ivolume, :, :, iimg, :], sample_weight)

                val = float(super().result())
                valls_by_img.append(val)
            vals_by_batch_volumes.append(np.mean(valls_by_img))
            vals_by_batch_volumes_.extend(valls_by_img)
            vals_by_batch_volumes__.append(val)

        print("result by volumes:", f"{np.mean(vals_by_batch_volumes):.16f}")
        print("result by imgs in volumes:", f"{np.mean(vals_by_batch_volumes_):.16f}")
        print("result by imgs in volumes_:", f"{np.mean(vals_by_batch_volumes__):.16f}")

        print("Посчитаем со сбросом по снимкам")
        for ivolume in range(y_true.shape[0]):
            valls_by_img = []

            for iimg in range(y_true.shape[depth_dim]):
                super().reset_state()
                if format == "channels_first":
                    super().update_state(y_true[ivolume, :, :, :, iimg], y_pred[ivolume, :, :, :, iimg], sample_weight)
                else:
                    super().update_state(y_true[ivolume, :, :, iimg, :], y_pred[ivolume, :, :, iimg, :], sample_weight)

                val = float(super().result())
                valls_by_img.append(val)
            vals_by_batch_volumes.append(np.mean(valls_by_img))
            vals_by_batch_volumes_.extend(valls_by_img)

        print("result by volumes:", f"{np.mean(vals_by_batch_volumes):.16f}")
        print("result by imgs in volumes:", f"{np.mean(vals_by_batch_volumes_):.16f}")


class BinaryLoss(BinaryCrossentropy):
    def __init__(
            self,
            name='binarized_binary_crossentropy',
            dtype=None,
            from_logits=False,
            label_smoothing=0,
            threshold=0.5
    ):
        self.threshold = threshold

        super(BinaryLoss, self).__init__(
            name=name,
            dtype=dtype,
            from_logits=from_logits,
            label_smoothing=label_smoothing
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.where(condition=tf.math.greater(y_pred, self.threshold), x=1.0, y=0.0)
        return super().update_state(y_true, y_pred, sample_weight)


