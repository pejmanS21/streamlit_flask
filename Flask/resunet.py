"""Residual U-Net implemented here."""
from typing import Tuple
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.optimizers import *
from tensorflow.keras import backend as K
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import ModelCheckpoint
from base_classes import SegmentationModel
import mlflow
import mlflow.tensorflow

"""DICE"""
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)


def dice_coef_loss(y_true, y_pred):
    return 1 - dice_coef(y_true, y_pred)

"""IoU(Jaccard)"""
def jaccard_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (intersection + 1.0) / (K.sum(y_true_f) + K.sum(y_pred_f) - intersection + 1.0)


def jaccard_coef_loss(y_true, y_pred):
    return 1 - jaccard_coef(y_true, y_pred)

"""
    :Custom layers for Res U-Net:
    :BatchNormalization:
"""
def bn_act(x, act=True):
    x = tensorflow.keras.layers.BatchNormalization()(x)
    if act == True:
        x = tensorflow.keras.layers.Activation("relu")(x)
    return x

"""
    :Custom layers for Res U-Net:    
"""
def conv_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = bn_act(x)
    conv = tensorflow.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(conv)
    return conv

"""
    :Custom layers for Res U-Net:  
"""
def stem(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    conv = tensorflow.keras.layers.Conv2D(filters, kernel_size, padding=padding, strides=strides)(x)
    conv = conv_block(conv, filters, kernel_size=kernel_size, padding=padding, strides=strides)

    shortcut = tensorflow.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)

    output = tensorflow.keras.layers.Add()([conv, shortcut])
    return output

"""
    :Custom layers for Res U-Net:
    :Residual layer:   
"""
def residual_block(x, filters, kernel_size=(3, 3), padding="same", strides=1):
    res = conv_block(x, filters, kernel_size=kernel_size, padding=padding, strides=strides)
    res = conv_block(res, filters, kernel_size=kernel_size, padding=padding, strides=1)

    shortcut = tensorflow.keras.layers.Conv2D(filters, kernel_size=(1, 1), padding=padding, strides=strides)(x)
    shortcut = bn_act(shortcut, act=False)

    output = tensorflow.keras.layers.Add()([shortcut, res])
    return output

"""
    :Custom layers for Res U-Net:
    :UpSample layer:   
"""
def upsample_concat_block(x, xskip):
    u = tensorflow.keras.layers.UpSampling2D((2, 2))(x)
    c = tensorflow.keras.layers.Concatenate()([u, xskip])
    return c


"""Residual U-Net implementation"""
class ResUnet_Builder(SegmentationModel):
    """
        :param pretrained_weights: where the pretarined_weight stored
        :param input_size: input shape for the model to be built with
        :Description: get a CXR image with size of (256, 256, 1) and return a mask with same size
        :Result: 99% AUC, 99% Accuracy
    """
    def __init__(self, pretrained_weights: str,
                 input_size: Tuple[int, int, int] = (256, 256, 1)):

        self.pretrained_weights = pretrained_weights
        self.input_size = input_size
        self.model = self.__load_model()


    def __load_model(self):
        f = [16, 32, 64, 128, 256]
        inputs = tensorflow.keras.layers.Input(self.input_size)

        # Encoder
        e0 = inputs
        e1 = stem(e0, f[0])
        e2 = residual_block(e1, f[1], strides=2)
        e3 = residual_block(e2, f[2], strides=2)
        e4 = residual_block(e3, f[3], strides=2)
        e5 = residual_block(e4, f[4], strides=2)

        # Bridge
        b0 = conv_block(e5, f[4], strides=1)
        b1 = conv_block(b0, f[4], strides=1)

        # Decoder
        u1 = upsample_concat_block(b1, e4)
        d1 = residual_block(u1, f[4])

        u2 = upsample_concat_block(d1, e3)
        d2 = residual_block(u2, f[3])

        u3 = upsample_concat_block(d2, e2)
        d3 = residual_block(u3, f[2])

        u4 = upsample_concat_block(d3, e1)
        d4 = residual_block(u4, f[1])

        outputs = tensorflow.keras.layers.Conv2D(1, (1, 1), padding="same", activation="sigmoid")(d4)
        model = tensorflow.keras.models.Model(inputs, outputs)

        metrics = [dice_coef, jaccard_coef,
                'binary_accuracy',
                tf.keras.metrics.Precision(),
                tf.keras.metrics.Recall()]
        
        loss = [dice_coef_loss,
                jaccard_coef_loss,
                'binary_crossentropy']

        adam = tensorflow.keras.optimizers.Adam()
        model.compile(optimizer=adam, loss=loss, metrics=metrics)

        model.load_weights(self.pretrained_weights)

        return model
    """----- mask prediction -----"""
    def predict(self, image):
        return self.model.predict(image)