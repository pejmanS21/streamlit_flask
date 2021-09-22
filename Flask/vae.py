import numpy as np
import cv2
import os
from tqdm import tqdm
import tensorflow as tf
import tensorflow.keras
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
# from tensorflow.keras import backend as K
# import tensorflow.compat.v1.keras.backend as K
# tf.compat.v1.disable_eager_execution()
import matplotlib.pyplot as plt

input_shape = (256, 256, 1)
latent_dim = 2


def decoder(pretrained_weights=None):
    latent_dim = 2
    decoder_input = Input(shape=(latent_dim, ), name='decoder_input')

    x = Dense(128 * 128 * 128, activation='relu')(decoder_input)
    x = Dropout(0.1)(x)
    x = Reshape((128, 128, 128))(x)

    x = Conv2DTranspose(64, (3, 3), padding='same', activation='relu', strides=(2, 2))(x)
    x = Conv2DTranspose(1, (3, 3), padding='same', activation='sigmoid', name='decoder_output')(x)
    model = Model(decoder_input, x, name='decoder')

    if pretrained_weights:
        model.load_weights(pretrained_weights)

    return model

