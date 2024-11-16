import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, BatchNormalization
from tensorflow.keras.models import Model

def build_discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    inputs = Input(shape=(128, 128, 3))

    # C64
    x = Conv2D(32, kernel_size=4, strides=2, padding='same', kernel_initializer=initializer)(inputs)
    x = LeakyReLU(0.2)(x)

    # C128
    x = Conv2D(64, kernel_size=4, strides=2, padding='same', kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    # C256
    x = Conv2D(128, kernel_size=4, strides=2, padding='same', kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    # C512
    x = Conv2D(256, kernel_size=4, strides=1, padding='same', kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU(0.2)(x)

    # Output layer
    x = Conv2D(1, kernel_size=4, strides=1, padding='same', kernel_initializer=initializer)(x)

    # The output is a patch of size (N x N x 1)
    model = Model(inputs, x)
    return model
