import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, ReLU, Add, Activation, BatchNormalization
from tensorflow.keras.models import Model

def resnet_block(x, filters, kernel_size=3):
    initializer = tf.random_normal_initializer(0., 0.02)

    # First convolutional layer
    y = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same', kernel_initializer=initializer)(x)
    y = BatchNormalization()(y)
    y = ReLU()(y)

    # Second convolutional layer
    y = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same', kernel_initializer=initializer)(y)
    y = BatchNormalization()(y)

    # Skip connection
    y = Add()([y, x])
    return y

def build_generator():
    initializer = tf.random_normal_initializer(0., 0.02)
    inputs = Input(shape=(128, 128, 3))

    # Downsampling layers
    x = Conv2D(32, kernel_size=7, strides=1, padding='same', kernel_initializer=initializer)(inputs)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(64, kernel_size=3, strides=2, padding='same', kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(128, kernel_size=3, strides=2, padding='same', kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Residual blocks
    for _ in range(4):
        x = resnet_block(x, 128)

    # Upsampling layers
    x = Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    x = Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', kernel_initializer=initializer)(x)
    x = BatchNormalization()(x)
    x = ReLU()(x)

    # Output layer
    x = Conv2D(3, kernel_size=7, strides=1, padding='same', kernel_initializer=initializer)(x)
    outputs = Activation('tanh')(x)

    # Define the model
    model = Model(inputs, outputs)
    return model
