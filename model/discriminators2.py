import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, LeakyReLU, Layer
from tensorflow.keras.models import Model

class InstanceNormalization(Layer):
    def __init__(self, epsilon=1e-5, **kwargs):
        super(InstanceNormalization, self).__init__(**kwargs)
        self.epsilon = epsilon

    def build(self, input_shape):
        self.scale = self.add_weight(
            name='scale',
            shape=(input_shape[-1],),
            initializer=tf.random_normal_initializer(1., 0.02),
            trainable=True)
        self.offset = self.add_weight(
            name='offset',
            shape=(input_shape[-1],),
            initializer='zeros',
            trainable=True)
        super(InstanceNormalization, self).build(input_shape)

    def call(self, inputs):
        mean, variance = tf.nn.moments(inputs, axes=[1, 2], keepdims=True)
        inv = tf.math.rsqrt(variance + self.epsilon)
        normalized = (inputs - mean) * inv
        return self.scale * normalized + self.offset

    def get_config(self):
        config = super(InstanceNormalization, self).get_config()
        config.update({'epsilon': self.epsilon})
        return config


def build_discriminator():
    initializer = tf.random_normal_initializer(0., 0.02)
    inputs = Input(shape=(128, 128, 3))

    # C64
    x = Conv2D(32, kernel_size=4, strides=2, padding='same',
               kernel_initializer=initializer)(inputs)
    x = LeakyReLU(0.2)(x)

    # C128
    x = Conv2D(64, kernel_size=4, strides=2, padding='same',
               kernel_initializer=initializer)(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(0.2)(x)

    # C256
    x = Conv2D(128, kernel_size=4, strides=2, padding='same',
               kernel_initializer=initializer)(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(0.2)(x)

    # C512
    x = Conv2D(256, kernel_size=4, strides=1, padding='same',
               kernel_initializer=initializer)(x)
    x = InstanceNormalization()(x)
    x = LeakyReLU(0.2)(x)

    # Output layer
    x = Conv2D(1, kernel_size=4, strides=1, padding='same',
               kernel_initializer=initializer)(x)

    # The output is a patch of size (N x N x 1)
    model = Model(inputs, x)
    return model

# Example usage
if __name__ == '__main__':
    discriminator = build_discriminator()
    discriminator.summary()