import tensorflow as tf
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, ReLU, Add, Activation, Layer
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

def resnet_block(x, filters, kernel_size=3):
    initializer = tf.random_normal_initializer(0., 0.02)

    # First convolutional layer
    y = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same', kernel_initializer=initializer)(x)
    y = InstanceNormalization()(y)
    y = ReLU()(y)

    # Second convolutional layer
    y = Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same', kernel_initializer=initializer)(y)
    y = InstanceNormalization()(y)

    # Skip connection
    y = Add()([y, x])
    return y

def build_generator():
    initializer = tf.random_normal_initializer(0., 0.02)
    inputs = Input(shape=(128, 128, 3))

    # Downsampling layers
    x = Conv2D(32, kernel_size=7, strides=1, padding='same', kernel_initializer=initializer)(inputs)
    x = InstanceNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(64, kernel_size=3, strides=2, padding='same', kernel_initializer=initializer)(x)
    x = InstanceNormalization()(x)
    x = ReLU()(x)

    x = Conv2D(128, kernel_size=3, strides=2, padding='same', kernel_initializer=initializer)(x)
    x = InstanceNormalization()(x)
    x = ReLU()(x)

    # Residual blocks
    for _ in range(4):
        x = resnet_block(x, 128)

    # Upsampling layers
    x = Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', kernel_initializer=initializer)(x)
    x = InstanceNormalization()(x)
    x = ReLU()(x)

    x = Conv2DTranspose(32, kernel_size=3, strides=2, padding='same', kernel_initializer=initializer)(x)
    x = InstanceNormalization()(x)
    x = ReLU()(x)

    # Output layer
    x = Conv2D(3, kernel_size=7, strides=1, padding='same', kernel_initializer=initializer)(x)
    outputs = Activation('tanh')(x)

    # Define the model
    model = Model(inputs, outputs)
    return model

def build_simple_generator():
    inputs = Input(shape=(128, 128, 3))
    x = Conv2D(32, kernel_size=7, strides=1, padding='same')(inputs)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(64, kernel_size=3, strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2DTranspose(32, kernel_size=3, strides=2, padding='same')(x)
    x = LeakyReLU(0.2)(x)
    x = Conv2D(3, kernel_size=7, strides=1, padding='same')(x)
    outputs = Activation('tanh')(x)
    model = Model(inputs, outputs)
    return model

# Example usage
if __name__ == '__main__':
    generator = build_generator()
    generator.summary()