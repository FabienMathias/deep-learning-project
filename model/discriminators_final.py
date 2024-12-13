import tensorflow as tf
from tensorflow.keras import layers, Model
from model.layers import InstanceNormalization

def build_discriminator(image_size=256):
    """
    Builds a PatchGAN discriminator model as per CycleGAN architecture.
    Args:
        image_size (int): Height and width of the input images.
    Returns:
        A Keras Model representing the discriminator.
    """
    initializer = tf.random_normal_initializer(0., 0.02)
    inputs = layers.Input(shape=(image_size, image_size, 3))

    # Layer 1: C64
    x = layers.Conv2D(64, kernel_size=4, strides=2, padding='same', kernel_initializer=initializer, use_bias=True)(inputs)
    x = layers.LeakyReLU(0.2)(x)

    # Layer 2: C128
    x = layers.Conv2D(128, kernel_size=4, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(x)
    x = InstanceNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    # Layer 3: C256
    x = layers.Conv2D(256, kernel_size=4, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(x)
    x = InstanceNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    # Layer 4: C512
    x = layers.Conv2D(512, kernel_size=4, strides=1, padding='same', kernel_initializer=initializer, use_bias=False)(x)
    x = InstanceNormalization()(x)
    x = layers.LeakyReLU(0.2)(x)

    # Output Layer
    x = layers.Conv2D(1, kernel_size=4, strides=1, padding='same', kernel_initializer=initializer)(x)

    return Model(inputs, x, name='PatchGANDiscriminator')

if __name__ == '__main__':
    discriminator = build_discriminator(image_size=256)
    discriminator.summary()
