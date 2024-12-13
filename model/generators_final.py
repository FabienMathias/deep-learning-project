import tensorflow as tf
from tensorflow.keras import layers, Model
from model.layers import InstanceNormalization

def resnet_block(x, filters=256, kernel_size=3):
    """
    Defines a ResNet block with two convolutional layers and a skip connection.
    """
    initializer = tf.random_normal_initializer(0., 0.02)

    # First convolutional layer
    y = layers.Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same', kernel_initializer=initializer, use_bias=False)(x)
    y = InstanceNormalization()(y)
    y = layers.ReLU()(y)

    # Second convolutional layer
    y = layers.Conv2D(filters, kernel_size=kernel_size, strides=1, padding='same', kernel_initializer=initializer, use_bias=False)(y)
    y = InstanceNormalization()(y)

    # Add skip connection
    return layers.Add()([y, x])


def build_generator(image_size=256, num_res_blocks=9):
    """
    Builds a ResNet-based generator model as per CycleGAN architecture.
    Args:
        image_size (int): Height and width of the input images.
        num_res_blocks (int): Number of residual blocks (9 for 256x256 images).
    Returns:
        A Keras Model representing the generator.
    """
    initializer = tf.random_normal_initializer(0., 0.02)
    inputs = layers.Input(shape=(image_size, image_size, 3))

    # Initial Convolutional Layer
    x = layers.Conv2D(64, kernel_size=7, strides=1, padding='same', kernel_initializer=initializer, use_bias=False)(inputs)
    x = InstanceNormalization()(x)
    x = layers.ReLU()(x)

    # Downsampling Layers
    x = layers.Conv2D(128, kernel_size=3, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(x)
    x = InstanceNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2D(256, kernel_size=3, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(x)
    x = InstanceNormalization()(x)
    x = layers.ReLU()(x)

    # Residual Blocks
    for _ in range(num_res_blocks):
        x = resnet_block(x, filters=256, kernel_size=3)

    # Upsampling Layers
    x = layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(x)
    x = InstanceNormalization()(x)
    x = layers.ReLU()(x)

    x = layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding='same', kernel_initializer=initializer, use_bias=False)(x)
    x = InstanceNormalization()(x)
    x = layers.ReLU()(x)

    # Output Layer
    x = layers.Conv2D(3, kernel_size=7, strides=1, padding='same', kernel_initializer=initializer, use_bias=False)(x)
    outputs = layers.Activation('tanh')(x)

    return Model(inputs, outputs, name='ResNetGenerator')

if __name__ == '__main__':
    # Example usage: build a generator and print its summary
    generator = build_generator(image_size=256, num_res_blocks=9)
    generator.summary()
