import os
import tensorflow as tf
from generators_final import build_generator
from tensorflow.keras.preprocessing import image as keras_image
import numpy as np
from PIL import Image

# Constants
SAVED_MODELS_DIR = './saved_models'
GENERATOR_G_PATH = os.path.join(SAVED_MODELS_DIR, 'generator_g')
OUTPUT_DIR = './output'
IMAGE_SIZE = (256, 256)

# Ensure Output Directory Exists
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Preprocessing Function
def preprocess_image(image_path):
    """
    Loads and preprocesses an image.
    Args:
        image_path (str): Path to the input JPEG image.
    Returns:
        tf.Tensor: Preprocessed image tensor.
    """
    img = Image.open(image_path).convert('RGB')
    img = img.resize(IMAGE_SIZE, Image.BICUBIC)
    img = np.array(img).astype(np.float32)
    # Normalize to [-1, 1]
    img = (img - 127.5) / 127.5
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    return tf.convert_to_tensor(img)

# Postprocessing Function
def postprocess_image(img_tensor):
    """
    Converts a tensor back to a PIL Image.
    Args:
        img_tensor (tf.Tensor): Output tensor from the generator.
    Returns:
        PIL.Image.Image: Postprocessed image.
    """
    img = img_tensor[0].numpy()
    img = (img * 127.5) + 127.5  # Denormalize to [0, 255]
    img = np.clip(img, 0, 255).astype(np.uint8)
    return Image.fromarray(img)

# Load Generator Model
def load_generator():
    """
    Loads the trained generator_g model.
    Returns:
        tf.keras.Model: Loaded generator model.
    """
    if not os.path.exists(GENERATOR_G_PATH):
        raise FileNotFoundError(f"Generator model not found at {GENERATOR_G_PATH}. Please train the model first.")
    generator_g = tf.keras.models.load_model(GENERATOR_G_PATH, compile=False)
    print(f"Generator model loaded from {GENERATOR_G_PATH}.")
    return generator_g

# Generate Monet-style Image
def generate_monet_image(input_image_path, output_image_path):
    """
    Transforms an input JPEG image into a Monet-style painting.
    Args:
        input_image_path (str): Path to the input JPEG image.
        output_image_path (str): Path to save the generated Monet-style image.
    """
    # Load and preprocess the input image
    input_tensor = preprocess_image(input_image_path)

    # Load the generator model
    generator_g = load_generator()

    # Generate the Monet-style image
    fake_y = generator_g(input_tensor, training=False)

    # Postprocess the output image
    output_image = postprocess_image(fake_y)

    # Save the output image
    output_image.save(output_image_path)
    print(f"Generated Monet-style image saved to {output_image_path}.")

# Main Function
def main():
    import argparse

    parser = argparse.ArgumentParser(description='Transform a photo into a Monet-style painting using CycleGAN.')
    parser.add_argument('input_image', type=str, help='Path to the input JPEG image.')
    parser.add_argument('--output', type=str, default=None,
                        help='Path to save the generated image. If not specified, saves as "output_<input_filename>".')

    args = parser.parse_args()

    input_image_path = args.input_image
    if not os.path.isfile(input_image_path):
        print(f"Input image file {input_image_path} does not exist.")
        return

    if args.output:
        output_image_path = args.output
    else:
        input_filename = os.path.basename(input_image_path)
        name, ext = os.path.splitext(input_filename)
        output_image_path = os.path.join(OUTPUT_DIR, f"output_{name}.jpg")

    generate_monet_image(input_image_path, output_image_path)

if __name__ == '__main__':
    main()