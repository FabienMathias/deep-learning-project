import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras import ops
from data import data_preprocessing as dp

from train import generator_g, generator_f

# Load the trained generator models
# generator_g = tf.keras.models.load_model('generator_g.keras')
# generator_f = tf.keras.models.load_model('generator_f.keras')

# Load test datasets
#monet_test_ds = dp.monet_test_ds
#photo_test_ds = dp.photo_test_ds
# Load the photo test file paths
photo_test_files = dp.photo_test_files
print(photo_test_files)

# Create a dataset with images and paths
def create_test_dataset(file_paths):
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    dataset = dataset.map(lambda path: (dp.load_and_preprocess_image(path), path))
    return dataset

photo_test_ds = create_test_dataset(photo_test_files)

# Create directories to save generated images
os.makedirs('generated_images/monet_style', exist_ok=True)
os.makedirs('generated_images/reconstructed_photos', exist_ok=True)

# Function to denormalize images from [-1, 1] to [0, 255]
def denormalize(image):
    if isinstance(image, tf.Tensor):
        image = image.numpy()  # Convert TensorFlow tensor to NumPy array
    image = (image + 1) * 127.5  # Scale to [0, 255]
    return np.clip(image, 0, 255).astype(np.uint8)

# Function to save images directly
def save_image(image, path):
    tf.keras.preprocessing.image.save_img(path, image)

# Generate Monet-style images from photos
print("Generating Monet-style images from photos...")
for idx, photo in enumerate(photo_test_ds.unbatch().take(15)):
    print(f"Processing photo {idx + 1}")
    photo = tf.expand_dims(photo, axis=0)  # Add batch dimension
    fake_monet = generator_g(photo, training=False)
    fake_monet_image = denormalize(fake_monet[0])  # Process single image

    # Debugging print
    print(f"Type of fake_monet_image: {type(fake_monet_image)}")
    print(f"Shape of fake_monet_image: {fake_monet_image.shape}")

    # Save the generated image
    save_path = f'generated_images/monet_style/fake_monet_{idx+1}.png'
    save_image(fake_monet_image, save_path)
    print(f"Saved {save_path}")
