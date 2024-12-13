import tensorflow as tf
import os
import matplotlib.pyplot as plt
import numpy as np
import keras
from keras import ops
from data import data_preprocessing as dp

from train import generator_g, generator_f
# Load the trained generator models
#generator_g = tf.keras.models.load_model('generator_g.keras')
#generator_f = tf.keras.models.load_model('generator_f.keras')

# Load test datasets
monet_test_ds = dp.monet_test_ds
photo_test_ds = dp.photo_test_ds

# Create directories to save generated images
os.makedirs('generated_images/monet_style', exist_ok=True)
os.makedirs('generated_images/reconstructed_photos', exist_ok=True)

# Function to denormalize images from [-1, 1] to [0, 255]
def denormalize(image):
    image = (image + 1) * 127.5
    return tf.cast(image, tf.uint8)

# Function to save generated images
def save_image(image, path):
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.axis('off')
    plt.savefig(path, bbox_inches='tight')
    plt.close()

print("Start generating Monet-style images...")
for idx, photo in enumerate(photo_test_ds.unbatch().take(12)):
    print(f"Processing photo {idx + 1}")
    photo = tf.expand_dims(photo, axis=0)  # Add batch dimension
    fake_monet = generator_g(photo, training=False)
    print(f"Generated Monet-style image for photo {idx + 1}")
    fake_monet_image = denormalize(fake_monet[0]).numpy()
    save_path = f'generated_images/monet_style/fake_monet_{idx+1}.png'
    save_image(fake_monet_image, save_path)
    print(f"Saved {save_path}")

print("Start generating reconstructed photos...")
for idx, monet in enumerate(monet_test_ds.unbatch().take(10)):
    print(f"Processing Monet painting {idx + 1}")
    monet = tf.expand_dims(monet, axis=0)  # Add batch dimension
    fake_photo = generator_f(monet, training=False)
    print(f"Generated reconstructed photo for Monet painting {idx + 1}")
    fake_photo_image = denormalize(fake_photo[0]).numpy()
    save_path = f'generated_images/reconstructed_photos/fake_photo_{idx+1}.png'
    save_image(fake_photo_image, save_path)
    print(f"Saved {save_path}")

print("Image generation completed.")

