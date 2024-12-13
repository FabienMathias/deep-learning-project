import tensorflow as tf
import numpy as np
import os

AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 32
IMAGE_SIZE = [256, 256]
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SHUFFLE_BUFFER_SIZE = 1000

current_dir = os.path.dirname(os.path.abspath(__file__))
MONET_JPEG_DIR = os.path.join(current_dir, 'monet_jpg')
PHOTO_JPEG_DIR = os.path.join(current_dir, 'photo_jpg')

def load_image_files(image_dir):
    """Loads image file paths from a directory."""
    image_files = tf.io.gfile.glob(os.path.join(image_dir, '*.jpg'))
    return image_files

def split_dataset(file_list, train_ratio=0.8, val_ratio=0.1):
    """Splits the dataset into training, validation, and testing sets."""
    np.random.shuffle(file_list)
    total_files = len(file_list)
    train_end = int(total_files * train_ratio)
    val_end = train_end + int(total_files * val_ratio)
    train_files = file_list[:train_end]
    val_files = file_list[train_end:val_end]
    test_files = file_list[val_end:]
    return train_files, val_files, test_files

def preprocess_image(image):
    """Resizes and normalizes the image."""
    image = tf.image.resize(image, IMAGE_SIZE)
    image = (image - 127.5) / 127.5  # Normalize to [-1, 1]
    return image

def load_and_preprocess_image(path):
    """Loads an image file and applies preprocessing."""
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = preprocess_image(image)
    return image

def create_dataset(file_paths, shuffle=False):
    """Creates a tf.data.Dataset from file paths with optimized pipeline."""
    dataset = tf.data.Dataset.from_tensor_slices(file_paths)
    if shuffle:
        dataset = dataset.shuffle(SHUFFLE_BUFFER_SIZE)
    dataset = dataset.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset

def count_elements(dataset):
    """Counts the number of batches in a dataset."""
    return sum(1 for _ in dataset)

# Load image file paths
monet_files = load_image_files(MONET_JPEG_DIR)
photo_files = load_image_files(PHOTO_JPEG_DIR)

print(f"Found {len(monet_files)} Monet image files.")
print(f"Found {len(photo_files)} Photo image files.")

# Split datasets
monet_train_files, monet_val_files, monet_test_files = split_dataset(monet_files, TRAIN_RATIO, VAL_RATIO)
photo_train_files, photo_val_files, photo_test_files = split_dataset(photo_files, TRAIN_RATIO, VAL_RATIO)

# Create tf.data.Datasets with optimized pipeline
monet_train_ds = create_dataset(monet_train_files, shuffle=True)
monet_val_ds = create_dataset(monet_val_files)
monet_test_ds = create_dataset(monet_test_files)

photo_train_ds = create_dataset(photo_train_files, shuffle=True)
photo_val_ds = create_dataset(photo_val_files)
photo_test_ds = create_dataset(photo_test_files)

# Count batches
monet_train_count = count_elements(monet_train_ds)
monet_val_count = count_elements(monet_val_ds)
monet_test_count = count_elements(monet_test_ds)

photo_train_count = count_elements(photo_train_ds)
photo_val_count = count_elements(photo_val_ds)
photo_test_count = count_elements(photo_test_ds)

print(f"Monet Dataset - Train batches: {monet_train_count}, Validation batches: {monet_val_count}, Test batches: {monet_test_count}")
print(f"Photo Dataset - Train batches: {photo_train_count}, Validation batches: {photo_val_count}, Test batches: {photo_test_count}")
