import tensorflow as tf
import os
import time
import matplotlib.pyplot as plt

from data import data_preprocessing as dp
from generators import build_generator
from discriminators import build_discriminator

# Load datasets
monet_train_ds = dp.monet_train_ds
photo_train_ds = dp.photo_train_ds

# Instantiate models
generator_g = build_generator()  # Transforms photos to Monet-style paintings
generator_f = build_generator()  # Transforms Monet paintings to photos
discriminator_x = build_discriminator()  # Discriminator for photos
discriminator_y = build_discriminator()  # Discriminator for Monet paintings

# Define loss functions
loss_obj = tf.keras.losses.MeanSquaredError()

def discriminator_loss(real, generated):
    real_loss = loss_obj(tf.ones_like(real), real)
    generated_loss = loss_obj(tf.zeros_like(generated), generated)
    total_disc_loss = (real_loss + generated_loss) * 0.5
    return total_disc_loss

def generator_loss(generated):
    return loss_obj(tf.ones_like(generated), generated)

def cycle_consistency_loss(real_image, cycled_image):
    return tf.reduce_mean(tf.abs(real_image - cycled_image))

def identity_loss(real_image, same_image):
    return tf.reduce_mean(tf.abs(real_image - same_image))

# Define optimizers
generator_g_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
generator_f_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_x_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)
discriminator_y_optimizer = tf.keras.optimizers.Adam(2e-4, beta_1=0.5)

# Checkpoints to save model weights
checkpoint_path = "./checkpoints"
ckpt = tf.train.Checkpoint(
    generator_g=generator_g,
    generator_f=generator_f,
    discriminator_x=discriminator_x,
    discriminator_y=discriminator_y,
    generator_g_optimizer=generator_g_optimizer,
    generator_f_optimizer=generator_f_optimizer,
    discriminator_x_optimizer=discriminator_x_optimizer,
    discriminator_y_optimizer=discriminator_y_optimizer
)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# Load latest checkpoint if available
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!')

# Training step
@tf.function
def train_step(real_x, real_y):
    # Persistent is set to True because the tape is used more than once to calculate gradients
    with tf.GradientTape(persistent=True) as tape:
        # Generator G translates X -> Y
        fake_y = generator_g(real_x, training=True)
        # Generator F translates Y -> X
        fake_x = generator_f(real_y, training=True)

        # Cycle consistency
        cycled_x = generator_f(fake_y, training=True)
        cycled_y = generator_g(fake_x, training=True)

        # Identity mapping
        same_x = generator_f(real_x, training=True)
        same_y = generator_g(real_y, training=True)

        # Discriminator output
        disc_real_x = discriminator_x(real_x, training=True)
        disc_real_y = discriminator_y(real_y, training=True)
        disc_fake_x = discriminator_x(fake_x, training=True)
        disc_fake_y = discriminator_y(fake_y, training=True)

        # Generator adversarial losses
        gen_g_loss = generator_loss(disc_fake_y)
        gen_f_loss = generator_loss(disc_fake_x)

        # Cycle consistency losses
        total_cycle_loss = cycle_consistency_loss(real_x, cycled_x) + cycle_consistency_loss(real_y, cycled_y)

        # Identity losses
        id_loss_x = identity_loss(real_x, same_x)
        id_loss_y = identity_loss(real_y, same_y)

        # Total generator losses
        total_gen_g_loss = gen_g_loss + (10.0 * total_cycle_loss) + (5.0 * id_loss_y)
        total_gen_f_loss = gen_f_loss + (10.0 * total_cycle_loss) + (5.0 * id_loss_x)

        # Discriminator losses
        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    # Calculate gradients
    generator_g_gradients = tape.gradient(total_gen_g_loss, generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss, generator_f.trainable_variables)

    discriminator_x_gradients = tape.gradient(disc_x_loss, discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss, discriminator_y.trainable_variables)

    # Apply gradients
    generator_g_optimizer.apply_gradients(zip(generator_g_gradients, generator_g.trainable_variables))
    generator_f_optimizer.apply_gradients(zip(generator_f_gradients, generator_f.trainable_variables))

    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients, discriminator_x.trainable_variables))
    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients, discriminator_y.trainable_variables))

# Training loop
EPOCHS = 5

for epoch in range(EPOCHS):
    start = time.time()
    print(f"Epoch {epoch+1}/{EPOCHS}")

    n = 0
    for real_x, real_y in tf.data.Dataset.zip((photo_train_ds, monet_train_ds)):
        train_step(real_x, real_y)
        if n % 100 == 0:
            print(f"Processed batch {n}")
        n += 1

    # Save the model every 5 epochs
    if (epoch + 1) % 5 == 0:
        ckpt_save_path = ckpt_manager.save()
        print(f'Saving checkpoint at epoch {epoch+1}')

    print(f'Time taken for epoch {epoch+1} is {time.time()-start:.2f} sec\n')

# Save the final models
#generator_g.save('generator_g.keras')
#generator_f.save('generator_f.keras')