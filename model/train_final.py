import os
import time
import tensorflow as tf
from tqdm import tqdm

# Configure CPU Threads to Utilize All CPU Cores
num_cores = os.cpu_count() or 4  # Fallback to 4 if os.cpu_count() is None
tf.config.threading.set_intra_op_parallelism_threads(num_cores)
tf.config.threading.set_inter_op_parallelism_threads(num_cores)

# Enable XLA (Accelerated Linear Algebra) Compiler
tf.config.optimizer.set_jit(True)

# Enable Mixed Precision
from tensorflow.keras import mixed_precision
mixed_precision.set_global_policy('mixed_float16')

# Now Import Other Modules
from data.data_preprocessing import (
    monet_train_ds, monet_val_ds, monet_test_ds,
    photo_train_ds, photo_val_ds, photo_test_ds
)
from model.generators_final import build_generator
from model.discriminators_final import build_discriminator

# Hyperparameters
EPOCHS = 40
LR = 2e-4
BETA_1 = 0.5
LAMBDA_CYCLE = 10.0       # Cycle consistency loss weight
LAMBDA_IDENTITY = 5.0     # Identity loss weight (set to 0 to disable)
CHECKPOINT_DIR = './checkpoints'
SAVE_FREQ = 5
LOG_DIR = './logs'
SAVED_MODELS_DIR = './saved_models'

# Ensure Saved Models Directory Exists
os.makedirs(SAVED_MODELS_DIR, exist_ok=True)

# Instantiate Models
generator_g = build_generator(image_size=256, num_res_blocks=9)
generator_f = build_generator(image_size=256, num_res_blocks=9)
discriminator_x = build_discriminator(image_size=256)
discriminator_y = build_discriminator(image_size=256)

# Optimizers
generator_g_optimizer = tf.keras.optimizers.Adam(LR, beta_1=BETA_1)
generator_f_optimizer = tf.keras.optimizers.Adam(LR, beta_1=BETA_1)
discriminator_x_optimizer = tf.keras.optimizers.Adam(LR, beta_1=BETA_1)
discriminator_y_optimizer = tf.keras.optimizers.Adam(LR, beta_1=BETA_1)

# Loss Functions
# Using Least-Squares GAN Loss (LSGAN)
loss_obj = tf.keras.losses.MeanSquaredError()

def discriminator_loss(real, generated):
    """LSGAN discriminator loss."""
    real_loss = loss_obj(tf.ones_like(real), real)   # real images close to 1
    generated_loss = loss_obj(tf.zeros_like(generated), generated) # fake close to 0
    total_disc_loss = (real_loss + generated_loss) * 0.5
    return total_disc_loss

def generator_loss(generated):
    """LSGAN generator loss."""
    # We want generated images to be classified as real (close to 1).
    return loss_obj(tf.ones_like(generated), generated)

def cycle_consistency_loss(real_image, cycled_image):
    """L1 cycle consistency loss."""
    return tf.reduce_mean(tf.abs(real_image - cycled_image))

def identity_loss(real_image, same_image):
    """L1 identity mapping loss (optional)."""
    return tf.reduce_mean(tf.abs(real_image - same_image))

# Checkpoints
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
ckpt_manager = tf.train.CheckpointManager(ckpt, CHECKPOINT_DIR, max_to_keep=5)

# Restore latest checkpoint if available
if ckpt_manager.latest_checkpoint:
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print('Latest checkpoint restored!')
else:
    print('Initializing from scratch.')

# TensorBoard Setup
summary_writer = tf.summary.create_file_writer(LOG_DIR)

# Training Step Function
@tf.function
def train_step(real_x, real_y):
    with tf.GradientTape(persistent=True) as tape:
        # Generator G: X -> Y
        fake_y = generator_g(real_x, training=True)
        # Generator F: Y -> X
        fake_x = generator_f(real_y, training=True)

        # Cycle consistency
        cycled_x = generator_f(fake_y, training=True)
        cycled_y = generator_g(fake_x, training=True)

        # Identity mapping (optional)
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
        cycle_loss_x = cycle_consistency_loss(real_x, cycled_x)
        cycle_loss_y = cycle_consistency_loss(real_y, cycled_y)
        total_cycle_loss = cycle_loss_x + cycle_loss_y

        # Identity losses
        if LAMBDA_IDENTITY > 0:
            id_loss_x = identity_loss(real_x, same_x)
            id_loss_y = identity_loss(real_y, same_y)
        else:
            id_loss_x = 0
            id_loss_y = 0

        # Total generator losses
        total_gen_g_loss = gen_g_loss + (LAMBDA_CYCLE * cycle_loss_y) + (LAMBDA_IDENTITY * id_loss_y)
        total_gen_f_loss = gen_f_loss + (LAMBDA_CYCLE * cycle_loss_x) + (LAMBDA_IDENTITY * id_loss_x)

        # Discriminator losses
        disc_x_loss = discriminator_loss(disc_real_x, disc_fake_x)
        disc_y_loss = discriminator_loss(disc_real_y, disc_fake_y)

    # Compute gradients
    generator_g_gradients = tape.gradient(total_gen_g_loss, generator_g.trainable_variables)
    generator_f_gradients = tape.gradient(total_gen_f_loss, generator_f.trainable_variables)

    discriminator_x_gradients = tape.gradient(disc_x_loss, discriminator_x.trainable_variables)
    discriminator_y_gradients = tape.gradient(disc_y_loss, discriminator_y.trainable_variables)

    # Apply gradients
    generator_g_optimizer.apply_gradients(zip(generator_g_gradients, generator_g.trainable_variables))
    generator_f_optimizer.apply_gradients(zip(generator_f_gradients, generator_f.trainable_variables))

    discriminator_x_optimizer.apply_gradients(zip(discriminator_x_gradients, discriminator_x.trainable_variables))
    discriminator_y_optimizer.apply_gradients(zip(discriminator_y_gradients, discriminator_y.trainable_variables))

    return {
        'gen_g_loss': total_gen_g_loss,
        'gen_f_loss': total_gen_f_loss,
        'disc_x_loss': disc_x_loss,
        'disc_y_loss': disc_y_loss,
        'cycle_loss': total_cycle_loss,
        'id_loss_x': id_loss_x,
        'id_loss_y': id_loss_y
    }

# Training Loop
def train():
    for epoch in range(EPOCHS):
        start_time = time.time()
        print(f"Epoch {epoch+1}/{EPOCHS}")

        # Combine datasets; zip stops at the shorter dataset. To ensure all data is used, use repeat.
        steps_per_epoch = max(len(monet_train_ds), len(photo_train_ds))
        monet_train_ds_repeated = monet_train_ds.repeat()
        photo_train_ds_repeated = photo_train_ds.repeat()
        combined = tf.data.Dataset.zip((photo_train_ds_repeated, monet_train_ds_repeated)).take(steps_per_epoch)

        # Initialize loss accumulators
        epoch_losses = {
            'gen_g_loss': 0.0,
            'gen_f_loss': 0.0,
            'disc_x_loss': 0.0,
            'disc_y_loss': 0.0,
            'cycle_loss': 0.0,
            'id_loss_x': 0.0,
            'id_loss_y': 0.0
        }
        step = 0

        # Initialize tqdm progress bar
        pbar = tqdm(enumerate(combined), total=steps_per_epoch, desc=f"Epoch {epoch+1}")

        for step, (real_x, real_y) in pbar:
            losses = train_step(real_x, real_y)
            for key in epoch_losses:
                epoch_losses[key] += losses[key]
            step += 1

            # Update progress bar with current losses
            if step % 50 == 0 or step == steps_per_epoch:
                pbar.set_postfix({
                    key: f"{value.numpy():.4f}"
                    for key, value in losses.items()
                })

        # Average losses
        for key in epoch_losses:
            epoch_losses[key] /= step

        # Log losses to TensorBoard
        with summary_writer.as_default():
            for key, value in epoch_losses.items():
                tf.summary.scalar(key, value, step=epoch)

        # Print epoch summary
        print(f"Epoch {epoch+1} Summary:")
        print(", ".join([f"{k}: {v:.4f}" for k, v in epoch_losses.items()]))
        print(f"Time taken for epoch {epoch+1}: {time.time() - start_time:.2f} sec\n")

        # Save checkpoints every SAVE_FREQ epochs
        if (epoch + 1) % SAVE_FREQ == 0:
            ckpt_save_path = ckpt_manager.save()
            print(f"Checkpoint saved at epoch {epoch+1}.\n")

    print("Training completed.")

    # Save the generator models for inference
    generator_g.save(os.path.join(SAVED_MODELS_DIR, 'generator_g'))
    generator_f.save(os.path.join(SAVED_MODELS_DIR, 'generator_f'))
    print(f"Generator models saved to {SAVED_MODELS_DIR}.")

if __name__ == '__main__':
    train()