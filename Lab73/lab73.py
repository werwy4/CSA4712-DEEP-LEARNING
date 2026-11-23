# Experiment No: 73
# Title: Generative Adversarial Network (GAN) - Basic
# Aim: Implement a basic GAN with generator and discriminator to generate digit-like images.
# Libraries: TensorFlow, Keras, NumPy, Matplotlib

# Install: pip install tensorflow numpy matplotlib

# PROGRAM
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def build_generator(latent_dim=100):
    """Constructs the Generator network mapping noise vector z -> 784 image."""
    model = keras.Sequential([
        keras.Input(shape=(latent_dim,)),
        layers.Dense(256),
        layers.LeakyReLU(negative_slope=0.2),
        layers.BatchNormalization(momentum=0.8),

        layers.Dense(512),
        layers.LeakyReLU(negative_slope=0.2),
        layers.BatchNormalization(momentum=0.8),

        layers.Dense(784, activation='tanh')  # Outputs normalized to [-1, 1]
    ], name="Generator")
    return model

def build_discriminator():
    """Constructs the Discriminator network mapping 784 image -> P(Real)."""
    model = keras.Sequential([
        keras.Input(shape=(784,)),
        layers.Dense(512),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Dropout(0.3),

        layers.Dense(256),
        layers.LeakyReLU(negative_slope=0.2),
        layers.Dropout(0.3),

        layers.Dense(1, activation='sigmoid')  # Real (1) vs. Fake (0)
    ], name="Discriminator")
    return model

def main():
    print("=" * 65)
    print("  LAB 73: BASIC GENERATIVE ADVERSARIAL NETWORK (GAN)")
    print("=" * 65)

    # Step 1: Load and Preprocess MNIST Dataset
    (x_train, _), _ = tf.keras.datasets.mnist.load_data()

    # Subsample for fast training and rescale to [-1, 1] for tanh activation
    train_size = 5000
    X_train = (x_train[:train_size].reshape(-1, 784).astype(np.float32) - 127.5) / 127.5
    print(f"\n[+] Loaded {len(X_train)} training images scaled to range [-1, 1].")

    # Step 2: Instantiate Models and Optimizers
    latent_dim = 100
    generator = build_generator(latent_dim=latent_dim)
    discriminator = build_discriminator()

    d_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    g_optimizer = keras.optimizers.Adam(learning_rate=0.0002, beta_1=0.5)
    bce_loss = keras.losses.BinaryCrossentropy()

    discriminator.summary()
    generator.summary()

    # Step 3: Adversarial Training Loop
    epochs = 30
    batch_size = 64
    n_batches = len(X_train) // batch_size

    # Fixed noise vector to inspect progressive image evolution across epochs
    seed_noise = np.random.normal(0, 1, size=(16, latent_dim))
    milestone_images = {}
    milestones = [1, 10, 20, 30]

    d_losses = []
    g_losses = []

    print("\n--- Commencing Adversarial Minimax Training ---")
    for epoch in range(1, epochs + 1):
        epoch_d_loss = []
        epoch_g_loss = []

        perm = np.random.permutation(len(X_train))

        for b in range(n_batches):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            real_imgs = X_train[perm[b * batch_size:(b + 1) * batch_size]]
            current_batch_size = len(real_imgs)
            noise = np.random.normal(0, 1, size=(current_batch_size, latent_dim))

            with tf.GradientTape() as d_tape:
                fake_imgs = generator(noise, training=True)

                real_output = discriminator(real_imgs, training=True)
                fake_output = discriminator(fake_imgs, training=True)

                # Discriminator loss with one-sided soft labels for stability
                d_loss_real = bce_loss(tf.ones_like(real_output) * 0.9, real_output)
                d_loss_fake = bce_loss(tf.zeros_like(fake_output), fake_output)
                d_loss = d_loss_real + d_loss_fake

            d_grads = d_tape.gradient(d_loss, discriminator.trainable_variables)
            d_optimizer.apply_gradients(zip(d_grads, discriminator.trainable_variables))

            # -----------------
            #  Train Generator
            # -----------------
            noise = np.random.normal(0, 1, size=(current_batch_size, latent_dim))

            with tf.GradientTape() as g_tape:
                gen_imgs = generator(noise, training=True)
                disc_output = discriminator(gen_imgs, training=True)
                # Generator wants discriminator to classify generated samples as real (label 1)
                g_loss = bce_loss(tf.ones_like(disc_output), disc_output)

            g_grads = g_tape.gradient(g_loss, generator.trainable_variables)
            g_optimizer.apply_gradients(zip(g_grads, generator.trainable_variables))

            epoch_d_loss.append(float(d_loss))
            epoch_g_loss.append(float(g_loss))

        avg_d = float(np.mean(epoch_d_loss))
        avg_g = float(np.mean(epoch_g_loss))
        d_losses.append(avg_d)
        g_losses.append(avg_g)

        if epoch in milestones:
            samples = generator(seed_noise, training=False).numpy()
            milestone_images[epoch] = samples
            print(f"    Epoch {epoch:2d}/{epochs:2d} | Discriminator Loss: {avg_d:.4f} | Generator Loss: {avg_g:.4f} [*Saved]")
        elif epoch % 5 == 0:
            print(f"    Epoch {epoch:2d}/{epochs:2d} | Discriminator Loss: {avg_d:.4f} | Generator Loss: {avg_g:.4f}")

    # Step 4: Visualizations
    fig = plt.figure(figsize=(16, 10))

    # Subplot 1: Adversarial Loss Curves
    ax_loss = fig.add_subplot(2, 2, 1)
    ax_loss.plot(range(1, epochs + 1), d_losses, 'b-', label='Discriminator Loss', linewidth=2)
    ax_loss.plot(range(1, epochs + 1), g_losses, 'r-', label='Generator Loss', linewidth=2)
    ax_loss.set_title('Adversarial Training Loss History', fontsize=12, fontweight='bold')
    ax_loss.set_xlabel('Epoch', fontsize=11)
    ax_loss.set_ylabel('Loss', fontsize=11)
    ax_loss.legend(fontsize=10)
    ax_loss.grid(True, linestyle='--', alpha=0.5)

    # Subplot 2: Milestone Evolutions (Epoch 1 vs Epoch 30)
    ax_evolve = fig.add_subplot(2, 2, 2)
    evolve_grid = np.zeros((28 * 2, 28 * 8))
    for i in range(8):
        # Epoch 1
        img_e1 = (milestone_images[1][i].reshape(28, 28) * 0.5 + 0.5)
        evolve_grid[0:28, i*28:(i+1)*28] = img_e1
        # Epoch 30
        img_e30 = (milestone_images[30][i].reshape(28, 28) * 0.5 + 0.5)
        evolve_grid[28:56, i*28:(i+1)*28] = img_e30
    ax_evolve.imshow(evolve_grid, cmap='gray')
    ax_evolve.set_title('Evolution: Top (Epoch 1) vs. Bottom (Epoch 30)', fontsize=12, fontweight='bold')
    ax_evolve.axis('off')

    # Subplot 3 & 4: Final 4x4 Grid of Generated Digits
    ax_grid = fig.add_subplot(2, 1, 2)
    final_samples = milestone_images[30]
    full_grid = np.zeros((28 * 2, 28 * 8))
    for idx in range(16):
        r, c = idx // 8, idx % 8
        digit = final_samples[idx].reshape(28, 28) * 0.5 + 0.5
        full_grid[r*28:(r+1)*28, c*28:(c+1)*28] = digit
    ax_grid.imshow(full_grid, cmap='gray')
    ax_grid.set_title('Final Generated Digits at Epoch 30 (16 Random Latent Vectors)', fontsize=12, fontweight='bold')
    ax_grid.axis('off')

    plt.tight_layout()
    plt.savefig("lab73_basic_gan.png", dpi=300)
    print("\n[+] GAN training visualisations saved to 'lab73_basic_gan.png'")
    plt.show()

if __name__ == "__main__":
    main()

# EXPECTED OUTPUT
# =================================================================
#   LAB 73: BASIC GENERATIVE ADVERSARIAL NETWORK (GAN)
# =================================================================
#
# [+] Loaded 5000 training images scaled to range [-1, 1].
#
# --- Commencing Adversarial Minimax Training ---
#     Epoch  1/30 | Discriminator Loss: ~1.2500 | Generator Loss: ~0.8500 [*Saved]
#     Epoch 10/30 | Discriminator Loss: ~1.0500 | Generator Loss: ~1.1500 [*Saved]
#     Epoch 20/30 | Discriminator Loss: ~0.9500 | Generator Loss: ~1.3000 [*Saved]
#     Epoch 30/30 | Discriminator Loss: ~0.9000 | Generator Loss: ~1.4000 [*Saved]
#
# [+] GAN training visualisations saved to 'lab73_basic_gan.png'

# RESULT
# Formulated and executed the zero-sum minimax game between Generator and Discriminator,
# demonstrating the progression from unstructured Gaussian noise to realistic digit-like
# stroke patterns through alternating gradient updates.
