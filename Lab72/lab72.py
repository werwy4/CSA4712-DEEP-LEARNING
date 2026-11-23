# Experiment No: 72
# Title: Variational Autoencoder (VAE)
# Aim: Implement a VAE and generate new samples by sampling from the learned latent space.
# Libraries: TensorFlow, Keras, NumPy, Matplotlib

# Install: pip install tensorflow numpy matplotlib

# PROGRAM
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

class Sampling(layers.Layer):
    """Reparameterisation trick: z = z_mean + exp(0.5 * z_log_var) * epsilon."""
    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.random.normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon

def build_vae(latent_dim=2):
    """Builds the Encoder, Decoder, and VAE Model."""
    # 1. Encoder
    encoder_inputs = keras.Input(shape=(784,), name="encoder_input")
    x = layers.Dense(256, activation="relu")(encoder_inputs)
    x = layers.Dense(128, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(encoder_inputs, [z_mean, z_log_var, z], name="encoder")

    # 2. Decoder
    latent_inputs = keras.Input(shape=(latent_dim,), name="z_sampling")
    x = layers.Dense(128, activation="relu")(latent_inputs)
    x = layers.Dense(256, activation="relu")(x)
    decoder_outputs = layers.Dense(784, activation="sigmoid")(x)
    decoder = keras.Model(latent_inputs, decoder_outputs, name="decoder")

    return encoder, decoder

class VAEModel(keras.Model):
    """End-to-end VAE with Evidence Lower Bound (ELBO) Loss."""
    def __init__(self, encoder, decoder, **kwargs):
        super().__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(name="recon_loss")
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [self.total_loss_tracker, self.reconstruction_loss_tracker, self.kl_loss_tracker]

    def train_step(self, data):
        # Support either x or (x, y) input
        if isinstance(data, tuple):
            data = data[0]

        with tf.GradientTape() as tape:
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)

            # Binary Cross-Entropy Reconstruction Loss across 784 pixels
            bce = keras.losses.binary_crossentropy(data, reconstruction)
            recon_loss = tf.reduce_mean(bce) * 784.0

            # Analytic KL Divergence: -0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
            kl_loss = -0.5 * tf.reduce_mean(
                tf.reduce_sum(1.0 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var), axis=1)
            )
            total_loss = recon_loss + kl_loss

        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))

        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(recon_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "loss": self.total_loss_tracker.result(),
            "recon_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

def main():
    print("=" * 65)
    print("  LAB 72: VARIATIONAL AUTOENCODER (VAE) & MANIFOLD SAMPLING")
    print("=" * 65)

    # Step 1: Load and Preprocess Dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    train_size = 8000
    test_size = 1500

    X_train = (x_train[:train_size].reshape(-1, 784) / 255.0).astype(np.float32)
    y_train_sub = y_train[:train_size]

    X_test = (x_test[:test_size].reshape(-1, 784) / 255.0).astype(np.float32)
    y_test_sub = y_test[:test_size]

    print(f"\n[+] Data Subsets Loaded: Train={X_train.shape}, Test={X_test.shape}")

    # Step 2: Build and Train VAE
    latent_dim = 2
    encoder, decoder = build_vae(latent_dim=latent_dim)
    vae = VAEModel(encoder, decoder)
    vae.compile(optimizer=keras.optimizers.Adam(learning_rate=1e-3))

    epochs = 15
    batch_size = 128
    print(f"\n--- Training VAE for {epochs} Epochs ---")
    vae.fit(X_train, epochs=epochs, batch_size=batch_size, verbose=1)

    # Step 3: Visualise 2D Latent Space Embeddings
    print("\n[+] Encoding test digits to 2D latent space...")
    z_mean, _, _ = encoder.predict(X_test, verbose=0)

    # Step 4: Latent Space Interpolation Between Digits
    print("[+] Performing linear latent space interpolation between two test digits...")
    idx1, idx2 = 0, 1  # Select two test examples
    z_start = z_mean[idx1]
    z_end = z_mean[idx2]

    n_steps = 10
    alphas = np.linspace(0, 1, n_steps)
    z_interp = np.array([(1 - a) * z_start + a * z_end for a in alphas])
    interpolated_digits = decoder.predict(z_interp, verbose=0)

    # Step 5: Visualisations
    fig = plt.figure(figsize=(16, 10))

    # Plot 1: 2D Latent Space Scatter Coloured by Class
    ax1 = fig.add_subplot(2, 2, 1)
    scatter = ax1.scatter(z_mean[:, 0], z_mean[:, 1], c=y_test_sub, cmap='tab10', alpha=0.7, s=15)
    plt.colorbar(scatter, ax=ax1, label='Digit Class')
    ax1.set_title('VAE 2D Latent Space (Learned Prior)', fontsize=11, fontweight='bold')
    ax1.set_xlabel('Latent z[0]', fontsize=10)
    ax1.set_ylabel('Latent z[1]', fontsize=10)
    ax1.grid(True, linestyle='--', alpha=0.5)

    # Plot 2: 2D Digit Manifold (Grid Sampling from Prior)
    ax2 = fig.add_subplot(2, 2, 2)
    grid_n = 12
    grid_x = np.linspace(-2.5, 2.5, grid_n)
    grid_y = np.linspace(-2.5, 2.5, grid_n)
    manifold_img = np.zeros((28 * grid_n, 28 * grid_n))

    for i, yi in enumerate(grid_y):
        for j, xi in enumerate(grid_x):
            z_sample = np.array([[xi, yi]])
            x_decoded = decoder.predict(z_sample, verbose=0)
            digit = x_decoded[0].reshape(28, 28)
            manifold_img[(grid_n - i - 1) * 28:(grid_n - i) * 28, j * 28:(j + 1) * 28] = digit

    ax2.imshow(manifold_img, cmap='Greys_r')
    ax2.set_title('Generated Digit Manifold via Latent Grid Sampling', fontsize=11, fontweight='bold')
    ax2.axis('off')

    # Plot 3: Unconditional Random Generations from Standard Normal N(0, I)
    ax3 = fig.add_subplot(2, 2, 3)
    np.random.seed(10)
    random_z = np.random.normal(loc=0.0, scale=1.0, size=(16, latent_dim))
    gen_digits = decoder.predict(random_z, verbose=0)

    gen_grid = np.zeros((28 * 4, 28 * 4))
    for idx in range(16):
        r, c = idx // 4, idx % 4
        gen_grid[r*28:(r+1)*28, c*28:(c+1)*28] = gen_digits[idx].reshape(28, 28)
    ax3.imshow(gen_grid, cmap='gray')
    ax3.set_title('Random Generated Digits ~ N(0, I)', fontsize=11, fontweight='bold')
    ax3.axis('off')

    # Plot 4: Smooth Latent Space Morphing / Interpolation
    ax4 = fig.add_subplot(2, 2, 4)
    morph_grid = np.zeros((28, 28 * n_steps))
    for step in range(n_steps):
        morph_grid[:, step*28:(step+1)*28] = interpolated_digits[step].reshape(28, 28)
    ax4.imshow(morph_grid, cmap='plasma')
    ax4.set_title(f"Smooth Morphing: Digit '{y_test_sub[idx1]}' -> Digit '{y_test_sub[idx2]}'",
                  fontsize=11, fontweight='bold')
    ax4.axis('off')

    plt.tight_layout()
    plt.savefig("lab72_vae_manifold.png", dpi=300)
    print("\n[+] VAE manifold visualizations saved to 'lab72_vae_manifold.png'")
    plt.show()

if __name__ == "__main__":
    main()

# EXPECTED OUTPUT
# =================================================================
#   LAB 72: VARIATIONAL AUTOENCODER (VAE) & MANIFOLD SAMPLING
# =================================================================
#
# [+] Data Subsets Loaded: Train=(8000, 784), Test=(1500, 784)
#
# --- Training VAE for 15 Epochs ---
# ...
# Epoch 15/15 | loss: ~140.0 - 150.0 | recon_loss: ~135.0 | kl_loss: ~10.0 - 15.0
#
# [+] Encoding test digits to 2D latent space...
# [+] Performing linear latent space interpolation between two test digits...
# [+] VAE manifold visualizations saved to 'lab72_vae_manifold.png'

# RESULT
# Successfully implemented the reparameterization trick and trained a Variational
# Autoencoder with Evidence Lower Bound (ELBO) objective, achieving a smooth,
# continuous latent manifold that enables unconstrained random generation and morphing.
