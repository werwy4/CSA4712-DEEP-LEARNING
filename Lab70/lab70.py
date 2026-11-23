# Experiment No: 70
# Title: Denoising Autoencoder
# Aim: Train a denoising autoencoder to reconstruct clean images from noisy inputs.
# Libraries: TensorFlow, Keras, NumPy, Matplotlib

# Install: pip install tensorflow numpy matplotlib

# PROGRAM
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def add_gaussian_noise(images, noise_factor=0.35):
    """Corrupts images with additive Gaussian white noise."""
    noisy = images + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=images.shape)
    return np.clip(noisy, 0.0, 1.0)

def add_salt_and_pepper_noise(images, salt_prob=0.10, pepper_prob=0.10):
    """Corrupts images with impulse (Salt & Pepper) noise."""
    noisy = images.copy()
    # Salt noise (white pixels)
    num_salt = np.ceil(salt_prob * images[0].size)
    # Pepper noise (black pixels)
    num_pepper = np.ceil(pepper_prob * images[0].size)

    for i in range(len(noisy)):
        # Salt
        coords = [np.random.randint(0, i - 1, int(num_salt)) for i in images.shape[1:3]]
        noisy[i, coords[0], coords[1], :] = 1.0
        # Pepper
        coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in images.shape[1:3]]
        noisy[i, coords[0], coords[1], :] = 0.0

    return np.clip(noisy, 0.0, 1.0)

def build_conv_autoencoder():
    """Builds a Convolutional Autoencoder for image denoising."""
    inputs = keras.Input(shape=(28, 28, 1))

    # Encoder
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.MaxPooling2D((2, 2), padding='same')(x)
    x = layers.Conv2D(16, (3, 3), activation='relu', padding='same')(x)
    encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

    # Decoder
    x = layers.Conv2DTranspose(16, (3, 3), strides=2, activation='relu', padding='same')(encoded)
    x = layers.Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(x)
    decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = keras.Model(inputs, decoded)
    autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
    return autoencoder

def main():
    print("=" * 65)
    print("  LAB 70: CONVOLUTIONAL DENOISING AUTOENCODER (DAE)")
    print("=" * 65)

    # Step 1: Load and Preprocess Dataset
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

    train_size = 5000
    test_size = 1000

    X_train_clean = (x_train[:train_size].astype(np.float32) / 255.0)[..., np.newaxis]
    X_test_clean = (x_test[:test_size].astype(np.float32) / 255.0)[..., np.newaxis]

    print(f"\n[+] Clean Data Loaded: Train={X_train_clean.shape}, Test={X_test_clean.shape}")

    # Step 2: Corrupt Images with Gaussian and Salt & Pepper Noise
    np.random.seed(42)
    X_train_noisy = add_gaussian_noise(X_train_clean, noise_factor=0.35)
    X_test_noisy_gauss = add_gaussian_noise(X_test_clean, noise_factor=0.35)
    X_test_noisy_sp = add_salt_and_pepper_noise(X_test_clean, salt_prob=0.10, pepper_prob=0.10)

    print("[+] Generated Corrupted Noise Sets (Gaussian sigma=0.35 & Salt-and-Pepper).")

    # Step 3: Train Denoising Autoencoder (Noisy -> Clean)
    print("\n--- Training Denoising Autoencoder (DAE) ---")
    dae = build_conv_autoencoder()
    dae_hist = dae.fit(
        X_train_noisy, X_train_clean,
        epochs=10,
        batch_size=128,
        validation_data=(X_test_noisy_gauss, X_test_clean),
        verbose=1
    )

    # Step 4: Train Standard Autoencoder for Comparison (Clean -> Clean)
    print("\n--- Training Standard Autoencoder (Clean -> Clean Baseline) ---")
    std_ae = build_conv_autoencoder()
    std_ae.fit(
        X_train_clean, X_train_clean,
        epochs=10,
        batch_size=128,
        validation_data=(X_test_clean, X_test_clean),
        verbose=0
    )

    # Step 5: Comparative Evaluation on Noisy Test Images
    dae_gauss_preds = dae.predict(X_test_noisy_gauss, verbose=0)
    std_gauss_preds = std_ae.predict(X_test_noisy_gauss, verbose=0)

    dae_sp_preds = dae.predict(X_test_noisy_sp, verbose=0)

    dae_mse = np.mean((X_test_clean - dae_gauss_preds) ** 2)
    std_mse = np.mean((X_test_clean - std_gauss_preds) ** 2)

    print("\n" + "=" * 65)
    print("  DENOISING FIDELITY COMPARISON (MSE vs. Clean Ground Truth)")
    print("=" * 65)
    print(f"  Standard Autoencoder on Noisy Inputs MSE : {std_mse:.5f}")
    print(f"  Denoising Autoencoder (DAE) MSE          : {dae_mse:.5f}")
    print(f"  Error Reduction by Denoising Autoencoder : {(1.0 - dae_mse / std_mse) * 100:.2f}%")

    # Step 6: Visualizations
    n_samples = 6
    fig, axes = plt.subplots(5, n_samples, figsize=(14, 11))

    for i in range(n_samples):
        # Row 1: Original Clean
        axes[0, i].imshow(X_test_clean[i].squeeze(), cmap='gray')
        axes[0, i].set_title(f"Clean #{i+1}", fontsize=9)
        axes[0, i].axis('off')

        # Row 2: Noisy Input (Gaussian)
        axes[1, i].imshow(X_test_noisy_gauss[i].squeeze(), cmap='gray')
        axes[1, i].set_title("Gaussian Noise", fontsize=9)
        axes[1, i].axis('off')

        # Row 3: Standard Autoencoder Output
        axes[2, i].imshow(std_gauss_preds[i].squeeze(), cmap='gray')
        axes[2, i].set_title("Standard AE", fontsize=9)
        axes[2, i].axis('off')

        # Row 4: DAE Output (Gaussian)
        axes[3, i].imshow(dae_gauss_preds[i].squeeze(), cmap='gray')
        axes[3, i].set_title("DAE Restored", fontsize=9, color='green', fontweight='bold')
        axes[3, i].axis('off')

        # Row 5: DAE Output on Salt & Pepper Noise
        axes[4, i].imshow(dae_sp_preds[i].squeeze(), cmap='gray')
        axes[4, i].set_title("DAE (S&P Noise)", fontsize=9, color='blue')
        axes[4, i].axis('off')

    axes[0, 0].set_ylabel("Clean Original", fontsize=10, fontweight='bold')
    axes[1, 0].set_ylabel("Noisy Input", fontsize=10, fontweight='bold')
    axes[2, 0].set_ylabel("Standard AE", fontsize=10, fontweight='bold')
    axes[3, 0].set_ylabel("DAE Gaussian", fontsize=10, fontweight='bold')
    axes[4, 0].set_ylabel("DAE Salt&Pepper", fontsize=10, fontweight='bold')

    plt.suptitle("Denoising Autoencoder Reconstruction vs. Standard Autoencoder", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("lab70_denoising_autoencoder.png", dpi=300)
    print("\n[+] Denoising visualisations saved to 'lab70_denoising_autoencoder.png'")
    plt.show()

if __name__ == "__main__":
    main()

# EXPECTED OUTPUT
# =================================================================
#   LAB 70: CONVOLUTIONAL DENOISING AUTOENCODER (DAE)
# =================================================================
#
# [+] Clean Data Loaded: Train=(5000, 28, 28, 1), Test=(1000, 28, 28, 1)
# [+] Generated Corrupted Noise Sets (Gaussian sigma=0.35 & Salt-and-Pepper).
#
# --- Training Denoising Autoencoder (DAE) ---
# ...
# --- Training Standard Autoencoder (Clean -> Clean Baseline) ---
# ...
#
# =================================================================
#   DENOISING FIDELITY COMPARISON (MSE vs. Clean Ground Truth)
# =================================================================
#   Standard Autoencoder on Noisy Inputs MSE : ~0.035 - 0.045
#   Denoising Autoencoder (DAE) MSE          : ~0.012 - 0.016
#   Error Reduction by Denoising Autoencoder : ~60% - 70%
#
# [+] Denoising visualisations saved to 'lab70_denoising_autoencoder.png'

# RESULT
# Formulated and evaluated a Convolutional Denoising Autoencoder (DAE), proving that forcing
# the network to reconstruct manifold-projected clean targets from corrupted inputs forces the
# latent layers to learn robust invariant topological features rather than trivial identity mappings.
