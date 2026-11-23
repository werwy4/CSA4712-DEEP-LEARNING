# Experiment No: 69
# Title: Autoencoder for Feature Learning and Reconstruction
# Aim: Build and train an autoencoder for feature learning and image reconstruction on MNIST.
# Libraries: TensorFlow, Keras, NumPy, Matplotlib, Scikit-learn

# Install: pip install tensorflow numpy matplotlib scikit-learn

# PROGRAM
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score

def build_deep_autoencoder(bottleneck_dim=32):
    """Constructs encoder and decoder for a specified bottleneck dimension."""
    # Encoder
    encoder_input = keras.Input(shape=(784,), name="encoder_input")
    x = layers.Dense(128, activation="relu", name="enc_dense1")(encoder_input)
    x = layers.Dense(64, activation="relu", name="enc_dense2")(x)
    latent = layers.Dense(bottleneck_dim, activation="relu", name="latent_space")(x)
    encoder = keras.Model(encoder_input, latent, name=f"encoder_{bottleneck_dim}")

    # Decoder
    decoder_input = keras.Input(shape=(bottleneck_dim,), name="decoder_input")
    x = layers.Dense(64, activation="relu", name="dec_dense1")(decoder_input)
    x = layers.Dense(128, activation="relu", name="dec_dense2")(x)
    decoded = layers.Dense(784, activation="sigmoid", name="output_reconstruction")(x)
    decoder = keras.Model(decoder_input, decoded, name=f"decoder_{bottleneck_dim}")

    # Autoencoder
    ae_output = decoder(encoder(encoder_input))
    autoencoder = keras.Model(encoder_input, ae_output, name=f"autoencoder_{bottleneck_dim}")
    autoencoder.compile(optimizer="adam", loss="mean_squared_error")

    return autoencoder, encoder, decoder

def main():
    print("=" * 65)
    print("  LAB 69: AUTOENCODER FOR FEATURE LEARNING & RECONSTRUCTION")
    print("=" * 65)

    # Step 1: Load and Preprocess MNIST Dataset
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

    # Normalize to [0, 1] and flatten to vectors of length 784
    train_size = 6000
    test_size = 1000

    X_train = (x_train[:train_size].reshape(-1, 784) / 255.0).astype(np.float32)
    y_train_sub = y_train[:train_size]

    X_test = (x_test[:test_size].reshape(-1, 784) / 255.0).astype(np.float32)
    y_test_sub = y_test[:test_size]

    print(f"\n[+] Data Preprocessed: Training={X_train.shape}, Test={X_test.shape}")

    # Step 2: Build and Train Primary Autoencoder (Bottleneck = 32)
    print("\n--- Training Deep Autoencoder (Bottleneck = 32) ---")
    ae_32, encoder_32, decoder_32 = build_deep_autoencoder(bottleneck_dim=32)
    ae_32.summary()

    history_32 = ae_32.fit(
        X_train, X_train,
        epochs=12,
        batch_size=128,
        validation_data=(X_test, X_test),
        verbose=1
    )

    test_mse_32 = ae_32.evaluate(X_test, X_test, verbose=0)
    print(f"\n[+] Primary Autoencoder (dim=32) Test MSE: {test_mse_32:.5f}")

    # Step 3: Unsupervised Clustering on Latent Representations
    print("\n--- Unsupervised Clustering with KMeans on Latent Space (dim=32) ---")
    latent_features = encoder_32.predict(X_test, verbose=0)
    kmeans = KMeans(n_clusters=10, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(latent_features)

    ari_score = adjusted_rand_score(y_test_sub, cluster_labels)
    sil_score = silhouette_score(latent_features, cluster_labels)
    print(f"    Adjusted Rand Index (ARI vs. True Digits) : {ari_score:.4f}")
    print(f"    Latent Space Silhouette Score             : {sil_score:.4f}")

    # Step 4: Compare Reconstruction Quality across Different Bottleneck Sizes
    bottlenecks = [2, 8, 32, 64]
    recon_mses = []

    print("\n--- Comparing Bottleneck Sizes [2, 8, 32, 64] ---")
    ae_2d = None
    encoder_2d = None

    for dim in bottlenecks:
        if dim == 32:
            recon_mses.append(test_mse_32)
            continue

        ae_dim, enc_dim, _ = build_deep_autoencoder(bottleneck_dim=dim)
        ae_dim.fit(X_train, X_train, epochs=8, batch_size=128, verbose=0)
        mse_dim = ae_dim.evaluate(X_test, X_test, verbose=0)
        recon_mses.append(mse_dim)
        print(f"    Bottleneck Dim = {dim:2d} | Test MSE: {mse_dim:.5f}")

        if dim == 2:
            ae_2d = ae_dim
            encoder_2d = enc_dim

    # Step 5: Visualizations
    fig = plt.figure(figsize=(16, 10))

    # Subplot 1: Original vs Reconstructed Digits (32-dim AE)
    reconstructed_32 = ae_32.predict(X_test[:8], verbose=0)
    for i in range(8):
        # Original
        ax_orig = plt.subplot(4, 8, i + 1)
        ax_orig.imshow(X_test[i].reshape(28, 28), cmap="gray")
        ax_orig.set_title(f"Orig: {y_test_sub[i]}", fontsize=9)
        ax_orig.axis("off")

        # Reconstructed
        ax_rec = plt.subplot(4, 8, i + 9)
        ax_rec.imshow(reconstructed_32[i].reshape(28, 28), cmap="gray")
        ax_rec.set_title("Recon (32D)", fontsize=9)
        ax_rec.axis("off")

    # Subplot 2: 2D Latent Space Projection
    latent_2d = encoder_2d.predict(X_test, verbose=0)
    ax_latent = plt.subplot(2, 2, 3)
    scatter = ax_latent.scatter(
        latent_2d[:, 0], latent_2d[:, 1],
        c=y_test_sub, cmap="tab10", alpha=0.7, s=15
    )
    plt.colorbar(scatter, ax=ax_latent, label="Digit Class")
    ax_latent.set_title("2D Latent Space Manifold", fontsize=11, fontweight="bold")
    ax_latent.set_xlabel("Latent Z_1", fontsize=10)
    ax_latent.set_ylabel("Latent Z_2", fontsize=10)
    ax_latent.grid(True, linestyle="--", alpha=0.5)

    # Subplot 3: Bottleneck Size vs Reconstruction Error
    ax_bar = plt.subplot(2, 2, 4)
    bars = ax_bar.bar([str(d) for d in bottlenecks], recon_mses, color=["#e74c3c", "#f39c12", "#27ae60", "#2980b9"])
    ax_bar.set_title("Reconstruction MSE vs. Bottleneck Dimension", fontsize=11, fontweight="bold")
    ax_bar.set_xlabel("Latent Bottleneck Dimension", fontsize=10)
    ax_bar.set_ylabel("Test Mean Squared Error (MSE)", fontsize=10)
    for bar in bars:
        yval = bar.get_height()
        ax_bar.text(bar.get_x() + bar.get_width()/2.0, yval + 0.001, f"{yval:.4f}", ha="center", va="bottom", fontsize=9)
    ax_bar.grid(axis="y", linestyle="--", alpha=0.6)

    plt.tight_layout()
    plt.savefig("lab69_autoencoder_reconstruction.png", dpi=300)
    print("\n[+] Visualisations saved to 'lab69_autoencoder_reconstruction.png'")
    plt.show()

if __name__ == "__main__":
    main()

# EXPECTED OUTPUT
# =================================================================
#   LAB 69: AUTOENCODER FOR FEATURE LEARNING & RECONSTRUCTION
# =================================================================
#
# [+] Data Preprocessed: Training=(6000, 784), Test=(1000, 784)
#
# --- Training Deep Autoencoder (Bottleneck = 32) ---
# ...
# [+] Primary Autoencoder (dim=32) Test MSE: ~0.012 - 0.018
#
# --- Unsupervised Clustering with KMeans on Latent Space (dim=32) ---
#     Adjusted Rand Index (ARI vs. True Digits) : ~0.40 - 0.55
#     Latent Space Silhouette Score             : ~0.15 - 0.25
#
# --- Comparing Bottleneck Sizes [2, 8, 32, 64] ---
#     Bottleneck Dim =  2 | Test MSE: ~0.045
#     Bottleneck Dim =  8 | Test MSE: ~0.024
#     Bottleneck Dim = 32 | Test MSE: ~0.015
#     Bottleneck Dim = 64 | Test MSE: ~0.010
#
# [+] Visualisations saved to 'lab69_autoencoder_reconstruction.png'

# RESULT
# Successfully trained a deep symmetric autoencoder for non-linear dimensionality reduction,
# demonstrating that compact latent representations capture intrinsic image structures,
# facilitate unsupervised clustering, and trade off compression ratio against reconstruction fidelity.
