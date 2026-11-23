# Experiment No: 71
# Title: Sparse Autoencoder
# Aim: Train a sparse autoencoder with L1 activity regularisation to learn sparse representations.
# Libraries: TensorFlow, Keras, NumPy, Matplotlib

# Install: pip install tensorflow numpy matplotlib

# PROGRAM
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, regularizers

def build_autoencoder(n_hidden=64, l1_penalty=0.0):
    """
    Constructs an autoencoder with optional L1 activity regularization
    on the hidden bottleneck representations.
    """
    inputs = keras.Input(shape=(784,), name="input_pixels")

    # Regularizer on hidden layer activations
    reg = regularizers.l1(l1_penalty) if l1_penalty > 0 else None

    # Hidden layer uses sigmoid so activations are strictly in [0, 1]
    hidden = layers.Dense(
        n_hidden,
        activation="sigmoid",
        activity_regularizer=reg,
        name="hidden_bottleneck"
    )(inputs)

    # Reconstruction layer
    reconstructed = layers.Dense(784, activation="sigmoid", name="reconstruction")(hidden)

    # Full autoencoder model
    autoencoder = keras.Model(inputs, reconstructed, name=f"AE_L1_{l1_penalty}")
    # Separate encoder to inspect latent activations and receptive fields
    encoder = keras.Model(inputs, hidden, name=f"Encoder_L1_{l1_penalty}")

    autoencoder.compile(optimizer="adam", loss="mean_squared_error")
    return autoencoder, encoder

def main():
    print("=" * 65)
    print("  LAB 71: SPARSE AUTOENCODER WITH L1 ACTIVITY REGULARIZATION")
    print("=" * 65)

    # Step 1: Load and Preprocess MNIST
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

    train_size = 5000
    test_size = 1000

    X_train = (x_train[:train_size].reshape(-1, 784) / 255.0).astype(np.float32)
    X_test = (x_test[:test_size].reshape(-1, 784) / 255.0).astype(np.float32)

    print(f"\n[+] Loaded Dataset: Training={X_train.shape}, Test={X_test.shape}")

    # Step 2: Train Models with Varying Sparsity Penalties
    penalties = [0.0, 1e-4, 1e-3]
    models = {}
    encoders = {}
    histories = {}

    epochs = 12
    batch_size = 128
    n_hidden = 64

    for penalty in penalties:
        tag = f"L1={penalty}"
        print(f"\n--- Training Autoencoder with Sparsity Penalty ({tag}) ---")
        ae, enc = build_autoencoder(n_hidden=n_hidden, l1_penalty=penalty)
        h = ae.fit(
            X_train, X_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, X_test),
            verbose=1
        )
        models[penalty] = ae
        encoders[penalty] = enc
        histories[penalty] = h

    # Step 3: Analyze Sparsity and Activation Profiles
    print("\n" + "=" * 65)
    print(f"{'Sparsity Penalty':<18} | {'Test MSE':<10} | {'Mean Activation':<16} | {'Sparsity (<0.05)':<15}")
    print("=" * 65)

    activations_dict = {}
    mean_acts = []
    sparsity_ratios = []
    test_mses = []

    for penalty in penalties:
        ae = models[penalty]
        enc = encoders[penalty]

        mse = ae.evaluate(X_test, X_test, verbose=0)
        acts = enc.predict(X_test, verbose=0)  # Shape: (1000, 64)
        activations_dict[penalty] = acts

        mean_act = float(np.mean(acts))
        # Sparsity ratio: fraction of units having activation < 0.05
        sparsity_frac = float(np.mean(acts < 0.05) * 100.0)

        mean_acts.append(mean_act)
        sparsity_ratios.append(sparsity_frac)
        test_mses.append(mse)

        print(f"L1 = {penalty:<13} | {mse:<10.5f} | {mean_act:<16.4f} | {sparsity_frac:<14.2f}%")

    # Step 4: Visualizations
    fig = plt.figure(figsize=(16, 10))

    # 1. Activation Distribution Histograms
    ax_hist = fig.add_subplot(2, 2, 1)
    ax_hist.hist(activations_dict[0.0].flatten(), bins=40, alpha=0.5, color='gray', label='Standard AE (No L1)', density=True)
    ax_hist.hist(activations_dict[1e-4].flatten(), bins=40, alpha=0.6, color='blue', label='Sparse AE (L1=1e-4)', density=True)
    ax_hist.hist(activations_dict[1e-3].flatten(), bins=40, alpha=0.6, color='red', label='Sparse AE (L1=1e-3)', density=True)
    ax_hist.set_title("Hidden Layer Activation Distribution", fontsize=11, fontweight='bold')
    ax_hist.set_xlabel("Neuron Activation a_j in [0, 1]", fontsize=10)
    ax_hist.set_ylabel("Probability Density", fontsize=10)
    ax_hist.legend(fontsize=9)
    ax_hist.grid(True, linestyle='--', alpha=0.5)

    # 2. Average Activation and Sparsity Percentage Comparison
    ax_bar = fig.add_subplot(2, 2, 2)
    x_indices = np.arange(len(penalties))
    width = 0.35
    ax_bar.bar(x_indices - width/2, mean_acts, width, label='Mean Activation', color='#2ecc71')
    ax_bar.bar(x_indices + width/2, [s / 100.0 for s in sparsity_ratios], width, label='Sparsity Ratio (<0.05)', color='#9b59b6')
    ax_bar.set_xticks(x_indices)
    ax_bar.set_xticklabels([f"L1={p}" for p in penalties], fontsize=10)
    ax_bar.set_title("Activation Density vs. Sparsity Penalty", fontsize=11, fontweight='bold')
    ax_bar.legend(fontsize=9)
    ax_bar.grid(axis='y', linestyle='--', alpha=0.5)

    # 3. Learned Filters / Receptive Fields (Standard AE)
    ax_filt_std = fig.add_subplot(2, 2, 3)
    weights_std = models[0.0].layers[1].get_weights()[0]  # Shape: (784, 64)
    grid_std = np.zeros((28 * 4, 28 * 4))
    for i in range(16):
        r, c = i // 4, i % 4
        filt = weights_std[:, i].reshape(28, 28)
        norm_f = (filt - filt.min()) / (filt.max() - filt.min() + 1e-8)
        grid_std[r*28:(r+1)*28, c*28:(c+1)*28] = norm_f
    ax_filt_std.imshow(grid_std, cmap='viridis')
    ax_filt_std.set_title("Learned Filters: Standard AE (No Sparsity)", fontsize=11, fontweight='bold')
    ax_filt_std.axis('off')

    # 4. Learned Filters / Receptive Fields (Sparse AE L1=1e-3)
    ax_filt_sp = fig.add_subplot(2, 2, 4)
    weights_sp = models[1e-3].layers[1].get_weights()[0]
    grid_sp = np.zeros((28 * 4, 28 * 4))
    for i in range(16):
        r, c = i // 4, i % 4
        filt = weights_sp[:, i].reshape(28, 28)
        norm_f = (filt - filt.min()) / (filt.max() - filt.min() + 1e-8)
        grid_sp[r*28:(r+1)*28, c*28:(c+1)*28] = norm_f
    ax_filt_sp.imshow(grid_sp, cmap='viridis')
    ax_filt_sp.set_title("Learned Filters: Sparse AE (L1=1e-3)", fontsize=11, fontweight='bold')
    ax_filt_sp.axis('off')

    plt.tight_layout()
    plt.savefig("lab71_sparse_autoencoder.png", dpi=300)
    print("\n[+] Sparsity comparison visualisations saved to 'lab71_sparse_autoencoder.png'")
    plt.show()

if __name__ == "__main__":
    main()

# EXPECTED OUTPUT
# =================================================================
#   LAB 71: SPARSE AUTOENCODER WITH L1 ACTIVITY REGULARIZATION
# =================================================================
#
# [+] Loaded Dataset: Training=(5000, 784), Test=(1000, 784)
#
# --- Training Autoencoder with Sparsity Penalty (L1=0.0) ---
# ...
# --- Training Autoencoder with Sparsity Penalty (L1=0.0001) ---
# ...
# --- Training Autoencoder with Sparsity Penalty (L1=0.001) ---
# ...
#
# =================================================================
# Sparsity Penalty   | Test MSE   | Mean Activation  | Sparsity (<0.05)
# =================================================================
# L1 = 0.0           | ~0.0150    | ~0.35 - 0.45     | ~2.00%
# L1 = 0.0001        | ~0.0185    | ~0.15 - 0.22     | ~35.00%
# L1 = 0.001         | ~0.0240    | ~0.04 - 0.08     | ~75.00%
#
# [+] Sparsity comparison visualisations saved to 'lab71_sparse_autoencoder.png'

# RESULT
# Established that incorporating an L1 activity penalty heavily penalizes continuous hidden
# activation, driving the vast majority of bottleneck neurons toward zero activation,
# yielding localized, stroke-like dictionary basis filters with minimal degradation in reconstruction MSE.
