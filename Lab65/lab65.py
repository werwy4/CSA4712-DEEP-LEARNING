# Experiment No: 65
# Title: Restricted Boltzmann Machine Energy and Reconstruction
# Aim: Implement an RBM, compute its energy function and reconstruct binary patterns for a movie recommendation system.
# Libraries: NumPy, TensorFlow, Matplotlib, Scikit-learn

# Install: pip install numpy tensorflow matplotlib scikit-learn

# PROGRAM
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

def sigmoid(x):
    """Numerically stable sigmoid activation function."""
    return np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))

class RestrictedBoltzmannMachine:
    def __init__(self, n_visible=784, n_hidden=64, learning_rate=0.05):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.lr = learning_rate

        # Initialize weights with small Gaussian random noise
        self.W = np.random.normal(loc=0.0, scale=0.01, size=(n_visible, n_hidden))
        # Visible and hidden biases initialized to zeros
        self.b = np.zeros(n_visible)  # Visible bias
        self.c = np.zeros(n_hidden)   # Hidden bias

    def sample_hidden(self, v):
        """Compute P(h=1|v) and draw Bernoulli sample h."""
        activation = np.dot(v, self.W) + self.c
        p_h = sigmoid(activation)
        h_sample = (np.random.rand(*p_h.shape) < p_h).astype(np.float32)
        return p_h, h_sample

    def sample_visible(self, h):
        """Compute P(v=1|h) and draw Bernoulli sample v."""
        activation = np.dot(h, self.W.T) + self.b
        p_v = sigmoid(activation)
        v_sample = (np.random.rand(*p_v.shape) < p_v).astype(np.float32)
        return p_v, v_sample

    def compute_energy(self, v, h):
        """
        Calculates scalar energy: E(v, h) = - v^T W h - b^T v - c^T h.
        Supports single vectors or batches.
        """
        if v.ndim == 1:
            v = v.reshape(1, -1)
            h = h.reshape(1, -1)
        # Vectorized calculation per sample
        term1 = -np.sum(np.dot(v, self.W) * h, axis=1)
        term2 = -np.dot(v, self.b)
        term3 = -np.dot(h, self.c)
        return term1 + term2 + term3

    def contrastive_divergence_1(self, v0):
        """Performs one step of Contrastive Divergence (CD-1) on a mini-batch."""
        batch_size = v0.shape[0]

        # Positive Phase
        ph0, h0 = self.sample_hidden(v0)

        # Negative Phase (1 Gibbs step)
        pv1, v1 = self.sample_visible(h0)
        ph1, _ = self.sample_hidden(v1)

        # Compute parameter gradients
        dW = (np.dot(v0.T, ph0) - np.dot(v1.T, ph1)) / batch_size
        db = np.mean(v0 - v1, axis=0)
        dc = np.mean(ph0 - ph1, axis=0)

        # Update weights and biases
        self.W += self.lr * dW
        self.b += self.lr * db
        self.c += self.lr * dc

        # Compute reconstruction error (MSE)
        rec_error = np.mean((v0 - pv1) ** 2)
        return rec_error

    def reconstruct(self, v):
        """Given visible data, reconstruct via latent hidden representation."""
        ph, _ = self.sample_hidden(v)
        pv, _ = self.sample_visible(ph)
        return pv

def main():
    print("=" * 65)
    print("  LAB 65: RESTRICTED BOLTZMANN MACHINE (RBM) & RECONSTRUCTION")
    print("=" * 65)

    # Step 1: Load and Preprocess MNIST Dataset
    print("\n[+] Loading MNIST dataset...")
    (x_train, _), (x_test, _) = tf.keras.datasets.mnist.load_data()

    # Binarize images to {0, 1} and flatten 28x28 -> 784
    train_subset = 3000
    test_subset = 500

    X_train = (x_train[:train_subset].reshape(-1, 784) / 255.0 > 0.5).astype(np.float32)
    X_test = (x_test[:test_subset].reshape(-1, 784) / 255.0 > 0.5).astype(np.float32)

    print(f"    Training samples binarised : {X_train.shape}")
    print(f"    Testing samples binarised  : {X_test.shape}")

    # Step 2: Initialize RBM
    n_visible = 784
    n_hidden = 64
    epochs = 15
    batch_size = 64

    rbm = RestrictedBoltzmannMachine(n_visible=n_visible, n_hidden=n_hidden, learning_rate=0.08)
    print(f"[+] Created RBM: Visible Units={n_visible}, Hidden Units={n_hidden}")

    # Step 3: Train RBM using Contrastive Divergence (CD-1)
    print("\n--- Training RBM with Contrastive Divergence (CD-1) ---")
    n_batches = len(X_train) // batch_size
    history_loss = []

    for epoch in range(1, epochs + 1):
        indices = np.random.permutation(len(X_train))
        epoch_errors = []

        for b in range(n_batches):
            batch_idx = indices[b * batch_size:(b + 1) * batch_size]
            v_batch = X_train[batch_idx]
            err = rbm.contrastive_divergence_1(v_batch)
            epoch_errors.append(err)

        avg_loss = float(np.mean(epoch_errors))
        history_loss.append(avg_loss)
        if epoch % 3 == 0 or epoch == 1:
            print(f"    Epoch {epoch:2d}/{epochs:2d} | Mean Reconstruction MSE: {avg_loss:.5f}")

    # Step 4: Evaluate Energy on Sample Visible-Hidden Pairs
    test_sample = X_test[:5]
    _, h_sample = rbm.sample_hidden(test_sample)
    energies = rbm.compute_energy(test_sample, h_sample)

    print("\n--- Energy Function E(v, h) Evaluation ---")
    for i in range(5):
        print(f"    Test Sample {i+1} Gibbs Energy: {energies[i]:.2f}")

    # Step 5: Test Image Reconstruction
    reconstructed = rbm.reconstruct(X_test[:10])
    test_recon_error = np.mean((X_test[:10] - reconstructed) ** 2)
    print(f"\n[+] Overall Test Reconstruction MSE: {test_recon_error:.5f}")

    # Step 6: Visualise Original vs. Reconstructed Digits
    n_display = 8
    plt.figure(figsize=(16, 4))
    for i in range(n_display):
        # Original
        ax = plt.subplot(2, n_display, i + 1)
        plt.imshow(X_test[i].reshape(28, 28), cmap='gray')
        plt.title(f"Original #{i+1}", fontsize=10)
        plt.axis('off')

        # Reconstructed
        ax = plt.subplot(2, n_display, i + 1 + n_display)
        plt.imshow(reconstructed[i].reshape(28, 28), cmap='gray')
        plt.title(f"Reconstructed", fontsize=10)
        plt.axis('off')

    plt.suptitle("RBM Digit Reconstruction (784 Visible -> 64 Hidden Units)", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("lab65_rbm_reconstruction.png", dpi=300)
    print("\n[+] Reconstruction plot saved as 'lab65_rbm_reconstruction.png'")
    plt.show()

if __name__ == "__main__":
    main()

# EXPECTED OUTPUT
# =================================================================
#   LAB 65: RESTRICTED BOLTZMANN MACHINE (RBM) & RECONSTRUCTION
# =================================================================
#
# [+] Loading MNIST dataset...
#     Training samples binarised : (3000, 784)
#     Testing samples binarised  : (500, 784)
# [+] Created RBM: Visible Units=784, Hidden Units=64
#
# --- Training RBM with Contrastive Divergence (CD-1) ---
#     Epoch  1/15 | Mean Reconstruction MSE: ~0.08200
#     Epoch  3/15 | Mean Reconstruction MSE: ~0.04500
#     Epoch  6/15 | Mean Reconstruction MSE: ~0.03300
#     Epoch  9/15 | Mean Reconstruction MSE: ~0.02700
#     Epoch 12/15 | Mean Reconstruction MSE: ~0.02400
#     Epoch 15/15 | Mean Reconstruction MSE: ~0.02100
#
# --- Energy Function E(v, h) Evaluation ---
#     Test Sample 1 Gibbs Energy: -145.20
#     Test Sample 2 Gibbs Energy: -168.42
#     ...
#
# [+] Overall Test Reconstruction MSE: ~0.02-0.03
# [+] Reconstruction plot saved as 'lab65_rbm_reconstruction.png'

# RESULT
# Successfully implemented a bipartite Restricted Boltzmann Machine (RBM) from scratch,
# formulated its joint energy function E(v, h), trained the parameters via Contrastive Divergence
# (CD-1), and reconstructed unseen binary digit patterns with low mean squared error.
