# Experiment No: 67
# Title: Deep Belief Network Construction and Training
# Aim: Build a Deep Belief Network by stacking RBMs layer by layer for digit recognition.
# Libraries: NumPy, TensorFlow, Keras, Matplotlib, Scikit-learn

# Install: pip install numpy tensorflow matplotlib scikit-learn

# PROGRAM
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

def sigmoid(x):
    """Numerically stable sigmoid."""
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))

class SimpleRBM:
    """Restricted Boltzmann Machine for layer-wise pretraining."""
    def __init__(self, n_visible, n_hidden, lr=0.05, seed=42):
        np.random.seed(seed)
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.lr = lr
        self.W = np.random.normal(loc=0.0, scale=0.01, size=(n_visible, n_hidden)).astype(np.float32)
        self.b = np.zeros(n_visible, dtype=np.float32)
        self.c = np.zeros(n_hidden, dtype=np.float32)

    def sample_h(self, v):
        p_h = sigmoid(np.dot(v, self.W) + self.c)
        h = (np.random.rand(*p_h.shape) < p_h).astype(np.float32)
        return p_h, h

    def sample_v(self, h):
        p_v = sigmoid(np.dot(h, self.W.T) + self.b)
        v = (np.random.rand(*p_v.shape) < p_v).astype(np.float32)
        return p_v, v

    def train_epoch(self, X, batch_size=64):
        n_batches = len(X) // batch_size
        perm = np.random.permutation(len(X))
        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            v0 = X[idx]

            # CD-1
            ph0, h0 = self.sample_h(v0)
            pv1, v1 = self.sample_v(h0)
            ph1, _ = self.sample_h(pv1)

            # Update
            self.W += self.lr * (np.dot(v0.T, ph0) - np.dot(pv1.T, ph1)) / batch_size
            self.b += self.lr * np.mean(v0 - pv1, axis=0)
            self.c += self.lr * np.mean(ph0 - ph1, axis=0)

    def get_hidden_representation(self, X):
        """Pass inputs forward to get expected hidden features."""
        return sigmoid(np.dot(X, self.W) + self.c)

def build_mlp_model(layer_sizes, output_classes=10):
    """Builds a standard Sequential MLP."""
    model = keras.Sequential()
    model.add(keras.Input(shape=(layer_sizes[0],)))
    for units in layer_sizes[1:]:
        model.add(layers.Dense(units, activation='sigmoid'))
    model.add(layers.Dense(output_classes, activation='softmax'))
    return model

def main():
    print("=" * 65)
    print("  LAB 67: DEEP BELIEF NETWORK (DBN) PRETRAINING & FINE-TUNING")
    print("=" * 65)

    # Step 1: Load and Preprocess Data
    (x_train_raw, y_train_raw), (x_test_raw, y_test_raw) = tf.keras.datasets.mnist.load_data()

    train_size = 4000
    test_size = 1000

    X_train = (x_train_raw[:train_size].reshape(-1, 784) / 255.0 > 0.5).astype(np.float32)
    y_train = y_train_raw[:train_size]

    X_test = (x_test_raw[:test_size].reshape(-1, 784) / 255.0 > 0.5).astype(np.float32)
    y_test = y_test_raw[:test_size]

    print(f"\n[+] Data Subsets Loaded: Train={X_train.shape}, Test={X_test.shape}")

    # Step 2: Greedy Layer-Wise RBM Pre-Training (784 -> 256 -> 128 -> 64)
    arch = [784, 256, 128, 64]
    rbm_layers = []
    current_input = X_train

    print("\n--- Greedy Layer-Wise RBM Pre-Training ---")
    pretrain_epochs = 5

    for l in range(len(arch) - 1):
        n_vis = arch[l]
        n_hid = arch[l + 1]
        print(f"\n[+] Pre-training RBM Layer {l+1}: Visible={n_vis} -> Hidden={n_hid}")
        rbm = SimpleRBM(n_visible=n_vis, n_hidden=n_hid, lr=0.08, seed=42 + l)

        for ep in range(1, pretrain_epochs + 1):
            rbm.train_epoch(current_input, batch_size=64)
            print(f"    RBM Layer {l+1} | Epoch {ep}/{pretrain_epochs} completed.")

        rbm_layers.append(rbm)
        # Compute activations for next layer
        current_input = rbm.get_hidden_representation(current_input)

    # Step 3: Construct DBN and Transfer Pretrained Weights
    print("\n[+] Constructing DBN and transferring learned RBM weights...")
    dbn_model = build_mlp_model(arch, output_classes=10)

    # Transfer pretrained weights to corresponding Dense layers
    for i, rbm in enumerate(rbm_layers):
        dbn_model.layers[i].set_weights([rbm.W, rbm.c])
    print("[+] RBM weights and hidden biases successfully transferred.")

    # Step 4: Construct Baseline Random-Initialized MLP
    print("[+] Constructing Baseline MLP (Random Glorot Initialisation)...")
    baseline_mlp = build_mlp_model(arch, output_classes=10)

    # Step 5: Supervised Fine-Tuning Comparison
    fine_tune_epochs = 12
    batch_size = 64

    dbn_model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.002),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    baseline_mlp.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.002),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    print("\n--- Fine-tuning Pretrained DBN ---")
    dbn_hist = dbn_model.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=fine_tune_epochs,
        batch_size=batch_size,
        verbose=1
    )

    print("\n--- Training Baseline MLP From Scratch ---")
    mlp_hist = baseline_mlp.fit(
        X_train, y_train,
        validation_data=(X_test, y_test),
        epochs=fine_tune_epochs,
        batch_size=batch_size,
        verbose=1
    )

    # Step 6: Evaluate Test Accuracies
    _, dbn_test_acc = dbn_model.evaluate(X_test, y_test, verbose=0)
    _, mlp_test_acc = baseline_mlp.evaluate(X_test, y_test, verbose=0)

    print("\n" + "=" * 65)
    print("  PERFORMANCE EVALUATION: DBN VS. BASELINE MLP")
    print("=" * 65)
    print(f"  Pretrained DBN Test Accuracy  : {dbn_test_acc * 100:.2f}%")
    print(f"  Random Init MLP Test Accuracy : {mlp_test_acc * 100:.2f}%")
    print(f"  Accuracy Advantage of DBN     : {(dbn_test_acc - mlp_test_acc) * 100:+.2f}%")

    # Step 7: Plot Training and Validation Curves
    plt.figure(figsize=(13, 5))

    # Loss comparison
    plt.subplot(1, 2, 1)
    plt.plot(dbn_hist.history['loss'], 'b-o', label='DBN Pretrained Train Loss')
    plt.plot(dbn_hist.history['val_loss'], 'b--', label='DBN Pretrained Val Loss')
    plt.plot(mlp_hist.history['loss'], 'r-s', label='MLP Baseline Train Loss')
    plt.plot(mlp_hist.history['val_loss'], 'r--', label='MLP Baseline Val Loss')
    plt.title('Cross-Entropy Loss vs. Epochs', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Loss', fontsize=11)
    plt.legend(fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.6)

    # Accuracy comparison
    plt.subplot(1, 2, 2)
    plt.plot(dbn_hist.history['accuracy'], 'b-o', label='DBN Pretrained Train Acc')
    plt.plot(dbn_hist.history['val_accuracy'], 'b--', label='DBN Pretrained Val Acc')
    plt.plot(mlp_hist.history['accuracy'], 'r-s', label='MLP Baseline Train Acc')
    plt.plot(mlp_hist.history['val_accuracy'], 'r--', label='MLP Baseline Val Acc')
    plt.title('Classification Accuracy vs. Epochs', fontsize=12, fontweight='bold')
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Accuracy', fontsize=11)
    plt.legend(fontsize=9)
    plt.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout()
    plt.savefig("lab67_dbn_training.png", dpi=300)
    print("\n[+] Training curves saved as 'lab67_dbn_training.png'")
    plt.show()

if __name__ == "__main__":
    main()

# EXPECTED OUTPUT
# =================================================================
#   LAB 67: DEEP BELIEF NETWORK (DBN) PRETRAINING & FINE-TUNING
# =================================================================
#
# [+] Data Subsets Loaded: Train=(4000, 784), Test=(1000, 784)
#
# --- Greedy Layer-Wise RBM Pre-Training ---
# [+] Pre-training RBM Layer 1: Visible=784 -> Hidden=256
# [+] Pre-training RBM Layer 2: Visible=256 -> Hidden=128
# [+] Pre-training RBM Layer 3: Visible=128 -> Hidden=64
#
# [+] Constructing DBN and transferring learned RBM weights...
# [+] Constructing Baseline MLP (Random Glorot Initialisation)...
#
# --- Fine-tuning Pretrained DBN ---
# ...
# --- Training Baseline MLP From Scratch ---
# ...
#
# =================================================================
#   PERFORMANCE EVALUATION: DBN VS. BASELINE MLP
# =================================================================
#   Pretrained DBN Test Accuracy  : ~92.50% - 94.00%
#   Random Init MLP Test Accuracy : ~88.00% - 90.50%
#   Accuracy Advantage of DBN     : +2.0% to +4.0%
#
# [+] Training curves saved as 'lab67_dbn_training.png'

# RESULT
# Demonstrated the power of unsupervised greedy layer-wise pre-training using stacked RBMs
# to form a Deep Belief Network (DBN), showing faster convergence and higher final test
# accuracy compared to an identically structured deep neural network initialized at random.
