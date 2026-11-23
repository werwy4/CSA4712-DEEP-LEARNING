# Experiment No: 24
# Title: Softmax Classification and Cross-Entropy Loss
# Aim: Implement multiclass neural network output using softmax with
#      categorical cross-entropy for equipment condition classification.
# Libraries: NumPy, TensorFlow, Keras, Matplotlib

# PROGRAM

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
tf.random.set_seed(42)

CLASS_NAMES = ['Normal', 'Warning', 'Critical']

# ── 1. Create 3-class equipment condition dataset ─────────────────
X_raw, y = make_blobs(n_samples=600, centers=3, n_features=4,
                      cluster_std=1.5, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X_raw)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# One-hot encode
y_train_oh = keras.utils.to_categorical(y_train, 3)
y_test_oh  = keras.utils.to_categorical(y_test, 3)

print(f"Dataset: {X.shape}, Classes: {np.unique(y)}")

# ── 2. Implement softmax manually ─────────────────────────────────
def softmax(logits):
    """Numerically stable softmax"""
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    exp_z   = np.exp(shifted)
    return exp_z / exp_z.sum(axis=-1, keepdims=True)

# Test on sample logits
sample_logits = np.array([[2.0, 1.0, 0.1],
                           [-1.0, 3.0, 2.0],
                           [0.5,  0.5, 0.5]])
probs = softmax(sample_logits)
print("\nSoftmax output:")
print(probs.round(4))
print(f"Row sums (must = 1): {probs.sum(axis=1).round(6)}")

# ── 3. Cross-entropy implementations ──────────────────────────────
def categorical_crossentropy(y_true_oh, y_pred_prob):
    eps = 1e-12
    y_pred_prob = np.clip(y_pred_prob, eps, 1-eps)
    return -np.mean(np.sum(y_true_oh * np.log(y_pred_prob), axis=1))

def sparse_categorical_crossentropy(y_true_int, y_pred_prob):
    eps = 1e-12
    y_pred_prob = np.clip(y_pred_prob, eps, 1-eps)
    n = len(y_true_int)
    return -np.mean(np.log(y_pred_prob[np.arange(n), y_true_int]))

# Example
y_true_sample = np.array([[1,0,0],[0,1,0],[0,0,1]])
ce_cat   = categorical_crossentropy(y_true_sample, probs)
ce_sparse = sparse_categorical_crossentropy(np.array([0,1,2]), probs)
print(f"\nCategorical CE:        {ce_cat:.4f}")
print(f"Sparse Categorical CE: {ce_sparse:.4f}")
print(f"(Both should be equal: {abs(ce_cat - ce_sparse) < 1e-6})")

# ── 4. Build and train a small neural classifier ───────────────────
model = keras.Sequential([
    keras.layers.Dense(32, activation='relu', input_shape=(4,)),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(3,  activation='softmax')
])
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

history = model.fit(X_train, y_train_oh,
                    validation_split=0.2,
                    epochs=50, batch_size=16, verbose=0)

test_loss, test_acc = model.evaluate(X_test, y_test_oh, verbose=0)
print(f"\nTest Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")

# ── 5. Compare predicted probabilities with true labels ───────────
sample_probs = model.predict(X_test[:6], verbose=0)
sample_preds = np.argmax(sample_probs, axis=1)
print("\nPredicted class probabilities (first 6 test samples):")
print(f"{'Sample':>8} {'Normal':>10} {'Warning':>10} {'Critical':>10} {'Pred':>8} {'True':>8}")
for i in range(6):
    print(f"{i+1:>8} {sample_probs[i,0]:>10.4f} {sample_probs[i,1]:>10.4f} "
          f"{sample_probs[i,2]:>10.4f} {CLASS_NAMES[sample_preds[i]]:>8} "
          f"{CLASS_NAMES[y_test[i]]:>8}")

# ── 6. Plots ───────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Loss curve
axes[0,0].plot(history.history['loss'],     label='Train')
axes[0,0].plot(history.history['val_loss'], label='Val')
axes[0,0].set_title('Training Loss (Categorical CE)')
axes[0,0].set_xlabel('Epoch'); axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

# Accuracy curve
axes[0,1].plot(history.history['accuracy'],     label='Train')
axes[0,1].plot(history.history['val_accuracy'], label='Val')
axes[0,1].set_title('Training Accuracy')
axes[0,1].set_xlabel('Epoch'); axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

# Softmax probability bars for 6 samples
x_pos = np.arange(3)
for i, ax in enumerate(axes[1, :2] if i < 2 else []):
    pass
# Show probability distribution for 3 samples
for idx, ax in enumerate([axes[1,0], axes[1,1]]):
    sample_idx = idx
    ax.bar(x_pos, sample_probs[sample_idx], color=['green','orange','red'], edgecolor='k')
    ax.set_xticks(x_pos); ax.set_xticklabels(CLASS_NAMES)
    ax.set_ylim(0, 1); ax.set_ylabel('Probability')
    ax.set_title(f"Sample {sample_idx+1}: Pred={CLASS_NAMES[sample_preds[sample_idx]]}, "
                 f"True={CLASS_NAMES[y_test[sample_idx]]}")
    ax.grid(True, alpha=0.3)

plt.suptitle('Softmax Classification and Categorical Cross-Entropy', fontsize=12)
plt.tight_layout()
plt.savefig('lab24_softmax.png', dpi=100)
plt.show()

# EXPECTED OUTPUT
# Softmax probabilities summing to 1. Cross-entropy values computed.
# Loss decreasing and accuracy increasing over epochs.
# Probability bars showing confidence distribution over 3 classes.

# RESULT
# Softmax converts raw logits into a valid probability distribution over
# classes. Categorical cross-entropy measures the divergence between the
# predicted and true distribution, serving as the loss for multi-class
# classification. The model correctly predicts equipment conditions with
# high confidence for clearly separable samples.
