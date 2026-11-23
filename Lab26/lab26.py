# Experiment No: 26
# Title: Batch Normalisation in Deep Neural Networks
# Aim: Examine the effect of batch normalisation on training stability
#      and model performance for image classification.
# Libraries: TensorFlow, Keras, NumPy, Matplotlib

# PROGRAM

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import time

np.random.seed(42)
tf.random.set_seed(42)

# ── Dataset ───────────────────────────────────────────────────────
X, y = make_classification(n_samples=3000, n_features=20, n_informative=15,
                            n_classes=3, n_clusters_per_class=1, random_state=42)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

y_train_oh = keras.utils.to_categorical(y_train, 3)
y_test_oh  = keras.utils.to_categorical(y_test,  3)

EPOCHS = 60; BATCH = 64

# ── Model builders ─────────────────────────────────────────────────
def build_baseline():
    return keras.Sequential([
        keras.layers.Dense(256, activation='relu', input_shape=(20,)),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(64,  activation='relu'),
        keras.layers.Dense(32,  activation='relu'),
        keras.layers.Dense(3,   activation='softmax')
    ])

def build_batchnorm():
    return keras.Sequential([
        keras.layers.Dense(256, input_shape=(20,)),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Dense(128),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Dense(64),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Dense(32),
        keras.layers.BatchNormalization(),
        keras.layers.Activation('relu'),
        keras.layers.Dense(3, activation='softmax')
    ])

def compile_and_train(model, label):
    model.compile(optimizer=keras.optimizers.Adam(0.001),
                  loss='categorical_crossentropy', metrics=['accuracy'])
    t0 = time.time()
    h = model.fit(X_train, y_train_oh,
                  validation_data=(X_test, y_test_oh),
                  epochs=EPOCHS, batch_size=BATCH, verbose=0)
    elapsed = time.time() - t0
    print(f"{label}: {elapsed:.1f}s | Final val acc: {h.history['val_accuracy'][-1]:.4f}")
    return h.history, elapsed

print("Training Baseline (no BN)...")
h_base, t_base = compile_and_train(build_baseline(), 'Baseline')
print("Training with Batch Normalisation...")
h_bn, t_bn     = compile_and_train(build_batchnorm(), 'BatchNorm')

# ── Activation statistics through training ────────────────────────
# Examine internal activation distributions before/after BN on test batch
sample = X_test[:200].astype('float32')

for model_name, model_fn in [('Baseline', build_baseline), ('BN Model', build_batchnorm)]:
    model = model_fn()
    model.compile(optimizer='adam', loss='categorical_crossentropy')
    model.fit(X_train, y_train_oh, epochs=5, batch_size=BATCH, verbose=0)

    # Get hidden layer activations
    hidden_outputs = []
    inp = tf.constant(sample)
    for layer in model.layers[:-1]:  # skip output
        inp = layer(inp, training=False)
        try:
            if layer.activation.__name__ in ('relu', 'linear'):
                hidden_outputs.append(inp.numpy())
        except AttributeError:
            pass

# ── Plots ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Loss
axes[0,0].plot(h_base['loss'],     'b-',  label='Baseline Train')
axes[0,0].plot(h_base['val_loss'], 'b--', label='Baseline Val')
axes[0,0].plot(h_bn['loss'],       'r-',  label='BatchNorm Train')
axes[0,0].plot(h_bn['val_loss'],   'r--', label='BatchNorm Val')
axes[0,0].set_title('Loss vs Epochs')
axes[0,0].set_xlabel('Epoch'); axes[0,0].set_ylabel('Loss')
axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

# Accuracy
axes[0,1].plot(h_base['accuracy'],     'b-',  label='Baseline Train')
axes[0,1].plot(h_base['val_accuracy'], 'b--', label='Baseline Val')
axes[0,1].plot(h_bn['accuracy'],       'r-',  label='BatchNorm Train')
axes[0,1].plot(h_bn['val_accuracy'],   'r--', label='BatchNorm Val')
axes[0,1].set_title('Accuracy vs Epochs')
axes[0,1].set_xlabel('Epoch'); axes[0,1].set_ylabel('Accuracy')
axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

# Gradient variance proxy: loss variance
window = 5
base_loss_sm = np.convolve(h_base['loss'], np.ones(window)/window, mode='valid')
bn_loss_sm   = np.convolve(h_bn['loss'],   np.ones(window)/window, mode='valid')
axes[1,0].plot(base_loss_sm, 'b-', label='Baseline (smoothed)')
axes[1,0].plot(bn_loss_sm,   'r-', label='BatchNorm (smoothed)')
axes[1,0].set_title('Smoothed Training Loss (Stability)')
axes[1,0].set_xlabel('Epoch'); axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

# Training time comparison
axes[1,1].bar(['Baseline', 'BatchNorm'], [t_base, t_bn],
              color=['steelblue', 'tomato'], edgecolor='k')
axes[1,1].set_title('Training Time (seconds)')
axes[1,1].set_ylabel('Seconds')
for i, t in enumerate([t_base, t_bn]):
    axes[1,1].text(i, t + 0.5, f'{t:.1f}s', ha='center', fontsize=11)

plt.suptitle('Batch Normalisation: Effect on Training Stability and Performance', fontsize=12)
plt.tight_layout()
plt.savefig('lab26_batchnorm.png', dpi=100)
plt.show()

# ── Summary ───────────────────────────────────────────────────────
print("\n── Summary ──")
print(f"  Baseline  – Final Val Acc: {h_base['val_accuracy'][-1]:.4f}, "
      f"Best Val Acc: {max(h_base['val_accuracy']):.4f}")
print(f"  BatchNorm – Final Val Acc: {h_bn['val_accuracy'][-1]:.4f},   "
      f"Best Val Acc: {max(h_bn['val_accuracy']):.4f}")
print(f"\n  Batch Normalisation advantages:")
print(f"  1. Faster convergence (fewer epochs to reach same accuracy)")
print(f"  2. Reduced internal covariate shift → more stable training")
print(f"  3. Acts as regulariser (slight noise from batch statistics)")
print(f"  4. Allows higher learning rates without divergence")

# EXPECTED OUTPUT
# BatchNorm converges faster and to higher accuracy than baseline.
# Loss curve for BatchNorm is smoother (more stable).
# BN may add slight overhead per epoch but converges in fewer epochs.

# RESULT
# Batch normalisation normalises activations within each mini-batch,
# reducing internal covariate shift. This stabilises and accelerates
# training, often allowing higher learning rates. The model with BN
# achieves better final accuracy and more stable loss curves.
