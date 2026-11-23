# Experiment No: 25
# Title: Weight Initialisation Strategies
# Aim: Compare weight initialisation methods and their influence on
#      forward activations and backpropagated gradients.
# Libraries: NumPy, TensorFlow, Matplotlib

# PROGRAM

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
tf.random.set_seed(42)

X, y = make_classification(n_samples=2000, n_features=20, n_informative=15,
                            random_state=42)
X = StandardScaler().fit_transform(X)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

n_layers  = 5
n_units   = 64
N_EPOCHS  = 30
BATCH     = 64

# ── Initialisation methods ─────────────────────────────────────────
init_methods = {
    'Zeros'         : keras.initializers.Zeros(),
    'Random Normal' : keras.initializers.RandomNormal(stddev=0.01),
    'Xavier/Glorot' : keras.initializers.GlorotUniform(),
    'He'            : keras.initializers.HeUniform(),
}

# ── Build model with given initialiser ────────────────────────────
def build_model(init, activation='relu'):
    layers = [keras.layers.Input(shape=(20,))]
    for _ in range(n_layers):
        layers.append(keras.layers.Dense(n_units, activation=activation,
                                         kernel_initializer=init))
    layers.append(keras.layers.Dense(1, activation='sigmoid'))
    model = keras.Sequential(layers)
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

# ── 1. Activation distribution at each layer ─────────────────────
print("Analysing activation distributions...")
fig, axes = plt.subplots(len(init_methods), n_layers,
                          figsize=(3*n_layers, 3*len(init_methods)))

for row, (name, init) in enumerate(init_methods.items()):
    model = build_model(init)
    # Intermediate models for each layer
    sample = X_train[:500]
    activations = [sample]
    inp = tf.constant(sample, dtype=tf.float32)
    for layer in model.layers:
        inp = layer(inp)
        if hasattr(layer, 'activation') and layer.activation is not None:
            activations.append(inp.numpy())
    actual_acts = activations[1:n_layers+1]

    for col in range(min(n_layers, len(actual_acts))):
        ax = axes[row, col]
        vals = actual_acts[col].ravel()
        ax.hist(vals, bins=30, color='steelblue', edgecolor='k', alpha=0.7)
        ax.set_title(f'{name}\nLayer {col+1}', fontsize=8)
        ax.set_xlabel('Activation value', fontsize=7)
        if col == 0:
            ax.set_ylabel('Count', fontsize=7)
        mean_v, std_v = vals.mean(), vals.std()
        ax.text(0.98, 0.95, f'μ={mean_v:.2f}\nσ={std_v:.3f}',
                transform=ax.transAxes, ha='right', va='top', fontsize=7)

plt.suptitle('Activation Distributions by Initialisation Method', fontsize=12)
plt.tight_layout()
plt.savefig('lab25_activations.png', dpi=100)
plt.show()

# ── 2. Training convergence comparison ────────────────────────────
histories = {}
for name, init in init_methods.items():
    print(f"Training with {name}...")
    model = build_model(init)
    h = model.fit(X_train, y_train, validation_data=(X_val, y_val),
                  epochs=N_EPOCHS, batch_size=BATCH, verbose=0)
    histories[name] = h.history

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['royalblue', 'tomato', 'forestgreen', 'darkorange']

for (name, hist), col in zip(histories.items(), colors):
    axes[0].plot(hist['loss'],         label=name, color=col)
    axes[1].plot(hist['val_accuracy'], label=name, color=col)

axes[0].set_title('Training Loss by Initialisation')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

axes[1].set_title('Validation Accuracy by Initialisation')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

plt.suptitle('Weight Initialisation – Convergence Comparison', fontsize=12)
plt.tight_layout()
plt.savefig('lab25_convergence.png', dpi=100)
plt.show()

# ── 3. Gradient magnitude analysis ────────────────────────────────
print("\n── Gradient Magnitude at First Layer ──")
print(f"{'Initialisation':>18} {'Mean |grad|':>14} {'Std |grad|':>13}")
print("-" * 48)

for name, init in init_methods.items():
    model = build_model(init)
    with tf.GradientTape() as tape:
        preds = model(X_train[:100].astype('float32'), training=True)
        loss  = keras.losses.binary_crossentropy(
                    y_train[:100].astype('float32'), tf.squeeze(preds))
    grads = tape.gradient(loss, model.trainable_variables)
    if grads[0] is not None:
        g = grads[0].numpy().ravel()
        print(f"{name:>18} {np.abs(g).mean():>14.6f} {np.abs(g).std():>13.6f}")
    else:
        print(f"{name:>18} {'None':>14}")

# ── 4. Recommendation ─────────────────────────────────────────────
print("\n── Recommendation ──")
print("  Zeros       : NEVER use – all neurons learn identical features.")
print("  Random Norm : Small stddev can help, but may still have vanishing gradients.")
print("  Xavier/Glorot: Best for sigmoid/tanh activations (maintains variance).")
print("  He          : Best for ReLU/Leaky ReLU (accounts for half-activation).")

# EXPECTED OUTPUT
# Activation histograms: Zeros → all activations equal 0.
# Xavier → reasonable spread; He → slightly wider, better for ReLU.
# Convergence: He/Xavier converge fast; Zeros/Random-Normal converge slowly or not at all.

# RESULT
# Proper weight initialisation ensures that activations and gradients
# maintain reasonable magnitudes across layers. He initialisation is
# optimal for ReLU-based networks; Xavier/Glorot suits sigmoid/tanh.
# Zero initialisation causes symmetry-breaking failure.
