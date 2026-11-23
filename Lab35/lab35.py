# Experiment No: 35
# Title: BPTT Flow and Gradient Analysis
# Aim: Trace backpropagation through time and measure gradient magnitude
#      at each time step for stock price RNN.
# Libraries: TensorFlow, Keras, NumPy, Matplotlib

# PROGRAM

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

np.random.seed(42)
tf.random.set_seed(42)

# ── 1. Generate synthetic stock-price data ─────────────────────────
n = 2000
t = np.arange(n)
price = (100 + 0.05 * t + 20 * np.sin(2 * np.pi * t / 252)
         + np.cumsum(np.random.normal(0, 1, n)))
price = (price - price.min()) / (price.max() - price.min())   # scale to [0,1]

def make_sequences(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i+look_back])
        y.append(data[i+look_back])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

# ── 2. Build RNN with different activations ────────────────────────
def build_rnn(seq_len, activation='tanh'):
    model = keras.Sequential([
        keras.layers.SimpleRNN(16, return_sequences=False,
                               activation=activation,
                               input_shape=(seq_len, 1)),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer='adam', loss='mse')
    return model

# ── 3. Gradient norm analysis across sequence lengths ─────────────
seq_lengths  = [10, 50, 100]
activations  = ['tanh', 'relu']

fig, axes = plt.subplots(len(seq_lengths), 2,
                          figsize=(14, 4 * len(seq_lengths)))

print("Gradient analysis:")
for row, seq_len in enumerate(seq_lengths):
    X, y_data = make_sequences(price, seq_len)
    X = X[:500, :, np.newaxis]
    y_data = y_data[:500]

    for col, act in enumerate(activations):
        model = build_rnn(seq_len, activation=act)
        # Quick training
        model.fit(X, y_data, epochs=5, batch_size=64, verbose=0)

        # Compute gradients w.r.t. each time step
        X_sample = X[:32]
        y_sample = y_data[:32]
        time_grads = []

        with tf.GradientTape() as tape:
            X_tf  = tf.constant(X_sample)
            tape.watch(X_tf)
            preds = model(X_tf, training=False)
            loss  = tf.reduce_mean(tf.square(preds - y_sample))

        input_grad = tape.gradient(loss, X_tf)
        if input_grad is not None:
            # Norm of gradient at each time step
            grad_norms = tf.norm(input_grad, axis=-1)   # (batch, seq)
            mean_norms = grad_norms.numpy().mean(axis=0)
            time_grads = mean_norms
        else:
            time_grads = np.ones(seq_len) * np.nan

        ax = axes[row, col]
        ax.plot(range(seq_len), time_grads, 'b-o', markersize=3)
        ax.set_title(f'seq_len={seq_len}, act={act}')
        ax.set_xlabel('Time Step')
        ax.set_ylabel('Gradient Norm')
        ax.grid(True, alpha=0.3)
        print(f"  seq={seq_len}, act={act}: "
              f"first_step_grad={time_grads[0]:.6f}, "
              f"last_step_grad={time_grads[-1]:.6f}")

plt.suptitle('BPTT Gradient Magnitude at Each Time Step', fontsize=12)
plt.tight_layout()
plt.savefig('lab35_bptt_grads.png', dpi=100)
plt.show()

# ── 4. Training loss comparison ────────────────────────────────────
print("\nTraining SimpleRNN for loss comparison...")
seq_len = 50
X50, y50 = make_sequences(price, seq_len)
split = int(0.8 * len(X50))
X_tr, X_te = X50[:split, :, np.newaxis], X50[split:, :, np.newaxis]
y_tr, y_te = y50[:split], y50[split:]

histories = {}
for act in activations:
    m = build_rnn(seq_len, activation=act)
    h = m.fit(X_tr, y_tr, validation_data=(X_te, y_te),
              epochs=20, batch_size=64, verbose=0)
    histories[act] = h.history

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for act, hist in histories.items():
    axes[0].plot(hist['loss'],     label=f'{act} train')
    axes[0].plot(hist['val_loss'], label=f'{act} val', linestyle='--')
axes[0].set_title('Training Loss: tanh vs ReLU (seq_len=50)')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('MSE')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

# ── 5. Gradient norms vs depth (vanishing gradient explanation) ────
n_layers = [1, 2, 3, 4, 5]
grads_by_depth = []

for n in n_layers:
    layers = []
    for _ in range(n):
        layers.append(keras.layers.SimpleRNN(16, return_sequences=True))
    layers.append(keras.layers.SimpleRNN(16, return_sequences=False))
    layers.append(keras.layers.Dense(1))
    model_d = keras.Sequential(layers)
    model_d.build(input_shape=(None, 10, 1))

    X_d = tf.constant(np.random.randn(16, 10, 1).astype('float32'))
    with tf.GradientTape() as tape:
        tape.watch(X_d)
        out  = model_d(X_d, training=False)
        loss = tf.reduce_mean(tf.square(out))
    grd = tape.gradient(loss, X_d)
    grd_norm = tf.norm(grd).numpy() if grd is not None else 0.0
    grads_by_depth.append(grd_norm)
    print(f"  {n} RNN layer(s): gradient norm = {grd_norm:.6f}")

axes[1].plot(n_layers, grads_by_depth, 'ro-', markersize=8)
axes[1].set_title('Gradient Norm vs Number of Stacked RNN Layers')
axes[1].set_xlabel('Number of RNN Layers'); axes[1].set_ylabel('Input Gradient Norm')
axes[1].grid(True, alpha=0.3)

plt.suptitle('Backpropagation Through Time (BPTT) Analysis', fontsize=12)
plt.tight_layout()
plt.savefig('lab35_training.png', dpi=100)
plt.show()

# EXPECTED OUTPUT
# Gradient norms decrease towards earlier time steps (vanishing gradient).
# ReLU activations may show different vanishing/exploding patterns than tanh.
# More stacked layers → smaller gradient norm at input.

# RESULT
# BPTT propagates gradients backwards through time steps. For long sequences,
# repeated multiplication by weight matrices causes vanishing gradients
# (gradients → 0) or exploding gradients (gradients → ∞).
# LSTM and GRU architectures address this via gating mechanisms.
