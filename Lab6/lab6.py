# Experiment No: 6
# Title: Gradient Descent, SGD, Momentum and Adam Comparison
# Aim: Implement and compare batch GD, SGD, Momentum and Adam optimisers
#      for a delivery-time prediction model.
# Libraries: NumPy, TensorFlow, Keras, Matplotlib

# PROGRAM

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import make_regression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

np.random.seed(42)
tf.random.set_seed(42)

# ════════════════════════════════════════════════════════════════════
# PART A – Optimisers on a 2-D loss surface (from scratch)
# ════════════════════════════════════════════════════════════════════

def loss_fn(w):
    """Simple bowl: f(w1,w2) = w1^2 + 5*w2^2"""
    return w[0]**2 + 5 * w[1]**2

def grad_fn(w):
    return np.array([2 * w[0], 10 * w[1]])

def run_optimizer(name, lr=0.1, n_steps=60, momentum=0.9):
    w    = np.array([3.0, 2.0])
    path = [w.copy()]
    vel  = np.zeros(2)   # for momentum
    m1   = np.zeros(2)   # Adam 1st moment
    m2   = np.zeros(2)   # Adam 2nd moment
    eps  = 1e-8
    b1, b2 = 0.9, 0.999

    for t in range(1, n_steps + 1):
        g = grad_fn(w)
        if name == 'GD':
            w = w - lr * g
        elif name == 'SGD':
            noise = np.random.randn(2) * 0.3    # simulate stochastic
            w = w - lr * (g + noise)
        elif name == 'Momentum':
            vel = momentum * vel + lr * g
            w   = w - vel
        elif name == 'Adam':
            m1 = b1 * m1 + (1 - b1) * g
            m2 = b2 * m2 + (1 - b2) * g**2
            m1_hat = m1 / (1 - b1**t)
            m2_hat = m2 / (1 - b2**t)
            w = w - lr * m1_hat / (np.sqrt(m2_hat) + eps)
        path.append(w.copy())
    return np.array(path)

# Plot contour + paths
w1 = np.linspace(-3.5, 3.5, 300)
w2 = np.linspace(-2.5, 2.5, 300)
W1, W2 = np.meshgrid(w1, w2)
Z = W1**2 + 5 * W2**2

fig, axes = plt.subplots(2, 2, figsize=(12, 10))
optimizers = ['GD', 'SGD', 'Momentum', 'Adam']
colors     = ['royalblue', 'tomato', 'forestgreen', 'darkorange']

for ax, opt, col in zip(axes.flatten(), optimizers, colors):
    path = run_optimizer(opt, lr=0.08)
    ax.contour(W1, W2, Z, levels=20, cmap='gray', alpha=0.6)
    ax.plot(path[:, 0], path[:, 1], '-o', color=col, markersize=3, label=opt)
    ax.plot(0, 0, 'r*', markersize=15)
    ax.set_title(f'{opt} Optimiser Path')
    ax.set_xlabel('w₁'); ax.set_ylabel('w₂')
    ax.legend()

plt.suptitle('Optimiser Paths on 2-D Loss Surface', fontsize=13)
plt.tight_layout()
plt.savefig('lab6_paths.png', dpi=100)
plt.show()

# ════════════════════════════════════════════════════════════════════
# PART B – Keras regression model with different optimisers
# ════════════════════════════════════════════════════════════════════
X, y = make_regression(n_samples=1000, n_features=4, noise=20, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

def build_model(optimizer):
    model = keras.Sequential([
        keras.layers.Dense(64, activation='relu', input_shape=(4,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(1)
    ])
    model.compile(optimizer=optimizer, loss='mse')
    return model

opts = {
    'SGD'          : keras.optimizers.SGD(learning_rate=0.01),
    'SGD+Momentum' : keras.optimizers.SGD(learning_rate=0.01, momentum=0.9),
    'Adam'         : keras.optimizers.Adam(learning_rate=0.01)
}

histories = {}
for name, opt in opts.items():
    print(f"Training with {name}...")
    m = build_model(opt)
    h = m.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=60, batch_size=32, verbose=0)
    histories[name] = h.history

plt.figure(figsize=(10, 5))
for name, hist in histories.items():
    plt.plot(hist['val_loss'], label=name)
plt.xlabel('Epoch')
plt.ylabel('Validation MSE Loss')
plt.title('Keras Regression: Loss vs Epochs for Different Optimisers')
plt.legend()
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lab6_keras.png', dpi=100)
plt.show()

# ── Learning-rate sensitivity ──────────────────────────────────────
print("\nLearning Rate Sensitivity (Adam):")
for lr in [0.1, 0.01, 0.001]:
    m = build_model(keras.optimizers.Adam(learning_rate=lr))
    h = m.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)
    final_loss = h.history['loss'][-1]
    print(f"  LR={lr}: final train loss = {final_loss:.2f}")

# EXPECTED OUTPUT
# 2-D contour plots showing each optimiser's path towards the minimum.
# Adam converges fastest; SGD without momentum may oscillate.
# Keras loss curves show Adam outperforming plain SGD.

# RESULT
# Adam adapts learning rates per parameter, achieving fastest convergence.
# SGD with momentum is faster than plain SGD. Very high learning rates
# cause divergence; very low rates cause slow convergence.
