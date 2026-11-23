# Experiment No: 16
# Title: Learning-Rate Scheduling and Optimisation Stability
# Aim: Examine learning-rate schedules and their influence on optimisation
#      stability and convergence speed for neural network training.
# Libraries: TensorFlow, Keras, NumPy, Matplotlib

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

# ── Dataset ───────────────────────────────────────────────────────
X, y = make_classification(n_samples=2000, n_features=20, n_informative=15,
                            n_classes=2, random_state=42)
scaler = StandardScaler()
X = scaler.fit_transform(X)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

EPOCHS = 60
BATCH  = 64

# ── Build model ───────────────────────────────────────────────────
def build_model():
    m = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(20,)),
        keras.layers.Dense(64,  activation='relu'),
        keras.layers.Dense(32,  activation='relu'),
        keras.layers.Dense(1,   activation='sigmoid')
    ])
    return m

# ── Learning-rate schedules ───────────────────────────────────────
# 1. Fixed LR = 0.01
def run_fixed():
    m = build_model()
    m.compile(optimizer=keras.optimizers.Adam(0.01),
              loss='binary_crossentropy', metrics=['accuracy'])
    h = m.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=EPOCHS, batch_size=BATCH, verbose=0)
    return h.history

# 2. Step decay: halve LR every 15 epochs, start at 0.01
def step_decay_schedule(epoch, lr):
    drop_every = 15
    drop_factor = 0.5
    new_lr = 0.01 * (drop_factor ** (epoch // drop_every))
    return float(new_lr)

def run_step_decay():
    m = build_model()
    m.compile(optimizer=keras.optimizers.Adam(),
              loss='binary_crossentropy', metrics=['accuracy'])
    cb = keras.callbacks.LearningRateScheduler(step_decay_schedule, verbose=0)
    h = m.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=EPOCHS, batch_size=BATCH, callbacks=[cb], verbose=0)
    return h.history

# 3. Exponential decay: LR = 0.01 * 0.95^epoch
def exp_decay_schedule(epoch, lr):
    return float(0.01 * (0.95 ** epoch))

def run_exp_decay():
    m = build_model()
    m.compile(optimizer=keras.optimizers.Adam(),
              loss='binary_crossentropy', metrics=['accuracy'])
    cb = keras.callbacks.LearningRateScheduler(exp_decay_schedule, verbose=0)
    h = m.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=EPOCHS, batch_size=BATCH, callbacks=[cb], verbose=0)
    return h.history

# 4. Cosine annealing: LR = 0.5 * lr_max * (1 + cos(pi * epoch / T))
def cosine_schedule(epoch, lr):
    lr_max = 0.01
    T = EPOCHS
    return float(0.5 * lr_max * (1 + np.cos(np.pi * epoch / T)))

def run_cosine():
    m = build_model()
    m.compile(optimizer=keras.optimizers.Adam(),
              loss='binary_crossentropy', metrics=['accuracy'])
    cb = keras.callbacks.LearningRateScheduler(cosine_schedule, verbose=0)
    h = m.fit(X_train, y_train, validation_data=(X_val, y_val),
              epochs=EPOCHS, batch_size=BATCH, callbacks=[cb], verbose=0)
    return h.history

# ── Run all schedules ─────────────────────────────────────────────
print("Training with Fixed LR...")
h_fixed = run_fixed()
print("Training with Step Decay...")
h_step  = run_step_decay()
print("Training with Exponential Decay...")
h_exp   = run_exp_decay()
print("Training with Cosine Annealing...")
h_cos   = run_cosine()

schedules = {
    'Fixed LR=0.01'      : h_fixed,
    'Step Decay'         : h_step,
    'Exponential Decay'  : h_exp,
    'Cosine Annealing'   : h_cos
}

# ── Compare LR curves ─────────────────────────────────────────────
epochs = np.arange(EPOCHS)
lr_fixed = [0.01] * EPOCHS
lr_step  = [step_decay_schedule(e, None) for e in epochs]
lr_exp   = [exp_decay_schedule(e, None) for e in epochs]
lr_cos   = [cosine_schedule(e, None) for e in epochs]

plt.figure(figsize=(10, 4))
plt.plot(epochs, lr_fixed, label='Fixed')
plt.plot(epochs, lr_step,  label='Step Decay')
plt.plot(epochs, lr_exp,   label='Exponential Decay')
plt.plot(epochs, lr_cos,   label='Cosine Annealing')
plt.xlabel('Epoch'); plt.ylabel('Learning Rate')
plt.title('Learning Rate Schedules')
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lab16_lr_schedules.png', dpi=100)
plt.show()

# ── Compare training/validation loss and accuracy ─────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
colors = ['royalblue', 'tomato', 'forestgreen', 'darkorange']

for (name, hist), col in zip(schedules.items(), colors):
    axes[0].plot(hist['val_loss'],     color=col, label=name)
    axes[1].plot(hist['val_accuracy'], color=col, label=name)

axes[0].set_title('Validation Loss vs Epochs')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Loss')
axes[0].legend(fontsize=9); axes[0].grid(True, alpha=0.3)

axes[1].set_title('Validation Accuracy vs Epochs')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
axes[1].legend(fontsize=9); axes[1].grid(True, alpha=0.3)

plt.suptitle('Learning Rate Schedule Comparison', fontsize=13)
plt.tight_layout()
plt.savefig('lab16_comparison.png', dpi=100)
plt.show()

# ── Summary ───────────────────────────────────────────────────────
print(f"\n{'Schedule':>22} {'Final Val Loss':>16} {'Final Val Acc':>14}")
print("-" * 55)
for name, hist in schedules.items():
    print(f"{name:>22} {hist['val_loss'][-1]:>16.4f} {hist['val_accuracy'][-1]:>14.4f}")

print("\nRecommendation:")
print("  Cosine Annealing: smooth decay, often best final accuracy.")
print("  Exponential Decay: stable, good for most tasks.")
print("  Step Decay: simple, effective when decay points are known.")
print("  Fixed LR: may overshoot near minimum; use as baseline only.")

# EXPECTED OUTPUT
# LR schedule curves showing different decay patterns.
# Validation loss and accuracy curves for all 4 schedules over 60 epochs.
# Cosine annealing and exponential decay typically outperform fixed LR.

# RESULT
# Learning-rate scheduling improves convergence stability.
# Starting with a higher LR for fast learning and reducing it later
# avoids oscillation near the minimum. Cosine annealing provides smooth
# warm restarts and is widely used in modern deep learning training.
