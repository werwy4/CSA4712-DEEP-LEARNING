# Experiment No: 20
# Title: L2 Regularization, Dropout and Early Stopping Comparison
# Aim: Compare L2 regularisation, dropout and early stopping as remedies
#      for overfitting on a diabetes prediction dataset.
# Libraries: TensorFlow, Keras, NumPy, Matplotlib, Scikit-learn

# PROGRAM

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
tf.random.set_seed(42)

# ── Dataset (synthetic diabetes-like) ─────────────────────────────
X, y = make_classification(n_samples=500, n_features=8, n_informative=6,
                            n_redundant=2, random_state=42)
X = StandardScaler().fit_transform(X)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val,  y_train, y_val  = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

EPOCHS = 100
BATCH  = 16

# ── Build models ───────────────────────────────────────────────────
def baseline():
    m = keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(8,)),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return m

def l2_model(lam):
    reg = keras.regularizers.l2(lam)
    m = keras.Sequential([
        keras.layers.Dense(512, activation='relu', kernel_regularizer=reg, input_shape=(8,)),
        keras.layers.Dense(256, activation='relu', kernel_regularizer=reg),
        keras.layers.Dense(128, activation='relu', kernel_regularizer=reg),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return m

def dropout_model(rate):
    m = keras.Sequential([
        keras.layers.Dense(512, activation='relu', input_shape=(8,)),
        keras.layers.Dropout(rate),
        keras.layers.Dense(256, activation='relu'),
        keras.layers.Dropout(rate),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(rate),
        keras.layers.Dense(1, activation='sigmoid')
    ])
    m.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return m

# ── Train all models ───────────────────────────────────────────────
results = {}

print("\nTraining Baseline (Overfit)...")
h = baseline().fit(X_train, y_train, validation_data=(X_val, y_val),
                   epochs=EPOCHS, batch_size=BATCH, verbose=0)
results['Baseline'] = h.history

for lam in [0.001, 0.01, 0.1]:
    print(f"Training L2 (λ={lam})...")
    h = l2_model(lam).fit(X_train, y_train, validation_data=(X_val, y_val),
                          epochs=EPOCHS, batch_size=BATCH, verbose=0)
    results[f'L2 λ={lam}'] = h.history

for rate in [0.2, 0.3, 0.5]:
    print(f"Training Dropout (rate={rate})...")
    h = dropout_model(rate).fit(X_train, y_train, validation_data=(X_val, y_val),
                                epochs=EPOCHS, batch_size=BATCH, verbose=0)
    results[f'Dropout {rate}'] = h.history

print("Training with Early Stopping...")
es_model = baseline()
es_cb    = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10,
                                          restore_best_weights=True)
h = es_model.fit(X_train, y_train, validation_data=(X_val, y_val),
                 epochs=EPOCHS, batch_size=BATCH, callbacks=[es_cb], verbose=0)
results['Early Stop'] = h.history
print(f"  Stopped at epoch: {len(h.history['loss'])}")

# ── Summary table ─────────────────────────────────────────────────
print(f"\n{'Model':>18} {'Train Acc':>11} {'Val Acc':>10} {'Test Acc':>10} {'Gap':>8}")
print("-" * 62)

def eval_model(m, X, y):
    return m.evaluate(X, y, verbose=0)[1]

# Re-train and evaluate each model type briefly
def get_metrics(hist):
    train_acc = hist['accuracy'][-1]
    val_acc   = hist['val_accuracy'][-1]
    return train_acc, val_acc

for name, hist in results.items():
    train_acc, val_acc = get_metrics(hist)
    gap = train_acc - val_acc
    print(f"{name:>18} {train_acc:>11.4f} {val_acc:>10.4f} {'--':>10} {gap:>8.4f}")

# ── Plots ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. Baseline overfitting
ax = axes[0, 0]
h  = results['Baseline']
ax.plot(h['accuracy'],     label='Train')
ax.plot(h['val_accuracy'], label='Val', linestyle='--')
ax.set_title('Baseline (Overfitting)')
ax.set_xlabel('Epoch'); ax.set_ylabel('Accuracy')
ax.legend(); ax.grid(True, alpha=0.3)

# 2. L2 comparison
ax = axes[0, 1]
for lam in [0.001, 0.01, 0.1]:
    h = results[f'L2 λ={lam}']
    ax.plot(h['val_accuracy'], label=f'λ={lam}')
ax.plot(results['Baseline']['val_accuracy'], 'k--', label='Baseline')
ax.set_title('L2 Regularisation – Val Accuracy')
ax.set_xlabel('Epoch'); ax.legend(); ax.grid(True, alpha=0.3)

# 3. Dropout comparison
ax = axes[1, 0]
for rate in [0.2, 0.3, 0.5]:
    h = results[f'Dropout {rate}']
    ax.plot(h['val_accuracy'], label=f'rate={rate}')
ax.plot(results['Baseline']['val_accuracy'], 'k--', label='Baseline')
ax.set_title('Dropout – Val Accuracy')
ax.set_xlabel('Epoch'); ax.legend(); ax.grid(True, alpha=0.3)

# 4. Early stopping
ax = axes[1, 1]
h  = results['Early Stop']
ax.plot(h['accuracy'],     label='Train')
ax.plot(h['val_accuracy'], label='Val', linestyle='--')
ax.axvline(len(h['accuracy'])-1, color='red', linestyle=':', label='Stopped')
ax.set_title(f'Early Stopping (stopped at epoch {len(h["accuracy"])})')
ax.set_xlabel('Epoch'); ax.legend(); ax.grid(True, alpha=0.3)

plt.suptitle('Overfitting Remedies Comparison: L2, Dropout, Early Stopping', fontsize=12)
plt.tight_layout()
plt.savefig('lab20_regularisation.png', dpi=100)
plt.show()

# EXPECTED OUTPUT
# Baseline shows high train accuracy but lower val accuracy (overfitting gap).
# L2 and dropout reduce the gap. Early stopping halts at the optimal epoch.
# λ=0.01 and dropout=0.3 often give best validation accuracy.

# RESULT
# Overfitting occurs when a large model memorises training data.
# L2 regularisation penalises large weights, dropout randomly disables
# neurons during training, and early stopping monitors validation loss.
# All three techniques reduce overfitting; the best choice depends on
# dataset size and model architecture.
