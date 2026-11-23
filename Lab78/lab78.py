# Experiment No: 78
# Title: Federated Learning Simulation
# Aim: Simulate a federated learning setup where multiple clients train
#      local models on private data, and a central server aggregates
#      updates using FedAvg.
# Libraries: TensorFlow, Keras, NumPy, Pandas, Matplotlib

# PROGRAM

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from copy import deepcopy

np.random.seed(42)
tf.random.set_seed(42)

# ── 1. Federated setup on MNIST ───────────────────────────────────
print("Loading MNIST...")
(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train.reshape(-1, 784).astype('float32') / 255.0
X_test  = X_test.reshape(-1,  784).astype('float32') / 255.0

N_CLIENTS = 5
ROUNDS    = 15
LOCAL_EPOCHS = 3
BATCH_SIZE   = 64

# IID partition: shuffle and split evenly
perm = np.random.permutation(len(X_train))
X_train, y_train = X_train[perm], y_train[perm]
client_size = len(X_train) // N_CLIENTS

client_data = []
for i in range(N_CLIENTS):
    s = i * client_size
    e = s + client_size
    client_data.append((X_train[s:e], y_train[s:e]))
    print(f"  Client {i+1}: {e-s} samples, classes={np.unique(y_train[s:e])}")

# ── 2. Build model factory ────────────────────────────────────────
def build_model():
    model = keras.Sequential([
        keras.layers.Dense(128, activation='relu', input_shape=(784,)),
        keras.layers.Dense(64,  activation='relu'),
        keras.layers.Dense(10,  activation='softmax')
    ])
    model.compile(optimizer='sgd', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ── 3. FedAvg helper functions ────────────────────────────────────
def get_weights(model):
    return [w.numpy() for w in model.trainable_weights]

def set_weights(model, weights):
    for var, w in zip(model.trainable_weights, weights):
        var.assign(w)

def fedavg(client_weights):
    """Average weights across all clients."""
    avg = [np.zeros_like(w) for w in client_weights[0]]
    for weights in client_weights:
        for i, w in enumerate(weights):
            avg[i] += w / len(client_weights)
    return avg

# ── 4. Federated training loop ────────────────────────────────────
global_model = build_model()
# Global model starts with random weights

round_metrics = []
central_metrics = []   # for comparison: central training on full data

print("\n── Federated Learning: FedAvg ──")
for rnd in range(1, ROUNDS + 1):
    client_weight_list = []
    client_accs = []

    for cid, (X_c, y_c) in enumerate(client_data):
        # Each client gets global weights and trains locally
        local_model = build_model()
        set_weights(local_model, get_weights(global_model))
        local_model.fit(X_c, y_c,
                        epochs=LOCAL_EPOCHS,
                        batch_size=BATCH_SIZE,
                        verbose=0)
        client_weight_list.append(get_weights(local_model))
        _, acc = local_model.evaluate(X_c, y_c, verbose=0)
        client_accs.append(acc)

    # Server aggregates
    new_weights = fedavg(client_weight_list)
    set_weights(global_model, new_weights)

    # Evaluate global model on test set
    test_loss, test_acc = global_model.evaluate(X_test, y_test, verbose=0)
    round_metrics.append({
        'round': rnd,
        'test_loss': test_loss,
        'test_acc':  test_acc,
        'avg_client_acc': np.mean(client_accs)
    })
    print(f"  Round {rnd:2d} | Test Acc={test_acc:.4f} | "
          f"Test Loss={test_loss:.4f} | Avg Client Acc={np.mean(client_accs):.4f}")

df = pd.DataFrame(round_metrics)

# ── 5. Central baseline (trained on full data) ────────────────────
print("\n── Central Baseline Training ──")
central_model = build_model()
central_hist  = central_model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    epochs=ROUNDS * LOCAL_EPOCHS,
    batch_size=BATCH_SIZE, verbose=0
)
_, central_acc = central_model.evaluate(X_test, y_test, verbose=0)
print(f"  Central Final Test Accuracy: {central_acc:.4f}")

# ── 6. Plots ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(df['round'], df['test_acc'], 'b-o', label='Federated (FedAvg)')
central_ep = range(LOCAL_EPOCHS, ROUNDS*LOCAL_EPOCHS+1, LOCAL_EPOCHS)
axes[0].axhline(central_acc, color='red', linestyle='--', label=f'Central ({central_acc:.4f})')
axes[0].set_title('Test Accuracy: Federated vs Central')
axes[0].set_xlabel('Communication Round')
axes[0].set_ylabel('Test Accuracy')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(df['round'], df['test_loss'], 'steelblue', marker='o')
axes[1].set_title('Federated Global Model – Test Loss per Round')
axes[1].set_xlabel('Communication Round')
axes[1].set_ylabel('Test Loss'); axes[1].grid(True, alpha=0.3)

# Per-round client accuracy
axes[2].plot(df['round'], df['avg_client_acc'], 'g-o', label='Avg Client Acc')
axes[2].plot(df['round'], df['test_acc'],        'b-s', label='Global Test Acc')
axes[2].set_title('Client Accuracy vs Global Accuracy')
axes[2].set_xlabel('Round'); axes[2].set_ylabel('Accuracy')
axes[2].legend(); axes[2].grid(True, alpha=0.3)

plt.suptitle(f'Federated Learning Simulation (5 Clients, FedAvg, {ROUNDS} Rounds)',
             fontsize=12)
plt.tight_layout()
plt.savefig('lab78_federated.png', dpi=100)
plt.show()

# ── 7. Final summary ───────────────────────────────────────────────
print("\n── Summary ──")
print(df.to_string(index=False))
print(f"\nFederated Final Accuracy : {df['test_acc'].iloc[-1]:.4f}")
print(f"Central Baseline Accuracy: {central_acc:.4f}")
print(f"Accuracy Gap             : {abs(central_acc - df['test_acc'].iloc[-1]):.4f}")

# EXPECTED OUTPUT
# Federated accuracy rises from ~90% to ~96-97% across 15 rounds.
# Central baseline achieves ~97-98% on full MNIST.
# FedAvg closes most of the gap without sharing raw data.

# RESULT
# Federated learning enables collaborative model training without sharing
# private data. FedAvg aggregates client gradients (weights) on the server.
# The global model converges to near-central performance while preserving
# data privacy on each client. Communication overhead is proportional to
# the number of rounds and clients.
