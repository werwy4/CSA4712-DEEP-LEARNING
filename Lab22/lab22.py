# Experiment No: 22
# Title: CNN Architecture and Hyperparameter Comparison
# Aim: Compare CNN architectures and hyperparameters systematically on CIFAR-10
#      to balance accuracy, model size and training time.
# Libraries: TensorFlow, Keras, NumPy, Matplotlib

# PROGRAM

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import time

np.random.seed(42)
tf.random.set_seed(42)

# ── Load CIFAR-10 (use small subset for faster comparison) ─────────
(X_train_full, y_train_full), (X_test, y_test) = keras.datasets.cifar10.load_data()
X_train_full = X_train_full.astype('float32') / 255.0
X_test       = X_test.astype('float32') / 255.0

# Small subset for speed
N = 10000
X_tr = X_train_full[:N];  y_tr = y_train_full[:N]
X_vl = X_train_full[N:N+2000]; y_vl = y_train_full[N:N+2000]

y_tr_oh = keras.utils.to_categorical(y_tr, 10)
y_vl_oh = keras.utils.to_categorical(y_vl, 10)
y_ts_oh = keras.utils.to_categorical(y_test, 10)

EPOCHS = 10;  BATCH = 64

# ── CNN builder ────────────────────────────────────────────────────
def build_cnn(n_conv_layers=2, filters=32, kernel_size=3,
              pool_type='max', use_batchnorm=False):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(32,32,3)))
    for _ in range(n_conv_layers):
        model.add(keras.layers.Conv2D(filters, (kernel_size, kernel_size),
                                      activation='relu', padding='same'))
        if use_batchnorm:
            model.add(keras.layers.BatchNormalization())
        if pool_type == 'max':
            model.add(keras.layers.MaxPooling2D(2, 2))
        else:
            model.add(keras.layers.AveragePooling2D(2, 2))

    model.add(keras.layers.Flatten())
    model.add(keras.layers.Dense(128, activation='relu'))
    model.add(keras.layers.Dense(10, activation='softmax'))
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

# ── Experiment configurations ──────────────────────────────────────
configs = [
    {'n_conv_layers': 2, 'filters': 32,  'kernel_size': 3, 'pool_type': 'max', 'use_batchnorm': False},
    {'n_conv_layers': 3, 'filters': 32,  'kernel_size': 3, 'pool_type': 'max', 'use_batchnorm': False},
    {'n_conv_layers': 2, 'filters': 64,  'kernel_size': 3, 'pool_type': 'max', 'use_batchnorm': False},
    {'n_conv_layers': 2, 'filters': 32,  'kernel_size': 5, 'pool_type': 'max', 'use_batchnorm': False},
    {'n_conv_layers': 2, 'filters': 32,  'kernel_size': 3, 'pool_type': 'avg', 'use_batchnorm': False},
    {'n_conv_layers': 2, 'filters': 32,  'kernel_size': 3, 'pool_type': 'max', 'use_batchnorm': True},
    {'n_conv_layers': 3, 'filters': 64,  'kernel_size': 3, 'pool_type': 'max', 'use_batchnorm': True},
]

records = []
for cfg in configs:
    name = (f"conv={cfg['n_conv_layers']}, f={cfg['filters']}, "
            f"k={cfg['kernel_size']}, pool={cfg['pool_type']}, "
            f"bn={cfg['use_batchnorm']}")
    print(f"Training: {name}")
    model = build_cnn(**cfg)
    n_params = model.count_params()
    t0 = time.time()
    model.fit(X_tr, y_tr_oh, validation_data=(X_vl, y_vl_oh),
              epochs=EPOCHS, batch_size=BATCH, verbose=0)
    elapsed = time.time() - t0
    _, val_acc = model.evaluate(X_vl, y_vl_oh, verbose=0)
    records.append({'Config': name, 'Params': n_params,
                    'Val Acc': round(val_acc, 4), 'Time(s)': round(elapsed, 1)})

df = pd.DataFrame(records)
print("\n── Results Table ──")
print(df.to_string(index=False))

# ── Plot comparison ────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

short_names = [f"C{i+1}" for i in range(len(records))]
val_accs = [r['Val Acc'] for r in records]
params   = [r['Params']  for r in records]
times    = [r['Time(s)'] for r in records]

axes[0].barh(short_names, val_accs, color='steelblue', edgecolor='k')
axes[0].set_xlabel('Validation Accuracy')
axes[0].set_title('Val Accuracy by Config')
axes[0].set_xlim(0, 1)

axes[1].barh(short_names, params, color='tomato', edgecolor='k')
axes[1].set_xlabel('Number of Parameters')
axes[1].set_title('Model Size (Parameters)')

axes[2].barh(short_names, times, color='forestgreen', edgecolor='k')
axes[2].set_xlabel('Training Time (seconds)')
axes[2].set_title('Training Time')

plt.suptitle('CNN Architecture and Hyperparameter Comparison (CIFAR-10 subset)',
             fontsize=12)
plt.tight_layout()
plt.savefig('lab22_cnn_compare.png', dpi=100)
plt.show()

# ── Legend for config names ────────────────────────────────────────
print("\nConfig Legend:")
for i, r in enumerate(records):
    print(f"  C{i+1}: {r['Config']}")

# EXPECTED OUTPUT
# Results table with validation accuracy, parameter count, and training time.
# BatchNorm + more filters generally improves accuracy at cost of params/time.

# RESULT
# More convolutional layers and filters improve accuracy but increase
# training time and model size. Batch normalisation stabilises training
# and often improves accuracy. 3×3 filters are more parameter-efficient
# than 5×5 for similar accuracy. Max pooling slightly outperforms average
# pooling on natural images.
