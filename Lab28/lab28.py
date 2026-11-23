# Experiment No: 28
# Title: Transfer Learning for Image Classification
# Aim: Apply transfer learning using pretrained MobileNetV2 for a
#      manufacturing defect detection binary classification task.
# Libraries: TensorFlow, Keras, Matplotlib, NumPy

# PROGRAM

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay

np.random.seed(42)
tf.random.set_seed(42)

# ── 1. Create synthetic defect dataset using CIFAR-10 ─────────────
# Use classes: airplane(0)=Acceptable, bird(2)=Defective (texture contrast)
(X_train_full, y_train_full), (X_test_full, y_test_full) = keras.datasets.cifar10.load_data()

def make_binary_dataset(X, y, class_a=0, class_b=2, n_per_class=1000):
    mask_a = (y.ravel() == class_a)
    mask_b = (y.ravel() == class_b)
    Xa = X[mask_a][:n_per_class]
    Xb = X[mask_b][:n_per_class]
    ya = np.zeros(len(Xa), dtype=int)
    yb = np.ones(len(Xb),  dtype=int)
    X_out = np.concatenate([Xa, Xb])
    y_out = np.concatenate([ya, yb])
    # Shuffle
    idx = np.random.permutation(len(y_out))
    return X_out[idx], y_out[idx]

X_train, y_train = make_binary_dataset(X_train_full, y_train_full, n_per_class=1500)
X_test,  y_test  = make_binary_dataset(X_test_full,  y_test_full,  n_per_class=300)

# Resize to 96×96 for MobileNetV2 (minimum 32×32)
def preprocess(X):
    X = tf.image.resize(X, (96, 96)).numpy()
    return tf.keras.applications.mobilenet_v2.preprocess_input(X)

print("Preprocessing images...")
X_train_p = preprocess(X_train.astype('float32'))
X_test_p  = preprocess(X_test.astype('float32'))

print(f"Train: {X_train_p.shape}, Test: {X_test_p.shape}")
print(f"Class distribution: {np.bincount(y_train)}")

# ── 2. Build transfer learning model ──────────────────────────────
base_model = keras.applications.MobileNetV2(
    input_shape=(96, 96, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False   # freeze base

x = keras.layers.GlobalAveragePooling2D()(base_model.output)
x = keras.layers.Dense(128, activation='relu')(x)
x = keras.layers.Dropout(0.3)(x)
outputs = keras.layers.Dense(1, activation='sigmoid')(x)

model_tl = keras.Model(inputs=base_model.input, outputs=outputs, name='TL_MobileNetV2')
model_tl.compile(optimizer=keras.optimizers.Adam(1e-3),
                 loss='binary_crossentropy', metrics=['accuracy'])
print(f"Total params: {model_tl.count_params():,}")
print(f"Trainable params: {sum([np.prod(v.shape) for v in model_tl.trainable_variables]):,}")

# ── 3. Phase 1: Train new head (frozen base) ───────────────────────
print("\nPhase 1: Training new classification head...")
h1 = model_tl.fit(X_train_p, y_train,
                  validation_split=0.2,
                  epochs=10, batch_size=32, verbose=1)

# ── 4. Phase 2: Fine-tune last 20 layers of base ──────────────────
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model_tl.compile(optimizer=keras.optimizers.Adam(1e-4),   # lower LR
                 loss='binary_crossentropy', metrics=['accuracy'])
print(f"Fine-tune trainable params: "
      f"{sum([np.prod(v.shape) for v in model_tl.trainable_variables]):,}")

print("\nPhase 2: Fine-tuning last 20 base layers...")
h2 = model_tl.fit(X_train_p, y_train,
                  validation_split=0.2,
                  epochs=10, batch_size=32, verbose=1)

# ── 5. Evaluate ────────────────────────────────────────────────────
y_pred = (model_tl.predict(X_test_p, verbose=0) > 0.5).astype(int).ravel()
test_acc = (y_pred == y_test).mean()
print(f"\nTest Accuracy: {test_acc*100:.2f}%")
print(classification_report(y_test, y_pred, target_names=['Acceptable', 'Defective']))

# ── 6. Plots ───────────────────────────────────────────────────────
all_acc  = h1.history['accuracy']     + h2.history['accuracy']
all_vac  = h1.history['val_accuracy'] + h2.history['val_accuracy']
all_loss = h1.history['loss']         + h2.history['loss']
all_vloss= h1.history['val_loss']     + h2.history['val_loss']

fig, axes = plt.subplots(1, 3, figsize=(16, 5))

axes[0].plot(all_acc,  label='Train Acc')
axes[0].plot(all_vac,  label='Val Acc')
axes[0].axvline(10, color='black', linestyle='--', label='Fine-tune starts')
axes[0].set_title('Accuracy over Training Phases')
axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(all_loss,  label='Train Loss')
axes[1].plot(all_vloss, label='Val Loss')
axes[1].axvline(10, color='black', linestyle='--', label='Fine-tune starts')
axes[1].set_title('Loss over Training Phases')
axes[1].set_xlabel('Epoch'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

cm = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['Acceptable', 'Defective'])
disp.plot(ax=axes[2], cmap='Blues', colorbar=False)
axes[2].set_title('Confusion Matrix')

plt.suptitle('Transfer Learning: MobileNetV2 for Defect Detection', fontsize=12)
plt.tight_layout()
plt.savefig('lab28_transfer.png', dpi=100)
plt.show()

# EXPECTED OUTPUT
# Phase 1 accuracy improves quickly (pretrained features used immediately).
# Phase 2 fine-tuning further improves accuracy slightly.
# High test accuracy (~85-95%) with few training samples.

# RESULT
# Transfer learning leverages ImageNet-pretrained features for a new task.
# Phase 1 trains only the classification head using rich frozen features.
# Phase 2 fine-tunes upper base layers for domain adaptation.
# This approach achieves high accuracy with limited labelled data,
# which is critical in manufacturing where defect samples are scarce.
