# Experiment No: 29
# Title: Data Augmentation for CNN Generalisation
# Aim: Evaluate image augmentation as a strategy for improving CNN
#      generalisation on Fashion-MNIST medical-style classification.
# Libraries: TensorFlow, Keras, NumPy, Matplotlib

# PROGRAM

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

np.random.seed(42)
tf.random.set_seed(42)

CLASS_NAMES = ['T-shirt', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Boot']

# ── 1. Load Fashion-MNIST ─────────────────────────────────────────
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

# Add channel dimension and normalise
X_train = X_train.astype('float32')[..., np.newaxis] / 255.0
X_test  = X_test.astype('float32')[..., np.newaxis]  / 255.0

# Small training set to highlight augmentation benefit
N = 5000
X_train_small = X_train[:N]
y_train_small = y_train[:N]

print(f"Train (small): {X_train_small.shape}, Test: {X_test.shape}")

# ── 2. Display sample images ──────────────────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for ax, img, label in zip(axes.flatten(), X_test[:10], y_test[:10]):
    ax.imshow(img.squeeze(), cmap='gray')
    ax.set_title(CLASS_NAMES[label], fontsize=9)
    ax.axis('off')
plt.suptitle('Fashion-MNIST Sample Images', fontsize=12)
plt.tight_layout()
plt.savefig('lab29_samples.png', dpi=100)
plt.show()

# ── 3. Augmentation pipeline ──────────────────────────────────────
augmentation = keras.Sequential([
    keras.layers.RandomFlip('horizontal'),
    keras.layers.RandomRotation(0.1),
    keras.layers.RandomTranslation(0.1, 0.1),
    keras.layers.RandomZoom(0.1),
    keras.layers.RandomContrast(0.1),
], name='augmentation')

# Show augmentation effect
fig, axes = plt.subplots(3, 6, figsize=(14, 7))
orig_img = X_train_small[0:1]
axes[0, 0].imshow(orig_img[0].squeeze(), cmap='gray')
axes[0, 0].set_title('Original', fontsize=8); axes[0, 0].axis('off')
for i in range(1, 6):
    aug = augmentation(orig_img, training=True).numpy()
    axes[0, i].imshow(aug[0].squeeze(), cmap='gray')
    axes[0, i].set_title(f'Aug {i}', fontsize=8); axes[0, i].axis('off')

for op_row, op_layer in enumerate([
    keras.layers.RandomRotation(0.2),
    keras.layers.RandomZoom(0.2)
], start=1):
    axes[op_row, 0].imshow(orig_img[0].squeeze(), cmap='gray')
    axes[op_row, 0].set_title('Original', fontsize=8); axes[op_row, 0].axis('off')
    for i in range(1, 6):
        aug = op_layer(orig_img, training=True).numpy()
        axes[op_row, i].imshow(aug[0].squeeze(), cmap='gray')
        op_name = op_layer.name.replace('random_', '').replace('_', ' ').title()
        axes[op_row, i].set_title(f'{op_name} {i}', fontsize=7)
        axes[op_row, i].axis('off')

plt.suptitle('Image Augmentation Effects', fontsize=12)
plt.tight_layout()
plt.savefig('lab29_augmentation.png', dpi=100)
plt.show()

# ── 4. CNN builder ─────────────────────────────────────────────────
def build_cnn(augment=False):
    layers = []
    if augment:
        layers.append(augmentation)
    layers += [
        keras.layers.Conv2D(32, 3, activation='relu', padding='same',
                            input_shape=(28,28,1)),
        keras.layers.MaxPooling2D(2,2),
        keras.layers.Conv2D(64, 3, activation='relu', padding='same'),
        keras.layers.MaxPooling2D(2,2),
        keras.layers.Flatten(),
        keras.layers.Dense(128, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(10, activation='softmax')
    ]
    model = keras.Sequential(layers)
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model

# ── 5. Train baseline (no augmentation) ───────────────────────────
EPOCHS = 20; BATCH = 64
print("\nTraining without augmentation...")
m_base = build_cnn(augment=False)
h_base = m_base.fit(X_train_small, y_train_small,
                     validation_data=(X_test, y_test),
                     epochs=EPOCHS, batch_size=BATCH, verbose=0)
base_test = m_base.evaluate(X_test, y_test, verbose=0)[1]
print(f"  Baseline Test Accuracy: {base_test*100:.2f}%")

# ── 6. Train with augmentation ────────────────────────────────────
print("Training with augmentation...")
m_aug = build_cnn(augment=True)
h_aug = m_aug.fit(X_train_small, y_train_small,
                   validation_data=(X_test, y_test),
                   epochs=EPOCHS, batch_size=BATCH, verbose=0)
aug_test = m_aug.evaluate(X_test, y_test, verbose=0)[1]
print(f"  Augmented Test Accuracy: {aug_test*100:.2f}%")

# ── 7. Compare learning curves ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(h_base.history['accuracy'],     'b-',  label='Base Train')
axes[0].plot(h_base.history['val_accuracy'], 'b--', label='Base Val')
axes[0].plot(h_aug.history['accuracy'],      'r-',  label='Aug Train')
axes[0].plot(h_aug.history['val_accuracy'],  'r--', label='Aug Val')
axes[0].set_title('Accuracy vs Epochs')
axes[0].set_xlabel('Epoch'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(h_base.history['loss'],     'b-',  label='Base Train Loss')
axes[1].plot(h_base.history['val_loss'], 'b--', label='Base Val Loss')
axes[1].plot(h_aug.history['loss'],      'r-',  label='Aug Train Loss')
axes[1].plot(h_aug.history['val_loss'],  'r--', label='Aug Val Loss')
axes[1].set_title('Loss vs Epochs')
axes[1].set_xlabel('Epoch'); axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.suptitle(f'Data Augmentation: Base Test={base_test:.3f}, Aug Test={aug_test:.3f}', fontsize=12)
plt.tight_layout()
plt.savefig('lab29_comparison.png', dpi=100)
plt.show()

# EXPECTED OUTPUT
# Baseline shows bigger train-val accuracy gap (more overfitting).
# Augmented model: smaller gap, better test accuracy.
# Augmented training accuracy may be lower (harder augmented images).

# RESULT
# Data augmentation artificially increases dataset diversity by applying
# random transformations. This reduces overfitting, especially with small
# datasets. The augmented CNN generalises better to unseen test images
# as measured by test accuracy and the train-validation accuracy gap.
