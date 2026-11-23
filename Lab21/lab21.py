# Experiment No: 21
# Title: CNN for Image Classification
# Aim: Build and train a CNN for multi-class image classification
#      using CIFAR-10 dataset for a recycling plant sorting system.
# Libraries: TensorFlow, Keras, NumPy, Matplotlib, Scikit-learn

# PROGRAM

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

np.random.seed(42)
tf.random.set_seed(42)

CLASS_NAMES = ['airplane','automobile','bird','cat','deer',
               'dog','frog','horse','ship','truck']

# ── 1. Load and preprocess CIFAR-10 ───────────────────────────────
(X_train, y_train), (X_test, y_test) = keras.datasets.cifar10.load_data()

# Normalise to [0,1]
X_train = X_train.astype('float32') / 255.0
X_test  = X_test.astype('float32')  / 255.0

# Validation split
X_val, y_val     = X_train[45000:], y_train[45000:]
X_train, y_train = X_train[:45000], y_train[:45000]

# One-hot encode
y_train_oh = keras.utils.to_categorical(y_train, 10)
y_val_oh   = keras.utils.to_categorical(y_val,   10)
y_test_oh  = keras.utils.to_categorical(y_test,  10)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ── 2. Build CNN ───────────────────────────────────────────────────
model = keras.Sequential([
    keras.layers.Conv2D(32, (3,3), activation='relu', padding='same',
                        input_shape=(32,32,3)),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(32, (3,3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Dropout(0.25),

    keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    keras.layers.BatchNormalization(),
    keras.layers.Conv2D(64, (3,3), activation='relu', padding='same'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Dropout(0.25),

    keras.layers.Flatten(),
    keras.layers.Dense(256, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(10,  activation='softmax')
], name='CIFAR10_CNN')

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

# ── 3. Train ───────────────────────────────────────────────────────
history = model.fit(X_train, y_train_oh,
                    validation_data=(X_val, y_val_oh),
                    epochs=20, batch_size=64, verbose=1)

# ── 4. Evaluate ────────────────────────────────────────────────────
test_loss, test_acc = model.evaluate(X_test, y_test_oh, verbose=0)
print(f"\nTest Accuracy: {test_acc*100:.2f}%  |  Test Loss: {test_loss:.4f}")

# ── 5. Accuracy and loss curves ────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
axes[0].plot(history.history['accuracy'],     label='Train')
axes[0].plot(history.history['val_accuracy'], label='Val')
axes[0].set_title('Accuracy vs Epochs'); axes[0].legend(); axes[0].grid(True, alpha=0.3)
axes[1].plot(history.history['loss'],     label='Train')
axes[1].plot(history.history['val_loss'], label='Val')
axes[1].set_title('Loss vs Epochs'); axes[1].legend(); axes[1].grid(True, alpha=0.3)
plt.suptitle('CNN Training on CIFAR-10', fontsize=12)
plt.tight_layout()
plt.savefig('lab21_curves.png', dpi=100)
plt.show()

# ── 6. Confusion matrix ────────────────────────────────────────────
y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
y_true = y_test.ravel()
cm = confusion_matrix(y_true, y_pred)

fig, ax = plt.subplots(figsize=(10, 8))
disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
disp.plot(ax=ax, cmap='Blues', colorbar=True)
plt.title('CIFAR-10 CNN – Confusion Matrix')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('lab21_cm.png', dpi=100)
plt.show()

# ── 7. Sample misclassifications ──────────────────────────────────
wrong_idx = np.where(y_pred != y_true)[0][:9]
fig, axes = plt.subplots(3, 3, figsize=(8, 8))
for ax, idx in zip(axes.flatten(), wrong_idx):
    ax.imshow(X_test[idx])
    ax.set_title(f"True:{CLASS_NAMES[y_true[idx]]}\nPred:{CLASS_NAMES[y_pred[idx]]}", fontsize=7)
    ax.axis('off')
plt.suptitle('Sample Misclassifications', fontsize=11)
plt.tight_layout()
plt.savefig('lab21_misclass.png', dpi=100)
plt.show()

# ── 8. Feature maps from first conv layer ─────────────────────────
feat_model = keras.Model(inputs=model.input,
                         outputs=model.layers[0].output)
sample_img = X_test[0:1]
feat_maps  = feat_model.predict(sample_img, verbose=0)

fig, axes = plt.subplots(4, 8, figsize=(16, 8))
for i, ax in enumerate(axes.flatten()):
    ax.imshow(feat_maps[0, :, :, i], cmap='viridis')
    ax.axis('off')
plt.suptitle('Feature Maps – First Conv Layer (32 filters)', fontsize=11)
plt.tight_layout()
plt.savefig('lab21_features.png', dpi=100)
plt.show()

# EXPECTED OUTPUT
# Test accuracy ~65-75% after 20 epochs on CIFAR-10.
# Confusion matrix shows most confusion between visually similar classes.
# Feature maps reveal learned edge and texture detectors.

# RESULT
# A CNN with two convolutional blocks achieves significantly better
# accuracy than fully-connected networks on image data by exploiting
# local spatial structure. Confusion is highest among animal classes
# (cat/dog, deer/horse) which share visual features.
