# Experiment No: 32
# Title: Model Evaluation and Error Analysis for CNNs
# Aim: Conduct structured error analysis using classification metrics
#      and visual inspection for Fashion-MNIST CNN.
# Libraries: TensorFlow, Keras, Scikit-learn, Pandas, Matplotlib

# PROGRAM

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay,
                             classification_report)

np.random.seed(42)
tf.random.set_seed(42)

CLASS_NAMES = ['T-shirt','Trouser','Pullover','Dress','Coat',
               'Sandal','Shirt','Sneaker','Bag','Boot']

# ── 1. Load Fashion-MNIST and train CNN ───────────────────────────
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train = X_train.astype('float32')[..., np.newaxis] / 255.0
X_test  = X_test.astype('float32')[..., np.newaxis]  / 255.0

model = keras.Sequential([
    keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28,28,1)),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64, 3, activation='relu'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dropout(0.3),
    keras.layers.Dense(10, activation='softmax')
])
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

print("Training CNN...")
model.fit(X_train, y_train, validation_split=0.1,
          epochs=12, batch_size=64, verbose=1)

# ── 2. Generate predictions ───────────────────────────────────────
y_pred_probs = model.predict(X_test, verbose=0)
y_pred       = np.argmax(y_pred_probs, axis=1)
y_true       = y_test

# ── 3. Classification metrics ─────────────────────────────────────
acc  = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, average='weighted', zero_division=0)
rec  = recall_score(y_true, y_pred, average='weighted', zero_division=0)
f1   = f1_score(y_true, y_pred, average='weighted', zero_division=0)

print(f"\n── Overall Test Metrics ──")
print(f"  Accuracy  : {acc:.4f}")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1-Score  : {f1:.4f}")

# Per-class metrics
print("\n── Per-Class Classification Report ──")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

# ── 4. Confusion matrix ────────────────────────────────────────────
cm = confusion_matrix(y_true, y_pred)
fig, ax = plt.subplots(figsize=(11, 9))
disp = ConfusionMatrixDisplay(cm, display_labels=CLASS_NAMES)
disp.plot(ax=ax, cmap='Blues', colorbar=True, values_format='d')
plt.title('Fashion-MNIST CNN – Confusion Matrix (Test Set)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.savefig('lab32_confusion.png', dpi=100)
plt.show()

# ── 5. Most confused class pairs ──────────────────────────────────
print("\n── Most Confused Class Pairs (off-diagonal) ──")
cm_copy = cm.copy()
np.fill_diagonal(cm_copy, 0)
top_confused = []
for _ in range(5):
    idx = np.unravel_index(cm_copy.argmax(), cm_copy.shape)
    true_cls  = CLASS_NAMES[idx[0]]
    pred_cls  = CLASS_NAMES[idx[1]]
    count     = cm_copy[idx]
    top_confused.append((true_cls, pred_cls, count))
    cm_copy[idx] = 0
    print(f"  True: {true_cls:12s} Predicted as: {pred_cls:12s} Count: {count}")

# ── 6. Correctly classified samples ───────────────────────────────
correct_idx = np.where(y_pred == y_true)[0][:9]
fig, axes = plt.subplots(3, 3, figsize=(9, 9))
fig.suptitle('Correctly Classified Examples', fontsize=12)
for ax, idx in zip(axes.flatten(), correct_idx):
    ax.imshow(X_test[idx].squeeze(), cmap='gray')
    conf = y_pred_probs[idx, y_pred[idx]]
    ax.set_title(f"{CLASS_NAMES[y_pred[idx]]}\n({conf:.2f})", fontsize=8)
    ax.axis('off')
plt.tight_layout()
plt.savefig('lab32_correct.png', dpi=100)
plt.show()

# ── 7. Misclassified samples ──────────────────────────────────────
wrong_idx = np.where(y_pred != y_true)[0][:9]
fig, axes = plt.subplots(3, 3, figsize=(9, 9))
fig.suptitle('Misclassified Examples', fontsize=12)
for ax, idx in zip(axes.flatten(), wrong_idx):
    ax.imshow(X_test[idx].squeeze(), cmap='gray')
    conf = y_pred_probs[idx, y_pred[idx]]
    ax.set_title(f"True: {CLASS_NAMES[y_true[idx]]}\n"
                 f"Pred: {CLASS_NAMES[y_pred[idx]]} ({conf:.2f})", fontsize=7)
    ax.axis('off')
plt.tight_layout()
plt.savefig('lab32_misclassified.png', dpi=100)
plt.show()

# ── 8. Per-class metrics table ────────────────────────────────────
per_class = []
for i, name in enumerate(CLASS_NAMES):
    p = precision_score(y_true == i, y_pred == i, zero_division=0)
    r = recall_score(y_true == i, y_pred == i, zero_division=0)
    f = f1_score(y_true == i, y_pred == i, zero_division=0)
    n = (y_true == i).sum()
    per_class.append({'Class': name, 'Support': n,
                      'Precision': round(p,4), 'Recall': round(r,4), 'F1': round(f,4)})

df_per = pd.DataFrame(per_class)
print("\n── Per-Class Metrics Table ──")
print(df_per.to_string(index=False))

# EXPECTED OUTPUT
# Overall accuracy ~88-91%. Shirt and Coat are most confused.
# T-shirt/Shirt confusion is highest off-diagonal entry.
# Trouser and Bag have highest precision/recall (distinctive shapes).

# RESULT
# Structured error analysis reveals which classes are most challenging.
# Visually similar categories (Shirt/T-shirt, Coat/Pullover) contribute
# most misclassifications. This guides targeted improvements such as
# class-specific augmentation or hierarchical classification.
