# Experiment No: 30
# Title: Imbalanced Image Classification
# Aim: Investigate class imbalance and compare strategies for improving
#      minority-class detection accuracy.
# Libraries: TensorFlow, Keras, Scikit-learn, Matplotlib

# PROGRAM

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (classification_report, confusion_matrix,
                             ConfusionMatrixDisplay, f1_score)

np.random.seed(42)
tf.random.set_seed(42)

# ── 1. Create imbalanced MNIST dataset ────────────────────────────
# Class 0 (normal): many samples; Class 1 (anomaly): few samples
(X_all, y_all), (X_test_full, y_test_full) = keras.datasets.mnist.load_data()

def make_imbalanced(X, y, majority_class=0, minority_class=1,
                    n_majority=5000, n_minority=250):
    idx_maj = np.where(y == majority_class)[0][:n_majority]
    idx_min = np.where(y == minority_class)[0][:n_minority]
    idx     = np.concatenate([idx_maj, idx_min])
    np.random.shuffle(idx)
    X_out = X[idx].astype('float32') / 255.0
    # Binary labels: 0 → class 0, 1 → minority
    y_out = (y[idx] == minority_class).astype(int)
    return X_out[..., np.newaxis], y_out

X_train, y_train = make_imbalanced(X_all, y_all)
X_test,  y_test  = make_imbalanced(X_test_full, y_test_full, n_majority=1000, n_minority=200)

print(f"Train – Class 0: {(y_train==0).sum()}, Class 1: {(y_train==1).sum()}")
print(f"Test  – Class 0: {(y_test==0).sum()},  Class 1: {(y_test==1).sum()}")
imbalance_ratio = (y_train==0).sum() / (y_train==1).sum()
print(f"Imbalance ratio: {imbalance_ratio:.1f}:1")

# ── 2. Plot class distribution ────────────────────────────────────
fig, ax = plt.subplots(figsize=(6, 4))
counts = [(y_train==0).sum(), (y_train==1).sum()]
ax.bar(['Class 0 (Normal)', 'Class 1 (Anomaly)'], counts,
       color=['steelblue', 'tomato'], edgecolor='k')
ax.set_title('Class Distribution (Imbalanced Training Set)')
ax.set_ylabel('Count')
for i, c in enumerate(counts):
    ax.text(i, c+50, str(c), ha='center')
plt.tight_layout()
plt.savefig('lab30_distribution.png', dpi=100)
plt.show()

# ── CNN builder ────────────────────────────────────────────────────
def build_cnn():
    return keras.Sequential([
        keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28,28,1)),
        keras.layers.MaxPooling2D(2,2),
        keras.layers.Conv2D(64, 3, activation='relu'),
        keras.layers.MaxPooling2D(2,2),
        keras.layers.Flatten(),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.Dense(1, activation='sigmoid')
    ])

EPOCHS = 15; BATCH = 64

def train_model(model, X_tr, y_tr, class_weight=None, label=''):
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    h = model.fit(X_tr, y_tr, epochs=EPOCHS, batch_size=BATCH,
                  validation_data=(X_test, y_test),
                  class_weight=class_weight, verbose=0)
    y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int).ravel()
    f1_min = f1_score(y_test, y_pred, pos_label=1, zero_division=0)
    acc    = (y_pred == y_test).mean()
    print(f"{label:30s}: Acc={acc:.3f}, F1(minority)={f1_min:.3f}")
    return h.history, y_pred, f1_min

# ── 3. Baseline ───────────────────────────────────────────────────
print("\n── Experiment Results ──")
h_base, pred_base, f1_base = train_model(build_cnn(), X_train, y_train, label='Baseline (no strategy)')

# ── 4. Class weights ──────────────────────────────────────────────
cw = {0: 1.0, 1: imbalance_ratio}
h_cw, pred_cw, f1_cw = train_model(build_cnn(), X_train, y_train,
                                     class_weight=cw, label='Class Weights')

# ── 5. Oversampled minority class (naive) ─────────────────────────
min_idx = np.where(y_train == 1)[0]
n_to_rep = (y_train==0).sum() - (y_train==1).sum()
rep_idx  = np.random.choice(min_idx, n_to_rep, replace=True)
X_over   = np.concatenate([X_train, X_train[rep_idx]])
y_over   = np.concatenate([y_train, y_train[rep_idx]])
shuf     = np.random.permutation(len(y_over))
h_over, pred_over, f1_over = train_model(build_cnn(), X_over[shuf], y_over[shuf],
                                          label='Oversampling (minority)')

# ── 6. Confusion matrices ─────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, preds, title in zip(axes,
                              [pred_base, pred_cw, pred_over],
                              ['Baseline', 'Class Weights', 'Oversampling']):
    cm   = confusion_matrix(y_test, preds)
    disp = ConfusionMatrixDisplay(cm, display_labels=['Normal','Anomaly'])
    disp.plot(ax=ax, cmap='Blues', colorbar=False)
    f1 = f1_score(y_test, preds, pos_label=1, zero_division=0)
    ax.set_title(f'{title}\nF1(minority)={f1:.3f}')

plt.suptitle('Imbalanced Classification: Strategy Comparison', fontsize=12)
plt.tight_layout()
plt.savefig('lab30_confusion.png', dpi=100)
plt.show()

# ── Summary ───────────────────────────────────────────────────────
print("\n── Summary ──")
print(f"{'Strategy':>25} {'F1 (minority)':>15}")
print("-" * 42)
for name, f1 in [('Baseline', f1_base), ('Class Weights', f1_cw),
                  ('Oversampling', f1_over)]:
    print(f"{name:>25} {f1:>15.4f}")

print("\nRecommendation:")
print("  Class weights: simple, effective, no data duplication.")
print("  Oversampling: effective, risk of overfitting to duplicated samples.")
print("  For severe imbalance (>100:1), use SMOTE or focal loss.")

# EXPECTED OUTPUT
# Baseline: high accuracy but very low F1 for minority class (predicts all as 0).
# Class weights: improved F1 for class 1 at slight accuracy cost.
# Oversampling: similar improvement in minority recall.

# RESULT
# Class imbalance causes models to ignore minority classes.
# Class weighting penalises minority misclassifications more during training.
# Both strategies improve minority-class F1-score compared to the baseline,
# demonstrating that accuracy alone is a misleading metric for imbalanced data.
