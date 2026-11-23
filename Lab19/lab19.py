# Experiment No: 19
# Title: Multilayer Perceptron Implementation
# Aim: Build, train and evaluate an MLP classifier in Keras for
#      telecom customer churn prediction.
# Libraries: TensorFlow, Keras, NumPy, Matplotlib, Scikit-learn, Pandas

# PROGRAM

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, ConfusionMatrixDisplay)

np.random.seed(42)
tf.random.set_seed(42)

# ── 1. Generate synthetic customer-churn dataset ───────────────────
n = 1000
tenure         = np.random.randint(1, 72, n)
monthly_charge = np.random.uniform(20, 120, n)
contract_type  = np.random.choice([0, 1, 2], n)   # 0=month, 1=1yr, 2=2yr
complaints     = np.random.randint(0, 10, n)
data_usage     = np.random.uniform(0, 50, n)

# Churn probability: higher complaints, lower tenure, month-to-month → more churn
churn_prob = (1 / (1 + np.exp(-(
    -0.05 * tenure
    + 0.02 * monthly_charge
    - 0.5  * contract_type
    + 0.3  * complaints
    - 0.01 * data_usage
    - 1.0
))))
churn = np.random.binomial(1, churn_prob, n)

df = pd.DataFrame({
    'Tenure': tenure, 'MonthlyCharge': monthly_charge,
    'ContractType': contract_type, 'Complaints': complaints,
    'DataUsage': data_usage, 'Churn': churn
})
print(f"Dataset shape: {df.shape}")
print(f"Churn rate: {churn.mean()*100:.1f}%")

# ── 2. Pre-process ─────────────────────────────────────────────────
X = df.drop('Churn', axis=1).values.astype(float)
y = df['Churn'].values

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
X_train, X_val, y_train, y_val   = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

scaler  = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val   = scaler.transform(X_val)
X_test  = scaler.transform(X_test)

print(f"Train: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")

# ── 3. Build MLP model ─────────────────────────────────────────────
def build_mlp(hidden_units=(64, 32)):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(5,)))
    for units in hidden_units:
        model.add(keras.layers.Dense(units, activation='relu'))
    model.add(keras.layers.Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model

model = build_mlp((64, 32))
model.summary()

# ── 4. Train ───────────────────────────────────────────────────────
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=60, batch_size=32, verbose=1
)

# ── 5. Evaluate ────────────────────────────────────────────────────
y_pred_prob = model.predict(X_test, verbose=0).ravel()
y_pred      = (y_pred_prob > 0.5).astype(int)

acc  = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)
rec  = recall_score(y_test, y_pred, zero_division=0)
f1   = f1_score(y_test, y_pred, zero_division=0)

print("\n── Test Set Metrics ──")
print(f"  Accuracy  : {acc:.4f}")
print(f"  Precision : {prec:.4f}")
print(f"  Recall    : {rec:.4f}")
print(f"  F1-Score  : {f1:.4f}")

# ── 6. Plots ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

# Training curves
axes[0].plot(history.history['loss'],     label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Val Loss')
axes[0].set_title('Training vs Validation Loss')
axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('BCE Loss')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['accuracy'],     label='Train Acc')
axes[1].plot(history.history['val_accuracy'], label='Val Acc')
axes[1].set_title('Training vs Validation Accuracy')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

# Confusion matrix
cm  = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=['No Churn', 'Churn'])
disp.plot(ax=axes[2], colorbar=False, cmap='Blues')
axes[2].set_title('Confusion Matrix')

plt.suptitle('MLP for Customer Churn Prediction', fontsize=13)
plt.tight_layout()
plt.savefig('lab19_mlp.png', dpi=100)
plt.show()

# ── 7. Effect of hidden layer width ───────────────────────────────
print("\n── Effect of Hidden Layer Width ──")
print(f"{'Architecture':>22} {'Val Acc':>10} {'Test Acc':>10} {'Params':>10}")
print("-" * 55)

for arch in [(16, 8), (64, 32), (128, 64), (256, 128)]:
    m = build_mlp(arch)
    m.fit(X_train, y_train, validation_data=(X_val, y_val),
          epochs=40, batch_size=32, verbose=0)
    val_acc  = m.evaluate(X_val,  y_val,  verbose=0)[1]
    test_acc = m.evaluate(X_test, y_test, verbose=0)[1]
    params   = m.count_params()
    print(f"{str(arch):>22} {val_acc:>10.4f} {test_acc:>10.4f} {params:>10,}")

# EXPECTED OUTPUT
# Training curves showing decreasing loss and increasing accuracy.
# Test accuracy ~75-85%, confusion matrix showing TP/TN/FP/FN.
# Wider hidden layers generally give better accuracy up to a point.

# RESULT
# An MLP with two hidden layers learns non-linear relationships in
# customer churn data. Binary cross-entropy with sigmoid output produces
# probability estimates. Precision/recall trade-off is important in churn
# prediction where false negatives (missed churners) are costly.
