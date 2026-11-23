# Experiment No: 36
# Title: LSTM for Long-Term Dependency Modelling
# Aim: Demonstrate LSTM's ability to retain long-term information
#      by learning multi-period patterns in a synthetic financial series.
# Libraries: TensorFlow, Keras, NumPy, Pandas, Matplotlib, Scikit-learn

# PROGRAM

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

np.random.seed(42)
tf.random.set_seed(42)

# ── 1. Synthetic multi-period financial series ─────────────────────
N = 5000
t = np.linspace(0, 100, N)
long_cycle  = 20 * np.sin(2 * np.pi * t / 50)   # 50-step period
short_cycle = 5  * np.sin(2 * np.pi * t / 10)   # 10-step period
trend       = 0.1 * t
noise       = np.random.normal(0, 1, N)
series      = long_cycle + short_cycle + trend + noise

scaler = MinMaxScaler()
s_scaled = scaler.fit_transform(series.reshape(-1, 1)).ravel().astype(np.float32)

LOOK_BACK = 100   # long look-back to test long-term dependency

def make_seq(data, lb):
    X, y = [], []
    for i in range(len(data) - lb):
        X.append(data[i:i+lb])
        y.append(data[i+lb])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)

X, y = make_seq(s_scaled, LOOK_BACK)
split = int(0.8 * len(X))
X_tr, X_te = X[:split, :, None], X[split:, :, None]
y_tr, y_te = y[:split], y[split:]
print(f"Train: {X_tr.shape}, Test: {X_te.shape}")

# ── 2. Build LSTM model ────────────────────────────────────────────
def build_lstm():
    m = keras.Sequential([
        keras.layers.LSTM(64, return_sequences=True, input_shape=(LOOK_BACK, 1)),
        keras.layers.LSTM(32),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1)
    ], name='LSTM_LongTerm')
    m.compile(optimizer='adam', loss='mse')
    return m

def build_simple_rnn():
    m = keras.Sequential([
        keras.layers.SimpleRNN(64, return_sequences=True, input_shape=(LOOK_BACK, 1)),
        keras.layers.SimpleRNN(32),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(1)
    ], name='SimpleRNN_LongTerm')
    m.compile(optimizer='adam', loss='mse')
    return m

es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=8,
                                   restore_best_weights=True)

print("\n--- Training LSTM ---")
lstm_model = build_lstm()
lstm_model.summary()
lstm_hist = lstm_model.fit(X_tr, y_tr, validation_split=0.1,
                           epochs=40, batch_size=64,
                           callbacks=[es], verbose=1)

print("\n--- Training SimpleRNN ---")
rnn_model = build_simple_rnn()
rnn_hist = rnn_model.fit(X_tr, y_tr, validation_split=0.1,
                         epochs=40, batch_size=64,
                         callbacks=[es], verbose=1)

# ── 3. Evaluate ────────────────────────────────────────────────────
def evaluate(model, X_te, y_te, name):
    yp_scaled = model.predict(X_te, verbose=0).ravel()
    yp = scaler.inverse_transform(yp_scaled.reshape(-1,1)).ravel()
    yt = scaler.inverse_transform(y_te.reshape(-1,1)).ravel()
    mae  = mean_absolute_error(yt, yp)
    rmse = np.sqrt(mean_squared_error(yt, yp))
    print(f"  {name}: MAE={mae:.4f}, RMSE={rmse:.4f}")
    return yp, yt

print("\n── Evaluation ──")
lstm_pred, y_true = evaluate(lstm_model, X_te, y_te, "LSTM")
rnn_pred,  _      = evaluate(rnn_model,  X_te, y_te, "SimpleRNN")

# ── 4. Plots ───────────────────────────────────────────────────────
n_plot = 300
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

axes[0,0].plot(y_true[:n_plot], label='Actual', lw=1.5)
axes[0,0].plot(lstm_pred[:n_plot], label='LSTM', lw=1.5, linestyle='--')
axes[0,0].plot(rnn_pred[:n_plot],  label='RNN',  lw=1.5, linestyle=':')
axes[0,0].set_title('Actual vs Predicted (Test Set)')
axes[0,0].legend(); axes[0,0].grid(True, alpha=0.3)

axes[0,1].plot(lstm_hist.history['loss'],     label='LSTM Train')
axes[0,1].plot(lstm_hist.history['val_loss'], label='LSTM Val', linestyle='--')
axes[0,1].plot(rnn_hist.history['loss'],      label='RNN Train')
axes[0,1].plot(rnn_hist.history['val_loss'],  label='RNN Val', linestyle=':')
axes[0,1].set_title('Training Loss Comparison')
axes[0,1].set_xlabel('Epoch'); axes[0,1].set_ylabel('MSE')
axes[0,1].legend(); axes[0,1].grid(True, alpha=0.3)

res_lstm = y_true - lstm_pred
res_rnn  = y_true - rnn_pred
axes[1,0].hist(res_lstm, bins=40, alpha=0.7, label='LSTM residuals', color='steelblue')
axes[1,0].hist(res_rnn,  bins=40, alpha=0.7, label='RNN residuals',  color='tomato')
axes[1,0].set_title('Residual Distributions')
axes[1,0].legend(); axes[1,0].grid(True, alpha=0.3)

axes[1,1].scatter(y_true[:500], lstm_pred[:500], alpha=0.3, s=5, label='LSTM', color='steelblue')
axes[1,1].scatter(y_true[:500], rnn_pred[:500],  alpha=0.3, s=5, label='RNN',  color='tomato')
lims = [y_true.min(), y_true.max()]
axes[1,1].plot(lims, lims, 'k--', label='Perfect prediction')
axes[1,1].set_xlabel('Actual'); axes[1,1].set_ylabel('Predicted')
axes[1,1].set_title('Scatter: Actual vs Predicted')
axes[1,1].legend(); axes[1,1].grid(True, alpha=0.3)

plt.suptitle('LSTM vs SimpleRNN – Long-Term Dependency (look_back=100)', fontsize=12)
plt.tight_layout()
plt.savefig('lab36_lstm_vs_rnn.png', dpi=100)
plt.show()

# EXPECTED OUTPUT
# LSTM achieves lower MAE and RMSE than SimpleRNN on 100-step look-back.
# LSTM loss curve converges more smoothly; residuals are tighter.

# RESULT
# LSTM's gating mechanisms (input, forget, output gates) allow it to
# selectively retain or discard information over long sequences.
# It significantly outperforms SimpleRNN on tasks with long-range
# temporal dependencies, demonstrating the practical advantage of LSTMs.
