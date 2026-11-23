# Experiment No: 33
# Title: Sequential Dataset Preparation and RNN Prediction
# Aim: Convert a time series into supervised sequences and predict
#      next-step electricity consumption with SimpleRNN.
# Libraries: TensorFlow, Keras, NumPy, Pandas, Matplotlib

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

# ── 1. Generate synthetic hourly electricity consumption series ────
n_points = 8760   # 365 days × 24 hours
t        = np.arange(n_points)

# Trend + seasonality (daily + weekly) + noise
daily_cycle   = 15 * np.sin(2 * np.pi * t / 24)
weekly_cycle  = 10 * np.sin(2 * np.pi * t / (24 * 7))
trend         = 0.001 * t
noise         = np.random.normal(0, 2, n_points)
series        = 100 + trend + daily_cycle + weekly_cycle + noise

dates = pd.date_range(start='2023-01-01', periods=n_points, freq='h')
df    = pd.DataFrame({'datetime': dates, 'consumption_kWh': series})
print(f"Time series shape: {df.shape}")
print(df.head())
print(f"\nRange: [{series.min():.2f}, {series.max():.2f}] kWh")

# ── 2. Visualise raw series ────────────────────────────────────────
fig, axes = plt.subplots(2, 1, figsize=(14, 6))
axes[0].plot(dates[:168], series[:168], 'steelblue', lw=1)
axes[0].set_title('Hourly Electricity Consumption – First Week')
axes[0].set_xlabel('Date'); axes[0].set_ylabel('kWh')
axes[1].plot(dates[:720], series[:720], 'tomato', lw=0.8)
axes[1].set_title('Hourly Electricity Consumption – First Month')
axes[1].set_xlabel('Date'); axes[1].set_ylabel('kWh')
plt.tight_layout()
plt.savefig('lab33_series.png', dpi=100)
plt.show()

# ── 3. Scale and create sliding windows ───────────────────────────
scaler = MinMaxScaler(feature_range=(0, 1))
series_scaled = scaler.fit_transform(series.reshape(-1, 1)).ravel()

LOOK_BACK = 24   # 24 hours past

def create_sequences(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:i+look_back])
        y.append(data[i+look_back])
    return np.array(X), np.array(y)

X_seq, y_seq = create_sequences(series_scaled, LOOK_BACK)
print(f"\nSequences: X={X_seq.shape}, y={y_seq.shape}")

# ── 4. Chronological train/test split ─────────────────────────────
split = int(len(X_seq) * 0.8)
X_train, X_test = X_seq[:split], X_seq[split:]
y_train, y_test = y_seq[:split], y_seq[split:]

# Reshape for RNN: (samples, timesteps, features)
X_train = X_train.reshape(-1, LOOK_BACK, 1)
X_test  = X_test.reshape(-1, LOOK_BACK, 1)

print(f"Train: {X_train.shape}, Test: {X_test.shape}")

# ── 5. Build and train SimpleRNN ──────────────────────────────────
model = keras.Sequential([
    keras.layers.SimpleRNN(64, return_sequences=True, input_shape=(LOOK_BACK, 1)),
    keras.layers.SimpleRNN(32),
    keras.layers.Dense(16, activation='relu'),
    keras.layers.Dense(1)
], name='SimpleRNN_Electricity')

model.compile(optimizer='adam', loss='mse')
model.summary()

es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=5,
                                   restore_best_weights=True)
print("\nTraining SimpleRNN...")
history = model.fit(X_train, y_train,
                    validation_split=0.1,
                    epochs=30, batch_size=64,
                    callbacks=[es], verbose=1)

# ── 6. Predict and evaluate ────────────────────────────────────────
y_pred_scaled = model.predict(X_test, verbose=0).ravel()
y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()
y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).ravel()

mae  = mean_absolute_error(y_true, y_pred)
rmse = np.sqrt(mean_squared_error(y_true, y_pred))
mape = np.mean(np.abs((y_true - y_pred) / (y_true + 1e-6))) * 100

print(f"\n── Test Metrics ──")
print(f"  MAE  : {mae:.4f} kWh")
print(f"  RMSE : {rmse:.4f} kWh")
print(f"  MAPE : {mape:.2f}%")

# ── 7. Plot actual vs predicted ────────────────────────────────────
n_plot = 168   # 1 week
fig, axes = plt.subplots(2, 1, figsize=(14, 8))

axes[0].plot(y_true[:n_plot], label='Actual', color='steelblue', lw=1.5)
axes[0].plot(y_pred[:n_plot], label='Predicted', color='tomato', lw=1.5, linestyle='--')
axes[0].set_title('RNN Forecast vs Actual – First Test Week')
axes[0].set_xlabel('Hours'); axes[0].set_ylabel('kWh')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['loss'],     label='Train Loss')
axes[1].plot(history.history['val_loss'], label='Val Loss')
axes[1].set_title('Training Loss')
axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('MSE')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.suptitle(f'SimpleRNN Electricity Forecasting (MAE={mae:.2f}, RMSE={rmse:.2f})', fontsize=12)
plt.tight_layout()
plt.savefig('lab33_forecast.png', dpi=100)
plt.show()

# EXPECTED OUTPUT
# Training loss decreasing. Forecast follows actual consumption patterns.
# Daily cycles captured. MAE typically 2-4 kWh on this synthetic data.

# RESULT
# Sliding window transformation converts a univariate time series into
# a supervised learning problem. SimpleRNN learns temporal dependencies
# within the 24-hour look-back window to forecast the next hour's
# electricity consumption. Early stopping prevents overfitting.
