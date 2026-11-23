# Experiment No: 37
# Title: GRU Architecture and Sentiment Classification
# Aim: Use a GRU-based model for binary sentiment classification on
#      IMDB movie reviews and compare speed vs LSTM.
# Libraries: TensorFlow, Keras, NumPy, Matplotlib

# PROGRAM

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
import time

np.random.seed(42)
tf.random.set_seed(42)

# ── 1. Load IMDB dataset ──────────────────────────────────────────
VOCAB_SIZE  = 10000
MAX_LEN     = 200
EMBED_DIM   = 64

print("Loading IMDB dataset...")
(X_train, y_train), (X_test, y_test) = keras.datasets.imdb.load_data(
    num_words=VOCAB_SIZE)

X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=MAX_LEN,
                                                      padding='post', truncating='post')
X_test  = keras.preprocessing.sequence.pad_sequences(X_test,  maxlen=MAX_LEN,
                                                      padding='post', truncating='post')
print(f"Train: {X_train.shape}, Test: {X_test.shape}")
print(f"Class distribution – train: {np.bincount(y_train)}")

# ── 2. Build GRU model ────────────────────────────────────────────
def build_gru():
    return keras.Sequential([
        keras.layers.Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAX_LEN),
        keras.layers.GRU(64, return_sequences=True),
        keras.layers.GRU(32),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation='sigmoid')
    ], name='GRU_Sentiment')

def build_lstm():
    return keras.Sequential([
        keras.layers.Embedding(VOCAB_SIZE, EMBED_DIM, input_length=MAX_LEN),
        keras.layers.LSTM(64, return_sequences=True),
        keras.layers.LSTM(32),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(1, activation='sigmoid')
    ], name='LSTM_Sentiment')

es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=3,
                                   restore_best_weights=True)

print("\n── Training GRU ──")
gru_model = build_gru()
gru_model.summary()
gru_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
t0 = time.time()
gru_hist = gru_model.fit(X_train, y_train, validation_split=0.1,
                         epochs=10, batch_size=128, callbacks=[es], verbose=1)
gru_time = time.time() - t0

print("\n── Training LSTM ──")
lstm_model = build_lstm()
lstm_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
t0 = time.time()
lstm_hist = lstm_model.fit(X_train, y_train, validation_split=0.1,
                           epochs=10, batch_size=128, callbacks=[es], verbose=1)
lstm_time = time.time() - t0

# ── 3. Evaluate ───────────────────────────────────────────────────
gru_loss,  gru_acc  = gru_model.evaluate(X_test,  y_test, verbose=0)
lstm_loss, lstm_acc = lstm_model.evaluate(X_test, y_test, verbose=0)

print(f"\n── Test Results ──")
print(f"  GRU  : Accuracy={gru_acc:.4f}, Loss={gru_loss:.4f}, Time={gru_time:.1f}s")
print(f"  LSTM : Accuracy={lstm_acc:.4f}, Loss={lstm_loss:.4f}, Time={lstm_time:.1f}s")

# ── 4. GRU gate explanation diagram ──────────────────────────────
fig_gate, ax_gate = plt.subplots(figsize=(12, 6))
ax_gate.axis('off')
ax_gate.set_title('GRU Gate Mechanism Overview', fontsize=13)

rows = [
    ('Reset Gate (r)',  'r = σ(Wr·[h_prev, x])',   'Controls how much past state to forget'),
    ('Update Gate (z)', 'z = σ(Wz·[h_prev, x])',   'Controls how much new info to add'),
    ('Candidate (h̃)',  'h̃ = tanh(W·[r·h_prev, x])','Proposed new hidden state'),
    ('New State (h)',   'h = (1−z)·h_prev + z·h̃',  'Blends old and new state'),
]
col_x = [0.05, 0.3, 0.6]
headers = ['Gate', 'Formula', 'Role']
for c, h in zip(col_x, headers):
    ax_gate.text(c, 0.9, h, fontsize=11, fontweight='bold',
                 transform=ax_gate.transAxes)
for i, (name, formula, role) in enumerate(rows):
    y = 0.75 - i * 0.17
    ax_gate.text(col_x[0], y, name,    fontsize=10, transform=ax_gate.transAxes)
    ax_gate.text(col_x[1], y, formula, fontsize=9,  transform=ax_gate.transAxes, family='monospace')
    ax_gate.text(col_x[2], y, role,    fontsize=9,  transform=ax_gate.transAxes)
    ax_gate.axhline(y - 0.07, xmin=0.02, xmax=0.98, color='gray', lw=0.5,
                    transform=ax_gate.transAxes)

ax_gate.text(0.5, 0.05,
             'GRU has 2 gates (fewer parameters than LSTM\'s 3 gates)',
             ha='center', fontsize=10, style='italic',
             transform=ax_gate.transAxes, color='steelblue')
plt.tight_layout()
plt.savefig('lab37_gru_gates.png', dpi=100)
plt.show()

# ── 5. Training & comparison plots ───────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(gru_hist.history['accuracy'],     label='GRU Train')
axes[0].plot(gru_hist.history['val_accuracy'], label='GRU Val', linestyle='--')
axes[0].plot(lstm_hist.history['accuracy'],    label='LSTM Train')
axes[0].plot(lstm_hist.history['val_accuracy'],label='LSTM Val', linestyle=':')
axes[0].set_title('Training Accuracy'); axes[0].set_xlabel('Epoch')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(gru_hist.history['loss'],     label='GRU Train')
axes[1].plot(gru_hist.history['val_loss'], label='GRU Val', linestyle='--')
axes[1].plot(lstm_hist.history['loss'],    label='LSTM Train')
axes[1].plot(lstm_hist.history['val_loss'],label='LSTM Val', linestyle=':')
axes[1].set_title('Training Loss'); axes[1].set_xlabel('Epoch')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

models    = ['GRU', 'LSTM']
accs      = [gru_acc, lstm_acc]
times_sec = [gru_time, lstm_time]
x = np.arange(2)
bars = axes[2].bar(x, accs, width=0.35, label='Test Accuracy', color=['steelblue','tomato'])
ax2 = axes[2].twinx()
ax2.plot(x, times_sec, 'go-', markersize=10, label='Training Time (s)')
axes[2].set_xticks(x); axes[2].set_xticklabels(models)
axes[2].set_ylabel('Test Accuracy'); ax2.set_ylabel('Training Time (s)')
axes[2].set_title('GRU vs LSTM – Accuracy & Speed')
axes[2].set_ylim(0.8, 1.0)
axes[2].legend(loc='upper left'); ax2.legend(loc='upper right')
axes[2].grid(True, alpha=0.3)

plt.suptitle('GRU Sentiment Classification – IMDB Reviews', fontsize=12)
plt.tight_layout()
plt.savefig('lab37_gru_sentiment.png', dpi=100)
plt.show()

# ── 6. Sample predictions ─────────────────────────────────────────
word_index = keras.datasets.imdb.get_word_index()
rev_index  = {v+3: k for k, v in word_index.items()}
rev_index.update({0: '<pad>', 1: '<sos>', 2: '<unk>', 3: '<unused>'})

print("\n── Sample Predictions ──")
for i in range(5):
    tokens  = X_test[i]
    text    = ' '.join([rev_index.get(tok, '?') for tok in tokens if tok > 0])[:120]
    pred    = gru_model.predict(X_test[i:i+1], verbose=0)[0][0]
    label   = 'Positive' if y_test[i] == 1 else 'Negative'
    pred_lbl = 'Positive' if pred > 0.5 else 'Negative'
    print(f"  Review: {text}...")
    print(f"  True={label}, Pred={pred_lbl} ({pred:.3f})\n")

# EXPECTED OUTPUT
# GRU achieves ~87-90% test accuracy, trains ~15-20% faster than LSTM.
# Both models converge within 5-7 epochs on IMDB.

# RESULT
# GRU with reset and update gates achieves comparable accuracy to LSTM
# with fewer parameters and faster training. The update gate controls
# how much past memory to retain, enabling long-range sentiment cues
# (like negation words far from the final verdict) to be captured.
