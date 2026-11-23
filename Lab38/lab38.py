# Experiment No: 38
# Title: Bidirectional RNN for POS Tagging
# Aim: Build a Bidirectional LSTM for Part-of-Speech tagging and
#      compare with unidirectional LSTM.
# Libraries: TensorFlow, Keras, NumPy, Pandas, Matplotlib, Scikit-learn

# PROGRAM

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import LabelEncoder

np.random.seed(42)
tf.random.set_seed(42)

# ── 1. Build a small synthetic POS dataset ────────────────────────
# (Replicates a small English corpus with deterministic POS assignments)
SENTENCES = [
    [('The', 'DET'), ('dog', 'NOUN'), ('runs', 'VERB'), ('fast', 'ADV')],
    [('A', 'DET'), ('cat', 'NOUN'), ('sleeps', 'VERB'), ('quietly', 'ADV')],
    [('Birds', 'NOUN'), ('sing', 'VERB'), ('beautiful', 'ADJ'), ('songs', 'NOUN')],
    [('The', 'DET'), ('quick', 'ADJ'), ('fox', 'NOUN'), ('jumps', 'VERB')],
    [('She', 'PRON'), ('loves', 'VERB'), ('beautiful', 'ADJ'), ('music', 'NOUN')],
    [('He', 'PRON'), ('ate', 'VERB'), ('the', 'DET'), ('apple', 'NOUN')],
    [('Children', 'NOUN'), ('play', 'VERB'), ('happily', 'ADV'), ('outside', 'ADV')],
    [('The', 'DET'), ('teacher', 'NOUN'), ('explains', 'VERB'), ('clearly', 'ADV')],
]
# Duplicate and shuffle for more samples
import random
random.seed(42)
SENTENCES = SENTENCES * 50
random.shuffle(SENTENCES)

# ── 2. Build vocabulary and tag set ───────────────────────────────
all_words = sorted(set(w for s in SENTENCES for w, _ in s))
all_tags  = sorted(set(t for s in SENTENCES for _, t in s))
print(f"Vocabulary size: {len(all_words)}, Tags: {all_tags}")

word2idx = {w: i+2 for i, w in enumerate(all_words)}
word2idx['<PAD>'] = 0; word2idx['<UNK>'] = 1
tag2idx  = {t: i+1 for i, t in enumerate(all_tags)}
tag2idx['<PAD>'] = 0

idx2tag = {v: k for k, v in tag2idx.items()}

VOCAB_SIZE = len(word2idx)
TAG_SIZE   = len(tag2idx)
MAX_LEN    = 4   # all our sentences have length 4

def encode(sentence):
    xs = [word2idx.get(w, 1) for w, _ in sentence]
    ys = [tag2idx[t] for _, t in sentence]
    return xs, ys

X_all = np.array([encode(s)[0] for s in SENTENCES], dtype=np.int32)
Y_all = np.array([encode(s)[1] for s in SENTENCES], dtype=np.int32)

# One-hot encode tags for sequence output
Y_cat = tf.keras.utils.to_categorical(Y_all, num_classes=TAG_SIZE)

split = int(0.8 * len(X_all))
X_tr, X_te = X_all[:split], X_all[split:]
Y_tr, Y_te = Y_cat[:split], Y_cat[split:]
Y_tr_int    = Y_all[:split]
Y_te_int    = Y_all[split:]

print(f"Train: {X_tr.shape}, Test: {X_te.shape}")

# ── 3. Build Bidirectional LSTM model ─────────────────────────────
EMBED_DIM = 32

def build_bidir():
    inp = keras.Input(shape=(MAX_LEN,))
    emb = keras.layers.Embedding(VOCAB_SIZE, EMBED_DIM)(inp)
    out = keras.layers.Bidirectional(
              keras.layers.LSTM(64, return_sequences=True))(emb)
    out = keras.layers.TimeDistributed(keras.layers.Dense(TAG_SIZE,
                                                          activation='softmax'))(out)
    m = keras.Model(inp, out, name='BiLSTM_POS')
    m.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return m

def build_unidir():
    inp = keras.Input(shape=(MAX_LEN,))
    emb = keras.layers.Embedding(VOCAB_SIZE, EMBED_DIM)(inp)
    out = keras.layers.LSTM(64, return_sequences=True)(emb)
    out = keras.layers.TimeDistributed(keras.layers.Dense(TAG_SIZE,
                                                          activation='softmax'))(out)
    m = keras.Model(inp, out, name='LSTM_POS')
    m.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
    return m

print("\n── Training Bidirectional LSTM ──")
bi_model = build_bidir()
bi_model.summary()
bi_hist = bi_model.fit(X_tr, Y_tr, validation_split=0.1,
                       epochs=30, batch_size=16, verbose=1)

print("\n── Training Unidirectional LSTM ──")
uni_model = build_unidir()
uni_hist  = uni_model.fit(X_tr, Y_tr, validation_split=0.1,
                          epochs=30, batch_size=16, verbose=1)

# ── 4. Evaluate ────────────────────────────────────────────────────
def evaluate_seq(model, X, Y_int, name):
    preds_all = model.predict(X, verbose=0)   # (N, T, C)
    preds_flat = np.argmax(preds_all, axis=-1).ravel()
    true_flat  = Y_int.ravel()
    # Exclude PAD (class 0)
    mask = true_flat != 0
    acc  = accuracy_score(true_flat[mask], preds_flat[mask])
    print(f"\n── {name} ──")
    print(f"  Token Accuracy (excl. PAD): {acc:.4f}")
    print(classification_report(true_flat[mask], preds_flat[mask],
                                target_names=[idx2tag[i] for i in sorted(set(true_flat[mask]))],
                                zero_division=0))
    return acc

bi_acc  = evaluate_seq(bi_model,  X_te, Y_te_int, "Bidirectional LSTM")
uni_acc = evaluate_seq(uni_model, X_te, Y_te_int, "Unidirectional LSTM")

# ── 5. Sample tagging ──────────────────────────────────────────────
print("\n── Sample POS Tag Predictions ──")
for i in range(3):
    words  = [w for w, _ in SENTENCES[split + i]]
    true_t = [t for _, t in SENTENCES[split + i]]
    pred_i = bi_model.predict(X_te[i:i+1], verbose=0)[0]
    pred_t = [idx2tag[np.argmax(pred_i[j])] for j in range(MAX_LEN)]
    print(f"  Words:    {words}")
    print(f"  True:     {true_t}")
    print(f"  BiLSTM:   {pred_t}\n")

# ── 6. Plots ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(bi_hist.history['accuracy'],     label='BiLSTM Train')
axes[0].plot(bi_hist.history['val_accuracy'], label='BiLSTM Val', linestyle='--')
axes[0].plot(uni_hist.history['accuracy'],    label='UniLSTM Train')
axes[0].plot(uni_hist.history['val_accuracy'],label='UniLSTM Val', linestyle=':')
axes[0].set_title('POS Tagging Accuracy'); axes[0].set_xlabel('Epoch')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].bar(['BiLSTM', 'UniLSTM'], [bi_acc, uni_acc],
            color=['steelblue', 'tomato'])
axes[1].set_ylim(0.8, 1.0)
axes[1].set_title('Test Token Accuracy'); axes[1].set_ylabel('Accuracy')
axes[1].grid(True, alpha=0.3)

plt.suptitle('Bidirectional LSTM – POS Tagging', fontsize=12)
plt.tight_layout()
plt.savefig('lab38_bidir_pos.png', dpi=100)
plt.show()

# EXPECTED OUTPUT
# BiLSTM reaches ~95-100% accuracy on this small synthetic dataset.
# Bidirectional model converges faster than unidirectional.

# RESULT
# Bidirectional LSTM reads sequences in both forward and backward directions,
# providing each time step with context from both past and future tokens.
# This is crucial for NLP tasks like POS tagging where the meaning of a
# word depends on words that follow it, not just those before it.
