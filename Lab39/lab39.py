# Experiment No: 39
# Title: Seq2Seq Architecture for Machine Translation
# Aim: Build an encoder-decoder (Seq2Seq) model for translating short
#      English phrases to French using character-level sequence mapping.
# Libraries: TensorFlow, Keras, NumPy, Matplotlib

# PROGRAM

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

np.random.seed(42)
tf.random.set_seed(42)

# ── 1. Small English→French phrase dictionary ─────────────────────
PAIRS = [
    ("hello",          "bonjour"),
    ("goodbye",        "au revoir"),
    ("thank you",      "merci"),
    ("yes",            "oui"),
    ("no",             "non"),
    ("please",         "sil vous plait"),
    ("good morning",   "bonjour"),
    ("good night",     "bonne nuit"),
    ("how are you",    "comment allez vous"),
    ("i love you",     "je t aime"),
    ("water",          "eau"),
    ("food",           "nourriture"),
    ("cat",            "chat"),
    ("dog",            "chien"),
    ("house",          "maison"),
    ("book",           "livre"),
    ("school",         "ecole"),
    ("friend",         "ami"),
    ("beautiful",      "beau"),
    ("happy",          "heureux"),
]

# Add decoder start/end tokens
START_TOK = '\t'
END_TOK   = '\n'
pairs_dec = [(inp, START_TOK + out + END_TOK) for inp, out in PAIRS]

# ── 2. Build character-level vocabularies ─────────────────────────
input_chars  = sorted(set(''.join(inp for inp, _ in pairs_dec)))
target_chars = sorted(set(''.join(out for _, out in pairs_dec)))

input_char2idx  = {c: i for i, c in enumerate(input_chars)}
target_char2idx = {c: i for i, c in enumerate(target_chars)}
input_idx2char  = {i: c for c, i in input_char2idx.items()}
target_idx2char = {i: c for c, i in target_char2idx.items()}

num_enc = len(input_chars)
num_dec = len(target_chars)
max_enc = max(len(inp) for inp, _ in pairs_dec)
max_dec = max(len(out) for _, out in pairs_dec)

print(f"Encoder vocab: {num_enc}, Decoder vocab: {num_dec}")
print(f"Max encoder len: {max_enc}, Max decoder len: {max_dec}")

# ── 3. One-hot encode data ────────────────────────────────────────
# Augment dataset by repeating
pairs_aug = pairs_dec * 50
N = len(pairs_aug)

enc_input  = np.zeros((N, max_enc, num_enc), dtype='float32')
dec_input  = np.zeros((N, max_dec, num_dec), dtype='float32')
dec_target = np.zeros((N, max_dec, num_dec), dtype='float32')

for i, (inp, out) in enumerate(pairs_aug):
    for t, c in enumerate(inp):
        enc_input[i, t, input_char2idx[c]] = 1.0
    for t, c in enumerate(out):
        dec_input[i, t, target_char2idx[c]] = 1.0
        if t > 0:
            dec_target[i, t-1, target_char2idx[c]] = 1.0

print(f"Encoder input shape: {enc_input.shape}")
print(f"Decoder input shape: {dec_input.shape}")

# ── 4. Build Seq2Seq model (training mode) ────────────────────────
LATENT_DIM = 128

# Encoder
enc_in  = keras.Input(shape=(None, num_enc))
enc_out, state_h, state_c = keras.layers.LSTM(LATENT_DIM,
                                               return_state=True)(enc_in)
enc_states = [state_h, state_c]

# Decoder
dec_in  = keras.Input(shape=(None, num_dec))
dec_lstm = keras.layers.LSTM(LATENT_DIM, return_sequences=True, return_state=True)
dec_out, _, _ = dec_lstm(dec_in, initial_state=enc_states)
dec_dense = keras.layers.Dense(num_dec, activation='softmax')
dec_preds  = dec_dense(dec_out)

model = keras.Model([enc_in, dec_in], dec_preds)
model.compile(optimizer='rmsprop', loss='categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

print("\nTraining Seq2Seq model...")
history = model.fit([enc_input, dec_input], dec_target,
                    validation_split=0.1,
                    epochs=80, batch_size=32, verbose=1)

# ── 5. Inference models ────────────────────────────────────────────
# Encoder inference
enc_model = keras.Model(enc_in, enc_states)

# Decoder inference
dec_state_h_in = keras.Input(shape=(LATENT_DIM,))
dec_state_c_in = keras.Input(shape=(LATENT_DIM,))
dec_in_inf     = keras.Input(shape=(None, num_dec))

dec_out_inf, out_h, out_c = dec_lstm(dec_in_inf,
                                     initial_state=[dec_state_h_in, dec_state_c_in])
dec_preds_inf = dec_dense(dec_out_inf)
dec_model = keras.Model([dec_in_inf, dec_state_h_in, dec_state_c_in],
                        [dec_preds_inf, out_h, out_c])

def translate(input_text):
    """Greedy decode inference."""
    seq = np.zeros((1, max_enc, num_enc))
    for t, c in enumerate(input_text[:max_enc]):
        if c in input_char2idx:
            seq[0, t, input_char2idx[c]] = 1.0

    states = enc_model.predict(seq, verbose=0)
    target_seq = np.zeros((1, 1, num_dec))
    target_seq[0, 0, target_char2idx[START_TOK]] = 1.0

    decoded = ''
    stop    = False
    while not stop:
        preds, h, c = dec_model.predict([target_seq] + states, verbose=0)
        idx = np.argmax(preds[0, -1, :])
        char = target_idx2char[idx]
        if char == END_TOK or len(decoded) > max_dec:
            stop = True
        else:
            decoded += char
        target_seq = np.zeros((1, 1, num_dec))
        target_seq[0, 0, idx] = 1.0
        states = [h, c]
    return decoded

# ── 6. Test translations ────────────────────────────────────────────
print("\n── Translation Results ──")
print(f"{'English':20s} {'True French':20s} {'Predicted French':20s}")
print("-" * 65)
for eng, fre in PAIRS[:10]:
    pred = translate(eng)
    print(f"  {eng:18s} {fre:20s} {pred:20s}")

# ── 7. Training plot ───────────────────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
axes[0].plot(history.history['loss'],     label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Val Loss', linestyle='--')
axes[0].set_title('Seq2Seq Training Loss'); axes[0].set_xlabel('Epoch')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(history.history['accuracy'],     label='Train Acc')
axes[1].plot(history.history['val_accuracy'], label='Val Acc', linestyle='--')
axes[1].set_title('Seq2Seq Training Accuracy'); axes[1].set_xlabel('Epoch')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.suptitle('Seq2Seq (Encoder-Decoder LSTM) – English to French', fontsize=12)
plt.tight_layout()
plt.savefig('lab39_seq2seq.png', dpi=100)
plt.show()

# EXPECTED OUTPUT
# After 80 epochs on the small phrase set, most translations are correct.
# "hello" → "bonjour", "thank you" → "merci", etc.

# RESULT
# The Seq2Seq architecture uses an LSTM encoder to compress the input
# sequence into a context vector (hidden states) and an LSTM decoder to
# generate the output sequence one character/token at a time.
# This forms the foundation of modern neural machine translation systems.
