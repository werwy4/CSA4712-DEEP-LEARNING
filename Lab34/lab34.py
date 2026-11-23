# Experiment No: 34
# Title: Unfolded RNN Architecture Visualization
# Aim: Visualise an RNN unfolded across time steps and trace hidden-state values.
# Libraries: TensorFlow, Keras, NetworkX, Matplotlib, NumPy

# PROGRAM

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow as tf
from tensorflow import keras

np.random.seed(42)
tf.random.set_seed(42)

# ── 1. Build a simple SimpleRNN ───────────────────────────────────
T         = 5    # time steps
INPUT_DIM = 2
HIDDEN    = 3

model = keras.Sequential([
    keras.layers.SimpleRNN(HIDDEN, return_sequences=True,
                           input_shape=(T, INPUT_DIM),
                           name='rnn')
])
model.build()

# ── 2. Extract weights ────────────────────────────────────────────
rnn_layer = model.get_layer('rnn')
W_x, W_h, b = rnn_layer.get_weights()   # input, recurrent, bias
print(f"W_x shape: {W_x.shape}")    # (input_dim, hidden)
print(f"W_h shape: {W_h.shape}")    # (hidden, hidden)
print(f"b shape:   {b.shape}")      # (hidden,)

# ── 3. Manual forward pass ────────────────────────────────────────
# Input sequence
X_seq = np.random.randn(T, INPUT_DIM)
print(f"\nInput sequence shape: {X_seq.shape}")

h = np.zeros(HIDDEN)
h_states = [h.copy()]
print("\nManual RNN Forward Pass:")
print(f"{'t':>3} {'h_state':>45}")
for t in range(T):
    h = np.tanh(X_seq[t] @ W_x + h @ W_h + b)
    h_states.append(h.copy())
    print(f"{t+1:>3} {np.round(h, 4)}")

# ── 4. Verify against Keras ───────────────────────────────────────
X_keras = X_seq[np.newaxis, :, :]   # (1, T, input_dim)
y_keras = model.predict(X_keras, verbose=0)[0]   # (T, hidden)
print("\nKeras RNN output (all timesteps):")
print(np.round(y_keras, 4))
print("\nManual h_states[1:]:")
print(np.round(np.array(h_states[1:]), 4))
match = np.allclose(y_keras, np.array(h_states[1:]), atol=1e-5)
print(f"\nManual == Keras: {match}")

# ── 5. Visualise hidden-state values ──────────────────────────────
h_arr = np.array(h_states[1:])   # (T, HIDDEN)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for i in range(HIDDEN):
    axes[0].plot(range(1, T+1), h_arr[:, i], '-o', label=f'h[{i}]')
axes[0].set_xlabel('Time Step')
axes[0].set_ylabel('Hidden State Value')
axes[0].set_title('Hidden State Trajectories Across Time Steps')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

im = axes[1].imshow(h_arr.T, cmap='coolwarm', aspect='auto')
axes[1].set_xlabel('Time Step'); axes[1].set_ylabel('Hidden Unit')
axes[1].set_xticks(range(T)); axes[1].set_xticklabels(range(1, T+1))
axes[1].set_yticks(range(HIDDEN)); axes[1].set_yticklabels([f'h{i}' for i in range(HIDDEN)])
axes[1].set_title('Hidden State Heatmap (time × units)')
for t in range(T):
    for u in range(HIDDEN):
        axes[1].text(t, u, f'{h_arr[t,u]:.2f}', ha='center', va='center', fontsize=9)
plt.colorbar(im, ax=axes[1])

plt.suptitle('RNN Hidden State Analysis', fontsize=12)
plt.tight_layout()
plt.savefig('lab34_hidden_states.png', dpi=100)
plt.show()

# ── 6. Draw unfolded RNN diagram ──────────────────────────────────
fig, ax = plt.subplots(figsize=(14, 6))
ax.set_xlim(-1, T+1); ax.set_ylim(-1, 3)
ax.axis('off')
ax.set_title('Unfolded RNN Architecture (T=5 time steps)', fontsize=13)

# Node positions
input_y  = 0.0
hidden_y = 1.5
out_y    = 2.8

colors_in  = '#AED6F1'
colors_hid = '#A9DFBF'
colors_out = '#FAD7A0'

for t in range(T):
    x = t + 0.5
    # Input node
    circ_in = plt.Circle((x, input_y), 0.3, color=colors_in, ec='black', zorder=3)
    ax.add_patch(circ_in)
    ax.text(x, input_y, f'x{t+1}', ha='center', va='center', fontsize=10, zorder=4)
    # Hidden node
    circ_hid = plt.Circle((x, hidden_y), 0.35, color=colors_hid, ec='black', zorder=3)
    ax.add_patch(circ_hid)
    ax.text(x, hidden_y, f'h{t+1}', ha='center', va='center', fontsize=10, zorder=4)
    # Output node
    circ_out = plt.Circle((x, out_y), 0.3, color=colors_out, ec='black', zorder=3)
    ax.add_patch(circ_out)
    ax.text(x, out_y, f'y{t+1}', ha='center', va='center', fontsize=10, zorder=4)

    # Arrows: input → hidden
    ax.annotate('', xy=(x, hidden_y-0.35), xytext=(x, input_y+0.3),
                arrowprops=dict(arrowstyle='->', color='royalblue', lw=1.5))
    # Arrows: hidden → output
    ax.annotate('', xy=(x, out_y-0.3), xytext=(x, hidden_y+0.35),
                arrowprops=dict(arrowstyle='->', color='forestgreen', lw=1.5))
    # Recurrent arrows: hidden[t] → hidden[t+1]
    if t < T - 1:
        ax.annotate('', xy=(x+0.65, hidden_y), xytext=(x+0.35, hidden_y),
                   arrowprops=dict(arrowstyle='->', color='tomato', lw=2.0))
        ax.text(x+0.5, hidden_y+0.5, 'W_h', ha='center', fontsize=8, color='tomato')

# Labels
ax.text(-0.5, input_y,  'Inputs', va='center', fontsize=10, style='italic')
ax.text(-0.5, hidden_y, 'Hidden', va='center', fontsize=10, style='italic')
ax.text(-0.5, out_y,    'Output', va='center', fontsize=10, style='italic')

legend_patches = [
    mpatches.Patch(color=colors_in,  label='Input nodes'),
    mpatches.Patch(color=colors_hid, label='Hidden state nodes'),
    mpatches.Patch(color=colors_out, label='Output nodes'),
    mpatches.Patch(color='tomato',   label='Recurrent connection (W_h)'),
]
ax.legend(handles=legend_patches, loc='lower right')
plt.tight_layout()
plt.savefig('lab34_unfolded.png', dpi=100)
plt.show()

# EXPECTED OUTPUT
# Manual forward pass matches Keras RNN output exactly.
# Hidden-state heatmap shows tanh-bounded values (-1 to +1).
# Unfolded diagram shows shared weights across all time steps.

# RESULT
# An RNN processes sequences by maintaining a hidden state that is updated
# at each time step. Unfolding across time visualises weight sharing and
# shows how information flows forward. The hidden state acts as a
# memory that accumulates information from all past inputs.
