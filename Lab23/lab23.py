# Experiment No: 23
# Title: Activation Function Behaviour and Gradient Analysis
# Aim: Compare common activation functions and their derivatives for
#      neural network layer selection.
# Libraries: NumPy, TensorFlow, Matplotlib

# PROGRAM

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# ── 1. Define activation functions ────────────────────────────────
def sigmoid(z):         return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
def sigmoid_deriv(z):   a = sigmoid(z); return a * (1 - a)
def tanh_fn(z):         return np.tanh(z)
def tanh_deriv(z):      return 1 - np.tanh(z)**2
def relu(z):            return np.maximum(0, z)
def relu_deriv(z):      return (z > 0).astype(float)
def leaky_relu(z, a=0.01): return np.where(z > 0, z, a * z)
def leaky_relu_deriv(z, a=0.01): return np.where(z > 0, 1.0, a)
def softmax(z):
    e = np.exp(z - np.max(z))
    return e / e.sum()

# ── 2. Plot all activation functions ──────────────────────────────
z = np.linspace(-5, 5, 500)
fig, axes = plt.subplots(2, 3, figsize=(15, 10))

act_data = [
    ('Sigmoid',    sigmoid(z),    sigmoid_deriv(z)),
    ('Tanh',       tanh_fn(z),    tanh_deriv(z)),
    ('ReLU',       relu(z),       relu_deriv(z)),
    ('Leaky ReLU', leaky_relu(z), leaky_relu_deriv(z)),
]

colors = ['royalblue', 'tomato', 'forestgreen', 'darkorange']
for ax, (name, act, deriv), col in zip(axes.flatten(), act_data, colors):
    ax.plot(z, act,   color=col, lw=2.5, label='f(z)')
    ax.plot(z, deriv, color=col, lw=2,   linestyle='--', alpha=0.7, label="f'(z)")
    ax.axhline(0, color='black', lw=0.5)
    ax.axvline(0, color='black', lw=0.5)
    ax.set_title(name, fontsize=13)
    ax.set_xlabel('z');  ax.set_ylabel('Value')
    ax.legend(); ax.grid(True, alpha=0.3)
    ax.set_ylim(-1.5, 2.5)

# Softmax (vector input)
z_vec = np.array([-2, -1, 0, 1, 2, 3])
sm    = softmax(z_vec)
ax_sm = axes[1, 2]
ax_sm.bar(range(len(z_vec)), sm, color='purple', edgecolor='k')
ax_sm.set_title('Softmax (class probabilities)')
ax_sm.set_xlabel('Class index'); ax_sm.set_ylabel('Probability')
ax_sm.set_xticks(range(len(z_vec)))
ax_sm.set_xticklabels([f'z={x}' for x in z_vec])
ax_sm.text(0.5, 0.95, f'Sum = {sm.sum():.4f}', transform=ax_sm.transAxes,
           ha='center', va='top', fontsize=10, color='red')
ax_sm.grid(True, alpha=0.3)

plt.suptitle('Activation Functions and Their Derivatives', fontsize=14)
plt.tight_layout()
plt.savefig('lab23_activations.png', dpi=100)
plt.show()

# ── 3. Gradient analysis for large inputs ─────────────────────────
large_vals = np.array([-10, -5, -2, 0, 2, 5, 10])
print("Gradient values at different z values:")
print(f"{'z':>6} {'σ\'(z)':>12} {'tanh\'(z)':>12} {'ReLU\'(z)':>12} {'LReLU\'(z)':>13}")
print("-" * 58)
for z_val in large_vals:
    print(f"{z_val:>6} {sigmoid_deriv(z_val):>12.6f} {tanh_deriv(z_val):>12.6f} "
          f"{relu_deriv(z_val):>12.6f} {leaky_relu_deriv(z_val):>13.6f}")

# ── 4. Vanishing gradient illustration ────────────────────────────
n_layers = 10
z_input  = 3.0   # large positive input
grad_sigmoid = [sigmoid_deriv(z_input)] * n_layers
grad_relu    = [relu_deriv(z_input)]    * n_layers
grad_tanh    = [tanh_deriv(z_input)]    * n_layers

cum_sig  = np.cumprod(grad_sigmoid)
cum_relu = np.cumprod(grad_relu)
cum_tanh = np.cumprod(grad_tanh)

plt.figure(figsize=(9, 5))
plt.semilogy(range(1, n_layers+1), cum_sig,  'b-o', label='Sigmoid')
plt.semilogy(range(1, n_layers+1), cum_tanh, 'r-s', label='Tanh')
plt.semilogy(range(1, n_layers+1), cum_relu, 'g-^', label='ReLU')
plt.xlabel('Layer depth')
plt.ylabel('Cumulative gradient magnitude (log scale)')
plt.title(f'Vanishing Gradient: Cumulative gradient over {n_layers} layers (z={z_input})')
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lab23_vanishing.png', dpi=100)
plt.show()

# ── 5. Summary ────────────────────────────────────────────────────
print("\n── Activation Function Summary ──")
summary = [
    ("Sigmoid",    "Output (binary)", "Vanishing gradient for |z|>3; not zero-centred"),
    ("Tanh",       "Hidden layers",   "Zero-centred; still vanishes for large |z|"),
    ("ReLU",       "Hidden layers",   "Fast; may cause 'dying ReLU' for z<0"),
    ("Leaky ReLU", "Hidden layers",   "Avoids dying ReLU; small gradient for z<0"),
    ("Softmax",    "Output (multi)",  "Converts logits to class probabilities"),
]
for name, use, note in summary:
    print(f"  {name:>12}: Use for {use}. Note: {note}")

# EXPECTED OUTPUT
# Plots of all 5 activation functions with derivatives. Gradient table
# showing sigmoid/tanh derivatives vanishing for large z. ReLU derivative
# is constant 1 for z>0 (no vanishing). Softmax probabilities sum to 1.

# RESULT
# ReLU and Leaky ReLU are preferred for hidden layers because they avoid
# the vanishing gradient problem. Sigmoid is used for binary output;
# Softmax for multiclass output. Tanh is zero-centred but still suffers
# from gradient saturation at extreme values.
