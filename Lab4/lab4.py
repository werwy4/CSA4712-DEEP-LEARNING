# Experiment No: 4
# Title: Entropy, Cross-Entropy and KL Divergence
# Aim: Compute entropy, cross-entropy and KL divergence for weather
#      forecast probability distributions and relate them to classification loss.
# Libraries: NumPy, SciPy, Matplotlib

# PROGRAM

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import xlogy

# ── Helper: safe log ───────────────────────────────────────────────
def safe_log(p):
    return np.where(p > 0, np.log(p), 0)

def entropy(p):
    """Shannon entropy H(P)"""
    return -np.sum(p * safe_log(p))

def cross_entropy(p, q):
    """Cross-entropy H(P, Q) = -Σ p * log(q)"""
    return -np.sum(p * safe_log(q))

def kl_divergence(p, q):
    """KL divergence KL(P || Q) = Σ p * log(p/q)"""
    return np.sum(xlogy(p, p / (q + 1e-12)))

# ── 1. Observed weather distribution ──────────────────────────────
# Classes: Sunny, Cloudy, Rainy
true_dist = np.array([0.50, 0.30, 0.20])   # observed over 30 days
labels    = ['Sunny', 'Cloudy', 'Rainy']

H = entropy(true_dist)
print("=" * 55)
print(f"Observed distribution  P: {true_dist}")
print(f"Entropy  H(P)            : {H:.4f} nats")

# ── 2. Two competing forecast models ──────────────────────────────
model_A = np.array([0.40, 0.35, 0.25])   # model A forecast
model_B = np.array([0.60, 0.20, 0.20])   # model B forecast

for name, q in [('Model A', model_A), ('Model B', model_B)]:
    ce  = cross_entropy(true_dist, q)
    kl  = kl_divergence(true_dist, q)
    print(f"\n{name}: Q = {q}")
    print(f"  Cross-Entropy H(P,Q)  = {ce:.4f}")
    print(f"  KL Divergence KL(P‖Q) = {kl:.4f}")
    print(f"  Verification: H(P) + KL = {H:.4f} + {kl:.4f} = {H+kl:.4f}  "
          f"[should equal H(P,Q)={ce:.4f}]")

# ── 3. KL divergence is asymmetric ────────────────────────────────
kl_pq = kl_divergence(true_dist, model_A)
kl_qp = kl_divergence(model_A, true_dist)
print("\n── KL Asymmetry ──")
print(f"KL(P ‖ Q_A) = {kl_pq:.4f}")
print(f"KL(Q_A ‖ P) = {kl_qp:.4f}  (different ⇒ asymmetric)")

# ── 4. Bar chart of distributions ─────────────────────────────────
x = np.arange(len(labels))
width = 0.25

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].bar(x - width, true_dist, width, label='True P', color='steelblue')
axes[0].bar(x,         model_A,   width, label='Model A', color='tomato')
axes[0].bar(x + width, model_B,   width, label='Model B', color='forestgreen')
axes[0].set_xticks(x); axes[0].set_xticklabels(labels)
axes[0].set_ylabel('Probability')
axes[0].set_title('Weather Forecast Distributions')
axes[0].legend()

# Summary bar: cross-entropy and KL for both models
ce_vals  = [cross_entropy(true_dist, model_A), cross_entropy(true_dist, model_B)]
kl_vals  = [kl_divergence(true_dist, model_A), kl_divergence(true_dist, model_B)]
x2 = np.array([0, 1])
axes[1].bar(x2 - 0.2, ce_vals, 0.35, label='Cross-Entropy H(P,Q)', color='coral')
axes[1].bar(x2 + 0.2, kl_vals, 0.35, label='KL Divergence KL(P‖Q)', color='teal')
axes[1].axhline(H, color='navy', linestyle='--', label=f'Entropy H(P)={H:.3f}')
axes[1].set_xticks(x2); axes[1].set_xticklabels(['Model A', 'Model B'])
axes[1].set_ylabel('Bits (nats)')
axes[1].set_title('Cross-Entropy and KL Divergence Comparison')
axes[1].legend()

plt.tight_layout()
plt.savefig('lab4_entropy.png', dpi=100)
plt.show()

# ── 5. Binary cross-entropy loss curve ────────────────────────────
p_pred = np.linspace(1e-6, 1 - 1e-6, 500)

bce_true1  = -np.log(p_pred)         # true label = 1
bce_true0  = -np.log(1 - p_pred)     # true label = 0

plt.figure(figsize=(8, 5))
plt.plot(p_pred, bce_true1, 'b-', lw=2, label='True label = 1: −log(p)')
plt.plot(p_pred, bce_true0, 'r-', lw=2, label='True label = 0: −log(1−p)')
plt.ylim(0, 6)
plt.xlabel('Predicted Probability p')
plt.ylabel('Binary Cross-Entropy Loss')
plt.title('Binary Cross-Entropy Loss vs Predicted Probability')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lab4_bce.png', dpi=100)
plt.show()

# EXPECTED OUTPUT
# H(P) ≈ 1.00 nats. Model A has lower cross-entropy than Model B.
# Cross-entropy = entropy + KL verified numerically.
# KL(P‖Q) ≠ KL(Q‖P) demonstrating asymmetry.
# Binary cross-entropy curve shows loss → ∞ for wrong-confident predictions.

# RESULT
# Entropy quantifies the inherent uncertainty of a distribution.
# Cross-entropy and KL divergence measure how much a predicted distribution
# deviates from the true one — forming the basis of classification loss functions.
