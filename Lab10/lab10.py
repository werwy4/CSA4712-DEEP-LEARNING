# Experiment No: 10
# Title: Singular Value Decomposition for Data Compression
# Aim: Apply SVD and reconstruct images using limited singular values
#      to demonstrate data compression for a digital archive.
# Libraries: NumPy, Scikit-learn, Matplotlib

# PROGRAM

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits

np.random.seed(42)

# ── 1. Load image – use sklearn digits (8×8), upsample to 32×32 ───
digits     = load_digits()
# Pick digit '5' for a richer visual
idx        = np.where(digits.target == 5)[0][0]
img_small  = digits.images[idx]              # 8×8

# Upsample using repeat to get a larger image (64×64)
from numpy import kron
img = kron(img_small, np.ones((8, 8)))       # 64×64

print(f"Image shape: {img.shape}")
print(f"Min pixel: {img.min():.2f}, Max pixel: {img.max():.2f}")

# ── 2. Compute SVD ─────────────────────────────────────────────────
U, S, Vt = np.linalg.svd(img, full_matrices=False)
print(f"\nSVD shapes: U={U.shape}, S={S.shape}, Vt={Vt.shape}")
print(f"Top 10 singular values: {np.round(S[:10], 2)}")

total_energy = (S ** 2).sum()
print(f"\nEnergy captured by top k singular values:")
for k in [1, 5, 10, 20, 32, 64]:
    energy = (S[:k] ** 2).sum() / total_energy * 100
    print(f"  k={k:3d}: {energy:.2f}%")

# ── 3. Reconstruct with k singular values ─────────────────────────
k_values = [1, 3, 5, 10, 20, 64]

def reconstruct(U, S, Vt, k):
    return U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]

# ── 4. Reconstruction error and compression ratio ─────────────────
print(f"\n{'k':>5} {'Error (Fro)':>14} {'Compression':>14} {'Energy%':>10}")
print("-" * 48)
orig_size = img.shape[0] * img.shape[1]
for k in k_values:
    rec   = reconstruct(U, S, Vt, k)
    error = np.linalg.norm(img - rec, 'fro')
    # Compressed size = k*(m + n + 1) floats
    comp_size  = k * (img.shape[0] + img.shape[1] + 1)
    comp_ratio = orig_size / comp_size
    energy     = (S[:k] ** 2).sum() / total_energy * 100
    print(f"{k:>5} {error:>14.4f} {comp_ratio:>14.2f}x {energy:>10.2f}%")

# ── 5. Visualise original and reconstructed images ─────────────────
n_plots = len(k_values) + 1
fig, axes = plt.subplots(1, n_plots, figsize=(3 * n_plots, 3.5))

axes[0].imshow(img, cmap='gray')
axes[0].set_title('Original')
axes[0].axis('off')

for ax, k in zip(axes[1:], k_values):
    rec = reconstruct(U, S, Vt, k)
    err = np.linalg.norm(img - rec, 'fro')
    ax.imshow(rec, cmap='gray')
    ax.set_title(f'k={k}\nErr={err:.1f}')
    ax.axis('off')

plt.suptitle('SVD Image Reconstruction with Different k Values', fontsize=12)
plt.tight_layout()
plt.savefig('lab10_svd.png', dpi=100)
plt.show()

# ── Singular value spectrum ────────────────────────────────────────
plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(S, 'b-o', markersize=4)
plt.xlabel('Index')
plt.ylabel('Singular Value')
plt.title('Singular Value Spectrum')
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
cum_energy = np.cumsum(S ** 2) / total_energy * 100
plt.plot(cum_energy, 'r-')
plt.axhline(95, color='gray', linestyle='--', label='95% energy')
plt.xlabel('Number of Singular Values (k)')
plt.ylabel('Cumulative Energy (%)')
plt.title('Cumulative Energy vs k')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lab10_spectrum.png', dpi=100)
plt.show()

# EXPECTED OUTPUT
# Singular value spectrum decreasing rapidly. Reconstructions improve as k
# increases. k=5 already captures most visible structure. Full k gives
# lossless reconstruction. Compression ratio decreases as k increases.

# RESULT
# SVD decomposes an image into orthogonal components ranked by importance.
# Keeping only the top k singular values achieves lossy compression while
# preserving the most significant visual features. This is the mathematical
# basis of techniques like PCA and latent-factor models.
