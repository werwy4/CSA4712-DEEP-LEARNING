# Experiment No: 2
# Title: Covariance, Eigenvalues and Eigenvectors Analysis
# Aim: Compute covariance/correlation matrices and extract eigenvalues/eigenvectors
#      to find principal directions of variation in hospital blood parameter data.
# Libraries: NumPy, Pandas, Matplotlib

# PROGRAM

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# ── 1. Generate synthetic patient dataset ──────────────────────────
n = 200
# Correlated features using a base signal
base = np.random.randn(n)
haemoglobin  = 13.5 + 1.5 * base + 0.5 * np.random.randn(n)
rbc          = 4.5  + 0.5 * base + 0.3 * np.random.randn(n)
glucose      = 90   + 15  * np.random.randn(n) - 5 * base
cholesterol  = 190  + 25  * np.random.randn(n) + 10 * base
bmi          = 25   + 4   * np.random.randn(n) + 2  * base

df = pd.DataFrame({
    'Haemoglobin': haemoglobin,
    'RBC_Count'  : rbc,
    'Glucose'    : glucose,
    'Cholesterol': cholesterol,
    'BMI'        : bmi
})

print("Dataset shape:", df.shape)
print(df.describe().round(2))

# ── 2. Standardise ─────────────────────────────────────────────────
df_std = (df - df.mean()) / df.std()
X = df_std.values   # (200, 5)
print("\nStandardised mean (should ≈ 0):", X.mean(axis=0).round(4))
print("Standardised std  (should ≈ 1):", X.std(axis=0).round(4))

# ── 3. Covariance and correlation matrices ─────────────────────────
cov_matrix  = np.cov(X.T)          # (5, 5)
corr_matrix = np.corrcoef(X.T)     # (5, 5)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))
labels = df.columns.tolist()

for ax, mat, title in zip(axes,
                           [cov_matrix, corr_matrix],
                           ['Covariance Matrix', 'Correlation Matrix']):
    im = ax.imshow(mat, cmap='coolwarm', vmin=-1, vmax=1)
    ax.set_xticks(range(5)); ax.set_xticklabels(labels, rotation=45, ha='right')
    ax.set_yticks(range(5)); ax.set_yticklabels(labels)
    ax.set_title(title)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f'{mat[i,j]:.2f}', ha='center', va='center', fontsize=8)
    plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig('lab2_heatmaps.png', dpi=100)
plt.show()

# ── 4. Eigenvalues and eigenvectors ────────────────────────────────
eigenvalues, eigenvectors = np.linalg.eig(cov_matrix)

# Sort descending
order = np.argsort(eigenvalues)[::-1]
eigenvalues  = eigenvalues[order].real
eigenvectors = eigenvectors[:, order].real

print("\nEigenvalues (sorted descending):", np.round(eigenvalues, 4))
print("Top eigenvector:\n", np.round(eigenvectors[:, 0], 4))

# ── 5. 2-D projection with eigenvector arrows ──────────────────────
proj = X @ eigenvectors[:, :2]   # project onto top 2 PCs

plt.figure(figsize=(7, 6))
plt.scatter(proj[:, 0], proj[:, 1], alpha=0.5, edgecolors='k', linewidths=0.3)
origin = [0, 0]
scale = 2.5
for i, label in enumerate(labels):
    plt.arrow(0, 0,
              scale * eigenvectors[i, 0],
              scale * eigenvectors[i, 1],
              head_width=0.08, head_length=0.05,
              fc='red', ec='red')
    plt.text(scale * eigenvectors[i, 0] * 1.15,
             scale * eigenvectors[i, 1] * 1.15,
             label, color='red', fontsize=9)
plt.axhline(0, color='grey', lw=0.5)
plt.axvline(0, color='grey', lw=0.5)
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.title('2D Projection with Top 2 Eigenvectors')
plt.tight_layout()
plt.savefig('lab2_pca_plot.png', dpi=100)
plt.show()

# ── 6. Variance explained ──────────────────────────────────────────
total_var = eigenvalues.sum()
pct_var   = eigenvalues / total_var * 100
cum_var   = np.cumsum(pct_var)

print("\nVariance explained per eigenvalue:")
for i, (ev, pct, cum) in enumerate(zip(eigenvalues, pct_var, cum_var)):
    print(f"  PC{i+1}: eigenvalue={ev:.4f}, var%={pct:.2f}%, cumulative={cum:.2f}%")

# EXPECTED OUTPUT
# Heatmaps of covariance/correlation matrices showing feature correlations.
# Eigenvalues in descending order. 2D scatter with eigenvector arrows.
# Variance explained table showing which PCs capture most information.

# RESULT
# Eigenvalue decomposition of the covariance matrix reveals the principal
# directions of variation. Correlated blood parameters show clear structure
# in the covariance matrix, and the top eigenvectors point toward the
# directions of maximum variance.
