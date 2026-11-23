# Experiment No: 7
# Title: Linear Algebra Operations for Neural Network Data Preparation
# Aim: Prepare and transform numerical customer data using fundamental
#      linear algebra operations for retail analytics.
# Libraries: NumPy, Pandas, Matplotlib

# PROGRAM

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

np.random.seed(42)

# ── 1. Customer-feature matrix ─────────────────────────────────────
# Features: age, spending, browsing_frequency, purchase_history
n_customers = 10
data = {
    'Age'              : np.random.randint(18, 65, n_customers),
    'Spending'         : np.random.randint(100, 5000, n_customers),
    'Browsing_Freq'    : np.random.randint(1, 30, n_customers),
    'Purchase_History' : np.random.randint(1, 50, n_customers)
}
df = pd.DataFrame(data)
print("Customer-Feature DataFrame:")
print(df.to_string())

X = df.values.astype(float)   # (10, 4) matrix
print(f"\nFeature matrix X shape: {X.shape}")

# ── 2. Matrix operations ───────────────────────────────────────────
# a) Addition and subtraction
X_shift = X + 10
print("\nX + 10 (first 3 rows):\n", X_shift[:3])
X_diff  = X - X.mean(axis=0)
print("X - column_mean (first 3 rows):\n", X_diff[:3].round(2))

# b) Matrix multiplication: X @ X.T → (10×10) similarity matrix
sim = X @ X.T
print(f"\nX @ X.T shape: {sim.shape}")

# c) Transpose
print(f"X.T shape: {X.T.shape}")

# d) Element-wise operations
X_norm = X / X.max(axis=0)   # scale each column to [0,1]
print("\nElement-wise normalisation (first 3 rows):\n", X_norm[:3].round(4))

# ── 3. Dimension mismatch demonstration ───────────────────────────
print("\n── Dimension Mismatch Demo ──")
A = np.random.randn(3, 4)
B = np.random.randn(5, 3)
try:
    C = A @ B
except ValueError as e:
    print(f"Error caught: {e}")
    print("Explanation: A is (3,4), B is (5,3) – inner dims don't match for matmul.")

# Fix: use B.T → (3,5), so A @ B.T works
C_fixed = A @ B.T
print(f"Fixed: A @ B.T shape = {C_fixed.shape}")

# ── 4. Matrix transformation ───────────────────────────────────────
# Apply a 4×3 projection matrix to get a 3-feature representation
W_proj = np.random.randn(4, 3) * 0.5
X_transformed = X_norm @ W_proj        # (10, 3)
print(f"\nProjection matrix W shape: {W_proj.shape}")
print(f"Transformed X shape:       {X_transformed.shape}")
print("Transformed data (first 3 rows):\n", X_transformed[:3].round(4))

# ── 5. Visualise original and transformed feature matrices ─────────
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

im1 = axes[0].imshow(X_norm, cmap='YlGnBu', aspect='auto')
axes[0].set_title('Original Normalised Feature Matrix (10×4)')
axes[0].set_xlabel('Features')
axes[0].set_ylabel('Customers')
axes[0].set_xticks(range(4))
axes[0].set_xticklabels(['Age', 'Spending', 'Browsing', 'Purchase'])
axes[0].set_yticks(range(10))
axes[0].set_yticklabels([f'C{i+1}' for i in range(10)])
for i in range(10):
    for j in range(4):
        axes[0].text(j, i, f'{X_norm[i,j]:.2f}', ha='center', va='center', fontsize=7)
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(X_transformed, cmap='coolwarm', aspect='auto')
axes[1].set_title('Transformed Feature Matrix (10×3)')
axes[1].set_xlabel('Projected Features')
axes[1].set_ylabel('Customers')
axes[1].set_xticks(range(3))
axes[1].set_xticklabels(['F1', 'F2', 'F3'])
axes[1].set_yticks(range(10))
axes[1].set_yticklabels([f'C{i+1}' for i in range(10)])
for i in range(10):
    for j in range(3):
        axes[1].text(j, i, f'{X_transformed[i,j]:.2f}', ha='center', va='center', fontsize=7)
plt.colorbar(im2, ax=axes[1])

plt.suptitle('Linear Algebra: Feature Matrix Transformation', fontsize=13)
plt.tight_layout()
plt.savefig('lab7_matrices.png', dpi=100)
plt.show()

# EXPECTED OUTPUT
# Customer-Feature DataFrame, matrix operation results,
# a dimension mismatch error (caught and explained), and two heatmaps
# showing the original and linearly transformed feature matrices.

# RESULT
# Linear algebra operations (addition, multiplication, transpose,
# transformation) are the fundamental building blocks of neural network
# data preparation and layer computations. Valid matrix dimensions are
# essential for correct computations.
