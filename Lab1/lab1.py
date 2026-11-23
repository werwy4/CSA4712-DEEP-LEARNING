# Experiment No: 1
# Title: Vector and Matrix Operations for Deep Learning
# Aim: Perform vector and matrix operations that underpin a neural network layer
#      for a bookstore recommendation engine with 6 customers and 5 genres.
# Libraries: NumPy, Matplotlib

# PROGRAM

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ── 1. Build matrices ──────────────────────────────────────────────
# Customer-Genre matrix (6 customers × 5 genres): ratings 0-5
customer_genre = np.array([
    [5, 3, 0, 1, 2],
    [4, 0, 4, 1, 2],
    [1, 1, 0, 5, 4],
    [1, 0, 0, 4, 4],
    [0, 1, 5, 4, 0],
    [2, 4, 3, 0, 1]
], dtype=float)

# Genre-Book matrix (5 genres × 4 books): genre weights
genre_book = np.array([
    [0.9, 0.2, 0.1, 0.0],
    [0.1, 0.8, 0.1, 0.1],
    [0.0, 0.1, 0.9, 0.2],
    [0.1, 0.0, 0.1, 0.9],
    [0.2, 0.3, 0.1, 0.5]
], dtype=float)

print("Customer-Genre Matrix (6×5):\n", customer_genre)
print("\nGenre-Book Matrix (5×4):\n", genre_book)

# ── 2. Matrix operations ───────────────────────────────────────────
# Dot product: customer 0 · genre 0 (first row · first column)
dot = np.dot(customer_genre[0], genre_book[:, 0])
print(f"\nDot product (customer 0, book 0): {dot:.4f}")

# Matrix product: customer-book score matrix (6×4)
score_matrix = customer_genre @ genre_book
print("\nScore Matrix (6 customers × 4 books):\n", np.round(score_matrix, 2))

# Transpose
print("\nTranspose of Customer-Genre Matrix (5×6):\n", customer_genre.T)

# Element-wise squaring
print("\nElement-wise squared Customer-Genre:\n", customer_genre ** 2)

# Inverse of a small square sub-matrix (3×3)
sub = customer_genre[:3, :3]
try:
    inv = np.linalg.inv(sub)
    print("\nInverse of 3×3 sub-matrix:\n", np.round(inv, 4))
    print("Verification A @ A_inv ≈ I:\n", np.round(sub @ inv, 4))
except np.linalg.LinAlgError:
    print("Sub-matrix is singular – inverse does not exist.")

# ── 3. Cosine similarity between customers ─────────────────────────
def cosine_similarity(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9)

n = customer_genre.shape[0]
sim_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        sim_matrix[i, j] = cosine_similarity(customer_genre[i], customer_genre[j])

print("\nCosine Similarity Matrix (6×6):\n", np.round(sim_matrix, 3))

# Find the two most similar customers (excluding diagonal)
np.fill_diagonal(sim_matrix, -1)
idx = np.unravel_index(np.argmax(sim_matrix), sim_matrix.shape)
print(f"\nMost similar customers: Customer {idx[0]} and Customer {idx[1]} "
      f"(similarity = {sim_matrix[idx]:.4f})")

# ── 4. Dense layer y = Wx + b ──────────────────────────────────────
W = np.random.randn(3, 5) * 0.5   # 3 output neurons, 5 input features
b = np.random.randn(3) * 0.1
x = customer_genre[0]              # Customer 0's genre vector
y = W @ x + b
print(f"\nDense layer y = Wx + b for Customer 0:")
print(f"  x (input): {x}")
print(f"  W shape:   {W.shape}")
print(f"  y (output): {np.round(y, 4)}")

# ── 5. Heatmap of score matrix ─────────────────────────────────────
np.fill_diagonal(sim_matrix, 1)   # restore diagonal for display

fig, axes = plt.subplots(1, 2, figsize=(12, 4))

im1 = axes[0].imshow(score_matrix, cmap='YlOrRd', aspect='auto')
axes[0].set_title('Customer–Book Score Matrix')
axes[0].set_xlabel('Books')
axes[0].set_ylabel('Customers')
axes[0].set_xticks(range(4)); axes[0].set_xticklabels([f'B{i+1}' for i in range(4)])
axes[0].set_yticks(range(6)); axes[0].set_yticklabels([f'C{i+1}' for i in range(6)])
plt.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(sim_matrix, cmap='Blues', vmin=0, vmax=1)
axes[1].set_title('Customer Cosine Similarity Matrix')
axes[1].set_xlabel('Customer')
axes[1].set_ylabel('Customer')
axes[1].set_xticks(range(6)); axes[1].set_xticklabels([f'C{i+1}' for i in range(6)])
axes[1].set_yticks(range(6)); axes[1].set_yticklabels([f'C{i+1}' for i in range(6)])
plt.colorbar(im2, ax=axes[1])

plt.tight_layout()
plt.savefig('lab1_output.png', dpi=100)
plt.show()
print("\nHeatmap saved as lab1_output.png")

# EXPECTED OUTPUT
# Customer-Genre and Score matrices printed, two most similar customers
# identified, dense layer output computed, and two heatmaps displayed.

# RESULT
# This program demonstrates that fundamental neural network operations
# (dot products, matrix multiplication, dense layers) are built on
# linear algebra, and cosine similarity enables customer-based
# recommendation by measuring vector alignment.
