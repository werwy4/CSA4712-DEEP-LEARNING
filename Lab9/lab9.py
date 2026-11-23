# Experiment No: 9
# Title: Matrix Rank, Determinant and Invertibility
# Aim: Examine matrix rank, determinant and invertibility for quality-control
#      sensor measurements.
# Libraries: NumPy, SciPy

# PROGRAM

import numpy as np
from scipy import linalg

np.random.seed(42)

# ── Helper ─────────────────────────────────────────────────────────
def analyse_matrix(M, name):
    print(f"\n{'='*55}")
    print(f"Matrix: {name}  (shape {M.shape})")
    print(M)
    rank = np.linalg.matrix_rank(M)
    print(f"  Rank        : {rank}")
    if M.shape[0] == M.shape[1]:
        det = np.linalg.det(M)
        print(f"  Determinant : {det:.6f}")
        is_invertible = abs(det) > 1e-10
        print(f"  Invertible  : {is_invertible}")
        if is_invertible:
            inv = np.linalg.inv(M)
            identity_check = M @ inv
            print(f"  A @ A⁻¹ ≈ I  (max error = {np.max(np.abs(identity_check - np.eye(M.shape[0]))):.2e})")
        else:
            print("  Cannot compute inverse (singular matrix)")
    else:
        print("  Determinant : N/A (non-square matrix)")

# ── 1. Full-rank square matrix ─────────────────────────────────────
A = np.array([[4, 2, 1],
              [1, 3, 2],
              [2, 1, 5]], dtype=float)
analyse_matrix(A, "A (3×3 full-rank)")

# ── 2. Another 4×4 matrix ─────────────────────────────────────────
B = np.random.randint(1, 10, (4, 4)).astype(float)
analyse_matrix(B, "B (4×4 random)")

# ── 3. Singular matrix (dependent rows) ───────────────────────────
C = np.array([[1, 2, 3],
              [4, 5, 6],
              [7, 8, 9]], dtype=float)   # row3 = row1 + row2 (approx)
analyse_matrix(C, "C (3×3 singular – row3 = row1 + row2)")

# ── 4. Rectangular matrix (non-square) ────────────────────────────
D = np.random.randn(4, 6)
analyse_matrix(D, "D (4×6 rectangular)")

# ── 5. Rank-deficient matrix: introduce linear dependence ──────────
E = np.array([[2, 4, 6],
              [1, 2, 3],
              [3, 7, 2]], dtype=float)   # col2 = 2*col1 for first two rows
E_modified = E.copy()
E_modified[:, 2] = 2 * E_modified[:, 0]   # make col3 = 2*col1 → rank drops

print("\n── Effect of Introducing Linear Dependence ──")
print("Original matrix E:")
print(E)
print(f"  Rank: {np.linalg.matrix_rank(E)}")

print("\nModified E (col3 = 2×col1 – linearly dependent):")
print(E_modified)
print(f"  Rank: {np.linalg.matrix_rank(E_modified)}  (reduced!)")
print(f"  Determinant: {np.linalg.det(E_modified):.6f}  (≈ 0 ⇒ singular)")

# ── 6. Eigenvalues of the singular matrix ─────────────────────────
print("\n── Eigenvalues of singular matrix E_modified ──")
evals = np.linalg.eigvals(E_modified)
print(f"  Eigenvalues: {np.round(evals, 6)}")
print("  (At least one eigenvalue ≈ 0 confirms singularity)")

# ── 7. Moore-Penrose pseudo-inverse for rectangular/singular ───────
pinv_C = np.linalg.pinv(C)
print("\n── Moore-Penrose Pseudo-Inverse of C ──")
print(f"pinv(C) shape: {pinv_C.shape}")
print(f"C @ pinv(C) @ C ≈ C: max error = "
      f"{np.max(np.abs(C @ pinv_C @ C - C)):.2e}")

# EXPECTED OUTPUT
# Full-rank matrices show non-zero determinants and successful inversion.
# Singular matrices show determinant ≈ 0, rank < n, and no inverse.
# Introducing a dependent column drops rank and makes det → 0.

# RESULT
# Matrix rank measures the number of linearly independent rows/columns.
# A square matrix is invertible if and only if its rank = n (full rank)
# and determinant ≠ 0. Linear dependence among sensor channels reduces
# rank, making the matrix singular and uninvertible.
