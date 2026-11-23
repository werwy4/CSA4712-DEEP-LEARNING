# Experiment No: 8
# Title: Vector Norms and Distance Measures
# Aim: Calculate vector norms and distance measures for learner similarity
#      on an online learning platform.
# Libraries: NumPy, SciPy, Matplotlib

# PROGRAM

import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cosine, cityblock, euclidean
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

np.random.seed(42)

# ── 1. Learner feature vectors ─────────────────────────────────────
# Features: [time_spent_hrs, lessons_completed, quiz_attempts, avg_score]
learners = np.array([
    [120, 45, 30, 85],
    [ 60, 20, 15, 72],
    [200, 80, 50, 91],
    [ 30, 10,  8, 55],
    [150, 60, 40, 88]
], dtype=float)

names = [f'Learner {i+1}' for i in range(5)]
print("Learner Feature Vectors:")
print("             Time  Lessons  Quizzes  Score")
for n, v in zip(names, learners):
    print(f"  {n}: {v}")

# ── 2. Vector norms ────────────────────────────────────────────────
print("\n── Vector Norms ──")
print(f"{'Learner':>12} {'L1 norm':>12} {'L2 norm':>12} {'L∞ norm':>12}")
for n, v in zip(names, learners):
    l1  = np.linalg.norm(v, ord=1)
    l2  = np.linalg.norm(v, ord=2)
    linf = np.linalg.norm(v, ord=np.inf)
    print(f"{n:>12} {l1:>12.2f} {l2:>12.2f} {linf:>12.2f}")

# ── 3. Pairwise distances (raw) ────────────────────────────────────
n = len(learners)
euc_mat  = np.zeros((n, n))
man_mat  = np.zeros((n, n))
cos_mat  = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        euc_mat[i, j]  = euclidean(learners[i], learners[j])
        man_mat[i, j]  = cityblock(learners[i], learners[j])
        cos_mat[i, j]  = cosine(learners[i], learners[j])   # 0=identical

print("\n── Euclidean Distance Matrix (raw) ──")
print(np.round(euc_mat, 2))

# ── 4. After standardisation ───────────────────────────────────────
scaler          = StandardScaler()
learners_scaled = scaler.fit_transform(learners)

euc_std = np.zeros((n, n))
man_std = np.zeros((n, n))
cos_std = np.zeros((n, n))

for i in range(n):
    for j in range(n):
        euc_std[i, j] = euclidean(learners_scaled[i], learners_scaled[j])
        man_std[i, j] = cityblock(learners_scaled[i], learners_scaled[j])
        cos_std[i, j] = cosine(learners_scaled[i], learners_scaled[j])

print("\n── Euclidean Distance Matrix (standardised) ──")
print(np.round(euc_std, 4))

# ── 5. Closest learner pairs ───────────────────────────────────────
def closest_pair(mat, metric_name):
    temp = mat.copy()
    np.fill_diagonal(temp, np.inf)
    idx = np.unravel_index(np.argmin(temp), temp.shape)
    print(f"  {metric_name:25s}: {names[idx[0]]} & {names[idx[1]]}  "
          f"(distance = {temp[idx]:.4f})")

print("\n── Closest Learner Pairs (standardised) ──")
closest_pair(euc_std, 'Euclidean')
closest_pair(man_std, 'Manhattan')
closest_pair(cos_std, 'Cosine')

# ── 6. 2-D PCA projection with annotations ─────────────────────────
pca  = PCA(n_components=2)
proj = pca.fit_transform(learners_scaled)

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Left: scatter with learner labels
ax = axes[0]
ax.scatter(proj[:, 0], proj[:, 1], s=150, c=range(n), cmap='tab10', zorder=3)
for i, name in enumerate(names):
    ax.annotate(name, (proj[i, 0], proj[i, 1]),
                textcoords='offset points', xytext=(8, 5), fontsize=9)
ax.set_xlabel('PC 1')
ax.set_ylabel('PC 2')
ax.set_title('Learners in 2-D PCA Space')
ax.grid(True, alpha=0.3)

# Right: Euclidean distance heatmap
ax2 = axes[1]
im  = ax2.imshow(euc_std, cmap='YlOrRd')
ax2.set_xticks(range(n)); ax2.set_xticklabels([f'L{i+1}' for i in range(n)])
ax2.set_yticks(range(n)); ax2.set_yticklabels([f'L{i+1}' for i in range(n)])
ax2.set_title('Euclidean Distance Matrix (Standardised)')
for i in range(n):
    for j in range(n):
        ax2.text(j, i, f'{euc_std[i,j]:.2f}', ha='center', va='center', fontsize=8)
plt.colorbar(im, ax=ax2)

plt.suptitle('Vector Norms and Distance Measures – Learner Similarity', fontsize=13)
plt.tight_layout()
plt.savefig('lab8_distances.png', dpi=100)
plt.show()

# EXPECTED OUTPUT
# Norm table for all 5 learners. Distance matrices before/after standardisation.
# Closest pair identified for each metric. 2-D projection scatter and heatmap.

# RESULT
# Different norms and distance metrics reveal different similarity aspects.
# Standardisation is critical: without it, features with large scales
# (time, lessons) dominate Euclidean and Manhattan distances.
# Cosine distance captures directional similarity regardless of magnitude.
