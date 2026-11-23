# Experiment No: 14
# Title: Sampling Methods and the Central Limit Theorem
# Aim: Demonstrate random sampling and the central limit theorem (CLT)
#      for manufacturing quality control of packet weights.
# Libraries: NumPy, SciPy, Matplotlib

# PROGRAM

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# ── 1. Skewed population distribution (exponential) ───────────────
lam = 0.5          # rate parameter → mean = 1/lam = 2 kg
pop_size = 100_000
population = np.random.exponential(scale=1/lam, size=pop_size)

pop_mean = population.mean()
pop_std  = population.std()

print(f"Population: Exponential(λ={lam})")
print(f"  True mean    = {1/lam:.4f}  |  Sample mean = {pop_mean:.4f}")
print(f"  True std     = {1/lam:.4f}  |  Sample std  = {pop_std:.4f}")
print(f"  Skewness     = {stats.skew(population):.4f}")

# ── 2. Draw repeated samples and compute means ────────────────────
sample_sizes  = [5, 10, 30, 100]
n_simulations = 2000

sample_means = {}
for n in sample_sizes:
    means = [np.random.choice(population, n, replace=False).mean()
             for _ in range(n_simulations)]
    sample_means[n] = np.array(means)

# ── 3. Plot sampling distributions vs Gaussian ────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, n in zip(axes, sample_sizes):
    means = sample_means[n]
    # CLT prediction
    clt_std = pop_std / np.sqrt(n)

    ax.hist(means, bins=40, density=True, color='steelblue',
            edgecolor='k', alpha=0.7, label=f'Sample means (n={n})')

    x = np.linspace(means.min(), means.max(), 300)
    ax.plot(x, stats.norm.pdf(x, pop_mean, clt_std),
            'r-', lw=2, label=f'Gaussian N(μ={pop_mean:.2f}, σ={clt_std:.3f})')

    ax.axvline(pop_mean, color='black', linestyle='--', lw=1.5,
               label=f'True mean={pop_mean:.2f}')

    ax.set_title(f'Sample Size n = {n}')
    ax.set_xlabel('Sample Mean')
    ax.set_ylabel('Density')
    ax.legend(fontsize=8)

plt.suptitle("Central Limit Theorem: Sampling Distribution of the Mean\n"
             "(Population: Exponential – right-skewed)", fontsize=12)
plt.tight_layout()
plt.savefig('lab14_clt.png', dpi=100)
plt.show()

# ── 4. Summary statistics of sampling distributions ───────────────
print(f"\n{'n':>5} {'E[X̄]':>10} {'Std[X̄]':>10} {'Pred Std':>10} {'Skewness':>10} {'Normal?':>10}")
print("-" * 60)
for n in sample_sizes:
    means    = sample_means[n]
    pred_std = pop_std / np.sqrt(n)
    skew     = stats.skew(means)
    # Shapiro-Wilk test (subsample for speed)
    _, p_val = stats.shapiro(means[:200])
    normal   = "Yes" if p_val > 0.05 else "No"
    print(f"{n:>5} {means.mean():>10.4f} {means.std():>10.4f} "
          f"{pred_std:>10.4f} {skew:>10.4f} {normal:>10}")

# ── 5. Population vs sample-mean distribution side-by-side ────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].hist(population[:5000], bins=50, density=True,
             color='tomato', edgecolor='k', alpha=0.7)
x = np.linspace(0, 15, 300)
axes[0].plot(x, stats.expon.pdf(x, scale=1/lam), 'k-', lw=2, label='Exponential PDF')
axes[0].set_title('Population Distribution (Skewed)')
axes[0].set_xlabel('Packet Weight (kg)')
axes[0].legend()

n_demo = 30
means  = sample_means[n_demo]
clt_std = pop_std / np.sqrt(n_demo)
axes[1].hist(means, bins=40, density=True,
             color='steelblue', edgecolor='k', alpha=0.7)
x2 = np.linspace(means.min(), means.max(), 300)
axes[1].plot(x2, stats.norm.pdf(x2, pop_mean, clt_std),
             'r-', lw=2, label=f'Gaussian N(μ,σ/{n_demo}^0.5)')
axes[1].set_title(f'Sampling Distribution of X̄  (n={n_demo})')
axes[1].set_xlabel('Sample Mean Weight (kg)')
axes[1].legend()

plt.suptitle('CLT: Skewed Population → Normal Sampling Distribution', fontsize=12)
plt.tight_layout()
plt.savefig('lab14_comparison.png', dpi=100)
plt.show()

# EXPECTED OUTPUT
# For small n (5, 10), sampling distribution is still right-skewed.
# For n=30+, it closely resembles a Gaussian (CLT kicks in).
# Standard deviation of sample means decreases as √n increases.

# RESULT
# The Central Limit Theorem states that the distribution of sample means
# approaches a Gaussian as n increases, regardless of the population shape.
# This justifies many statistical inference procedures and is fundamental
# to understanding why neural network parameter updates behave predictably.
