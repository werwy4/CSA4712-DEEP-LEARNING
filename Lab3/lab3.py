# Experiment No: 3
# Title: Probability Distributions and Bayes' Theorem
# Aim: Model data with standard distributions and apply Bayes' theorem
#      for medical diagnostic decision-making.
# Libraries: NumPy, SciPy, Matplotlib

# PROGRAM

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

# ── 1. Standard probability distributions ─────────────────────────
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
fig.suptitle('Standard Probability Distributions', fontsize=14)

# Bernoulli
p = 0.3
samples_bern = np.random.binomial(1, p, 1000)
axes[0, 0].hist(samples_bern, bins=[-0.5, 0.5, 1.5], density=True,
                color='steelblue', edgecolor='k', rwidth=0.6)
axes[0, 0].bar([0, 1], [1-p, p], alpha=0.4, color='orange', label='PMF')
axes[0, 0].set_title('Bernoulli (p=0.3)')
axes[0, 0].legend()

# Binomial
n_bin, p_bin = 20, 0.4
x_bin = np.arange(0, n_bin+1)
axes[0, 1].hist(np.random.binomial(n_bin, p_bin, 1000), bins=20,
                density=True, color='steelblue', edgecolor='k', label='Samples')
axes[0, 1].plot(x_bin, stats.binom.pmf(x_bin, n_bin, p_bin),
                'ro-', label='PMF')
axes[0, 1].set_title(f'Binomial (n={n_bin}, p={p_bin})')
axes[0, 1].legend()

# Poisson
lam = 4
x_pois = np.arange(0, 15)
axes[0, 2].hist(np.random.poisson(lam, 1000), bins=15,
                density=True, color='steelblue', edgecolor='k', label='Samples')
axes[0, 2].plot(x_pois, stats.poisson.pmf(x_pois, lam), 'ro-', label='PMF')
axes[0, 2].set_title(f'Poisson (λ={lam})')
axes[0, 2].legend()

# Uniform
a, b = 0, 10
x_uni = np.linspace(-1, 11, 200)
axes[1, 0].hist(np.random.uniform(a, b, 1000), bins=30,
                density=True, color='steelblue', edgecolor='k', label='Samples')
axes[1, 0].plot(x_uni, stats.uniform.pdf(x_uni, a, b-a), 'r-', lw=2, label='PDF')
axes[1, 0].set_title(f'Uniform ({a}, {b})')
axes[1, 0].legend()

# Gaussian
mu, sigma = 5, 2
x_gauss = np.linspace(-2, 12, 200)
axes[1, 1].hist(np.random.normal(mu, sigma, 1000), bins=30,
                density=True, color='steelblue', edgecolor='k', label='Samples')
axes[1, 1].plot(x_gauss, stats.norm.pdf(x_gauss, mu, sigma),
                'r-', lw=2, label='PDF')
axes[1, 1].set_title(f'Gaussian (μ={mu}, σ={sigma})')
axes[1, 1].legend()

axes[1, 2].axis('off')
plt.tight_layout()
plt.savefig('lab3_distributions.png', dpi=100)
plt.show()

# ── 2. Bayes' Theorem ─────────────────────────────────────────────
prevalence  = 0.01   # P(Disease)
sensitivity = 0.95   # P(Positive | Disease)
specificity = 0.90   # P(Negative | Healthy)

false_positive_rate = 1 - specificity   # P(Positive | Healthy)

# P(Positive) = P(Pos|Dis)*P(Dis) + P(Pos|Healthy)*P(Healthy)
p_positive = sensitivity * prevalence + false_positive_rate * (1 - prevalence)

# Bayes: P(Disease | Positive)
p_disease_given_pos = (sensitivity * prevalence) / p_positive

print("=" * 50)
print("Bayes' Theorem – Medical Diagnosis")
print("=" * 50)
print(f"  P(Disease)              = {prevalence*100:.1f}%")
print(f"  P(Positive | Disease)   = {sensitivity*100:.1f}%  [sensitivity]")
print(f"  P(Negative | Healthy)   = {specificity*100:.1f}%  [specificity]")
print(f"  P(Positive)             = {p_positive*100:.4f}%")
print(f"  P(Disease | Positive)   = {p_disease_given_pos*100:.2f}%")

# ── 3. Empirical verification with 100,000 patients ───────────────
n_patients = 100_000
diseases   = np.random.binomial(1, prevalence, n_patients)

# Generate test results
test_pos = np.where(diseases == 1,
                    np.random.binomial(1, sensitivity, n_patients),
                    np.random.binomial(1, false_positive_rate, n_patients))

# Among positive tests, how many actually have disease?
positive_mask      = test_pos == 1
true_positives     = diseases[positive_mask].sum()
total_positive     = positive_mask.sum()
empirical_prob     = true_positives / total_positive

print(f"\nEmpirical simulation ({n_patients:,} patients):")
print(f"  Total positive tests    = {total_positive}")
print(f"  True positives          = {true_positives}")
print(f"  Empirical P(D|Pos)      = {empirical_prob*100:.2f}%  "
      f"(theoretical = {p_disease_given_pos*100:.2f}%)")

# ── 4. Posterior vs prevalence ────────────────────────────────────
prev_range = np.linspace(0.001, 0.20, 500)
posterior  = (sensitivity * prev_range) / (
    sensitivity * prev_range + false_positive_rate * (1 - prev_range))

plt.figure(figsize=(8, 5))
plt.plot(prev_range * 100, posterior * 100, 'b-', lw=2)
plt.axvline(prevalence * 100, color='red', linestyle='--',
            label=f'Current prevalence ({prevalence*100}%)')
plt.axhline(p_disease_given_pos * 100, color='green', linestyle=':',
            label=f'Current P(D|+) = {p_disease_given_pos*100:.1f}%')
plt.xlabel('Disease Prevalence (%)')
plt.ylabel('P(Disease | Positive Test) (%)')
plt.title("Posterior Probability vs Disease Prevalence")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lab3_bayes.png', dpi=100)
plt.show()

# EXPECTED OUTPUT
# Plots of 5 distributions. Bayesian posterior ≈ 8.7% at 1% prevalence.
# Empirical simulation should closely match the analytical result.
# Posterior-vs-prevalence curve shows how rare diseases have low PPV.

# RESULT
# Bayes' theorem quantifies how the posterior probability of disease
# depends on prevalence. Even with high sensitivity and specificity,
# low prevalence leads to a surprisingly low positive predictive value,
# demonstrating the base-rate fallacy.
