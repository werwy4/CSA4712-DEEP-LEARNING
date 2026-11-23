# Experiment No: 12
# Title: Maximum Likelihood Estimation for Gaussian Data
# Aim: Estimate Gaussian distribution parameters using MLE and examine
#      how sample size affects the estimates for temperature sensor data.
# Libraries: NumPy, SciPy, Matplotlib

# PROGRAM

import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

np.random.seed(42)

TRUE_MU    = 25.0   # true mean temperature (°C)
TRUE_SIGMA = 3.0    # true standard deviation

# ── 1. MLE formulas ────────────────────────────────────────────────
# MLE for Gaussian:
#   mu_hat    = (1/n) * sum(x_i)
#   sigma_hat = sqrt( (1/n) * sum((x_i - mu_hat)^2) )
# Note: MLE for variance is biased (divides by n, not n-1)

def mle_gaussian(samples):
    mu_hat    = np.mean(samples)
    sigma_hat = np.sqrt(np.mean((samples - mu_hat) ** 2))   # biased MLE
    sigma_unb = np.std(samples, ddof=1)                     # unbiased
    return mu_hat, sigma_hat, sigma_unb

# ── 2. Compare for different sample sizes ─────────────────────────
sample_sizes = [10, 50, 100, 500, 1000, 5000]

print(f"True μ = {TRUE_MU},  True σ = {TRUE_SIGMA}")
print(f"\n{'n':>6} {'μ̂ (MLE)':>12} {'σ̂ (MLE)':>12} {'σ (unbiased)':>14} "
      f"{'μ error':>10} {'σ error':>10}")
print("-" * 70)

rows = []
for n in sample_sizes:
    samples         = np.random.normal(TRUE_MU, TRUE_SIGMA, n)
    mu_hat, s_hat, s_unb = mle_gaussian(samples)
    mu_err  = abs(mu_hat - TRUE_MU)
    sig_err = abs(s_hat  - TRUE_SIGMA)
    rows.append((n, mu_hat, s_hat, s_unb, mu_err, sig_err))
    print(f"{n:>6} {mu_hat:>12.4f} {s_hat:>12.4f} {s_unb:>14.4f} "
          f"{mu_err:>10.4f} {sig_err:>10.4f}")

# ── 3. Error vs sample size plot ───────────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

ns       = [r[0] for r in rows]
mu_errs  = [r[4] for r in rows]
sig_errs = [r[5] for r in rows]

axes[0].plot(ns, mu_errs,  'bo-', label='|μ̂ - μ|')
axes[0].set_xscale('log')
axes[0].set_xlabel('Sample Size (n)')
axes[0].set_ylabel('Absolute Error')
axes[0].set_title('MLE μ Estimation Error vs Sample Size')
axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].plot(ns, sig_errs, 'ro-', label='|σ̂ - σ|')
axes[1].set_xscale('log')
axes[1].set_xlabel('Sample Size (n)')
axes[1].set_ylabel('Absolute Error')
axes[1].set_title('MLE σ Estimation Error vs Sample Size')
axes[1].legend(); axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('lab12_mle_errors.png', dpi=100)
plt.show()

# ── 4. Likelihood surface ─────────────────────────────────────────
# Use a moderate sample for visualisation
samples = np.random.normal(TRUE_MU, TRUE_SIGMA, 100)

mu_grid    = np.linspace(20, 30, 100)
sigma_grid = np.linspace(1, 6, 100)
MU, SIG    = np.meshgrid(mu_grid, sigma_grid)

# Log-likelihood: sum of log N(x | mu, sigma)
def log_likelihood(mu, sigma, data):
    n = len(data)
    return (-n * np.log(sigma)
            - np.sum((data - mu)**2) / (2 * sigma**2)
            - n/2 * np.log(2 * np.pi))

LL = np.array([[log_likelihood(m, s, samples) for m in mu_grid]
               for s in sigma_grid])

mu_mle, _, _ = mle_gaussian(samples)
sig_mle       = np.sqrt(np.mean((samples - mu_mle) ** 2))

plt.figure(figsize=(8, 6))
cp = plt.contourf(MU, SIG, LL, levels=30, cmap='RdYlGn')
plt.colorbar(cp, label='Log-Likelihood')
plt.contour(MU, SIG, LL, levels=15, colors='k', linewidths=0.5, alpha=0.4)
plt.plot(mu_mle, sig_mle, 'r*', markersize=15, label=f'MLE (μ={mu_mle:.2f}, σ={sig_mle:.2f})')
plt.plot(TRUE_MU, TRUE_SIGMA, 'b^', markersize=10, label=f'True (μ={TRUE_MU}, σ={TRUE_SIGMA})')
plt.xlabel('μ (mean)')
plt.ylabel('σ (std dev)')
plt.title('Log-Likelihood Surface for Gaussian MLE (n=100)')
plt.legend()
plt.tight_layout()
plt.savefig('lab12_likelihood_surface.png', dpi=100)
plt.show()

# ── 5. Distribution fit ───────────────────────────────────────────
large_samples = np.random.normal(TRUE_MU, TRUE_SIGMA, 1000)
mu_hat, s_hat, _ = mle_gaussian(large_samples)

x_range = np.linspace(TRUE_MU - 4*TRUE_SIGMA, TRUE_MU + 4*TRUE_SIGMA, 300)
plt.figure(figsize=(8, 4))
plt.hist(large_samples, bins=40, density=True, alpha=0.5,
         color='steelblue', edgecolor='k', label='Samples (n=1000)')
plt.plot(x_range, stats.norm.pdf(x_range, TRUE_MU, TRUE_SIGMA),
         'b-', lw=2, label=f'True N({TRUE_MU},{TRUE_SIGMA}²)')
plt.plot(x_range, stats.norm.pdf(x_range, mu_hat, s_hat),
         'r--', lw=2, label=f'MLE N({mu_hat:.2f},{s_hat:.2f}²)')
plt.xlabel('Temperature (°C)')
plt.ylabel('Density')
plt.title('MLE Gaussian Fit to Temperature Sensor Data')
plt.legend(); plt.tight_layout()
plt.savefig('lab12_fit.png', dpi=100)
plt.show()

# EXPECTED OUTPUT
# Error table showing MLE estimates converging to true values as n increases.
# Log-likelihood surface with MLE maximum near true parameters.
# Histogram with fitted Gaussian closely matching true distribution for large n.

# RESULT
# MLE estimates converge to true parameters as sample size increases.
# The MLE for variance is slightly biased (divides by n), while the unbiased
# estimator divides by n-1. The likelihood surface confirms the MLE is a
# global maximum in the (mu, sigma) space.
