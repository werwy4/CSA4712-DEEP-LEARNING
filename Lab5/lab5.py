# Experiment No: 5
# Title: Bias-Variance, Overfitting and Underfitting Analysis
# Aim: Demonstrate underfitting, good fit and overfitting using polynomial
#      regression and estimate bias/variance through resampling.
# Libraries: Scikit-learn, NumPy, Matplotlib

# PROGRAM

import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(42)

# ── 1. Generate synthetic house-price data ─────────────────────────
n = 150
X_raw = np.linspace(50, 500, n)
y = 0.0003 * X_raw**2 - 0.1 * X_raw + 50 + np.random.normal(0, 15, n)

X = X_raw.reshape(-1, 1)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

degrees = [1, 3, 9, 15]
x_plot  = np.linspace(50, 500, 300).reshape(-1, 1)

# ── 2. Fit polynomial regression for each degree ───────────────────
train_errors, test_errors = [], []
models = {}

for deg in degrees:
    model = Pipeline([
        ('poly', PolynomialFeatures(degree=deg)),
        ('reg',  LinearRegression())
    ])
    model.fit(X_train, y_train)
    models[deg] = model
    train_errors.append(mean_squared_error(y_train, model.predict(X_train)))
    test_errors.append(mean_squared_error(y_test,  model.predict(X_test)))

# ── 3. Plot fitted curves ──────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for ax, deg in zip(axes, degrees):
    ax.scatter(X_train, y_train, s=20, alpha=0.6, label='Train', color='steelblue')
    ax.scatter(X_test,  y_test,  s=20, alpha=0.6, label='Test',  color='tomato')
    y_plot = models[deg].predict(x_plot)
    ax.plot(x_plot, y_plot, 'k-', lw=2, label=f'Degree {deg}')
    ax.set_ylim(-50, 350)
    ax.set_title(f'Polynomial Degree {deg}')
    ax.set_xlabel('Area (sq ft)')
    ax.set_ylabel('Price (k)')
    ax.legend(fontsize=8)

plt.suptitle('Polynomial Regression: Underfitting vs Overfitting', fontsize=13)
plt.tight_layout()
plt.savefig('lab5_curves.png', dpi=100)
plt.show()

# ── 4. Training vs Validation error curve ─────────────────────────
plt.figure(figsize=(8, 5))
plt.plot(degrees, train_errors, 'bo-', label='Train MSE')
plt.plot(degrees, test_errors,  'ro-', label='Test  MSE')
plt.xlabel('Polynomial Degree')
plt.ylabel('Mean Squared Error')
plt.title('Bias-Variance Trade-off: Error vs Model Complexity')
plt.legend()
plt.xticks(degrees)
plt.yscale('log')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lab5_bv.png', dpi=100)
plt.show()

# ── 5. Bias² and Variance estimation via bootstrap ─────────────────
print("\nBias² and Variance Estimation (via bootstrap resampling):")
print(f"{'Degree':>8} {'Bias²':>12} {'Variance':>12} {'Test MSE':>12}")
print("-" * 48)

for deg in degrees:
    preds_all = []
    for _ in range(100):
        idx = np.random.choice(len(X_train), len(X_train), replace=True)
        m = Pipeline([('poly', PolynomialFeatures(degree=deg)),
                      ('reg',  LinearRegression())])
        m.fit(X_train[idx], y_train[idx])
        preds_all.append(m.predict(X_test))

    preds_all = np.array(preds_all)       # (100, n_test)
    mean_pred = preds_all.mean(axis=0)
    bias2     = np.mean((mean_pred - y_test) ** 2)
    variance  = np.mean(preds_all.var(axis=0))
    test_mse  = mean_squared_error(y_test, mean_pred)
    print(f"{deg:>8} {bias2:>12.2f} {variance:>12.2f} {test_mse:>12.2f}")

# ── 6. Recommendation ─────────────────────────────────────────────
best_deg = degrees[np.argmin(test_errors)]
print(f"\nBest polynomial degree: {best_deg}  (lowest test MSE = {min(test_errors):.2f})")
print("Degree 1 → underfitting (high bias)")
print("Degree 9/15 → overfitting (high variance)")
print(f"Degree {best_deg} → best bias-variance trade-off")

# EXPECTED OUTPUT
# Degree 1: straight line (underfitting). Degree 3: good fit.
# Degree 9/15: wavy curves that fit noise (overfitting).
# Error plot shows train error drops but test error rises for high degrees.

# RESULT
# Polynomial degree controls model complexity. Too simple → underfitting
# (high bias). Too complex → overfitting (high variance). The best degree
# minimises test error, balancing bias and variance.
