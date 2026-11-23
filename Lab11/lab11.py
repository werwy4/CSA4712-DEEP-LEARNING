# Experiment No: 11
# Title: Feature Scaling and Numerical Conditioning
# Aim: Compare common scaling methods and examine their effect on
#      numerical conditioning and model training for vehicle-price prediction.
# Libraries: Pandas, Scikit-learn, NumPy, Matplotlib

# PROGRAM

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, StandardScaler, RobustScaler
from sklearn.linear_model import SGDRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

np.random.seed(42)

# ── 1. Generate synthetic vehicle dataset ─────────────────────────
n = 500
mileage  = np.random.randint(5000,  200000, n).astype(float)
engine   = np.random.randint(500,   5000,   n).astype(float)
age      = np.random.randint(1,     20,     n).astype(float)
# Price: inverse of mileage and age, positive with engine size
price    = (500000 / (mileage ** 0.3)
            + 200 * engine ** 0.5
            - 15000 * np.log(age + 1)
            + np.random.normal(0, 20000, n))
price    = np.clip(price, 50000, 3000000)

df = pd.DataFrame({'Mileage': mileage, 'Engine_cc': engine,
                   'Age': age, 'Price': price})
print("Dataset Statistics:")
print(df.describe().round(2))

X = df[['Mileage', 'Engine_cc', 'Age']].values
y = df['Price'].values

# ── 2. Apply scaling methods ───────────────────────────────────────
scalers = {
    'No Scaling'      : None,
    'Min-Max Scaling' : MinMaxScaler(),
    'Standardisation' : StandardScaler(),
    'Robust Scaling'  : RobustScaler()
}

X_scaled = {}
for name, scaler in scalers.items():
    if scaler is None:
        X_scaled[name] = X.copy()
    else:
        X_scaled[name] = scaler.fit_transform(X)

# ── 3. Compare distributions before/after scaling ─────────────────
fig, axes = plt.subplots(4, 3, figsize=(14, 12))
feat_names = ['Mileage', 'Engine_cc', 'Age']

for row_idx, (name, Xs) in enumerate(X_scaled.items()):
    for col_idx, feat in enumerate(feat_names):
        axes[row_idx, col_idx].hist(Xs[:, col_idx], bins=30,
                                    color='steelblue', edgecolor='k', alpha=0.7)
        axes[row_idx, col_idx].set_title(f'{name}\n{feat}', fontsize=9)
        axes[row_idx, col_idx].set_xlabel('Value')
        axes[row_idx, col_idx].set_ylabel('Frequency')

plt.suptitle('Feature Distributions Before and After Scaling', fontsize=12)
plt.tight_layout()
plt.savefig('lab11_distributions.png', dpi=100)
plt.show()

# ── 4. Train same SGD regression model on each representation ──────
print(f"\n{'Scaling Method':>22} {'Train MSE':>14} {'Val MSE':>14} {'Converged':>12}")
print("-" * 66)

results = {}
for name, Xs in X_scaled.items():
    # Normalise y for fair comparison
    y_mean, y_std = y.mean(), y.std()
    y_norm = (y - y_mean) / y_std

    X_tr, X_val, y_tr, y_val = train_test_split(Xs, y_norm, test_size=0.2, random_state=42)
    model = SGDRegressor(max_iter=1000, tol=1e-4, random_state=42)
    model.fit(X_tr, y_tr)

    train_mse = mean_squared_error(y_tr,  model.predict(X_tr))
    val_mse   = mean_squared_error(y_val, model.predict(X_val))
    converged  = model.n_iter_ < model.max_iter
    results[name] = {'train': train_mse, 'val': val_mse}
    print(f"{name:>22} {train_mse:>14.4f} {val_mse:>14.4f} {str(converged):>12}")

# ── 5. Bar chart of validation MSE ────────────────────────────────
plt.figure(figsize=(8, 5))
methods   = list(results.keys())
val_mses  = [results[m]['val'] for m in methods]
colors    = ['tomato', 'steelblue', 'forestgreen', 'darkorange']
plt.bar(methods, val_mses, color=colors, edgecolor='k')
plt.xticks(rotation=15, ha='right')
plt.ylabel('Validation MSE (normalised)')
plt.title('Scaling Method vs SGDRegressor Validation MSE')
plt.tight_layout()
plt.savefig('lab11_mse.png', dpi=100)
plt.show()

# ── 6. Recommendation ─────────────────────────────────────────────
print("\nRecommendation:")
print("  • StandardScaler is generally best for gradient-based models.")
print("  • RobustScaler is preferable when data contains outliers.")
print("  • MinMaxScaler is useful when features must be bounded to [0,1].")
print("  • No scaling leads to slow convergence or divergence in SGD.")

# EXPECTED OUTPUT
# Distribution histograms showing different scale ranges.
# No-scaling SGD may not converge; scaled versions converge faster with lower MSE.

# RESULT
# Feature scaling is essential for gradient-based models. Without scaling,
# features with large magnitudes dominate the gradient, causing slow
# or failed convergence. Standardisation (zero mean, unit variance) and
# RobustScaler (using median/IQR) are most effective in practice.
