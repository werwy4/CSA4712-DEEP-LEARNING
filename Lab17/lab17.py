# Experiment No: 17
# Title: Perceptron and Forward Propagation
# Aim: Implement a single perceptron with forward propagation and the
#      perceptron learning rule for loan approval classification.
# Libraries: NumPy, Matplotlib

# PROGRAM

import numpy as np
import matplotlib.pyplot as plt

np.random.seed(42)

# ── 1. Perceptron class ───────────────────────────────────────────
class Perceptron:
    def __init__(self, n_features, lr=0.1):
        self.w  = np.zeros(n_features)
        self.b  = 0.0
        self.lr = lr

    def step_activation(self, z):
        return 1 if z >= 0 else 0

    def predict_one(self, x):
        z = np.dot(self.w, x) + self.b
        return self.step_activation(z)

    def predict(self, X):
        return np.array([self.predict_one(x) for x in X])

    def fit(self, X, y, epochs=50):
        weight_history = [self.w.copy()]
        errors_per_epoch = []
        for epoch in range(epochs):
            errors = 0
            for xi, yi in zip(X, y):
                y_hat = self.predict_one(xi)
                error = yi - y_hat
                if error != 0:
                    self.w += self.lr * error * xi
                    self.b += self.lr * error
                    errors += 1
            errors_per_epoch.append(errors)
            weight_history.append(self.w.copy())
            if errors == 0:
                print(f"  Converged at epoch {epoch+1}")
                break
        return errors_per_epoch, weight_history

# ── 2. Linearly separable loan dataset ────────────────────────────
n = 100
# Class 0: Rejected – low credit, high debt
c0 = np.column_stack([np.random.uniform(300, 550, n//2),
                      np.random.uniform(0.5, 1.0,  n//2)])
# Class 1: Approved – high credit, low debt
c1 = np.column_stack([np.random.uniform(600, 850, n//2),
                      np.random.uniform(0.1, 0.45, n//2)])

X_loan = np.vstack([c0, c1])
y_loan = np.array([0]*(n//2) + [1]*(n//2))

# Normalise
X_loan_norm = (X_loan - X_loan.mean(axis=0)) / X_loan.std(axis=0)

print("── Training on Loan Dataset ──")
p_loan = Perceptron(n_features=2, lr=0.1)
errors, weights = p_loan.fit(X_loan_norm, y_loan, epochs=100)
acc = (p_loan.predict(X_loan_norm) == y_loan).mean()
print(f"  Training accuracy: {acc*100:.1f}%")

# ── 3. Decision boundary plot ─────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

ax = axes[0]
ax.scatter(X_loan_norm[y_loan==0, 0], X_loan_norm[y_loan==0, 1],
           color='tomato', label='Rejected', edgecolors='k', s=40)
ax.scatter(X_loan_norm[y_loan==1, 0], X_loan_norm[y_loan==1, 1],
           color='steelblue', label='Approved', edgecolors='k', s=40)
# Decision boundary: w0*x0 + w1*x1 + b = 0  →  x1 = -(w0*x0 + b)/w1
x_line = np.linspace(-2.5, 2.5, 100)
w = p_loan.w;  b = p_loan.b
if abs(w[1]) > 1e-6:
    y_line = -(w[0] * x_line + b) / w[1]
    ax.plot(x_line, y_line, 'k-', lw=2, label='Decision Boundary')
ax.set_title('Perceptron: Loan Approval')
ax.set_xlabel('Credit Score (norm)')
ax.set_ylabel('Debt Ratio (norm)')
ax.legend(); ax.set_ylim(-3, 3)

# Error per epoch
axes[1].plot(errors, 'b-o', markersize=4)
axes[1].set_title('Perceptron Training: Misclassifications per Epoch')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Number of Errors')
axes[1].grid(True, alpha=0.3)

# ── 4. XOR failure ────────────────────────────────────────────────
X_xor = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y_xor = np.array([0, 1, 1, 0])

print("\n── Training on XOR Dataset ──")
p_xor = Perceptron(n_features=2, lr=0.1)
xor_errors, _ = p_xor.fit(X_xor, y_xor, epochs=50)
print(f"  Final errors per epoch: {xor_errors[-5:]}")
print(f"  Perceptron FAILS on XOR (not linearly separable)")

axes[2].plot(xor_errors, 'r-o', markersize=5)
axes[2].set_title('XOR Dataset: Perceptron Does NOT Converge')
axes[2].set_xlabel('Epoch')
axes[2].set_ylabel('Misclassifications')
axes[2].set_ylim(-0.5, 4.5)
axes[2].grid(True, alpha=0.3)

plt.suptitle('Single Perceptron – Forward Propagation and Learning Rule', fontsize=12)
plt.tight_layout()
plt.savefig('lab17_perceptron.png', dpi=100)
plt.show()

# ── 5. Activation function comparison ─────────────────────────────
z_vals = np.linspace(-5, 5, 300)

def sigmoid(z):  return 1 / (1 + np.exp(-z))
def relu(z):     return np.maximum(0, z)
def step(z):     return (z >= 0).astype(float)

plt.figure(figsize=(9, 4))
plt.plot(z_vals, step(z_vals),    'k-',  lw=2, label='Step (Perceptron)')
plt.plot(z_vals, sigmoid(z_vals), 'b--', lw=2, label='Sigmoid')
plt.plot(z_vals, relu(z_vals),    'r:',  lw=2, label='ReLU')
plt.axhline(0, color='gray', lw=0.5)
plt.axvline(0, color='gray', lw=0.5)
plt.xlabel('z (pre-activation)');  plt.ylabel('Activation output')
plt.title('Activation Function Comparison: Step vs Sigmoid vs ReLU')
plt.legend(); plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('lab17_activations.png', dpi=100)
plt.show()

# EXPECTED OUTPUT
# Loan data: perceptron converges (perfect separation achieved).
# XOR data: errors oscillate and never reach 0 (failure).
# Activation plot shows step (binary), sigmoid (smooth), ReLU (piecewise linear).

# RESULT
# A single perceptron correctly classifies linearly separable data using
# the perceptron learning rule (weight update ∝ error). It fails on
# non-linearly separable problems like XOR, motivating the need for
# multilayer networks with non-linear activations.
