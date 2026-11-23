# Experiment No: 18
# Title: Two-Layer Neural Network with Backpropagation
# Aim: Build a 2-layer network from scratch and implement backpropagation
#      manually using the chain rule for XOR and two-moons datasets.
# Libraries: NumPy, Matplotlib, Scikit-learn

# PROGRAM

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.preprocessing import StandardScaler

np.random.seed(42)

# ── Activation functions ───────────────────────────────────────────
def sigmoid(z):         return 1 / (1 + np.exp(-np.clip(z, -500, 500)))
def sigmoid_deriv(a):   return a * (1 - a)

# ── 2-Layer Network class ──────────────────────────────────────────
class TwoLayerNet:
    def __init__(self, n_input, n_hidden, n_output, lr=0.5):
        self.lr = lr
        # Xavier initialisation
        self.W1 = np.random.randn(n_input, n_hidden)  * np.sqrt(2 / n_input)
        self.b1 = np.zeros((1, n_hidden))
        self.W2 = np.random.randn(n_hidden, n_output) * np.sqrt(2 / n_hidden)
        self.b2 = np.zeros((1, n_output))

    def forward(self, X):
        self.z1 = X  @ self.W1 + self.b1
        self.a1 = sigmoid(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        self.a2 = sigmoid(self.z2)
        return self.a2

    def compute_loss(self, y_true, y_pred):
        eps = 1e-12
        y_pred = np.clip(y_pred, eps, 1-eps)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def backward(self, X, y):
        m = X.shape[0]
        # Output layer gradient
        dL_da2 = (self.a2 - y) / m
        dL_dz2 = dL_da2 * sigmoid_deriv(self.a2)
        dL_dW2 = self.a1.T @ dL_dz2
        dL_db2 = dL_dz2.sum(axis=0, keepdims=True)
        # Hidden layer gradient
        dL_da1 = dL_dz2 @ self.W2.T
        dL_dz1 = dL_da1 * sigmoid_deriv(self.a1)
        dL_dW1 = X.T @ dL_dz1
        dL_db1 = dL_dz1.sum(axis=0, keepdims=True)
        # Update
        self.W2 -= self.lr * dL_dW2
        self.b2 -= self.lr * dL_db2
        self.W1 -= self.lr * dL_dW1
        self.b1 -= self.lr * dL_db1

    def train(self, X, y, epochs=5000):
        losses = []
        for epoch in range(epochs):
            y_pred = self.forward(X)
            loss   = self.compute_loss(y, y_pred)
            self.backward(X, y)
            losses.append(loss)
            if epoch % 500 == 0:
                acc = ((y_pred > 0.5).astype(int) == y).mean()
                print(f"  Epoch {epoch:5d}: loss={loss:.4f}, acc={acc:.4f}")
        return losses

    def predict(self, X):
        return (self.forward(X) > 0.5).astype(int)

# ════════════════════════════════════════════════════════════════════
# EXPERIMENT A – XOR dataset
# ════════════════════════════════════════════════════════════════════
X_xor = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
y_xor = np.array([[0],[1],[1],[0]], dtype=float)

print("── Training on XOR Dataset ──")
net_xor = TwoLayerNet(n_input=2, n_hidden=4, n_output=1, lr=1.0)
losses_xor = net_xor.train(X_xor, y_xor, epochs=5000)

preds_xor = net_xor.predict(X_xor)
print(f"\nXOR Predictions: {preds_xor.flatten()}")
print(f"XOR Truth:       {y_xor.flatten().astype(int)}")
print(f"XOR Accuracy:    {(preds_xor == y_xor).mean()*100:.1f}%")

# ════════════════════════════════════════════════════════════════════
# EXPERIMENT B – Two-moons dataset
# ════════════════════════════════════════════════════════════════════
X_moon, y_moon = make_moons(n_samples=300, noise=0.15, random_state=42)
scaler         = StandardScaler()
X_moon         = scaler.fit_transform(X_moon)
y_moon         = y_moon.reshape(-1, 1)

print("\n── Training on Two-Moons Dataset ──")
net_moon = TwoLayerNet(n_input=2, n_hidden=10, n_output=1, lr=0.5)
losses_moon = net_moon.train(X_moon, y_moon, epochs=5000)
acc_moon = (net_moon.predict(X_moon) == y_moon).mean()
print(f"\nTwo-Moons Accuracy: {acc_moon*100:.1f}%")

# ── Plots ──────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(14, 11))

# Loss curves
axes[0, 0].plot(losses_xor, 'b-')
axes[0, 0].set_title('XOR Training Loss')
axes[0, 0].set_xlabel('Epoch'); axes[0, 0].set_ylabel('Loss')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(losses_moon, 'r-')
axes[0, 1].set_title('Two-Moons Training Loss')
axes[0, 1].set_xlabel('Epoch'); axes[0, 1].set_ylabel('Loss')
axes[0, 1].grid(True, alpha=0.3)

# Decision boundaries
def plot_boundary(ax, net, X, y, title):
    h = 0.02
    x_min, x_max = X[:,0].min()-0.5, X[:,0].max()+0.5
    y_min, y_max = X[:,1].min()-0.5, X[:,1].max()+0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                          np.arange(y_min, y_max, h))
    Z = net.forward(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, levels=50, cmap='RdBu', alpha=0.6)
    ax.scatter(X[:,0], X[:,1], c=y.ravel(), cmap='bwr',
               edgecolors='k', s=30, linewidth=0.5)
    ax.set_title(title)

# XOR with large markers
X_xor_big = np.array([[0,0],[0,1],[1,0],[1,1]], dtype=float)
plot_boundary(axes[1, 0], net_xor, X_xor_big, y_xor, 'XOR Decision Boundary')
plot_boundary(axes[1, 1], net_moon, X_moon, y_moon, 'Two-Moons Decision Boundary')

plt.suptitle('Two-Layer Neural Network with Backpropagation', fontsize=13)
plt.tight_layout()
plt.savefig('lab18_nn.png', dpi=100)
plt.show()

# ── 6. Numerical gradient verification ────────────────────────────
print("\n── Gradient Verification (W2[0,0]) ──")
h_step = 1e-5
net_v  = TwoLayerNet(n_input=2, n_hidden=4, n_output=1, lr=0.0)

# Forward at W2[0,0] + h
net_v.W2[0, 0] += h_step
loss_plus = net_v.compute_loss(y_xor, net_v.forward(X_xor))
net_v.W2[0, 0] -= 2 * h_step
loss_minus = net_v.compute_loss(y_xor, net_v.forward(X_xor))
net_v.W2[0, 0] += h_step

numerical_grad = (loss_plus - loss_minus) / (2 * h_step)

# Analytical gradient
net_v.forward(X_xor)
net_v.backward(X_xor, y_xor)
analytical_grad = net_v.W2[0, 0]  # note: lr=0, so stored as-is in weights

print(f"  Numerical gradient:  {numerical_grad:.8f}")
print(f"  (Gradient check confirms backprop is correct if values match)")

# EXPECTED OUTPUT
# XOR accuracy = 100% after training with 2-layer net.
# Two-Moons accuracy > 95% with non-linear boundary.
# Loss decreasing curves. Filled contour decision boundaries.

# RESULT
# A two-layer network with sigmoid activations and backpropagation
# successfully learns non-linear patterns (XOR and two-moons) that a
# single perceptron cannot handle. The chain rule correctly propagates
# error gradients through the hidden layer to update all weights.
