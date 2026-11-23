# Experiment No: 15
# Title: Numerical Differentiation and Gradient Verification
# Aim: Compare analytical derivatives with numerical finite-difference estimates
#      for a custom multivariable loss function.
# Libraries: NumPy, Matplotlib

# PROGRAM

import numpy as np
import matplotlib.pyplot as plt

# ── 1. Define the loss function ───────────────────────────────────
# f(x, y) = x^2 * sin(y) + y^3 * cos(x)
def f(x, y):
    return x**2 * np.sin(y) + y**3 * np.cos(x)

# ── 2. Analytical partial derivatives ─────────────────────────────
# df/dx = 2x * sin(y) - y^3 * sin(x)
# df/dy = x^2 * cos(y) + 3y^2 * cos(x)
def df_dx_analytical(x, y):
    return 2 * x * np.sin(y) - y**3 * np.sin(x)

def df_dy_analytical(x, y):
    return x**2 * np.cos(y) + 3 * y**2 * np.cos(x)

# ── 3. Numerical gradient estimates ───────────────────────────────
def forward_diff(func, x, y, h, var='x'):
    if var == 'x':
        return (func(x + h, y) - func(x, y)) / h
    else:
        return (func(x, y + h) - func(x, y)) / h

def central_diff(func, x, y, h, var='x'):
    if var == 'x':
        return (func(x + h, y) - func(x - h, y)) / (2 * h)
    else:
        return (func(x, y + h) - func(x, y - h)) / (2 * h)

# ── 4. Compare at point (x0, y0) ──────────────────────────────────
x0, y0 = 1.0, 0.5
step_sizes = [1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10]

grad_x_true = df_dx_analytical(x0, y0)
grad_y_true = df_dy_analytical(x0, y0)

print(f"Point: (x={x0}, y={y0})")
print(f"Analytical ∂f/∂x = {grad_x_true:.8f}")
print(f"Analytical ∂f/∂y = {grad_y_true:.8f}")

print(f"\n{'h':>12} {'FD ∂f/∂x err':>16} {'CD ∂f/∂x err':>16} "
      f"{'FD ∂f/∂y err':>16} {'CD ∂f/∂y err':>16}")
print("-" * 80)

fd_x_errs, cd_x_errs = [], []
fd_y_errs, cd_y_errs = [], []

for h in step_sizes:
    fd_x = forward_diff(f, x0, y0, h, 'x')
    cd_x = central_diff(f, x0, y0, h, 'x')
    fd_y = forward_diff(f, x0, y0, h, 'y')
    cd_y = central_diff(f, x0, y0, h, 'y')

    err_fd_x = abs(fd_x - grad_x_true)
    err_cd_x = abs(cd_x - grad_x_true)
    err_fd_y = abs(fd_y - grad_y_true)
    err_cd_y = abs(cd_y - grad_y_true)

    fd_x_errs.append(err_fd_x); cd_x_errs.append(err_cd_x)
    fd_y_errs.append(err_fd_y); cd_y_errs.append(err_cd_y)

    print(f"{h:>12.0e} {err_fd_x:>16.2e} {err_cd_x:>16.2e} "
          f"{err_fd_y:>16.2e} {err_cd_y:>16.2e}")

# Best step size recommendation
best_h_idx = np.argmin(cd_x_errs)
print(f"\nBest step size for central diff: h = {step_sizes[best_h_idx]:.0e} "
      f"(∂f/∂x error = {cd_x_errs[best_h_idx]:.2e})")

# ── 5. Plot gradient error vs step size ───────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

for ax, fd_errs, cd_errs, var in zip(axes,
                                      [fd_x_errs, fd_y_errs],
                                      [cd_x_errs, cd_y_errs],
                                      ['x', 'y']):
    ax.loglog(step_sizes, fd_errs, 'bo-', label='Forward Difference')
    ax.loglog(step_sizes, cd_errs, 'rs-', label='Central Difference')
    best_cd = step_sizes[np.argmin(cd_errs)]
    ax.axvline(best_cd, color='gray', linestyle='--', alpha=0.7,
               label=f'Best h={best_cd:.0e}')
    ax.set_xlabel('Step Size h')
    ax.set_ylabel('Absolute Error')
    ax.set_title(f'Gradient Error vs Step Size (∂f/∂{var})')
    ax.legend()
    ax.grid(True, which='both', alpha=0.3)

plt.suptitle('Numerical Differentiation: Forward vs Central Difference', fontsize=12)
plt.tight_layout()
plt.savefig('lab15_grad_error.png', dpi=100)
plt.show()

# ── 6. Visualise the function surface ─────────────────────────────
xg = np.linspace(-2, 2, 60)
yg = np.linspace(-2, 2, 60)
XG, YG = np.meshgrid(xg, yg)
ZG = f(XG, YG)

fig = plt.figure(figsize=(10, 6))
ax3d = fig.add_subplot(111, projection='3d')
ax3d.plot_surface(XG, YG, ZG, cmap='viridis', alpha=0.8)
ax3d.scatter([x0], [y0], [f(x0, y0)], color='red', s=100, zorder=5,
             label=f'Eval point ({x0},{y0})')
ax3d.set_xlabel('x'); ax3d.set_ylabel('y'); ax3d.set_zlabel('f(x,y)')
ax3d.set_title('f(x,y) = x²·sin(y) + y³·cos(x)')
ax3d.legend()
plt.tight_layout()
plt.savefig('lab15_surface.png', dpi=100)
plt.show()

# EXPECTED OUTPUT
# Error table showing forward diff error ∝ h (first order) and central
# diff error ∝ h² (second order). Both worsen for very small h due to
# floating-point cancellation. Best h ≈ 1e-5 for central difference.

# RESULT
# Central difference approximates gradients more accurately than forward
# difference (O(h²) vs O(h)). Very small h causes floating-point
# cancellation errors. This gradient verification technique (gradient checking)
# is used to debug custom neural network implementations.
