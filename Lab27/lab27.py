# Experiment No: 27
# Title: Convolution and Pooling Operations
# Aim: Implement convolution and pooling operations manually and examine
#      their effects on image features.
# Libraries: NumPy, TensorFlow, Matplotlib

# PROGRAM

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

np.random.seed(42)

# ── 1. Create a synthetic test image ──────────────────────────────
# 32x32 checkerboard with gradient
size = 32
img = np.zeros((size, size))
for i in range(size):
    for j in range(size):
        if (i // 4 + j // 4) % 2 == 0:
            img[i, j] = 200 + i * 1.5
        else:
            img[i, j] = 50 + j * 1.5
img = img / 255.0   # normalise to [0,1]
print(f"Image shape: {img.shape}, Range: [{img.min():.3f}, {img.max():.3f}]")

# ── 2. Define kernels ─────────────────────────────────────────────
kernels = {
    'Horizontal Edge': np.array([[-1,-1,-1],
                                  [ 2, 2, 2],
                                  [-1,-1,-1]], dtype=float),
    'Vertical Edge'  : np.array([[-1, 2,-1],
                                  [-1, 2,-1],
                                  [-1, 2,-1]], dtype=float),
    'Sobel X'        : np.array([[-1, 0, 1],
                                  [-2, 0, 2],
                                  [-1, 0, 1]], dtype=float),
    'Gaussian Blur'  : np.array([[1,2,1],
                                  [2,4,2],
                                  [1,2,1]], dtype=float) / 16.0,
    'Sharpen'        : np.array([[ 0,-1, 0],
                                  [-1, 5,-1],
                                  [ 0,-1, 0]], dtype=float),
}

# ── 3. Manual 2-D convolution (NumPy) ─────────────────────────────
def conv2d_manual(image, kernel, stride=1, padding='same'):
    h, w   = image.shape
    kh, kw = kernel.shape
    if padding == 'same':
        ph = kh // 2
        pw = kw // 2
        image = np.pad(image, ((ph, ph), (pw, pw)), mode='constant')
    out_h = (image.shape[0] - kh) // stride + 1
    out_w = (image.shape[1] - kw) // stride + 1
    output = np.zeros((out_h, out_w))
    for i in range(0, out_h):
        for j in range(0, out_w):
            patch = image[i*stride:i*stride+kh, j*stride:j*stride+kw]
            output[i, j] = np.sum(patch * kernel)
    return output

# ── 4. Max and average pooling ────────────────────────────────────
def max_pool(image, size=2, stride=2):
    h, w = image.shape
    oh   = (h - size) // stride + 1
    ow   = (w - size) // stride + 1
    out  = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[i, j] = image[i*stride:i*stride+size, j*stride:j*stride+size].max()
    return out

def avg_pool(image, size=2, stride=2):
    h, w = image.shape
    oh   = (h - size) // stride + 1
    ow   = (w - size) // stride + 1
    out  = np.zeros((oh, ow))
    for i in range(oh):
        for j in range(ow):
            out[i, j] = image[i*stride:i*stride+size, j*stride:j*stride+size].mean()
    return out

# ── 5. Apply all kernels ──────────────────────────────────────────
results = {}
for name, kernel in kernels.items():
    conv_out = conv2d_manual(img, kernel, stride=1, padding='same')
    conv_out = np.clip(conv_out, 0, 1)
    mp_out   = max_pool(conv_out, size=2, stride=2)
    ap_out   = avg_pool(conv_out, size=2, stride=2)
    results[name] = {'conv': conv_out, 'max_pool': mp_out, 'avg_pool': ap_out}
    print(f"{name:20s}: conv={conv_out.shape}, max_pool={mp_out.shape}, avg_pool={ap_out.shape}")

# ── 6. Visualise ─────────────────────────────────────────────────
n_kernels = len(kernels)
fig, axes = plt.subplots(n_kernels + 1, 4, figsize=(16, (n_kernels+1)*3))
fig.suptitle('Convolution and Pooling Operations', fontsize=13)

axes[0, 0].imshow(img, cmap='gray'); axes[0, 0].set_title('Original Image'); axes[0, 0].axis('off')
axes[0, 1].axis('off'); axes[0, 2].axis('off'); axes[0, 3].axis('off')

col_titles = ['Filter', 'Conv Output', 'Max Pooling', 'Avg Pooling']
for col_idx, title in enumerate(col_titles):
    axes[0, col_idx].set_title(title if col_idx == 0 else '', fontsize=10)

for row, (name, kernel) in enumerate(kernels.items(), start=1):
    r = results[name]
    # Filter visualisation
    im = axes[row, 0].imshow(kernel, cmap='bwr')
    axes[row, 0].set_title(f'{name}\n(kernel)', fontsize=8)
    axes[row, 0].axis('off')
    plt.colorbar(im, ax=axes[row, 0], fraction=0.04)

    axes[row, 1].imshow(r['conv'], cmap='gray')
    axes[row, 1].set_title(f'Conv {r["conv"].shape}', fontsize=8)
    axes[row, 1].axis('off')

    axes[row, 2].imshow(r['max_pool'], cmap='gray')
    axes[row, 2].set_title(f'MaxPool {r["max_pool"].shape}', fontsize=8)
    axes[row, 2].axis('off')

    axes[row, 3].imshow(r['avg_pool'], cmap='gray')
    axes[row, 3].set_title(f'AvgPool {r["avg_pool"].shape}', fontsize=8)
    axes[row, 3].axis('off')

plt.tight_layout()
plt.savefig('lab27_conv_pooling.png', dpi=100)
plt.show()

# ── 7. Dimension analysis ─────────────────────────────────────────
print("\n── Dimension Analysis ──")
for name, r in results.items():
    orig = img.shape
    conv = r['conv'].shape
    mp   = r['max_pool'].shape
    print(f"{name:20s}: {orig} → conv {conv} → pool {mp}")

print("\nFormula: out_size = (in_size + 2*pad - kernel) / stride + 1")
print("MaxPool 2×2 stride 2 halves each spatial dimension.")

# EXPECTED OUTPUT
# 5 filter outputs showing different feature responses (edges, blur, sharpness).
# Max/Avg pooling outputs are half the size of conv outputs.
# Edge filters respond to transitions; blur smooths; sharpen enhances detail.

# RESULT
# Convolution extracts local spatial features using learnable filters.
# Different kernels detect edges, blurs, and textures. Pooling reduces
# spatial dimensions while preserving dominant features, providing
# translation invariance and reducing computation in subsequent layers.
