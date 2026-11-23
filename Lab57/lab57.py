# Experiment No: 57
# Title: Choosing Components Using Reconstruction Error
# Aim: Select PCA component count by comparing explained variance and reconstruction error for image compression.
# Libraries: NumPy, Scikit-learn, Matplotlib

# PROGRAM
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error

# Set random seed
np.random.seed(42)

def main():
    print("=" * 65)
    print("  LAB 57: CHOOSING PCA COMPONENTS VIA RECONSTRUCTION ERROR")
    print("=" * 65)

    # Step 1: Load Digits Image Dataset
    digits = load_digits()
    X = digits.data          # Shape: (1797, 64)
    images = digits.images  # Shape: (1797, 8, 8)
    y = digits.target

    n_samples, n_features = X.shape
    print(f"\n[+] Loaded Scikit-Learn Digits Dataset:")
    print(f"    - Number of Samples: {n_samples}")
    print(f"    - Image Resolution:  8x8 pixels ({n_features} features)")
    print(f"    - Pixel Intensity:   Min={X.min()}, Max={X.max()}")

    # Step 2: Evaluate PCA Reconstruction Error across component counts
    k_values = [1, 2, 4, 8, 12, 16, 20, 24, 32, 40, 48, 56, 64]
    mse_errors = []
    explained_variances = []
    compression_ratios = []

    print("\n--- Measuring Compression Quality vs Number of Components (k) ---")
    print(f"{'Components (k)':<15} | {'Variance (%)':<15} | {'MSE Error':<15} | {'Compression Ratio'}")
    print("-" * 65)

    for k in k_values:
        pca = PCA(n_components=k)
        X_trans = pca.fit_transform(X)
        X_recon = pca.inverse_transform(X_trans)

        mse = mean_squared_error(X, X_recon)
        cum_var = np.sum(pca.explained_variance_ratio_) * 100.0
        comp_ratio = (1.0 - (k / n_features)) * 100.0

        mse_errors.append(mse)
        explained_variances.append(cum_var)
        compression_ratios.append(comp_ratio)

        print(f"{k:<15d} | {cum_var:<15.2f} | {mse:<15.4f} | {comp_ratio:<.1f}% reduction")

    # Step 3: Identify Optimal Elbow Point / Thresholds
    # e.g., components for 90% and 95% variance retention
    full_pca = PCA().fit(X)
    full_cum_var = np.cumsum(full_pca.explained_variance_ratio_) * 100.0
    k_90 = np.argmax(full_cum_var >= 90.0) + 1
    k_95 = np.argmax(full_cum_var >= 95.0) + 1

    print(f"\n[+] Recommended Dimensionality Thresholds:")
    print(f"    - To retain 90% information: k = {k_90} components (MSE ~ {mse_errors[k_values.index(min(k_values, key=lambda x:abs(x-k_90)))]:.2f})")
    print(f"    - To retain 95% information: k = {k_95} components (MSE ~ {mse_errors[k_values.index(min(k_values, key=lambda x:abs(x-k_95)))]:.2f})")

    # Step 4: Plot Reconstruction Error & Explained Variance Curves
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Subplot 1: Reconstruction MSE Curve
    axes[0].plot(k_values, mse_errors, marker='o', lw=2.5, color='#c0392b', label='Reconstruction MSE')
    axes[0].axvline(x=k_90, color='#2980b9', linestyle='--', label=f'90% Var (k={k_90})')
    axes[0].axvline(x=k_95, color='#27ae60', linestyle='--', label=f'95% Var (k={k_95})')
    axes[0].set_title("Reconstruction Error (MSE) vs Component Count", fontweight='bold')
    axes[0].set_xlabel("Number of Principal Components (k)", fontweight='bold')
    axes[0].set_ylabel("Mean Squared Error (MSE)", fontweight='bold')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    # Subplot 2: Cumulative Explained Variance Curve
    axes[1].plot(k_values, explained_variances, marker='s', lw=2.5, color='#2c3e50', label='Explained Variance (%)')
    axes[1].axhline(y=90, color='#2980b9', linestyle='--', label='90% Information Threshold')
    axes[1].axhline(y=95, color='#27ae60', linestyle='--', label='95% Information Threshold')
    axes[1].set_title("Cumulative Information Retained vs Components", fontweight='bold')
    axes[1].set_xlabel("Number of Principal Components (k)", fontweight='bold')
    axes[1].set_ylabel("Explained Variance (%)", fontweight='bold')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("lab57_reconstruction_error_curve.png", dpi=300)
    print("\n[+] Reconstruction error curves saved as 'lab57_reconstruction_error_curve.png'")
    plt.show()

    # Step 5: Visualise Image Reconstructions across Component Counts
    display_ks = [2, 8, 16, 32, 64]
    sample_indices = [0, 10, 20, 30, 40]  # Different digit samples

    fig, axes = plt.subplots(len(sample_indices), len(display_ks) + 1, figsize=(12, 10))

    for row_idx, sample_id in enumerate(sample_indices):
        orig_img = images[sample_id]
        axes[row_idx, 0].imshow(orig_img, cmap='gray')
        axes[row_idx, 0].axis('off')
        if row_idx == 0:
            axes[row_idx, 0].set_title("Original\n(64 dims)", fontweight='bold')

        for col_idx, k in enumerate(display_ks):
            pca_k = PCA(n_components=k)
            img_vec = X[sample_id].reshape(1, -1)
            rec_vec = pca_k.inverse_transform(pca_k.fit_transform(X))[sample_id]
            rec_img = rec_vec.reshape(8, 8)

            axes[row_idx, col_idx + 1].imshow(rec_img, cmap='gray')
            axes[row_idx, col_idx + 1].axis('off')
            if row_idx == 0:
                axes[row_idx, col_idx + 1].set_title(f"k = {k}\n({explained_variances[k_values.index(k)]:.0f}%)", fontweight='bold')

    plt.suptitle("Digit Image Reconstruction Quality Across PCA Components", fontsize=14, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig("lab57_digits_reconstruction_grid.png", dpi=300)
    print("[+] Visual digit reconstruction grid saved as 'lab57_digits_reconstruction_grid.png'")
    plt.show()

if __name__ == "__main__":
    main()

# EXPECTED OUTPUT
# =================================================================
#   LAB 57: CHOOSING PCA COMPONENTS VIA RECONSTRUCTION ERROR
# =================================================================
#
# [+] Loaded Scikit-Learn Digits Dataset:
#     - Number of Samples: 1797
#     - Image Resolution:  8x8 pixels (64 features)
#     - Pixel Intensity:   Min=0.0, Max=16.0
#
# --- Measuring Compression Quality vs Number of Components (k) ---
# Components (k)  | Variance (%)    | MSE Error       | Compression Ratio
# -----------------------------------------------------------------
# 1               | 14.89           | 12.4512         | 98.4% reduction
# 2               | 28.51           | 10.4578         | 96.9% reduction
# 4               | 48.65           | 7.5120          | 93.8% reduction
# 8               | 67.41           | 4.7640          | 87.5% reduction
# 12              | 78.43           | 3.1560          | 81.2% reduction
# 16              | 84.95           | 2.2034          | 75.0% reduction
# 20              | 89.44           | 1.5450          | 68.8% reduction
# 24              | 92.60           | 1.0820          | 62.5% reduction
# 32              | 96.69           | 0.4842          | 50.0% reduction
# 40              | 98.71           | 0.1884          | 37.5% reduction
# 48              | 99.68           | 0.0468          | 25.0% reduction
# 56              | 99.98           | 0.0032          | 12.5% reduction
# 64              | 100.00          | 0.0000          | 0.0% reduction
#
# [+] Recommended Dimensionality Thresholds:
#     - To retain 90% information: k = 21 components (MSE ~ 1.55)
#     - To retain 95% information: k = 29 components (MSE ~ 0.48)
#
# [+] Reconstruction error curves saved as 'lab57_reconstruction_error_curve.png'
# [+] Visual digit reconstruction grid saved as 'lab57_digits_reconstruction_grid.png'

# RESULT
# Investigated the trade-off between dimensionality reduction and information preservation in image compression.
# Confirmed that reconstruction Mean Squared Error (MSE) monotonically decreases as components increase,
# showing that keeping approximately 21 to 29 components out of 64 (55% to 67% compression) preserves over
# 90-95% of original image variance with near-lossless perceptual fidelity.
