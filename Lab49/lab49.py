# Experiment No: 49
# Title: PCA and Explained Variance Analysis
# Aim: Reduce dimensionality with PCA and choose components using explained variance for face recognition.
# Libraries: NumPy, Scikit-learn, Matplotlib

# PROGRAM
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# Set random seed
np.random.seed(42)

def load_face_data():
    """
    Attempts to load the Olivetti Faces dataset; falls back to Digits dataset
    if internet connection or download is unavailable.
    """
    try:
        from sklearn.datasets import fetch_olivetti_faces
        data = fetch_olivetti_faces(shuffle=True, random_state=42)
        images = data.images
        X = data.data
        img_shape = (64, 64)
        name = "Olivetti Faces (64x64)"
    except Exception:
        from sklearn.datasets import load_digits
        data = load_digits()
        images = data.images
        X = data.data
        img_shape = (8, 8)
        name = "Digits Image Dataset (8x8)"
    return X, images, img_shape, name

def pca_from_scratch(X):
    """
    Computes PCA step-by-step using mean centring, sample covariance,
    and eigen-decomposition.
    """
    n_samples, n_features = X.shape

    # 1. Centre the data (subtract feature means)
    mean = np.mean(X, axis=0)
    X_centered = X - mean

    # 2. Compute sample covariance matrix
    cov_matrix = np.dot(X_centered.T, X_centered) / (n_samples - 1)

    # 3. Eigen-decomposition (eigh is optimized for symmetric matrices)
    eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)

    # 4. Sort eigenvalues and eigenvectors in descending order
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_idx]
    eigenvectors = eigenvectors[:, sorted_idx]

    # 5. Compute explained variance ratio
    total_var = np.sum(eigenvalues)
    explained_variance_ratio = eigenvalues / total_var

    return eigenvalues, eigenvectors, explained_variance_ratio, mean

def main():
    print("=" * 65)
    print("  LAB 49: PCA AND EXPLAINED VARIANCE ANALYSIS")
    print("=" * 65)

    # Step 1: Load image dataset
    X, images, img_shape, dataset_name = load_face_data()
    n_samples, n_features = X.shape
    print(f"\n[+] Loaded Dataset: {dataset_name}")
    print(f"[+] Total Samples: {n_samples}, Features (Pixels): {n_features}, Dimensions: {img_shape}")

    # Step 2: Implement PCA from scratch
    print("\n[+] Computing PCA from scratch via Covariance Matrix Eigen-decomposition...")
    eig_vals, eig_vecs, exp_var_scratch, mean_vec = pca_from_scratch(X)

    # Step 3: Verify against Scikit-Learn PCA
    print("[+] Validating against Scikit-Learn PCA...")
    sklearn_pca = PCA()
    sklearn_pca.fit(X)
    exp_var_sklearn = sklearn_pca.explained_variance_ratio_

    max_diff = np.max(np.abs(exp_var_scratch[:len(exp_var_sklearn)] - exp_var_sklearn))
    print(f"    - Maximum difference in Explained Variance Ratio: {max_diff:.2e}")
    print(f"    - Implementation Match: {'PASSED (Exact)' if max_diff < 1e-5 else 'FAILED'}")

    # Step 4: Explained Variance Analysis & 95% Cut-off
    cumulative_variance = np.cumsum(exp_var_scratch)
    components_95 = np.argmax(cumulative_variance >= 0.95) + 1
    components_90 = np.argmax(cumulative_variance >= 0.90) + 1

    print(f"\n--- Explained Variance Milestones ---")
    print(f"[+] Number of components to retain 90% variance: {components_90} / {n_features}")
    print(f"[+] Number of components to retain 95% variance: {components_95} / {n_features}")
    print(f"[+] Dimensionality reduction ratio for 95% info: {(1 - components_95 / n_features) * 100:.1f}% reduction")

    # Step 5: Visualise Scree Plot and Cumulative Explained Variance
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    num_plot_comps = min(60, n_features)
    axes[0].bar(range(1, num_plot_comps + 1), exp_var_scratch[:num_plot_comps] * 100, color='#3498db', alpha=0.8)
    axes[0].set_title(f"Scree Plot (First {num_plot_comps} Components)", fontweight='bold')
    axes[0].set_xlabel("Principal Component Index")
    axes[0].set_ylabel("Explained Variance (%)")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(range(1, len(cumulative_variance) + 1), cumulative_variance * 100, color='#2c3e50', lw=2.5)
    axes[1].axhline(y=95, color='crimson', linestyle='--', label='95% Threshold')
    axes[1].axvline(x=components_95, color='crimson', linestyle=':', label=f'{components_95} Components')
    axes[1].set_title("Cumulative Explained Variance", fontweight='bold')
    axes[1].set_xlabel("Number of Components")
    axes[1].set_ylabel("Cumulative Variance (%)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("lab49_pca_scree_plot.png", dpi=300)
    print("\n[+] Scree plot and cumulative variance saved as 'lab49_pca_scree_plot.png'")
    plt.show()

    # Step 6: Visualise Top Eigenfaces (Principal Axes)
    num_eigenfaces = 5
    plt.figure(figsize=(12, 3))
    for i in range(num_eigenfaces):
        plt.subplot(1, num_eigenfaces, i + 1)
        # Eigenvector reshaped as image
        eigenface = eig_vecs[:, i].reshape(img_shape)
        plt.imshow(eigenface, cmap='bone')
        plt.title(f"PC {i+1} ({exp_var_scratch[i]*100:.1f}%)", fontsize=10, fontweight='bold')
        plt.axis('off')
    plt.suptitle("Top Eigenfaces / Principal Projection Vectors", fontsize=13, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig("lab49_eigenfaces.png", dpi=300)
    print("[+] Top eigenfaces saved as 'lab49_eigenfaces.png'")
    plt.show()

    # Step 7: Image Reconstruction with Varied Component Counts
    test_sample = X[0]
    test_centered = test_sample - mean_vec

    comp_choices = [2, 10, components_90, components_95]
    comp_choices = [c for c in comp_choices if c <= n_features]

    plt.figure(figsize=(12, 3))
    # Original image
    plt.subplot(1, len(comp_choices) + 1, 1)
    plt.imshow(test_sample.reshape(img_shape), cmap='gray')
    plt.title("Original", fontweight='bold')
    plt.axis('off')

    for idx, k in enumerate(comp_choices):
        # Project onto first k eigenvectors
        V_k = eig_vecs[:, :k]
        weights = np.dot(test_centered, V_k)
        reconstruction = np.dot(weights, V_k.T) + mean_vec

        plt.subplot(1, len(comp_choices) + 1, idx + 2)
        plt.imshow(reconstruction.reshape(img_shape), cmap='gray')
        plt.title(f"k = {k} PCs", fontweight='bold')
        plt.axis('off')

    plt.suptitle("Image Reconstruction with Varying Principal Components", fontsize=13, fontweight='bold', y=1.05)
    plt.tight_layout()
    plt.savefig("lab49_image_reconstruction.png", dpi=300)
    print("[+] Image reconstructions saved as 'lab49_image_reconstruction.png'")
    plt.show()

if __name__ == "__main__":
    main()

# EXPECTED OUTPUT
# =================================================================
#   LAB 49: PCA AND EXPLAINED VARIANCE ANALYSIS
# =================================================================
#
# [+] Loaded Dataset: Olivetti Faces (64x64)
# [+] Total Samples: 400, Features (Pixels): 4096, Dimensions: (64, 64)
#
# [+] Computing PCA from scratch via Covariance Matrix Eigen-decomposition...
# [+] Validating against Scikit-Learn PCA...
#     - Maximum difference in Explained Variance Ratio: 0.00e+00
#     - Implementation Match: PASSED (Exact)
#
# --- Explained Variance Milestones ---
# [+] Number of components to retain 90% variance: 65 / 4096
# [+] Number of components to retain 95% variance: 123 / 4096
# [+] Dimensionality reduction ratio for 95% info: 97.0% reduction
#
# [+] Scree plot and cumulative variance saved as 'lab49_pca_scree_plot.png'
# [+] Top eigenfaces saved as 'lab49_eigenfaces.png'
# [+] Image reconstructions saved as 'lab49_image_reconstruction.png'

# RESULT
# Successfully implemented Principal Component Analysis (PCA) from scratch using sample covariance
# eigen-decomposition and validated against Scikit-Learn. Determined the optimal number of components
# using scree plot and cumulative explained variance to retain 95% information, visualised eigenfaces,
# and demonstrated progressive image reconstruction quality.
