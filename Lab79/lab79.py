# Experiment No: 79
# Title: Explainability with SHAP and Grad-CAM
# Aim: Apply SHAP for tabular model explanation and Grad-CAM for
#      CNN visual explanation on a medical imaging dataset.
# Libraries: TensorFlow, Keras, NumPy, Matplotlib, Scikit-learn, SHAP

# PROGRAM

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
tf.random.set_seed(42)

# ════════════════════════════════════════════════════════════════
# PART A – SHAP Explanation for Tabular Neural Network
# ════════════════════════════════════════════════════════════════
print("=" * 60)
print("PART A: SHAP for Breast Cancer Classifier")
print("=" * 60)

# ── A1. Load and prepare data ─────────────────────────────────────
bc   = load_breast_cancer()
X, y = bc.data.astype(np.float32), bc.target.astype(np.float32)
feat_names = bc.feature_names

scaler = StandardScaler()
X_sc   = scaler.fit_transform(X)

X_tr, X_te, y_tr, y_te = train_test_split(X_sc, y, test_size=0.2, random_state=42)

# ── A2. Build and train tabular MLP ───────────────────────────────
mlp = keras.Sequential([
    keras.layers.Dense(64, activation='relu', input_shape=(30,)),
    keras.layers.Dense(32, activation='relu'),
    keras.layers.Dense(1,  activation='sigmoid')
], name='BreastCancer_MLP')
mlp.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
mlp.fit(X_tr, y_tr, epochs=50, batch_size=32, validation_split=0.1, verbose=0)

_, acc = mlp.evaluate(X_te, y_te, verbose=0)
print(f"MLP Test Accuracy: {acc:.4f}")

# ── A3. SHAP explanation ──────────────────────────────────────────
# Try importing shap; fall back to manual gradient-based feature importance
try:
    import shap
    HAS_SHAP = True
    print("Using SHAP library for explanations...")
except ImportError:
    HAS_SHAP = False
    print("SHAP not installed — using gradient-based feature importance.")

if HAS_SHAP:
    explainer = shap.GradientExplainer(mlp, X_tr[:100])
    shap_values = explainer.shap_values(X_te[:50])
    if isinstance(shap_values, list):
        shap_values = shap_values[0]
    shap_values = shap_values.squeeze()

    mean_abs_shap = np.abs(shap_values).mean(axis=0)
    top_idx  = np.argsort(mean_abs_shap)[::-1][:10]
    top_names = [feat_names[i] for i in top_idx]
    top_vals  = mean_abs_shap[top_idx]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    axes[0].barh(range(10), top_vals[::-1], color='steelblue')
    axes[0].set_yticks(range(10))
    axes[0].set_yticklabels(top_names[::-1])
    axes[0].set_title('Top-10 Features by Mean |SHAP|')
    axes[0].set_xlabel('Mean |SHAP Value|')
    axes[0].grid(True, alpha=0.3)

    # SHAP scatter for top feature
    axes[1].scatter(X_te[:50, top_idx[0]], shap_values[:50, top_idx[0]],
                   c=y_te[:50], cmap='bwr', alpha=0.7)
    axes[1].set_xlabel(top_names[0])
    axes[1].set_ylabel('SHAP Value')
    axes[1].set_title(f'SHAP Scatter – {top_names[0]}')
    axes[1].grid(True, alpha=0.3)
    plt.suptitle('SHAP Explanations – Breast Cancer MLP', fontsize=11)
    plt.tight_layout()
    plt.savefig('lab79_shap.png', dpi=100); plt.show()
else:
    # Gradient-based importance (∂output/∂input)
    X_tf = tf.constant(X_te[:50], dtype=tf.float32)
    with tf.GradientTape() as tape:
        tape.watch(X_tf)
        preds = mlp(X_tf)
    grads = tape.gradient(preds, X_tf).numpy()
    importance = np.abs(grads).mean(axis=0)
    top_idx   = np.argsort(importance)[::-1][:10]
    top_names = [feat_names[i] for i in top_idx]
    top_vals  = importance[top_idx]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.barh(range(10), top_vals[::-1], color='steelblue')
    ax.set_yticks(range(10)); ax.set_yticklabels(top_names[::-1])
    ax.set_title('Top-10 Features by Gradient Importance')
    ax.set_xlabel('Mean |Gradient|'); ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('lab79_shap.png', dpi=100); plt.show()
    print("Feature importances (gradient-based):")
    for name, val in zip(top_names, top_vals):
        print(f"  {name:35s}: {val:.6f}")

# ════════════════════════════════════════════════════════════════
# PART B – Grad-CAM for CNN Explanation
# ════════════════════════════════════════════════════════════════
print("\n" + "=" * 60)
print("PART B: Grad-CAM for CNN on Fashion-MNIST")
print("=" * 60)

# ── B1. Train small CNN on Fashion-MNIST ─────────────────────────
CLASS_NAMES = ['T-shirt','Trouser','Pullover','Dress','Coat',
               'Sandal','Shirt','Sneaker','Bag','Boot']

(X_fm_tr, y_fm_tr), (X_fm_te, y_fm_te) = keras.datasets.fashion_mnist.load_data()
X_fm_tr = X_fm_tr.astype('float32')[..., np.newaxis] / 255.0
X_fm_te = X_fm_te.astype('float32')[..., np.newaxis] / 255.0

cnn = keras.Sequential([
    keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28,28,1), name='conv1'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64, 3, activation='relu', name='conv2'),
    keras.layers.MaxPooling2D(2,2),
    keras.layers.Conv2D(64, 3, activation='relu', name='conv3'),
    keras.layers.Flatten(),
    keras.layers.Dense(128, activation='relu'),
    keras.layers.Dense(10, activation='softmax')
], name='FashionMNIST_CNN')
cnn.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
            metrics=['accuracy'])

print("Training CNN...")
cnn.fit(X_fm_tr, y_fm_tr, epochs=10, batch_size=64,
        validation_split=0.1, verbose=1)

_, fm_acc = cnn.evaluate(X_fm_te, y_fm_te, verbose=0)
print(f"CNN Test Accuracy: {fm_acc:.4f}")

# ── B2. Grad-CAM implementation ───────────────────────────────────
def grad_cam(model, img_array, layer_name, class_idx=None):
    """Compute Grad-CAM heatmap for a given layer."""
    grad_model = keras.Model(model.inputs,
                             [model.get_layer(layer_name).output, model.output])
    with tf.GradientTape() as tape:
        img_tf = tf.cast(img_array[np.newaxis], tf.float32)
        conv_out, preds = grad_model(img_tf)
        if class_idx is None:
            class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads    = tape.gradient(loss, conv_out)
    pooled   = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap  = tf.reduce_sum(conv_out * pooled, axis=-1).numpy()
    heatmap  = np.maximum(heatmap, 0)
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
    return heatmap

def overlay_heatmap(img, heatmap, alpha=0.4):
    import cv2  # noqa
    h = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    h = np.uint8(255 * h)
    colormap = plt.cm.jet(h)[:,:,:3]
    img_rgb = np.repeat(img, 3, axis=-1)
    return alpha * colormap + (1 - alpha) * img_rgb

# ── B3. Visualise Grad-CAM for sample images ──────────────────────
n_samples = 6
indices   = np.random.choice(len(X_fm_te), n_samples, replace=False)

fig, axes = plt.subplots(n_samples, 3, figsize=(10, n_samples * 3))
fig.suptitle('Grad-CAM Visualisation – Fashion-MNIST CNN', fontsize=12)

for row, idx in enumerate(indices):
    img  = X_fm_te[idx]           # (28,28,1)
    true = y_fm_te[idx]
    pred_probs = cnn.predict(img[np.newaxis], verbose=0)[0]
    pred_cls   = np.argmax(pred_probs)

    heatmap = grad_cam(cnn, img, 'conv3', pred_cls)

    # Try cv2 overlay, else matplotlib imshow
    try:
        overlay = overlay_heatmap(img[:,:,0:1], heatmap)
    except ImportError:
        overlay = plt.cm.hot(heatmap)[:,:,:3]

    axes[row, 0].imshow(img.squeeze(), cmap='gray')
    axes[row, 0].set_title(f'Input: {CLASS_NAMES[true]}', fontsize=8)
    axes[row, 0].axis('off')

    axes[row, 1].imshow(heatmap, cmap='hot')
    axes[row, 1].set_title(f'Grad-CAM Heatmap', fontsize=8)
    axes[row, 1].axis('off')

    axes[row, 2].imshow(overlay)
    axes[row, 2].set_title(f'Pred: {CLASS_NAMES[pred_cls]} ({pred_probs[pred_cls]:.2f})',
                           fontsize=8)
    axes[row, 2].axis('off')

plt.tight_layout()
plt.savefig('lab79_gradcam.png', dpi=100)
plt.show()

# EXPECTED OUTPUT
# SHAP: "worst radius" and "worst perimeter" are top features for BC.
# Grad-CAM: heatmaps highlight distinctive regions (shoe sole, collar, etc.)
# The overlay shows what spatial areas activated the CNN's prediction.

# RESULT
# SHAP explains which tabular features drive each prediction (global/local).
# Grad-CAM visualises which spatial regions in an image activate the CNN's
# final convolutional layer for a given class prediction.
# Both tools are essential for building trustworthy, explainable AI systems.
