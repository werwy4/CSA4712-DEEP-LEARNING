# Experiment No: 31
# Title: CNN Feature Extraction and Visualisation
# Aim: Visualise intermediate feature maps and investigate how convolutional
#      representations evolve across layers for Fashion-MNIST.
# Libraries: TensorFlow, Keras, Matplotlib

# PROGRAM

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras

np.random.seed(42)
tf.random.set_seed(42)

CLASS_NAMES = ['T-shirt','Trouser','Pullover','Dress','Coat',
               'Sandal','Shirt','Sneaker','Bag','Boot']

# ── 1. Load and train CNN on Fashion-MNIST ─────────────────────────
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()
X_train = X_train.astype('float32')[..., np.newaxis] / 255.0
X_test  = X_test.astype('float32')[..., np.newaxis]  / 255.0

model = keras.Sequential([
    keras.layers.Conv2D(32, 3, activation='relu', padding='same',
                        input_shape=(28, 28, 1), name='conv1'),
    keras.layers.MaxPooling2D(2, 2, name='pool1'),
    keras.layers.Conv2D(64, 3, activation='relu', padding='same', name='conv2'),
    keras.layers.MaxPooling2D(2, 2, name='pool2'),
    keras.layers.Conv2D(64, 3, activation='relu', padding='same', name='conv3'),
    keras.layers.Flatten(name='flatten'),
    keras.layers.Dense(64, activation='relu', name='dense1'),
    keras.layers.Dense(10, activation='softmax', name='output')
], name='FashionMNIST_CNN')

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.summary()

print("\nTraining CNN on Fashion-MNIST...")
model.fit(X_train, y_train, validation_split=0.1,
          epochs=10, batch_size=64, verbose=1)

test_acc = model.evaluate(X_test, y_test, verbose=0)[1]
print(f"Test Accuracy: {test_acc*100:.2f}%")

# ── 2. Select representative sample images ─────────────────────────
# Pick one image per class
samples, sample_labels = [], []
for class_id in range(10):
    idx = np.where(y_test == class_id)[0][0]
    samples.append(X_test[idx])
    sample_labels.append(class_id)

# Show samples
fig, axes = plt.subplots(2, 5, figsize=(12, 5))
for ax, img, lbl in zip(axes.flatten(), samples, sample_labels):
    ax.imshow(img.squeeze(), cmap='gray')
    ax.set_title(CLASS_NAMES[lbl], fontsize=9)
    ax.axis('off')
plt.suptitle('Selected Sample Images (one per class)', fontsize=12)
plt.tight_layout()
plt.savefig('lab31_samples.png', dpi=100)
plt.show()

# ── 3. Create intermediate feature-map models ─────────────────────
target_layers = ['conv1', 'pool1', 'conv2', 'pool2', 'conv3']
feat_models   = {
    name: keras.Model(inputs=model.input,
                      outputs=model.get_layer(name).output)
    for name in target_layers
}

# ── 4. Visualise feature maps for one test image ───────────────────
test_image = X_test[0:1]   # shape (1,28,28,1)
true_label  = y_test[0]

for layer_name in target_layers:
    feat_maps = feat_models[layer_name].predict(test_image, verbose=0)
    n_maps    = min(feat_maps.shape[-1], 32)
    n_cols    = 8
    n_rows    = int(np.ceil(n_maps / n_cols))

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols*1.5, n_rows*1.5))
    fig.suptitle(f'Feature Maps – {layer_name}  '
                 f'(shape: {feat_maps.shape[1:]}, showing {n_maps} filters)\n'
                 f'Input class: {CLASS_NAMES[true_label]}', fontsize=10)

    for i, ax in enumerate(axes.flatten()):
        if i < n_maps:
            ax.imshow(feat_maps[0, :, :, i], cmap='viridis')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig(f'lab31_{layer_name}.png', dpi=80)
    plt.show()

# ── 5. Compare feature maps across different classes ───────────────
fig, axes = plt.subplots(3, 5, figsize=(15, 9))
fig.suptitle('conv1 Feature Map 0 Across All 10 Classes', fontsize=12)

for col, (img, lbl) in enumerate(zip(samples[:5], sample_labels[:5])):
    inp   = img[np.newaxis, ...]
    fmaps = feat_models['conv1'].predict(inp, verbose=0)
    # Show original
    axes[0, col].imshow(img.squeeze(), cmap='gray')
    axes[0, col].set_title(CLASS_NAMES[lbl], fontsize=9)
    axes[0, col].axis('off')
    # Feature map 0
    axes[1, col].imshow(fmaps[0, :, :, 0], cmap='viridis')
    axes[1, col].set_title(f'Filter 0', fontsize=8)
    axes[1, col].axis('off')
    # Feature map 5
    axes[2, col].imshow(fmaps[0, :, :, min(5, fmaps.shape[-1]-1)], cmap='plasma')
    axes[2, col].set_title(f'Filter 5', fontsize=8)
    axes[2, col].axis('off')

plt.tight_layout()
plt.savefig('lab31_cross_class.png', dpi=100)
plt.show()

# ── 6. Discussion ─────────────────────────────────────────────────
print("\n── Feature Map Statistics ──")
for layer_name in target_layers:
    fm = feat_models[layer_name].predict(test_image, verbose=0)
    print(f"  {layer_name:8s}: shape={fm.shape[1:]}, "
          f"mean={fm.mean():.4f}, std={fm.std():.4f}, "
          f"sparsity={( fm == 0).mean():.2%}")

# EXPECTED OUTPUT
# Early layers (conv1): edge and texture detectors, dense activations.
# Middle layers (conv2): more complex local patterns.
# Deeper layers (conv3): abstract, sparse representations.
# Feature maps differ significantly across clothing categories.

# RESULT
# Earlier convolutional layers learn low-level features (edges, gradients).
# Deeper layers learn high-level semantic features (shapes, textures specific
# to clothing types). This hierarchical feature learning is the key advantage
# of deep CNNs over shallow classifiers.
