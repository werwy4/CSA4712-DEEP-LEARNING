# Experiment No: 77
# Title: Graph Neural Network for Molecular Property Prediction
# Aim: Build a simplified Message Passing Neural Network (MPNN) to
#      predict molecular properties from graph-structured data.
# Libraries: TensorFlow, Keras, NumPy, Pandas, Matplotlib, Scikit-learn

# PROGRAM

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

np.random.seed(42)
tf.random.set_seed(42)

# ── 1. Synthetic molecular graph dataset ──────────────────────────
# Each molecule is represented as:
#   - Node feature matrix X: (N_atoms, node_feat_dim)
#   - Adjacency matrix A:    (N_atoms, N_atoms)  [bonds]
#   - Target property y:     scalar (e.g., boiling point)
#
# We generate random small molecules (4-8 atoms) with synthetic features.

def make_molecule(n_atoms, node_feat_dim=5):
    """Create a random molecule graph."""
    X = np.random.randn(n_atoms, node_feat_dim).astype(np.float32)
    # Random adjacency (symmetric, no self-loops)
    A = np.random.rand(n_atoms, n_atoms)
    A = (A + A.T) / 2
    A = (A > 0.5).astype(np.float32)
    np.fill_diagonal(A, 0)
    # Degree-normalize adjacency
    deg = A.sum(axis=1, keepdims=True) + 1e-6
    A_norm = A / deg
    return X, A_norm

NODE_FEAT = 5
N_MOL     = 2000
MAX_ATOMS = 8

# Pad all graphs to MAX_ATOMS for batching
all_X, all_A, all_y = [], [], []
for _ in range(N_MOL):
    n_atoms = np.random.randint(4, MAX_ATOMS + 1)
    X, A    = make_molecule(n_atoms, NODE_FEAT)
    # Pad to MAX_ATOMS
    X_pad = np.zeros((MAX_ATOMS, NODE_FEAT), dtype=np.float32)
    A_pad = np.zeros((MAX_ATOMS, MAX_ATOMS), dtype=np.float32)
    X_pad[:n_atoms, :] = X
    A_pad[:n_atoms, :n_atoms] = A
    all_X.append(X_pad)
    all_A.append(A_pad)
    # Synthetic target: sum of node features × connectivity
    target = float(X.sum() + A.sum() + np.random.normal(0, 0.5))
    all_y.append(target)

all_X = np.array(all_X)   # (N_MOL, MAX_ATOMS, NODE_FEAT)
all_A = np.array(all_A)   # (N_MOL, MAX_ATOMS, MAX_ATOMS)
all_y = np.array(all_y, dtype=np.float32)

y_scaler = StandardScaler()
all_y_sc = y_scaler.fit_transform(all_y.reshape(-1,1)).ravel().astype(np.float32)

(X_tr, X_te, A_tr, A_te,
 y_tr, y_te) = train_test_split(all_X, all_A, all_y_sc,
                                 test_size=0.2, random_state=42)
print(f"X_train: {X_tr.shape}, A_train: {A_tr.shape}")

# ── 2. Graph Convolutional Layer (GCN-style) ───────────────────────
class GraphConvLayer(keras.layers.Layer):
    """Single GCN message-passing layer: H' = σ(A_norm · H · W)."""
    def __init__(self, units, activation='relu', **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.activation = keras.activations.get(activation)

    def build(self, input_shape):
        node_feat_dim = input_shape[0][-1]
        self.W = self.add_weight('W', shape=(node_feat_dim, self.units),
                                 initializer='glorot_uniform', trainable=True)
        self.b = self.add_weight('b', shape=(self.units,),
                                 initializer='zeros', trainable=True)

    def call(self, inputs):
        H, A = inputs     # H: (batch, nodes, feat), A: (batch, nodes, nodes)
        AH   = tf.matmul(A, H)       # (batch, nodes, feat)
        out  = tf.matmul(AH, self.W) + self.b
        return self.activation(out)

# ── 3. Build GNN model ────────────────────────────────────────────
node_inp = keras.Input(shape=(MAX_ATOMS, NODE_FEAT), name='node_features')
adj_inp  = keras.Input(shape=(MAX_ATOMS, MAX_ATOMS),  name='adjacency')

h = GraphConvLayer(32, activation='relu', name='GCN1')([node_inp, adj_inp])
h = GraphConvLayer(64, activation='relu', name='GCN2')([h, adj_inp])
h = GraphConvLayer(64, activation='relu', name='GCN3')([h, adj_inp])

# Global mean pooling (graph-level readout)
h_pool = keras.layers.GlobalAveragePooling1D(name='mean_pool')(h)

# MLP head
h_pool = keras.layers.Dense(64, activation='relu')(h_pool)
h_pool = keras.layers.Dropout(0.2)(h_pool)
h_pool = keras.layers.Dense(32, activation='relu')(h_pool)
output = keras.layers.Dense(1, activation='linear', name='property')(h_pool)

gnn_model = keras.Model([node_inp, adj_inp], output, name='GNN_MolProp')
gnn_model.compile(optimizer=keras.optimizers.Adam(1e-3), loss='mse', metrics=['mae'])
gnn_model.summary()

# ── 4. Train ──────────────────────────────────────────────────────
es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=15,
                                   restore_best_weights=True)
print("\nTraining GNN...")
history = gnn_model.fit(
    [X_tr, A_tr], y_tr,
    validation_split=0.1,
    epochs=100, batch_size=64,
    callbacks=[es], verbose=1
)

# ── 5. Evaluate ───────────────────────────────────────────────────
y_pred_sc = gnn_model.predict([X_te, A_te], verbose=0).ravel()
y_pred    = y_scaler.inverse_transform(y_pred_sc.reshape(-1,1)).ravel()
y_true    = y_scaler.inverse_transform(y_te.reshape(-1,1)).ravel()

mae = mean_absolute_error(y_true, y_pred)
r2  = r2_score(y_true, y_pred)
print(f"\n── Test Metrics ──")
print(f"  MAE : {mae:.4f}")
print(f"  R²  : {r2:.4f}")

# ── 6. Plots ───────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

axes[0].plot(history.history['loss'],     label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Val Loss',  linestyle='--')
axes[0].set_title('GNN Training Loss'); axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('MSE'); axes[0].legend(); axes[0].grid(True, alpha=0.3)

axes[1].scatter(y_true, y_pred, alpha=0.3, s=5, color='steelblue')
lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
axes[1].plot(lims, lims, 'r--', label='Perfect prediction')
axes[1].set_xlabel('True Property'); axes[1].set_ylabel('Predicted Property')
axes[1].set_title(f'Scatter (R²={r2:.3f})'); axes[1].legend()
axes[1].grid(True, alpha=0.3)

residuals = y_true - y_pred
axes[2].hist(residuals, bins=40, color='steelblue', alpha=0.7)
axes[2].axvline(0, color='red', linestyle='--')
axes[2].set_title('Residual Distribution'); axes[2].set_xlabel('Residual')
axes[2].set_ylabel('Count'); axes[2].grid(True, alpha=0.3)

plt.suptitle('Graph Neural Network – Molecular Property Prediction', fontsize=12)
plt.tight_layout()
plt.savefig('lab77_gnn_molecular.png', dpi=100)
plt.show()

# EXPECTED OUTPUT
# GNN converges with decreasing loss. R² > 0.90 on synthetic data.
# Scatter plot shows predicted vs true values clustered around y=x.

# RESULT
# Graph Neural Networks process irregular graph-structured data by
# propagating feature messages along edges (bonds). Each GCN layer
# aggregates neighbour features, building richer molecular representations.
# Global pooling reduces node features to a fixed-size molecular embedding
# for downstream property prediction.
