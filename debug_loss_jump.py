"""
Debug script to identify the source of loss jump after compute_V in GCN AWB pipeline.

Tests:
1. Compare get_AWBT() output with __call__() output after compute_V
2. Verify V = A @ W @ B.T transformation is correct
3. Compare loss on same batch with both forward passes
"""

import sys
sys.path.insert(0, 'src')

import jax
import jax.numpy as jnp
import equinox as eqx
import optax
import numpy as np

from cl.models.gcn import GCN, GCNLayer
from cl.core.awb import compute_V_from_AWB_gcn
from cl.arch_search.gcn_search import prepABs_GCN

# Set seed for reproducibility
key = jax.random.PRNGKey(42)

# Create a simple GCN model with original architecture
original_gcn_sizes = [10, 32, 32]  # [in_size, hidden, hidden]
original_feed_sizes = [32, 32, 16, 5]  # [gcn_out, h1, h2, n_class]

print("=" * 70)
print("DEBUG: AWB Loss Jump Investigation")
print("=" * 70)

# Step 1: Create model
print("\n[1] Creating GCN model with original architecture")
model = GCN(
    in_size=10,
    gcn_sizes=original_gcn_sizes,
    feed_sizes=original_feed_sizes,
    SEED=42,
    graph=True
)

print(f"  Original gcn_sizes: {original_gcn_sizes}")
print(f"  Original feed_sizes: {original_feed_sizes}")
print(f"  gcn_layers[0].weight shape: {model.gcn_layers[0].weight.shape}")
print(f"  gcn_layers[1].weight shape: {model.gcn_layers[1].weight.shape}")
print(f"  feed_layers[0].weight shape: {model.feed_layers[0].weight.shape}")

# Step 2: Create fake input data
print("\n[2] Creating fake input data")
batch_size = 4
nodes_per_graph = 20
total_nodes = batch_size * nodes_per_graph

key, subkey = jax.random.split(key)
x = jax.random.normal(subkey, (total_nodes, 10))

# Create block-diagonal adjacency matrix
adj = jnp.zeros((total_nodes, total_nodes))
for i in range(batch_size):
    start = i * nodes_per_graph
    end = (i + 1) * nodes_per_graph
    # Random edges within each graph
    key, subkey = jax.random.split(key)
    graph_adj = jax.random.bernoulli(subkey, 0.3, (nodes_per_graph, nodes_per_graph))
    graph_adj = (graph_adj + graph_adj.T) / 2  # Symmetric
    adj = adj.at[start:end, start:end].set(graph_adj)

# Add self-loops and normalize (mimicking T.GCNNorm())
adj = adj + jnp.eye(total_nodes)
degree = jnp.sum(adj, axis=1)
deg_inv_sqrt = jnp.power(degree, -0.5)
deg_inv_sqrt = jnp.where(jnp.isinf(deg_inv_sqrt), 0., deg_inv_sqrt)
deg_mat = jnp.diag(deg_inv_sqrt)
adj_norm = deg_mat @ adj @ deg_mat

# Batch indices and node counts
batch = jnp.repeat(jnp.arange(batch_size), nodes_per_graph)
n_nodes = jnp.full((batch_size,), nodes_per_graph)

# Random labels
key, subkey = jax.random.split(key)
y = jax.random.randint(subkey, (batch_size,), 0, 5)

print(f"  x shape: {x.shape}")
print(f"  adj_norm shape: {adj_norm.shape}")
print(f"  batch shape: {batch.shape}")
print(f"  y shape: {y.shape}")

# Step 3: Test original model forward pass
print("\n[3] Testing original model forward pass")
pred_original = model(x, adj_norm, batch, n_nodes)
loss_original = jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels(pred_original, y))
print(f"  Original model __call__ loss: {loss_original:.6f}")

# Step 4: Set up new architecture (larger)
print("\n[4] Setting up AWB transformation to larger architecture")
new_gcn_sizes = [10, 72, 72]
new_feed_sizes = [72, 172, 196, 5]
print(f"  New gcn_sizes: {new_gcn_sizes}")
print(f"  New feed_sizes: {new_feed_sizes}")

# Update architecture metadata
model = eqx.tree_at(lambda x: x.gcn_sizes, model, new_gcn_sizes)
model = eqx.tree_at(lambda x: x.feed_sizes, model, new_feed_sizes)

# Get A/B matrices
A_feed, B_feed, A_gcn, B_gcn = prepABs_GCN(model, original_feed_sizes, original_gcn_sizes)

# Set A/B matrices on model
model = eqx.tree_at(
    lambda x: (x.A_feed, x.B_feed, x.A_gcn, x.B_gcn),
    model,
    replace=(A_feed, B_feed, A_gcn, B_gcn)
)

print("\n  A/B matrix shapes:")
for i in range(len(A_gcn)):
    print(f"    A_gcn[{i}]: {A_gcn[i].shape}, B_gcn[{i}]: {B_gcn[i].shape}")
for i in range(len(A_feed)):
    print(f"    A_feed[{i}]: {A_feed[i].shape}, B_feed[{i}]: {B_feed[i].shape}")

# Step 5: Test get_AWBT before compute_V (simulating AB training end)
print("\n[5] Testing get_AWBT BEFORE compute_V (AB training phase)")
pred_awbt_before = model.get_AWBT(x, adj_norm, batch, n_nodes)
loss_awbt_before = jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels(pred_awbt_before, y))
print(f"  get_AWBT loss BEFORE compute_V: {loss_awbt_before:.6f}")

# Step 6: Apply compute_V transformation
print("\n[6] Applying compute_V transformation")
model_v = compute_V_from_AWB_gcn(model)

print("  Weight shapes after compute_V:")
print(f"    gcn_layers[0].weight: {model_v.gcn_layers[0].weight.shape}")
print(f"    gcn_layers[1].weight: {model_v.gcn_layers[1].weight.shape}")
print(f"    feed_layers[0].weight: {model_v.feed_layers[0].weight.shape}")

# Step 7: Test __call__ after compute_V (V training phase)
print("\n[7] Testing __call__ AFTER compute_V (V training phase)")
pred_call_after = model_v(x, adj_norm, batch, n_nodes)
loss_call_after = jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels(pred_call_after, y))
print(f"  __call__ loss AFTER compute_V: {loss_call_after:.6f}")

# Step 8: Test get_AWBT after compute_V (SHOULD give different result!)
print("\n[8] Testing get_AWBT AFTER compute_V (should be WRONG)")
pred_awbt_after = model_v.get_AWBT(x, adj_norm, batch, n_nodes)
loss_awbt_after = jnp.mean(optax.losses.softmax_cross_entropy_with_integer_labels(pred_awbt_after, y))
print(f"  get_AWBT loss AFTER compute_V: {loss_awbt_after:.6f} (expected to be wrong)")

# Step 9: Summary and analysis
print("\n" + "=" * 70)
print("SUMMARY:")
print("=" * 70)
print(f"  [BEFORE compute_V] get_AWBT: {loss_awbt_before:.6f}")
print(f"  [AFTER compute_V]  __call__: {loss_call_after:.6f}")
print(f"  Loss difference: {abs(loss_awbt_before - loss_call_after):.6f}")

if abs(loss_awbt_before - loss_call_after) < 1e-4:
    print("\n  ✓ PASS: Losses match! compute_V is working correctly.")
else:
    print("\n  ✗ FAIL: Loss mismatch detected!")
    print("    This explains the loss jump after AWB training.")

    # Detailed diagnostics
    print("\n  DETAILED DIAGNOSTICS:")
    print(f"    Output difference (max): {jnp.max(jnp.abs(pred_awbt_before - pred_call_after)):.10f}")
    print(f"    Output difference (mean): {jnp.mean(jnp.abs(pred_awbt_before - pred_call_after)):.10f}")

    # Check layer by layer for GCN
    print("\n  Checking GCN layer outputs:")
    x_before = x.copy()
    x_after = x.copy()

    for i in range(len(model.gcn_layers)):
        # get_AWBT path
        V_weight_awbt = model.A_gcn[i] @ model.gcn_layers[i].weight @ model.B_gcn[i].T
        support_awbt = x_before @ V_weight_awbt
        x_before = adj_norm @ support_awbt
        if model.gcn_layers[i].bias_flag:
            x_before += (model.gcn_layers[i].bias @ model.B_gcn[i].T)
        x_before = jax.nn.leaky_relu(x_before)

        # __call__ path (after compute_V)
        support_call = x_after @ model_v.gcn_layers[i].weight
        x_after = adj_norm @ support_call
        if model_v.gcn_layers[i].bias_flag:
            x_after += model_v.gcn_layers[i].bias
        x_after = jax.nn.leaky_relu(x_after)

        diff = jnp.max(jnp.abs(x_before - x_after))
        print(f"    GCN layer {i} output diff (max): {diff:.10f}")

        # Check weight equality
        weight_diff = jnp.max(jnp.abs(V_weight_awbt - model_v.gcn_layers[i].weight))
        print(f"    GCN layer {i} weight V vs computed: {weight_diff:.10f}")

        # Check bias equality
        bias_awbt = model.gcn_layers[i].bias @ model.B_gcn[i].T
        bias_diff = jnp.max(jnp.abs(bias_awbt - model_v.gcn_layers[i].bias))
        print(f"    GCN layer {i} bias V vs computed: {bias_diff:.10f}")

print("\n" + "=" * 70)
