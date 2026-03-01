"""
Transformer encoder for continual learning classification.

Implements a lightweight encoder-only transformer with optional AWB support.
Designed for sequence-shaped inputs (e.g., MNIST flattened to 784 tokens).
"""

import jax
from jax import core
import jax.numpy as jnp
import equinox as eqx
from typing import List, Optional, Dict, Any

from .layers import Linear, AWBLayerSpec, AWBMixin


class TransformerBlock(eqx.Module):
    qkv: Linear
    out_proj: Linear
    mlp1: Linear
    mlp2: Linear
    norm1: eqx.nn.LayerNorm
    norm2: eqx.nn.LayerNorm
    n_heads: int
    head_dim: int

    def __init__(self, embed_dim: int, n_heads: int, mlp_dim: int, key: jax.Array):
        if embed_dim % n_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by n_heads {n_heads}")

        key_qkv, key_out, key_mlp1, key_mlp2 = jax.random.split(key, 4)
        self.qkv = Linear(embed_dim, 3 * embed_dim, key=key_qkv)
        self.out_proj = Linear(embed_dim, embed_dim, key=key_out)
        self.mlp1 = Linear(embed_dim, mlp_dim, key=key_mlp1)
        self.mlp2 = Linear(mlp_dim, embed_dim, key=key_mlp2)
        self.norm1 = eqx.nn.LayerNorm(embed_dim, use_weight=True, use_bias=True)
        self.norm2 = eqx.nn.LayerNorm(embed_dim, use_weight=True, use_bias=True)
        self.n_heads = n_heads
        self.head_dim = embed_dim // n_heads

    def _self_attention(self, x: jax.Array) -> jax.Array:
        seq_len = x.shape[0]
        qkv = self.qkv(x)
        qkv = qkv.reshape(seq_len, 3, self.n_heads, self.head_dim)

        q = jnp.transpose(qkv[:, 0], (1, 0, 2))
        k = jnp.transpose(qkv[:, 1], (1, 0, 2))
        v = jnp.transpose(qkv[:, 2], (1, 0, 2))

        scale = 1.0 / jnp.sqrt(self.head_dim)
        attn_scores = jnp.einsum('hqd,hkd->hqk', q, k) * scale
        attn_weights = jax.nn.softmax(attn_scores, axis=-1)
        attn_out = jnp.einsum('hqk,hkd->hqd', attn_weights, v)
        attn_out = jnp.transpose(attn_out, (1, 0, 2)).reshape(seq_len, self.n_heads * self.head_dim)
        return self.out_proj(attn_out)

    def __call__(self, x: jax.Array) -> jax.Array:
        h = jax.vmap(self.norm1)(x)
        x = x + self._self_attention(h)
        h = jax.vmap(self.norm2)(x)
        x = x + self.mlp2(jax.nn.gelu(self.mlp1(h)))
        return x


class TransformerEncoder(AWBMixin, eqx.Module):
    input_proj: Linear
    blocks: List[TransformerBlock]
    head: Linear
    pos_embed: jax.Array
    seq_len: int
    input_dim: int
    embed_dim: int
    n_heads: int
    mlp_dim: int
    n_layers: int
    awb_enabled: bool
    A: Optional[List[jax.Array]]
    B: Optional[List[jax.Array]]

    def __init__(
        self,
        seq_len: int,
        input_dim: int,
        embed_dim: int,
        n_heads: int,
        mlp_dim: int,
        n_layers: int,
        output_dim: int,
        key: Optional[jax.Array] = None,
        awb_enabled: bool = False,
    ):
        if key is None:
            key = jax.random.PRNGKey(0)

        if embed_dim % n_heads != 0:
            raise ValueError(f"embed_dim {embed_dim} must be divisible by n_heads {n_heads}")

        self.seq_len = seq_len
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.n_heads = n_heads
        self.mlp_dim = mlp_dim
        self.n_layers = n_layers
        self.awb_enabled = awb_enabled

        key_in, key_pos, key_blocks, key_head = jax.random.split(key, 4)
        self.input_proj = Linear(input_dim, embed_dim, key=key_in)
        self.pos_embed = jax.random.normal(key_pos, (seq_len, embed_dim)) * 0.02

        block_keys = jax.random.split(key_blocks, n_layers)
        self.blocks = [
            TransformerBlock(embed_dim, n_heads, mlp_dim, key=block_keys[i])
            for i in range(n_layers)
        ]

        self.head = Linear(embed_dim, output_dim, key=key_head)

        if awb_enabled:
            self.A, self.B = self._init_awb_matrices()
        else:
            self.A, self.B = None, None

    def _linear_shapes(self) -> List[tuple]:
        shapes = [(self.input_proj.weight.shape[0], self.input_proj.weight.shape[1])]
        for block in self.blocks:
            shapes.append((block.qkv.weight.shape[0], block.qkv.weight.shape[1]))
            shapes.append((block.out_proj.weight.shape[0], block.out_proj.weight.shape[1]))
            shapes.append((block.mlp1.weight.shape[0], block.mlp1.weight.shape[1]))
            shapes.append((block.mlp2.weight.shape[0], block.mlp2.weight.shape[1]))
        shapes.append((self.head.weight.shape[0], self.head.weight.shape[1]))
        return shapes

    def _init_awb_matrices(self) -> tuple:
        A_list = []
        B_list = []
        for out_dim, in_dim in self._linear_shapes():
            A_list.append(jnp.eye(out_dim, dtype=jnp.float32))
            B_list.append(jnp.eye(in_dim, dtype=jnp.float32))
        return A_list, B_list

    def _linear_paths(self):
        paths = []
        paths.append((lambda m: m.input_proj.weight, lambda m: m.input_proj.bias))
        for i in range(len(self.blocks)):
            paths.append((
                lambda m, idx=i: m.blocks[idx].qkv.weight,
                lambda m, idx=i: m.blocks[idx].qkv.bias,
            ))
            paths.append((
                lambda m, idx=i: m.blocks[idx].out_proj.weight,
                lambda m, idx=i: m.blocks[idx].out_proj.bias,
            ))
            paths.append((
                lambda m, idx=i: m.blocks[idx].mlp1.weight,
                lambda m, idx=i: m.blocks[idx].mlp1.bias,
            ))
            paths.append((
                lambda m, idx=i: m.blocks[idx].mlp2.weight,
                lambda m, idx=i: m.blocks[idx].mlp2.bias,
            ))
        paths.append((lambda m: m.head.weight, lambda m: m.head.bias))
        return paths

    def _normalize_input(self, x: jax.Array) -> jax.Array:
        if isinstance(x, core.Tracer):
            return x

        if x.ndim == 0:
            raise ValueError("Transformer input must be at least 1D")

        expected = self.seq_len * self.input_dim
        if x.ndim == 2 and x.shape == (self.seq_len, self.input_dim):
            return x
        if x.size != expected:
            raise ValueError(
                f"Expected input with {expected} elements (seq_len={self.seq_len}, token_dim={self.input_dim}), "
                f"got shape {x.shape}"
            )

        return jnp.reshape(x, (self.seq_len, self.input_dim))

    def __call__(self, x: jax.Array) -> jax.Array:
        x = self._normalize_input(x)
        x = self.input_proj(x)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = jnp.mean(x, axis=0)
        x = self.head(x)
        if x.ndim == 2 and x.shape[0] == 1:
            x = jnp.squeeze(x, axis=0)
        return x

    def _apply_linear_awb(self, x: jax.Array, layer: Linear, A: jax.Array, B: jax.Array) -> jax.Array:
        V_weight = self.awb_transform_linear(A, layer.weight, B)
        V_bias = self.awb_transform_bias_linear(A, layer.bias, bias_shape='row')
        return x @ V_weight.T + V_bias

    def _attention_awb(self, x: jax.Array, block: TransformerBlock, A_qkv, B_qkv, A_out, B_out) -> jax.Array:
        seq_len = x.shape[0]
        qkv = self._apply_linear_awb(x, block.qkv, A_qkv, B_qkv)
        qkv = qkv.reshape(seq_len, 3, block.n_heads, block.head_dim)

        q = jnp.transpose(qkv[:, 0], (1, 0, 2))
        k = jnp.transpose(qkv[:, 1], (1, 0, 2))
        v = jnp.transpose(qkv[:, 2], (1, 0, 2))

        scale = 1.0 / jnp.sqrt(block.head_dim)
        attn_scores = jnp.einsum('hqd,hkd->hqk', q, k) * scale
        attn_weights = jax.nn.softmax(attn_scores, axis=-1)
        attn_out = jnp.einsum('hqk,hkd->hqd', attn_weights, v)
        attn_out = jnp.transpose(attn_out, (1, 0, 2)).reshape(seq_len, block.n_heads * block.head_dim)
        return self._apply_linear_awb(attn_out, block.out_proj, A_out, B_out)

    def get_AWBT(self, x: jax.Array) -> jax.Array:
        if not self.awb_enabled or self.A is None or self.B is None:
            raise ValueError("AWB not enabled - A/B matrices are not initialized")

        x = self._normalize_input(x)

        idx = 0
        x = self._apply_linear_awb(x, self.input_proj, self.A[idx], self.B[idx])
        idx += 1
        x = x + self.pos_embed

        for block in self.blocks:
            h = jax.vmap(block.norm1)(x)
            attn_out = self._attention_awb(h, block, self.A[idx], self.B[idx], self.A[idx + 1], self.B[idx + 1])
            idx += 2
            x = x + attn_out

            h = jax.vmap(block.norm2)(x)
            mlp_out = self._apply_linear_awb(h, block.mlp1, self.A[idx], self.B[idx])
            idx += 1
            mlp_out = jax.nn.gelu(mlp_out)
            mlp_out = self._apply_linear_awb(mlp_out, block.mlp2, self.A[idx], self.B[idx])
            idx += 1
            x = x + mlp_out

        x = jnp.mean(x, axis=0)
        x = self._apply_linear_awb(x, self.head, self.A[idx], self.B[idx])
        if x.ndim == 2 and x.shape[0] == 1:
            x = jnp.squeeze(x, axis=0)
        return x

    def getAWB(self, x: jax.Array) -> jax.Array:
        return self.get_AWBT(x)

    def get_awb_layer_specs(self) -> List[AWBLayerSpec]:
        layers = [self.input_proj]
        for block in self.blocks:
            layers.extend([block.qkv, block.out_proj, block.mlp1, block.mlp2])
        layers.append(self.head)

        specs = []
        for i, layer in enumerate(layers):
            A = self.A[i] if self.A is not None else None
            B = self.B[i] if self.B is not None else None
            specs.append(AWBLayerSpec(layer=layer, A=A, B=B, layer_type='linear', layer_index=i))
        return specs

    def apply_V_transformation(self) -> 'TransformerEncoder':
        if not self.awb_enabled or self.A is None or self.B is None:
            raise ValueError("apply_V_transformation requires initialized A/B matrices")

        model = self
        paths = self._linear_paths()
        for i, (w_path, b_path) in enumerate(paths):
            layer = w_path(model)
            bias = b_path(model)
            V_weight = self.awb_transform_linear(model.A[i], layer, model.B[i])
            V_bias = self.awb_transform_bias_linear(model.A[i], bias, bias_shape='row')
            model = eqx.tree_at(w_path, model, V_weight)
            model = eqx.tree_at(b_path, model, V_bias)

        return model

    def partition_for_AB_training(self):
        filter_spec = eqx.tree_map(lambda _: False, self)
        filter_spec = eqx.tree_at(
            lambda x: (x.A, x.B),
            filter_spec,
            replace=(True, True)
        )
        diff_model, static_model = eqx.partition(self, filter_spec)
        return diff_model, static_model

    def partition_for_standard_training(self):
        params, static = eqx.partition(self, eqx.is_array)

        if self.awb_enabled and self.A is not None and self.B is not None:
            static = eqx.tree_at(
                lambda x: (x.A, x.B),
                static,
                replace=(self.A, self.B)
            )
            params = eqx.tree_at(
                lambda x: (x.A, x.B),
                params,
                replace=(None, None)
            )

        return params, static

    def with_new_AB_matrices(self) -> 'TransformerEncoder':
        model = self
        A_list, B_list = self._init_awb_matrices()
        model = eqx.tree_at(lambda x: x.A, model, A_list)
        model = eqx.tree_at(lambda x: x.B, model, B_list)
        model = eqx.tree_at(lambda x: x.awb_enabled, model, True)
        return model

    def reinitialize_weights(self, seed: int = 0) -> 'TransformerEncoder':
        initializer = jax.nn.initializers.glorot_uniform()
        model = self
        paths = self._linear_paths()
        for i, (w_path, b_path) in enumerate(paths):
            weight = initializer(jax.random.PRNGKey(seed + i), w_path(model).shape)
            bias = initializer(jax.random.PRNGKey(seed + i + 100), b_path(model).shape)
            model = eqx.tree_at(w_path, model, weight)
            model = eqx.tree_at(b_path, model, bias)
        return model


def create_transformer(config: Dict[str, Any]) -> TransformerEncoder:
    input_size = config['input_size']
    seq_len = config.get('transformer_seq_len')
    token_dim = config.get('transformer_token_dim', 1)

    if seq_len is None:
        if input_size % token_dim != 0:
            raise ValueError(f"input_size {input_size} not divisible by token_dim {token_dim}")
        seq_len = input_size // token_dim
    else:
        expected = seq_len * token_dim
        if expected != input_size:
            raise ValueError(f"seq_len*token_dim={expected} does not match input_size={input_size}")

    embed_dim = config.get('transformer_embed_dim', 128)
    n_heads = config.get('transformer_n_heads', 4)
    mlp_dim = config.get('transformer_mlp_dim', 256)
    n_layers = config.get('transformer_n_layers', 2)
    output_dim = config['output_size']
    awb_enabled = config.get('awb_enabled', False)

    return TransformerEncoder(
        seq_len=seq_len,
        input_dim=token_dim,
        embed_dim=embed_dim,
        n_heads=n_heads,
        mlp_dim=mlp_dim,
        n_layers=n_layers,
        output_dim=output_dim,
        key=jax.random.PRNGKey(0),
        awb_enabled=awb_enabled,
    )


class TransformerAWBOps:
    def search_architecture(self, model, task_id, baseline_loss, dataloader_curr,
                            dataloader_exp, test_loader_curr, test_loader_exp, config, trainer=None):
        return self.get_model_architecture(model)

    def set_AB_matrices(self, model, original_arch, new_arch):
        if original_arch != new_arch:
            raise ValueError("Transformer architecture changes are not supported yet")
        return model.with_new_AB_matrices()

    def partition_for_AB_training(self, model):
        return model.partition_for_AB_training()

    def compute_V(self, model):
        return model.apply_V_transformation()

    def partition_for_standard_training(self, model):
        return model.partition_for_standard_training()

    def get_model_architecture(self, model):
        return {
            'seq_len': model.seq_len,
            'token_dim': model.input_dim,
            'embed_dim': model.embed_dim,
            'n_heads': model.n_heads,
            'mlp_dim': model.mlp_dim,
            'n_layers': model.n_layers,
        }

    def save_weights(self, model):
        paths = model._linear_paths()
        weights = [w_path(model) for w_path, _ in paths]
        biases = [b_path(model) for _, b_path in paths]
        return weights, biases

    def restore_weights(self, model, saved_weights):
        weights, biases = saved_weights
        paths = model._linear_paths()
        for i, (w_path, b_path) in enumerate(paths):
            model = eqx.tree_at(w_path, model, weights[i])
            model = eqx.tree_at(b_path, model, biases[i])
        return model
