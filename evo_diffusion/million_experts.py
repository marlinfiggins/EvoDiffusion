from collections.abc import Callable
from typing import Optional

import jax
import jax.numpy as jnp
from flax import linen as nn


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def rearrange_b_p_d(x, p):
    """
    Rearrange tensor from shape [batch_size, p * dim] to [batch_size, p, dim].

    Parameters:
    - x: Input tensor of shape [batch_size, p * dim]
    - p: The first split dimension

    Returns:
    - rearranged tensor of shape [batch_size, p, dim]
    """
    batch_size, p_dim = x.shape
    d = p_dim // p
    x = x.reshape(batch_size, p, d)
    return x

def rearrange_b_n_p_h_d(x, p, h):
    """
    Rearrange tensor from shape [batch_size, seq_len, p * h * dim] to [p, batch_size, seq_len, h, dim].
    
    Parameters:
    - x: Input tensor of shape [batch_size, seq_len, p * h * dim]
    - p: The first split dimension
    - h: The number of heads (second split dimension)
    
    Returns:
    - rearranged tensor of shape [p, batch_size, seq_len, h, dim]
    """
    batch_size, seq_len, _ = x.shape
    dim = x.shape[-1] // (p * h)
    x = x.reshape(batch_size, seq_len, p, h, dim)
    x = x.transpose(2, 0, 1, 3, 4)
    return x

class RMSNorm(nn.Module):
    dim: int

    @nn.compact
    def __call__(self, x):
        scale = self.dim ** 0.5
        gamma = self.param('gamma', nn.initializers.ones, (self.dim,))
        return nn.LayerNorm()(x) * scale * gamma

class ProductKeyMemory(nn.Module):
    dim: int
    num_keys: int

    @nn.compact
    def __call__(self, query):
        keys = self.param('keys', nn.initializers.normal(), (self.num_keys, self.dim // 2))
        query = rearrange_b_p_d(query, p=2)
        dots = jnp.einsum('bpd,kd->bpk', query, keys)
        return dots.reshape(query.shape[0], -1)

class PEER(nn.Module):
    dim: int
    heads: int = 8
    num_experts: int = 1_000_000
    num_experts_per_head: int = 16
    activation: Callable = nn.gelu
    dim_key: Optional[int] = None
    product_key_topk: Optional[int] = None
    separate_embed_per_head: bool = False
    pre_rmsnorm: bool = False
    dropout: float = 0.0

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        norm = RMSNorm(self.dim) if self.pre_rmsnorm else None
        if norm is not None:
            x = norm(x)

        heads = self.heads
        num_experts = self.num_experts
        separate_embed_per_head = self.separate_embed_per_head

        num_expert_sets = heads if separate_embed_per_head else 1

        weight_down_embed = self.param('weight_down_embed', nn.initializers.normal(), (num_experts * num_expert_sets, self.dim))
        weight_up_embed = self.param('weight_up_embed', nn.initializers.normal(), (num_experts * num_expert_sets, self.dim))

        assert (num_experts ** 0.5).is_integer(), '`num_experts` needs to be a square'
        assert (self.dim % 2) == 0, 'feature dimension should be divisible by 2'

        dim_key = default(self.dim_key, self.dim // 2)
        num_keys = int(num_experts ** 0.5)

        to_queries = nn.Sequential([
            nn.Dense(dim_key * heads * 2, use_bias=False),
            nn.BatchNorm(use_running_average=not deterministic, axis=-1),
            nn.Dropout(rate=self.dropout, deterministic=deterministic)
        ])

        self.product_key_topk = default(self.product_key_topk, self.num_experts_per_head)
        self.num_experts_per_head = self.product_key_topk

        keys = self.param('keys', nn.initializers.normal(), (heads, num_keys, 2, dim_key))

        x = to_queries(x)
        queries = rearrange_b_n_p_h_d(x, p=2, h=heads)

        sim = jnp.einsum('pbnhd,hkpd->pbnhk', queries, keys)
        (scores_x, scores_y), (indices_x, indices_y) = [jax.lax.top_k(s, self.product_key_topk) for s in sim]

        all_scores = scores_x[..., None] + scores_y[..., None, :]
        all_indices = indices_x[..., None] * num_keys + indices_y[..., None, :]

        all_scores = all_scores.reshape(*all_scores.shape[:-2], -1)
        all_indices = all_indices.reshape(*all_indices.shape[:-2], -1)

        scores, pk_indices = jax.lax.top_k(all_scores, self.num_experts_per_head)
        indices = jnp.take_along_axis(all_indices, pk_indices, axis=-1)

        if self.separate_embed_per_head:
            head_expert_offsets = jnp.arange(heads) * num_experts
            indices = indices + head_expert_offsets[..., None, None]

        weights_down = jnp.take(weight_down_embed, pk_indices, axis=0)
        weights_up = jnp.take(weight_up_embed, pk_indices, axis=0)

        x = jnp.einsum('bnd,bnhkd->bnhk', x, weights_down)

        x = self.activation(x)
        x = nn.Dropout(rate=self.dropout, deterministic=deterministic)(x)

        x = x * nn.softmax(scores, axis=-1)

        x = jnp.einsum('bnhk,bnhkd->bnd', x, weights_up)

        return x

class TransformerBlock(nn.Module):
    dim: int
    num_heads: int
    num_experts: int
    num_experts_per_head: int
    dropout: float = 0.1

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        attn_output = nn.MultiHeadDotProductAttention(
            num_heads=self.num_heads, deterministic=deterministic)(x, x, x)
        x = x + nn.Dropout(rate=self.dropout, deterministic=deterministic)(attn_output)
        x = nn.LayerNorm()(x)

        peer1 = PEER(self.dim, heads=self.num_heads, num_experts=self.num_experts, num_experts_per_head=self.num_experts_per_head)
        peer2 = PEER(self.dim, heads=self.num_heads, num_experts=self.num_experts, num_experts_per_head=self.num_experts_per_head)
        
        peer_output1 = peer1(x, deterministic=deterministic)
        peer_output2 = peer2(nn.gelu(peer_output1), deterministic=deterministic)
        x = x + nn.Dropout(rate=self.dropout, deterministic=deterministic)(peer_output2)
        x = nn.LayerNorm()(x)

        return x

class PEERLanguageModel(nn.Module):
    vocab_size: int
    dim: int
    num_layers: int
    num_heads: int
    num_experts: int
    top_k: int

    @nn.compact
    def __call__(self, x, deterministic: bool = True):
        b, s = x.shape
        positions = jnp.arange(s)[None, :]

        token_embedding = nn.Embed(self.vocab_size, self.dim)(x)
        position_embedding = nn.Embed(512, self.dim)(positions)
        x = token_embedding + position_embedding

        for _ in range(self.num_layers):
            x = TransformerBlock(self.dim, self.num_heads, self.num_experts, self.top_k)(x, deterministic=deterministic)

        x = nn.LayerNorm()(x)
        logits = nn.Dense(self.vocab_size)(x)
        return logits
