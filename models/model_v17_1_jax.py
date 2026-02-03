"""
DAWN v17.1-JAX: JAX/Flax port of model_v17_1_tpu_memopt.py

Native JAX implementation:
  - jax.checkpoint for feature/restore recompute (memory optimization)
  - flax.linen modules
  - Weight tying via nn.Embed.attend()
  - EMA usage tracking via mutable 'ema' state collection

Structure (same as PyTorch):
  SharedNeurons, UnifiedNeuronRouter, GlobalRouters,
  AttentionCircuit, KnowledgeCircuit, DAWNBlock, DAWN
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Dict, Any, List, Tuple


# ================================================================
# Recompute helpers (jax.checkpoint = jax.remat)
# ================================================================

@jax.checkpoint
def feature_fn(x, neurons, weights):
    """Feature: x [B,S,D] -> h [B,S,R]. Intermediate [B,S,N,R] not saved."""
    all_h = jnp.einsum('bsd,ndr->bsnr', x, neurons)
    return jnp.einsum('bsnr,bsn->bsr', all_h, weights)


@jax.checkpoint
def restore_fn(h, neurons, weights):
    """Restore: h [B,S,R] -> out [B,S,D]. Intermediate [B,S,N,D] not saved."""
    all_out = jnp.einsum('bsr,nrd->bsnd', h, neurons)
    return jnp.einsum('bsnd,bsn->bsd', all_out, weights)


# ================================================================
# Custom initializers
# ================================================================

def per_slice_orthogonal():
    """Orthogonal init applied independently to each slice of a 3D tensor [N, *, *]."""
    base = nn.initializers.orthogonal()
    def init(key, shape, dtype=jnp.float32):
        keys = jax.random.split(key, shape[0])
        return jax.vmap(lambda k: base(k, shape[1:], dtype))(keys)
    return init


def scaled_normal(scale=0.02):
    """Normal init with specified std."""
    def init(key, shape, dtype=jnp.float32):
        return jax.random.normal(key, shape, dtype) * scale
    return init


# ================================================================
# Utility functions
# ================================================================

def topk_sparsify(weights, k):
    """Top-k sparsification with renormalization.

    Args:
        weights: [..., N] softmax probabilities
        k: number of top elements to keep
    Returns:
        sparse_weights: [..., N] with only top-k nonzero, renormalized
        topk_idx: [..., k] indices
    """
    topk_vals, topk_idx = jax.lax.top_k(weights, k)
    N = weights.shape[-1]
    one_hot_mask = jax.nn.one_hot(topk_idx, N, dtype=weights.dtype)
    sparse = (one_hot_mask * topk_vals[..., jnp.newaxis]).sum(axis=-2)
    sparse = sparse / (sparse.sum(axis=-1, keepdims=True) + 1e-8)
    return sparse, topk_idx


# ================================================================
# UnifiedNeuronRouter
# ================================================================

class UnifiedNeuronRouter(nn.Module):
    """
    v17.1: 6 attention projections + 2 knowledge projections.
    Token-level routing via learned neuron embeddings.
    """
    n_feature_qk: int
    n_feature_v: int
    n_restore_qk: int
    n_restore_v: int
    n_feature_know: int
    n_restore_know: int
    d_space: int = 64
    dropout_rate: float = 0.1
    ema_alpha: float = 0.01

    def setup(self):
        total = (self.n_feature_qk + self.n_feature_v +
                 self.n_restore_qk + self.n_restore_v +
                 self.n_feature_know + self.n_restore_know)
        self._total_neurons = total

        self._feature_qk_end = self.n_feature_qk
        self._feature_v_end = self.n_feature_qk + self.n_feature_v
        self._restore_qk_end = self._feature_v_end + self.n_restore_qk
        self._restore_v_end = self._restore_qk_end + self.n_restore_v
        self._feature_know_end = self._restore_v_end + self.n_feature_know

        self.proj_all = nn.Dense(self.d_space * 6)
        self.proj_feature_know = nn.Dense(self.d_space)
        self.proj_restore_know = nn.Dense(self.d_space)
        self.dropout = nn.Dropout(rate=self.dropout_rate)

        self.neuron_emb = self.param('neuron_emb', scaled_normal(0.02),
                                     (total, self.d_space))

        # EMA usage tracking (mutable state, not gradient-tracked)
        self.ema_feature_q = self.variable('ema', 'usage_feature_q',
                                           jnp.zeros, (self.n_feature_qk,))
        self.ema_feature_k = self.variable('ema', 'usage_feature_k',
                                           jnp.zeros, (self.n_feature_qk,))
        self.ema_feature_v = self.variable('ema', 'usage_feature_v',
                                           jnp.zeros, (self.n_feature_v,))
        self.ema_restore_q = self.variable('ema', 'usage_restore_q',
                                           jnp.zeros, (self.n_restore_qk,))
        self.ema_restore_k = self.variable('ema', 'usage_restore_k',
                                           jnp.zeros, (self.n_restore_qk,))
        self.ema_restore_v = self.variable('ema', 'usage_restore_v',
                                           jnp.zeros, (self.n_restore_v,))
        self.ema_feature_know = self.variable('ema', 'usage_feature_know',
                                              jnp.zeros, (self.n_feature_know,))
        self.ema_restore_know = self.variable('ema', 'usage_restore_know',
                                              jnp.zeros, (self.n_restore_know,))

    def _normalize_emb(self):
        return self.neuron_emb / (jnp.linalg.norm(self.neuron_emb, axis=-1, keepdims=True) + 1e-8)

    def get_knowledge_logits(self, x, deterministic=False):
        emb_norm = self._normalize_emb()

        h_fk = self.dropout(self.proj_feature_know(x), deterministic=deterministic)
        emb_fk = emb_norm[self._restore_v_end:self._feature_know_end]
        logits_fk = jnp.einsum('bsd,nd->bsn', h_fk, emb_fk)

        h_rk = self.dropout(self.proj_restore_know(x), deterministic=deterministic)
        emb_rk = emb_norm[self._feature_know_end:]
        logits_rk = jnp.einsum('bsd,nd->bsn', h_rk, emb_rk)

        return logits_fk, logits_rk

    def get_all_logits(self, x, deterministic=False):
        emb_norm = self._normalize_emb()
        all_proj = self.dropout(self.proj_all(x), deterministic=deterministic)
        h_fqk_Q, h_fqk_K, h_fv, h_rqk_Q, h_rqk_K, h_rv = jnp.split(all_proj, 6, axis=-1)

        fqk_emb = emb_norm[:self._feature_qk_end]
        fv_emb = emb_norm[self._feature_qk_end:self._feature_v_end]
        rqk_emb = emb_norm[self._feature_v_end:self._restore_qk_end]
        rv_emb = emb_norm[self._restore_qk_end:self._restore_v_end]

        logits_fqk_Q = jnp.einsum('bsd,nd->bsn', h_fqk_Q, fqk_emb)
        logits_fqk_K = jnp.einsum('bsd,nd->bsn', h_fqk_K, fqk_emb)
        logits_fv = jnp.einsum('bsd,nd->bsn', h_fv, fv_emb)
        logits_rqk_Q = jnp.einsum('bsd,nd->bsn', h_rqk_Q, rqk_emb)
        logits_rqk_K = jnp.einsum('bsd,nd->bsn', h_rqk_K, rqk_emb)
        logits_rv = jnp.einsum('bsd,nd->bsn', h_rv, rv_emb)

        return logits_fqk_Q, logits_fqk_K, logits_fv, logits_rqk_Q, logits_rqk_K, logits_rv

    def update_usage(self, weights, neuron_type, attention_mask=None, deterministic=False):
        if deterministic:
            return
        if weights.ndim == 3:
            active = (weights > 0).astype(jnp.float32)
            if attention_mask is not None:
                mask = attention_mask[..., jnp.newaxis].astype(jnp.float32)
                active = active * mask
                count = mask.sum() + 1e-8
                usage = active.sum(axis=(0, 1)) / count
            else:
                usage = active.mean(axis=(0, 1))
        else:
            usage = (weights > 0).astype(jnp.float32).mean(axis=0)

        decay = 1.0 - self.ema_alpha
        ema_var = getattr(self, f'ema_{neuron_type}')
        ema_var.value = ema_var.value * decay + usage * self.ema_alpha


# ================================================================
# SharedNeurons
# ================================================================

class SharedNeurons(nn.Module):
    """Shared neuron pools for attention and knowledge circuits."""
    d_model: int
    rank: int
    n_feature_qk: int
    n_feature_v: int
    n_restore_qk: int
    n_restore_v: int
    n_feature_know: int
    n_restore_know: int
    knowledge_rank: int = 128

    def setup(self):
        orth_init = per_slice_orthogonal()
        self.f_neurons = self.param(
            'f_neurons', orth_init,
            (self.n_feature_qk + self.n_feature_v, self.d_model, self.rank))
        self.r_neurons = self.param(
            'r_neurons', orth_init,
            (self.n_restore_qk + self.n_restore_v, self.rank, self.d_model))
        self.feature_know = self.param(
            'feature_know', orth_init,
            (self.n_feature_know, self.d_model, self.knowledge_rank))
        self.restore_know = self.param(
            'restore_know', orth_init,
            (self.n_restore_know, self.knowledge_rank, self.d_model))

    @property
    def feature_qk_neurons(self):
        return self.f_neurons[:self.n_feature_qk]

    @property
    def feature_v_neurons(self):
        return self.f_neurons[self.n_feature_qk:]

    @property
    def restore_qk_neurons(self):
        return self.r_neurons[:self.n_restore_qk]

    @property
    def restore_v_neurons(self):
        return self.r_neurons[self.n_restore_qk:]


# ================================================================
# GlobalRouters
# ================================================================

class GlobalRouters(nn.Module):
    """Top-k routing for attention and knowledge circuits."""
    d_model: int
    n_feature_qk: int
    n_feature_v: int
    n_restore_qk: int
    n_restore_v: int
    n_feature_know: int
    n_restore_know: int
    top_k_feature_qk: int = 8
    top_k_feature_v: int = 8
    top_k_restore_qk: int = 8
    top_k_restore_v: int = 8
    top_k_feature_know: int = 4
    top_k_restore_know: int = 4
    d_space: int = 64
    router_dropout: float = 0.1

    def setup(self):
        self.neuron_router = UnifiedNeuronRouter(
            n_feature_qk=self.n_feature_qk,
            n_feature_v=self.n_feature_v,
            n_restore_qk=self.n_restore_qk,
            n_restore_v=self.n_restore_v,
            n_feature_know=self.n_feature_know,
            n_restore_know=self.n_restore_know,
            d_space=self.d_space,
            dropout_rate=self.router_dropout,
        )

    def get_attention_weights(self, x, attention_mask=None,
                              deterministic=False, return_routing_info=False):
        (fqk_logits_Q, fqk_logits_K, fv_logits,
         rqk_logits_Q, rqk_logits_K, rv_logits) = self.neuron_router.get_all_logits(
            x, deterministic=deterministic)

        fqk_pref_Q = jax.nn.softmax(fqk_logits_Q, axis=-1)
        fqk_pref_K = jax.nn.softmax(fqk_logits_K, axis=-1)
        fv_pref = jax.nn.softmax(fv_logits, axis=-1)
        rqk_pref_Q = jax.nn.softmax(rqk_logits_Q, axis=-1)
        rqk_pref_K = jax.nn.softmax(rqk_logits_K, axis=-1)
        rv_pref = jax.nn.softmax(rv_logits, axis=-1)

        # Load-balancing aux loss
        aux_loss = jnp.float32(0.0)
        if not deterministic:
            if attention_mask is not None:
                mask = attention_mask[..., jnp.newaxis].astype(jnp.float32)
                count = mask.sum() + 1e-8
                usage_fqk_Q = (fqk_pref_Q * mask).sum(axis=(0, 1)) / count
                usage_fqk_K = (fqk_pref_K * mask).sum(axis=(0, 1)) / count
                usage_fv = (fv_pref * mask).sum(axis=(0, 1)) / count
                usage_rqk_Q = (rqk_pref_Q * mask).sum(axis=(0, 1)) / count
                usage_rqk_K = (rqk_pref_K * mask).sum(axis=(0, 1)) / count
                usage_rv = (rv_pref * mask).sum(axis=(0, 1)) / count
            else:
                usage_fqk_Q = fqk_pref_Q.mean(axis=(0, 1))
                usage_fqk_K = fqk_pref_K.mean(axis=(0, 1))
                usage_fv = fv_pref.mean(axis=(0, 1))
                usage_rqk_Q = rqk_pref_Q.mean(axis=(0, 1))
                usage_rqk_K = rqk_pref_K.mean(axis=(0, 1))
                usage_rv = rv_pref.mean(axis=(0, 1))

            target_fqk = 1.0 / self.n_feature_qk
            target_fv = 1.0 / self.n_feature_v
            target_rqk = 1.0 / self.n_restore_qk
            target_rv = 1.0 / self.n_restore_v

            aux_loss = aux_loss + ((usage_fqk_Q - target_fqk) ** 2).sum() * self.n_feature_qk
            aux_loss = aux_loss + ((usage_fqk_K - target_fqk) ** 2).sum() * self.n_feature_qk
            aux_loss = aux_loss + ((usage_fv - target_fv) ** 2).sum() * self.n_feature_v
            aux_loss = aux_loss + ((usage_rqk_Q - target_rqk) ** 2).sum() * self.n_restore_qk
            aux_loss = aux_loss + ((usage_rqk_K - target_rqk) ** 2).sum() * self.n_restore_qk
            aux_loss = aux_loss + ((usage_rv - target_rv) ** 2).sum() * self.n_restore_v

        # Top-k sparsification
        fqk_weights_Q, _ = topk_sparsify(fqk_pref_Q, self.top_k_feature_qk)
        fqk_weights_K, _ = topk_sparsify(fqk_pref_K, self.top_k_feature_qk)
        fv_weights, _ = topk_sparsify(fv_pref, self.top_k_feature_v)
        rqk_weights_Q, _ = topk_sparsify(rqk_pref_Q, self.top_k_restore_qk)
        rqk_weights_K, _ = topk_sparsify(rqk_pref_K, self.top_k_restore_qk)
        rv_weights, _ = topk_sparsify(rv_pref, self.top_k_restore_v)

        # EMA usage update (training only)
        if not deterministic:
            self.neuron_router.update_usage(fqk_weights_Q, 'feature_q',
                                            attention_mask, deterministic)
            self.neuron_router.update_usage(fqk_weights_K, 'feature_k',
                                            attention_mask, deterministic)
            self.neuron_router.update_usage(fv_weights, 'feature_v',
                                            attention_mask, deterministic)
            self.neuron_router.update_usage(rqk_weights_Q, 'restore_q',
                                            attention_mask, deterministic)
            self.neuron_router.update_usage(rqk_weights_K, 'restore_k',
                                            attention_mask, deterministic)
            self.neuron_router.update_usage(rv_weights, 'restore_v',
                                            attention_mask, deterministic)

        routing_info = None
        if return_routing_info:
            routing_info = {
                'fqk_weights_Q': jax.lax.stop_gradient(fqk_weights_Q),
                'fqk_weights_K': jax.lax.stop_gradient(fqk_weights_K),
                'fv_weights': jax.lax.stop_gradient(fv_weights),
                'rqk_weights_Q': jax.lax.stop_gradient(rqk_weights_Q),
                'rqk_weights_K': jax.lax.stop_gradient(rqk_weights_K),
                'rv_weights': jax.lax.stop_gradient(rv_weights),
                'fqk_q_pref': jax.lax.stop_gradient(fqk_pref_Q),
                'fqk_k_pref': jax.lax.stop_gradient(fqk_pref_K),
                'fv_pref': jax.lax.stop_gradient(fv_pref),
                'rqk_q_pref': jax.lax.stop_gradient(rqk_pref_Q),
                'rqk_k_pref': jax.lax.stop_gradient(rqk_pref_K),
                'rv_pref': jax.lax.stop_gradient(rv_pref),
                'token_routing': True,
            }

        return (fqk_weights_Q, fqk_weights_K, fv_weights,
                rqk_weights_Q, rqk_weights_K, rv_weights,
                routing_info, aux_loss)

    def get_knowledge_weights(self, x, attention_mask=None,
                              deterministic=False, return_routing_info=False):
        logits_f, logits_r = self.neuron_router.get_knowledge_logits(
            x, deterministic=deterministic)
        pref_f = jax.nn.softmax(logits_f, axis=-1)
        pref_r = jax.nn.softmax(logits_r, axis=-1)

        aux_loss = jnp.float32(0.0)
        if not deterministic:
            if attention_mask is not None:
                mask = attention_mask[..., jnp.newaxis].astype(jnp.float32)
                count = mask.sum() + 1e-8
                usage_f = (pref_f * mask).sum(axis=(0, 1)) / count
                usage_r = (pref_r * mask).sum(axis=(0, 1)) / count
            else:
                usage_f = pref_f.mean(axis=(0, 1))
                usage_r = pref_r.mean(axis=(0, 1))

            target_f = 1.0 / self.n_feature_know
            target_r = 1.0 / self.n_restore_know
            aux_loss = aux_loss + ((usage_f - target_f) ** 2).sum() * self.n_feature_know
            aux_loss = aux_loss + ((usage_r - target_r) ** 2).sum() * self.n_restore_know

        feature_know_w, _ = topk_sparsify(pref_f, self.top_k_feature_know)
        restore_know_w, _ = topk_sparsify(pref_r, self.top_k_restore_know)

        if not deterministic:
            self.neuron_router.update_usage(feature_know_w, 'feature_know',
                                            attention_mask, deterministic)
            self.neuron_router.update_usage(restore_know_w, 'restore_know',
                                            attention_mask, deterministic)

        routing_info = None
        if return_routing_info:
            routing_info = {
                'feature_know_w': jax.lax.stop_gradient(feature_know_w),
                'restore_know_w': jax.lax.stop_gradient(restore_know_w),
                'feature_know_pref': jax.lax.stop_gradient(pref_f),
                'restore_know_pref': jax.lax.stop_gradient(pref_r),
            }

        return feature_know_w, restore_know_w, routing_info, aux_loss


# ================================================================
# AttentionCircuit
# ================================================================

class AttentionCircuit(nn.Module):
    """
    v17.1 attention with jax.checkpoint recompute.

    Feature + Restore both use jax.checkpoint so intermediate
    tensors [B,S,N,R] and [B,S,N,D] are not saved for backward.
    QK shared pool: x @ f_qk computed separately for Q and K.
    """
    d_model: int
    n_heads: int
    rank: int
    dropout_rate: float = 0.1

    def setup(self):
        self.d_head = self.d_model // self.n_heads
        self.expand_O = nn.Dense(self.d_model, use_bias=False)
        self.out_dropout = nn.Dropout(rate=self.dropout_rate)

    def __call__(self, x, shared_neurons, fqk_weights_Q, fqk_weights_K,
                 fv_weights, rqk_weights_Q, rqk_weights_K, rv_weights,
                 attention_mask=None, deterministic=False):
        B, S, D = x.shape

        # === Feature: jax.checkpoint recompute ===
        f_qk = shared_neurons.feature_qk_neurons               # [N_fqk, D, R]
        h_q = feature_fn(x, f_qk, fqk_weights_Q)               # [B, S, R]
        h_k = feature_fn(x, f_qk, fqk_weights_K)               # [B, S, R]

        f_v = shared_neurons.feature_v_neurons                  # [N_fv, D, R]
        h_v = feature_fn(x, f_v, fv_weights)                    # [B, S, R]

        # === Restore: jax.checkpoint recompute ===
        r_qk = shared_neurons.restore_qk_neurons               # [N_rqk, R, D]
        Q = restore_fn(h_q, r_qk, rqk_weights_Q)               # [B, S, D]
        K = restore_fn(h_k, r_qk, rqk_weights_K)               # [B, S, D]

        r_v = shared_neurons.restore_v_neurons                  # [N_rv, R, D]
        V = restore_fn(h_v, r_v, rv_weights)                    # [B, S, D]

        # Multi-head attention
        Q = Q.reshape(B, S, self.n_heads, self.d_head).transpose(0, 2, 1, 3)  # [B,H,S,Dh]
        K = K.reshape(B, S, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        V = V.reshape(B, S, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

        scale = jnp.sqrt(jnp.float32(self.d_head))
        scores = jnp.einsum('bhsd,bhtd->bhst', Q, K) / scale

        # Causal mask
        causal_mask = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
        scores = jnp.where(causal_mask, scores, jnp.finfo(scores.dtype).min)

        attn_weights = jax.nn.softmax(scores, axis=-1)

        # Attention dropout
        if not deterministic and self.dropout_rate > 0.0:
            dropout_rng = self.make_rng('dropout')
            keep = jax.random.bernoulli(dropout_rng, 1.0 - self.dropout_rate,
                                        attn_weights.shape)
            attn_weights = jnp.where(keep, attn_weights / (1.0 - self.dropout_rate), 0.0)

        attn_out = jnp.einsum('bhst,bhtd->bhsd', attn_weights, V)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, S, D)

        output = self.expand_O(attn_out)
        output = self.out_dropout(output, deterministic=deterministic)
        return output


# ================================================================
# KnowledgeCircuit
# ================================================================

class KnowledgeCircuit(nn.Module):
    """
    v17.1 knowledge circuit with jax.checkpoint recompute.
    Feature + Restore intermediate tensors not saved.
    """
    d_model: int
    n_feature_know: int
    n_restore_know: int
    knowledge_rank: int
    top_k_feature_know: int = 4
    top_k_restore_know: int = 4
    dropout_rate: float = 0.1

    def setup(self):
        self.dropout = nn.Dropout(rate=self.dropout_rate)

    def __call__(self, x, shared_neurons, feature_know_w, restore_know_w,
                 attention_mask=None, deterministic=False):
        # Feature: jax.checkpoint recompute
        f_know = shared_neurons.feature_know                    # [N_f, D, KR]
        h = feature_fn(x, f_know, feature_know_w)               # [B, S, KR]

        # Restore: jax.checkpoint recompute
        r_know = shared_neurons.restore_know                    # [N_r, KR, D]
        output = restore_fn(h, r_know, restore_know_w)           # [B, S, D]

        return self.dropout(output, deterministic=deterministic)


# ================================================================
# DAWNBlock
# ================================================================

class DAWNBlock(nn.Module):
    """Single transformer block: attention + knowledge with pre-norm."""
    d_model: int
    n_heads: int
    rank: int
    n_feature_know: int
    n_restore_know: int
    knowledge_rank: int = 128
    top_k_feature_know: int = 4
    top_k_restore_know: int = 4
    dropout_rate: float = 0.1

    def setup(self):
        self.attn = AttentionCircuit(
            d_model=self.d_model, n_heads=self.n_heads,
            rank=self.rank, dropout_rate=self.dropout_rate)
        self.knowledge = KnowledgeCircuit(
            d_model=self.d_model,
            n_feature_know=self.n_feature_know,
            n_restore_know=self.n_restore_know,
            knowledge_rank=self.knowledge_rank,
            top_k_feature_know=self.top_k_feature_know,
            top_k_restore_know=self.top_k_restore_know,
            dropout_rate=self.dropout_rate)
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()

    def __call__(self, x, shared_neurons, router, attention_mask=None,
                 deterministic=False, return_routing_info=False):
        # Attention
        normed_x = self.norm1(x)
        (fqk_w_Q, fqk_w_K, fv_w, rqk_w_Q, rqk_w_K, rv_w,
         attn_routing_info, aux_loss) = router.get_attention_weights(
            normed_x, attention_mask, deterministic, return_routing_info)
        attn_out = self.attn(
            normed_x, shared_neurons,
            fqk_w_Q, fqk_w_K, fv_w, rqk_w_Q, rqk_w_K, rv_w,
            attention_mask, deterministic)
        x = x + attn_out

        # Knowledge
        normed_x = self.norm2(x)
        feature_know_w, restore_know_w, know_routing_info, know_aux_loss = \
            router.get_knowledge_weights(
                normed_x, attention_mask, deterministic, return_routing_info)
        know_out = self.knowledge(
            normed_x, shared_neurons,
            feature_know_w, restore_know_w,
            attention_mask, deterministic)
        x = x + know_out

        routing_info = None
        if return_routing_info:
            routing_info = {
                'attention': attn_routing_info,
                'knowledge': know_routing_info,
                'attn_out_norm': jax.lax.stop_gradient(
                    jnp.linalg.norm(attn_out, axis=-1).mean()),
                'know_out_norm': jax.lax.stop_gradient(
                    jnp.linalg.norm(know_out, axis=-1).mean()),
            }

        return x, routing_info, aux_loss + know_aux_loss


# ================================================================
# DAWN
# ================================================================

class DAWN(nn.Module):
    """
    DAWN v17.1-JAX

    JAX/Flax port of v17.1-TPU-MemOpt.
    - jax.checkpoint for feature/restore intermediate tensor recompute
    - Weight tying via nn.Embed.attend()
    - Functional API: returns dict with loss, logits, aux_loss

    Usage:
        model = DAWN(vocab_size=30522, d_model=384, ...)
        variables = model.init(rng, input_ids)
        output = model.apply(variables, input_ids, labels=labels,
                             deterministic=False,
                             rngs={'dropout': dropout_rng},
                             mutable=['ema'])
    """
    __version__ = "17.1-JAX"

    vocab_size: int = 30000
    d_model: int = 320
    n_layers: int = 4
    n_heads: int = 8
    rank: int = 64
    max_seq_len: int = 512
    d_space: int = 64
    n_feature_qk: int = 56
    n_feature_v: int = 24
    top_k_feature_qk: int = 16
    top_k_feature_v: int = 6
    n_restore_qk: int = 56
    n_restore_v: int = 24
    top_k_restore_qk: int = 16
    top_k_restore_v: int = 6
    n_feature_know: int = 24
    n_restore_know: int = 24
    top_k_feature_know: int = 4
    top_k_restore_know: int = 4
    knowledge_rank: int = 128
    dropout_rate: float = 0.1
    router_dropout: float = 0.1
    gradient_checkpointing: bool = False

    def setup(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by n_heads ({self.n_heads})")

        self.token_emb = nn.Embed(self.vocab_size, self.d_model,
                                  embedding_init=scaled_normal(0.02))
        self.pos_emb = nn.Embed(self.max_seq_len, self.d_model,
                                embedding_init=scaled_normal(0.02))
        self.emb_dropout = nn.Dropout(rate=self.dropout_rate)

        self.shared_neurons = SharedNeurons(
            d_model=self.d_model, rank=self.rank,
            n_feature_qk=self.n_feature_qk, n_feature_v=self.n_feature_v,
            n_restore_qk=self.n_restore_qk, n_restore_v=self.n_restore_v,
            n_feature_know=self.n_feature_know,
            n_restore_know=self.n_restore_know,
            knowledge_rank=self.knowledge_rank)

        self.router = GlobalRouters(
            d_model=self.d_model,
            n_feature_qk=self.n_feature_qk, n_feature_v=self.n_feature_v,
            n_restore_qk=self.n_restore_qk, n_restore_v=self.n_restore_v,
            n_feature_know=self.n_feature_know,
            n_restore_know=self.n_restore_know,
            top_k_feature_qk=self.top_k_feature_qk,
            top_k_feature_v=self.top_k_feature_v,
            top_k_restore_qk=self.top_k_restore_qk,
            top_k_restore_v=self.top_k_restore_v,
            top_k_feature_know=self.top_k_feature_know,
            top_k_restore_know=self.top_k_restore_know,
            d_space=self.d_space, router_dropout=self.router_dropout)

        block_cls = DAWNBlock
        if self.gradient_checkpointing:
            block_cls = nn.remat(DAWNBlock)

        self.layers = [
            block_cls(
                d_model=self.d_model, n_heads=self.n_heads,
                rank=self.rank,
                n_feature_know=self.n_feature_know,
                n_restore_know=self.n_restore_know,
                knowledge_rank=self.knowledge_rank,
                top_k_feature_know=self.top_k_feature_know,
                top_k_restore_know=self.top_k_restore_know,
                dropout_rate=self.dropout_rate,
                name=f'block_{i}')
            for i in range(self.n_layers)
        ]

        self.norm = nn.LayerNorm()
        # lm_head uses weight tying via token_emb.attend()

    def __call__(self, input_ids, labels=None, attention_mask=None,
                 deterministic=False, return_routing_info=False):
        B, S = input_ids.shape
        if S > self.max_seq_len:
            raise ValueError(
                f"Sequence length {S} exceeds max_seq_len {self.max_seq_len}")

        positions = jnp.arange(S)[jnp.newaxis, :]  # [1, S]
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        x = self.emb_dropout(x, deterministic=deterministic)

        routing_infos = [] if return_routing_info else None
        total_aux_loss = jnp.float32(0.0)

        for layer in self.layers:
            x, routing_info, aux_loss = layer(
                x, self.shared_neurons, self.router,
                attention_mask, deterministic, return_routing_info)
            if routing_infos is not None:
                routing_infos.append(routing_info)
            total_aux_loss = total_aux_loss + aux_loss

        x = self.norm(x)

        # Weight-tied lm_head: logits = x @ embedding.T
        logits = self.token_emb.attend(x)

        result = {
            'logits': logits,
            'aux_loss': total_aux_loss,
        }

        if labels is not None:
            shift_logits = logits[:, :-1, :]
            shift_labels = labels[:, 1:].astype(jnp.int32)
            # Cross-entropy with ignore_index=-100
            valid_mask = (shift_labels != -100)
            safe_labels = jnp.where(valid_mask, shift_labels, 0)
            log_probs = jax.nn.log_softmax(shift_logits, axis=-1)
            token_losses = -jnp.take_along_axis(
                log_probs, safe_labels[..., jnp.newaxis], axis=-1).squeeze(-1)
            token_losses = token_losses * valid_mask
            loss = token_losses.sum() / (valid_mask.sum() + 1e-8)
            result['loss'] = loss

        if return_routing_info:
            result['routing_infos'] = routing_infos

        return result

    # ------------------------------------------------------------------
    # Auxiliary losses (computed from parameters, no forward needed)
    # ------------------------------------------------------------------

    def orthogonality_loss(self):
        """Enforce orthogonality of neuron matrices."""
        I_rank = jnp.eye(self.rank)[jnp.newaxis]
        I_know = jnp.eye(self.knowledge_rank)[jnp.newaxis]

        W_fqk = self.shared_neurons.feature_qk_neurons  # [N, D, R]
        WtW_fqk = jnp.matmul(
            W_fqk.transpose(0, 2, 1), W_fqk)               # [N, R, R]
        loss_fqk = ((WtW_fqk - I_rank) ** 2).mean()

        W_fv = self.shared_neurons.feature_v_neurons
        WtW_fv = jnp.matmul(W_fv.transpose(0, 2, 1), W_fv)
        loss_fv = ((WtW_fv - I_rank) ** 2).mean()

        W_rqk = self.shared_neurons.restore_qk_neurons      # [N, R, D]
        WWt_rqk = jnp.matmul(W_rqk, W_rqk.transpose(0, 2, 1))  # [N, R, R]
        loss_rqk = ((WWt_rqk - I_rank) ** 2).mean()

        W_rv = self.shared_neurons.restore_v_neurons
        WWt_rv = jnp.matmul(W_rv, W_rv.transpose(0, 2, 1))
        loss_rv = ((WWt_rv - I_rank) ** 2).mean()

        W_fknow = self.shared_neurons.feature_know           # [N, D, KR]
        WtW_fknow = jnp.matmul(
            W_fknow.transpose(0, 2, 1), W_fknow)            # [N, KR, KR]
        loss_fknow = ((WtW_fknow - I_know) ** 2).mean()

        W_rknow = self.shared_neurons.restore_know           # [N, KR, D]
        WWt_rknow = jnp.matmul(
            W_rknow, W_rknow.transpose(0, 2, 1))            # [N, KR, KR]
        loss_rknow = ((WWt_rknow - I_know) ** 2).mean()

        return (loss_fqk + loss_fv + loss_rqk + loss_rv +
                loss_fknow + loss_rknow) / 6

    def knowledge_diversity_loss(self):
        """Encourage diversity among knowledge neurons."""
        feat_know = self.shared_neurons.feature_know
        feat_flat = feat_know.reshape(feat_know.shape[0], -1)
        feat_norm = feat_flat / (jnp.linalg.norm(feat_flat, axis=-1, keepdims=True) + 1e-8)
        feat_sim = jnp.matmul(feat_norm, feat_norm.T)
        mask_f = ~jnp.eye(feat_sim.shape[0], dtype=jnp.bool_)
        feat_loss = jnp.abs(feat_sim * mask_f).sum() / mask_f.sum()

        rest_know = self.shared_neurons.restore_know
        rest_flat = rest_know.reshape(rest_know.shape[0], -1)
        rest_norm = rest_flat / (jnp.linalg.norm(rest_flat, axis=-1, keepdims=True) + 1e-8)
        rest_sim = jnp.matmul(rest_norm, rest_norm.T)
        mask_r = ~jnp.eye(rest_sim.shape[0], dtype=jnp.bool_)
        rest_loss = jnp.abs(rest_sim * mask_r).sum() / mask_r.sum()

        return (feat_loss + rest_loss) / 2

    def get_auxiliary_losses(self):
        return {
            'orth_total': self.orthogonality_loss(),
            'knowledge_div': self.knowledge_diversity_loss(),
        }

    # ------------------------------------------------------------------
    # Config / info
    # ------------------------------------------------------------------

    def get_config(self):
        return {
            'model_version': self.__version__,
            'vocab_size': self.vocab_size, 'd_model': self.d_model,
            'n_layers': self.n_layers, 'n_heads': self.n_heads,
            'rank': self.rank, 'knowledge_rank': self.knowledge_rank,
            'max_seq_len': self.max_seq_len,
            'n_feature_qk': self.n_feature_qk, 'n_feature_v': self.n_feature_v,
            'top_k_feature_qk': self.top_k_feature_qk,
            'top_k_feature_v': self.top_k_feature_v,
            'n_restore_qk': self.n_restore_qk, 'n_restore_v': self.n_restore_v,
            'top_k_restore_qk': self.top_k_restore_qk,
            'top_k_restore_v': self.top_k_restore_v,
            'n_feature_know': self.n_feature_know,
            'n_restore_know': self.n_restore_know,
            'top_k_feature_know': self.top_k_feature_know,
            'top_k_restore_know': self.top_k_restore_know,
            'd_space': self.d_space,
        }

    def get_model_info(self):
        return [
            f"DAWN v{self.__version__}: Q/K Shared + Knowledge Feature-Restore (JAX)",
            f"  d_model={self.d_model}, n_layers={self.n_layers}, n_heads={self.n_heads}",
            f"  rank={self.rank}, knowledge_rank={self.knowledge_rank}",
            f"  max_seq_len={self.max_seq_len}, dropout={self.dropout_rate}",
            f"",
            f"  [Attention - Q/K Shared Pool]",
            f"  Feature_QK: {self.n_feature_qk} x {self.d_model} x {self.rank} (top-k={self.top_k_feature_qk})",
            f"  Feature_V: {self.n_feature_v} x {self.d_model} x {self.rank} (top-k={self.top_k_feature_v})",
            f"  Restore_QK: {self.n_restore_qk} x {self.rank} x {self.d_model} (top-k={self.top_k_restore_qk})",
            f"  Restore_V: {self.n_restore_v} x {self.rank} x {self.d_model} (top-k={self.top_k_restore_v})",
            f"",
            f"  [Knowledge - Feature-Restore]",
            f"  Feature_Know: {self.n_feature_know} x {self.d_model} x {self.knowledge_rank} (top-k={self.top_k_feature_know})",
            f"  Restore_Know: {self.n_restore_know} x {self.knowledge_rank} x {self.d_model} (top-k={self.top_k_restore_know})",
            f"",
            f"  [Memory Optimization]",
            f"  jax.checkpoint: ON (Feature + Restore intermediate recompute)",
            f"",
            f"  [Router]",
            f"  d_space={self.d_space}, router_dropout={self.router_dropout}",
            f"  token_routing=True (SSM removed)",
            f"",
            f"  [Other]",
            f"  gradient_checkpointing={self.gradient_checkpointing}",
        ]
