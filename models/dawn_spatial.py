"""
DAWN-Spatial: Rank-1 Neuron Architecture with Hierarchical Routing (JAX/Flax)

Key concept:
  - Every neuron is a single vector v_i [D] (rank-1).
  - Sense: activation_i = v_i · x  (scalar)
  - Fire:  gate decides which neurons fire (threshold tau)
  - Emit:  out = Σ gate_i * activation_i * v_i
  - Equivalent to: out = V^T diag(gate) V x  (sum of rank-1 outer products)
  - Complex transforms emerge from multi-layer residual, not single neuron.

Hierarchical Routing (2-stage):
  Stage 1 (Coarse): Select top-k clusters via cluster embeddings
  Stage 2 (Fine):   Route within selected clusters via neuron embeddings
  Complexity: O(n_clusters + k_cluster × cluster_size) << O(N)

Architecture:
  NeuronPool   — shared [N, D] vectors + cluster embeddings [n_clusters, d_space]
  Router       — hierarchical: cluster selection → neuron routing → threshold tau
  sense_emit   — checkpointed rank-1 transform: V^T diag(gate) V x
  AttentionCircuit — Q/K/V via hierarchical sense_emit + causal self-attention
  KnowledgeCircuit — FFN-equivalent via hierarchical sense_emit
  DAWNBlock    — norm → attn → residual → norm → knowledge → residual
  DAWN         — embedding + jax.lax.scan layer loop + weight-tied lm_head

Reference:
  - model_v17_1_jax.py: scan, checkpoint, cached forward, weight tying
  - model_v18_5.py: threshold/tau gate mechanism
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Dict


# ================================================================
# 1. Trace-safe dropout (from v17.1 — compatible with nn.remat)
# ================================================================

def safe_dropout(x, rate, deterministic, rng):
    """Dropout that is fully trace-safe — no Python bool branch."""
    if rate == 0.0:
        return x
    keep_rate = 1.0 - rate
    mask = jax.random.bernoulli(rng, keep_rate, x.shape)
    mask = jnp.where(deterministic, jnp.ones_like(mask), mask)
    return jnp.where(mask, x / keep_rate, 0.0)


# ================================================================
# 2. Pure functional LayerNorm (from v17.1)
# ================================================================

def _layer_norm(x, scale, bias, eps=1e-6):
    """Pure functional LayerNorm (matches Flax nn.LayerNorm)."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * scale + bias


# ================================================================
# 3. Initializers
# ================================================================

def scaled_normal(scale=0.02):
    """Normal init with specified std."""
    def init(key, shape, dtype=jnp.float32):
        return jax.random.normal(key, shape, dtype) * scale
    return init


def unit_norm_init(scale=1.0):
    """Initialize each row as a unit-norm random vector.

    Shape: [N, D] — each of N vectors is independently normalized.
    """
    def init(key, shape, dtype=jnp.float32):
        x = jax.random.normal(key, shape, dtype)
        norms = jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8
        return x / norms * scale
    return init


# ================================================================
# 4. Threshold gate (v18.5 tau mechanism → JAX)
# ================================================================

def threshold_gate(scores, tau, max_k=None):
    """Threshold-based gating with dead-neuron gradient flow.

    Adapted from v18.5 _topk_select_and_chunk.

    Args:
        scores: [B, S, N] routing scores (logits)
        tau:    [B, S, 1] input-dependent threshold
        max_k:  optional safety cap on active neuron count

    Returns:
        gates: [B, S, N] sparse gate values (normalized)
    """
    raw_gate = scores - tau

    # Positive: pass through. Negative: tiny gradient flow.
    gate = jnp.where(raw_gate > 0, raw_gate, 1e-8 * jnp.exp(raw_gate))

    # Exponential scaling (amplifies differences, 0 when gate=0)
    exp_gate = jnp.exp(gate) - 1.0

    # max_k safety cap: keep only top-max_k neurons
    if max_k is not None:
        topk_vals, _ = jax.lax.top_k(exp_gate, max_k)
        threshold = topk_vals[:, :, -1:]  # k-th largest value
        exp_gate = jnp.where(exp_gate >= threshold, exp_gate, 0.0)

    # Normalize: ratio * gate_strength
    gate_sum = exp_gate.sum(axis=-1, keepdims=True) + 1e-8
    gate_strength = jnp.tanh(exp_gate.max(axis=-1, keepdims=True))
    return (exp_gate / gate_sum) * gate_strength


# ================================================================
# 5. Rank-1 Sense-Emit (checkpointed core operation)
# ================================================================

@jax.checkpoint
def sense_emit(x, neurons, gates):
    """Rank-1 sense + emit with gradient checkpoint.

    x:       [B, S, D]
    neurons: [N, D]     — rank-1 neuron vectors
    gates:   [B, S, N]  — sparse gate values

    Returns: [B, S, D]

    Computes: out = V^T diag(gate * (Vx)) V  applied to x
      1) sense:  activations = x @ V^T   →  [B, S, N]
      2) fire:   gated = activations * gates
      3) emit:   out = gated @ V          →  [B, S, D]

    The intermediate [B, S, N] activations are recomputed in backward
    (not saved) thanks to jax.checkpoint.
    """
    activations = jnp.einsum('bsd,nd->bsn', x, neurons)  # sense
    gated = activations * gates                            # fire
    return jnp.einsum('bsn,nd->bsd', gated, neurons)      # emit


# ================================================================
# 6. NeuronPool — shared rank-1 vectors (replaces SharedNeurons)
# ================================================================

class NeuronPool(nn.Module):
    """Shared rank-1 neuron pool with hierarchical cluster structure.

    Each neuron is a single [D] vector. Three pools:
      - qk_neurons [N_qk, D]: shared for Q and K
      - v_neurons  [N_v, D]:  for V
      - know_neurons [N_know, D]: for knowledge (FFN-equivalent)

    Hierarchical routing adds cluster embeddings for coarse selection:
      - cluster_emb_qk  [n_clusters_qk, d_space]
      - cluster_emb_v   [n_clusters_v, d_space]
      - cluster_emb_know [n_clusters_know, d_space]
    Cluster assignments are fixed: neuron i -> cluster i // cluster_size.
    """
    n_qk: int
    n_v: int
    n_know: int
    d_model: int
    d_space: int = 256
    n_clusters_qk: int = 64
    n_clusters_v: int = 64
    n_clusters_know: int = 128

    def setup(self):
        self.qk_neurons = self.param(
            'qk_neurons', unit_norm_init(), (self.n_qk, self.d_model))
        self.v_neurons = self.param(
            'v_neurons', unit_norm_init(), (self.n_v, self.d_model))
        self.know_neurons = self.param(
            'know_neurons', unit_norm_init(), (self.n_know, self.d_model))

        # Cluster embeddings for hierarchical routing
        self.cluster_emb_qk = self.param(
            'cluster_emb_qk', scaled_normal(0.02),
            (self.n_clusters_qk, self.d_space))
        self.cluster_emb_v = self.param(
            'cluster_emb_v', scaled_normal(0.02),
            (self.n_clusters_v, self.d_space))
        self.cluster_emb_know = self.param(
            'cluster_emb_know', scaled_normal(0.02),
            (self.n_clusters_know, self.d_space))


# ================================================================
# 6b. Hierarchical routing helpers
# ================================================================

def _hierarchical_gate(h, neuron_emb, cluster_emb, tau,
                       n_neurons, n_clusters, k_cluster, max_k):
    """2-stage hierarchical routing: cluster selection → neuron routing.

    Args:
        h:           [B, S, d_space] projected input
        neuron_emb:  [N, d_space] neuron embeddings (already normalized)
        cluster_emb: [n_clusters, d_space] cluster embeddings (learnable)
        tau:         [B, S, 1] input-dependent threshold
        n_neurons:   total neuron count N
        n_clusters:  number of clusters
        k_cluster:   how many clusters to select
        max_k:       safety cap on total active neurons

    Returns:
        gates: [B, S, N] sparse gate values (most entries zero)
        cluster_aux: scalar cluster-level load balance loss
        neuron_aux:  scalar neuron-level load balance loss
    """
    B, S, _ = h.shape
    cluster_size = n_neurons // n_clusters

    # --- Stage 1: Coarse cluster selection ---
    cluster_emb_norm = cluster_emb / (
        jnp.linalg.norm(cluster_emb, axis=-1, keepdims=True) + 1e-8)
    cluster_scores = jnp.einsum('bsd,cd->bsc', h, cluster_emb_norm)  # [B,S,n_clusters]

    # Top-k clusters
    topk_vals, topk_ids = jax.lax.top_k(cluster_scores, k_cluster)  # [B,S,k_cluster]

    # Cluster load balance: encourage uniform cluster usage
    # Soft assignment via softmax for differentiable balance
    cluster_probs = jax.nn.softmax(cluster_scores, axis=-1)  # [B,S,n_clusters]
    cluster_freq = cluster_probs.mean(axis=(0, 1))  # [n_clusters]
    target_cluster = 1.0 / n_clusters
    cluster_aux = ((cluster_freq - target_cluster) ** 2).sum() * n_clusters

    # --- Stage 2: Fine neuron selection within selected clusters ---
    # Build mask of active neurons from selected clusters
    # topk_ids: [B, S, k_cluster] — each entry is a cluster index
    # For each selected cluster c, neurons c*cluster_size .. (c+1)*cluster_size-1 are candidates

    # Expand cluster ids to neuron ids: [B, S, k_cluster * cluster_size]
    # offsets within cluster: [cluster_size]
    offsets = jnp.arange(cluster_size)  # [cluster_size]
    # base indices: [B, S, k_cluster, 1] * cluster_size + [cluster_size]
    base = topk_ids * cluster_size  # [B, S, k_cluster]
    # [B, S, k_cluster, cluster_size]
    active_ids = base[:, :, :, jnp.newaxis] + offsets[jnp.newaxis, jnp.newaxis, jnp.newaxis, :]
    active_ids = active_ids.reshape(B, S, k_cluster * cluster_size)  # [B, S, n_active]

    # Gather neuron embeddings for active neurons
    active_emb = neuron_emb[active_ids.reshape(-1)]  # [B*S*n_active, d_space]
    active_emb = active_emb.reshape(B, S, k_cluster * cluster_size, -1)  # [B,S,n_active,d_space]

    # Compute scores only for active neurons
    # h: [B,S,d_space], active_emb: [B,S,n_active,d_space]
    active_scores = jnp.einsum('bsd,bsnd->bsn', h, active_emb)  # [B,S,n_active]

    # Apply threshold gate on active scores
    active_gates = threshold_gate(active_scores, tau, max_k)  # [B,S,n_active]

    # Scatter back to full [B, S, N] gate tensor
    gates = jnp.zeros((B, S, n_neurons), dtype=active_gates.dtype)
    # Use scatter-add: for each (b, s), place active_gates into gates at active_ids positions
    b_idx = jnp.arange(B)[:, jnp.newaxis, jnp.newaxis]  # [B,1,1]
    s_idx = jnp.arange(S)[jnp.newaxis, :, jnp.newaxis]  # [1,S,1]
    gates = gates.at[b_idx, s_idx, active_ids].set(active_gates)

    # Neuron load balance (within active neurons)
    target_neuron = 1.0 / n_neurons
    neuron_freq = gates.mean(axis=(0, 1))  # [N]
    neuron_aux = ((neuron_freq - target_neuron) ** 2).sum() * n_neurons

    return gates, cluster_aux, neuron_aux


def _hierarchical_gate_pure(h, neuron_emb, cluster_emb, tau,
                            n_neurons, n_clusters, k_cluster, max_k):
    """Pure-function version of hierarchical gate (for scan body).

    Same as _hierarchical_gate but takes raw arrays, not module params.
    """
    return _hierarchical_gate(h, neuron_emb, cluster_emb, tau,
                              n_neurons, n_clusters, k_cluster, max_k)


# ================================================================
# 7. Router — learned embeddings + hierarchical threshold tau gating
# ================================================================

class Router(nn.Module):
    """Hierarchical router: cluster selection → neuron routing → threshold tau.

    2-stage routing for each pool (QK, V, Know):
      Stage 1: score against cluster embeddings, top-k_cluster selection
      Stage 2: score against neuron embeddings within selected clusters
    """
    d_model: int
    d_space: int
    n_qk: int
    n_v: int
    n_know: int
    n_clusters_qk: int = 64
    n_clusters_v: int = 64
    n_clusters_know: int = 128
    k_cluster_qk: int = 8
    k_cluster_v: int = 8
    k_cluster_know: int = 8
    max_k: int = 32
    router_dropout: float = 0.1

    def setup(self):
        total = self.n_qk + self.n_v + self.n_know
        self.neuron_emb = self.param(
            'neuron_emb', scaled_normal(0.02), (total, self.d_space))

        # Attention: 3 projections (Q, K share qk_emb; V uses v_emb)
        self.proj_attn = nn.Dense(self.d_space * 3, name='proj_attn')
        # Knowledge: 1 projection
        self.proj_know = nn.Dense(self.d_space, name='proj_know')

        # Learnable tau (v18.5 style)
        self.tau_attn = nn.Dense(3, name='tau_attn')   # tau_Q, tau_K, tau_V
        self.tau_know = nn.Dense(1, name='tau_know')

    def get_attention_gates(self, x, neuron_pool, deterministic, rng):
        """Compute Q, K, V gates via hierarchical routing.

        Returns: (gate_Q [B,S,N_qk], gate_K [B,S,N_qk],
                  gate_V [B,S,N_v], aux_loss)
        """
        emb = self.neuron_emb
        emb_norm = emb / (jnp.linalg.norm(emb, axis=-1, keepdims=True) + 1e-8)

        qk_emb = emb_norm[:self.n_qk]
        v_emb = emb_norm[self.n_qk:self.n_qk + self.n_v]

        rng, rng1 = jax.random.split(rng)
        h_all = self.proj_attn(x)
        h_all = safe_dropout(h_all, self.router_dropout, deterministic, rng1)
        h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)

        tau_all = self.tau_attn(x)  # [B, S, 3]
        tau_Q = tau_all[:, :, 0:1]
        tau_K = tau_all[:, :, 1:2]
        tau_V = tau_all[:, :, 2:3]

        # Hierarchical routing for Q, K (shared QK pool)
        gate_Q, caux_q, naux_q = _hierarchical_gate(
            h_Q, qk_emb, neuron_pool.cluster_emb_qk, tau_Q,
            self.n_qk, self.n_clusters_qk, self.k_cluster_qk, self.max_k)
        gate_K, caux_k, naux_k = _hierarchical_gate(
            h_K, qk_emb, neuron_pool.cluster_emb_qk, tau_K,
            self.n_qk, self.n_clusters_qk, self.k_cluster_qk, self.max_k)
        # Hierarchical routing for V
        gate_V, caux_v, naux_v = _hierarchical_gate(
            h_V, v_emb, neuron_pool.cluster_emb_v, tau_V,
            self.n_v, self.n_clusters_v, self.k_cluster_v, self.max_k)

        aux = (caux_q + caux_k + caux_v) + (naux_q + naux_k + naux_v)
        return gate_Q, gate_K, gate_V, aux

    def get_knowledge_gates(self, x, neuron_pool, deterministic, rng):
        """Compute knowledge gates via hierarchical routing.

        Returns: (gate_know [B,S,N_know], aux_loss)
        """
        emb = self.neuron_emb
        emb_norm = emb / (jnp.linalg.norm(emb, axis=-1, keepdims=True) + 1e-8)
        know_emb = emb_norm[self.n_qk + self.n_v:]

        rng, rng1 = jax.random.split(rng)
        h_know = self.proj_know(x)
        h_know = safe_dropout(h_know, self.router_dropout, deterministic, rng1)

        tau = self.tau_know(x)

        gate, caux, naux = _hierarchical_gate(
            h_know, know_emb, neuron_pool.cluster_emb_know, tau,
            self.n_know, self.n_clusters_know, self.k_cluster_know, self.max_k)

        aux = caux + naux
        return gate, aux


# ================================================================
# 8. Pure functions for jax.lax.scan forward path
# ================================================================

def _router_attn_gates(x, router_params, pool_params,
                       n_qk, n_v, d_space, max_k,
                       n_clusters_qk, n_clusters_v,
                       k_cluster_qk, k_cluster_v,
                       router_dropout, deterministic, rng):
    """Pure function: hierarchical attention gates + aux_loss (for scan body)."""
    neuron_emb = router_params['neuron_emb']
    emb_norm = neuron_emb / (jnp.linalg.norm(neuron_emb, axis=-1, keepdims=True) + 1e-8)

    qk_emb = emb_norm[:n_qk]
    v_emb = emb_norm[n_qk:n_qk + n_v]

    rng, rng1 = jax.random.split(rng)
    h_all = x @ router_params['proj_attn']['kernel'] + router_params['proj_attn']['bias']
    h_all = safe_dropout(h_all, router_dropout, deterministic, rng1)
    h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)

    tau_all = x @ router_params['tau_attn']['kernel'] + router_params['tau_attn']['bias']
    tau_Q = tau_all[:, :, 0:1]
    tau_K = tau_all[:, :, 1:2]
    tau_V = tau_all[:, :, 2:3]

    # Hierarchical routing
    cluster_emb_qk = pool_params['cluster_emb_qk']
    cluster_emb_v = pool_params['cluster_emb_v']

    gate_Q, caux_q, naux_q = _hierarchical_gate(
        h_Q, qk_emb, cluster_emb_qk, tau_Q,
        n_qk, n_clusters_qk, k_cluster_qk, max_k)
    gate_K, caux_k, naux_k = _hierarchical_gate(
        h_K, qk_emb, cluster_emb_qk, tau_K,
        n_qk, n_clusters_qk, k_cluster_qk, max_k)
    gate_V, caux_v, naux_v = _hierarchical_gate(
        h_V, v_emb, cluster_emb_v, tau_V,
        n_v, n_clusters_v, k_cluster_v, max_k)

    aux = (caux_q + caux_k + caux_v) + (naux_q + naux_k + naux_v)
    return gate_Q, gate_K, gate_V, aux


def _router_know_gates(x, router_params, pool_params,
                       n_qk, n_v, n_know, max_k,
                       n_clusters_know, k_cluster_know,
                       router_dropout, deterministic, rng):
    """Pure function: hierarchical knowledge gates + aux_loss (for scan body)."""
    neuron_emb = router_params['neuron_emb']
    emb_norm = neuron_emb / (jnp.linalg.norm(neuron_emb, axis=-1, keepdims=True) + 1e-8)
    know_emb = emb_norm[n_qk + n_v:]

    rng, rng1 = jax.random.split(rng)
    h_know = x @ router_params['proj_know']['kernel'] + router_params['proj_know']['bias']
    h_know = safe_dropout(h_know, router_dropout, deterministic, rng1)

    tau = x @ router_params['tau_know']['kernel'] + router_params['tau_know']['bias']

    cluster_emb_know = pool_params['cluster_emb_know']
    gate, caux, naux = _hierarchical_gate(
        h_know, know_emb, cluster_emb_know, tau,
        n_know, n_clusters_know, k_cluster_know, max_k)

    aux = caux + naux
    return gate, aux


def _attn_forward(x, pool_params, gate_Q, gate_K, gate_V,
                  expand_O_kernel, n_heads, d_model,
                  dropout_rate, deterministic, rng):
    """Pure function: rank-1 attention circuit (for scan body)."""
    B, S, D = x.shape
    d_head = d_model // n_heads

    qk = pool_params['qk_neurons']
    v = pool_params['v_neurons']

    Q = sense_emit(x, qk, gate_Q)   # [B, S, D]
    K = sense_emit(x, qk, gate_K)   # [B, S, D]
    V = sense_emit(x, v, gate_V)    # [B, S, D]

    Q = Q.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
    K = K.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
    V = V.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)

    scale = jnp.sqrt(jnp.float32(d_head))
    scores = jnp.einsum('bhsd,bhtd->bhst', Q, K) / scale
    causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
    scores = jnp.where(causal, scores, jnp.finfo(scores.dtype).min)
    attn_w = jax.nn.softmax(scores, axis=-1)

    rng, rng1, rng2 = jax.random.split(rng, 3)
    attn_w = safe_dropout(attn_w, dropout_rate, deterministic, rng1)

    out = jnp.einsum('bhst,bhtd->bhsd', attn_w, V)
    out = out.transpose(0, 2, 1, 3).reshape(B, S, D)
    out = out @ expand_O_kernel
    out = safe_dropout(out, dropout_rate, deterministic, rng2)
    return out


def _know_forward(x, pool_params, gate_know,
                  dropout_rate, deterministic, rng):
    """Pure function: rank-1 knowledge circuit (for scan body)."""
    know = pool_params['know_neurons']
    out = sense_emit(x, know, gate_know)
    out = safe_dropout(out, dropout_rate, deterministic, rng)
    return out


# ================================================================
# 9. Flax modules (used during init; scan uses pure functions above)
# ================================================================

class AttentionCircuit(nn.Module):
    """Rank-1 attention: Q/K/V via sense_emit + causal self-attention."""
    d_model: int
    n_heads: int
    dropout_rate: float = 0.1

    def setup(self):
        self.expand_O = nn.Dense(
            self.d_model, use_bias=False, kernel_init=scaled_normal(0.02))

    def __call__(self, x, neuron_pool, router, attention_mask, deterministic):
        rng = self.make_rng('dropout')
        rng, rng_router, rng_drop, rng_out = jax.random.split(rng, 4)

        gate_Q, gate_K, gate_V, aux = router.get_attention_gates(
            x, neuron_pool, deterministic, rng_router)

        Q = sense_emit(x, neuron_pool.qk_neurons, gate_Q)
        K = sense_emit(x, neuron_pool.qk_neurons, gate_K)
        V = sense_emit(x, neuron_pool.v_neurons, gate_V)

        B, S, D = x.shape
        d_head = D // self.n_heads
        Q = Q.reshape(B, S, self.n_heads, d_head).transpose(0, 2, 1, 3)
        K = K.reshape(B, S, self.n_heads, d_head).transpose(0, 2, 1, 3)
        V = V.reshape(B, S, self.n_heads, d_head).transpose(0, 2, 1, 3)

        scale = jnp.sqrt(jnp.float32(d_head))
        scores = jnp.einsum('bhsd,bhtd->bhst', Q, K) / scale
        causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
        scores = jnp.where(causal, scores, jnp.finfo(scores.dtype).min)
        attn_w = jax.nn.softmax(scores, axis=-1)
        attn_w = safe_dropout(attn_w, self.dropout_rate, deterministic, rng_drop)

        out = jnp.einsum('bhst,bhtd->bhsd', attn_w, V)
        out = out.transpose(0, 2, 1, 3).reshape(B, S, D)
        out = self.expand_O(out)
        out = safe_dropout(out, self.dropout_rate, deterministic, rng_out)
        return out, aux


class KnowledgeCircuit(nn.Module):
    """Rank-1 knowledge: FFN-equivalent via sense_emit on know_neurons."""
    d_model: int
    dropout_rate: float = 0.1

    def __call__(self, x, neuron_pool, router, attention_mask, deterministic):
        rng = self.make_rng('dropout')
        rng, rng_router = jax.random.split(rng)

        gate, aux = router.get_knowledge_gates(x, neuron_pool, deterministic, rng_router)
        out = sense_emit(x, neuron_pool.know_neurons, gate)
        out = safe_dropout(out, self.dropout_rate, deterministic, rng)
        return out, aux


class DAWNBlock(nn.Module):
    """Single DAWN block: norm → attn → residual → norm → knowledge → residual."""
    d_model: int
    n_heads: int
    dropout_rate: float = 0.1

    def setup(self):
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.attn = AttentionCircuit(
            d_model=self.d_model, n_heads=self.n_heads,
            dropout_rate=self.dropout_rate)
        self.knowledge = KnowledgeCircuit(
            d_model=self.d_model, dropout_rate=self.dropout_rate)

    def __call__(self, x, neuron_pool, router, attention_mask, deterministic):
        normed = self.norm1(x)
        attn_out, attn_aux = self.attn(
            normed, neuron_pool, router, attention_mask, deterministic)
        x = x + attn_out

        normed = self.norm2(x)
        know_out, know_aux = self.knowledge(
            normed, neuron_pool, router, attention_mask, deterministic)
        x = x + know_out

        return x, attn_aux + know_aux


# ================================================================
# 10. DAWN Model
# ================================================================

class DAWN(nn.Module):
    """DAWN-Spatial: Rank-1 Neuron Architecture.

    Each neuron is a single vector [D].
    Transform per circuit: out = V^T diag(gate) V x.
    Uses jax.lax.scan for O(1) XLA compile, jax.checkpoint for memory.
    Weight tying: lm_head reuses token_emb via nn.Embed.attend().

    Usage:
        model = DAWN(vocab_size=30522, d_model=384, ...)
        variables = model.init(rng, input_ids)
        output = model.apply(variables, input_ids, labels=labels,
                             deterministic=False,
                             rngs={'dropout': dropout_rng})
    """
    __version__ = "spatial-r1"

    vocab_size: int = 30000
    d_model: int = 384
    n_layers: int = 12
    n_heads: int = 6
    max_seq_len: int = 512
    d_space: int = 64
    n_qk: int = 256
    n_v: int = 256
    n_know: int = 512
    max_k: int = 32
    dropout_rate: float = 0.1
    router_dropout: float = 0.1
    gradient_checkpointing: bool = False
    # Hierarchical routing
    n_clusters_qk: int = 64
    n_clusters_v: int = 64
    n_clusters_know: int = 128
    k_cluster_qk: int = 8
    k_cluster_v: int = 8
    k_cluster_know: int = 8

    def setup(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"n_heads ({self.n_heads})")

        self.token_emb = nn.Embed(
            self.vocab_size, self.d_model,
            embedding_init=scaled_normal(0.02))
        self.pos_emb = nn.Embed(
            self.max_seq_len, self.d_model,
            embedding_init=scaled_normal(0.02))

        self.neuron_pool = NeuronPool(
            n_qk=self.n_qk, n_v=self.n_v, n_know=self.n_know,
            d_model=self.d_model, d_space=self.d_space,
            n_clusters_qk=self.n_clusters_qk,
            n_clusters_v=self.n_clusters_v,
            n_clusters_know=self.n_clusters_know)

        self.router = Router(
            d_model=self.d_model, d_space=self.d_space,
            n_qk=self.n_qk, n_v=self.n_v, n_know=self.n_know,
            n_clusters_qk=self.n_clusters_qk,
            n_clusters_v=self.n_clusters_v,
            n_clusters_know=self.n_clusters_know,
            k_cluster_qk=self.k_cluster_qk,
            k_cluster_v=self.k_cluster_v,
            k_cluster_know=self.k_cluster_know,
            max_k=self.max_k, router_dropout=self.router_dropout)

        # Blocks: used for init; forward uses scan with pure functions
        self.layers = [
            DAWNBlock(
                d_model=self.d_model, n_heads=self.n_heads,
                dropout_rate=self.dropout_rate,
                name=f'block_{i}')
            for i in range(self.n_layers)
        ]

        self.norm = nn.LayerNorm()
        # lm_head uses weight tying via token_emb.attend()

    def __call__(self, input_ids, labels=None, attention_mask=None,
                 deterministic=False):
        B, S = input_ids.shape
        if S > self.max_seq_len:
            raise ValueError(
                f"Sequence length {S} exceeds max_seq_len {self.max_seq_len}")

        positions = jnp.arange(S)[jnp.newaxis, :]  # [1, S]
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        emb_rng = self.make_rng('dropout')
        x = safe_dropout(x, self.dropout_rate, deterministic, emb_rng)

        if self.is_initializing():
            # Init path: for-loop to trigger lazy parameter creation
            total_aux = jnp.float32(0.0)
            for layer in self.layers:
                x, aux = layer(
                    x, self.neuron_pool, self.router,
                    attention_mask, deterministic)
                total_aux = total_aux + aux
        else:
            # Forward path: jax.lax.scan with pure functions
            # neuron_pool + router params captured via closure (true sharing)
            all_params = self.variables['params']
            pool_params = all_params['neuron_pool']
            router_params = all_params['router']

            # Stack per-block params: [n_layers, ...]
            block_params_list = [all_params[f'block_{i}']
                                 for i in range(self.n_layers)]
            stacked = jax.tree.map(
                lambda *arrays: jnp.stack(arrays), *block_params_list)

            # Pre-split dropout RNGs: one per layer
            base_rng = self.make_rng('dropout')
            layer_rngs = jax.random.split(base_rng, self.n_layers)

            def scan_body(carry, xs):
                x = carry
                bp = xs['params']
                rng = xs['rng']
                rng, rng_ar, rng_kr, rng_a, rng_k = \
                    jax.random.split(rng, 5)

                # --- Attention sub-block ---
                normed = _layer_norm(
                    x, bp['norm1']['scale'], bp['norm1']['bias'])

                gate_Q, gate_K, gate_V, attn_aux = _router_attn_gates(
                    normed, router_params, pool_params,
                    self.n_qk, self.n_v, self.d_space, self.max_k,
                    self.n_clusters_qk, self.n_clusters_v,
                    self.k_cluster_qk, self.k_cluster_v,
                    self.router_dropout, deterministic, rng_ar)

                attn_out = _attn_forward(
                    normed, pool_params, gate_Q, gate_K, gate_V,
                    bp['attn']['expand_O']['kernel'],
                    self.n_heads, self.d_model,
                    self.dropout_rate, deterministic, rng_a)

                x = x + attn_out

                # --- Knowledge sub-block ---
                normed = _layer_norm(
                    x, bp['norm2']['scale'], bp['norm2']['bias'])

                gate_know, know_aux = _router_know_gates(
                    normed, router_params, pool_params,
                    self.n_qk, self.n_v, self.n_know, self.max_k,
                    self.n_clusters_know, self.k_cluster_know,
                    self.router_dropout, deterministic, rng_kr)

                know_out = _know_forward(
                    normed, pool_params, gate_know,
                    self.dropout_rate, deterministic, rng_k)

                x = x + know_out
                return x, attn_aux + know_aux

            if self.gradient_checkpointing:
                scan_body = jax.checkpoint(scan_body)

            xs = {'params': stacked, 'rng': layer_rngs}
            x, aux_losses = jax.lax.scan(scan_body, x, xs)
            total_aux = aux_losses.sum()

        x = self.norm(x)

        result = {'aux_loss': total_aux}

        if labels is not None:
            # Loss+accuracy under checkpoint — avoids materializing [B,S,V]
            embedding_matrix = self.token_emb.embedding  # [V, D]
            shift_x = x[:, :-1, :]                       # [B, S-1, D]
            shift_labels = labels[:, 1:].astype(jnp.int32)
            valid_mask = (shift_labels != -100)

            @jax.checkpoint
            def compute_loss_and_acc(x_chunk, emb, labs, vmask):
                logits = x_chunk @ emb.T                  # [B, S-1, V]
                log_probs = jax.nn.log_softmax(logits, axis=-1)
                safe = jnp.where(vmask, labs, 0)
                tl = -jnp.take_along_axis(
                    log_probs, safe[..., jnp.newaxis], axis=-1).squeeze(-1)
                loss = (tl * vmask).sum() / (vmask.sum() + 1e-8)
                preds = jnp.argmax(logits, axis=-1)
                correct = jnp.sum((preds == labs) & vmask)
                valid_count = jnp.sum(vmask)
                return loss, correct, valid_count

            loss, correct, valid_count = compute_loss_and_acc(
                shift_x, embedding_matrix, shift_labels, valid_mask)
            result['loss'] = loss
            result['correct'] = correct
            result['valid_count'] = valid_count
        else:
            # Inference: need full logits
            logits = self.token_emb.attend(x)
            result['logits'] = logits

        return result

    # ------------------------------------------------------------------
    # Auxiliary losses (computed from parameters, no forward needed)
    # ------------------------------------------------------------------

    def diversity_loss(self):
        """Encourage diversity among neurons via cosine similarity penalty."""
        def _pool_div(neurons):
            n = neurons / (jnp.linalg.norm(neurons, axis=-1, keepdims=True) + 1e-8)
            sim = n @ n.T
            mask = ~jnp.eye(sim.shape[0], dtype=jnp.bool_)
            return jnp.abs(sim * mask).sum() / mask.sum()

        return (_pool_div(self.neuron_pool.qk_neurons) +
                _pool_div(self.neuron_pool.v_neurons) +
                _pool_div(self.neuron_pool.know_neurons)) / 3

    def get_auxiliary_losses(self):
        """Return dict of all auxiliary losses."""
        return {
            'neuron_diversity': self.diversity_loss(),
        }

    # ------------------------------------------------------------------
    # Config / info
    # ------------------------------------------------------------------

    def get_config(self):
        return {
            'model_version': self.__version__,
            'vocab_size': self.vocab_size, 'd_model': self.d_model,
            'n_layers': self.n_layers, 'n_heads': self.n_heads,
            'max_seq_len': self.max_seq_len,
            'n_qk': self.n_qk, 'n_v': self.n_v, 'n_know': self.n_know,
            'max_k': self.max_k, 'd_space': self.d_space,
            'n_clusters_qk': self.n_clusters_qk,
            'n_clusters_v': self.n_clusters_v,
            'n_clusters_know': self.n_clusters_know,
            'k_cluster_qk': self.k_cluster_qk,
            'k_cluster_v': self.k_cluster_v,
            'k_cluster_know': self.k_cluster_know,
        }

    def get_model_info(self):
        cs_qk = self.n_qk // self.n_clusters_qk
        cs_v = self.n_v // self.n_clusters_v
        cs_know = self.n_know // self.n_clusters_know
        return [
            f"DAWN v{self.__version__}: Rank-1 Neuron + Hierarchical Routing (JAX)",
            f"  d_model={self.d_model}, n_layers={self.n_layers}, "
            f"n_heads={self.n_heads}",
            f"  max_seq_len={self.max_seq_len}, dropout={self.dropout_rate}",
            f"",
            f"  [Neuron Pool — Rank-1 vectors]",
            f"  QK neurons: {self.n_qk} x {self.d_model}  "
            f"(shared Q/K, max_k={self.max_k})",
            f"  V neurons:  {self.n_v} x {self.d_model}  "
            f"(max_k={self.max_k})",
            f"  Know neurons: {self.n_know} x {self.d_model}  "
            f"(max_k={self.max_k})",
            f"",
            f"  [Hierarchical Routing — 2-stage]",
            f"  QK:   {self.n_clusters_qk} clusters x {cs_qk} neurons, "
            f"top-{self.k_cluster_qk} → search {self.k_cluster_qk * cs_qk}/{self.n_qk}",
            f"  V:    {self.n_clusters_v} clusters x {cs_v} neurons, "
            f"top-{self.k_cluster_v} → search {self.k_cluster_v * cs_v}/{self.n_v}",
            f"  Know: {self.n_clusters_know} clusters x {cs_know} neurons, "
            f"top-{self.k_cluster_know} → search {self.k_cluster_know * cs_know}/{self.n_know}",
            f"  d_space={self.d_space}, router_dropout={self.router_dropout}",
            f"",
            f"  [Memory Optimization]",
            f"  jax.checkpoint: sense_emit (rank-1 intermediate recompute)",
            f"  gradient_checkpointing={self.gradient_checkpointing}",
            f"  jax.lax.scan layer loop (O(1) XLA compile)",
        ]
