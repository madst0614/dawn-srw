"""
DAWN-Spatial v2: Rank-1 Neuron Architecture with 2D Positional Routing (JAX/Flax)

Changelog:
  spatial-r1-v2.0.6 (2026-03-31):
    - Both Know AND Attention pipelines fully chunked with carry accumulation
    - _attn_pipeline_chunked: Q/K/V sense_emit chunked, self-attention full seq
    - _know_pipeline_chunked: carry accumulation via dynamic_update_slice
    - Removed dead code: _router_attn_gates, _router_know_gates, _attn_forward,
      _know_forward, _router_and_attn, sense_emit_sparse, _chunked_distance_topk
    - scan_body computes pos/tau outside pipelines, passes as args

  spatial-r1-v2.0.5 (2026-03-31):
    - Know pipeline fully chunked over sequence axis to fix gather OOM
      ([B,S,n_cand,D] too large). distance->topk->gather->score->gate->
      sense_emit all inside per-chunk scan. candidates_multiplier 3->2.
    - Add know_chunk_size hyperparameter (default 16)

  spatial-r1-v2.0.4 (2026-03-31):
    - Fix TracerBoolConversionError: use jax.checkpoint with static_argnums
      for int/float/bool hyperparameters (max_k, n_heads, dropout, etc.)
    - Reorder function signatures: dynamic tensor args first, static last

  spatial-r1-v2.0.3 (2026-03-31):
    - Fix jax.checkpoint dynamic slice error: pre-slice neuron_pos in
      scan_body (outside checkpoint), pass npos_qk/npos_v/npos_know as
      args to checkpointed functions

  spatial-r1-v2.0.2 (2026-03-31):
    - Wrap routing+sense_emit in single jax.checkpoint (_router_and_attn,
      _router_and_know) to eliminate backward OOM from intermediate tensors
    - Remove redundant @jax.checkpoint from sense_emit_sparse (now inside
      higher-level checkpoint)

  spatial-r1-v2.0.1 (2026-03-31):
    - CRITICAL: Know distance OOM fix (chunked sequence processing)
    - pos_loss distance reuse (take_along_axis / chunked return)
    - diversity_loss: random sampling -> deterministic strided sampling

  spatial-r1-v2.0.0 (2026-03-31):
    - neuron_emb dot-product routing -> 2D positional routing
    - Per-circuit max_k (max_k_qk, max_k_v, max_k_know)
    - Balanced neuron pool ratios (1 : 1.5 : 5)
    - sense_emit -> sense_emit_sparse (candidate-only computation)
    - pos_loss for neuron_pos learning
    - Removed: d_space, cluster embeddings, hierarchical routing

Key concept:
  - Every neuron is a single vector v_i [D] (rank-1).
  - Sense: activation_i = v_i . x  (scalar)
  - Fire:  gate decides which neurons fire (threshold tau)
  - Emit:  out = sum gate_i * activation_i * v_i
  - Equivalent to: out = V^T diag(gate) V x  (sum of rank-1 outer products)

2D Positional Routing:
  - Each neuron has a learnable 2D coordinate neuron_pos [N, 2]
  - Input generates query_pos [B, S, 2] via projection
  - Distance-based candidate selection (top-K nearest neurons)
  - Candidate neurons scored via input @ neurons[candidates].T (differentiable)
  - neuron_pos learned via pos_loss (pull activated neurons toward query_pos)

Architecture:
  NeuronPool   -- shared [N, D] rank-1 vectors (no cluster embeddings)
  Router       -- 2D pos: distance candidate selection -> threshold tau gate
  sense_emit_sparse -- checkpointed rank-1 transform on candidates only
  AttentionCircuit -- Q/K/V via sparse sense_emit + causal self-attention
  KnowledgeCircuit -- FFN-equivalent via sparse sense_emit
  DAWNBlock    -- norm -> attn -> residual -> norm -> knowledge -> residual
  DAWN         -- embedding + jax.lax.scan layer loop + weight-tied lm_head
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Dict


# ================================================================
# 1. Trace-safe dropout (from v17.1 -- compatible with nn.remat)
# ================================================================

def safe_dropout(x, rate, deterministic, rng):
    """Dropout that is fully trace-safe -- no Python bool branch."""
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

    Shape: [N, D] -- each of N vectors is independently normalized.
    """
    def init(key, shape, dtype=jnp.float32):
        x = jax.random.normal(key, shape, dtype)
        norms = jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8
        return x / norms * scale
    return init


# ================================================================
# 4. Threshold gate (v18.5 tau mechanism -> JAX)
# ================================================================

def threshold_gate(scores, tau, max_k=None):
    """Threshold-based gating with dead-neuron gradient flow.

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


# (sense_emit_sparse removed in v2.0.6 — inlined into chunked pipelines)


# ================================================================
# 6. NeuronPool -- shared rank-1 vectors (no cluster structure)
# ================================================================

class NeuronPool(nn.Module):
    """Shared rank-1 neuron pool.

    Each neuron is a single [D] vector. Three pools:
      - qk_neurons [N_qk, D]: shared for Q and K
      - v_neurons  [N_v, D]:  for V
      - know_neurons [N_know, D]: for knowledge (FFN-equivalent)
    """
    n_qk: int
    n_v: int
    n_know: int
    d_model: int

    def setup(self):
        self.qk_neurons = self.param(
            'qk_neurons', unit_norm_init(), (self.n_qk, self.d_model))
        self.v_neurons = self.param(
            'v_neurons', unit_norm_init(), (self.n_v, self.d_model))
        self.know_neurons = self.param(
            'know_neurons', unit_norm_init(), (self.n_know, self.d_model))


# ================================================================
# 7. Router -- 2D positional routing + threshold tau gating
# ================================================================

class Router(nn.Module):
    """2D positional router: distance-based candidate selection -> threshold tau.

    Each neuron has a learnable 2D coordinate. Input is projected to a 2D
    query position. Candidates are selected by proximity, then scored via
    dot product with the actual neuron vectors (differentiable path).
    """
    d_model: int
    pos_dim: int          # 2
    n_qk: int
    n_v: int
    n_know: int
    max_k_qk: int
    max_k_v: int
    max_k_know: int
    candidates_multiplier: int = 3
    router_dropout: float = 0.1

    def setup(self):
        total = self.n_qk + self.n_v + self.n_know

        # Learnable 2D coordinates for all neurons
        self.neuron_pos = self.param(
            'neuron_pos', scaled_normal(1.0), (total, self.pos_dim))

        # Input -> query position projections (per circuit)
        self.proj_pos_qk = nn.Dense(self.pos_dim, name='proj_pos_qk')
        self.proj_pos_v = nn.Dense(self.pos_dim, name='proj_pos_v')
        self.proj_pos_know = nn.Dense(self.pos_dim, name='proj_pos_know')

        # Learnable tau (v18.5 style)
        self.tau_attn = nn.Dense(3, name='tau_attn')   # tau_Q, tau_K, tau_V
        self.tau_know = nn.Dense(1, name='tau_know')

    def get_attention_gates(self, x, neuron_pool, deterministic, rng):
        """Compute Q, K, V gates via 2D positional routing.

        Returns: (gate_Q, gate_K, gate_V, cand_idx_qk, cand_idx_v, aux_loss)
          gate_Q:      [B, S, n_cand_qk]
          gate_K:      [B, S, n_cand_qk]
          gate_V:      [B, S, n_cand_v]
          cand_idx_qk: [B, S, n_cand_qk]
          cand_idx_v:  [B, S, n_cand_v]
          aux_loss:    scalar (pos_loss)
        """
        B, S, D = x.shape
        qk_neurons = neuron_pool.qk_neurons   # [N_qk, D]
        v_neurons = neuron_pool.v_neurons      # [N_v, D]

        # Query positions (differentiable)
        qk_pos = self.proj_pos_qk(x)   # [B, S, 2]
        v_pos = self.proj_pos_v(x)      # [B, S, 2]

        # Neuron positions (sliced per pool)
        npos_qk = self.neuron_pos[:self.n_qk]                          # [N_qk, 2]
        npos_v = self.neuron_pos[self.n_qk:self.n_qk + self.n_v]      # [N_v, 2]

        n_cand_qk = self.max_k_qk * self.candidates_multiplier
        n_cand_v = self.max_k_v * self.candidates_multiplier

        # --- Candidate selection (distance-based, stop_gradient) ---
        # QK candidates
        dist_qk = jnp.sum(
            (qk_pos[:, :, None, :] - npos_qk[None, None, :, :]) ** 2,
            axis=-1)  # [B, S, N_qk]
        _, cand_idx_qk = jax.lax.top_k(-dist_qk, n_cand_qk)  # [B, S, n_cand_qk]
        cand_idx_qk = jax.lax.stop_gradient(cand_idx_qk)

        # V candidates
        dist_v = jnp.sum(
            (v_pos[:, :, None, :] - npos_v[None, None, :, :]) ** 2,
            axis=-1)  # [B, S, N_v]
        _, cand_idx_v = jax.lax.top_k(-dist_v, n_cand_v)
        cand_idx_v = jax.lax.stop_gradient(cand_idx_v)

        # --- Precise scoring within candidates (differentiable) ---
        cand_qk = qk_neurons[cand_idx_qk]   # [B, S, n_cand_qk, D]
        cand_v = v_neurons[cand_idx_v]       # [B, S, n_cand_v, D]

        rng, rng1 = jax.random.split(rng)
        x_drop = safe_dropout(x, self.router_dropout, deterministic, rng1)

        scores_Q = jnp.einsum('bsd,bsnd->bsn', x_drop, cand_qk)  # [B, S, n_cand_qk]
        scores_K = jnp.einsum('bsd,bsnd->bsn', x_drop, cand_qk)
        scores_V = jnp.einsum('bsd,bsnd->bsn', x_drop, cand_v)   # [B, S, n_cand_v]

        # Tau
        tau_all = self.tau_attn(x)  # [B, S, 3]
        tau_Q = tau_all[:, :, 0:1]
        tau_K = tau_all[:, :, 1:2]
        tau_V = tau_all[:, :, 2:3]

        # Threshold gate (within candidates only)
        gate_Q = threshold_gate(scores_Q, tau_Q, self.max_k_qk)
        gate_K = threshold_gate(scores_K, tau_K, self.max_k_qk)
        gate_V = threshold_gate(scores_V, tau_V, self.max_k_v)

        # --- pos_loss: reuse distances from candidate selection ---
        cand_dist_qk = jnp.take_along_axis(dist_qk, cand_idx_qk, axis=-1)
        pos_loss_qk = (jax.lax.stop_gradient(gate_Q) * cand_dist_qk).mean()

        cand_dist_v = jnp.take_along_axis(dist_v, cand_idx_v, axis=-1)
        pos_loss_v = (jax.lax.stop_gradient(gate_V) * cand_dist_v).mean()

        pos_loss = pos_loss_qk + pos_loss_v

        return gate_Q, gate_K, gate_V, cand_idx_qk, cand_idx_v, pos_loss

    def get_knowledge_gates(self, x, neuron_pool, deterministic, rng):
        """Compute knowledge gates via 2D positional routing.

        Returns: (gate_know, cand_idx_know, pos_loss)
          gate_know:     [B, S, n_cand_know]
          cand_idx_know: [B, S, n_cand_know]
          pos_loss:      scalar
        """
        B, S, D = x.shape
        know_neurons = neuron_pool.know_neurons  # [N_know, D]

        know_pos = self.proj_pos_know(x)  # [B, S, 2]
        npos_know = self.neuron_pos[self.n_qk + self.n_v:]  # [N_know, 2]

        n_cand = self.max_k_know * self.candidates_multiplier

        # Direct distance + top_k (init path uses small batch, no OOM)
        dist = jnp.sum(
            (know_pos[:, :, None, :] - npos_know[None, None, :, :]) ** 2,
            axis=-1)  # [B, S, N_know]
        _, cand_idx = jax.lax.top_k(-dist, n_cand)
        cand_idx = jax.lax.stop_gradient(cand_idx)

        # Precise scoring
        cand_neurons = know_neurons[cand_idx]  # [B, S, n_cand, D]

        rng, rng1 = jax.random.split(rng)
        x_drop = safe_dropout(x, self.router_dropout, deterministic, rng1)
        scores = jnp.einsum('bsd,bsnd->bsn', x_drop, cand_neurons)

        tau = self.tau_know(x)  # [B, S, 1]
        gate = threshold_gate(scores, tau, self.max_k_know)

        # pos_loss: recompute distance for gradient flow to neuron_pos & proj_pos
        cand_npos = npos_know[cand_idx]  # [B, S, n_cand, 2]
        cand_dist_grad = jnp.sum(
            (know_pos[:, :, None, :] - cand_npos) ** 2,
            axis=-1)  # [B, S, n_cand] — small tensor, OK
        pos_loss = (jax.lax.stop_gradient(gate) * cand_dist_grad).mean()

        return gate, cand_idx, pos_loss


# ================================================================
# 8. Chunked pipeline functions for jax.lax.scan forward path
#    All pipelines use carry accumulation (dynamic_update_slice)
#    to avoid scan output stacking OOM.
# ================================================================

def _attn_pipeline_chunked(
    # --- dynamic (tensors) ---
    x, qk_pos, v_pos, tau_Q, tau_K, tau_V,
    qk_neurons, v_neurons, npos_qk, npos_v,
    expand_O_kernel, rng,
    # --- static (int/float/bool) ---
    max_k_qk, max_k_v, candidates_multiplier,
    n_heads, d_model,
    router_dropout, dropout_rate, deterministic,
    chunk_size,
):
    """Attention pipeline: Q/K/V sense_emit chunked, self-attention full sequence.

    Gather tensors only exist within each chunk. Q/K/V buffers are [B,S,D] (small).
    No jax.checkpoint here -- outer scan_body handles that if needed.
    """
    B, S, D = x.shape
    n_cand_qk = max_k_qk * candidates_multiplier
    n_cand_v = max_k_v * candidates_multiplier
    pos_dim = qk_pos.shape[-1]

    # Pad sequence to chunk_size multiple
    pad_S = ((S + chunk_size - 1) // chunk_size) * chunk_size
    if pad_S > S:
        pad_seq = ((0, 0), (0, pad_S - S), (0, 0))
        x = jnp.pad(x, pad_seq)
        qk_pos = jnp.pad(qk_pos, pad_seq[:2] + ((0, 0),))
        v_pos = jnp.pad(v_pos, pad_seq[:2] + ((0, 0),))
        tau_Q = jnp.pad(tau_Q, pad_seq[:2] + ((0, 0),))
        tau_K = jnp.pad(tau_K, pad_seq[:2] + ((0, 0),))
        tau_V = jnp.pad(tau_V, pad_seq[:2] + ((0, 0),))

    n_chunks = pad_S // chunk_size
    Q_buf = jnp.zeros((B, pad_S, D))
    K_buf = jnp.zeros((B, pad_S, D))
    V_buf = jnp.zeros((B, pad_S, D))
    total_pos_loss = jnp.float32(0.0)
    chunk_rngs = jax.random.split(rng, n_chunks)

    def process_chunk(carry, chunk_idx):
        Q_buf, K_buf, V_buf, total_pos_loss = carry
        start = chunk_idx * chunk_size

        x_c = jax.lax.dynamic_slice(x, (0, start, 0), (B, chunk_size, D))
        qk_pos_c = jax.lax.dynamic_slice(
            qk_pos, (0, start, 0), (B, chunk_size, pos_dim))
        v_pos_c = jax.lax.dynamic_slice(
            v_pos, (0, start, 0), (B, chunk_size, pos_dim))
        tau_Q_c = jax.lax.dynamic_slice(
            tau_Q, (0, start, 0), (B, chunk_size, 1))
        tau_K_c = jax.lax.dynamic_slice(
            tau_K, (0, start, 0), (B, chunk_size, 1))
        tau_V_c = jax.lax.dynamic_slice(
            tau_V, (0, start, 0), (B, chunk_size, 1))
        rng_c = chunk_rngs[chunk_idx]

        # --- QK candidates ---
        dist_qk = jnp.sum(
            (qk_pos_c[:, :, None, :] - npos_qk[None, None, :, :]) ** 2,
            axis=-1)
        _, cand_idx_qk = jax.lax.top_k(-dist_qk, n_cand_qk)
        cand_idx_qk = jax.lax.stop_gradient(cand_idx_qk)
        cand_qk = qk_neurons[cand_idx_qk]  # [B, cs, n_cand_qk, D]

        # --- V candidates ---
        dist_v = jnp.sum(
            (v_pos_c[:, :, None, :] - npos_v[None, None, :, :]) ** 2,
            axis=-1)
        _, cand_idx_v = jax.lax.top_k(-dist_v, n_cand_v)
        cand_idx_v = jax.lax.stop_gradient(cand_idx_v)
        cand_v = v_neurons[cand_idx_v]  # [B, cs, n_cand_v, D]

        # --- Score ---
        rng_c, rng_drop = jax.random.split(rng_c)
        x_drop = safe_dropout(x_c, router_dropout, deterministic, rng_drop)
        scores_Q = jnp.einsum('bsd,bsnd->bsn', x_drop, cand_qk)
        scores_K = jnp.einsum('bsd,bsnd->bsn', x_drop, cand_qk)
        scores_V = jnp.einsum('bsd,bsnd->bsn', x_drop, cand_v)

        # --- Gate ---
        gate_Q = threshold_gate(scores_Q, tau_Q_c, max_k_qk)
        gate_K = threshold_gate(scores_K, tau_K_c, max_k_qk)
        gate_V = threshold_gate(scores_V, tau_V_c, max_k_v)

        # --- Sense-emit Q/K/V (inline) ---
        act_Q = jnp.einsum('bsd,bsnd->bsn', x_c, cand_qk)
        Q_chunk = jnp.einsum('bsn,bsnd->bsd', act_Q * gate_Q, cand_qk)

        act_K = jnp.einsum('bsd,bsnd->bsn', x_c, cand_qk)
        K_chunk = jnp.einsum('bsn,bsnd->bsd', act_K * gate_K, cand_qk)

        act_V = jnp.einsum('bsd,bsnd->bsn', x_c, cand_v)
        V_chunk = jnp.einsum('bsn,bsnd->bsd', act_V * gate_V, cand_v)

        # --- pos_loss ---
        pos_dist_qk = jnp.sum(
            (qk_pos_c[:, :, None, :] - npos_qk[cand_idx_qk]) ** 2,
            axis=-1)
        pos_loss_qk = (jax.lax.stop_gradient(gate_Q) * pos_dist_qk).mean()

        pos_dist_v = jnp.sum(
            (v_pos_c[:, :, None, :] - npos_v[cand_idx_v]) ** 2,
            axis=-1)
        pos_loss_v = (jax.lax.stop_gradient(gate_V) * pos_dist_v).mean()

        # --- Accumulate into carry ---
        Q_buf = jax.lax.dynamic_update_slice(Q_buf, Q_chunk, (0, start, 0))
        K_buf = jax.lax.dynamic_update_slice(K_buf, K_chunk, (0, start, 0))
        V_buf = jax.lax.dynamic_update_slice(V_buf, V_chunk, (0, start, 0))
        total_pos_loss = total_pos_loss + pos_loss_qk + pos_loss_v

        return (Q_buf, K_buf, V_buf, total_pos_loss), None

    (Q_buf, K_buf, V_buf, total_pos_loss), _ = jax.lax.scan(
        process_chunk,
        (Q_buf, K_buf, V_buf, total_pos_loss),
        jnp.arange(n_chunks))

    # Remove padding
    Q = Q_buf[:, :S, :]
    K = K_buf[:, :S, :]
    V = V_buf[:, :S, :]
    pos_loss = total_pos_loss / n_chunks

    # === Self-attention (full sequence, Q/K/V are [B,S,D] = small) ===
    d_head = d_model // n_heads
    Q = Q.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
    K = K.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
    V = V.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)

    scale = jnp.sqrt(jnp.float32(d_head))
    attn_scores = jnp.einsum('bhsd,bhtd->bhst', Q, K) / scale
    causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
    attn_scores = jnp.where(causal, attn_scores,
                            jnp.finfo(attn_scores.dtype).min)
    attn_w = jax.nn.softmax(attn_scores, axis=-1)

    rng_attn, rng_out = jax.random.split(rng)
    attn_w = safe_dropout(attn_w, dropout_rate, deterministic, rng_attn)

    out = jnp.einsum('bhst,bhtd->bhsd', attn_w, V)
    out = out.transpose(0, 2, 1, 3).reshape(B, S, D)
    out = out @ expand_O_kernel
    out = safe_dropout(out, dropout_rate, deterministic, rng_out)

    return out, pos_loss


def _know_pipeline_chunked(
    # --- dynamic (tensors) ---
    x, know_pos, tau,
    know_neurons, npos_know, rng,
    # --- static (int/float/bool) ---
    max_k_know, candidates_multiplier,
    router_dropout, dropout_rate, deterministic,
    chunk_size,
):
    """Know pipeline fully chunked over sequence axis.

    Uses carry accumulation (dynamic_update_slice). Scan returns None output.
    No jax.checkpoint here -- outer scan_body handles that if needed.

    Peak memory per chunk: [B, chunk_size, n_cand, D].
    """
    B, S, D = x.shape
    n_cand = max_k_know * candidates_multiplier
    pos_dim = know_pos.shape[-1]

    # Pad sequence to chunk_size multiple
    pad_S = ((S + chunk_size - 1) // chunk_size) * chunk_size
    if pad_S > S:
        x = jnp.pad(x, ((0, 0), (0, pad_S - S), (0, 0)))
        know_pos = jnp.pad(know_pos, ((0, 0), (0, pad_S - S), (0, 0)))
        tau = jnp.pad(tau, ((0, 0), (0, pad_S - S), (0, 0)))

    n_chunks = pad_S // chunk_size
    out_buf = jnp.zeros((B, pad_S, D))
    total_pos_loss = jnp.float32(0.0)
    chunk_rngs = jax.random.split(rng, n_chunks)

    def process_chunk(carry, chunk_idx):
        out_buf, total_pos_loss = carry
        start = chunk_idx * chunk_size

        x_c = jax.lax.dynamic_slice(x, (0, start, 0), (B, chunk_size, D))
        pos_c = jax.lax.dynamic_slice(
            know_pos, (0, start, 0), (B, chunk_size, pos_dim))
        tau_c = jax.lax.dynamic_slice(tau, (0, start, 0), (B, chunk_size, 1))
        rng_c = chunk_rngs[chunk_idx]

        # 1. Distance -> candidate selection
        dist = jnp.sum(
            (pos_c[:, :, None, :] - npos_know[None, None, :, :]) ** 2,
            axis=-1)
        _, cand_idx = jax.lax.top_k(-dist, n_cand)
        cand_idx = jax.lax.stop_gradient(cand_idx)

        # 2. Gather + Score
        cand_neurons = know_neurons[cand_idx]
        rng_c, rng_drop = jax.random.split(rng_c)
        x_drop = safe_dropout(x_c, router_dropout, deterministic, rng_drop)
        scores = jnp.einsum('bsd,bsnd->bsn', x_drop, cand_neurons)

        # 3. Gate
        gate = threshold_gate(scores, tau_c, max_k_know)

        # 4. Sense-emit (inline)
        activations = jnp.einsum('bsd,bsnd->bsn', x_c, cand_neurons)
        gated = activations * gate
        chunk_out = jnp.einsum('bsn,bsnd->bsd', gated, cand_neurons)
        rng_c, rng_out = jax.random.split(rng_c)
        chunk_out = safe_dropout(
            chunk_out, dropout_rate, deterministic, rng_out)

        # 5. pos_loss
        cand_npos = npos_know[cand_idx]
        pos_dist = jnp.sum(
            (pos_c[:, :, None, :] - cand_npos) ** 2, axis=-1)
        chunk_pos_loss = (jax.lax.stop_gradient(gate) * pos_dist).mean()

        # Accumulate into carry
        out_buf = jax.lax.dynamic_update_slice(
            out_buf, chunk_out, (0, start, 0))
        total_pos_loss = total_pos_loss + chunk_pos_loss

        return (out_buf, total_pos_loss), None

    (out_buf, total_pos_loss), _ = jax.lax.scan(
        process_chunk,
        (out_buf, total_pos_loss),
        jnp.arange(n_chunks))

    return out_buf[:, :S, :], total_pos_loss / n_chunks


# ================================================================
# 9. Flax modules (used during init; scan uses pure functions above)
# ================================================================

class AttentionCircuit(nn.Module):
    """Rank-1 sparse attention: Q/K/V via inline sense_emit + causal self-attention."""
    d_model: int
    n_heads: int
    dropout_rate: float = 0.1

    def setup(self):
        self.expand_O = nn.Dense(
            self.d_model, use_bias=False, kernel_init=scaled_normal(0.02))

    def __call__(self, x, neuron_pool, router, attention_mask, deterministic):
        rng = self.make_rng('dropout')
        rng, rng_router, rng_drop, rng_out = jax.random.split(rng, 4)

        gate_Q, gate_K, gate_V, cand_qk, cand_v, aux = \
            router.get_attention_gates(x, neuron_pool, deterministic, rng_router)

        # Inline sense_emit_sparse (function was removed in v2.0.6)
        def _sense_emit(x, neurons, gates, cand_idx):
            cn = neurons[cand_idx]
            act = jnp.einsum('bsd,bsnd->bsn', x, cn)
            return jnp.einsum('bsn,bsnd->bsd', act * gates, cn)

        Q = _sense_emit(x, neuron_pool.qk_neurons, gate_Q, cand_qk)
        K = _sense_emit(x, neuron_pool.qk_neurons, gate_K, cand_qk)
        V = _sense_emit(x, neuron_pool.v_neurons, gate_V, cand_v)

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
    """Rank-1 sparse knowledge: FFN-equivalent via inline sense_emit."""
    d_model: int
    dropout_rate: float = 0.1

    def __call__(self, x, neuron_pool, router, attention_mask, deterministic):
        rng = self.make_rng('dropout')
        rng, rng_router = jax.random.split(rng)

        gate, cand_idx, aux = router.get_knowledge_gates(
            x, neuron_pool, deterministic, rng_router)

        # Inline sense_emit_sparse
        cn = neuron_pool.know_neurons[cand_idx]
        act = jnp.einsum('bsd,bsnd->bsn', x, cn)
        out = jnp.einsum('bsn,bsnd->bsd', act * gate, cn)

        out = safe_dropout(out, self.dropout_rate, deterministic, rng)
        return out, aux


class DAWNBlock(nn.Module):
    """Single DAWN block: norm -> attn -> residual -> norm -> knowledge -> residual."""
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
    """DAWN-Spatial v2: Rank-1 Neuron Architecture with 2D Positional Routing.

    Each neuron is a single vector [D].
    Transform per circuit: out = V^T diag(gate) V x (sparse, candidate-only).
    Routing: 2D positional distance -> candidate selection -> dot-product scoring.
    Uses jax.lax.scan for O(1) XLA compile, jax.checkpoint for memory.
    Weight tying: lm_head reuses token_emb via nn.Embed.attend().
    """
    __version__ = "spatial-r1-v2.0.6"

    vocab_size: int = 30000
    d_model: int = 384
    n_layers: int = 12
    n_heads: int = 6
    max_seq_len: int = 512
    dropout_rate: float = 0.1
    gradient_checkpointing: bool = False

    # 2D spatial routing
    pos_dim: int = 2
    grid_size: int = 64
    candidates_multiplier: int = 2
    grid_rebuild_interval: int = 100
    pos_loss_weight: float = 0.01
    know_chunk_size: int = 16

    # Balanced neuron pools (ratio 1 : 1.5 : 5)
    n_qk: int = 3140
    n_v: int = 5240
    n_know: int = 42000

    # Per-circuit max_k
    max_k_qk: int = 157
    max_k_v: int = 262
    max_k_know: int = 1536

    router_dropout: float = 0.1

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
            d_model=self.d_model)

        self.router = Router(
            d_model=self.d_model, pos_dim=self.pos_dim,
            n_qk=self.n_qk, n_v=self.n_v, n_know=self.n_know,
            max_k_qk=self.max_k_qk, max_k_v=self.max_k_v,
            max_k_know=self.max_k_know,
            candidates_multiplier=self.candidates_multiplier,
            router_dropout=self.router_dropout)

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

            # Pre-slice neuron_pos outside checkpoint (static indices)
            neuron_pos = router_params['neuron_pos']
            npos_qk = neuron_pos[:self.n_qk]
            npos_v = neuron_pos[self.n_qk:self.n_qk + self.n_v]
            npos_know = neuron_pos[self.n_qk + self.n_v:]

            def scan_body(carry, xs):
                x = carry
                bp = xs['params']
                rng = xs['rng']
                rng, rng_attn, rng_know = jax.random.split(rng, 3)

                # --- Attention sub-block (chunked Q/K/V + full self-attn) ---
                normed = _layer_norm(
                    x, bp['norm1']['scale'], bp['norm1']['bias'])

                # Compute small tensors (pos, tau) outside chunked pipeline
                qk_pos = (normed @ router_params['proj_pos_qk']['kernel']
                          + router_params['proj_pos_qk']['bias'])
                v_pos = (normed @ router_params['proj_pos_v']['kernel']
                         + router_params['proj_pos_v']['bias'])
                tau_all = (normed @ router_params['tau_attn']['kernel']
                           + router_params['tau_attn']['bias'])
                tau_Q = tau_all[:, :, 0:1]
                tau_K = tau_all[:, :, 1:2]
                tau_V = tau_all[:, :, 2:3]

                attn_out, attn_aux = _attn_pipeline_chunked(
                    normed, qk_pos, v_pos, tau_Q, tau_K, tau_V,
                    pool_params['qk_neurons'], pool_params['v_neurons'],
                    npos_qk, npos_v,
                    bp['attn']['expand_O']['kernel'], rng_attn,
                    # static
                    self.max_k_qk, self.max_k_v,
                    self.candidates_multiplier,
                    self.n_heads, self.d_model,
                    self.router_dropout, self.dropout_rate,
                    deterministic, self.know_chunk_size)

                x = x + attn_out

                # --- Knowledge sub-block (fully chunked) ---
                normed = _layer_norm(
                    x, bp['norm2']['scale'], bp['norm2']['bias'])

                know_pos = (normed @ router_params['proj_pos_know']['kernel']
                            + router_params['proj_pos_know']['bias'])
                tau_know = (normed @ router_params['tau_know']['kernel']
                            + router_params['tau_know']['bias'])

                know_out, know_aux = _know_pipeline_chunked(
                    normed, know_pos, tau_know,
                    pool_params['know_neurons'], npos_know, rng_know,
                    # static
                    self.max_k_know, self.candidates_multiplier,
                    self.router_dropout, self.dropout_rate,
                    deterministic, self.know_chunk_size)

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
            # Loss+accuracy under checkpoint -- avoids materializing [B,S,V]
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
        """Encourage diversity among neurons via cosine similarity penalty.

        For large pools (>4096), uses deterministic strided sampling
        to avoid O(N^2). No rng needed — safe outside forward pass.
        """
        def _pool_div(neurons, max_sample=4096):
            N = neurons.shape[0]
            if N > max_sample:
                # Deterministic strided sampling (no rng needed)
                stride = N // max_sample
                neurons = neurons[::stride][:max_sample]
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
            'max_k_qk': self.max_k_qk, 'max_k_v': self.max_k_v,
            'max_k_know': self.max_k_know,
            'pos_dim': self.pos_dim,
            'grid_size': self.grid_size,
            'candidates_multiplier': self.candidates_multiplier,
            'grid_rebuild_interval': self.grid_rebuild_interval,
            'pos_loss_weight': self.pos_loss_weight,
            'know_chunk_size': self.know_chunk_size,
        }

    def get_model_info(self):
        n_cand_qk = self.max_k_qk * self.candidates_multiplier
        n_cand_v = self.max_k_v * self.candidates_multiplier
        n_cand_know = self.max_k_know * self.candidates_multiplier
        return [
            f"DAWN v{self.__version__}: Rank-1 Neuron + 2D Positional Routing (JAX)",
            f"  d_model={self.d_model}, n_layers={self.n_layers}, "
            f"n_heads={self.n_heads}",
            f"  max_seq_len={self.max_seq_len}, dropout={self.dropout_rate}",
            f"",
            f"  [Neuron Pool -- Rank-1 vectors]",
            f"  QK neurons: {self.n_qk} x {self.d_model}  "
            f"(shared Q/K, max_k={self.max_k_qk})",
            f"  V neurons:  {self.n_v} x {self.d_model}  "
            f"(max_k={self.max_k_v})",
            f"  Know neurons: {self.n_know} x {self.d_model}  "
            f"(max_k={self.max_k_know})",
            f"",
            f"  [2D Positional Routing]",
            f"  pos_dim={self.pos_dim}, candidates_multiplier={self.candidates_multiplier}",
            f"  QK: {self.n_qk} neurons -> {n_cand_qk} candidates -> top-{self.max_k_qk}",
            f"  V:  {self.n_v} neurons -> {n_cand_v} candidates -> top-{self.max_k_v}",
            f"  Know: {self.n_know} neurons -> {n_cand_know} candidates -> top-{self.max_k_know}",
            f"  router_dropout={self.router_dropout}",
            f"",
            f"  [Memory Optimization]",
            f"  Chunked pipelines: carry accumulation + dynamic_update_slice",
            f"  gradient_checkpointing={self.gradient_checkpointing}",
            f"  jax.lax.scan layer loop (O(1) XLA compile)",
        ]
