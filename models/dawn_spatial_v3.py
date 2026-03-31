"""
DAWN-Spatial v3: Rank-1 Neuron Architecture with Cell-Dense Routing (JAX/Flax)

Changelog:
  spatial-r1-v3.1.0 (2026-03-31):
    - Cell-dense routing: block_data stores original neuron indices per cell
      (not neuron copies). Actual neurons gathered from latest params.
    - _attn_dense: QK/V full sequence (small candidate count)
    - _know_dense_chunked: Python for loop + jax.checkpoint per chunk
    - No distance computation. cell_id integer division + 3x3 neighbor lookup.
    - block_data rebuilt every map_rebuild_interval steps (index map only)

  spatial-r1-v3.0.0 (2026-03-31):
    - Map-based routing (cell lookup + gather + chunked). 30s/step.

Key concept:
  - Every neuron is a single vector v_i [D] (rank-1).
  - Transform: out = V^T diag(gate) V x
  - Cell-based routing: neuron_pos -> cell assignment -> index map
  - Forward: query_pos -> cell_id -> neighbor indices -> original gather

Architecture:
  build_cell_index_map  -- numpy CPU: neuron_pos -> index-only lookup table
  get_cell_block_indices -- JAX: query_pos -> original neuron indices
  _attn_dense           -- full sequence QK/V, block index -> original gather
  _know_dense_chunked   -- chunked Know, Python loop + checkpoint
  DAWN                  -- embedding + jax.lax.scan + weight-tied lm_head
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
import numpy as np
from typing import Optional, Dict


# ================================================================
# 1. Trace-safe dropout
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
# 2. Pure functional LayerNorm
# ================================================================

def _layer_norm(x, scale, bias, eps=1e-6):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * scale + bias


# ================================================================
# 3. Initializers
# ================================================================

def scaled_normal(scale=0.02):
    def init(key, shape, dtype=jnp.float32):
        return jax.random.normal(key, shape, dtype) * scale
    return init


def unit_norm_init(scale=1.0):
    def init(key, shape, dtype=jnp.float32):
        x = jax.random.normal(key, shape, dtype)
        norms = jnp.linalg.norm(x, axis=-1, keepdims=True) + 1e-8
        return x / norms * scale
    return init


# ================================================================
# 4. Threshold gate
# ================================================================

def threshold_gate(scores, tau, max_k=None):
    """Threshold-based gating with dead-neuron gradient flow."""
    raw_gate = scores - tau
    gate = jnp.where(raw_gate > 0, raw_gate, 1e-8 * jnp.exp(raw_gate))
    exp_gate = jnp.exp(gate) - 1.0

    if max_k is not None:
        # Clamp max_k to available candidates (safety for cell-map routing)
        n_cand = scores.shape[-1]
        effective_k = min(max_k, n_cand)
        topk_vals, _ = jax.lax.top_k(exp_gate, effective_k)
        threshold = topk_vals[:, :, -1:]
        exp_gate = jnp.where(exp_gate >= threshold, exp_gate, 0.0)

    gate_sum = exp_gate.sum(axis=-1, keepdims=True) + 1e-8
    gate_strength = jnp.tanh(exp_gate.max(axis=-1, keepdims=True))
    return (exp_gate / gate_sum) * gate_strength


# ================================================================
# 5. Cell Index Map Builder (numpy, CPU -- called outside JIT)
#    Stores original neuron indices per cell, NOT neuron copies.
# ================================================================

def build_cell_index_map(neuron_pos_np, n_cells_per_side, block_size):
    """Build cell index map: which original neuron indices are in each cell.

    Args:
        neuron_pos_np: [N, 2] numpy array
        n_cells_per_side: grid resolution per axis
        block_size: max neurons per cell (overflow truncated)

    Returns:
        cell_idx: [n_cells, block_size] int32 (-1 = padding)
        cell_valid: [n_cells, block_size] bool
        pos_min: [2], pos_range: [2]
    """
    N = neuron_pos_np.shape[0]
    n_cells = n_cells_per_side ** 2

    pos_min = neuron_pos_np.min(axis=0)
    pos_range = neuron_pos_np.max(axis=0) - pos_min + 1e-8
    normalized = (neuron_pos_np - pos_min) / pos_range

    cell_xy = np.clip(
        (normalized * n_cells_per_side).astype(np.int32),
        0, n_cells_per_side - 1)
    cell_ids = cell_xy[:, 0] * n_cells_per_side + cell_xy[:, 1]

    cell_idx = np.full((n_cells, block_size), -1, dtype=np.int32)
    cell_valid = np.zeros((n_cells, block_size), dtype=np.bool_)
    counts = np.zeros(n_cells, dtype=np.int32)

    for i in range(N):
        cid = cell_ids[i]
        if counts[cid] < block_size:
            cell_idx[cid, counts[cid]] = i  # original index
            cell_valid[cid, counts[cid]] = True
            counts[cid] += 1

    return cell_idx, cell_valid, pos_min.astype(np.float32), pos_range.astype(np.float32)


def build_all_blocks(params, n_qk, n_v, n_cells_per_side,
                     max_k_qk=157, max_k_v=262, max_k_know=1536):
    """Build block_data (index maps) for all pools. Called outside JIT.

    block_data contains only original neuron indices per cell.
    Actual neuron vectors are gathered from latest params at forward time.
    """
    neuron_pos = np.array(params['router']['neuron_pos'])
    pos_qk = neuron_pos[:n_qk]
    pos_v = neuron_pos[n_qk:n_qk + n_v]
    pos_know = neuron_pos[n_qk + n_v:]

    n_cells = n_cells_per_side ** 2
    # block_size: ensure 9*block >= max_k, also ~4x avg
    bs_qk = max(max_k_qk // 9 + 4, n_qk // n_cells * 4, 16)
    bs_v = max(max_k_v // 9 + 4, n_v // n_cells * 4, 16)
    bs_know = max(max_k_know // 9 + 4, len(pos_know) // n_cells * 4, 64)

    idx_qk, val_qk, pmin_qk, prng_qk = build_cell_index_map(
        pos_qk, n_cells_per_side, bs_qk)
    idx_v, val_v, pmin_v, prng_v = build_cell_index_map(
        pos_v, n_cells_per_side, bs_v)
    idx_know, val_know, pmin_know, prng_know = build_cell_index_map(
        pos_know, n_cells_per_side, bs_know)

    return {
        'qk_idx': jnp.array(idx_qk),        # [n_cells, bs_qk]
        'qk_valid': jnp.array(val_qk),
        'v_idx': jnp.array(idx_v),
        'v_valid': jnp.array(val_v),
        'know_idx': jnp.array(idx_know),
        'know_valid': jnp.array(val_know),
        'pos_min_qk': jnp.array(pmin_qk),
        'pos_range_qk': jnp.array(prng_qk),
        'pos_min_v': jnp.array(pmin_v),
        'pos_range_v': jnp.array(prng_v),
        'pos_min_know': jnp.array(pmin_know),
        'pos_range_know': jnp.array(prng_know),
    }


# ================================================================
# 6. Cell Block Index Lookup (JAX, inside JIT)
# ================================================================

_NEIGHBOR_OFFSETS = jnp.array([
    [-1, -1], [-1, 0], [-1, 1],
    [0, -1],  [0, 0],  [0, 1],
    [1, -1],  [1, 0],  [1, 1],
], dtype=jnp.int32)  # [9, 2]


def get_cell_block_indices(query_pos, cell_idx_map, cell_valid_map,
                           pos_min, pos_range, n_cells_per_side):
    """Look up original neuron indices from cell index map.

    Args:
        query_pos:      [B, S, 2]
        cell_idx_map:   [n_cells, block_size] int32 (original indices, -1=pad)
        cell_valid_map: [n_cells, block_size] bool
        pos_min, pos_range: normalization params
        n_cells_per_side: int

    Returns:
        cand_idx:   [B, S, 9*block_size] int32 (safe, 0 for invalid)
        valid_mask: [B, S, 9*block_size] bool
    """
    B, S, _ = query_pos.shape
    block_size = cell_idx_map.shape[1]

    normalized = (query_pos - pos_min) / pos_range
    cell_xy = jnp.clip(
        (normalized * n_cells_per_side).astype(jnp.int32),
        0, n_cells_per_side - 1)

    neighbor_xy = cell_xy[:, :, None, :] + _NEIGHBOR_OFFSETS[None, None, :, :]
    neighbor_xy = jnp.clip(neighbor_xy, 0, n_cells_per_side - 1)
    neighbor_cell = (neighbor_xy[:, :, :, 0] * n_cells_per_side
                     + neighbor_xy[:, :, :, 1])  # [B, S, 9]

    # Gather index blocks: [B, S, 9, block_size]
    cand_idx = cell_idx_map[neighbor_cell]
    valid_mask = cell_valid_map[neighbor_cell]

    # Flatten
    cand_idx = cand_idx.reshape(B, S, 9 * block_size)
    valid_mask = valid_mask.reshape(B, S, 9 * block_size)

    # Safe indices (replace -1 with 0 for gather safety)
    cand_idx = jnp.where(valid_mask, cand_idx, 0)

    return cand_idx, valid_mask


# ================================================================
# 8. NeuronPool -- shared rank-1 vectors
# ================================================================

class NeuronPool(nn.Module):
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
# 8. Router -- 2D positional projections + tau
# ================================================================

class Router(nn.Module):
    """Router: produces query positions and tau. Cell lookup in pipelines."""
    d_model: int
    pos_dim: int
    n_qk: int
    n_v: int
    n_know: int
    max_k_qk: int
    max_k_v: int
    max_k_know: int
    router_dropout: float = 0.1

    def setup(self):
        total = self.n_qk + self.n_v + self.n_know
        self.neuron_pos = self.param(
            'neuron_pos', scaled_normal(1.0), (total, self.pos_dim))
        self.proj_pos_qk = nn.Dense(self.pos_dim, name='proj_pos_qk')
        self.proj_pos_v = nn.Dense(self.pos_dim, name='proj_pos_v')
        self.proj_pos_know = nn.Dense(self.pos_dim, name='proj_pos_know')
        self.tau_attn = nn.Dense(3, name='tau_attn')
        self.tau_know = nn.Dense(1, name='tau_know')

    def get_attention_gates(self, x, neuron_pool, deterministic, rng):
        """Init path: brute-force fallback (small batch, no OOM)."""
        B, S, D = x.shape
        qk_pos = self.proj_pos_qk(x)
        v_pos = self.proj_pos_v(x)
        npos_qk = self.neuron_pos[:self.n_qk]
        npos_v = self.neuron_pos[self.n_qk:self.n_qk + self.n_v]

        # Brute-force distance (init only)
        n_cand_qk = min(self.max_k_qk * 2, self.n_qk)
        dist_qk = jnp.sum(
            (qk_pos[:, :, None, :] - npos_qk[None, None, :, :]) ** 2, axis=-1)
        _, ci_qk = jax.lax.top_k(-dist_qk, n_cand_qk)
        n_cand_v = min(self.max_k_v * 2, self.n_v)
        dist_v = jnp.sum(
            (v_pos[:, :, None, :] - npos_v[None, None, :, :]) ** 2, axis=-1)
        _, ci_v = jax.lax.top_k(-dist_v, n_cand_v)

        ci_qk = jax.lax.stop_gradient(ci_qk)
        ci_v = jax.lax.stop_gradient(ci_v)

        cand_qk = neuron_pool.qk_neurons[ci_qk]
        cand_v = neuron_pool.v_neurons[ci_v]

        rng, rng1 = jax.random.split(rng)
        x_drop = safe_dropout(x, self.router_dropout, deterministic, rng1)
        s_Q = jnp.einsum('bsd,bsnd->bsn', x_drop, cand_qk)
        s_K = jnp.einsum('bsd,bsnd->bsn', x_drop, cand_qk)
        s_V = jnp.einsum('bsd,bsnd->bsn', x_drop, cand_v)

        tau_all = self.tau_attn(x)
        g_Q = threshold_gate(s_Q, tau_all[:, :, 0:1], self.max_k_qk)
        g_K = threshold_gate(s_K, tau_all[:, :, 1:2], self.max_k_qk)
        g_V = threshold_gate(s_V, tau_all[:, :, 2:3], self.max_k_v)

        return g_Q, g_K, g_V, ci_qk, ci_v, jnp.float32(0.0)

    def get_knowledge_gates(self, x, neuron_pool, deterministic, rng):
        """Init path: brute-force fallback."""
        know_pos = self.proj_pos_know(x)
        npos_know = self.neuron_pos[self.n_qk + self.n_v:]

        n_cand = min(self.max_k_know * 2, self.n_know)
        dist = jnp.sum(
            (know_pos[:, :, None, :] - npos_know[None, None, :, :]) ** 2,
            axis=-1)
        _, ci = jax.lax.top_k(-dist, n_cand)
        ci = jax.lax.stop_gradient(ci)

        cand = neuron_pool.know_neurons[ci]
        rng, rng1 = jax.random.split(rng)
        x_drop = safe_dropout(x, self.router_dropout, deterministic, rng1)
        scores = jnp.einsum('bsd,bsnd->bsn', x_drop, cand)
        tau = self.tau_know(x)
        gate = threshold_gate(scores, tau, self.max_k_know)

        return gate, ci, jnp.float32(0.0)


# ================================================================
# 9. Dense Pipeline Functions (scan body)
#    block_data has only index maps. Neurons from latest params.
# ================================================================

def _attn_dense(x, router_params, pool_params, block_data,
                expand_O_kernel, rng,
                n_qk, n_v, n_cells_per_side,
                max_k_qk, max_k_v,
                n_heads, d_model,
                router_dropout, dropout_rate, deterministic):
    """Cell-dense attention: block index lookup -> original gather -> dense ops.

    QK/V processed full sequence (small candidate count).
    block_data has only original indices; neurons from pool_params (latest).
    """
    B, S, D = x.shape
    qk_neurons = pool_params['qk_neurons']
    v_neurons = pool_params['v_neurons']

    # Query positions + tau
    qk_pos = x @ router_params['proj_pos_qk']['kernel'] + router_params['proj_pos_qk']['bias']
    v_pos = x @ router_params['proj_pos_v']['kernel'] + router_params['proj_pos_v']['bias']

    # Cell lookup -> original indices (no distance)
    ci_qk, m_qk = get_cell_block_indices(
        qk_pos, block_data['qk_idx'], block_data['qk_valid'],
        block_data['pos_min_qk'], block_data['pos_range_qk'], n_cells_per_side)
    ci_v, m_v = get_cell_block_indices(
        v_pos, block_data['v_idx'], block_data['v_valid'],
        block_data['pos_min_v'], block_data['pos_range_v'], n_cells_per_side)
    ci_qk = jax.lax.stop_gradient(ci_qk)
    ci_v = jax.lax.stop_gradient(ci_v)

    # Gather from ORIGINAL params (latest, gradient flows)
    cand_qk = qk_neurons[ci_qk]   # [B, S, n_cand_qk, D]
    cand_v = v_neurons[ci_v]       # [B, S, n_cand_v, D]

    # Score
    rng, rng_drop = jax.random.split(rng)
    x_drop = safe_dropout(x, router_dropout, deterministic, rng_drop)
    s_Q = jnp.where(m_qk, jnp.einsum('bsd,bsnd->bsn', x_drop, cand_qk), -1e9)
    s_K = jnp.where(m_qk, jnp.einsum('bsd,bsnd->bsn', x_drop, cand_qk), -1e9)
    s_V = jnp.where(m_v, jnp.einsum('bsd,bsnd->bsn', x_drop, cand_v), -1e9)

    # Tau + Gate
    tau_all = x @ router_params['tau_attn']['kernel'] + router_params['tau_attn']['bias']
    g_Q = threshold_gate(s_Q, tau_all[:, :, 0:1], max_k_qk)
    g_K = threshold_gate(s_K, tau_all[:, :, 1:2], max_k_qk)
    g_V = threshold_gate(s_V, tau_all[:, :, 2:3], max_k_v)

    # Sense-emit Q/K/V
    act_Q = jnp.einsum('bsd,bsnd->bsn', x, cand_qk)
    Q = jnp.einsum('bsn,bsnd->bsd', act_Q * g_Q * m_qk, cand_qk)
    act_K = jnp.einsum('bsd,bsnd->bsn', x, cand_qk)
    K = jnp.einsum('bsn,bsnd->bsd', act_K * g_K * m_qk, cand_qk)
    act_V = jnp.einsum('bsd,bsnd->bsn', x, cand_v)
    V = jnp.einsum('bsn,bsnd->bsd', act_V * g_V * m_v, cand_v)

    # pos_loss (gradient flows to original neuron_pos)
    npos_qk = router_params['neuron_pos'][:n_qk]
    npos_v = router_params['neuron_pos'][n_qk:n_qk + n_v]
    pd_qk = jnp.sum((qk_pos[:, :, None, :] - npos_qk[ci_qk]) ** 2, axis=-1)
    pl_qk = (jax.lax.stop_gradient(g_Q) * pd_qk * m_qk).sum() / (m_qk.sum() + 1e-8)
    pd_v = jnp.sum((v_pos[:, :, None, :] - npos_v[ci_v]) ** 2, axis=-1)
    pl_v = (jax.lax.stop_gradient(g_V) * pd_v * m_v).sum() / (m_v.sum() + 1e-8)

    # Self-attention (full sequence)
    d_head = d_model // n_heads
    Q = Q.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
    K = K.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
    V = V.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)

    scale = jnp.sqrt(jnp.float32(d_head))
    attn_scores = jnp.einsum('bhsd,bhtd->bhst', Q, K) / scale
    causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
    attn_scores = jnp.where(causal, attn_scores, jnp.finfo(attn_scores.dtype).min)
    attn_w = jax.nn.softmax(attn_scores, axis=-1)

    rng, rng_attn, rng_out = jax.random.split(rng, 3)
    attn_w = safe_dropout(attn_w, dropout_rate, deterministic, rng_attn)

    out = jnp.einsum('bhst,bhtd->bhsd', attn_w, V)
    out = out.transpose(0, 2, 1, 3).reshape(B, S, D)
    out = out @ expand_O_kernel
    out = safe_dropout(out, dropout_rate, deterministic, rng_out)

    return out, pl_qk + pl_v


def _know_dense_chunked(x, router_params, pool_params, block_data, rng,
                        n_qk, n_v, n_cells_per_side,
                        max_k_know,
                        router_dropout, dropout_rate, deterministic,
                        chunk_size):
    """Cell-dense knowledge: cell lookup full seq + chunked gather/score/gate/emit.

    Python for loop + jax.checkpoint per chunk to avoid scan stacking OOM.
    Neurons gathered from original pool_params (latest, gradient flows).
    """
    B, S, D = x.shape
    know_neurons = pool_params['know_neurons']

    # Query pos + tau (full sequence, small tensors)
    know_pos = (x @ router_params['proj_pos_know']['kernel']
                + router_params['proj_pos_know']['bias'])
    tau = (x @ router_params['tau_know']['kernel']
           + router_params['tau_know']['bias'])

    # Cell lookup (full sequence — index tensors are small)
    ci, vm = get_cell_block_indices(
        know_pos, block_data['know_idx'], block_data['know_valid'],
        block_data['pos_min_know'], block_data['pos_range_know'],
        n_cells_per_side)
    ci = jax.lax.stop_gradient(ci)
    n_cand = ci.shape[-1]
    pos_dim = know_pos.shape[-1]

    # Pad sequence
    pad_S = ((S + chunk_size - 1) // chunk_size) * chunk_size
    if pad_S > S:
        x = jnp.pad(x, ((0, 0), (0, pad_S - S), (0, 0)))
        know_pos = jnp.pad(know_pos, ((0, 0), (0, pad_S - S), (0, 0)))
        tau = jnp.pad(tau, ((0, 0), (0, pad_S - S), (0, 0)))
        ci = jnp.pad(ci, ((0, 0), (0, pad_S - S), (0, 0)))
        vm = jnp.pad(vm, ((0, 0), (0, pad_S - S), (0, 0)))

    n_chunks = pad_S // chunk_size
    npos_know = router_params['neuron_pos'][n_qk + n_v:]
    chunk_rngs = jax.random.split(rng, n_chunks)

    def _know_chunk(x_c, pos_c, tau_c, ci_c, vm_c, rng_c,
                    max_k_know, router_dropout, dropout_rate, deterministic):
        cand = know_neurons[ci_c]  # [B, cs, n_cand, D]
        rng_c, rng_drop = jax.random.split(rng_c)
        x_drop = safe_dropout(x_c, router_dropout, deterministic, rng_drop)
        scores = jnp.where(
            vm_c, jnp.einsum('bsd,bsnd->bsn', x_drop, cand), -1e9)
        gate = threshold_gate(scores, tau_c, max_k_know)
        act = jnp.einsum('bsd,bsnd->bsn', x_c, cand)
        chunk_out = jnp.einsum('bsn,bsnd->bsd', act * gate * vm_c, cand)
        rng_c, rng_out = jax.random.split(rng_c)
        chunk_out = safe_dropout(chunk_out, dropout_rate, deterministic, rng_out)
        # pos_loss
        cand_pos = npos_know[ci_c]
        pd = jnp.sum((pos_c[:, :, None, :] - cand_pos) ** 2, axis=-1)
        pl = (jax.lax.stop_gradient(gate) * pd * vm_c).sum() / (vm_c.sum() + 1e-8)
        return chunk_out, pl

    _know_chunk_ckpt = jax.checkpoint(
        _know_chunk, static_argnums=(6, 7, 8, 9))

    out_parts = []
    total_pl = jnp.float32(0.0)
    for i in range(n_chunks):
        s = i * chunk_size
        x_c = jax.lax.dynamic_slice(x, (0, s, 0), (B, chunk_size, D))
        pos_c = jax.lax.dynamic_slice(know_pos, (0, s, 0), (B, chunk_size, pos_dim))
        tau_c = jax.lax.dynamic_slice(tau, (0, s, 0), (B, chunk_size, 1))
        ci_c = jax.lax.dynamic_slice(ci, (0, s, 0), (B, chunk_size, n_cand))
        vm_c = jax.lax.dynamic_slice(vm, (0, s, 0), (B, chunk_size, n_cand))
        co, pl = _know_chunk_ckpt(
            x_c, pos_c, tau_c, ci_c, vm_c, chunk_rngs[i],
            max_k_know, router_dropout, dropout_rate, deterministic)
        out_parts.append(co)
        total_pl = total_pl + pl

    out = jnp.concatenate(out_parts, axis=1)[:, :S, :]
    return out, total_pl / n_chunks


# ================================================================
# 10. Flax modules (init path only)
# ================================================================

class AttentionCircuit(nn.Module):
    d_model: int
    n_heads: int
    dropout_rate: float = 0.1

    def setup(self):
        self.expand_O = nn.Dense(
            self.d_model, use_bias=False, kernel_init=scaled_normal(0.02))

    def __call__(self, x, neuron_pool, router, attention_mask, deterministic):
        rng = self.make_rng('dropout')
        rng, rng_r, rng_d, rng_o = jax.random.split(rng, 4)
        g_Q, g_K, g_V, ci_qk, ci_v, aux = router.get_attention_gates(
            x, neuron_pool, deterministic, rng_r)

        def _se(x, neurons, g, ci):
            cn = neurons[ci]
            act = jnp.einsum('bsd,bsnd->bsn', x, cn)
            return jnp.einsum('bsn,bsnd->bsd', act * g, cn)

        Q = _se(x, neuron_pool.qk_neurons, g_Q, ci_qk)
        K = _se(x, neuron_pool.qk_neurons, g_K, ci_qk)
        V = _se(x, neuron_pool.v_neurons, g_V, ci_v)

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
        attn_w = safe_dropout(attn_w, self.dropout_rate, deterministic, rng_d)
        out = jnp.einsum('bhst,bhtd->bhsd', attn_w, V)
        out = out.transpose(0, 2, 1, 3).reshape(B, S, D)
        out = self.expand_O(out)
        out = safe_dropout(out, self.dropout_rate, deterministic, rng_o)
        return out, aux


class KnowledgeCircuit(nn.Module):
    d_model: int
    dropout_rate: float = 0.1

    def __call__(self, x, neuron_pool, router, attention_mask, deterministic):
        rng = self.make_rng('dropout')
        rng, rng_r = jax.random.split(rng)
        gate, ci, aux = router.get_knowledge_gates(
            x, neuron_pool, deterministic, rng_r)
        cn = neuron_pool.know_neurons[ci]
        act = jnp.einsum('bsd,bsnd->bsn', x, cn)
        out = jnp.einsum('bsn,bsnd->bsd', act * gate, cn)
        out = safe_dropout(out, self.dropout_rate, deterministic, rng)
        return out, aux


class DAWNBlock(nn.Module):
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
        attn_out, a_aux = self.attn(
            normed, neuron_pool, router, attention_mask, deterministic)
        x = x + attn_out
        normed = self.norm2(x)
        know_out, k_aux = self.knowledge(
            normed, neuron_pool, router, attention_mask, deterministic)
        x = x + know_out
        return x, a_aux + k_aux


# ================================================================
# 11. DAWN Model
# ================================================================

class DAWN(nn.Module):
    """DAWN-Spatial v3.1: Rank-1 Neuron + Cell-Dense Routing."""
    __version__ = "spatial-r1-v3.1.0"

    vocab_size: int = 30000
    d_model: int = 384
    n_layers: int = 12
    n_heads: int = 6
    max_seq_len: int = 512
    dropout_rate: float = 0.1
    gradient_checkpointing: bool = False

    pos_dim: int = 2
    n_cells_per_side: int = 32
    map_rebuild_interval: int = 100
    pos_loss_weight: float = 0.01

    n_qk: int = 3140
    n_v: int = 5240
    n_know: int = 42000
    max_k_qk: int = 157
    max_k_v: int = 262
    max_k_know: int = 1536
    router_dropout: float = 0.1
    chunk_size: int = 64

    def setup(self):
        if self.d_model % self.n_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"n_heads ({self.n_heads})")
        self.token_emb = nn.Embed(
            self.vocab_size, self.d_model, embedding_init=scaled_normal(0.02))
        self.pos_emb = nn.Embed(
            self.max_seq_len, self.d_model, embedding_init=scaled_normal(0.02))
        self.neuron_pool = NeuronPool(
            n_qk=self.n_qk, n_v=self.n_v, n_know=self.n_know,
            d_model=self.d_model)
        self.router = Router(
            d_model=self.d_model, pos_dim=self.pos_dim,
            n_qk=self.n_qk, n_v=self.n_v, n_know=self.n_know,
            max_k_qk=self.max_k_qk, max_k_v=self.max_k_v,
            max_k_know=self.max_k_know, router_dropout=self.router_dropout)
        self.layers = [
            DAWNBlock(d_model=self.d_model, n_heads=self.n_heads,
                      dropout_rate=self.dropout_rate, name=f'block_{i}')
            for i in range(self.n_layers)]
        self.norm = nn.LayerNorm()

    def __call__(self, input_ids, labels=None, attention_mask=None,
                 deterministic=False, block_data=None):
        B, S = input_ids.shape
        if S > self.max_seq_len:
            raise ValueError(f"Sequence length {S} exceeds max_seq_len")

        positions = jnp.arange(S)[jnp.newaxis, :]
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        emb_rng = self.make_rng('dropout')
        x = safe_dropout(x, self.dropout_rate, deterministic, emb_rng)

        if self.is_initializing():
            total_aux = jnp.float32(0.0)
            for layer in self.layers:
                x, aux = layer(x, self.neuron_pool, self.router,
                               attention_mask, deterministic)
                total_aux = total_aux + aux
        else:
            all_params = self.variables['params']
            pool_params = all_params['neuron_pool']
            router_params = all_params['router']

            block_params_list = [all_params[f'block_{i}']
                                 for i in range(self.n_layers)]
            stacked = jax.tree.map(
                lambda *arrays: jnp.stack(arrays), *block_params_list)

            base_rng = self.make_rng('dropout')
            layer_rngs = jax.random.split(base_rng, self.n_layers)

            def scan_body(carry, xs):
                x = carry
                bp = xs['params']
                rng = xs['rng']
                rng, rng_attn, rng_know = jax.random.split(rng, 3)

                # --- Attention (full sequence, cell-dense) ---
                normed = _layer_norm(
                    x, bp['norm1']['scale'], bp['norm1']['bias'])
                attn_out, attn_aux = _attn_dense(
                    normed, router_params, pool_params, block_data,
                    bp['attn']['expand_O']['kernel'], rng_attn,
                    self.n_qk, self.n_v, self.n_cells_per_side,
                    self.max_k_qk, self.max_k_v,
                    self.n_heads, self.d_model,
                    self.router_dropout, self.dropout_rate, deterministic)
                x = x + attn_out

                # --- Knowledge (chunked, cell-dense) ---
                normed = _layer_norm(
                    x, bp['norm2']['scale'], bp['norm2']['bias'])
                know_out, know_aux = _know_dense_chunked(
                    normed, router_params, pool_params, block_data,
                    rng_know,
                    self.n_qk, self.n_v, self.n_cells_per_side,
                    self.max_k_know,
                    self.router_dropout, self.dropout_rate, deterministic,
                    self.chunk_size)
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
            embedding_matrix = self.token_emb.embedding
            shift_x = x[:, :-1, :]
            shift_labels = labels[:, 1:].astype(jnp.int32)
            valid_mask = (shift_labels != -100)

            @jax.checkpoint
            def compute_loss_and_acc(x_chunk, emb, labs, vmask):
                logits = x_chunk @ emb.T
                log_probs = jax.nn.log_softmax(logits, axis=-1)
                safe = jnp.where(vmask, labs, 0)
                tl = -jnp.take_along_axis(
                    log_probs, safe[..., jnp.newaxis], axis=-1).squeeze(-1)
                loss = (tl * vmask).sum() / (vmask.sum() + 1e-8)
                preds = jnp.argmax(logits, axis=-1)
                correct = jnp.sum((preds == labs) & vmask)
                return loss, correct, jnp.sum(vmask)

            loss, correct, valid_count = compute_loss_and_acc(
                shift_x, embedding_matrix, shift_labels, valid_mask)
            result['loss'] = loss
            result['correct'] = correct
            result['valid_count'] = valid_count
        else:
            result['logits'] = self.token_emb.attend(x)

        return result

    def diversity_loss(self):
        def _pool_div(neurons, max_sample=4096):
            N = neurons.shape[0]
            if N > max_sample:
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
        return {'neuron_diversity': self.diversity_loss()}

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
            'n_cells_per_side': self.n_cells_per_side,
            'map_rebuild_interval': self.map_rebuild_interval,
            'chunk_size': self.chunk_size,
        }

    def get_model_info(self):
        return [
            f"DAWN v{self.__version__}: Rank-1 + Cell-Dense Routing (JAX)",
            f"  d_model={self.d_model}, n_layers={self.n_layers}, "
            f"n_heads={self.n_heads}",
            f"  QK: {self.n_qk}, V: {self.n_v}, Know: {self.n_know}",
            f"  grid={self.n_cells_per_side}x{self.n_cells_per_side}, "
            f"chunk_size={self.chunk_size}",
        ]
