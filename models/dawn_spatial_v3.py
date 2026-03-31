"""
DAWN-Spatial v3: Rank-1 Neuron Architecture with Map-Based Routing (JAX/Flax)

Changelog:
  spatial-r1-v3.0.0 (2026-03-31):
    - Replace distance brute-force routing with cell map lookup
    - build_cell_map (numpy, CPU) creates fixed-shape lookup table
    - cell_lookup (JAX) does O(1) grid lookup instead of O(N) distance
    - Both Attn and Know pipelines use cell lookup + chunked gather
    - Remove candidates_multiplier (cell size determines candidates)
    - Add n_cells_per_side, map_rebuild_interval config

  spatial-r1-v2.0.6 (2026-03-31):
    - Both Attn and Know fully chunked with carry accumulation

Key concept:
  - Every neuron is a single vector v_i [D] (rank-1).
  - Sense: activation_i = v_i . x  (scalar)
  - Fire:  gate decides which neurons fire (threshold tau)
  - Emit:  out = sum gate_i * activation_i * v_i

Map-Based Routing:
  - Each neuron has a learnable 2D coordinate neuron_pos [N, 2]
  - 2D space divided into grid cells (n_cells_per_side x n_cells_per_side)
  - cell_map[cell_id] = list of neuron indices in that cell (fixed shape)
  - Forward: query_pos -> cell_id (integer division) -> 3x3 neighbor lookup
  - No distance computation. O(1) candidate selection.
  - cell_map rebuilt periodically (every map_rebuild_interval steps, CPU)

Architecture:
  build_cell_map      -- numpy CPU: neuron_pos -> cell lookup table
  cell_lookup          -- JAX: query_pos -> candidate indices via grid
  sense_emit_mapped    -- rank-1 transform with valid_mask
  NeuronPool           -- shared [N, D] rank-1 vectors
  Router               -- 2D pos projections + tau (no distance logic)
  _attn_pipeline_mapped_chunked  -- cell lookup + chunked Q/K/V + full self-attn
  _know_pipeline_mapped_chunked  -- cell lookup + chunked gather/score/gate/emit
  DAWN                 -- embedding + jax.lax.scan + weight-tied lm_head
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
# 5. Cell Map Builder (numpy, CPU -- called outside JIT)
# ================================================================

def build_cell_map(neuron_pos_np, n_cells_per_side, max_per_cell):
    """Build cell lookup table from neuron positions.

    Args:
        neuron_pos_np: [N, 2] numpy array
        n_cells_per_side: grid resolution per axis
        max_per_cell: max neurons per cell (overflow truncated)

    Returns:
        cell_map: [n_cells, max_per_cell] int32 (-1 = padding)
        cell_counts: [n_cells] int32
        pos_min: [2] float32
        pos_range: [2] float32
    """
    N = neuron_pos_np.shape[0]
    n_cells = n_cells_per_side ** 2

    pos_min = neuron_pos_np.min(axis=0)
    pos_max = neuron_pos_np.max(axis=0)
    pos_range = pos_max - pos_min + 1e-8
    normalized = (neuron_pos_np - pos_min) / pos_range  # [0, 1)

    cell_xy = np.clip(
        (normalized * n_cells_per_side).astype(np.int32),
        0, n_cells_per_side - 1)
    cell_ids = cell_xy[:, 0] * n_cells_per_side + cell_xy[:, 1]

    cell_map = np.full((n_cells, max_per_cell), -1, dtype=np.int32)
    cell_counts = np.zeros(n_cells, dtype=np.int32)

    for i in range(N):
        cid = cell_ids[i]
        if cell_counts[cid] < max_per_cell:
            cell_map[cid, cell_counts[cid]] = i
            cell_counts[cid] += 1

    return cell_map, cell_counts, pos_min.astype(np.float32), pos_range.astype(np.float32)


def build_all_cell_maps(params, n_qk, n_v, n_cells_per_side,
                        max_k_qk=157, max_k_v=262, max_k_know=1536):
    """Build cell maps for all three pools. Called outside JIT.

    Args:
        params: model params dict (has 'router'/'neuron_pos')
        n_qk, n_v: pool sizes
        n_cells_per_side: grid resolution
        max_k_qk, max_k_v, max_k_know: per-circuit max_k (for sizing)

    Returns:
        dict of JAX arrays (fixed shapes, no recompilation on rebuild)
    """
    neuron_pos = np.array(params['router']['neuron_pos'])
    pos_qk = neuron_pos[:n_qk]
    pos_v = neuron_pos[n_qk:n_qk + n_v]
    pos_know = neuron_pos[n_qk + n_v:]

    n_cells = n_cells_per_side ** 2
    # max_per_cell must ensure 9*max_per_cell >= max_k for each pool
    # ceil(max_k / 9) + 2 for safety margin
    mpc_qk = max(max_k_qk // 9 + 2, n_qk // n_cells * 4, 20)
    mpc_v = max(max_k_v // 9 + 2, n_v // n_cells * 4, 32)
    mpc_know = max(max_k_know // 9 + 2, len(pos_know) // n_cells * 3, 176)

    cm_qk, _, min_qk, range_qk = build_cell_map(pos_qk, n_cells_per_side, mpc_qk)
    cm_v, _, min_v, range_v = build_cell_map(pos_v, n_cells_per_side, mpc_v)
    cm_know, _, min_know, range_know = build_cell_map(pos_know, n_cells_per_side, mpc_know)

    return {
        'qk': jnp.array(cm_qk),
        'v': jnp.array(cm_v),
        'know': jnp.array(cm_know),
        'pos_min_qk': jnp.array(min_qk),
        'pos_range_qk': jnp.array(range_qk),
        'pos_min_v': jnp.array(min_v),
        'pos_range_v': jnp.array(range_v),
        'pos_min_know': jnp.array(min_know),
        'pos_range_know': jnp.array(range_know),
    }


# ================================================================
# 6. Cell Lookup (JAX, inside JIT)
# ================================================================

# 3x3 neighbor offsets (self + 8 directions)
_NEIGHBOR_OFFSETS = jnp.array([
    [-1, -1], [-1, 0], [-1, 1],
    [0, -1],  [0, 0],  [0, 1],
    [1, -1],  [1, 0],  [1, 1],
], dtype=jnp.int32)  # [9, 2]


def cell_lookup(query_pos, cell_map, pos_min, pos_range, n_cells_per_side):
    """Look up candidate neuron indices from cell map via grid position.

    Args:
        query_pos:        [B, S, 2]
        cell_map:         [n_cells, max_per_cell] int32 (-1 = padding)
        pos_min:          [2] normalization offset
        pos_range:        [2] normalization scale
        n_cells_per_side: int

    Returns:
        cand_idx:   [B, S, 9 * max_per_cell] int32 (padding replaced with 0)
        valid_mask: [B, S, 9 * max_per_cell] bool
    """
    B, S, _ = query_pos.shape
    max_per_cell = cell_map.shape[1]

    # Normalize to [0, 1) and compute cell coordinates
    normalized = (query_pos - pos_min) / pos_range  # [B, S, 2]
    cell_xy = jnp.clip(
        (normalized * n_cells_per_side).astype(jnp.int32),
        0, n_cells_per_side - 1)  # [B, S, 2]

    # 3x3 neighbor cells: [B, S, 9, 2]
    neighbor_xy = cell_xy[:, :, None, :] + _NEIGHBOR_OFFSETS[None, None, :, :]
    neighbor_xy = jnp.clip(neighbor_xy, 0, n_cells_per_side - 1)

    # Flatten to cell index: [B, S, 9]
    neighbor_idx = (neighbor_xy[:, :, :, 0] * n_cells_per_side
                    + neighbor_xy[:, :, :, 1])

    # Gather from cell_map: [B, S, 9, max_per_cell]
    cand_idx = cell_map[neighbor_idx]
    cand_idx = cand_idx.reshape(B, S, 9 * max_per_cell)

    # Valid mask and safe indices
    valid_mask = (cand_idx >= 0)
    cand_idx = jnp.where(valid_mask, cand_idx, 0)

    return cand_idx, valid_mask


# ================================================================
# 7. Sense-Emit with valid mask
# ================================================================

def sense_emit_mapped(x, neurons, gate, cand_idx, valid_mask):
    """Rank-1 sense-emit with masking for invalid (padding) candidates.

    x:          [B, S, D]
    neurons:    [N, D]
    gate:       [B, S, n_cand]
    cand_idx:   [B, S, n_cand]
    valid_mask: [B, S, n_cand] bool
    """
    cand_neurons = neurons[cand_idx]                        # [B, S, n_cand, D]
    activations = jnp.einsum('bsd,bsnd->bsn', x, cand_neurons)
    masked_gate = gate * valid_mask
    gated = activations * masked_gate
    return jnp.einsum('bsn,bsnd->bsd', gated, cand_neurons)


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
# 9. Router -- 2D positional projections + tau (no distance logic)
# ================================================================

class Router(nn.Module):
    """Router for map-based routing. Only produces query positions and tau.
    Cell lookup and scoring happen in the pipeline functions.
    """
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

        # Learnable 2D coordinates for all neurons
        self.neuron_pos = self.param(
            'neuron_pos', scaled_normal(1.0), (total, self.pos_dim))

        # Input -> query position projections (per circuit)
        self.proj_pos_qk = nn.Dense(self.pos_dim, name='proj_pos_qk')
        self.proj_pos_v = nn.Dense(self.pos_dim, name='proj_pos_v')
        self.proj_pos_know = nn.Dense(self.pos_dim, name='proj_pos_know')

        # Learnable tau
        self.tau_attn = nn.Dense(3, name='tau_attn')
        self.tau_know = nn.Dense(1, name='tau_know')

    def get_attention_gates(self, x, neuron_pool, deterministic, rng,
                            cell_maps=None, n_cells_per_side=32):
        """Compute Q, K, V gates via cell map lookup (init path).

        If cell_maps is None, falls back to brute-force distance (init only).
        """
        B, S, D = x.shape
        qk_neurons = neuron_pool.qk_neurons
        v_neurons = neuron_pool.v_neurons

        qk_pos = self.proj_pos_qk(x)
        v_pos = self.proj_pos_v(x)

        npos_qk = self.neuron_pos[:self.n_qk]
        npos_v = self.neuron_pos[self.n_qk:self.n_qk + self.n_v]

        if cell_maps is not None:
            cand_idx_qk, mask_qk = cell_lookup(
                qk_pos, cell_maps['qk'],
                cell_maps['pos_min_qk'], cell_maps['pos_range_qk'],
                n_cells_per_side)
            cand_idx_v, mask_v = cell_lookup(
                v_pos, cell_maps['v'],
                cell_maps['pos_min_v'], cell_maps['pos_range_v'],
                n_cells_per_side)
        else:
            # Brute-force fallback (init path, small batch)
            dist_qk = jnp.sum(
                (qk_pos[:, :, None, :] - npos_qk[None, None, :, :]) ** 2,
                axis=-1)
            n_cand_qk = min(self.max_k_qk * 2, self.n_qk)
            _, cand_idx_qk = jax.lax.top_k(-dist_qk, n_cand_qk)
            mask_qk = jnp.ones_like(cand_idx_qk, dtype=jnp.bool_)

            dist_v = jnp.sum(
                (v_pos[:, :, None, :] - npos_v[None, None, :, :]) ** 2,
                axis=-1)
            n_cand_v = min(self.max_k_v * 2, self.n_v)
            _, cand_idx_v = jax.lax.top_k(-dist_v, n_cand_v)
            mask_v = jnp.ones_like(cand_idx_v, dtype=jnp.bool_)

        cand_idx_qk = jax.lax.stop_gradient(cand_idx_qk)
        cand_idx_v = jax.lax.stop_gradient(cand_idx_v)

        cand_qk = qk_neurons[cand_idx_qk]
        cand_v = v_neurons[cand_idx_v]

        rng, rng1 = jax.random.split(rng)
        x_drop = safe_dropout(x, self.router_dropout, deterministic, rng1)

        scores_Q = jnp.einsum('bsd,bsnd->bsn', x_drop, cand_qk)
        scores_K = jnp.einsum('bsd,bsnd->bsn', x_drop, cand_qk)
        scores_V = jnp.einsum('bsd,bsnd->bsn', x_drop, cand_v)

        scores_Q = jnp.where(mask_qk, scores_Q, -1e9)
        scores_K = jnp.where(mask_qk, scores_K, -1e9)
        scores_V = jnp.where(mask_v, scores_V, -1e9)

        tau_all = self.tau_attn(x)
        gate_Q = threshold_gate(scores_Q, tau_all[:, :, 0:1], self.max_k_qk)
        gate_K = threshold_gate(scores_K, tau_all[:, :, 1:2], self.max_k_qk)
        gate_V = threshold_gate(scores_V, tau_all[:, :, 2:3], self.max_k_v)

        # pos_loss
        pos_dist_qk = jnp.sum(
            (qk_pos[:, :, None, :] - npos_qk[cand_idx_qk]) ** 2, axis=-1)
        pos_loss_qk = (jax.lax.stop_gradient(gate_Q) * pos_dist_qk * mask_qk
                        ).sum() / (mask_qk.sum() + 1e-8)
        pos_dist_v = jnp.sum(
            (v_pos[:, :, None, :] - npos_v[cand_idx_v]) ** 2, axis=-1)
        pos_loss_v = (jax.lax.stop_gradient(gate_V) * pos_dist_v * mask_v
                       ).sum() / (mask_v.sum() + 1e-8)

        return (gate_Q, gate_K, gate_V, cand_idx_qk, cand_idx_v,
                mask_qk, mask_v, pos_loss_qk + pos_loss_v)

    def get_knowledge_gates(self, x, neuron_pool, deterministic, rng,
                            cell_maps=None, n_cells_per_side=32):
        """Compute knowledge gates via cell map lookup (init path)."""
        B, S, D = x.shape
        know_neurons = neuron_pool.know_neurons

        know_pos = self.proj_pos_know(x)
        npos_know = self.neuron_pos[self.n_qk + self.n_v:]

        if cell_maps is not None:
            cand_idx, valid_mask = cell_lookup(
                know_pos, cell_maps['know'],
                cell_maps['pos_min_know'], cell_maps['pos_range_know'],
                n_cells_per_side)
        else:
            dist = jnp.sum(
                (know_pos[:, :, None, :] - npos_know[None, None, :, :]) ** 2,
                axis=-1)
            n_cand = min(self.max_k_know * 2, self.n_know)
            _, cand_idx = jax.lax.top_k(-dist, n_cand)
            valid_mask = jnp.ones_like(cand_idx, dtype=jnp.bool_)

        cand_idx = jax.lax.stop_gradient(cand_idx)
        cand_neurons = know_neurons[cand_idx]

        rng, rng1 = jax.random.split(rng)
        x_drop = safe_dropout(x, self.router_dropout, deterministic, rng1)
        scores = jnp.einsum('bsd,bsnd->bsn', x_drop, cand_neurons)
        scores = jnp.where(valid_mask, scores, -1e9)

        tau = self.tau_know(x)
        gate = threshold_gate(scores, tau, self.max_k_know)

        cand_npos = npos_know[cand_idx]
        pos_dist = jnp.sum(
            (know_pos[:, :, None, :] - cand_npos) ** 2, axis=-1)
        pos_loss = (jax.lax.stop_gradient(gate) * pos_dist * valid_mask
                    ).sum() / (valid_mask.sum() + 1e-8)

        return gate, cand_idx, valid_mask, pos_loss


# ================================================================
# 10. Chunked Pipeline Functions (scan body)
# ================================================================

def _attn_pipeline_mapped_chunked(
    x, qk_pos, v_pos, tau_Q, tau_K, tau_V,
    qk_neurons, v_neurons, npos_qk, npos_v,
    cell_map_qk, cell_map_v,
    pos_min_qk, pos_range_qk, pos_min_v, pos_range_v,
    expand_O_kernel, rng,
    # static:
    n_cells_per_side, max_k_qk, max_k_v,
    n_heads, d_model,
    router_dropout, dropout_rate, deterministic,
    chunk_size,
):
    """Attention: cell lookup (full seq) + chunked Q/K/V sense_emit + full self-attn.

    Cell lookup produces small index tensors [B,S,n_cand].
    Gather+score+gate+emit are chunked to limit peak memory.
    Self-attention runs on full sequence (Q/K/V are [B,S,D] = small).
    """
    B, S, D = x.shape
    pos_dim = qk_pos.shape[-1]

    # Cell lookup over full sequence (int tensors, small)
    cand_idx_qk, mask_qk = cell_lookup(
        qk_pos, cell_map_qk, pos_min_qk, pos_range_qk, n_cells_per_side)
    cand_idx_v, mask_v = cell_lookup(
        v_pos, cell_map_v, pos_min_v, pos_range_v, n_cells_per_side)
    cand_idx_qk = jax.lax.stop_gradient(cand_idx_qk)
    cand_idx_v = jax.lax.stop_gradient(cand_idx_v)

    n_cand_qk = cand_idx_qk.shape[-1]
    n_cand_v = cand_idx_v.shape[-1]

    # Pad sequence
    pad_S = ((S + chunk_size - 1) // chunk_size) * chunk_size
    if pad_S > S:
        pad2 = ((0, 0), (0, pad_S - S), (0, 0))
        x = jnp.pad(x, pad2)
        qk_pos = jnp.pad(qk_pos, pad2[:2] + ((0, 0),))
        v_pos = jnp.pad(v_pos, pad2[:2] + ((0, 0),))
        tau_Q = jnp.pad(tau_Q, pad2[:2] + ((0, 0),))
        tau_K = jnp.pad(tau_K, pad2[:2] + ((0, 0),))
        tau_V = jnp.pad(tau_V, pad2[:2] + ((0, 0),))
        cand_idx_qk = jnp.pad(cand_idx_qk, pad2[:2] + ((0, 0),))
        mask_qk = jnp.pad(mask_qk, pad2[:2] + ((0, 0),))
        cand_idx_v = jnp.pad(cand_idx_v, pad2[:2] + ((0, 0),))
        mask_v = jnp.pad(mask_v, pad2[:2] + ((0, 0),))

    n_chunks = pad_S // chunk_size
    Q_buf = jnp.zeros((B, pad_S, D))
    K_buf = jnp.zeros((B, pad_S, D))
    V_buf = jnp.zeros((B, pad_S, D))
    total_pos_loss = jnp.float32(0.0)
    chunk_rngs = jax.random.split(rng, n_chunks + 1)
    rng_final = chunk_rngs[-1]
    chunk_rngs = chunk_rngs[:-1]

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
        ci_qk = jax.lax.dynamic_slice(
            cand_idx_qk, (0, start, 0), (B, chunk_size, n_cand_qk))
        m_qk = jax.lax.dynamic_slice(
            mask_qk, (0, start, 0), (B, chunk_size, n_cand_qk))
        ci_v = jax.lax.dynamic_slice(
            cand_idx_v, (0, start, 0), (B, chunk_size, n_cand_v))
        m_v = jax.lax.dynamic_slice(
            mask_v, (0, start, 0), (B, chunk_size, n_cand_v))
        rng_c = chunk_rngs[chunk_idx]

        # Gather
        cand_qk = qk_neurons[ci_qk]
        cand_v = v_neurons[ci_v]

        # Score
        rng_c, rng_drop = jax.random.split(rng_c)
        x_drop = safe_dropout(x_c, router_dropout, deterministic, rng_drop)
        scores_Q = jnp.where(m_qk, jnp.einsum('bsd,bsnd->bsn', x_drop, cand_qk), -1e9)
        scores_K = jnp.where(m_qk, jnp.einsum('bsd,bsnd->bsn', x_drop, cand_qk), -1e9)
        scores_V = jnp.where(m_v, jnp.einsum('bsd,bsnd->bsn', x_drop, cand_v), -1e9)

        # Gate
        gate_Q = threshold_gate(scores_Q, tau_Q_c, max_k_qk)
        gate_K = threshold_gate(scores_K, tau_K_c, max_k_qk)
        gate_V = threshold_gate(scores_V, tau_V_c, max_k_v)

        # Sense-emit Q/K/V (inline)
        act_Q = jnp.einsum('bsd,bsnd->bsn', x_c, cand_qk)
        Q_c = jnp.einsum('bsn,bsnd->bsd', act_Q * gate_Q * m_qk, cand_qk)
        act_K = jnp.einsum('bsd,bsnd->bsn', x_c, cand_qk)
        K_c = jnp.einsum('bsn,bsnd->bsd', act_K * gate_K * m_qk, cand_qk)
        act_V = jnp.einsum('bsd,bsnd->bsn', x_c, cand_v)
        V_c = jnp.einsum('bsn,bsnd->bsd', act_V * gate_V * m_v, cand_v)

        # pos_loss
        pd_qk = jnp.sum(
            (qk_pos_c[:, :, None, :] - npos_qk[ci_qk]) ** 2, axis=-1)
        pl_qk = (jax.lax.stop_gradient(gate_Q) * pd_qk * m_qk
                  ).sum() / (m_qk.sum() + 1e-8)
        pd_v = jnp.sum(
            (v_pos_c[:, :, None, :] - npos_v[ci_v]) ** 2, axis=-1)
        pl_v = (jax.lax.stop_gradient(gate_V) * pd_v * m_v
                ).sum() / (m_v.sum() + 1e-8)

        Q_buf = jax.lax.dynamic_update_slice(Q_buf, Q_c, (0, start, 0))
        K_buf = jax.lax.dynamic_update_slice(K_buf, K_c, (0, start, 0))
        V_buf = jax.lax.dynamic_update_slice(V_buf, V_c, (0, start, 0))
        total_pos_loss = total_pos_loss + pl_qk + pl_v
        return (Q_buf, K_buf, V_buf, total_pos_loss), None

    (Q_buf, K_buf, V_buf, total_pos_loss), _ = jax.lax.scan(
        process_chunk,
        (Q_buf, K_buf, V_buf, total_pos_loss),
        jnp.arange(n_chunks))

    Q = Q_buf[:, :S, :]
    K = K_buf[:, :S, :]
    V = V_buf[:, :S, :]
    pos_loss = total_pos_loss / n_chunks

    # --- Self-attention (full sequence) ---
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

    rng_attn, rng_out = jax.random.split(rng_final)
    attn_w = safe_dropout(attn_w, dropout_rate, deterministic, rng_attn)

    out = jnp.einsum('bhst,bhtd->bhsd', attn_w, V)
    out = out.transpose(0, 2, 1, 3).reshape(B, S, D)
    out = out @ expand_O_kernel
    out = safe_dropout(out, dropout_rate, deterministic, rng_out)

    return out, pos_loss


def _know_pipeline_mapped_chunked(
    x, know_pos, tau,
    know_neurons, npos_know,
    cell_map_know, pos_min_know, pos_range_know,
    rng,
    # static:
    n_cells_per_side, max_k_know,
    router_dropout, dropout_rate, deterministic,
    chunk_size,
):
    """Know pipeline: cell lookup (full seq) + chunked gather/score/gate/emit.

    Cell lookup over full sequence (small int tensors).
    Gather+score+gate+emit chunked via carry accumulation.
    """
    B, S, D = x.shape
    pos_dim = know_pos.shape[-1]

    # Cell lookup (full sequence, small tensors)
    cand_idx, valid_mask = cell_lookup(
        know_pos, cell_map_know, pos_min_know, pos_range_know,
        n_cells_per_side)
    cand_idx = jax.lax.stop_gradient(cand_idx)
    n_cand = cand_idx.shape[-1]

    # Pad
    pad_S = ((S + chunk_size - 1) // chunk_size) * chunk_size
    if pad_S > S:
        x = jnp.pad(x, ((0, 0), (0, pad_S - S), (0, 0)))
        know_pos = jnp.pad(know_pos, ((0, 0), (0, pad_S - S), (0, 0)))
        tau = jnp.pad(tau, ((0, 0), (0, pad_S - S), (0, 0)))
        cand_idx = jnp.pad(cand_idx, ((0, 0), (0, pad_S - S), (0, 0)))
        valid_mask = jnp.pad(valid_mask, ((0, 0), (0, pad_S - S), (0, 0)))

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
        ci = jax.lax.dynamic_slice(
            cand_idx, (0, start, 0), (B, chunk_size, n_cand))
        vm = jax.lax.dynamic_slice(
            valid_mask, (0, start, 0), (B, chunk_size, n_cand))
        rng_c = chunk_rngs[chunk_idx]

        # Gather
        cand_neurons = know_neurons[ci]

        # Score
        rng_c, rng_drop = jax.random.split(rng_c)
        x_drop = safe_dropout(x_c, router_dropout, deterministic, rng_drop)
        scores = jnp.where(vm, jnp.einsum('bsd,bsnd->bsn', x_drop, cand_neurons), -1e9)

        # Gate
        gate = threshold_gate(scores, tau_c, max_k_know)

        # Sense-emit (inline)
        activations = jnp.einsum('bsd,bsnd->bsn', x_c, cand_neurons)
        gated = activations * gate * vm
        chunk_out = jnp.einsum('bsn,bsnd->bsd', gated, cand_neurons)
        rng_c, rng_out = jax.random.split(rng_c)
        chunk_out = safe_dropout(
            chunk_out, dropout_rate, deterministic, rng_out)

        # pos_loss
        cand_npos = npos_know[ci]
        pos_dist = jnp.sum(
            (pos_c[:, :, None, :] - cand_npos) ** 2, axis=-1)
        chunk_pl = (jax.lax.stop_gradient(gate) * pos_dist * vm
                    ).sum() / (vm.sum() + 1e-8)

        out_buf = jax.lax.dynamic_update_slice(
            out_buf, chunk_out, (0, start, 0))
        total_pos_loss = total_pos_loss + chunk_pl
        return (out_buf, total_pos_loss), None

    (out_buf, total_pos_loss), _ = jax.lax.scan(
        process_chunk,
        (out_buf, total_pos_loss),
        jnp.arange(n_chunks))

    return out_buf[:, :S, :], total_pos_loss / n_chunks


# ================================================================
# 11. Flax modules (init path only -- scan uses pure functions)
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
        rng, rng_router, rng_drop, rng_out = jax.random.split(rng, 4)

        (gate_Q, gate_K, gate_V, cand_qk, cand_v,
         mask_qk, mask_v, aux) = router.get_attention_gates(
            x, neuron_pool, deterministic, rng_router)

        def _se(x, neurons, gate, ci, m):
            cn = neurons[ci]
            act = jnp.einsum('bsd,bsnd->bsn', x, cn)
            return jnp.einsum('bsn,bsnd->bsd', act * gate * m, cn)

        Q = _se(x, neuron_pool.qk_neurons, gate_Q, cand_qk, mask_qk)
        K = _se(x, neuron_pool.qk_neurons, gate_K, cand_qk, mask_qk)
        V = _se(x, neuron_pool.v_neurons, gate_V, cand_v, mask_v)

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
    d_model: int
    dropout_rate: float = 0.1

    def __call__(self, x, neuron_pool, router, attention_mask, deterministic):
        rng = self.make_rng('dropout')
        rng, rng_router = jax.random.split(rng)

        gate, cand_idx, valid_mask, aux = router.get_knowledge_gates(
            x, neuron_pool, deterministic, rng_router)

        cn = neuron_pool.know_neurons[cand_idx]
        act = jnp.einsum('bsd,bsnd->bsn', x, cn)
        out = jnp.einsum('bsn,bsnd->bsd', act * gate * valid_mask, cn)
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
        attn_out, attn_aux = self.attn(
            normed, neuron_pool, router, attention_mask, deterministic)
        x = x + attn_out
        normed = self.norm2(x)
        know_out, know_aux = self.knowledge(
            normed, neuron_pool, router, attention_mask, deterministic)
        x = x + know_out
        return x, attn_aux + know_aux


# ================================================================
# 12. DAWN Model
# ================================================================

class DAWN(nn.Module):
    """DAWN-Spatial v3: Rank-1 Neuron + Map-Based Routing."""
    __version__ = "spatial-r1-v3.0.0"

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
    chunk_size: int = 16

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
            router_dropout=self.router_dropout)

        self.layers = [
            DAWNBlock(
                d_model=self.d_model, n_heads=self.n_heads,
                dropout_rate=self.dropout_rate, name=f'block_{i}')
            for i in range(self.n_layers)
        ]
        self.norm = nn.LayerNorm()

    def __call__(self, input_ids, labels=None, attention_mask=None,
                 deterministic=False, cell_maps=None):
        B, S = input_ids.shape
        if S > self.max_seq_len:
            raise ValueError(
                f"Sequence length {S} exceeds max_seq_len {self.max_seq_len}")

        positions = jnp.arange(S)[jnp.newaxis, :]
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        emb_rng = self.make_rng('dropout')
        x = safe_dropout(x, self.dropout_rate, deterministic, emb_rng)

        if self.is_initializing():
            total_aux = jnp.float32(0.0)
            for layer in self.layers:
                x, aux = layer(
                    x, self.neuron_pool, self.router,
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

            # Pre-slice neuron_pos (static indices, outside scan)
            neuron_pos = router_params['neuron_pos']
            npos_qk = neuron_pos[:self.n_qk]
            npos_v = neuron_pos[self.n_qk:self.n_qk + self.n_v]
            npos_know = neuron_pos[self.n_qk + self.n_v:]

            def scan_body(carry, xs):
                x = carry
                bp = xs['params']
                rng = xs['rng']
                rng, rng_attn, rng_know = jax.random.split(rng, 3)

                # --- Attention ---
                normed = _layer_norm(
                    x, bp['norm1']['scale'], bp['norm1']['bias'])

                qk_pos = (normed @ router_params['proj_pos_qk']['kernel']
                          + router_params['proj_pos_qk']['bias'])
                v_pos = (normed @ router_params['proj_pos_v']['kernel']
                         + router_params['proj_pos_v']['bias'])
                tau_all = (normed @ router_params['tau_attn']['kernel']
                           + router_params['tau_attn']['bias'])

                attn_out, attn_aux = _attn_pipeline_mapped_chunked(
                    normed, qk_pos, v_pos,
                    tau_all[:, :, 0:1], tau_all[:, :, 1:2], tau_all[:, :, 2:3],
                    pool_params['qk_neurons'], pool_params['v_neurons'],
                    npos_qk, npos_v,
                    cell_maps['qk'], cell_maps['v'],
                    cell_maps['pos_min_qk'], cell_maps['pos_range_qk'],
                    cell_maps['pos_min_v'], cell_maps['pos_range_v'],
                    bp['attn']['expand_O']['kernel'], rng_attn,
                    # static
                    self.n_cells_per_side, self.max_k_qk, self.max_k_v,
                    self.n_heads, self.d_model,
                    self.router_dropout, self.dropout_rate,
                    deterministic, self.chunk_size)

                x = x + attn_out

                # --- Knowledge ---
                normed = _layer_norm(
                    x, bp['norm2']['scale'], bp['norm2']['bias'])

                know_pos = (normed @ router_params['proj_pos_know']['kernel']
                            + router_params['proj_pos_know']['bias'])
                tau_know = (normed @ router_params['tau_know']['kernel']
                            + router_params['tau_know']['bias'])

                know_out, know_aux = _know_pipeline_mapped_chunked(
                    normed, know_pos, tau_know,
                    pool_params['know_neurons'], npos_know,
                    cell_maps['know'],
                    cell_maps['pos_min_know'], cell_maps['pos_range_know'],
                    rng_know,
                    # static
                    self.n_cells_per_side, self.max_k_know,
                    self.router_dropout, self.dropout_rate,
                    deterministic, self.chunk_size)

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
                valid_count = jnp.sum(vmask)
                return loss, correct, valid_count

            loss, correct, valid_count = compute_loss_and_acc(
                shift_x, embedding_matrix, shift_labels, valid_mask)
            result['loss'] = loss
            result['correct'] = correct
            result['valid_count'] = valid_count
        else:
            logits = self.token_emb.attend(x)
            result['logits'] = logits

        return result

    # ------------------------------------------------------------------
    # Auxiliary losses
    # ------------------------------------------------------------------

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
            'n_cells_per_side': self.n_cells_per_side,
            'map_rebuild_interval': self.map_rebuild_interval,
            'pos_loss_weight': self.pos_loss_weight,
            'chunk_size': self.chunk_size,
        }

    def get_model_info(self):
        return [
            f"DAWN v{self.__version__}: Rank-1 + Map-Based Routing (JAX)",
            f"  d_model={self.d_model}, n_layers={self.n_layers}, "
            f"n_heads={self.n_heads}",
            f"  [Neuron Pool]",
            f"  QK: {self.n_qk}, V: {self.n_v}, Know: {self.n_know}",
            f"  [Map Routing]",
            f"  grid={self.n_cells_per_side}x{self.n_cells_per_side}, "
            f"chunk_size={self.chunk_size}",
            f"  rebuild_interval={self.map_rebuild_interval}",
        ]
