"""
DAWN-Spatial v4.0.1: Sense-Read-Write (JAX/Flax)

Changelog:
  spatial-r1-v4.0.1 (2026-04-12):
    - GELU gate (z × Φ(z)) — confidence × intensity structure.
      Φ(z) = normal CDF provides statistical confidence from s_std-normalized z.
    - den floor=1.0 from v3.9.4 (prevents tau runaway to dense regime).

  spatial-r1-v3.9.9 (2026-04-12):
    - Gate function: sigmoid(z) → GELU-like z*sigmoid(z) with z>0 mask
    - Separate confidence (sigmoid) and intensity (z) gradient paths
    - Same denominator structure as v3.9.4 (gate sum)
    - Unbounded gate range restores neuron importance differentiation
    - Active count threshold: gate > 0 (replaces sigmoid > 0.5)

  spatial-r1-v3.9.8.1 (2026-04-12):
    - Remove xr² from denominator: den = Σ gate (pure confidence sum)
    - Motivation: xr² in denominator compensated when large-xr neurons
      were dropped, making pruning "free" → over-sparse
    - den = Σ sigmoid gives "confidence-weighted average" structure
    - Same cancellation property as v3.9.4 (gate in both num and den)

  spatial-r1-v3.9.8 (2026-04-12):
    - Sigmoid gate replaces binary+STE
    - gate = sigmoid(raw/s_std): bounded [0,1], continuous, no STE needed
    - Numerator and denominator share same gate → structural cancellation
    - den = Σ gate × xr² (no stop_gradient on xr²)
    - Bounded gate separates routing confidence from contribution magnitude
    - Linear denominator (no √) + learnable per-pool scale (init=√d_model, WD excluded)

  spatial-r1-v3.9.7.1 (2026-04-12):
    - Remove √ from denominator: den = Σ sigmoid(raw/s_std) × sg(xr²) (linear)
    - Replace fixed output scale with per-pool learnable output_scale (init=1.0)
      - know_scale, qk_scale, v_scale: scalar params, WD excluded
    - Linear den restores bidirectional feedback:
      fewer neurons → smaller den → larger output → loss pressure

  spatial-r1-v3.9.7 (2026-04-12):
    - Binary gate + xr²-weighted soft denominator
    - numerator: pure binary gate (STE via sigmoid(raw/s_std))
    - denominator: √(Σ sigmoid(raw/s_std) × xr²) — xr-weighted smooth count
    - Remove stop_gradient from xr² in denominator
    - Allows denominator to provide self-regulation gradient to read
    - Prevents runaway xr growth (positive feedback loop)
    - sigmoid scale = 1/s_std (adaptive, no hyperparameter)
    - STE changed from ReLU-based to sigmoid-based
    - den floor = 1e-3

  spatial-r1-v3.9.6 (2026-04-11):
    - STE binary gate + soft denominator
    - numerator: gate_hard * xr @ wc (binary selection, STE gradient)
    - denominator: soft_gate_sum = Σ ReLU(raw) (continuous, gradient flows)
    - out = raw_out / max(soft_gate_sum, 1.0)
    - Output scale √d_model unchanged from v3.9.4
    - gate_norm_mode removed (single mode only)

  spatial-r1-v3.9.5 (2026-04-11):
    - STE binary gate: forward 0/1, backward continuous gradient
    - gate = gate_hard + gate_soft - stop_gradient(gate_soft)  (STE trick)
    - gate_hard = (raw > 0).astype(dtype), gate_soft = ReLU(raw)
    - gate_norm_mode config: "sqrt_active" or "active_n"
      - sqrt_active: out = raw_out / √(active_N + 1)
      - active_n: out = raw_out / max(active_N, 1)
    - clip(0, 10) removed (binary gate, unnecessary)
    - gate_concentration logging replaced by active_n_mean

  spatial-r1-v3.9.4 (2026-04-08):
    - Remove tanh(gate_max) heuristic from all output paths
    - gate_sum normalize (ratio) + fixed √d_model scale only
    - x·read naturally modulates per-token output magnitude
    - No learnable strength parameters, no gate_strength variable
    - gate_sum floor=1.0: backward gradient 1/gate_sum² 폭발 방지

  spatial-r1-v3.9.1 (2026-04-05):
    - LB loss: gate-based → score-based (pre-ReLU)
    - All neurons receive LB gradient (no ReLU barrier)
    - Naturally adaptive: weak when scores uniform, strong when biased
    - gate LB (ng_sum/ng_sq) removed from pass 2
    - read/write: forward normalize (unit direction), init unit_norm_init
    - score_lb: CV² with adaptive epsilon (spread-invariant, stable at mean≈0)
    - gate_strength: pmax across model shards (global max)
    - Fixed output_scale = √d_model (not learnable, no WD issues)
    - Attn aux /3, layer .mean() (N/layer/pool invariant)

  spatial-r1-v3.9.0 (2026-04-05):
    - Gate: exp(gate)-1 → ReLU (linear gate). No dead neuron gradient.
    - Gate: σ-normalization removed (raw = scores - tau, no /std)
    - LB loss: 39M-style uniform target MSE per neuron
    - Attn dropout restored inside checkpoint (rng passed as arg)
    - New logging: score_std, gate_concentration
    - Inference APIs updated to match linear gate

  spatial-r1-v3.8.2 (2026-04-01):
    - d_route 64->128 (routing resolution improvement)
    - threshold_gate clamp (NaN prevention)
    - Neuron counts adjusted for param budget

  spatial-r1-v3.8.1 (2026-04-01):
    - threshold_gate: v18.5 relative tau (mean + offset*std)
    - Dead neuron gradient flow (1e-6 * exp(raw))
    - Exp scaling + gate_strength (tanh)
    - tau init: bias=-0.5, kernel=zeros (initially ~70% activation)

  spatial-r1-v3.8.0 (2026-04-01):
    - Sense-Read-Write: each neuron has emb[64] + w_read[384] + w_write[384]
    - out = sum(gate_i * (x . read_i) * write_i)
    - x participates directly in output computation (rank-1 F-R)
    - All ops: matmul + element-wise. TPU optimal.

  spatial-r1-v3.7.0 (2026-04-01):
    - Sense + direct emit (gate @ w). 4s/step.

Architecture:
  NeuronPool        -- emb[N,d_bn] + w_read[N,D] + w_write[N,D]
  Router            -- proj + tau. Uses pool emb for routing.
  threshold_gate    -- element-wise threshold, no top_k
  _srw_chunked    -- N-axis chunked gate+srw (dynamic_slice, bf16, 2-pass)
  _attn_forward   -- _srw_chunked per Q/K/V -> self-attn
  _know_forward   -- _srw_chunked
  DAWN              -- embedding + jax.lax.scan + weight-tied lm_head
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Dict
from functools import partial
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map


# ================================================================
# 1. Helpers
# ================================================================

def safe_dropout(x, rate, deterministic, rng):
    if rate == 0.0:
        return x
    keep_rate = 1.0 - rate
    mask = jax.random.bernoulli(rng, keep_rate, x.shape)
    mask = jnp.where(deterministic, jnp.ones_like(mask), mask)
    return jnp.where(mask, x / keep_rate, 0.0)


def _layer_norm(x, scale, bias, eps=1e-6):
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * scale + bias


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
# 2. Threshold gate (v18.5 relative tau + dead neuron gradient + exp)
# ================================================================

def threshold_gate(scores, tau_offset):
    """Relative tau threshold gate with linear (ReLU) activation.
    scores: [B,S,N], tau_offset: [B,S,1]
    Returns normalized gate: [B,S,N]
    """
    scores_f32 = scores.astype(jnp.float32)
    s_mean = scores_f32.mean(axis=-1, keepdims=True)
    s_std = jnp.sqrt(jnp.mean(jnp.square(scores_f32 - s_mean), axis=-1, keepdims=True)) + 1e-8
    tau = s_mean + tau_offset * s_std

    raw = scores - tau.astype(scores.dtype)
    z = raw.astype(jnp.float32) / s_std
    gate = jnp.where(z > 0, z * 0.5 * (1.0 + jax.lax.erf(z * 0.7071067811865476)), 0.0)
    active_n = (gate > 0.0).sum(axis=-1, keepdims=True).astype(jnp.float32)
    norm_factor = jnp.maximum(active_n, 1.0)
    return gate.astype(scores.dtype) / norm_factor.astype(scores.dtype)


# ================================================================
# 3. shard_map based gate + sense_read_write
#    Per-device code with explicit psum communication.
#    fori_loop inside for N_local chunking.
# ================================================================

def make_sharded_srw(mesh, max_chunk_size=2048):
    """Create fused shard_map'd gate+srw. Gate never materialized full.

    2-pass chunked inside shard_map:
      Pass 1: scores stats -> tau (psum for cross-chip mean/std)
      Pass 2: gate+srw fused per chunk (gate computed and consumed per chunk)

    Returns fused_gate_srw function (single call does gate+srw+psum).
    """
    _model_axis_size = mesh.shape['model']
    _data_axis_size = mesh.shape['data']

    @partial(shard_map, mesh=mesh,
             in_specs=(P('data', None, None),    # x [B,S,D]
                       P('data', None, None),    # h [B,S,d_bn]
                       P('model', None),          # emb_norm [N_local, d_bn]
                       P('data', None, None),    # tau_offset [B,S,1]
                       P('model', None),          # read [N_local, D]
                       P('model', None)),         # write [N_local, D]
             out_specs=(P('data', None, None),   # out [B,S,D]
                        P('data', None, None),   # active [B,S,1]
                        P('data', None, None),   # gate_max [B,S,1]
                        P(),                     # lb_loss scalar
                        P(),                     # score_std scalar
                        P(),                     # gate_sum scalar
                        P(),                     # active_n_mean scalar
                        P('data', None, None),   # strong [B,S,1]
                        P('data', None, None),   # phi_binary [B,S,1]
                        P('data', None, None),   # z_mean_active [B,S,1]
                        P(),                     # tau_abs_mean scalar
                        P(),                     # z_lt_075_frac scalar
                        P(),                     # z_lt_030_frac scalar
                        P(),                     # score_skew scalar
                        P(),                     # active_per_token_std scalar
                        P()),                    # gate_entropy scalar
             check_rep=False)
    def fused_gate_srw(x, h, emb_local, tau_offset, read_local, write_local):
        N_local = emb_local.shape[0]
        nc = max(1, N_local // max_chunk_size)
        while N_local % nc != 0 and nc < N_local:
            nc += 1
        cs = N_local // nc

        B, S, D = x.shape
        h_bf = h.astype(jnp.bfloat16)
        x_bf = x.astype(jnp.bfloat16)
        emb_bf = emb_local.astype(jnp.bfloat16)
        read_bf = read_local.astype(jnp.bfloat16)
        write_bf = write_local.astype(jnp.bfloat16)
        z1 = jnp.zeros((B, S, 1))

        # --- Pass 1: exact stats over ALL chunks (scan + checkpoint) ---
        @jax.checkpoint
        def stats_step(carry, i):
            s_sum, sq_sum, cube_sum, ns_sum, ns_sq = carry
            s = i * cs
            ec = jax.lax.dynamic_slice_in_dim(emb_bf, s, cs, axis=0)
            scores = h_bf @ ec.T
            scores_f = scores.astype(jnp.float32)
            s_sum = s_sum + scores_f.sum(axis=-1, keepdims=True)
            sq_sum = sq_sum + (scores_f ** 2).sum(axis=-1, keepdims=True)
            cube_sum = cube_sum + (scores_f ** 3).sum(axis=-1, keepdims=True)
            # Score LB: per-neuron score mean stats
            per_neuron_score = scores_f.mean(axis=(0, 1))  # [cs]
            ns_sum = ns_sum + per_neuron_score.sum()
            ns_sq = ns_sq + (per_neuron_score ** 2).sum()
            return (s_sum, sq_sum, cube_sum, ns_sum, ns_sq), None

        z_bs1 = jnp.zeros((B, S, 1))
        z_scalar = jnp.float32(0.0)
        (local_sum, local_sq, local_cube, ns_sum, ns_sq), _ = jax.lax.scan(
            stats_step, (z_bs1, z_bs1, z_bs1, z_scalar, z_scalar), jnp.arange(nc))

        global_sum = jax.lax.psum(local_sum, 'model')
        global_sq = jax.lax.psum(local_sq, 'model')
        global_cube = jax.lax.psum(local_cube, 'model')
        N_total = N_local * _model_axis_size

        s_mean = global_sum / N_total
        s_std = jnp.sqrt(global_sq / N_total - s_mean ** 2) + 1e-8
        tau = s_mean + tau_offset * s_std

        # Skewness via E[(X-μ)^3] = E[X^3] - 3μσ² - μ³
        cube_mean = global_cube / N_total
        central_third = cube_mean - 3.0 * s_mean * (s_std ** 2) - s_mean ** 3
        score_skew = jax.lax.stop_gradient((central_third / (s_std ** 3 + 1e-8)).mean())

        # Score LB: variance of per-neuron score mean * N
        ns_sum = jax.lax.psum(ns_sum, 'data') / _data_axis_size
        ns_sq = jax.lax.psum(ns_sq, 'data') / _data_axis_size
        global_ns_sum = jax.lax.psum(ns_sum, 'model')
        global_ns_sq = jax.lax.psum(ns_sq, 'model')
        mean_score = global_ns_sum / N_total
        var_score = global_ns_sq / N_total - mean_score ** 2
        score_lb = var_score / (mean_score ** 2 + var_score + 1e-2)

        # --- Pass 2: gate + srw fused (scan + checkpoint) ---
        @jax.checkpoint
        def gate_srw_step(carry, i):
            (out, total_weighted_cost, total_gate_max, total_active,
             total_strong, total_phi_binary, total_z_sum,
             total_z_lt_075, total_z_lt_030, total_g_log_g) = carry
            s = i * cs
            ec = jax.lax.dynamic_slice_in_dim(emb_bf, s, cs, axis=0)
            rc = jax.lax.dynamic_slice_in_dim(read_bf, s, cs, axis=0)
            wc = jax.lax.dynamic_slice_in_dim(write_bf, s, cs, axis=0)
            rc = rc / (jnp.linalg.norm(rc, axis=-1, keepdims=True) + 1e-8)
            wc = wc / (jnp.linalg.norm(wc, axis=-1, keepdims=True) + 1e-8)
            scores = h_bf @ ec.T
            raw = scores.astype(jnp.float32) - tau  # f32 (tau already f32)
            z = raw / s_std
            phi = 0.5 * (1.0 + jax.lax.erf(z * 0.7071067811865476))
            gate = jnp.where(z > 0, z * phi, 0.0)
            gate_bf = gate.astype(jnp.bfloat16)
            xr = x_bf @ rc.T
            c_out = ((gate_bf * xr) @ wc).astype(jnp.float32)
            chunk_weighted = gate.sum(axis=-1, keepdims=True)
            chunk_active = (gate > 0.0).astype(jnp.float32).sum(axis=-1, keepdims=True)
            chunk_strong = (gate > 0.5).astype(jnp.float32).sum(axis=-1, keepdims=True)
            chunk_phi_binary = ((phi > 0.95) | (phi < 0.05)).astype(jnp.float32).sum(axis=-1, keepdims=True)
            chunk_z_sum = jnp.where(z > 0, z, 0.0).sum(axis=-1, keepdims=True)
            chunk_z_lt_075 = ((z > 0) & (z < 0.75)).astype(jnp.float32).sum(axis=-1, keepdims=True)
            chunk_z_lt_030 = ((z > 0) & (z < 0.30)).astype(jnp.float32).sum(axis=-1, keepdims=True)
            g_safe = gate + 1e-8
            chunk_g_log_g = (gate * jnp.log(g_safe)).sum(axis=-1, keepdims=True)
            return (out + c_out,
                    total_weighted_cost + chunk_weighted,
                    jnp.maximum(total_gate_max, gate.max(axis=-1, keepdims=True)),
                    total_active + chunk_active,
                    total_strong + chunk_strong,
                    total_phi_binary + chunk_phi_binary,
                    total_z_sum + chunk_z_sum,
                    total_z_lt_075 + chunk_z_lt_075,
                    total_z_lt_030 + chunk_z_lt_030,
                    total_g_log_g + chunk_g_log_g), None

        (raw_out, total_weighted_cost, total_gate_max, total_active, total_strong,
         total_phi_binary, total_z_sum, total_z_lt_075, total_z_lt_030,
         total_g_log_g), _ = jax.lax.scan(
            gate_srw_step,
            (jnp.zeros((B, S, D), dtype=jnp.float32),
             z1, jnp.full((B, S, 1), -1e9), z1, z1, z1, z1, z1, z1, z1),
            jnp.arange(nc))

        global_weighted_cost = jax.lax.psum(total_weighted_cost, 'model')
        global_gate_max = jax.lax.pmax(jax.lax.stop_gradient(total_gate_max), 'model')
        den = jnp.maximum(global_weighted_cost, 1.0)
        out = raw_out / den
        out = jax.lax.psum(out.astype(jnp.bfloat16), 'model')

        global_active = jax.lax.psum(total_active, 'model')
        active_frac = global_active / N_total
        strong_frac = jax.lax.psum(total_strong, 'model') / N_total
        phi_binary_frac = jax.lax.psum(total_phi_binary, 'model') / N_total
        z_mean_active = jax.lax.psum(total_z_sum, 'model') / (global_active + 1e-8)
        z_lt_075_frac = jax.lax.stop_gradient(
            (jax.lax.psum(total_z_lt_075, 'model') / (global_active + 1e-8)).mean())
        z_lt_030_frac = jax.lax.stop_gradient(
            (jax.lax.psum(total_z_lt_030, 'model') / (global_active + 1e-8)).mean())

        score_std_out = s_std.mean()
        es_out = global_weighted_cost.mean()
        active_n_mean = global_active.mean()
        tau_abs_mean = jax.lax.stop_gradient(tau).mean()
        active_per_token_std = jax.lax.stop_gradient(global_active).std()
        # Gate entropy: per-token H = -(1/S) Σ g log g + log S over global pool
        global_g_log_g = jax.lax.psum(total_g_log_g, 'model')
        gate_sum_eps = global_weighted_cost + 1e-8
        entropy_per_token = -global_g_log_g / gate_sum_eps + jnp.log(gate_sum_eps)
        gate_entropy = jax.lax.stop_gradient(entropy_per_token).mean()
        return (out.astype(jnp.float32), active_frac, global_gate_max, score_lb,
                score_std_out, es_out, active_n_mean, strong_frac, phi_binary_frac, z_mean_active,
                tau_abs_mean, z_lt_075_frac, z_lt_030_frac,
                score_skew, active_per_token_std, gate_entropy)

    return fused_gate_srw


def make_sharded_srw_paired(mesh, max_chunk_size=2048):
    """Fused Q+K shard_map: two routes sharing same pool in one shard_map call.

    h is [B,S,2,d_bn] (h_Q, h_K stacked on axis=2).
    tau_offset is [B,S,2,1].
    x @ read.T computed once (shared by both routes).
    Scores stats computed independently per route.
    Returns out [B,S,2,D], active [B,S,1], gate_max [B,S,1].
    """
    _model_axis_size = mesh.shape['model']
    _data_axis_size = mesh.shape['data']

    @partial(shard_map, mesh=mesh,
             in_specs=(P('data', None, None),        # x [B,S,D]
                       P('data', None, None, None),  # h [B,S,2,d_bn]
                       P('model', None),              # emb_norm [N_local, d_bn]
                       P('data', None, None, None),  # tau_offset [B,S,2,1]
                       P('model', None),              # read [N_local, D]
                       P('model', None)),             # write [N_local, D]
             out_specs=(P('data', None, None, None), # out [B,S,2,D]
                        P('data', None, None),       # active [B,S,1]
                        P('data', None, None),       # gate_max [B,S,1]
                        P(),                         # lb_loss scalar
                        P(),                         # score_std scalar
                        P(),                         # gate_sum scalar
                        P(),                         # active_n_mean scalar
                        P('data', None, None),       # strong [B,S,1]
                        P('data', None, None),       # phi_binary [B,S,1]
                        P('data', None, None),       # z_mean_active [B,S,1]
                        P(),                         # tau_abs_mean scalar
                        P(),                         # z_lt_075_frac scalar
                        P(),                         # z_lt_030_frac scalar
                        P(),                         # score_skew scalar
                        P(),                         # active_per_token_std scalar
                        P()),                        # gate_entropy scalar
             check_rep=False)
    def fused_gate_srw_paired(x, h, emb_local, tau_offset, read_local, write_local):
        N_local = emb_local.shape[0]
        nc = max(1, N_local // max_chunk_size)
        while N_local % nc != 0 and nc < N_local:
            nc += 1
        cs = N_local // nc

        B, S, D = x.shape
        # h: [B,S,2,d_route], tau_offset: [B,S,2,1]
        h_bf = h.astype(jnp.bfloat16)
        x_bf = x.astype(jnp.bfloat16)
        emb_bf = emb_local.astype(jnp.bfloat16)
        read_bf = read_local.astype(jnp.bfloat16)
        write_bf = write_local.astype(jnp.bfloat16)
        z1_r = jnp.zeros((B, S, 2, 1))

        # --- Pass 1: exact stats over ALL chunks (scan + checkpoint) ---
        @jax.checkpoint
        def stats_step(carry, i):
            s_sum, sq_sum, cube_sum, ns_sum, ns_sq = carry
            s = i * cs
            ec = jax.lax.dynamic_slice_in_dim(emb_bf, s, cs, axis=0)
            scores = jnp.einsum('bsrd,nd->bsrn', h_bf, ec)
            scores_f = scores.astype(jnp.float32)
            s_sum = s_sum + scores_f.sum(axis=-1, keepdims=True)
            sq_sum = sq_sum + (scores_f ** 2).sum(axis=-1, keepdims=True)
            cube_sum = cube_sum + (scores_f ** 3).sum(axis=-1, keepdims=True)
            # Score LB: per-neuron score mean stats (averaged over batch, seq, route)
            per_neuron_score = scores_f.mean(axis=(0, 1, 2))  # [cs]
            ns_sum = ns_sum + per_neuron_score.sum()
            ns_sq = ns_sq + (per_neuron_score ** 2).sum()
            return (s_sum, sq_sum, cube_sum, ns_sum, ns_sq), None

        z_bsr1 = jnp.zeros((B, S, 2, 1))
        z_scalar = jnp.float32(0.0)
        (local_sum, local_sq, local_cube, ns_sum, ns_sq), _ = jax.lax.scan(
            stats_step, (z_bsr1, z_bsr1, z_bsr1, z_scalar, z_scalar), jnp.arange(nc))

        global_sum = jax.lax.psum(local_sum, 'model')  # [B,S,2,1]
        global_sq = jax.lax.psum(local_sq, 'model')
        global_cube = jax.lax.psum(local_cube, 'model')
        N_total = N_local * _model_axis_size

        s_mean = global_sum / N_total      # [B,S,2,1]
        s_std = jnp.sqrt(global_sq / N_total - s_mean ** 2) + 1e-8
        tau = s_mean + tau_offset * s_std   # [B,S,2,1]

        # Skewness via E[(X-μ)^3] = E[X^3] - 3μσ² - μ³
        cube_mean = global_cube / N_total
        central_third = cube_mean - 3.0 * s_mean * (s_std ** 2) - s_mean ** 3
        score_skew = jax.lax.stop_gradient((central_third / (s_std ** 3 + 1e-8)).mean())

        # Score LB: variance of per-neuron score mean * N
        ns_sum = jax.lax.psum(ns_sum, 'data') / _data_axis_size
        ns_sq = jax.lax.psum(ns_sq, 'data') / _data_axis_size
        global_ns_sum = jax.lax.psum(ns_sum, 'model')
        global_ns_sq = jax.lax.psum(ns_sq, 'model')
        mean_score = global_ns_sum / N_total
        var_score = global_ns_sq / N_total - mean_score ** 2
        score_lb = var_score / (mean_score ** 2 + var_score + 1e-2)

        # --- Pass 2: gate + srw fused ---
        @jax.checkpoint
        def gate_srw_step(carry, i):
            (out, total_weighted_cost, total_gate_max, total_active,
             total_strong, total_phi_binary, total_z_sum,
             total_z_lt_075, total_z_lt_030, total_g_log_g) = carry
            s = i * cs
            ec = jax.lax.dynamic_slice_in_dim(emb_bf, s, cs, axis=0)
            rc = jax.lax.dynamic_slice_in_dim(read_bf, s, cs, axis=0)
            wc = jax.lax.dynamic_slice_in_dim(write_bf, s, cs, axis=0)
            rc = rc / (jnp.linalg.norm(rc, axis=-1, keepdims=True) + 1e-8)
            wc = wc / (jnp.linalg.norm(wc, axis=-1, keepdims=True) + 1e-8)
            scores = jnp.einsum('bsrd,nd->bsrn', h_bf, ec)
            raw = scores.astype(jnp.float32) - tau
            z = raw / s_std  # s_std [B,S,2,1] broadcasts to [B,S,2,N]
            phi = 0.5 * (1.0 + jax.lax.erf(z * 0.7071067811865476))
            gate = jnp.where(z > 0, z * phi, 0.0)
            gate_bf = gate.astype(jnp.bfloat16)
            xr = x_bf @ rc.T  # [B,S,N]
            c_out = jnp.einsum('bsrn,nd->bsrd', gate_bf * xr[:, :, None, :], wc).astype(jnp.float32)
            chunk_weighted = gate.sum(axis=-1, keepdims=True)  # [B,S,2,1]
            chunk_active = (gate > 0.0).astype(jnp.float32).sum(axis=-1, keepdims=True)
            chunk_strong = (gate > 0.5).astype(jnp.float32).sum(axis=-1, keepdims=True)
            chunk_phi_binary = ((phi > 0.95) | (phi < 0.05)).astype(jnp.float32).sum(axis=-1, keepdims=True)
            chunk_z_sum = jnp.where(z > 0, z, 0.0).sum(axis=-1, keepdims=True)
            chunk_z_lt_075 = ((z > 0) & (z < 0.75)).astype(jnp.float32).sum(axis=-1, keepdims=True)
            chunk_z_lt_030 = ((z > 0) & (z < 0.30)).astype(jnp.float32).sum(axis=-1, keepdims=True)
            g_safe = gate + 1e-8
            chunk_g_log_g = (gate * jnp.log(g_safe)).sum(axis=-1, keepdims=True)
            return (out + c_out,
                    total_weighted_cost + chunk_weighted,
                    jnp.maximum(total_gate_max, gate.max(axis=-1, keepdims=True)),
                    total_active + chunk_active,
                    total_strong + chunk_strong,
                    total_phi_binary + chunk_phi_binary,
                    total_z_sum + chunk_z_sum,
                    total_z_lt_075 + chunk_z_lt_075,
                    total_z_lt_030 + chunk_z_lt_030,
                    total_g_log_g + chunk_g_log_g), None

        (raw_out, total_weighted_cost, total_gate_max, total_active, total_strong,
         total_phi_binary, total_z_sum, total_z_lt_075, total_z_lt_030,
         total_g_log_g), _ = jax.lax.scan(
            gate_srw_step,
            (jnp.zeros((B, S, 2, D), dtype=jnp.float32),
             z1_r, jnp.full((B, S, 2, 1), -1e9),
             z1_r, z1_r, z1_r, z1_r, z1_r, z1_r, z1_r),
            jnp.arange(nc))

        # Normalize per route independently
        global_weighted_cost = jax.lax.psum(total_weighted_cost, 'model')
        global_gate_max = jax.lax.pmax(jax.lax.stop_gradient(total_gate_max), 'model')
        den = jnp.maximum(global_weighted_cost, 1.0)
        out = raw_out / den
        out = jax.lax.psum(out.astype(jnp.bfloat16), 'model')

        global_active = jax.lax.psum(total_active, 'model')
        active_frac = global_active / N_total
        active_frac_mean = active_frac.mean(axis=2)
        strong_frac = jax.lax.psum(total_strong, 'model') / N_total
        strong_frac_mean = strong_frac.mean(axis=2)
        phi_binary_frac = jax.lax.psum(total_phi_binary, 'model') / N_total
        phi_binary_frac_mean = phi_binary_frac.mean(axis=2)
        z_mean_active = jax.lax.psum(total_z_sum, 'model') / (global_active + 1e-8)
        z_mean_active_mean = z_mean_active.mean(axis=2)
        raw_gate_max_mean = global_gate_max.mean(axis=2)
        z_lt_075_frac = jax.lax.stop_gradient(
            (jax.lax.psum(total_z_lt_075, 'model') / (global_active + 1e-8)).mean())
        z_lt_030_frac = jax.lax.stop_gradient(
            (jax.lax.psum(total_z_lt_030, 'model') / (global_active + 1e-8)).mean())

        score_std_out = s_std.mean()
        es_out = global_weighted_cost.mean()
        active_n_mean = global_active.mean()
        tau_abs_mean = jax.lax.stop_gradient(tau).mean()
        active_per_token_std = jax.lax.stop_gradient(global_active).std()
        global_g_log_g = jax.lax.psum(total_g_log_g, 'model')
        gate_sum_eps = global_weighted_cost + 1e-8
        entropy_per_token = -global_g_log_g / gate_sum_eps + jnp.log(gate_sum_eps)
        gate_entropy = jax.lax.stop_gradient(entropy_per_token).mean()
        return (out.astype(jnp.float32), active_frac_mean, raw_gate_max_mean, score_lb,
                score_std_out, es_out, active_n_mean, strong_frac_mean, phi_binary_frac_mean,
                z_mean_active_mean, tau_abs_mean, z_lt_075_frac, z_lt_030_frac,
                score_skew, active_per_token_std, gate_entropy)

    return fused_gate_srw_paired


def _srw_chunked(x, h, emb_unit, tau_offset, w_read, w_write, n_chunks):
    """Fallback: non-sharded chunked srw (for mesh_model=1 or 40M scale).

    Uses scan + checkpoint. No shard_map.
    """
    B, S, D = x.shape
    N = emb_unit.shape[0]
    cs = N // n_chunks

    h_bf = h.astype(jnp.bfloat16)
    x_bf = x.astype(jnp.bfloat16)
    emb_bf = emb_unit.astype(jnp.bfloat16)
    read_bf = w_read.astype(jnp.bfloat16)
    write_bf = w_write.astype(jnp.bfloat16)

    z1 = jnp.zeros((B, S, 1))

    # --- Exact stats over ALL chunks (scan + checkpoint) ---
    @jax.checkpoint
    def stats_step(carry, i):
        s_sum, sq_sum, cube_sum, ns_sum, ns_sq = carry
        s = i * cs
        ec = jax.lax.dynamic_slice_in_dim(emb_bf, s, cs, axis=0)
        scores = h_bf @ ec.T
        scores_f = scores.astype(jnp.float32)
        s_sum = s_sum + scores_f.sum(axis=-1, keepdims=True)
        sq_sum = sq_sum + (scores_f ** 2).sum(axis=-1, keepdims=True)
        cube_sum = cube_sum + (scores_f ** 3).sum(axis=-1, keepdims=True)
        # Score LB: per-neuron score mean stats
        per_neuron_score = scores_f.mean(axis=(0, 1))  # [cs]
        ns_sum = ns_sum + per_neuron_score.sum()
        ns_sq = ns_sq + (per_neuron_score ** 2).sum()
        return (s_sum, sq_sum, cube_sum, ns_sum, ns_sq), None

    z_bs1 = jnp.zeros((B, S, 1))
    z_scalar = jnp.float32(0.0)
    (s_sum, sq_sum, cube_sum, ns_sum, ns_sq), _ = jax.lax.scan(
        stats_step, (z_bs1, z_bs1, z_bs1, z_scalar, z_scalar), jnp.arange(n_chunks))
    s_mean = s_sum / N
    s_std = jnp.sqrt(sq_sum / N - s_mean**2) + 1e-8
    tau = s_mean + tau_offset * s_std

    # Skewness via E[(X-μ)^3] = E[X^3] - 3μσ² - μ³
    cube_mean = cube_sum / N
    central_third = cube_mean - 3.0 * s_mean * (s_std ** 2) - s_mean ** 3
    score_skew = jax.lax.stop_gradient((central_third / (s_std ** 3 + 1e-8)).mean())

    # Score LB: CV² of per-neuron score mean (spread-invariant)
    mean_score = ns_sum / N
    var_score = ns_sq / N - mean_score ** 2
    score_lb = var_score / (mean_score ** 2 + var_score + 1e-2)

    @jax.checkpoint
    def gate_srw_step(carry, i):
        (out, total_weighted_cost, total_gate_max, total_active,
         total_strong, total_phi_binary, total_z_sum,
         total_z_lt_075, total_z_lt_030, total_g_log_g) = carry
        s = i * cs
        ec = jax.lax.dynamic_slice_in_dim(emb_bf, s, cs, axis=0)
        rc = jax.lax.dynamic_slice_in_dim(read_bf, s, cs, axis=0)
        wc = jax.lax.dynamic_slice_in_dim(write_bf, s, cs, axis=0)
        rc = rc / (jnp.linalg.norm(rc, axis=-1, keepdims=True) + 1e-8)
        wc = wc / (jnp.linalg.norm(wc, axis=-1, keepdims=True) + 1e-8)
        scores = h_bf @ ec.T
        raw = scores.astype(jnp.float32) - tau
        z = raw / s_std
        phi = 0.5 * (1.0 + jax.lax.erf(z * 0.7071067811865476))
        gate = jnp.where(z > 0, z * phi, 0.0)
        gate_bf = gate.astype(jnp.bfloat16)
        xr = x_bf @ rc.T
        c_out = ((gate_bf * xr) @ wc).astype(jnp.float32)
        chunk_weighted = gate.sum(axis=-1, keepdims=True)
        chunk_active = (gate > 0.0).astype(jnp.float32).sum(axis=-1, keepdims=True)
        chunk_strong = (gate > 0.5).astype(jnp.float32).sum(axis=-1, keepdims=True)
        chunk_phi_binary = ((phi > 0.95) | (phi < 0.05)).astype(jnp.float32).sum(axis=-1, keepdims=True)
        chunk_z_sum = jnp.where(z > 0, z, 0.0).sum(axis=-1, keepdims=True)
        chunk_z_lt_075 = ((z > 0) & (z < 0.75)).astype(jnp.float32).sum(axis=-1, keepdims=True)
        chunk_z_lt_030 = ((z > 0) & (z < 0.30)).astype(jnp.float32).sum(axis=-1, keepdims=True)
        # g*log(g) for gate entropy; use stop_gradient inside log to avoid NaN at 0.
        g_safe = gate + 1e-8
        chunk_g_log_g = (gate * jnp.log(g_safe)).sum(axis=-1, keepdims=True)
        return (out + c_out,
                total_weighted_cost + chunk_weighted,
                jnp.maximum(total_gate_max, gate.max(axis=-1, keepdims=True)),
                total_active + chunk_active,
                total_strong + chunk_strong,
                total_phi_binary + chunk_phi_binary,
                total_z_sum + chunk_z_sum,
                total_z_lt_075 + chunk_z_lt_075,
                total_z_lt_030 + chunk_z_lt_030,
                total_g_log_g + chunk_g_log_g), None

    (raw_out, total_weighted_cost, total_gate_max, total_active, total_strong,
     total_phi_binary, total_z_sum, total_z_lt_075, total_z_lt_030,
     total_g_log_g), _ = jax.lax.scan(
        gate_srw_step,
        (jnp.zeros((B, S, D), dtype=jnp.float32), z1, jnp.full((B, S, 1), -1e9),
         z1, z1, z1, z1, z1, z1, z1),
        jnp.arange(n_chunks))

    D = x.shape[-1]
    den = jnp.maximum(total_weighted_cost, 1.0)
    out = (raw_out / den).astype(jnp.bfloat16)

    score_std_out = s_std.mean()
    es_out = total_weighted_cost.mean()
    active_n_mean = total_active.mean()
    z_mean_active = total_z_sum / (total_active + 1e-8)
    active_eps = total_active + 1e-8
    z_lt_075_frac = jax.lax.stop_gradient((total_z_lt_075 / active_eps).mean())
    z_lt_030_frac = jax.lax.stop_gradient((total_z_lt_030 / active_eps).mean())
    tau_abs_mean = jax.lax.stop_gradient(tau).mean()
    active_per_token_std = jax.lax.stop_gradient(total_active).std()
    # Gate entropy: H = -(1/S) Σ g log g + log S, where S = Σ g
    gate_sum_eps = total_weighted_cost + 1e-8
    entropy_per_token = -total_g_log_g / gate_sum_eps + jnp.log(gate_sum_eps)
    gate_entropy = jax.lax.stop_gradient(entropy_per_token).mean()
    return (out.astype(jnp.float32), total_active / N, total_gate_max, score_lb,
            score_std_out, es_out, active_n_mean, total_strong / N,
            total_phi_binary / N, z_mean_active,
            tau_abs_mean, z_lt_075_frac, z_lt_030_frac,
            score_skew, active_per_token_std, gate_entropy)


# ================================================================
# 4. NeuronPool -- emb + w_read + w_write
# ================================================================

class NeuronPool(nn.Module):
    n_qk: int
    n_v: int
    n_know: int
    d_model: int
    d_route: int

    def setup(self):
        db = self.d_route
        dm = self.d_model

        # Sense (routing, low-dim)
        self.qk_emb = self.param('qk_emb', unit_norm_init(), (self.n_qk, db))
        self.v_emb = self.param('v_emb', unit_norm_init(), (self.n_v, db))
        self.know_emb = self.param('know_emb', unit_norm_init(), (self.n_know, db))

        # Read (what to extract from x) — unit norm, direction only
        self.qk_read = self.param('qk_read', unit_norm_init(), (self.n_qk, dm))
        self.v_read = self.param('v_read', unit_norm_init(), (self.n_v, dm))
        self.know_read = self.param('know_read', unit_norm_init(), (self.n_know, dm))

        # Write (direction to push) — unit norm, direction only
        self.qk_write = self.param('qk_write', unit_norm_init(), (self.n_qk, dm))
        self.v_write = self.param('v_write', unit_norm_init(), (self.n_v, dm))
        self.know_write = self.param('know_write', unit_norm_init(), (self.n_know, dm))

        # Per-pool learnable output scale (init=√d_model, WD excluded)
        self.qk_scale = self.param('qk_scale',
            lambda k, s, d: jnp.full(s, jnp.sqrt(d)), (1,), self.d_model)
        self.v_scale = self.param('v_scale',
            lambda k, s, d: jnp.full(s, jnp.sqrt(d)), (1,), self.d_model)
        self.know_scale = self.param('know_scale',
            lambda k, s, d: jnp.full(s, jnp.sqrt(d)), (1,), self.d_model)


# ================================================================
# 5. Router -- proj + tau (unchanged)
# ================================================================

class Router(nn.Module):
    d_model: int
    d_route: int
    n_qk: int
    n_v: int
    n_know: int
    router_dropout: float = 0.1

    def setup(self):
        db = self.d_route
        self.proj_attn = nn.Dense(db * 3, name='proj_attn')
        self.proj_know = nn.Dense(db, name='proj_know')
        self.tau_attn = nn.Dense(3, name='tau_attn',
            kernel_init=nn.initializers.zeros,
            bias_init=lambda k, s, d: jnp.full(s, -0.5))
        self.tau_know = nn.Dense(1, name='tau_know',
            kernel_init=nn.initializers.zeros,
            bias_init=lambda k, s, d: jnp.full(s, -0.5))

    def get_attention_gates(self, x, neuron_pool, deterministic, rng):
        qk_emb_unit = neuron_pool.qk_emb / (
            jnp.linalg.norm(neuron_pool.qk_emb, axis=-1, keepdims=True) + 1e-8)
        v_emb_unit = neuron_pool.v_emb / (
            jnp.linalg.norm(neuron_pool.v_emb, axis=-1, keepdims=True) + 1e-8)
        rng, rng_drop = jax.random.split(rng)
        h_all = self.proj_attn(x)
        h_all = safe_dropout(h_all, self.router_dropout, deterministic, rng_drop)
        h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)

        tau_all = self.tau_attn(x)
        g_Q = threshold_gate(h_Q @ qk_emb_unit.T, tau_all[:, :, 0:1])
        g_K = threshold_gate(h_K @ qk_emb_unit.T, tau_all[:, :, 1:2])
        g_V = threshold_gate(h_V @ v_emb_unit.T, tau_all[:, :, 2:3])

        t_qk = 1.0 / self.n_qk
        t_v = 1.0 / self.n_v
        aux = (
            ((g_Q.mean(axis=(0, 1)) - t_qk) ** 2).sum() * self.n_qk +
            ((g_K.mean(axis=(0, 1)) - t_qk) ** 2).sum() * self.n_qk +
            ((g_V.mean(axis=(0, 1)) - t_v) ** 2).sum() * self.n_v
        )
        return g_Q, g_K, g_V, aux

    def get_knowledge_gates(self, x, neuron_pool, deterministic, rng):
        know_emb_unit = neuron_pool.know_emb / (
            jnp.linalg.norm(neuron_pool.know_emb, axis=-1, keepdims=True) + 1e-8)
        rng, rng_drop = jax.random.split(rng)
        h = self.proj_know(x)
        h = safe_dropout(h, self.router_dropout, deterministic, rng_drop)

        tau = self.tau_know(x)
        gate = threshold_gate(h @ know_emb_unit.T, tau)

        t = 1.0 / self.n_know
        aux = ((gate.mean(axis=(0, 1)) - t) ** 2).sum() * self.n_know
        return gate, aux


# ================================================================
# 6. Pure functions for scan body
# ================================================================

def _attn_forward(x, pool_params, router_params, expand_O_kernel, rng,
                  n_qk, n_v,
                  n_heads, d_model,
                  router_dropout, dropout_rate, deterministic,
                  n_chunks_qk=1, n_chunks_v=1,
                  sharded_fns=None):
    B, S, D = x.shape
    qk_emb = pool_params['qk_emb']
    qk_read = pool_params['qk_read']
    qk_write = pool_params['qk_write']
    v_emb = pool_params['v_emb']
    v_read = pool_params['v_read']
    v_write = pool_params['v_write']

    qk_emb_unit = qk_emb / (jnp.linalg.norm(qk_emb, axis=-1, keepdims=True) + 1e-8)
    v_emb_unit = v_emb / (jnp.linalg.norm(v_emb, axis=-1, keepdims=True) + 1e-8)

    rng, rng_drop = jax.random.split(rng)
    h_all = x @ router_params['proj_attn']['kernel'] + router_params['proj_attn']['bias']
    h_all = safe_dropout(h_all, router_dropout, deterministic, rng_drop)
    h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)

    tau_all = x @ router_params['tau_attn']['kernel'] + router_params['tau_attn']['bias']
    # Pure observational metrics — wrap in stop_gradient to avoid NaN VJP
    # hazards (norm of zero kernel at init; std of constant when kernel=0).
    _tau_all_sg = jax.lax.stop_gradient(tau_all)
    attn_tau_std = _tau_all_sg.std(axis=(0, 1))  # [3] — per-Q/K/V, per-token offset std
    attn_tau_kernel_norm = jnp.sqrt(
        jnp.sum(jax.lax.stop_gradient(router_params['tau_attn']['kernel']) ** 2) + 1e-12)

    qk_scale = pool_params['qk_scale']
    v_scale = pool_params['v_scale']

    if sharded_fns is not None:
        fused_single, fused_paired = sharded_fns
        h_QK = jnp.stack([h_Q, h_K], axis=2)
        tau_QK = jnp.stack([tau_all[:, :, 0:1], tau_all[:, :, 1:2]], axis=2)
        (QK_out, qk_active, qk_raw_gmax, qk_lb, qk_sstd, qk_es, qk_anm,
         qk_strong, qk_phi_bin, qk_z_act,
         qk_tau_abs, qk_z075, qk_z030,
         qk_skew, qk_apt_std, qk_entropy) = fused_paired(
            x, h_QK, qk_emb_unit, tau_QK, qk_read, qk_write)
        qk_raw_norm = jnp.linalg.norm(QK_out, axis=-1).mean()
        Q = QK_out[:, :, 0, :] * qk_scale
        K = QK_out[:, :, 1, :] * qk_scale
        (V, v_active, v_raw_gmax, v_lb, v_sstd, v_es, v_anm,
         v_strong, v_phi_bin, v_z_act,
         v_tau_abs, v_z075, v_z030,
         v_skew, v_apt_std, v_entropy) = fused_single(
            x, h_V, v_emb_unit, tau_all[:, :, 2:3], v_read, v_write)
        v_raw_norm = jnp.linalg.norm(V, axis=-1).mean()
        V = V * v_scale
    else:
        (Q, q_active, q_raw_gmax, q_lb, q_sstd, q_es, q_anm,
         q_strong, q_phi_bin, q_z_act,
         q_tau_abs, q_z075, q_z030,
         q_skew, q_apt_std, q_entropy) = _srw_chunked(
            x, h_Q, qk_emb_unit, tau_all[:, :, 0:1], qk_read, qk_write, n_chunks_qk)
        (K, k_active, k_raw_gmax, k_lb, k_sstd, k_es, k_anm,
         k_strong, k_phi_bin, k_z_act,
         k_tau_abs, k_z075, k_z030,
         k_skew, k_apt_std, k_entropy) = _srw_chunked(
            x, h_K, qk_emb_unit, tau_all[:, :, 1:2], qk_read, qk_write, n_chunks_qk)
        (V, v_active, v_raw_gmax, v_lb, v_sstd, v_es, v_anm,
         v_strong, v_phi_bin, v_z_act,
         v_tau_abs, v_z075, v_z030,
         v_skew, v_apt_std, v_entropy) = _srw_chunked(
            x, h_V, v_emb_unit, tau_all[:, :, 2:3], v_read, v_write, n_chunks_v)
        qk_raw_norm = (jnp.linalg.norm(Q, axis=-1).mean() + jnp.linalg.norm(K, axis=-1).mean()) / 2
        v_raw_norm = jnp.linalg.norm(V, axis=-1).mean()
        Q = Q * qk_scale
        K = K * qk_scale
        V = V * v_scale
        qk_lb = q_lb + k_lb
        qk_sstd = (q_sstd + k_sstd) / 2
        qk_es = (q_es + k_es) / 2
        qk_active = (q_active + k_active) / 2
        qk_raw_gmax = jnp.maximum(q_raw_gmax, k_raw_gmax)
        qk_anm = (q_anm + k_anm) / 2
        qk_strong = (q_strong + k_strong) / 2
        qk_phi_bin = (q_phi_bin + k_phi_bin) / 2
        qk_z_act = (q_z_act + k_z_act) / 2
        qk_tau_abs = (q_tau_abs + k_tau_abs) / 2
        qk_z075 = (q_z075 + k_z075) / 2
        qk_z030 = (q_z030 + k_z030) / 2
        qk_skew = (q_skew + k_skew) / 2
        qk_apt_std = (q_apt_std + k_apt_std) / 2
        qk_entropy = (q_entropy + k_entropy) / 2

    d_head = d_model // n_heads
    Q = Q.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
    K = K.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
    V = V.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)

    scale = jnp.sqrt(jnp.float32(d_head))
    rng, rng_attn_drop = jax.random.split(rng)

    @jax.checkpoint
    def _attn_scores(Q, K, V, rng_drop):
        attn_scores = jnp.einsum('bhsd,bhtd->bhst', Q, K) / scale
        causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
        attn_scores = jnp.where(causal, attn_scores,
                                jnp.finfo(attn_scores.dtype).min)
        attn_w = jax.nn.softmax(attn_scores, axis=-1)
        attn_w = safe_dropout(attn_w, dropout_rate, deterministic, rng_drop)
        return jnp.einsum('bhst,bhtd->bhsd', attn_w, V)

    # Debug metrics
    q_norm = jnp.linalg.norm(Q, axis=-1).mean()
    k_norm = jnp.linalg.norm(K, axis=-1).mean()
    v_norm_dbg = jnp.linalg.norm(V, axis=-1).mean()
    attn_logit_max = (q_norm * k_norm / scale)

    out = _attn_scores(Q, K, V, rng_attn_drop)
    o_input_norm = jnp.linalg.norm(out, axis=-1).mean()
    out = out.transpose(0, 2, 1, 3).reshape(B, S, D)
    out = out @ expand_O_kernel
    attn_out_norm = jnp.linalg.norm(out, axis=-1).mean()
    rng, rng_out = jax.random.split(rng)
    out = safe_dropout(out, dropout_rate, deterministic, rng_out)

    # Load balance loss from gate distributions + tau regularization
    tau_reg = jnp.maximum(tau_all, 0.0).mean() * 0.01
    aux = (qk_lb + v_lb) / 3.0 + tau_reg
    attn_raw_gmax = jnp.maximum(qk_raw_gmax.mean(), v_raw_gmax.mean())
    attn_score_std = (qk_sstd + v_sstd) / 2
    attn_gate_sum = (qk_es + v_es) / 2
    attn_active_n_mean = (qk_anm + v_anm) / 2
    attn_tau_mean = tau_all.mean()
    attn_strong = (qk_strong.mean() + v_strong.mean()) / 2
    attn_qk_phi_binary = qk_phi_bin.mean()
    attn_v_phi_binary = v_phi_bin.mean()
    attn_qk_z_mean_active = qk_z_act.mean()
    attn_v_z_mean_active = v_z_act.mean()
    attn_tau_abs_mean = (qk_tau_abs + v_tau_abs) / 2
    attn_z_lt_075_frac = (qk_z075 + v_z075) / 2
    attn_z_lt_030_frac = (qk_z030 + v_z030) / 2
    attn_score_skew = (qk_skew + v_skew) / 2
    attn_active_per_token_std = (qk_apt_std + v_apt_std) / 2
    attn_gate_entropy = (qk_entropy + v_entropy) / 2
    return (out, aux, qk_active.mean(), v_active.mean(), attn_raw_gmax,
            attn_score_std, attn_gate_sum, attn_active_n_mean,
            attn_out_norm, attn_tau_mean, qk_raw_norm, v_raw_norm,
            q_norm, k_norm, v_norm_dbg, attn_logit_max, o_input_norm,
            attn_strong, attn_qk_phi_binary, attn_v_phi_binary,
            attn_qk_z_mean_active, attn_v_z_mean_active,
            attn_tau_std, attn_tau_kernel_norm,
            attn_tau_abs_mean, attn_z_lt_075_frac, attn_z_lt_030_frac,
            attn_score_skew, attn_active_per_token_std, attn_gate_entropy)


def _know_forward(x, pool_params, router_params, rng,
                  router_dropout, dropout_rate, deterministic,
                  n_chunks_know=1, sharded_fns=None):
    know_emb = pool_params['know_emb']
    know_read = pool_params['know_read']
    know_write = pool_params['know_write']

    rng, rng_drop = jax.random.split(rng)
    h = x @ router_params['proj_know']['kernel'] + router_params['proj_know']['bias']
    h = safe_dropout(h, router_dropout, deterministic, rng_drop)

    know_emb_unit = know_emb / (jnp.linalg.norm(know_emb, axis=-1, keepdims=True) + 1e-8)
    tau = x @ router_params['tau_know']['kernel'] + router_params['tau_know']['bias']
    # Pure observational metrics — stop_gradient to avoid NaN VJP hazards at init.
    know_tau_std = jax.lax.stop_gradient(tau).std()
    know_tau_kernel_norm = jnp.sqrt(
        jnp.sum(jax.lax.stop_gradient(router_params['tau_know']['kernel']) ** 2) + 1e-12)

    know_scale = pool_params['know_scale']

    if sharded_fns is not None:
        fused_single, fused_paired = sharded_fns
        (out, active_frac, raw_gate_max, lb_loss, score_std, gate_sum, active_n_mean,
         strong_frac, phi_binary_frac, z_mean_act,
         know_tau_abs_mean, know_z_lt_075_frac, know_z_lt_030_frac,
         know_score_skew, know_active_per_token_std, know_gate_entropy) = fused_single(
            x, h, know_emb_unit, tau, know_read, know_write)
    else:
        (out, active_frac, raw_gate_max, lb_loss, score_std, gate_sum, active_n_mean,
         strong_frac, phi_binary_frac, z_mean_act,
         know_tau_abs_mean, know_z_lt_075_frac, know_z_lt_030_frac,
         know_score_skew, know_active_per_token_std, know_gate_entropy) = _srw_chunked(
            x, h, know_emb_unit, tau, know_read, know_write, n_chunks_know)

    know_raw_out_norm = jnp.linalg.norm(out, axis=-1).mean()
    out = out * know_scale
    know_out_norm = jnp.linalg.norm(out, axis=-1).mean()
    rng, rng_out = jax.random.split(rng)
    out = safe_dropout(out, dropout_rate, deterministic, rng_out)

    tau_reg = jnp.maximum(tau, 0.0).mean() * 0.01
    aux = lb_loss + tau_reg
    emb_norm_val = jnp.linalg.norm(know_emb, axis=-1).mean()
    read_norm_val = jnp.linalg.norm(know_read, axis=-1).mean()
    write_norm_val = jnp.linalg.norm(know_write, axis=-1).mean()
    know_tau_mean = tau.mean()
    know_strong = strong_frac.mean()
    know_phi_binary = phi_binary_frac.mean()
    know_z_mean_active = z_mean_act.mean()
    return (out, aux, active_frac, raw_gate_max, score_std, gate_sum, active_n_mean,
            emb_norm_val, read_norm_val, write_norm_val, know_out_norm,
            know_tau_mean, know_raw_out_norm, know_strong, know_phi_binary, know_z_mean_active,
            know_tau_std, know_tau_kernel_norm,
            know_tau_abs_mean, know_z_lt_075_frac, know_z_lt_030_frac,
            know_score_skew, know_active_per_token_std, know_gate_entropy)


# ================================================================
# 7. Flax modules (init path only)
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

        g_Q, g_K, g_V, aux = router.get_attention_gates(
            x, neuron_pool, deterministic, rng_r)

        def _se(x, g, rd, wr):
            return (g * (x @ rd.T)) @ wr

        Q = _se(x, g_Q, neuron_pool.qk_read, neuron_pool.qk_write)
        K = _se(x, g_K, neuron_pool.qk_read, neuron_pool.qk_write)
        V = _se(x, g_V, neuron_pool.v_read, neuron_pool.v_write)

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
        gate, aux = router.get_knowledge_gates(
            x, neuron_pool, deterministic, rng_r)
        out = (gate * (x @ neuron_pool.know_read.T)) @ neuron_pool.know_write
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
# 8. DAWN Model
# ================================================================

class DAWN(nn.Module):
    """DAWN-Spatial v3.8: Sense-Read-Write."""
    __version__ = "spatial-r1-v4.0.1"

    vocab_size: int = 30000
    d_model: int = 384
    n_layers: int = 12
    n_heads: int = 6
    max_seq_len: int = 512
    dropout_rate: float = 0.1
    gradient_checkpointing: bool = False

    d_route: int = 128
    n_qk: int = 1580
    n_v: int = 2600
    n_know: int = 25200
    router_dropout: float = 0.1
    n_chunks_know: int = 1    # N-axis chunking for know pool
    n_chunks_qk: int = 1     # N-axis chunking for qk pool
    n_chunks_v: int = 1      # N-axis chunking for v pool

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
            d_model=self.d_model, d_route=self.d_route)
        self.router = Router(
            d_model=self.d_model, d_route=self.d_route,
            n_qk=self.n_qk, n_v=self.n_v, n_know=self.n_know,
            router_dropout=self.router_dropout)
        self.layers = [
            DAWNBlock(d_model=self.d_model, n_heads=self.n_heads,
                      dropout_rate=self.dropout_rate, name=f'block_{i}')
            for i in range(self.n_layers)]
        self.norm = nn.LayerNorm()

    def __call__(self, input_ids, labels=None, attention_mask=None,
                 deterministic=False, sharded_fns=None):
        B, S = input_ids.shape
        if S > self.max_seq_len:
            raise ValueError(f"Sequence length {S} exceeds max_seq_len")

        positions = jnp.arange(S)[jnp.newaxis, :]
        x = self.token_emb(input_ids) + self.pos_emb(positions)
        emb_rng = self.make_rng('dropout')
        x = safe_dropout(x, self.dropout_rate, deterministic, emb_rng)

        if self.is_initializing():
            _z = jnp.float32(0.0)
            total_aux = _z
            attn_auxes = _z
            know_auxes = _z
            know_active_all = _z
            know_raw_gmax_all = _z
            know_sstd_all = _z
            know_gsum_all = _z
            know_active_n_mean_all = _z
            know_strong_all = _z
            attn_qk_active_all = _z
            attn_v_active_all = _z
            attn_raw_gmax_all = _z
            attn_sstd_all = _z
            attn_gsum_all = _z
            attn_active_n_mean_all = _z
            attn_strong_all = _z
            know_phi_bin_all = _z
            attn_qk_phi_bin_all = _z
            attn_v_phi_bin_all = _z
            know_z_act_all = _z
            attn_qk_z_act_all = _z
            attn_v_z_act_all = _z
            k_emb_n_all = _z
            k_read_n_all = _z
            k_write_n_all = _z
            know_out_norm_all = _z
            attn_out_norm_all = _z
            attn_tau_mean_all = _z
            know_tau_mean_all = _z
            attn_qk_raw_norm_all = _z
            attn_v_raw_norm_all = _z
            know_raw_out_norm_all = _z
            attn_q_norm_all = _z
            attn_k_norm_all = _z
            attn_v_norm_dbg_all = _z
            attn_logit_max_all = _z
            attn_o_input_norm_all = _z
            attn_tau_std_all = jnp.zeros((1, 3), dtype=jnp.float32)
            know_tau_std_all = _z
            attn_tau_kernel_norm_all = _z
            know_tau_kernel_norm_all = _z
            attn_tau_abs_all = _z
            know_tau_abs_all = _z
            attn_z075_all = _z
            know_z075_all = _z
            attn_z030_all = _z
            know_z030_all = _z
            attn_skew_all = _z
            know_skew_all = _z
            attn_apt_std_all = _z
            know_apt_std_all = _z
            attn_entropy_all = _z
            know_entropy_all = _z
            for layer in self.layers:
                x, aux = layer(x, self.neuron_pool, self.router,
                               attention_mask, deterministic)
                total_aux = total_aux + aux
        else:
            all_params = self.variables['params']
            pool_params = all_params['neuron_pool']
            router_params = all_params['router']

            _sharded = sharded_fns

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

                normed = _layer_norm(
                    x, bp['norm1']['scale'], bp['norm1']['bias'])
                (attn_out, attn_aux, a_qk_active, a_v_active, a_raw_gmax,
                 a_sstd, a_gsum, a_active_n_mean,
                 a_out_norm, a_tau_mean, a_qk_raw_norm, a_v_raw_norm,
                 a_q_norm, a_k_norm, a_v_norm_dbg, a_logit_max, a_o_input_norm,
                 a_strong, a_qk_phi_bin, a_v_phi_bin,
                 a_qk_z_act, a_v_z_act,
                 a_tau_std, a_tau_kernel_norm,
                 a_tau_abs, a_z075, a_z030,
                 a_skew, a_apt_std, a_entropy
                ) = _attn_forward(
                    normed, pool_params, router_params,
                    bp['attn']['expand_O']['kernel'], rng_attn,
                    self.n_qk, self.n_v,
                    self.n_heads, self.d_model,
                    self.router_dropout, self.dropout_rate, deterministic,
                    self.n_chunks_qk, self.n_chunks_v,
                    sharded_fns=_sharded)
                x = x + attn_out

                normed = _layer_norm(
                    x, bp['norm2']['scale'], bp['norm2']['bias'])
                (know_out, know_aux, k_active, k_raw_gmax, k_sstd, k_gsum, k_active_n_mean,
                 k_emb_n, k_read_n, k_write_n, k_out_norm,
                 k_tau_mean, k_raw_out_norm, k_strong, k_phi_bin, k_z_act,
                 k_tau_std, k_tau_kernel_norm,
                 k_tau_abs, k_z075, k_z030,
                 k_skew, k_apt_std, k_entropy
                ) = _know_forward(
                    normed, pool_params, router_params, rng_know,
                    self.router_dropout, self.dropout_rate, deterministic,
                    self.n_chunks_know, sharded_fns=_sharded)
                x = x + know_out
                return x, (attn_aux, know_aux,
                           k_active, k_raw_gmax, k_sstd, k_gsum, k_active_n_mean,
                           a_qk_active, a_v_active, a_raw_gmax, a_sstd, a_gsum, a_active_n_mean,
                           k_emb_n, k_read_n, k_write_n,
                           k_out_norm,
                           a_out_norm, a_tau_mean, k_tau_mean,
                           a_qk_raw_norm, a_v_raw_norm, k_raw_out_norm,
                           a_q_norm, a_k_norm, a_v_norm_dbg, a_logit_max, a_o_input_norm,
                           k_strong, a_strong,
                           k_phi_bin, a_qk_phi_bin, a_v_phi_bin,
                           k_z_act, a_qk_z_act, a_v_z_act,
                           a_tau_std, k_tau_std,
                           a_tau_kernel_norm, k_tau_kernel_norm,
                           a_tau_abs, k_tau_abs,
                           a_z075, k_z075,
                           a_z030, k_z030,
                           a_skew, k_skew,
                           a_apt_std, k_apt_std,
                           a_entropy, k_entropy)

            if self.gradient_checkpointing:
                scan_body = jax.checkpoint(scan_body)

            xs = {'params': stacked, 'rng': layer_rngs}
            x, (attn_auxes, know_auxes,
                know_active_all, know_raw_gmax_all, know_sstd_all, know_gsum_all, know_active_n_mean_all,
                attn_qk_active_all, attn_v_active_all, attn_raw_gmax_all, attn_sstd_all, attn_gsum_all, attn_active_n_mean_all,
                k_emb_n_all, k_read_n_all, k_write_n_all,
                know_out_norm_all,
                attn_out_norm_all, attn_tau_mean_all, know_tau_mean_all,
                attn_qk_raw_norm_all, attn_v_raw_norm_all, know_raw_out_norm_all,
                attn_q_norm_all, attn_k_norm_all, attn_v_norm_dbg_all,
                attn_logit_max_all, attn_o_input_norm_all,
                know_strong_all, attn_strong_all,
                know_phi_bin_all, attn_qk_phi_bin_all, attn_v_phi_bin_all,
                know_z_act_all, attn_qk_z_act_all, attn_v_z_act_all,
                attn_tau_std_all, know_tau_std_all,
                attn_tau_kernel_norm_all, know_tau_kernel_norm_all,
                attn_tau_abs_all, know_tau_abs_all,
                attn_z075_all, know_z075_all,
                attn_z030_all, know_z030_all,
                attn_skew_all, know_skew_all,
                attn_apt_std_all, know_apt_std_all,
                attn_entropy_all, know_entropy_all) = jax.lax.scan(
                scan_body, x, xs)
            total_aux = (attn_auxes + know_auxes).mean()

        # Debug norms
        _residual_norm = jnp.linalg.norm(x, axis=-1).mean()
        _emb_norm = jnp.linalg.norm(self.token_emb.embedding, axis=-1).mean()
        if not self.is_initializing():
            _o_proj_norm = jnp.linalg.norm(stacked['attn']['expand_O']['kernel'], axis=(-2, -1)).mean()
        else:
            _o_proj_norm = jnp.float32(0.0)

        x = self.norm(x)
        result = {
            'aux_loss': total_aux,
            'attn_aux': attn_auxes.mean(),
            'know_aux': know_auxes.mean(),

            'know_active': know_active_all.mean(),
            'know_raw_gate_max': know_raw_gmax_all.mean(),
            'know_score_std': know_sstd_all.mean(),
            'know_gate_sum': know_gsum_all.mean(),
            'know_active_n_mean': know_active_n_mean_all.mean(),
            'know_strong': know_strong_all.mean(),
            'know_phi_binary': know_phi_bin_all.mean(),
            'know_z_mean_active': know_z_act_all.mean(),

            'attn_qk_active': attn_qk_active_all.mean(),
            'attn_v_active': attn_v_active_all.mean(),
            'attn_raw_gate_max': attn_raw_gmax_all.mean(),
            'attn_score_std': attn_sstd_all.mean(),
            'attn_gate_sum': attn_gsum_all.mean(),
            'attn_active_n_mean': attn_active_n_mean_all.mean(),
            'attn_strong': attn_strong_all.mean(),
            'attn_qk_phi_binary': attn_qk_phi_bin_all.mean(),
            'attn_v_phi_binary': attn_v_phi_bin_all.mean(),
            'attn_qk_z_mean_active': attn_qk_z_act_all.mean(),
            'attn_v_z_mean_active': attn_v_z_act_all.mean(),

            'know_emb_norm': k_emb_n_all.mean(),
            'know_read_norm': k_read_n_all.mean(),
            'know_write_norm': k_write_n_all.mean(),

            'know_out_norm': know_out_norm_all.mean(),
            'attn_out_norm': attn_out_norm_all.mean(),
            'attn_tau_mean': attn_tau_mean_all.mean(),
            'know_tau_mean': know_tau_mean_all.mean(),
            'attn_tau_std': attn_tau_std_all.mean(axis=0),   # [3] Q/K/V
            'know_tau_std': know_tau_std_all.mean(),
            'attn_tau_kernel_norm': attn_tau_kernel_norm_all.mean(),
            'know_tau_kernel_norm': know_tau_kernel_norm_all.mean(),
            'attn_tau_abs_mean': attn_tau_abs_all.mean(),
            'know_tau_abs_mean': know_tau_abs_all.mean(),
            'attn_z_lt_075': attn_z075_all.mean(),
            'know_z_lt_075': know_z075_all.mean(),
            'attn_z_lt_030': attn_z030_all.mean(),
            'know_z_lt_030': know_z030_all.mean(),
            'attn_score_skew': attn_skew_all.mean(),
            'know_score_skew': know_skew_all.mean(),
            'attn_active_per_token_std': attn_apt_std_all.mean(),
            'know_active_per_token_std': know_apt_std_all.mean(),
            'attn_gate_entropy': attn_entropy_all.mean(),
            'know_gate_entropy': know_entropy_all.mean(),
            'attn_qk_raw_norm': attn_qk_raw_norm_all.mean(),
            'attn_v_raw_norm': attn_v_raw_norm_all.mean(),
            'know_raw_out_norm': know_raw_out_norm_all.mean(),

            'debug_residual_norm': _residual_norm,
            'debug_emb_norm': _emb_norm,
            'debug_o_proj_norm': _o_proj_norm,
            'debug_q_norm': attn_q_norm_all.mean(),
            'debug_k_norm': attn_k_norm_all.mean(),
            'debug_v_norm': attn_v_norm_dbg_all.mean(),
            'debug_logit_max': attn_logit_max_all.mean(),
            'debug_o_input_norm': attn_o_input_norm_all.mean(),

            'per_layer_attn_out_norm': attn_out_norm_all,
            'per_layer_know_out_norm': know_out_norm_all,
        }

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

    def get_config(self):
        return {
            'model_version': self.__version__,
            'vocab_size': self.vocab_size, 'd_model': self.d_model,
            'n_layers': self.n_layers, 'n_heads': self.n_heads,
            'max_seq_len': self.max_seq_len,
            'd_route': self.d_route,
            'n_qk': self.n_qk, 'n_v': self.n_v, 'n_know': self.n_know,
        }

    def get_model_info(self):
        return [
            f"DAWN v{self.__version__}: Sense-Read-Write",
            f"  d_model={self.d_model}, d_route={self.d_route}, "
            f"n_layers={self.n_layers}, n_heads={self.n_heads}",
            f"  QK: {self.n_qk}, V: {self.n_v}, Know: {self.n_know}",
            f"  Per neuron: emb[{self.d_route}] + read[{self.d_model}] "
            f"+ write[{self.d_model}]",
        ]


# ================================================================
# 9. INFERENCE API — KV-cache prefill + decode
#    Pure functions only. Training code above is untouched.
# ================================================================

def _squeeze_params(params):
    """Remove leading singleton dim from all param arrays.

    Device-replicated checkpoints store params with shape (1, ...).
    Squeeze axis 0 when it is size 1 so indexing and matmul work correctly.
    """
    def _sq(x):
        if hasattr(x, 'ndim') and x.ndim >= 2 and x.shape[0] == 1:
            return x.squeeze(0)
        return x
    return jax.tree.map(_sq, params)


def _srw_inference(x, h, emb_norm, tau_offset, w_read, w_write):
    """Non-chunked SRW for inference. Binary gate + xr²-weighted soft denominator."""
    scores = h @ emb_norm.T
    scores_f32 = scores.astype(jnp.float32)
    s_mean = scores_f32.mean(axis=-1, keepdims=True)
    s_std = jnp.sqrt(jnp.mean(jnp.square(scores_f32 - s_mean),
                               axis=-1, keepdims=True)) + 1e-8
    tau = s_mean + tau_offset * s_std

    raw = scores - tau.astype(scores.dtype)
    z = raw.astype(jnp.float32) / s_std
    gate = jnp.where(z > 0, z * 0.5 * (1.0 + jax.lax.erf(z * 0.7071067811865476)), 0.0)

    r_n = w_read / (jnp.linalg.norm(w_read, axis=-1, keepdims=True) + 1e-8)
    w_n = w_write / (jnp.linalg.norm(w_write, axis=-1, keepdims=True) + 1e-8)
    xr = x @ r_n.T
    raw_out = (gate.astype(scores.dtype) * xr) @ w_n
    weighted = gate.sum(axis=-1, keepdims=True)
    den = jnp.maximum(weighted, 1.0)
    out = raw_out.astype(jnp.float32) / den
    return out.astype(jnp.float32)


def _srw_inference_with_gates(x, h, emb_norm, tau_offset, w_read, w_write):
    """Like _srw_inference but also returns raw and normalized gate for analysis.

    Returns: (out, gate_raw, gate_norm)
    """
    scores = h @ emb_norm.T
    scores_f32 = scores.astype(jnp.float32)
    s_mean = scores_f32.mean(axis=-1, keepdims=True)
    s_std = jnp.sqrt(jnp.mean(jnp.square(scores_f32 - s_mean),
                               axis=-1, keepdims=True)) + 1e-8
    tau = s_mean + tau_offset * s_std

    raw = scores - tau.astype(scores.dtype)
    z = raw.astype(jnp.float32) / s_std
    gate = jnp.where(z > 0, z * 0.5 * (1.0 + jax.lax.erf(z * 0.7071067811865476)), 0.0)
    gate_norm = gate.astype(jnp.float32) / jnp.maximum(gate.sum(axis=-1, keepdims=True), 1e-8)

    r_n = w_read / (jnp.linalg.norm(w_read, axis=-1, keepdims=True) + 1e-8)
    w_n = w_write / (jnp.linalg.norm(w_write, axis=-1, keepdims=True) + 1e-8)
    xr = x @ r_n.T
    raw_out = (gate.astype(scores.dtype) * xr) @ w_n
    weighted = gate.sum(axis=-1, keepdims=True)
    den = jnp.maximum(weighted, 1.0)
    out = raw_out.astype(jnp.float32) / den
    return out.astype(jnp.float32), gate, gate_norm


def _attn_forward_cached(x, pool_params, router_params, expand_O_kernel,
                         n_heads, d_model,
                         cache_K, cache_V, cache_len):
    """Cached attention decode step. x: [B, 1, D]."""
    B = x.shape[0]
    d_head = d_model // n_heads

    qk_emb = pool_params['qk_emb']
    qk_norm = qk_emb / (jnp.linalg.norm(qk_emb, axis=-1, keepdims=True) + 1e-8)
    v_emb = pool_params['v_emb']
    v_norm = v_emb / (jnp.linalg.norm(v_emb, axis=-1, keepdims=True) + 1e-8)

    h_all = x @ router_params['proj_attn']['kernel'] + router_params['proj_attn']['bias']
    h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)
    tau_all = x @ router_params['tau_attn']['kernel'] + router_params['tau_attn']['bias']

    Q = _srw_inference(x, h_Q, qk_norm, tau_all[:, :, 0:1],
                       pool_params['qk_read'], pool_params['qk_write'])
    K_new = _srw_inference(x, h_K, qk_norm, tau_all[:, :, 1:2],
                           pool_params['qk_read'], pool_params['qk_write'])
    V_new = _srw_inference(x, h_V, v_norm, tau_all[:, :, 2:3],
                           pool_params['v_read'], pool_params['v_write'])
    _qk_s = pool_params['qk_scale']
    _v_s = pool_params['v_scale']
    Q = Q * _qk_s
    K_new = K_new * _qk_s
    V_new = V_new * _v_s

    Q = Q.reshape(B, 1, n_heads, d_head).transpose(0, 2, 1, 3)
    K_new_h = K_new.reshape(B, 1, n_heads, d_head).transpose(0, 2, 1, 3)
    V_new_h = V_new.reshape(B, 1, n_heads, d_head).transpose(0, 2, 1, 3)

    cache_K = cache_K.at[:, :, cache_len, :].set(K_new_h[:, :, 0, :])
    cache_V = cache_V.at[:, :, cache_len, :].set(V_new_h[:, :, 0, :])

    scale = jnp.sqrt(jnp.float32(d_head))
    attn_scores = jnp.einsum('bhqd,bhkd->bhqk', Q, cache_K) / scale
    pos_mask = jnp.arange(cache_K.shape[2]) < (cache_len + 1)
    attn_scores = jnp.where(pos_mask[None, None, None, :], attn_scores,
                            jnp.finfo(attn_scores.dtype).min)
    attn_w = jax.nn.softmax(attn_scores, axis=-1)
    out = jnp.einsum('bhqk,bhkd->bhqd', attn_w, cache_V)

    out = out.transpose(0, 2, 1, 3).reshape(B, 1, d_model)
    out = out @ expand_O_kernel
    return out, cache_K, cache_V


def _know_forward_inference(x, pool_params, router_params):
    """Inference-only know forward. No chunking, no LB, no dropout."""
    know_emb = pool_params['know_emb']
    know_norm = know_emb / (jnp.linalg.norm(know_emb, axis=-1, keepdims=True) + 1e-8)
    h = x @ router_params['proj_know']['kernel'] + router_params['proj_know']['bias']
    tau = x @ router_params['tau_know']['kernel'] + router_params['tau_know']['bias']
    out = _srw_inference(x, h, know_norm, tau,
                         pool_params['know_read'], pool_params['know_write'])
    return out * pool_params['know_scale']


def prefill(params, model_cfg, input_ids):
    """Run full forward on prompt, populate KV cache.

    Returns: logits [B,S,vocab], cache_K, cache_V [n_layers,B,H,max_seq,d_head], cache_len
    """
    params = _squeeze_params(params)
    B, S = input_ids.shape
    d_model = model_cfg['d_model']
    n_layers = model_cfg['n_layers']
    n_heads = model_cfg['n_heads']
    max_seq = model_cfg['max_seq_len']
    d_head = d_model // n_heads

    pool_params = params['neuron_pool']
    router_params = params['router']

    positions = jnp.arange(S)[jnp.newaxis, :]
    x = params['token_emb']['embedding'][input_ids] + params['pos_emb']['embedding'][positions]

    qk_norm = pool_params['qk_emb'] / (
        jnp.linalg.norm(pool_params['qk_emb'], axis=-1, keepdims=True) + 1e-8)
    v_norm = pool_params['v_emb'] / (
        jnp.linalg.norm(pool_params['v_emb'], axis=-1, keepdims=True) + 1e-8)

    block_params_list = [params[f'block_{i}'] for i in range(n_layers)]
    stacked = jax.tree.map(lambda *arrays: jnp.stack(arrays), *block_params_list)

    cache_K = jnp.zeros((n_layers, B, n_heads, max_seq, d_head))
    cache_V = jnp.zeros((n_layers, B, n_heads, max_seq, d_head))

    def prefill_layer(carry, xs):
        x, cK, cV = carry
        bp = xs['params']
        layer_idx = xs['layer_idx']

        normed = _layer_norm(x, bp['norm1']['scale'], bp['norm1']['bias'])
        h_all = normed @ router_params['proj_attn']['kernel'] + router_params['proj_attn']['bias']
        h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)
        tau_all = normed @ router_params['tau_attn']['kernel'] + router_params['tau_attn']['bias']

        Q = _srw_inference(normed, h_Q, qk_norm, tau_all[:, :, 0:1],
                           pool_params['qk_read'], pool_params['qk_write'])
        K_val = _srw_inference(normed, h_K, qk_norm, tau_all[:, :, 1:2],
                               pool_params['qk_read'], pool_params['qk_write'])
        V_val = _srw_inference(normed, h_V, v_norm, tau_all[:, :, 2:3],
                               pool_params['v_read'], pool_params['v_write'])
        _qk_s = pool_params['qk_scale']
        _v_s = pool_params['v_scale']
        Q = Q * _qk_s
        K_val = K_val * _qk_s
        V_val = V_val * _v_s

        Q_h = Q.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
        K_h = K_val.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
        V_h = V_val.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)

        cK = cK.at[layer_idx, :, :, :S, :].set(K_h)
        cV = cV.at[layer_idx, :, :, :S, :].set(V_h)

        scale = jnp.sqrt(jnp.float32(d_head))
        scores = jnp.einsum('bhsd,bhtd->bhst', Q_h, K_h) / scale
        causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
        scores = jnp.where(causal, scores, jnp.finfo(scores.dtype).min)
        attn_w = jax.nn.softmax(scores, axis=-1)
        attn_out = jnp.einsum('bhst,bhtd->bhsd', attn_w, V_h)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, S, d_model)
        attn_out = attn_out @ bp['attn']['expand_O']['kernel']
        x = x + attn_out

        normed = _layer_norm(x, bp['norm2']['scale'], bp['norm2']['bias'])
        know_out = _know_forward_inference(normed, pool_params, router_params)
        x = x + know_out
        return (x, cK, cV), None

    xs = {'params': stacked, 'layer_idx': jnp.arange(n_layers)}
    (x, cache_K, cache_V), _ = jax.lax.scan(prefill_layer, (x, cache_K, cache_V), xs)

    norm_p = params['norm']
    x = _layer_norm(x, norm_p['scale'], norm_p['bias'])
    logits = x @ params['token_emb']['embedding'].T
    return logits, cache_K, cache_V, S


def decode_step(params, model_cfg, token_id, cache_K, cache_V, cache_len):
    """Single token decode with KV cache. Returns logits [B,vocab], updated cache."""
    params = _squeeze_params(params)
    token_id = token_id.reshape(-1, 1)
    B = token_id.shape[0]
    d_model = model_cfg['d_model']
    n_layers = model_cfg['n_layers']
    n_heads = model_cfg['n_heads']

    pool_params = params['neuron_pool']
    router_params = params['router']

    x = (params['token_emb']['embedding'][token_id]
         + params['pos_emb']['embedding'][cache_len][jnp.newaxis, :])

    block_params_list = [params[f'block_{i}'] for i in range(n_layers)]
    stacked = jax.tree.map(lambda *arrays: jnp.stack(arrays), *block_params_list)

    def decode_layer(carry, xs):
        x, cK, cV, pos = carry
        bp = xs['params']
        layer_idx = xs['layer_idx']

        normed = _layer_norm(x, bp['norm1']['scale'], bp['norm1']['bias'])
        attn_out, new_cK, new_cV = _attn_forward_cached(
            normed, pool_params, router_params,
            bp['attn']['expand_O']['kernel'],
            n_heads, d_model, cK[layer_idx], cV[layer_idx], pos)
        cK = cK.at[layer_idx].set(new_cK)
        cV = cV.at[layer_idx].set(new_cV)
        x = x + attn_out

        normed = _layer_norm(x, bp['norm2']['scale'], bp['norm2']['bias'])
        know_out = _know_forward_inference(normed, pool_params, router_params)
        x = x + know_out
        return (x, cK, cV, pos), None

    xs = {'params': stacked, 'layer_idx': jnp.arange(n_layers)}
    (x, cache_K, cache_V, _), _ = jax.lax.scan(
        decode_layer, (x, cache_K, cache_V, cache_len), xs)

    norm_p = params['norm']
    x = _layer_norm(x, norm_p['scale'], norm_p['bias'])
    logits = (x @ params['token_emb']['embedding'].T)[:, 0, :]
    return logits, cache_K, cache_V, cache_len + 1


# ================================================================
# 10. Vectorized analysis helpers (inference-only)
# ================================================================

def vectorized_eval(params, model_cfg, all_tokens, batch_size=32):
    """Validation without Python loops. all_tokens: [N_seqs, max_seq] on device.

    Uses jax.lax.scan over batches, _srw_inference per layer (no chunking).
    Returns: (avg_loss, ppl, accuracy, total_valid) — all jnp scalars.
    """
    params = _squeeze_params(params)
    n_seqs = all_tokens.shape[0]
    n_batches = n_seqs // batch_size
    tokens = all_tokens[:n_batches * batch_size].reshape(n_batches, batch_size, -1).astype(jnp.int32)

    d_model = model_cfg['d_model']
    n_layers = model_cfg['n_layers']
    n_heads = model_cfg['n_heads']
    max_seq = model_cfg['max_seq_len']

    pool_params = params['neuron_pool']
    router_params = params['router']
    norm_params = params['norm']
    emb_matrix = jnp.asarray(params['token_emb']['embedding'])
    pos_matrix = jnp.asarray(params['pos_emb']['embedding'])

    qk_norm = pool_params['qk_emb'] / (
        jnp.linalg.norm(pool_params['qk_emb'], axis=-1, keepdims=True) + 1e-8)
    v_norm = pool_params['v_emb'] / (
        jnp.linalg.norm(pool_params['v_emb'], axis=-1, keepdims=True) + 1e-8)
    know_norm = pool_params['know_emb'] / (
        jnp.linalg.norm(pool_params['know_emb'], axis=-1, keepdims=True) + 1e-8)

    block_params_list = [params[f'block_{i}'] for i in range(n_layers)]
    stacked = jax.tree.map(lambda *arrays: jnp.stack(arrays), *block_params_list)

    def forward_batch(input_ids):
        B, S = input_ids.shape
        positions = jnp.arange(S)[jnp.newaxis, :]
        x = emb_matrix[input_ids.astype(jnp.int32)] + pos_matrix[positions]

        def layer_fn(x, bp):
            normed = _layer_norm(x, bp['norm1']['scale'], bp['norm1']['bias'])
            h_all = normed @ router_params['proj_attn']['kernel'] + router_params['proj_attn']['bias']
            h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)
            tau_all = normed @ router_params['tau_attn']['kernel'] + router_params['tau_attn']['bias']

            Q = _srw_inference(normed, h_Q, qk_norm, tau_all[:, :, 0:1],
                               pool_params['qk_read'], pool_params['qk_write'])
            K = _srw_inference(normed, h_K, qk_norm, tau_all[:, :, 1:2],
                               pool_params['qk_read'], pool_params['qk_write'])
            V = _srw_inference(normed, h_V, v_norm, tau_all[:, :, 2:3],
                               pool_params['v_read'], pool_params['v_write'])
            _qk_s = pool_params['qk_scale']
            _v_s = pool_params['v_scale']
            Q = Q * _qk_s
            K = K * _qk_s
            V = V * _v_s

            d_head = d_model // n_heads
            Qr = Q.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
            Kr = K.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
            Vr = V.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)

            scale = jnp.sqrt(jnp.float32(d_head))
            scores = jnp.einsum('bhsd,bhtd->bhst', Qr, Kr) / scale
            causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
            scores = jnp.where(causal, scores, jnp.finfo(scores.dtype).min)
            attn_w = jax.nn.softmax(scores, axis=-1)
            attn_out = jnp.einsum('bhst,bhtd->bhsd', attn_w, Vr)
            attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, S, d_model)
            attn_out = attn_out @ bp['attn']['expand_O']['kernel']
            x = x + attn_out

            normed = _layer_norm(x, bp['norm2']['scale'], bp['norm2']['bias'])
            h_k = normed @ router_params['proj_know']['kernel'] + router_params['proj_know']['bias']
            tau_k = normed @ router_params['tau_know']['kernel'] + router_params['tau_know']['bias']
            know_out = _srw_inference(normed, h_k, know_norm, tau_k,
                                     pool_params['know_read'], pool_params['know_write'])
            x = x + know_out * pool_params['know_scale']
            return x, None

        x, _ = jax.lax.scan(layer_fn, x, stacked)
        x = _layer_norm(x, norm_params['scale'], norm_params['bias'])

        shift_x = x[:, :-1, :]
        shift_labels = input_ids[:, 1:].astype(jnp.int32)
        valid_mask = shift_labels > 0

        logits = shift_x @ emb_matrix.T
        log_probs = jax.nn.log_softmax(logits, axis=-1)
        safe_labels = jnp.where(valid_mask, shift_labels, 0)
        token_loss = -jnp.take_along_axis(
            log_probs, safe_labels[..., jnp.newaxis], axis=-1).squeeze(-1)
        total_loss = (token_loss * valid_mask).sum()
        preds = jnp.argmax(logits, axis=-1)
        correct = ((preds == shift_labels) & valid_mask).sum()
        valid_count = valid_mask.sum()
        return total_loss, correct, valid_count

    def scan_batches(carry, batch):
        tl, tc, tv = carry
        loss, correct, valid = forward_batch(batch)
        return (tl + loss, tc + correct, tv + valid), None

    init = (jnp.float32(0.0), jnp.int32(0), jnp.int32(0))
    (total_loss, total_correct, total_valid), _ = jax.lax.scan(
        scan_batches, init, tokens)

    avg_loss = total_loss / (total_valid + 1e-8)
    ppl = jnp.exp(avg_loss)
    acc = total_correct.astype(jnp.float32) / (total_valid + 1e-8) * 100.0
    return avg_loss, ppl, acc, total_valid


def vectorized_neuron_health(params):
    """All neuron health stats. Returns dict of jnp values (single device_get later)."""
    params = _squeeze_params(params)
    pool = params['neuron_pool']
    results = {}
    for pool_name, emb_key in [('QK', 'qk_emb'), ('V', 'v_emb'), ('Know', 'know_emb')]:
        emb = pool[emb_key]
        read = pool[emb_key.replace('emb', 'read')]
        write = pool[emb_key.replace('emb', 'write')]
        emb_n = jnp.linalg.norm(emb, axis=-1)
        read_n = jnp.linalg.norm(read, axis=-1)
        write_n = jnp.linalg.norm(write, axis=-1)
        results[pool_name] = {
            'N': emb.shape[0],
            'emb_mean': emb_n.mean(), 'emb_std': emb_n.std(),
            'emb_dead': (emb_n < 1e-6).sum(),
            'read_mean': read_n.mean(), 'read_std': read_n.std(),
            'read_dead': (read_n < 1e-6).sum(),
            'write_mean': write_n.mean(), 'write_std': write_n.std(),
            'write_dead': (write_n < 1e-6).sum(),
        }
    results['tau_attn_bias'] = params['router']['tau_attn']['bias']
    results['tau_know_bias'] = params['router']['tau_know']['bias']
    return results


def vectorized_weight_analysis(params, max_sample=2048):
    """Weight analysis: effective rank + cosine sim. All on device."""
    params = _squeeze_params(params)
    pool = params['neuron_pool']
    results = {}
    for pool_name, emb_key in [('QK', 'qk_emb'), ('V', 'v_emb'), ('Know', 'know_emb')]:
        emb = pool[emb_key]
        N, d = emb.shape
        if N > max_sample:
            idx = jnp.linspace(0, N - 1, max_sample, dtype=jnp.int32)
            emb_s = emb[idx]
        else:
            emb_s = emb
        norms = jnp.linalg.norm(emb_s, axis=-1, keepdims=True) + 1e-8
        emb_normed = emb_s / norms
        n_s = emb_normed.shape[0]

        gram = emb_normed @ emb_normed.T
        gram = gram - jnp.eye(n_s) * gram
        mean_sim = jnp.abs(gram).sum() / (n_s * (n_s - 1))
        max_sim = jnp.abs(gram).max()

        sv = jnp.linalg.svd(emb_s, compute_uv=False)
        sv_norm = sv / (sv.sum() + 1e-8)
        entropy = -(sv_norm * jnp.log(sv_norm + 1e-10)).sum()
        eff_rank = jnp.exp(entropy)

        results[pool_name] = {
            'N': N, 'd': d,
            'mean_cosine_sim': mean_sim,
            'max_cosine_sim': max_sim,
            'effective_rank': eff_rank,
            'top5_sv': sv[:5],
        }
    return results


def analysis_forward(params, model_cfg, input_ids, mode='full'):
    """Forward returning per-layer gate distributions + output norms.

    mode='full': returns gate_raw + gate_norm (R.1, P2, P3 etc.)
    mode='light': returns gate_norm only (R.2, D6 — half the memory)

    Returns:
        logits: [B, S, vocab]
        layer_info: dict with stacked arrays:
            gate_Q: [n_layers, B, S, n_qk]
            gate_K: [n_layers, B, S, n_qk]
            gate_V: [n_layers, B, S, n_v]
            gate_Know: [n_layers, B, S, n_know]
            (mode='full' only) gate_Q_raw, gate_K_raw, gate_V_raw, gate_Know_raw
            attn_out_norm: [n_layers]
            know_out_norm: [n_layers]
    """
    params = _squeeze_params(params)
    B, S = input_ids.shape
    d_model = model_cfg['d_model']
    n_layers = model_cfg['n_layers']
    n_heads = model_cfg['n_heads']

    pool_params = params['neuron_pool']
    router_params = params['router']

    positions = jnp.arange(S)[jnp.newaxis, :]
    x = params['token_emb']['embedding'][input_ids] + params['pos_emb']['embedding'][positions]

    qk_norm = pool_params['qk_emb'] / (
        jnp.linalg.norm(pool_params['qk_emb'], axis=-1, keepdims=True) + 1e-8)
    v_norm = pool_params['v_emb'] / (
        jnp.linalg.norm(pool_params['v_emb'], axis=-1, keepdims=True) + 1e-8)
    know_norm_w = pool_params['know_emb'] / (
        jnp.linalg.norm(pool_params['know_emb'], axis=-1, keepdims=True) + 1e-8)

    block_params_list = [params[f'block_{i}'] for i in range(n_layers)]
    stacked = jax.tree.map(lambda *arrays: jnp.stack(arrays), *block_params_list)

    _return_raw = (mode == 'full')

    def analysis_layer(carry, xs):
        x = carry
        bp = xs['params']

        normed = _layer_norm(x, bp['norm1']['scale'], bp['norm1']['bias'])
        h_all = normed @ router_params['proj_attn']['kernel'] + router_params['proj_attn']['bias']
        h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)
        tau_all = normed @ router_params['tau_attn']['kernel'] + router_params['tau_attn']['bias']

        Q, gate_Q_raw, gate_Q = _srw_inference_with_gates(
            normed, h_Q, qk_norm, tau_all[:, :, 0:1],
            pool_params['qk_read'], pool_params['qk_write'])
        K, gate_K_raw, gate_K = _srw_inference_with_gates(
            normed, h_K, qk_norm, tau_all[:, :, 1:2],
            pool_params['qk_read'], pool_params['qk_write'])
        V, gate_V_raw, gate_V = _srw_inference_with_gates(
            normed, h_V, v_norm, tau_all[:, :, 2:3],
            pool_params['v_read'], pool_params['v_write'])
        _qk_s = pool_params['qk_scale']
        _v_s = pool_params['v_scale']
        Q = Q * _qk_s
        K = K * _qk_s
        V = V * _v_s

        d_head = d_model // n_heads
        Qr = Q.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
        Kr = K.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
        Vr = V.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
        scale = jnp.sqrt(jnp.float32(d_head))
        scores = jnp.einsum('bhsd,bhtd->bhst', Qr, Kr) / scale
        causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
        scores = jnp.where(causal, scores, jnp.finfo(scores.dtype).min)
        attn_w = jax.nn.softmax(scores, axis=-1)
        attn_out = jnp.einsum('bhst,bhtd->bhsd', attn_w, Vr)
        attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, S, d_model)
        attn_out = attn_out @ bp['attn']['expand_O']['kernel']
        attn_out_norm = jnp.linalg.norm(attn_out, axis=-1).mean()
        x = x + attn_out

        normed = _layer_norm(x, bp['norm2']['scale'], bp['norm2']['bias'])
        h_k = normed @ router_params['proj_know']['kernel'] + router_params['proj_know']['bias']
        tau_k = normed @ router_params['tau_know']['kernel'] + router_params['tau_know']['bias']
        know_out, gate_Know_raw, gate_Know = _srw_inference_with_gates(
            normed, h_k, know_norm_w, tau_k,
            pool_params['know_read'], pool_params['know_write'])
        know_out = know_out * pool_params['know_scale']
        know_out_norm = jnp.linalg.norm(know_out, axis=-1).mean()
        x = x + know_out

        info = {
            'gate_Q': gate_Q, 'gate_K': gate_K,
            'gate_V': gate_V, 'gate_Know': gate_Know,
            'attn_out_norm': attn_out_norm,
            'know_out_norm': know_out_norm,
        }
        if _return_raw:
            info['gate_Q_raw'] = gate_Q_raw
            info['gate_K_raw'] = gate_K_raw
            info['gate_V_raw'] = gate_V_raw
            info['gate_Know_raw'] = gate_Know_raw
        return x, info

    xs = {'params': stacked}
    x, layer_info = jax.lax.scan(analysis_layer, x, xs)

    norm_p = params['norm']
    x = _layer_norm(x, norm_p['scale'], norm_p['bias'])
    logits = x @ params['token_emb']['embedding'].T
    return logits, layer_info


def build_suppressed_forward(params, model_cfg, suppress_masks):
    """Build forward with specific neurons suppressed (gate zeroed).

    suppress_masks: dict with 'qk':[n_qk] bool, 'v':[n_v], 'know':[n_know].
    True = suppress.
    Returns: forward_fn(input_ids) -> logits [B, S, vocab]
    """
    params = _squeeze_params(params)
    params = jax.tree.map(jnp.asarray, params)
    qk_mult = jnp.where(suppress_masks.get('qk', jnp.zeros(1, dtype=bool)), 0.0, 1.0) \
        if 'qk' in suppress_masks else None
    v_mult = jnp.where(suppress_masks.get('v', jnp.zeros(1, dtype=bool)), 0.0, 1.0) \
        if 'v' in suppress_masks else None
    know_mult = jnp.where(suppress_masks.get('know', jnp.zeros(1, dtype=bool)), 0.0, 1.0) \
        if 'know' in suppress_masks else None

    def _srw_sup(x, h, emb_n, tau_off, w_read, w_write, mult):
        """SRW with optional gate suppression. Sigmoid gate + xr²-weighted denominator."""
        scores = h @ emb_n.T
        sf = scores.astype(jnp.float32)
        s_mean = sf.mean(axis=-1, keepdims=True)
        s_std = jnp.sqrt(jnp.mean(jnp.square(sf - s_mean), axis=-1, keepdims=True)) + 1e-8
        tau = s_mean + tau_off * s_std
        raw = scores - tau.astype(scores.dtype)
        z = raw.astype(jnp.float32) / s_std
        gate = jnp.where(z > 0, z * 0.5 * (1.0 + jax.lax.erf(z * 0.7071067811865476)), 0.0)
        if mult is not None:
            gate = gate * mult[None, None, :]
        r_n = w_read / (jnp.linalg.norm(w_read, axis=-1, keepdims=True) + 1e-8)
        w_n = w_write / (jnp.linalg.norm(w_write, axis=-1, keepdims=True) + 1e-8)
        xr = x @ r_n.T
        out = (gate.astype(scores.dtype) * xr) @ w_n
        weighted = gate.sum(axis=-1, keepdims=True)
        den = jnp.maximum(weighted, 1.0)
        return (out.astype(jnp.float32) / den).astype(jnp.float32)

    def forward_fn(input_ids):
        B, S = input_ids.shape
        d_model = model_cfg['d_model']
        n_layers = model_cfg['n_layers']
        n_heads = model_cfg['n_heads']
        d_head = d_model // n_heads
        pp = params['neuron_pool']
        rp = params['router']

        positions = jnp.arange(S)[jnp.newaxis, :]
        x = params['token_emb']['embedding'][input_ids] + params['pos_emb']['embedding'][positions]
        qk_n = pp['qk_emb'] / (jnp.linalg.norm(pp['qk_emb'], axis=-1, keepdims=True) + 1e-8)
        v_n = pp['v_emb'] / (jnp.linalg.norm(pp['v_emb'], axis=-1, keepdims=True) + 1e-8)
        kn_n = pp['know_emb'] / (jnp.linalg.norm(pp['know_emb'], axis=-1, keepdims=True) + 1e-8)

        for i in range(n_layers):
            bp = params[f'block_{i}']
            normed = _layer_norm(x, bp['norm1']['scale'], bp['norm1']['bias'])
            h_all = normed @ rp['proj_attn']['kernel'] + rp['proj_attn']['bias']
            h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)
            tau_all = normed @ rp['tau_attn']['kernel'] + rp['tau_attn']['bias']

            Q = _srw_sup(normed, h_Q, qk_n, tau_all[:,:,0:1], pp['qk_read'], pp['qk_write'], qk_mult)
            K = _srw_sup(normed, h_K, qk_n, tau_all[:,:,1:2], pp['qk_read'], pp['qk_write'], qk_mult)
            V = _srw_sup(normed, h_V, v_n, tau_all[:,:,2:3], pp['v_read'], pp['v_write'], v_mult)
            _qk_s = pp['qk_scale']
            _v_s = pp['v_scale']
            Q = Q * _qk_s
            K = K * _qk_s
            V = V * _v_s

            Qr = Q.reshape(B,S,n_heads,d_head).transpose(0,2,1,3)
            Kr = K.reshape(B,S,n_heads,d_head).transpose(0,2,1,3)
            Vr = V.reshape(B,S,n_heads,d_head).transpose(0,2,1,3)
            sc = jnp.sqrt(jnp.float32(d_head))
            attn_s = jnp.einsum('bhsd,bhtd->bhst', Qr, Kr) / sc
            causal = jnp.tril(jnp.ones((S,S), dtype=jnp.bool_))
            attn_s = jnp.where(causal, attn_s, jnp.finfo(attn_s.dtype).min)
            attn_w = jax.nn.softmax(attn_s, axis=-1)
            attn_out = jnp.einsum('bhst,bhtd->bhsd', attn_w, Vr)
            attn_out = attn_out.transpose(0,2,1,3).reshape(B,S,d_model) @ bp['attn']['expand_O']['kernel']
            x = x + attn_out

            normed = _layer_norm(x, bp['norm2']['scale'], bp['norm2']['bias'])
            h_k = normed @ rp['proj_know']['kernel'] + rp['proj_know']['bias']
            tau_k = normed @ rp['tau_know']['kernel'] + rp['tau_know']['bias']
            x = x + _srw_sup(normed, h_k, kn_n, tau_k, pp['know_read'], pp['know_write'], know_mult) * pp['know_scale']

        norm_p = params['norm']
        x = _layer_norm(x, norm_p['scale'], norm_p['bias'])
        return x @ params['token_emb']['embedding'].T

    return forward_fn
