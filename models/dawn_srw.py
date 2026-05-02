"""
DAWN-SRW v4.1.5.5: SRW neurons and Residual State Transition.

This file contains the active v4.1.5.5 implementation. DAWN neurons are
implemented here as SRW neurons: a DAWN neuron has a signature and an
operation; in DAWN-SRW the signature is a route-space embedding, and the
operation is an RW operator parameterized by read/write vectors.

The model decision is the input-conditioned process by which the model forms
route queries, matches them against neuron signatures, scores candidate
neurons, selects a sparse subset, and composes their operations. In DAWN-SRW:

    route query -> signature matching -> Select -> selected SRW neurons
    -> RW operator composition -> transition output

Routing is performed in the learned route space:

    score_i = h @ emb_i

The read vector extracts a scalar component from the residual state, and the
write vector writes that scalar back to the residual stream:

    O_i^RW(x) = <x, r_i> w_i
    out = sum_i gate_i * O_i^RW(x) / max(sum_i gate_i, 1.0)

Core objects
------------
NeuronPool:
    attn_qk_emb, attn_v_emb, rst_emb         [N, d_route] signature embeddings
    attn_qk_read, attn_v_read, rst_read      [N, d_model] read vectors
    attn_qk_write, attn_v_write, rst_write   [N, d_model] write vectors
    attn_qk_scale, attn_v_scale, rst_scale   scalar output scales

Router:
    proj_attn: x -> h_Q, h_K, h_V in d_route space
    proj_rst: x -> h_RST in d_route space
    tau_*: learned relative threshold offsets
    raw_scan_offset_*: raw learned parameter for bounded scan offset

Layer flow
----------
    token + position embedding
    for each layer:
        LayerNorm
        Attention Layer:
            model decisions over attention-qk and attention-v pools -> Q/K/V
            causal self-attention for relational state interaction
        residual add
        LayerNorm
        RST Layer:
            model decision over the RST pool
            selected RW operators refine/contextualize the post-attention
            residual state and move it to the next representation state
        residual add
    final LayerNorm
    tied LM head

Each DAWN-SRW Block is Attention Layer + RST Layer. At the model level, the
Residual State Transition view is that the network repeatedly moves the
residual stream through representation states.

Historical note: old "know" parameter names have been replaced by RST
terminology. Legacy checkpoint migration is provided for old qk/v/know and
scan_bias parameter names.

SRW equations
-------------
For each token and pool:

    scores = h @ emb.T
    s_mean = mean(scores over neurons)
    s_std  = std(scores over neurons)

    scan_offset = scan_scale * tanh(raw_scan_offset)
    tau  = s_mean + tau_offset * s_std - scan_offset / max(s_std, scan_std_floor)

    raw           = scores - tau
    margin        = raw - activation_threshold
    activation    = sigmoid(sharpness * margin)
    active_margin = max(margin - activation_cutoff, 0)
    intensity     = epsilon + min(active_margin, max_intensity)
    gate          = activation * intensity

    O_i^RW(x) = <x, r_i> w_i
    out = sum_i gate_i * O_i^RW(x) / max(sum_i gate_i, 1.0)

Implementation notes
--------------------
* Routing is score-based in route space: scores = h @ emb.T.
* Read/write vectors are used raw in SRW; norm/product stats are diagnostics.
* The denominator remains activation-weighted intensity: sum(gate).
* Dynamic tau alpha parameters are not present; tau is relative threshold
  plus bounded scan offset.
* The neuron pool and router are shared across layers.  Per-layer parameters
  are LayerNorms and the attention output projection.
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from flax.core import FrozenDict, freeze, unfreeze
from typing import Optional, Dict
from functools import partial
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map


# ================================================================
# V4.1 physical constants (defaults; overridable via config).
#
#   margin        = (score - tau) - ACTIVATION_THRESHOLD
#   activation    = sigmoid(SHARPNESS * margin)
#   active_margin = max(margin - ACTIVATION_CUTOFF, 0)
#   intensity     = EPSILON + min(active_margin, MAX_INTENSITY)
#   gate          = activation * intensity
#   den           = max(sum(gate), 1.0)
# ================================================================

SHARPNESS = 500.0              # activation sigmoid sharpness (near-binary)
ACTIVATION_THRESHOLD = 0.5     # raw must exceed tau by this to enter zone
ACTIVATION_CUTOFF = 0.01       # intensity starts margin beyond this point
EPSILON = 1e-4                 # minimum intensity floor
MAX_INTENSITY = 10.0           # intensity cap (drift safety)
SCAN_SCALE = 0.01              # max absolute scan movement before /std
SCAN_STD_FLOOR = 0.5           # caps low-std scan amplification
DEFAULT_D_ROUTE = 64


# ================================================================
# 1. Helpers
# ================================================================

def safe_dropout(x, rate, deterministic, rng):
    if rate == 0.0:
        return x
    keep_rate = 1.0 - rate
    mask = jax.random.bernoulli(rng, keep_rate, x.shape)
    dropped = jnp.where(mask, x / keep_rate, 0.0)
    # Eval path: return x unscaled. Previous version returned x/keep_rate
    # here (mask forced to ones but the where-branch still divided), which
    # inflated all eval activations by 1/keep_rate and put a structural
    # offset into val_loss.
    return jnp.where(deterministic, x, dropped)


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
# 2. shard_map based gate + sense_read_write
# ================================================================


# ================================================================
# 3. shard_map based gate + sense_read_write
#    Per-device code with explicit psum communication.
#    fori_loop inside for N_local chunking.
# ================================================================

def make_sharded_srw(mesh, max_chunk_size=2048, dead_threshold=0.01,
                     sharpness=SHARPNESS,
                     activation_threshold=ACTIVATION_THRESHOLD,
                     activation_cutoff=ACTIVATION_CUTOFF,
                     epsilon=EPSILON,
                     max_intensity=MAX_INTENSITY,
                     scan_scale=SCAN_SCALE,
                     scan_std_floor=SCAN_STD_FLOOR,
                     analysis=False):
    """Create fused shard_map'd gate+srw. Gate never materialised full.

    2-pass chunked inside shard_map:
      Pass 1: scores stats -> static tau (psum for cross-chip mean/std)
      Pass 2: gate+srw fused per chunk (gate computed and consumed per chunk)

    v4.1 gate:
        raw           = score - tau
        margin        = raw - activation_threshold
        activation    = sigmoid(sharpness * margin)
        active_margin = max(margin - activation_cutoff, 0)
        intensity     = epsilon + min(active_margin, max_intensity)
        gate          = activation * intensity
        den           = max(sum(gate), 1.0)

    scan_offset = scan_scale * tanh(raw_scan_offset)
    tau = s_mean + tau_offset * s_std - scan_offset / max(s_std, scan_std_floor).
    All v4.1 constants are closure-baked.

    `analysis=False` (default, train path): returns the SLIM tuple plus
    four gate-concentration diagnostics, and skips distribution-shape stats
    (skew/kurt), boundary/entropy counters and intensity-cap fraction.
    XLA DCE's the unused work.
    `analysis=True`: returns the SLIM/concentration tuple followed by 11 extra
    observational scalars/arrays (score_skew, score_kurt, apt_std,
    gate_entropy, phi_binary, z_lt_075_frac, z_lt_030_frac,
    den_cost_out, activation_cost_out, current_cost_out, int_cap_frac).
    Used by analysis_step at val time only.
    """
    _model_axis_size = mesh.shape['model']
    _data_axis_size = mesh.shape['data']
    _dead_thresh = jnp.float32(dead_threshold)
    _sharp = jnp.float32(sharpness)
    _act_thr = jnp.float32(activation_threshold)
    _act_cut = jnp.float32(activation_cutoff)
    _eps = jnp.float32(epsilon)
    _max_int = jnp.float32(max_intensity)
    _scan_scale = jnp.float32(scan_scale)
    _scan_std_floor = jnp.float32(scan_std_floor)

    # SLIM out_specs: train path.
    _slim_out_specs = (
        P('data', None, None),   # out [B,S,D]
        P('data', None, None),   # active [B,S,1]
        P('data', None, None),   # gate_max [B,S,1]
        P(),                     # lb_loss scalar
        P(),                     # score_std scalar
        P(),                     # gate_sum scalar (sum gate, observational)
        P(),                     # active_n_mean scalar
        P('data', None, None),   # strong [B,S,1]
        P('data', None, None),   # z_mean_active [B,S,1]
        P(),                     # tau_abs_mean scalar
        P(),                     # dead_penalty scalar
        P(),                     # dead_count scalar
        P(),                     # int_max scalar (v4.1 diag)
        P(),                     # den_cost_mean scalar
        P(),                     # activation_cost_mean scalar
        P(),                     # current_cost_mean scalar
    )
    # ANALYSIS extras appended after slim.
    _analysis_extra_specs = (
        P('data', None, None),   # phi_binary (boundary frac) [B,S,1]
        P(),                     # z_lt_075_frac scalar
        P(),                     # z_lt_030_frac scalar
        P(),                     # score_skew scalar
        P(),                     # active_per_token_std scalar
        P(),                     # gate_entropy scalar
        P(),                     # den_cost_out
        P(),                     # activation_cost_out
        P(),                     # current_cost_out
        P(),                     # score_kurt scalar
        P(),                     # int_cap_frac scalar
    )
    _conc_out_specs = (
        P(),                     # gate_eff_n_mean scalar
        P(),                     # gate_eff_ratio_mean scalar
        P(),                     # top1_gate_frac_mean scalar
        P(),                     # top1_gate_frac_max scalar
    )
    _out_specs = (_slim_out_specs + _conc_out_specs + _analysis_extra_specs
                  if analysis else _slim_out_specs + _conc_out_specs)

    @partial(shard_map, mesh=mesh,
             in_specs=(P('data', None, None),    # x [B,S,D]
                       P('data', None, None),    # h [B,S,d_route]
                       P('model', None),          # route emb [N_local,d_route]
                       P('data', None, None),    # tau_offset [B,S,1]
                       P('data', None, None),    # raw_scan_offset [B,S,1]
                       P('model', None),          # read [N_local, D]
                       P('model', None)),         # write [N_local, D]
             out_specs=_out_specs,
             check_rep=False)
    def fused_gate_srw(x, h, emb_local, tau_offset, raw_scan_offset,
                       read_local, write_local):
        N_local = emb_local.shape[0]
        nc = max(1, (N_local + max_chunk_size - 1) // max_chunk_size)
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

        def route_chunk(start):
            ec = jax.lax.dynamic_slice_in_dim(emb_bf, start, cs, axis=0)
            rc = jax.lax.dynamic_slice_in_dim(read_bf, start, cs, axis=0)
            wc = jax.lax.dynamic_slice_in_dim(write_bf, start, cs, axis=0)
            rc_f = rc.astype(jnp.float32)
            wc_f = wc.astype(jnp.float32)
            # v4.1.5.5: use raw read/write vectors.
            # Their norms are natural vector magnitudes.
            # WD/pool_weight_decay regulates these norms.
            return ec, rc_f.astype(jnp.bfloat16), wc_f.astype(jnp.bfloat16)

        # --- Pass 1: exact stats over ALL chunks (scan + checkpoint) ---
        if analysis:
            @jax.checkpoint
            def stats_step(carry, i):
                s_sum, sq_sum, cube_sum, quad_sum, ns_sum, ns_sq = carry
                s = i * cs
                route, _, _ = route_chunk(s)
                scores = h_bf @ route.T
                scores_f = scores.astype(jnp.float32)
                s_sum = s_sum + scores_f.sum(axis=-1, keepdims=True)
                sq_sum = sq_sum + (scores_f ** 2).sum(axis=-1, keepdims=True)
                cube_sum = cube_sum + (scores_f ** 3).sum(axis=-1, keepdims=True)
                quad_sum = quad_sum + (scores_f ** 4).sum(axis=-1, keepdims=True)
                per_neuron_score = scores_f.mean(axis=(0, 1))  # [cs]
                ns_sum = ns_sum + per_neuron_score.sum()
                ns_sq = ns_sq + (per_neuron_score ** 2).sum()
                return (s_sum, sq_sum, cube_sum, quad_sum, ns_sum, ns_sq), None

            z_bs1 = jnp.zeros((B, S, 1))
            z_scalar = jnp.float32(0.0)
            (local_sum, local_sq, local_cube, local_quad, ns_sum, ns_sq), _ = jax.lax.scan(
                stats_step, (z_bs1, z_bs1, z_bs1, z_bs1, z_scalar, z_scalar), jnp.arange(nc))
        else:
            @jax.checkpoint
            def stats_step(carry, i):
                s_sum, sq_sum, ns_sum, ns_sq = carry
                s = i * cs
                route, _, _ = route_chunk(s)
                scores = h_bf @ route.T
                scores_f = scores.astype(jnp.float32)
                s_sum = s_sum + scores_f.sum(axis=-1, keepdims=True)
                sq_sum = sq_sum + (scores_f ** 2).sum(axis=-1, keepdims=True)
                per_neuron_score = scores_f.mean(axis=(0, 1))  # [cs]
                ns_sum = ns_sum + per_neuron_score.sum()
                ns_sq = ns_sq + (per_neuron_score ** 2).sum()
                return (s_sum, sq_sum, ns_sum, ns_sq), None

            z_bs1 = jnp.zeros((B, S, 1))
            z_scalar = jnp.float32(0.0)
            (local_sum, local_sq, ns_sum, ns_sq), _ = jax.lax.scan(
                stats_step, (z_bs1, z_bs1, z_scalar, z_scalar), jnp.arange(nc))

        global_sum = jax.lax.psum(local_sum, 'model')
        global_sq = jax.lax.psum(local_sq, 'model')
        N_total = N_local * _model_axis_size

        s_mean = global_sum / N_total
        s_std = jnp.sqrt(global_sq / N_total - s_mean ** 2) + 1e-8
        scan_offset = _scan_scale * jnp.tanh(raw_scan_offset)
        tau = s_mean + tau_offset * s_std - scan_offset / jnp.maximum(s_std, _scan_std_floor)

        if analysis:
            global_cube = jax.lax.psum(local_cube, 'model')
            global_quad = jax.lax.psum(local_quad, 'model')
            # Skewness from raw moments over the per-token score distribution.
            cube_mean = global_cube / N_total
            central_third = cube_mean - 3.0 * s_mean * (s_std ** 2) - s_mean ** 3
            score_skew = jax.lax.stop_gradient((central_third / (s_std ** 3 + 1e-8)).mean())
            quad_mean = global_quad / N_total
            # Kurtosis from raw moments over the per-token score distribution.
            central_fourth = (quad_mean - 4.0 * s_mean * cube_mean
                              + 6.0 * (s_mean ** 2) * (s_std ** 2) + 3.0 * s_mean ** 4)
            score_kurt = jax.lax.stop_gradient((central_fourth / (s_std ** 4 + 1e-8)).mean())

        # Score LB: variance of per-neuron score mean * N
        ns_sum = jax.lax.psum(ns_sum, 'data') / _data_axis_size
        ns_sq = jax.lax.psum(ns_sq, 'data') / _data_axis_size
        global_ns_sum = jax.lax.psum(ns_sum, 'model')
        global_ns_sq = jax.lax.psum(ns_sq, 'model')
        mean_score = global_ns_sum / N_total
        var_score = global_ns_sq / N_total - mean_score ** 2
        score_lb = var_score / (mean_score ** 2 + var_score + 1e-2)

        # --- Pass 2: gate + srw fused (scan + checkpoint) ---
        # v4.1 diagnostic: ceiling on intensity relative to cap (1e-3 below).
        if analysis:
            _int_cap_thresh = _eps + _max_int - jnp.float32(1e-3)

            @jax.checkpoint
            def gate_srw_step(carry, i):
                (out, total_weighted_cost, total_gate_sq, total_gate_max, total_active,
                 total_strong, total_phi_binary, total_den_cost,
                 total_activation_cost, total_current_cost,
                 total_z_lt_075, total_z_lt_030, total_g_log_g,
                 total_dead_penalty, total_dead_count,
                 total_int_max, total_int_cap_count) = carry
                s = i * cs
                route, rc, wc = route_chunk(s)
                scores = h_bf @ route.T
                scores_f = scores.astype(jnp.float32)
                raw = scores_f - tau
                margin = raw - _act_thr
                activation = jax.nn.sigmoid(_sharp * margin)
                active_margin = jnp.maximum(margin - _act_cut, 0.0)
                intensity = _eps + jnp.minimum(active_margin, _max_int)
                gate = activation * intensity
                chunk_int_max = intensity.max()
                chunk_int_cap_count = (intensity >= _int_cap_thresh
                                        ).astype(jnp.float32).sum()
                xr = x_bf @ rc.T
                xr_f = xr.astype(jnp.float32)
                a = gate * xr_f
                c_out = (a.astype(jnp.bfloat16) @ wc).astype(jnp.float32)
                chunk_weighted = gate.sum(axis=-1, keepdims=True)
                chunk_gate_sq = jnp.square(gate).sum(axis=-1, keepdims=True)
                chunk_intensity = gate.sum(axis=-1, keepdims=True)
                chunk_active = (activation > 0.5).astype(jnp.float32).sum(axis=-1, keepdims=True)
                chunk_strong = (activation > 0.9).astype(jnp.float32).sum(axis=-1, keepdims=True)
                chunk_phi_binary = ((activation > 0.1) & (activation < 0.9)
                                    ).astype(jnp.float32).sum(axis=-1, keepdims=True)
                chunk_z_lt_075 = ((activation > 0.05) & (activation < 0.95)
                                  ).astype(jnp.float32).sum(axis=-1, keepdims=True)
                chunk_z_lt_030 = ((activation > 0.2) & (activation < 0.8)
                                  ).astype(jnp.float32).sum(axis=-1, keepdims=True)
                g_safe = gate + 1e-8
                chunk_g_log_g = (gate * jnp.log(g_safe)).sum(axis=-1, keepdims=True)
                max_gate_chunk = gate.max(axis=(0, 1))
                mean_score_chunk = scores_f.mean(axis=(0, 1))
                max_gate_chunk = jax.lax.pmax(
                    jax.lax.stop_gradient(max_gate_chunk), 'data')
                mean_score_chunk = jax.lax.pmean(mean_score_chunk, 'data')
                dead_mask_chunk = jax.lax.stop_gradient(
                    (max_gate_chunk < _dead_thresh).astype(jnp.float32))
                penalty_chunk = jax.nn.relu(-mean_score_chunk) * dead_mask_chunk
                chunk_dead_penalty = penalty_chunk.sum()
                chunk_dead_count = dead_mask_chunk.sum()
                return (out + c_out,
                        total_weighted_cost + chunk_weighted,
                        total_gate_sq + chunk_gate_sq,
                        jnp.maximum(total_gate_max, gate.max(axis=-1, keepdims=True)),
                        total_active + chunk_active,
                        total_strong + chunk_strong,
                        total_phi_binary + chunk_phi_binary,
                        total_den_cost + chunk_intensity,
                        total_activation_cost,
                        total_current_cost,
                        total_z_lt_075 + chunk_z_lt_075,
                        total_z_lt_030 + chunk_z_lt_030,
                        total_g_log_g + chunk_g_log_g,
                        total_dead_penalty + chunk_dead_penalty,
                        total_dead_count + chunk_dead_count,
                        jnp.maximum(total_int_max, chunk_int_max),
                        total_int_cap_count + chunk_int_cap_count), None

            (raw_out, total_weighted_cost, total_gate_sq, total_gate_max, total_active, total_strong,
             total_phi_binary, total_den_cost, total_activation_cost,
             total_current_cost, total_z_lt_075, total_z_lt_030,
             total_g_log_g, total_dead_penalty, total_dead_count,
             total_int_max, total_int_cap_count), _ = jax.lax.scan(
                gate_srw_step,
                (jnp.zeros((B, S, D), dtype=jnp.float32),
                 z1, z1, jnp.full((B, S, 1), -1e9), z1, z1, z1, z1, z1, z1, z1, z1, z1,
                 jnp.float32(0.0), jnp.float32(0.0),
                 jnp.float32(0.0), jnp.float32(0.0)),
                jnp.arange(nc))
        else:
            @jax.checkpoint
            def gate_srw_step(carry, i):
                (out, total_weighted_cost, total_gate_sq, total_gate_max, total_active,
                 total_strong, total_den_cost,
                 total_activation_cost, total_current_cost,
                 total_dead_penalty, total_dead_count,
                 total_int_max) = carry
                s = i * cs
                route, rc, wc = route_chunk(s)
                scores = h_bf @ route.T
                scores_f = scores.astype(jnp.float32)
                raw = scores_f - tau
                margin = raw - _act_thr
                activation = jax.nn.sigmoid(_sharp * margin)
                active_margin = jnp.maximum(margin - _act_cut, 0.0)
                intensity = _eps + jnp.minimum(active_margin, _max_int)
                gate = activation * intensity
                chunk_int_max = intensity.max()
                xr = x_bf @ rc.T
                xr_f = xr.astype(jnp.float32)
                a = gate * xr_f
                c_out = (a.astype(jnp.bfloat16) @ wc).astype(jnp.float32)
                chunk_weighted = gate.sum(axis=-1, keepdims=True)
                chunk_gate_sq = jnp.square(gate).sum(axis=-1, keepdims=True)
                chunk_intensity = gate.sum(axis=-1, keepdims=True)
                chunk_active = (activation > 0.5).astype(jnp.float32).sum(axis=-1, keepdims=True)
                chunk_strong = (activation > 0.9).astype(jnp.float32).sum(axis=-1, keepdims=True)
                # sum(gate) feeds the denominator after the chunk scan.
                max_gate_chunk = gate.max(axis=(0, 1))
                mean_score_chunk = scores_f.mean(axis=(0, 1))
                max_gate_chunk = jax.lax.pmax(
                    jax.lax.stop_gradient(max_gate_chunk), 'data')
                mean_score_chunk = jax.lax.pmean(mean_score_chunk, 'data')
                dead_mask_chunk = jax.lax.stop_gradient(
                    (max_gate_chunk < _dead_thresh).astype(jnp.float32))
                penalty_chunk = jax.nn.relu(-mean_score_chunk) * dead_mask_chunk
                chunk_dead_penalty = penalty_chunk.sum()
                chunk_dead_count = dead_mask_chunk.sum()
                return (out + c_out,
                        total_weighted_cost + chunk_weighted,
                        total_gate_sq + chunk_gate_sq,
                        jnp.maximum(total_gate_max, gate.max(axis=-1, keepdims=True)),
                        total_active + chunk_active,
                        total_strong + chunk_strong,
                        total_den_cost + chunk_intensity,
                        total_activation_cost,
                        total_current_cost,
                        total_dead_penalty + chunk_dead_penalty,
                        total_dead_count + chunk_dead_count,
                        jnp.maximum(total_int_max, chunk_int_max)), None

            (raw_out, total_weighted_cost, total_gate_sq, total_gate_max, total_active, total_strong,
             total_den_cost, total_activation_cost, total_current_cost,
             total_dead_penalty, total_dead_count,
             total_int_max), _ = jax.lax.scan(
                gate_srw_step,
                (jnp.zeros((B, S, D), dtype=jnp.float32),
                 z1, z1, jnp.full((B, S, 1), -1e9), z1, z1, z1, z1, z1,
                 jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0)),
                jnp.arange(nc))

        global_weighted_cost = jax.lax.psum(total_weighted_cost, 'model')  # sum(gate)
        global_gate_sq = jax.lax.psum(total_gate_sq, 'model')
        # Denominator matches the numerator gate weight:
        # max(sum(activation * intensity), 1.0).
        global_den_cost = jax.lax.psum(total_den_cost, 'model')
        global_activation_cost = jax.lax.psum(total_activation_cost, 'model')
        global_current_cost = jax.lax.psum(total_current_cost, 'model')
        global_gate_max = jax.lax.pmax(jax.lax.stop_gradient(total_gate_max), 'model')
        den = jnp.maximum(global_den_cost, 1.0)
        out = raw_out / den
        out = jax.lax.psum(out.astype(jnp.bfloat16), 'model')

        global_active = jax.lax.psum(total_active, 'model')
        active_frac = global_active / N_total
        strong_frac = jax.lax.psum(total_strong, 'model') / N_total
        z_mean_active = jax.lax.psum(total_weighted_cost, 'model') / (global_active + 1e-8)

        score_std_out = s_std.mean()
        es_out = global_weighted_cost.mean()          # sum(gate), observational
        active_n_mean = global_active.mean()
        gate_eff_n = jax.lax.stop_gradient(
            (global_weighted_cost ** 2) / (global_gate_sq + 1e-8))
        gate_eff_ratio = jax.lax.stop_gradient(
            gate_eff_n / jnp.maximum(global_active, 1.0))
        top1_gate_frac = jax.lax.stop_gradient(
            global_gate_max / jnp.maximum(global_weighted_cost, 1e-8))
        tau_abs_mean = jax.lax.stop_gradient(tau).mean()
        dead_penalty_out = jax.lax.psum(total_dead_penalty, 'model')
        dead_count_out = jax.lax.stop_gradient(
            jax.lax.psum(total_dead_count, 'model'))
        # pmax has no VJP; wrap the input in stop_gradient.
        int_max_out = jax.lax.pmax(
            jax.lax.stop_gradient(total_int_max), 'model')

        den_cost_mean = global_den_cost.mean()
        activation_cost_mean = global_activation_cost.mean()
        current_cost_mean = global_current_cost.mean()

        slim_out = (out.astype(jnp.float32), active_frac, global_gate_max, score_lb,
                    score_std_out, es_out, active_n_mean, strong_frac, z_mean_active,
                    tau_abs_mean, dead_penalty_out, dead_count_out, int_max_out,
                    den_cost_mean, activation_cost_mean, current_cost_mean)
        conc_out = (gate_eff_n.mean(), gate_eff_ratio.mean(),
                    top1_gate_frac.mean(), top1_gate_frac.max())
        if not analysis:
            return slim_out + conc_out

        # --- Analysis-only extras ---
        phi_binary_frac = jax.lax.psum(total_phi_binary, 'model') / N_total
        # Safety floor: active can collapse to 0 at init; clamp to 1.0.
        _active_denom = jnp.maximum(global_active, 1.0)
        z_lt_075_frac = jax.lax.stop_gradient(
            (jax.lax.psum(total_z_lt_075, 'model') / _active_denom).mean())
        z_lt_030_frac = jax.lax.stop_gradient(
            (jax.lax.psum(total_z_lt_030, 'model') / _active_denom).mean())
        active_per_token_std = jax.lax.stop_gradient(global_active).std()
        global_g_log_g = jax.lax.psum(total_g_log_g, 'model')
        gate_sum_eps = jnp.maximum(global_weighted_cost, 1e-6)
        safe_glogg = jnp.where(global_weighted_cost > 1e-6, global_g_log_g, 0.0)
        entropy_per_token = -safe_glogg / gate_sum_eps + jnp.log(gate_sum_eps)
        entropy_per_token = jnp.where(
            jnp.isfinite(entropy_per_token), entropy_per_token, 0.0)
        gate_entropy = jax.lax.stop_gradient(entropy_per_token).mean()
        den_cost_out = global_den_cost.mean()
        activation_cost_out = jax.lax.psum(total_activation_cost, 'model').mean()
        current_cost_out = jax.lax.psum(total_current_cost, 'model').mean()
        int_cap_frac_out = jax.lax.stop_gradient(
            jax.lax.psum(total_int_cap_count, 'model')
            / jnp.float32(B * S * N_total))
        return slim_out + conc_out + (phi_binary_frac, z_lt_075_frac, z_lt_030_frac,
                           score_skew, active_per_token_std, gate_entropy,
                           den_cost_out, activation_cost_out, current_cost_out,
                           score_kurt, int_cap_frac_out)

    return fused_gate_srw


def make_sharded_srw_paired(mesh, max_chunk_size=2048, dead_threshold=0.01,
                            sharpness=SHARPNESS,
                            activation_threshold=ACTIVATION_THRESHOLD,
                            activation_cutoff=ACTIVATION_CUTOFF,
                            epsilon=EPSILON,
                            max_intensity=MAX_INTENSITY,
                            scan_scale=SCAN_SCALE,
                            scan_std_floor=SCAN_STD_FLOOR,
                            analysis=False):
    """Fused Q+K shard_map: two routes sharing same pool in one shard_map call.

    h is [B,S,2,d_route] (h_Q, h_K stacked on axis=2).
    tau_offset and raw_scan_offset are [B,S,2,1].
    x @ read.T computed once (shared by both routes).
    Scores stats computed independently per route.
    Returns out [B,S,2,D], active [B,S,1], gate_max [B,S,1].

    v4.1 gate: activation * intensity (see make_sharded_srw docstring).
    analysis: see make_sharded_srw docstring.
    """
    _model_axis_size = mesh.shape['model']
    _data_axis_size = mesh.shape['data']
    _dead_thresh = jnp.float32(dead_threshold)
    _sharp = jnp.float32(sharpness)
    _act_thr = jnp.float32(activation_threshold)
    _act_cut = jnp.float32(activation_cutoff)
    _eps = jnp.float32(epsilon)
    _max_int = jnp.float32(max_intensity)
    _scan_scale = jnp.float32(scan_scale)
    _scan_std_floor = jnp.float32(scan_std_floor)

    _slim_out_specs = (
        P('data', None, None, None),  # out [B,S,2,D]
        P('data', None, None),        # active [B,S,1]
        P('data', None, None),        # gate_max [B,S,1]
        P(),                          # lb_loss scalar
        P(),                          # score_std scalar
        P(),                          # gate_sum scalar
        P(),                          # active_n_mean scalar
        P('data', None, None),        # strong [B,S,1]
        P('data', None, None),        # z_mean_active [B,S,1]
        P(),                          # tau_abs_mean scalar
        P(),                          # dead_penalty scalar
        P(),                          # dead_count scalar
        P(),                          # int_max scalar (v4.1 diag)
        P(),                          # den_cost_mean scalar
        P(),                          # activation_cost_mean scalar
        P(),                          # current_cost_mean scalar
    )
    _analysis_extra_specs = (
        P('data', None, None),        # phi_binary [B,S,1]
        P(),                          # z_lt_075_frac scalar
        P(),                          # z_lt_030_frac scalar
        P(),                          # score_skew scalar
        P(),                          # active_per_token_std scalar
        P(),                          # gate_entropy scalar
        P(),                          # den_cost_out scalar
        P(),                          # activation_cost_out scalar
        P(),                          # current_cost_out scalar
        P(),                          # score_kurt scalar
        P(),                          # int_cap_frac scalar
    )
    _conc_out_specs = (
        P(),                          # gate_eff_n_mean scalar
        P(),                          # gate_eff_ratio_mean scalar
        P(),                          # top1_gate_frac_mean scalar
        P(),                          # top1_gate_frac_max scalar
    )
    _out_specs = (_slim_out_specs + _conc_out_specs + _analysis_extra_specs
                  if analysis else _slim_out_specs + _conc_out_specs)

    @partial(shard_map, mesh=mesh,
             in_specs=(P('data', None, None),        # x [B,S,D]
                       P('data', None, None, None),  # h [B,S,2,d_route]
                       P('model', None),              # route emb [N_local,d_route]
                       P('data', None, None, None),  # tau_offset [B,S,2,1]
                       P('data', None, None, None),  # raw_scan_offset [B,S,2,1]
                       P('model', None),              # read [N_local, D]
                       P('model', None)),             # write [N_local, D]
             out_specs=_out_specs,
             check_rep=False)
    def fused_gate_srw_paired(x, h, emb_local, tau_offset, raw_scan_offset,
                              read_local, write_local):
        N_local = emb_local.shape[0]
        nc = max(1, (N_local + max_chunk_size - 1) // max_chunk_size)
        while N_local % nc != 0 and nc < N_local:
            nc += 1
        cs = N_local // nc

        B, S, D = x.shape
        # h: [B,S,2,d_route], tau_offset/raw_scan_offset: [B,S,2,1]
        h_bf = h.astype(jnp.bfloat16)
        x_bf = x.astype(jnp.bfloat16)
        emb_bf = emb_local.astype(jnp.bfloat16)
        read_bf = read_local.astype(jnp.bfloat16)
        write_bf = write_local.astype(jnp.bfloat16)
        z1_r = jnp.zeros((B, S, 2, 1))

        def route_chunk(start):
            ec = jax.lax.dynamic_slice_in_dim(emb_bf, start, cs, axis=0)
            rc = jax.lax.dynamic_slice_in_dim(read_bf, start, cs, axis=0)
            wc = jax.lax.dynamic_slice_in_dim(write_bf, start, cs, axis=0)
            rc_f = rc.astype(jnp.float32)
            wc_f = wc.astype(jnp.float32)
            # v4.1.5.5: use raw read/write vectors.
            # Their norms are natural vector magnitudes.
            # WD/pool_weight_decay regulates these norms.
            return ec, rc_f.astype(jnp.bfloat16), wc_f.astype(jnp.bfloat16)

        # --- Pass 1: exact stats over ALL chunks (scan + checkpoint) ---
        if analysis:
            @jax.checkpoint
            def stats_step(carry, i):
                s_sum, sq_sum, cube_sum, quad_sum, ns_sum, ns_sq = carry
                s = i * cs
                route, _, _ = route_chunk(s)
                scores = jnp.einsum('bsrd,nd->bsrn', h_bf, route)
                scores_f = scores.astype(jnp.float32)
                s_sum = s_sum + scores_f.sum(axis=-1, keepdims=True)
                sq_sum = sq_sum + (scores_f ** 2).sum(axis=-1, keepdims=True)
                cube_sum = cube_sum + (scores_f ** 3).sum(axis=-1, keepdims=True)
                quad_sum = quad_sum + (scores_f ** 4).sum(axis=-1, keepdims=True)
                per_neuron_score = scores_f.mean(axis=(0, 1, 2))  # [cs]
                ns_sum = ns_sum + per_neuron_score.sum()
                ns_sq = ns_sq + (per_neuron_score ** 2).sum()
                return (s_sum, sq_sum, cube_sum, quad_sum, ns_sum, ns_sq), None

            z_bsr1 = jnp.zeros((B, S, 2, 1))
            z_scalar = jnp.float32(0.0)
            (local_sum, local_sq, local_cube, local_quad, ns_sum, ns_sq), _ = jax.lax.scan(
                stats_step, (z_bsr1, z_bsr1, z_bsr1, z_bsr1, z_scalar, z_scalar), jnp.arange(nc))
        else:
            @jax.checkpoint
            def stats_step(carry, i):
                s_sum, sq_sum, ns_sum, ns_sq = carry
                s = i * cs
                route, _, _ = route_chunk(s)
                scores = jnp.einsum('bsrd,nd->bsrn', h_bf, route)
                scores_f = scores.astype(jnp.float32)
                s_sum = s_sum + scores_f.sum(axis=-1, keepdims=True)
                sq_sum = sq_sum + (scores_f ** 2).sum(axis=-1, keepdims=True)
                per_neuron_score = scores_f.mean(axis=(0, 1, 2))  # [cs]
                ns_sum = ns_sum + per_neuron_score.sum()
                ns_sq = ns_sq + (per_neuron_score ** 2).sum()
                return (s_sum, sq_sum, ns_sum, ns_sq), None

            z_bsr1 = jnp.zeros((B, S, 2, 1))
            z_scalar = jnp.float32(0.0)
            (local_sum, local_sq, ns_sum, ns_sq), _ = jax.lax.scan(
                stats_step, (z_bsr1, z_bsr1, z_scalar, z_scalar), jnp.arange(nc))

        global_sum = jax.lax.psum(local_sum, 'model')  # [B,S,2,1]
        global_sq = jax.lax.psum(local_sq, 'model')
        N_total = N_local * _model_axis_size

        s_mean = global_sum / N_total      # [B,S,2,1]
        s_std = jnp.sqrt(global_sq / N_total - s_mean ** 2) + 1e-8
        scan_offset = _scan_scale * jnp.tanh(raw_scan_offset)
        tau = s_mean + tau_offset * s_std - scan_offset / jnp.maximum(s_std, _scan_std_floor)

        if analysis:
            global_cube = jax.lax.psum(local_cube, 'model')
            global_quad = jax.lax.psum(local_quad, 'model')
            cube_mean = global_cube / N_total
            central_third = cube_mean - 3.0 * s_mean * (s_std ** 2) - s_mean ** 3
            score_skew = jax.lax.stop_gradient((central_third / (s_std ** 3 + 1e-8)).mean())
            quad_mean = global_quad / N_total
            central_fourth = (quad_mean - 4.0 * s_mean * cube_mean
                              + 6.0 * (s_mean ** 2) * (s_std ** 2) + 3.0 * s_mean ** 4)
            score_kurt = jax.lax.stop_gradient((central_fourth / (s_std ** 4 + 1e-8)).mean())

        # Score LB: variance of per-neuron score mean * N
        ns_sum = jax.lax.psum(ns_sum, 'data') / _data_axis_size
        ns_sq = jax.lax.psum(ns_sq, 'data') / _data_axis_size
        global_ns_sum = jax.lax.psum(ns_sum, 'model')
        global_ns_sq = jax.lax.psum(ns_sq, 'model')
        mean_score = global_ns_sum / N_total
        var_score = global_ns_sq / N_total - mean_score ** 2
        score_lb = var_score / (mean_score ** 2 + var_score + 1e-2)

        # --- Pass 2: gate + srw fused ---
        if analysis:
            _int_cap_thresh_paired = _eps + _max_int - jnp.float32(1e-3)

            @jax.checkpoint
            def gate_srw_step(carry, i):
                (out, total_weighted_cost, total_gate_sq, total_gate_max, total_active,
                 total_strong, total_phi_binary, total_den_cost,
                 total_activation_cost, total_current_cost,
                 total_z_lt_075, total_z_lt_030, total_g_log_g,
                 total_dead_penalty, total_dead_count,
                 total_int_max, total_int_cap_count) = carry
                s = i * cs
                route, rc, wc = route_chunk(s)
                scores = jnp.einsum('bsrd,nd->bsrn', h_bf, route)
                scores_f = scores.astype(jnp.float32)
                raw = scores_f - tau
                margin = raw - _act_thr
                activation = jax.nn.sigmoid(_sharp * margin)
                active_margin = jnp.maximum(margin - _act_cut, 0.0)
                intensity = _eps + jnp.minimum(active_margin, _max_int)
                gate = activation * intensity
                chunk_int_max = intensity.max()
                chunk_int_cap_count = (intensity >= _int_cap_thresh_paired
                                        ).astype(jnp.float32).sum()
                xr = x_bf @ rc.T  # [B,S,N]
                xr_f = xr.astype(jnp.float32)
                a = gate * xr_f[:, :, None, :]
                c_out = jnp.einsum('bsrn,nd->bsrd', a.astype(jnp.bfloat16), wc).astype(jnp.float32)
                chunk_weighted = gate.sum(axis=-1, keepdims=True)           # [B,S,2,1]
                chunk_gate_sq = jnp.square(gate).sum(axis=-1, keepdims=True)
                chunk_intensity = gate.sum(axis=-1, keepdims=True)
                chunk_active = (activation > 0.5).astype(jnp.float32).sum(axis=-1, keepdims=True)
                chunk_strong = (activation > 0.9).astype(jnp.float32).sum(axis=-1, keepdims=True)
                chunk_phi_binary = ((activation > 0.1) & (activation < 0.9)
                                    ).astype(jnp.float32).sum(axis=-1, keepdims=True)
                chunk_z_lt_075 = ((activation > 0.05) & (activation < 0.95)
                                  ).astype(jnp.float32).sum(axis=-1, keepdims=True)
                chunk_z_lt_030 = ((activation > 0.2) & (activation < 0.8)
                                  ).astype(jnp.float32).sum(axis=-1, keepdims=True)
                g_safe = gate + 1e-8
                chunk_g_log_g = (gate * jnp.log(g_safe)).sum(axis=-1, keepdims=True)
                max_gate_chunk = gate.max(axis=(0, 1, 2))
                mean_score_chunk = scores_f.mean(axis=(0, 1, 2))
                max_gate_chunk = jax.lax.pmax(
                    jax.lax.stop_gradient(max_gate_chunk), 'data')
                mean_score_chunk = jax.lax.pmean(mean_score_chunk, 'data')
                dead_mask_chunk = jax.lax.stop_gradient(
                    (max_gate_chunk < _dead_thresh).astype(jnp.float32))
                penalty_chunk = jax.nn.relu(-mean_score_chunk) * dead_mask_chunk
                chunk_dead_penalty = penalty_chunk.sum()
                chunk_dead_count = dead_mask_chunk.sum()
                return (out + c_out,
                        total_weighted_cost + chunk_weighted,
                        total_gate_sq + chunk_gate_sq,
                        jnp.maximum(total_gate_max, gate.max(axis=-1, keepdims=True)),
                        total_active + chunk_active,
                        total_strong + chunk_strong,
                        total_phi_binary + chunk_phi_binary,
                        total_den_cost + chunk_intensity,
                        total_activation_cost,
                        total_current_cost,
                        total_z_lt_075 + chunk_z_lt_075,
                        total_z_lt_030 + chunk_z_lt_030,
                        total_g_log_g + chunk_g_log_g,
                        total_dead_penalty + chunk_dead_penalty,
                        total_dead_count + chunk_dead_count,
                        jnp.maximum(total_int_max, chunk_int_max),
                        total_int_cap_count + chunk_int_cap_count), None

            (raw_out, total_weighted_cost, total_gate_sq, total_gate_max, total_active, total_strong,
             total_phi_binary, total_den_cost, total_activation_cost,
             total_current_cost, total_z_lt_075, total_z_lt_030,
             total_g_log_g, total_dead_penalty, total_dead_count,
             total_int_max, total_int_cap_count), _ = jax.lax.scan(
                gate_srw_step,
                (jnp.zeros((B, S, 2, D), dtype=jnp.float32),
                 z1_r, z1_r, jnp.full((B, S, 2, 1), -1e9),
                 z1_r, z1_r, z1_r, z1_r, z1_r, z1_r, z1_r, z1_r, z1_r,
                 jnp.float32(0.0), jnp.float32(0.0),
                 jnp.float32(0.0), jnp.float32(0.0)),
                jnp.arange(nc))
        else:
            @jax.checkpoint
            def gate_srw_step(carry, i):
                (out, total_weighted_cost, total_gate_sq, total_gate_max, total_active,
                 total_strong, total_den_cost,
                 total_activation_cost, total_current_cost,
                 total_dead_penalty, total_dead_count,
                 total_int_max) = carry
                s = i * cs
                route, rc, wc = route_chunk(s)
                scores = jnp.einsum('bsrd,nd->bsrn', h_bf, route)
                scores_f = scores.astype(jnp.float32)
                raw = scores_f - tau
                margin = raw - _act_thr
                activation = jax.nn.sigmoid(_sharp * margin)
                active_margin = jnp.maximum(margin - _act_cut, 0.0)
                intensity = _eps + jnp.minimum(active_margin, _max_int)
                gate = activation * intensity
                chunk_int_max = intensity.max()
                xr = x_bf @ rc.T
                xr_f = xr.astype(jnp.float32)
                a = gate * xr_f[:, :, None, :]
                c_out = jnp.einsum('bsrn,nd->bsrd', a.astype(jnp.bfloat16), wc).astype(jnp.float32)
                chunk_weighted = gate.sum(axis=-1, keepdims=True)
                chunk_gate_sq = jnp.square(gate).sum(axis=-1, keepdims=True)
                chunk_intensity = gate.sum(axis=-1, keepdims=True)
                chunk_active = (activation > 0.5).astype(jnp.float32).sum(axis=-1, keepdims=True)
                chunk_strong = (activation > 0.9).astype(jnp.float32).sum(axis=-1, keepdims=True)
                max_gate_chunk = gate.max(axis=(0, 1, 2))
                mean_score_chunk = scores_f.mean(axis=(0, 1, 2))
                max_gate_chunk = jax.lax.pmax(
                    jax.lax.stop_gradient(max_gate_chunk), 'data')
                mean_score_chunk = jax.lax.pmean(mean_score_chunk, 'data')
                dead_mask_chunk = jax.lax.stop_gradient(
                    (max_gate_chunk < _dead_thresh).astype(jnp.float32))
                penalty_chunk = jax.nn.relu(-mean_score_chunk) * dead_mask_chunk
                chunk_dead_penalty = penalty_chunk.sum()
                chunk_dead_count = dead_mask_chunk.sum()
                return (out + c_out,
                        total_weighted_cost + chunk_weighted,
                        total_gate_sq + chunk_gate_sq,
                        jnp.maximum(total_gate_max, gate.max(axis=-1, keepdims=True)),
                        total_active + chunk_active,
                        total_strong + chunk_strong,
                        total_den_cost + chunk_intensity,
                        total_activation_cost,
                        total_current_cost,
                        total_dead_penalty + chunk_dead_penalty,
                        total_dead_count + chunk_dead_count,
                        jnp.maximum(total_int_max, chunk_int_max)), None

            (raw_out, total_weighted_cost, total_gate_sq, total_gate_max, total_active, total_strong,
             total_den_cost, total_activation_cost, total_current_cost,
             total_dead_penalty, total_dead_count,
             total_int_max), _ = jax.lax.scan(
                gate_srw_step,
                (jnp.zeros((B, S, 2, D), dtype=jnp.float32),
                 z1_r, z1_r, jnp.full((B, S, 2, 1), -1e9),
                 z1_r, z1_r, z1_r, z1_r, z1_r,
                 jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0)),
                jnp.arange(nc))

        # Normalize per route independently
        global_weighted_cost = jax.lax.psum(total_weighted_cost, 'model')   # sum(gate)
        global_gate_sq = jax.lax.psum(total_gate_sq, 'model')
        global_den_cost = jax.lax.psum(total_den_cost, 'model')
        global_activation_cost = jax.lax.psum(total_activation_cost, 'model')
        global_current_cost = jax.lax.psum(total_current_cost, 'model')
        global_gate_max = jax.lax.pmax(jax.lax.stop_gradient(total_gate_max), 'model')
        den = jnp.maximum(global_den_cost, 1.0)
        out = raw_out / den
        out = jax.lax.psum(out.astype(jnp.bfloat16), 'model')

        global_active = jax.lax.psum(total_active, 'model')
        active_frac = global_active / N_total
        active_frac_mean = active_frac.mean(axis=2)
        strong_frac = jax.lax.psum(total_strong, 'model') / N_total
        strong_frac_mean = strong_frac.mean(axis=2)
        z_mean_active = global_weighted_cost / (global_active + 1e-8)
        z_mean_active_mean = z_mean_active.mean(axis=2)
        raw_gate_max_mean = global_gate_max.mean(axis=2)

        score_std_out = s_std.mean()
        es_out = global_weighted_cost.mean()
        active_n_mean = global_active.mean()
        gate_eff_n = jax.lax.stop_gradient(
            (global_weighted_cost ** 2) / (global_gate_sq + 1e-8))
        gate_eff_ratio = jax.lax.stop_gradient(
            gate_eff_n / jnp.maximum(global_active, 1.0))
        top1_gate_frac = jax.lax.stop_gradient(
            global_gate_max / jnp.maximum(global_weighted_cost, 1e-8))
        tau_abs_mean = jax.lax.stop_gradient(tau).mean()
        dead_penalty_out = jax.lax.psum(total_dead_penalty, 'model')
        dead_count_out = jax.lax.stop_gradient(
            jax.lax.psum(total_dead_count, 'model'))
        int_max_out = jax.lax.pmax(
            jax.lax.stop_gradient(total_int_max), 'model')

        den_cost_mean = global_den_cost.mean()
        activation_cost_mean = global_activation_cost.mean()
        current_cost_mean = global_current_cost.mean()

        slim_out = (out.astype(jnp.float32), active_frac_mean, raw_gate_max_mean, score_lb,
                    score_std_out, es_out, active_n_mean, strong_frac_mean,
                    z_mean_active_mean, tau_abs_mean, dead_penalty_out, dead_count_out,
                    int_max_out, den_cost_mean, activation_cost_mean, current_cost_mean)
        conc_out = (gate_eff_n.mean(), gate_eff_ratio.mean(),
                    top1_gate_frac.mean(), top1_gate_frac.max())
        if not analysis:
            return slim_out + conc_out

        # --- Analysis-only extras ---
        phi_binary_frac = jax.lax.psum(total_phi_binary, 'model') / N_total
        phi_binary_frac_mean = phi_binary_frac.mean(axis=2)
        _active_denom = jnp.maximum(global_active, 1.0)
        z_lt_075_frac = jax.lax.stop_gradient(
            (jax.lax.psum(total_z_lt_075, 'model') / _active_denom).mean())
        z_lt_030_frac = jax.lax.stop_gradient(
            (jax.lax.psum(total_z_lt_030, 'model') / _active_denom).mean())
        active_per_token_std = jax.lax.stop_gradient(global_active).std()
        global_g_log_g = jax.lax.psum(total_g_log_g, 'model')
        gate_sum_eps = jnp.maximum(global_weighted_cost, 1e-6)
        safe_glogg = jnp.where(global_weighted_cost > 1e-6, global_g_log_g, 0.0)
        entropy_per_token = -safe_glogg / gate_sum_eps + jnp.log(gate_sum_eps)
        entropy_per_token = jnp.where(
            jnp.isfinite(entropy_per_token), entropy_per_token, 0.0)
        gate_entropy = jax.lax.stop_gradient(entropy_per_token).mean()
        den_cost_out = global_den_cost.mean()
        activation_cost_out = jax.lax.psum(total_activation_cost, 'model').mean()
        current_cost_out = jax.lax.psum(total_current_cost, 'model').mean()
        int_cap_frac_out = jax.lax.stop_gradient(
            jax.lax.psum(total_int_cap_count, 'model')
            / jnp.float32(B * S * 2 * N_total))
        return slim_out + conc_out + (phi_binary_frac_mean, z_lt_075_frac, z_lt_030_frac,
                           score_skew, active_per_token_std, gate_entropy,
                           den_cost_out, activation_cost_out, current_cost_out,
                           score_kurt, int_cap_frac_out)

    return fused_gate_srw_paired


# ================================================================
# 4. NeuronPool -- route emb + read/write operator vectors
# ================================================================

class NeuronPool(nn.Module):
    n_qk: int
    n_v: int
    d_model: int
    d_route: int
    n_rst: Optional[int] = None
    n_know: Optional[int] = None  # Legacy alias accepted from older configs.

    def setup(self):
        db = self.d_route
        dm = self.d_model
        n_rst_eff = self.n_rst if self.n_rst is not None else self.n_know
        if n_rst_eff is None:
            raise ValueError("NeuronPool requires n_rst or legacy n_know.")

        # Learned route embeddings keep norm as a routing-strength DoF.
        self.attn_qk_emb = self.param('attn_qk_emb', unit_norm_init(), (self.n_qk, db))
        self.attn_v_emb = self.param('attn_v_emb', unit_norm_init(), (self.n_v, db))
        self.rst_emb = self.param('rst_emb', unit_norm_init(), (n_rst_eff, db))

        # Read vectors define what each neuron extracts from x.
        # v4.1.5.5 uses raw read/write vector magnitudes.
        self.attn_qk_read = self.param('attn_qk_read', unit_norm_init(), (self.n_qk, dm))
        self.attn_v_read = self.param('attn_v_read', unit_norm_init(), (self.n_v, dm))
        self.rst_read = self.param('rst_read', unit_norm_init(), (n_rst_eff, dm))

        # Write vectors define the output direction for each neuron.
        # v4.1.5.5 uses raw read/write vector magnitudes.
        self.attn_qk_write = self.param('attn_qk_write', unit_norm_init(), (self.n_qk, dm))
        self.attn_v_write = self.param('attn_v_write', unit_norm_init(), (self.n_v, dm))
        self.rst_write = self.param('rst_write', unit_norm_init(), (n_rst_eff, dm))

        # Per-pool learnable output scale, initialized to sqrt(d_model).
        self.attn_qk_scale = self.param('attn_qk_scale',
            lambda k, s, d: jnp.full(s, jnp.sqrt(d)), (1,), self.d_model)
        self.attn_v_scale = self.param('attn_v_scale',
            lambda k, s, d: jnp.full(s, jnp.sqrt(d)), (1,), self.d_model)
        self.rst_scale = self.param('rst_scale',
            lambda k, s, d: jnp.full(s, jnp.sqrt(d)), (1,), self.d_model)

        # No dynamic-tau alpha params; tau comes from Router offsets + scan.


# ================================================================
# 5. Router -- route queries, tau offsets, bounded scan offset
# ================================================================

class Router(nn.Module):
    d_model: int
    d_route: int
    n_qk: int
    n_v: int
    n_rst: Optional[int] = None
    n_know: Optional[int] = None  # Legacy alias accepted from older configs.
    router_dropout: float = 0.1

    def setup(self):
        db = self.d_route
        self.proj_attn = nn.Dense(db * 3, name='proj_attn')
        self.proj_rst = nn.Dense(db, name='proj_rst')
        self.tau_attn = nn.Dense(3, name='tau_attn',
            kernel_init=nn.initializers.zeros,
            bias_init=lambda k, s, d: jnp.full(s, -0.5))
        self.tau_rst = nn.Dense(1, name='tau_rst',
            kernel_init=nn.initializers.zeros,
            bias_init=lambda k, s, d: jnp.full(s, -0.5))
        # Raw learned parameters for bounded scan offsets.
        # Zero-init preserves old behavior at step 0.
        self.raw_scan_offset_attn = nn.Dense(3, name='raw_scan_offset_attn',
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros)
        self.raw_scan_offset_rst = nn.Dense(1, name='raw_scan_offset_rst',
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros)


# ================================================================
# 6. Pure functions for scan body
# ================================================================

def _attn_forward(x, pool_params, router_params, expand_O_kernel, rng,
                  n_qk, n_v,
                  n_heads, d_model,
                  router_dropout, dropout_rate, deterministic,
                  sharded_fns, analysis=False):
    """v4.1: sharded-only. sharded_fns=(fused_single, fused_paired) required.

    `analysis=False` (train path): returns the SLIM tuple. `analysis=True`:
    returns the SLIM tuple extended with observational ANALYSIS stats
    (see scan_body below for the full unpack shape).
    """
    B, S, D = x.shape
    qk_emb = pool_params['attn_qk_emb']
    qk_read = pool_params['attn_qk_read']
    qk_write = pool_params['attn_qk_write']
    v_emb = pool_params['attn_v_emb']
    v_read = pool_params['attn_v_read']
    v_write = pool_params['attn_v_write']

    # Route embeddings are used as-is; their norm is a routing-strength DoF.
    qk_emb_unit = qk_emb
    v_emb_unit = v_emb

    # Emb-norm monitoring is observational only.
    _qk_emb_norms = jax.lax.stop_gradient(jnp.linalg.norm(qk_emb, axis=-1))
    attn_qk_emb_norm_mean = _qk_emb_norms.mean()
    attn_qk_emb_norm_min = _qk_emb_norms.min()
    attn_qk_emb_norm_std = _qk_emb_norms.std()
    _v_emb_norms = jax.lax.stop_gradient(jnp.linalg.norm(v_emb, axis=-1))
    attn_v_emb_norm_mean = _v_emb_norms.mean()
    attn_v_emb_norm_min = _v_emb_norms.min()
    attn_v_emb_norm_std = _v_emb_norms.std()
    if analysis:
        attn_qk_emb_norm_max = _qk_emb_norms.max()
        attn_v_emb_norm_max = _v_emb_norms.max()

    rng, rng_drop = jax.random.split(rng)
    h_all = x @ router_params['proj_attn']['kernel'] + router_params['proj_attn']['bias']
    h_all = safe_dropout(h_all, router_dropout, deterministic, rng_drop)
    h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)

    tau_all = x @ router_params['tau_attn']['kernel'] + router_params['tau_attn']['bias']
    raw_scan_offset_all = x @ router_params['raw_scan_offset_attn']['kernel'] + router_params['raw_scan_offset_attn']['bias']
    if analysis:
        _tau_all_sg = jax.lax.stop_gradient(tau_all)
        attn_tau_std = _tau_all_sg.std(axis=(0, 1))  # [3] Q/K/V
        attn_tau_kernel_norm = jnp.sqrt(
            jnp.sum(jax.lax.stop_gradient(router_params['tau_attn']['kernel']) ** 2) + 1e-12)

    qk_scale = pool_params['attn_qk_scale']
    v_scale = pool_params['attn_v_scale']

    if isinstance(sharded_fns, dict):
        fused_paired = sharded_fns.get('attn_qk_paired', sharded_fns.get('qk_paired', sharded_fns['paired']))
        fused_single_v = sharded_fns.get('attn_v_single', sharded_fns.get('v_single', sharded_fns['single']))
    else:
        fused_single_v, fused_paired = sharded_fns
    h_QK = jnp.stack([h_Q, h_K], axis=2)
    tau_QK = jnp.stack([tau_all[:, :, 0:1], tau_all[:, :, 1:2]], axis=2)
    raw_scan_offset_QK = jnp.stack([raw_scan_offset_all[:, :, 0:1], raw_scan_offset_all[:, :, 1:2]], axis=2)
    qk_ret = fused_paired(x, h_QK, qk_emb_unit, tau_QK, raw_scan_offset_QK,
                           qk_read, qk_write)
    (QK_out, qk_active, qk_raw_gmax, qk_lb, qk_sstd, qk_es, qk_anm,
     qk_strong, qk_z_act, qk_tau_abs,
     qk_dead_pen, qk_dead_cnt, qk_int_max,
     qk_den_cost_mean, qk_activation_cost_mean, qk_current_cost_mean) = qk_ret[:16]
    (qk_gate_eff_n, qk_gate_eff_ratio,
     qk_top1_gate_frac, qk_top1_gate_frac_max) = qk_ret[16:20]
    if analysis:
        (qk_phi_bin, qk_z075, qk_z030, qk_skew, qk_apt_std, qk_entropy,
         qk_den_cost, qk_activation_cost, qk_current_cost,
         qk_kurt, qk_int_cap) = qk_ret[20:]
        qk_raw_norm = jnp.linalg.norm(QK_out, axis=-1).mean()
    Q = QK_out[:, :, 0, :] * qk_scale
    K = QK_out[:, :, 1, :] * qk_scale
    v_ret = fused_single_v(x, h_V, v_emb_unit, tau_all[:, :, 2:3],
                           raw_scan_offset_all[:, :, 2:3], v_read, v_write)
    (V, v_active, v_raw_gmax, v_lb, v_sstd, v_es, v_anm,
     v_strong, v_z_act, v_tau_abs,
     v_dead_pen, v_dead_cnt, v_int_max,
     v_den_cost_mean, v_activation_cost_mean, v_current_cost_mean) = v_ret[:16]
    (v_gate_eff_n, v_gate_eff_ratio,
     v_top1_gate_frac, v_top1_gate_frac_max) = v_ret[16:20]
    if analysis:
        (v_phi_bin, v_z075, v_z030, v_skew, v_apt_std, v_entropy,
         v_den_cost, v_activation_cost, v_current_cost,
         v_kurt, v_int_cap) = v_ret[20:]
        v_raw_norm = jnp.linalg.norm(V, axis=-1).mean()
    V = V * v_scale

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

    if analysis:
        q_norm = jnp.linalg.norm(Q, axis=-1).mean()
        k_norm = jnp.linalg.norm(K, axis=-1).mean()
        v_norm_dbg = jnp.linalg.norm(V, axis=-1).mean()
        attn_logit_max = (q_norm * k_norm / scale)

    out = _attn_scores(Q, K, V, rng_attn_drop)
    if analysis:
        o_input_norm = jnp.linalg.norm(out, axis=-1).mean()
    out = out.transpose(0, 2, 1, 3).reshape(B, S, D)
    out = out @ expand_O_kernel
    attn_out_norm = jnp.linalg.norm(out, axis=-1).mean()
    rng, rng_out = jax.random.split(rng)
    out = safe_dropout(out, dropout_rate, deterministic, rng_out)

    # Load-balance loss from gate distributions + tau regularization.
    tau_reg = jnp.maximum(tau_all, 0.0).mean() * 0.01
    # Q/K share the qk pool, while V has its own pool.  Keep the historical
    # /3 scaling so the aux magnitude stays comparable to older runs.
    aux = (qk_lb + v_lb) / 3.0 + tau_reg
    attn_raw_gmax = jnp.maximum(qk_raw_gmax.mean(), v_raw_gmax.mean())
    attn_score_std = (qk_sstd + v_sstd) / 2
    attn_gate_sum = (qk_es + v_es) / 2
    attn_active_n_mean = (qk_anm + v_anm) / 2
    attn_tau_mean = tau_all.mean()
    attn_strong = (qk_strong.mean() + v_strong.mean()) / 2
    attn_qk_z_mean_active = qk_z_act.mean()
    attn_v_z_mean_active = v_z_act.mean()
    attn_tau_abs_mean = (qk_tau_abs + v_tau_abs) / 2
    attn_dead_penalty = qk_dead_pen + v_dead_pen
    attn_dead_count = jax.lax.stop_gradient(qk_dead_cnt + v_dead_cnt)
    attn_int_max = jnp.maximum(qk_int_max, v_int_max)
    attn_den_cost_mean = (qk_den_cost_mean + v_den_cost_mean) / 2
    attn_activation_cost_mean = (qk_activation_cost_mean + v_activation_cost_mean) / 2
    attn_current_cost_mean = (qk_current_cost_mean + v_current_cost_mean) / 2
    attn_gate_eff_n = (qk_gate_eff_n + v_gate_eff_n) / 2
    attn_gate_eff_ratio = (qk_gate_eff_ratio + v_gate_eff_ratio) / 2
    attn_top1_gate_frac = (qk_top1_gate_frac + v_top1_gate_frac) / 2
    attn_top1_gate_frac_max = jnp.maximum(qk_top1_gate_frac_max,
                                          v_top1_gate_frac_max)
    # Exploration loss consumes tau offsets per layer: [B, S, 3].
    attn_tau_offset = tau_all
    slim_ret = (out, aux, qk_active.mean(), v_active.mean(), attn_raw_gmax,
                attn_score_std, attn_gate_sum, attn_active_n_mean,
                attn_out_norm, attn_tau_mean,
                attn_strong,
                qk_strong.mean(), v_strong.mean(),
                attn_qk_z_mean_active, attn_v_z_mean_active,
                attn_tau_abs_mean,
                attn_qk_emb_norm_mean, attn_v_emb_norm_mean,
                attn_qk_emb_norm_min, attn_qk_emb_norm_std,
                attn_v_emb_norm_min, attn_v_emb_norm_std,
                attn_dead_penalty, attn_dead_count,
                attn_tau_offset,
                attn_int_max,
                attn_den_cost_mean, attn_activation_cost_mean,
                attn_current_cost_mean,
                attn_gate_eff_n, attn_gate_eff_ratio,
                attn_top1_gate_frac, attn_top1_gate_frac_max)
    if not analysis:
        return slim_ret

    attn_qk_phi_binary = qk_phi_bin.mean()
    attn_v_phi_binary = v_phi_bin.mean()
    attn_z_lt_075_frac = (qk_z075 + v_z075) / 2
    attn_z_lt_030_frac = (qk_z030 + v_z030) / 2
    attn_score_skew = (qk_skew + v_skew) / 2
    attn_active_per_token_std = (qk_apt_std + v_apt_std) / 2
    attn_gate_entropy = (qk_entropy + v_entropy) / 2
    attn_den_cost = (qk_den_cost + v_den_cost) / 2
    attn_activation_cost = (qk_activation_cost + v_activation_cost) / 2
    attn_current_cost = (qk_current_cost + v_current_cost) / 2
    attn_score_kurt = (qk_kurt + v_kurt) / 2
    attn_int_cap_frac = (qk_int_cap + v_int_cap) / 2.0
    return slim_ret + (
        qk_raw_norm, v_raw_norm,
        q_norm, k_norm, v_norm_dbg, attn_logit_max, o_input_norm,
        attn_qk_phi_binary, attn_v_phi_binary,
        attn_tau_std, attn_tau_kernel_norm,
        attn_z_lt_075_frac, attn_z_lt_030_frac,
        attn_score_skew, attn_active_per_token_std, attn_gate_entropy,
        attn_den_cost,
        attn_activation_cost, attn_current_cost,
        attn_qk_emb_norm_max, attn_v_emb_norm_max,
        attn_score_kurt,
        attn_int_cap_frac,
    )


def _rst_forward(x, pool_params, router_params, rng,
                  router_dropout, dropout_rate, deterministic,
                  sharded_fns, analysis=False):
    """v4.1: sharded-only. sharded_fns=(fused_single, fused_paired) required.

    `analysis` see _attn_forward docstring.
    """
    rst_emb = pool_params['rst_emb']
    rst_read = pool_params['rst_read']
    rst_write = pool_params['rst_write']

    rng, rng_drop = jax.random.split(rng)
    h = x @ router_params['proj_rst']['kernel'] + router_params['proj_rst']['bias']
    h = safe_dropout(h, router_dropout, deterministic, rng_drop)

    # Route embeddings are used as-is; their norm is a routing-strength DoF.
    rst_emb_unit = rst_emb
    tau = x @ router_params['tau_rst']['kernel'] + router_params['tau_rst']['bias']
    raw_scan_offset = x @ router_params['raw_scan_offset_rst']['kernel'] + router_params['raw_scan_offset_rst']['bias']
    if analysis:
        rst_tau_std = jax.lax.stop_gradient(tau).std()
        rst_tau_kernel_norm = jnp.sqrt(
            jnp.sum(jax.lax.stop_gradient(router_params['tau_rst']['kernel']) ** 2) + 1e-12)

    rst_scale = pool_params['rst_scale']

    if isinstance(sharded_fns, dict):
        fused_single = sharded_fns.get('rst_single', sharded_fns['single'])
    else:
        fused_single, _ = sharded_fns
    rst_ret = fused_single(x, h, rst_emb_unit, tau, raw_scan_offset,
                            rst_read, rst_write)
    (out, active_frac, raw_gate_max, lb_loss, score_std, gate_sum, active_n_mean,
     strong_frac, z_mean_act, rst_tau_abs_mean,
     rst_dead_penalty, rst_dead_count, rst_int_max,
     rst_den_cost_mean, rst_activation_cost_mean, rst_current_cost_mean) = rst_ret[:16]
    (rst_gate_eff_n, rst_gate_eff_ratio,
     rst_top1_gate_frac, rst_top1_gate_frac_max) = rst_ret[16:20]
    if analysis:
        (phi_binary_frac, rst_z_lt_075_frac, rst_z_lt_030_frac,
         rst_score_skew, rst_active_per_token_std, rst_gate_entropy,
         rst_den_cost, rst_activation_cost, rst_current_cost,
         rst_score_kurt, rst_int_cap_frac) = rst_ret[20:]
        rst_raw_out_norm = jnp.linalg.norm(out, axis=-1).mean()
    out = out * rst_scale
    rst_out_norm = jnp.linalg.norm(out, axis=-1).mean()
    rng, rng_out = jax.random.split(rng)
    out = safe_dropout(out, dropout_rate, deterministic, rng_out)

    tau_reg = jnp.maximum(tau, 0.0).mean() * 0.01
    aux = lb_loss + tau_reg
    _rst_emb_norms = jax.lax.stop_gradient(jnp.linalg.norm(rst_emb, axis=-1))
    emb_norm_val = _rst_emb_norms.mean()
    rst_emb_norm_min = _rst_emb_norms.min()
    rst_emb_norm_std = _rst_emb_norms.std()
    if analysis:
        rst_emb_norm_max = _rst_emb_norms.max()
    read_norm_val = jnp.linalg.norm(rst_read, axis=-1).mean()
    write_norm_val = jnp.linalg.norm(rst_write, axis=-1).mean()
    rst_tau_mean = tau.mean()
    rst_strong = strong_frac.mean()
    rst_z_mean_active = z_mean_act.mean()
    slim_ret = (out, aux, active_frac, raw_gate_max, score_std, gate_sum, active_n_mean,
                emb_norm_val, read_norm_val, write_norm_val, rst_out_norm,
                rst_tau_mean, rst_strong, rst_z_mean_active,
                rst_tau_abs_mean,
                rst_emb_norm_min, rst_emb_norm_std,
                rst_dead_penalty, rst_dead_count,
                tau,
                rst_int_max,
                rst_den_cost_mean, rst_activation_cost_mean,
                rst_current_cost_mean,
                rst_gate_eff_n, rst_gate_eff_ratio,
                rst_top1_gate_frac, rst_top1_gate_frac_max)
    if not analysis:
        return slim_ret

    rst_phi_binary = phi_binary_frac.mean()
    return slim_ret + (
        rst_raw_out_norm,
        rst_tau_std, rst_tau_kernel_norm,
        rst_z_lt_075_frac, rst_z_lt_030_frac,
        rst_score_skew, rst_active_per_token_std, rst_gate_entropy,
        rst_den_cost,
        rst_activation_cost, rst_current_cost,
        rst_emb_norm_max,
        rst_score_kurt,
        rst_phi_binary,
        rst_int_cap_frac,
    )


# ================================================================
# 7. Flax modules (init path only)
# ================================================================

class AttentionLayer(nn.Module):
    """Attention Layer container.

    The Attention Layer performs model decisions over the attention-qk and
    attention-v pools to construct Q/K/V, then applies causal self-attention
    for relational state interaction. The real forward path is _attn_forward().
    """
    d_model: int
    n_heads: int
    dropout_rate: float = 0.1

    def setup(self):
        self.expand_O = nn.Dense(
            self.d_model, use_bias=False, kernel_init=scaled_normal(0.02))


class DAWNBlock(nn.Module):
    """DAWN-SRW Block = Attention Layer + RST Layer.

    The RST Layer is the concrete post-attention layer that selects and
    composes RW operators to refine the residual state and transition it to the
    next representation state.

    Container for per-layer norms + attn (expand_O) submodules.
    The real forward path is scan_body in DAWN.__call__."""
    d_model: int
    n_heads: int
    dropout_rate: float = 0.1

    def setup(self):
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.attn = AttentionLayer(
            d_model=self.d_model, n_heads=self.n_heads,
            dropout_rate=self.dropout_rate)




# ================================================================
# 8. DAWN Model
# ================================================================

class DAWN(nn.Module):
    """DAWN-SRW v4.1.5.5 with Attention Layers and RST Layers."""
    __version__ = "dawn_srw"

    vocab_size: int = 30000
    d_model: int = 384
    n_layers: int = 12
    n_heads: int = 6
    max_seq_len: int = 512
    dropout_rate: float = 0.1
    gradient_checkpointing: bool = False

    d_route: int = DEFAULT_D_ROUTE
    n_qk: int = 1580
    n_v: int = 2600
    n_rst: Optional[int] = None
    n_know: Optional[int] = None  # Legacy config alias; prefer n_rst in new configs.
    router_dropout: float = 0.1
    n_chunks_rst: Optional[int] = None
    n_chunks_know: int = 1    # Legacy config alias; prefer n_chunks_rst.
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
        n_rst_eff = self.n_rst if self.n_rst is not None else (
            self.n_know if self.n_know is not None else 25200)
        self.neuron_pool = NeuronPool(
            n_qk=self.n_qk, n_v=self.n_v, n_rst=n_rst_eff,
            d_model=self.d_model, d_route=self.d_route)
        self.router = Router(
            d_model=self.d_model, d_route=self.d_route,
            n_qk=self.n_qk, n_v=self.n_v, n_rst=n_rst_eff,
            router_dropout=self.router_dropout)
        self.layers = [
            DAWNBlock(d_model=self.d_model, n_heads=self.n_heads,
                      dropout_rate=self.dropout_rate, name=f'block_{i}')
            for i in range(self.n_layers)]
        self.norm = nn.LayerNorm()

    def __call__(self, input_ids, labels=None, attention_mask=None,
                 deterministic=False, sharded_fns=None, analysis=False):
        """Run the shared-pool SRW Transformer forward pass.

        analysis=False is the train/eval path and returns only regular
        training metrics.  analysis=True enables extra observational stats
        such as distribution shape, boundary fraction, entropy, tau stats,
        raw norms, and debug norms.
        """
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
            rst_auxes = _z
            rst_active_all = _z
            rst_raw_gmax_all = _z
            rst_sstd_all = _z
            rst_gsum_all = _z
            rst_active_n_mean_all = _z
            rst_strong_all = _z
            attn_qk_active_all = _z
            attn_v_active_all = _z
            attn_raw_gmax_all = _z
            attn_sstd_all = _z
            attn_gsum_all = _z
            attn_active_n_mean_all = _z
            attn_strong_all = _z
            attn_qk_strong_all = _z
            attn_v_strong_all = _z
            rst_z_act_all = _z
            attn_qk_z_act_all = _z
            attn_v_z_act_all = _z
            k_emb_n_all = _z
            k_read_n_all = _z
            k_write_n_all = _z
            rst_out_norm_all = _z
            attn_out_norm_all = _z
            attn_tau_mean_all = _z
            rst_tau_mean_all = _z
            attn_tau_abs_all = _z
            rst_tau_abs_all = _z
            attn_qk_emb_n_mean_all = _z
            attn_v_emb_n_mean_all = _z
            rst_emb_n_std_all = _z
            attn_qk_emb_n_min_all = _z
            attn_qk_emb_n_std_all = _z
            attn_v_emb_n_min_all = _z
            attn_v_emb_n_std_all = _z
            rst_emb_n_min_all = _z
            attn_dead_penalty_all = _z
            rst_dead_penalty_all = _z
            attn_dead_count_all = _z
            rst_dead_count_all = _z
            attn_tau_offset_all = _z
            rst_tau_offset_all = _z
            attn_int_max_all = _z
            rst_int_max_all = _z
            attn_den_cost_mean_all = _z
            rst_den_cost_mean_all = _z
            attn_activation_cost_mean_all = _z
            rst_activation_cost_mean_all = _z
            attn_current_cost_mean_all = _z
            rst_current_cost_mean_all = _z
            attn_gate_eff_n_all = _z
            attn_gate_eff_ratio_all = _z
            attn_top1_gate_frac_all = _z
            attn_top1_gate_frac_max_all = _z
            rst_gate_eff_n_all = _z
            rst_gate_eff_ratio_all = _z
            rst_top1_gate_frac_all = _z
            rst_top1_gate_frac_max_all = _z
            # Trigger Flax param realization for all submodules (init-only).
            # The real forward runs through scan_body in the else branch and
            # accesses params by path, not via these module calls.
            _ = self.neuron_pool.attn_qk_emb  # triggers NeuronPool.setup
            _ = self.router.proj_attn(x)
            _ = self.router.proj_rst(x)
            _ = self.router.tau_attn(x)
            _ = self.router.tau_rst(x)
            _ = self.router.raw_scan_offset_attn(x)
            _ = self.router.raw_scan_offset_rst(x)
            for layer in self.layers:
                _ = layer.norm1(x)
                _ = layer.norm2(x)
                _ = layer.attn.expand_O(x)
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
                rng, rng_attn, rng_rst = jax.random.split(rng, 3)

                normed = _layer_norm(
                    x, bp['norm1']['scale'], bp['norm1']['bias'])
                attn_ret = _attn_forward(
                    normed, pool_params, router_params,
                    bp['attn']['expand_O']['kernel'], rng_attn,
                    self.n_qk, self.n_v,
                    self.n_heads, self.d_model,
                    self.router_dropout, self.dropout_rate, deterministic,
                    sharded_fns=_sharded, analysis=analysis)
                (attn_out, attn_aux, a_qk_active, a_v_active, a_raw_gmax,
                 a_sstd, a_gsum, a_active_n_mean,
                 a_out_norm, a_tau_mean, a_strong,
                 a_qk_strong, a_v_strong,
                 a_qk_z_act, a_v_z_act,
                 a_tau_abs,
                 a_qk_emb_n_mean, a_v_emb_n_mean,
                 a_qk_emb_n_min, a_qk_emb_n_std,
                 a_v_emb_n_min, a_v_emb_n_std,
                 a_dead_penalty, a_dead_count,
                 a_tau_offset,
                 a_int_max,
                 a_den_cost_mean, a_activation_cost_mean,
                 a_current_cost_mean,
                 a_gate_eff_n, a_gate_eff_ratio,
                 a_top1_gate_frac, a_top1_gate_frac_max) = attn_ret[:33]
                if analysis:
                    (a_qk_raw_norm, a_v_raw_norm,
                     a_q_norm, a_k_norm, a_v_norm_dbg, a_logit_max, a_o_input_norm,
                     a_qk_phi_bin, a_v_phi_bin,
                     a_tau_std, a_tau_kernel_norm,
                     a_z075, a_z030,
                     a_skew, a_apt_std, a_entropy,
                     a_den_cost, a_activation_cost, a_current_cost,
                     a_qk_emb_n_max, a_v_emb_n_max,
                     a_score_kurt, a_int_cap_frac) = attn_ret[33:]
                x = x + attn_out

                normed = _layer_norm(
                    x, bp['norm2']['scale'], bp['norm2']['bias'])
                rst_ret = _rst_forward(
                    normed, pool_params, router_params, rng_rst,
                    self.router_dropout, self.dropout_rate, deterministic,
                    sharded_fns=_sharded, analysis=analysis)
                (rst_out, rst_aux, k_active, k_raw_gmax, k_sstd, k_gsum,
                 k_active_n_mean, k_emb_n, k_read_n, k_write_n, k_out_norm,
                 k_tau_mean, k_strong, k_z_act, k_tau_abs,
                 k_emb_n_min, k_emb_n_std,
                 k_dead_penalty, k_dead_count,
                 k_tau_offset,
                 k_int_max,
                 k_den_cost_mean, k_activation_cost_mean,
                 k_current_cost_mean,
                 k_gate_eff_n, k_gate_eff_ratio,
                 k_top1_gate_frac, k_top1_gate_frac_max) = rst_ret[:28]
                if analysis:
                    (k_raw_out_norm,
                     k_tau_std, k_tau_kernel_norm,
                     k_z075, k_z030,
                     k_skew, k_apt_std, k_entropy,
                     k_den_cost, k_activation_cost, k_current_cost,
                     k_emb_n_max, k_score_kurt, k_phi_bin,
                     k_int_cap_frac) = rst_ret[28:]
                x = x + rst_out

                slim_ys = (attn_aux, rst_aux,
                           k_active, k_raw_gmax, k_sstd, k_gsum, k_active_n_mean,
                           a_qk_active, a_v_active, a_raw_gmax, a_sstd, a_gsum, a_active_n_mean,
                           k_emb_n, k_read_n, k_write_n,
                           k_out_norm,
                           a_out_norm, a_tau_mean, k_tau_mean,
                           k_strong, a_strong,
                           a_qk_strong, a_v_strong,
                           k_z_act, a_qk_z_act, a_v_z_act,
                           a_tau_abs, k_tau_abs,
                           a_qk_emb_n_mean, a_v_emb_n_mean,
                           k_emb_n_std,
                           a_qk_emb_n_min, a_qk_emb_n_std,
                           a_v_emb_n_min, a_v_emb_n_std,
                           k_emb_n_min,
                           a_dead_penalty, k_dead_penalty,
                           a_dead_count, k_dead_count,
                           a_tau_offset, k_tau_offset,
                           a_int_max, k_int_max,
                           a_den_cost_mean, k_den_cost_mean,
                           a_activation_cost_mean, k_activation_cost_mean,
                           a_current_cost_mean, k_current_cost_mean,
                           a_gate_eff_n, a_gate_eff_ratio,
                           a_top1_gate_frac, a_top1_gate_frac_max,
                           k_gate_eff_n, k_gate_eff_ratio,
                           k_top1_gate_frac, k_top1_gate_frac_max,
                           )
                if not analysis:
                    return x, slim_ys
                return x, slim_ys + (
                    a_qk_raw_norm, a_v_raw_norm, k_raw_out_norm,
                    a_q_norm, a_k_norm, a_v_norm_dbg, a_logit_max, a_o_input_norm,
                    k_phi_bin, a_qk_phi_bin, a_v_phi_bin,
                    a_tau_std, k_tau_std,
                    a_tau_kernel_norm, k_tau_kernel_norm,
                    a_z075, k_z075,
                    a_z030, k_z030,
                    a_skew, k_skew,
                    a_apt_std, k_apt_std,
                    a_entropy, k_entropy,
                    a_den_cost, k_den_cost,
                    a_activation_cost, k_activation_cost,
                    a_current_cost, k_current_cost,
                    a_qk_emb_n_max, a_v_emb_n_max,
                    k_emb_n_max,
                    a_score_kurt, k_score_kurt,
                    a_int_cap_frac, k_int_cap_frac,
                )

            if self.gradient_checkpointing:
                scan_body = jax.checkpoint(scan_body)

            xs = {'params': stacked, 'rng': layer_rngs}
            x, scan_ys = jax.lax.scan(scan_body, x, xs)

            (attn_auxes, rst_auxes,
             rst_active_all, rst_raw_gmax_all, rst_sstd_all, rst_gsum_all, rst_active_n_mean_all,
             attn_qk_active_all, attn_v_active_all, attn_raw_gmax_all, attn_sstd_all, attn_gsum_all, attn_active_n_mean_all,
             k_emb_n_all, k_read_n_all, k_write_n_all,
             rst_out_norm_all,
             attn_out_norm_all, attn_tau_mean_all, rst_tau_mean_all,
             rst_strong_all, attn_strong_all,
             attn_qk_strong_all, attn_v_strong_all,
             rst_z_act_all, attn_qk_z_act_all, attn_v_z_act_all,
             attn_tau_abs_all, rst_tau_abs_all,
             attn_qk_emb_n_mean_all, attn_v_emb_n_mean_all,
             rst_emb_n_std_all,
             attn_qk_emb_n_min_all, attn_qk_emb_n_std_all,
             attn_v_emb_n_min_all, attn_v_emb_n_std_all,
             rst_emb_n_min_all,
            attn_dead_penalty_all, rst_dead_penalty_all,
            attn_dead_count_all, rst_dead_count_all,
            attn_tau_offset_all, rst_tau_offset_all,
            attn_int_max_all, rst_int_max_all,
            attn_den_cost_mean_all, rst_den_cost_mean_all,
            attn_activation_cost_mean_all, rst_activation_cost_mean_all,
            attn_current_cost_mean_all, rst_current_cost_mean_all,
            attn_gate_eff_n_all, attn_gate_eff_ratio_all,
            attn_top1_gate_frac_all, attn_top1_gate_frac_max_all,
            rst_gate_eff_n_all, rst_gate_eff_ratio_all,
            rst_top1_gate_frac_all, rst_top1_gate_frac_max_all) = scan_ys[:59]
            if analysis:
                (attn_qk_raw_norm_all, attn_v_raw_norm_all, rst_raw_out_norm_all,
                 attn_q_norm_all, attn_k_norm_all, attn_v_norm_dbg_all,
                 attn_logit_max_all, attn_o_input_norm_all,
                 rst_phi_bin_all, attn_qk_phi_bin_all, attn_v_phi_bin_all,
                 attn_tau_std_all, rst_tau_std_all,
                 attn_tau_kernel_norm_all, rst_tau_kernel_norm_all,
                 attn_z075_all, rst_z075_all,
                 attn_z030_all, rst_z030_all,
                 attn_skew_all, rst_skew_all,
                 attn_apt_std_all, rst_apt_std_all,
                 attn_entropy_all, rst_entropy_all,
                 attn_den_cost_all, rst_den_cost_all,
                 attn_activation_cost_all, rst_activation_cost_all,
                 attn_current_cost_all, rst_current_cost_all,
                 attn_qk_emb_n_max_all, attn_v_emb_n_max_all,
                 rst_emb_n_max_all,
                 attn_score_kurt_all, rst_score_kurt_all,
                 attn_int_cap_frac_all, rst_int_cap_frac_all) = scan_ys[59:]
            # Aux is averaged over layers after attention and RST terms are
            # collected.  Attention keeps historical Q/K/V scaling upstream.
            total_aux = (attn_auxes + rst_auxes).mean()

        x = self.norm(x)
        result = {
            'aux_loss': total_aux,
            'attn_aux': attn_auxes.mean(),
            'rst_aux': rst_auxes.mean(),

            'rst_active': rst_active_all.mean(),
            'rst_raw_gate_max': rst_raw_gmax_all.mean(),
            'rst_score_std': rst_sstd_all.mean(),
            'rst_gate_sum': rst_gsum_all.mean(),
            'rst_active_n_mean': rst_active_n_mean_all.mean(),
            'rst_strong': rst_strong_all.mean(),
            'rst_z_mean_active': rst_z_act_all.mean(),

            'attn_qk_active': attn_qk_active_all.mean(),
            'attn_v_active': attn_v_active_all.mean(),
            'attn_raw_gate_max': attn_raw_gmax_all.mean(),
            'attn_score_std': attn_sstd_all.mean(),
            'attn_gate_sum': attn_gsum_all.mean(),
            'attn_active_n_mean': attn_active_n_mean_all.mean(),
            'attn_strong': attn_strong_all.mean(),
            'attn_qk_strong': attn_qk_strong_all.mean(),
            'attn_v_strong': attn_v_strong_all.mean(),
            'attn_qk_z_mean_active': attn_qk_z_act_all.mean(),
            'attn_v_z_mean_active': attn_v_z_act_all.mean(),

            'rst_emb_norm': k_emb_n_all.mean(),
            'rst_read_norm': k_read_n_all.mean(),
            'rst_write_norm': k_write_n_all.mean(),

            'rst_out_norm': rst_out_norm_all.mean(),
            'attn_out_norm': attn_out_norm_all.mean(),
            'attn_tau_mean': attn_tau_mean_all.mean(),
            'rst_tau_mean': rst_tau_mean_all.mean(),
            'attn_tau_abs_mean': attn_tau_abs_all.mean(),
            'rst_tau_abs_mean': rst_tau_abs_all.mean(),
            'attn_qk_emb_norm_mean': attn_qk_emb_n_mean_all.mean(),
            'attn_qk_emb_norm_min': attn_qk_emb_n_min_all.min(),
            'attn_qk_emb_norm_std': attn_qk_emb_n_std_all.mean(),
            'attn_v_emb_norm_mean': attn_v_emb_n_mean_all.mean(),
            'attn_v_emb_norm_min': attn_v_emb_n_min_all.min(),
            'attn_v_emb_norm_std': attn_v_emb_n_std_all.mean(),
            'rst_emb_norm_min': rst_emb_n_min_all.min(),
            'rst_emb_norm_std': rst_emb_n_std_all.mean(),

            # Dead-only penalty is separate from aux and weighted in train_jax.
            # Mean across layers so the training weight is layer-count-agnostic.
            'attn_dead_penalty': attn_dead_penalty_all.mean(),
            'rst_dead_penalty': rst_dead_penalty_all.mean(),
            'dead_penalty': (attn_dead_penalty_all.mean()
                             + rst_dead_penalty_all.mean()),
            'attn_dead_count': attn_dead_count_all.mean(),
            'rst_dead_count': rst_dead_count_all.mean(),

            'per_layer_attn_out_norm': attn_out_norm_all,
            'per_layer_rst_out_norm': rst_out_norm_all,
            # Per-layer tau offset stacks for exploration loss.
            # Shapes: attn [L, B, S, 3], RST [L, B, S, 1].
            'attn_tau_offset': attn_tau_offset_all,
            'rst_tau_offset': rst_tau_offset_all,
            # Denominator diagnostic: sum(activation * intensity).
            'attn_int_max': attn_int_max_all.max(),
            'rst_int_max': rst_int_max_all.max(),
            'attn_gate_den_sum_mean': attn_den_cost_mean_all.mean(),
            'rst_gate_den_sum_mean': rst_den_cost_mean_all.mean(),
            'attn_gate_eff_n': attn_gate_eff_n_all.mean(),
            'attn_gate_eff_ratio': attn_gate_eff_ratio_all.mean(),
            'attn_top1_gate_frac': attn_top1_gate_frac_all.mean(),
            'attn_top1_gate_frac_max': attn_top1_gate_frac_max_all.max(),
            'rst_gate_eff_n': rst_gate_eff_n_all.mean(),
            'rst_gate_eff_ratio': rst_gate_eff_ratio_all.mean(),
            'rst_top1_gate_frac': rst_top1_gate_frac_all.mean(),
            'rst_top1_gate_frac_max': rst_top1_gate_frac_max_all.max(),
        }
        if analysis and not self.is_initializing():
            _residual_norm = jnp.linalg.norm(x, axis=-1).mean()
            _emb_norm = jnp.linalg.norm(self.token_emb.embedding, axis=-1).mean()
            _o_proj_norm = jnp.linalg.norm(
                stacked['attn']['expand_O']['kernel'], axis=(-2, -1)).mean()
            result.update({
                'rst_phi_binary': rst_phi_bin_all.mean(),
                'attn_qk_phi_binary': attn_qk_phi_bin_all.mean(),
                'attn_v_phi_binary': attn_v_phi_bin_all.mean(),
                'attn_tau_std': attn_tau_std_all.mean(axis=0),
                'rst_tau_std': rst_tau_std_all.mean(),
                'attn_tau_kernel_norm': attn_tau_kernel_norm_all.mean(),
                'rst_tau_kernel_norm': rst_tau_kernel_norm_all.mean(),
                'attn_z_lt_075': attn_z075_all.mean(),
                'rst_z_lt_075': rst_z075_all.mean(),
                'attn_z_lt_030': attn_z030_all.mean(),
                'rst_z_lt_030': rst_z030_all.mean(),
                'attn_score_skew': attn_skew_all.mean(),
                'rst_score_skew': rst_skew_all.mean(),
                'attn_active_per_token_std': attn_apt_std_all.mean(),
                'rst_active_per_token_std': rst_apt_std_all.mean(),
                'attn_gate_entropy': attn_entropy_all.mean(),
                'rst_gate_entropy': rst_entropy_all.mean(),
                'attn_gate_den_sum': attn_den_cost_all.mean(),
                'rst_gate_den_sum': rst_den_cost_all.mean(),
                'attn_qk_emb_norm_max': attn_qk_emb_n_max_all.max(),
                'attn_v_emb_norm_max': attn_v_emb_n_max_all.max(),
                'rst_emb_norm_max': rst_emb_n_max_all.max(),
                'attn_score_kurt': attn_score_kurt_all.mean(),
                'rst_score_kurt': rst_score_kurt_all.mean(),
                'attn_qk_raw_norm': attn_qk_raw_norm_all.mean(),
                'attn_v_raw_norm': attn_v_raw_norm_all.mean(),
                'rst_raw_out_norm': rst_raw_out_norm_all.mean(),
                'debug_residual_norm': _residual_norm,
                'debug_emb_norm': _emb_norm,
                'debug_o_proj_norm': _o_proj_norm,
                'debug_q_norm': attn_q_norm_all.mean(),
                'debug_k_norm': attn_k_norm_all.mean(),
                'debug_v_norm': attn_v_norm_dbg_all.mean(),
                'debug_logit_max': attn_logit_max_all.mean(),
                'debug_o_input_norm': attn_o_input_norm_all.mean(),
                'attn_int_cap_frac': attn_int_cap_frac_all.mean(),
                'rst_int_cap_frac': rst_int_cap_frac_all.mean(),
            })


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
                per_token_ce = tl * vmask            # [B, S-1], 0 on invalid
                loss = per_token_ce.sum() / (vmask.sum() + 1e-8)
                preds = jnp.argmax(logits, axis=-1)
                correct = jnp.sum((preds == labs) & vmask)
                return loss, per_token_ce, correct, jnp.sum(vmask)

            loss, per_token_ce, correct, valid_count = compute_loss_and_acc(
                shift_x, embedding_matrix, shift_labels, valid_mask)
            result['loss'] = loss
            result['correct'] = correct
            result['valid_count'] = valid_count
            # v4.1 explore: expose per-token CE + valid mask for RPE loss.
            result['per_token_ce'] = per_token_ce
            result['valid_mask'] = valid_mask
        else:
            result['logits'] = self.token_emb.attend(x)

        return result

    def get_config(self):
        n_rst_eff = self.n_rst if self.n_rst is not None else (
            self.n_know if self.n_know is not None else 25200)
        return {
            'model_version': self.__version__,
            'vocab_size': self.vocab_size, 'd_model': self.d_model,
            'n_layers': self.n_layers, 'n_heads': self.n_heads,
            'max_seq_len': self.max_seq_len,
            'd_route': self.d_route,
            'n_qk': self.n_qk, 'n_v': self.n_v, 'n_rst': n_rst_eff,
            'n_know': n_rst_eff,
        }

    def get_model_info(self):
        n_rst_eff = self.n_rst if self.n_rst is not None else (
            self.n_know if self.n_know is not None else 25200)
        return [
            f"DAWN-SRW ({self.__version__})",
            f"  d_model={self.d_model}, d_route={self.d_route}, "
            f"n_layers={self.n_layers}, n_heads={self.n_heads}",
            f"  Attention-QK: {self.n_qk}, Attention-V: {self.n_v}, RST: {n_rst_eff}",
            f"  Route: learned d_route embedding [{self.d_route}]",
        ]


# ================================================================
# 9. Inference API: KV-cache prefill + decode.
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


def _srw_inference(x, h, emb, tau_offset, raw_scan_offset, w_read, w_write):
    """Non-chunked SRW for inference."""
    # v4.1.5.5: raw read/write vectors with natural vector magnitudes.
    r_n = w_read.astype(jnp.float32)
    w_n = w_write.astype(jnp.float32)
    scores = h @ emb.T
    scores_f32 = scores.astype(jnp.float32)
    s_mean = scores_f32.mean(axis=-1, keepdims=True)
    s_std = jnp.sqrt(jnp.mean(jnp.square(scores_f32 - s_mean),
                               axis=-1, keepdims=True)) + 1e-8
    scan_offset = SCAN_SCALE * jnp.tanh(raw_scan_offset)
    tau = s_mean + tau_offset * s_std - scan_offset / jnp.maximum(s_std, SCAN_STD_FLOOR)

    raw = scores_f32 - tau
    margin = raw - ACTIVATION_THRESHOLD
    activation = jax.nn.sigmoid(SHARPNESS * margin)
    active_margin = jnp.maximum(margin - ACTIVATION_CUTOFF, 0.0)
    intensity = EPSILON + jnp.minimum(active_margin, MAX_INTENSITY)
    gate = activation * intensity

    xr = x.astype(jnp.float32) @ r_n.T
    a = gate * xr
    raw_out = a @ w_n
    den = jnp.maximum(gate.sum(axis=-1, keepdims=True), 1.0)
    out = raw_out.astype(jnp.float32) / den
    return out.astype(jnp.float32)


def _srw_inference_with_gates(x, h, emb, tau_offset, raw_scan_offset, w_read, w_write):
    """Like _srw_inference but also returns raw and normalized gate for analysis."""
    # v4.1.5.5: raw read/write vectors with natural vector magnitudes.
    r_n = w_read.astype(jnp.float32)
    w_n = w_write.astype(jnp.float32)
    scores = h @ emb.T
    scores_f32 = scores.astype(jnp.float32)
    s_mean = scores_f32.mean(axis=-1, keepdims=True)
    s_std = jnp.sqrt(jnp.mean(jnp.square(scores_f32 - s_mean),
                               axis=-1, keepdims=True)) + 1e-8
    scan_offset = SCAN_SCALE * jnp.tanh(raw_scan_offset)
    tau = s_mean + tau_offset * s_std - scan_offset / jnp.maximum(s_std, SCAN_STD_FLOOR)

    raw = scores_f32 - tau
    margin = raw - ACTIVATION_THRESHOLD
    activation = jax.nn.sigmoid(SHARPNESS * margin)
    active_margin = jnp.maximum(margin - ACTIVATION_CUTOFF, 0.0)
    intensity = EPSILON + jnp.minimum(active_margin, MAX_INTENSITY)
    gate = activation * intensity
    gate_norm = gate / jnp.maximum(gate.sum(axis=-1, keepdims=True), 1e-8)

    xr = x.astype(jnp.float32) @ r_n.T
    a = gate * xr
    raw_out = a @ w_n
    den = jnp.maximum(gate.sum(axis=-1, keepdims=True), 1.0)
    out = raw_out.astype(jnp.float32) / den
    return out.astype(jnp.float32), gate, gate_norm



def _attn_forward_cached(x, pool_params, router_params, expand_O_kernel,
                         n_heads, d_model,
                         cache_K, cache_V, cache_len):
    """Cached attention decode step. x: [B, 1, D]."""
    B = x.shape[0]
    d_head = d_model // n_heads

    # Route embeddings are used as-is, matching the training path.
    qk_norm = pool_params['attn_qk_emb']
    v_norm = pool_params['attn_v_emb']
    h_all = x @ router_params['proj_attn']['kernel'] + router_params['proj_attn']['bias']
    h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)
    tau_all = x @ router_params['tau_attn']['kernel'] + router_params['tau_attn']['bias']
    raw_scan_offset_all = x @ router_params['raw_scan_offset_attn']['kernel'] + router_params['raw_scan_offset_attn']['bias']

    Q = _srw_inference(x, h_Q, qk_norm, tau_all[:, :, 0:1], raw_scan_offset_all[:, :, 0:1],
                       pool_params['attn_qk_read'], pool_params['attn_qk_write'])
    K_new = _srw_inference(x, h_K, qk_norm, tau_all[:, :, 1:2], raw_scan_offset_all[:, :, 1:2],
                           pool_params['attn_qk_read'], pool_params['attn_qk_write'])
    V_new = _srw_inference(x, h_V, v_norm, tau_all[:, :, 2:3], raw_scan_offset_all[:, :, 2:3],
                           pool_params['attn_v_read'], pool_params['attn_v_write'])
    _qk_s = pool_params['attn_qk_scale']
    _v_s = pool_params['attn_v_scale']
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


def _rst_forward_inference(x, pool_params, router_params):
    """Inference-only RST Layer forward. No chunking, no LB, no dropout."""
    # emb used as-is (matches training path).
    rst_norm = pool_params['rst_emb']
    h = x @ router_params['proj_rst']['kernel'] + router_params['proj_rst']['bias']
    tau = x @ router_params['tau_rst']['kernel'] + router_params['tau_rst']['bias']
    raw_scan_offset = x @ router_params['raw_scan_offset_rst']['kernel'] + router_params['raw_scan_offset_rst']['bias']
    out = _srw_inference(x, h, rst_norm, tau, raw_scan_offset,
                         pool_params['rst_read'], pool_params['rst_write'])
    return out * pool_params['rst_scale']




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

    # Route embeddings are used as-is, matching the training path.
    qk_norm = pool_params['attn_qk_emb']
    v_norm = pool_params['attn_v_emb']

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
        raw_scan_offset_all = normed @ router_params['raw_scan_offset_attn']['kernel'] + router_params['raw_scan_offset_attn']['bias']

        Q = _srw_inference(normed, h_Q, qk_norm, tau_all[:, :, 0:1], raw_scan_offset_all[:, :, 0:1],
                           pool_params['attn_qk_read'], pool_params['attn_qk_write'])
        K_val = _srw_inference(normed, h_K, qk_norm, tau_all[:, :, 1:2], raw_scan_offset_all[:, :, 1:2],
                               pool_params['attn_qk_read'], pool_params['attn_qk_write'])
        V_val = _srw_inference(normed, h_V, v_norm, tau_all[:, :, 2:3], raw_scan_offset_all[:, :, 2:3],
                               pool_params['attn_v_read'], pool_params['attn_v_write'])
        _qk_s = pool_params['attn_qk_scale']
        _v_s = pool_params['attn_v_scale']
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
        rst_out = _rst_forward_inference(normed, pool_params, router_params)
        x = x + rst_out
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
        rst_out = _rst_forward_inference(normed, pool_params, router_params)
        x = x + rst_out
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
    Returns jnp scalars: avg_loss, ppl, accuracy, total_valid.
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

    # Route embeddings are used as-is, matching the training path.
    qk_norm = pool_params['attn_qk_emb']
    v_norm = pool_params['attn_v_emb']
    rst_norm = pool_params['rst_emb']

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
            raw_scan_offset_all = normed @ router_params['raw_scan_offset_attn']['kernel'] + router_params['raw_scan_offset_attn']['bias']

            Q = _srw_inference(normed, h_Q, qk_norm, tau_all[:, :, 0:1], raw_scan_offset_all[:, :, 0:1],
                               pool_params['attn_qk_read'], pool_params['attn_qk_write'])
            K = _srw_inference(normed, h_K, qk_norm, tau_all[:, :, 1:2], raw_scan_offset_all[:, :, 1:2],
                               pool_params['attn_qk_read'], pool_params['attn_qk_write'])
            V = _srw_inference(normed, h_V, v_norm, tau_all[:, :, 2:3], raw_scan_offset_all[:, :, 2:3],
                               pool_params['attn_v_read'], pool_params['attn_v_write'])
            _qk_s = pool_params['attn_qk_scale']
            _v_s = pool_params['attn_v_scale']
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
            h_k = normed @ router_params['proj_rst']['kernel'] + router_params['proj_rst']['bias']
            tau_k = normed @ router_params['tau_rst']['kernel'] + router_params['tau_rst']['bias']
            raw_scan_offset_k = normed @ router_params['raw_scan_offset_rst']['kernel'] + router_params['raw_scan_offset_rst']['bias']
            rst_out = _srw_inference(normed, h_k, rst_norm, tau_k, raw_scan_offset_k,
                                     pool_params['rst_read'], pool_params['rst_write'])
            x = x + rst_out * pool_params['rst_scale']
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
    for pool_name, emb_key in [
            ('Attention-QK', 'attn_qk_emb'),
            ('Attention-V', 'attn_v_emb'),
            ('RST', 'rst_emb')]:
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
    results['tau_rst_bias'] = params['router']['tau_rst']['bias']
    return results


def vectorized_weight_analysis(params, max_sample=2048):
    """Weight analysis: effective rank + cosine sim. All on device."""
    params = _squeeze_params(params)
    pool = params['neuron_pool']
    results = {}
    for pool_name, emb_key in [
            ('Attention-QK', 'attn_qk_emb'),
            ('Attention-V', 'attn_v_emb'),
            ('RST', 'rst_emb')]:
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
    mode='light': returns gate_norm only.

    Returns:
        logits: [B, S, vocab]
        layer_info: dict with stacked arrays:
            gate_Q: [n_layers, B, S, n_qk]
            gate_K: [n_layers, B, S, n_qk]
            gate_V: [n_layers, B, S, n_v]
            gate_RST: [n_layers, B, S, n_rst]
            (mode='full' only) gate_Q_raw, gate_K_raw, gate_V_raw, gate_RST_raw
            attn_out_norm: [n_layers]
            rst_out_norm: [n_layers]
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

    # Route embeddings are used as-is, matching the training path.
    qk_norm = pool_params['attn_qk_emb']
    v_norm = pool_params['attn_v_emb']
    rst_norm_w = pool_params['rst_emb']

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
        raw_scan_offset_all = normed @ router_params['raw_scan_offset_attn']['kernel'] + router_params['raw_scan_offset_attn']['bias']

        Q, gate_Q_raw, gate_Q = _srw_inference_with_gates(
            normed, h_Q, qk_norm, tau_all[:, :, 0:1], raw_scan_offset_all[:, :, 0:1],
            pool_params['attn_qk_read'], pool_params['attn_qk_write'])
        K, gate_K_raw, gate_K = _srw_inference_with_gates(
            normed, h_K, qk_norm, tau_all[:, :, 1:2], raw_scan_offset_all[:, :, 1:2],
            pool_params['attn_qk_read'], pool_params['attn_qk_write'])
        V, gate_V_raw, gate_V = _srw_inference_with_gates(
            normed, h_V, v_norm, tau_all[:, :, 2:3], raw_scan_offset_all[:, :, 2:3],
            pool_params['attn_v_read'], pool_params['attn_v_write'])
        _qk_s = pool_params['attn_qk_scale']
        _v_s = pool_params['attn_v_scale']
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
        h_k = normed @ router_params['proj_rst']['kernel'] + router_params['proj_rst']['bias']
        tau_k = normed @ router_params['tau_rst']['kernel'] + router_params['tau_rst']['bias']
        raw_scan_offset_k = normed @ router_params['raw_scan_offset_rst']['kernel'] + router_params['raw_scan_offset_rst']['bias']
        rst_out, gate_RST_raw, gate_RST = _srw_inference_with_gates(
            normed, h_k, rst_norm_w, tau_k, raw_scan_offset_k,
            pool_params['rst_read'], pool_params['rst_write'])
        rst_out = rst_out * pool_params['rst_scale']
        rst_out_norm = jnp.linalg.norm(rst_out, axis=-1).mean()
        x = x + rst_out

        info = {
            'gate_Q': gate_Q, 'gate_K': gate_K,
            'gate_V': gate_V, 'gate_RST': gate_RST,
            'attn_out_norm': attn_out_norm,
            'rst_out_norm': rst_out_norm,
        }
        if _return_raw:
            info['gate_Q_raw'] = gate_Q_raw
            info['gate_K_raw'] = gate_K_raw
            info['gate_V_raw'] = gate_V_raw
            info['gate_RST_raw'] = gate_RST_raw
        # Legacy analysis aliases; remove after downstream analysis is migrated.
        info['gate_Know'] = info['gate_RST']
        if _return_raw:
            info['gate_Know_raw'] = info['gate_RST_raw']
        return x, info

    xs = {'params': stacked}
    x, layer_info = jax.lax.scan(analysis_layer, x, xs)

    norm_p = params['norm']
    x = _layer_norm(x, norm_p['scale'], norm_p['bias'])
    logits = x @ params['token_emb']['embedding'].T
    return logits, layer_info


def build_suppressed_forward(params, model_cfg, suppress_masks):
    """Build forward with specific neurons suppressed (gate zeroed).

    suppress_masks: dict with 'qk':[n_qk] bool, 'v':[n_v], 'rst':[n_rst] bool.
    Legacy key 'know' is still accepted.
    True = suppress.
    Returns: forward_fn(input_ids) -> logits [B, S, vocab]
    """
    params = _squeeze_params(params)
    params = jax.tree.map(jnp.asarray, params)
    qk_mult = jnp.where(suppress_masks.get('qk', jnp.zeros(1, dtype=bool)), 0.0, 1.0) \
        if 'qk' in suppress_masks else None
    v_mult = jnp.where(suppress_masks.get('v', jnp.zeros(1, dtype=bool)), 0.0, 1.0) \
        if 'v' in suppress_masks else None
    rst_mask = suppress_masks.get('rst', suppress_masks.get('know', None))
    rst_mult = jnp.where(rst_mask, 0.0, 1.0) if rst_mask is not None else None

    def _srw_sup(x, h, emb, tau_off, raw_scan_offset, w_read, w_write, mult):
        """SRW with optional gate suppression."""
        # v4.1.5.5: raw read/write vectors with natural vector magnitudes.
        r_n = w_read.astype(jnp.float32)
        w_n = w_write.astype(jnp.float32)
        scores = h @ emb.T
        sf = scores.astype(jnp.float32)
        s_mean = sf.mean(axis=-1, keepdims=True)
        s_std = jnp.sqrt(jnp.mean(jnp.square(sf - s_mean), axis=-1, keepdims=True)) + 1e-8
        scan_offset = SCAN_SCALE * jnp.tanh(raw_scan_offset)
        tau = s_mean + tau_off * s_std - scan_offset / jnp.maximum(s_std, SCAN_STD_FLOOR)
        raw = scores - tau.astype(scores.dtype)
        margin = raw.astype(jnp.float32) - ACTIVATION_THRESHOLD
        activation = jax.nn.sigmoid(SHARPNESS * margin)
        active_margin = jnp.maximum(margin - ACTIVATION_CUTOFF, 0.0)
        intensity = EPSILON + jnp.minimum(active_margin, MAX_INTENSITY)
        gate = activation * intensity
        if mult is not None:
            gate = gate * mult[None, None, :]
            activation = activation * mult[None, None, :]
        xr = x.astype(jnp.float32) @ r_n.T
        a = gate * xr
        out = a @ w_n
        den = jnp.maximum(gate.sum(axis=-1, keepdims=True), 1.0)
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
        # Route embeddings are used as-is, matching the training path.
        qk_n = pp['attn_qk_emb']
        v_n = pp['attn_v_emb']
        kn_n = pp['rst_emb']

        for i in range(n_layers):
            bp = params[f'block_{i}']
            normed = _layer_norm(x, bp['norm1']['scale'], bp['norm1']['bias'])
            h_all = normed @ rp['proj_attn']['kernel'] + rp['proj_attn']['bias']
            h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)
            tau_all = normed @ rp['tau_attn']['kernel'] + rp['tau_attn']['bias']
            raw_scan_offset_all = normed @ rp['raw_scan_offset_attn']['kernel'] + rp['raw_scan_offset_attn']['bias']

            Q = _srw_sup(normed, h_Q, qk_n, tau_all[:,:,0:1], raw_scan_offset_all[:,:,0:1], pp['attn_qk_read'], pp['attn_qk_write'], qk_mult)
            K = _srw_sup(normed, h_K, qk_n, tau_all[:,:,1:2], raw_scan_offset_all[:,:,1:2], pp['attn_qk_read'], pp['attn_qk_write'], qk_mult)
            V = _srw_sup(normed, h_V, v_n, tau_all[:,:,2:3], raw_scan_offset_all[:,:,2:3], pp['attn_v_read'], pp['attn_v_write'], v_mult)
            _qk_s = pp['attn_qk_scale']
            _v_s = pp['attn_v_scale']
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
            h_k = normed @ rp['proj_rst']['kernel'] + rp['proj_rst']['bias']
            tau_k = normed @ rp['tau_rst']['kernel'] + rp['tau_rst']['bias']
            raw_scan_offset_k = normed @ rp['raw_scan_offset_rst']['kernel'] + rp['raw_scan_offset_rst']['bias']
            x = x + _srw_sup(normed, h_k, kn_n, tau_k, raw_scan_offset_k, pp['rst_read'], pp['rst_write'], rst_mult) * pp['rst_scale']

        norm_p = params['norm']
        x = _layer_norm(x, norm_p['scale'], norm_p['bias'])
        return x @ params['token_emb']['embedding'].T

    return forward_fn


def _rename_key_if_needed(d, old, new):
    """Rename old -> new in a mutable mapping, preserving an existing new key."""
    if old in d:
        if new not in d:
            d[new] = d[old]
        del d[old]


def migrate_legacy_v4155_params(params):
    """
    Convert legacy v4.1.5.5 parameter names to the new DAWN-SRW/RST names.

    Legacy:
        qk_* -> attn_qk_*
        v_* -> attn_v_*
        know_* -> rst_*
        proj_know -> proj_rst
        tau_know -> tau_rst
        scan_bias_attn -> raw_scan_offset_attn
        scan_bias_know -> raw_scan_offset_rst

    The function is safe to call on already-migrated params. It does not mutate
    the input in-place, preserves unknown keys, and handles plain dicts and
    Flax FrozenDict parameter trees. If old and new keys both exist, the new
    key is preferred.
    """
    was_frozen = isinstance(params, FrozenDict)
    tree = unfreeze(params) if was_frozen else jax.tree.map(
        lambda x: x, params, is_leaf=lambda x: not isinstance(x, dict))

    def _migrate_container(container):
        if not isinstance(container, dict):
            return
        if 'neuron_pool' in container and isinstance(container['neuron_pool'], dict):
            pool = container['neuron_pool']
            for old, new in (
                    ('qk_emb', 'attn_qk_emb'),
                    ('qk_read', 'attn_qk_read'),
                    ('qk_write', 'attn_qk_write'),
                    ('qk_scale', 'attn_qk_scale'),
                    ('v_emb', 'attn_v_emb'),
                    ('v_read', 'attn_v_read'),
                    ('v_write', 'attn_v_write'),
                    ('v_scale', 'attn_v_scale'),
                    ('know_emb', 'rst_emb'),
                    ('know_read', 'rst_read'),
                    ('know_write', 'rst_write'),
                    ('know_scale', 'rst_scale'),
            ):
                _rename_key_if_needed(pool, old, new)
        if 'router' in container and isinstance(container['router'], dict):
            router = container['router']
            for old, new in (
                    ('proj_know', 'proj_rst'),
                    ('tau_know', 'tau_rst'),
                    ('scan_bias_attn', 'raw_scan_offset_attn'),
                    ('scan_bias_know', 'raw_scan_offset_rst'),
            ):
                _rename_key_if_needed(router, old, new)

    def _walk(node):
        if isinstance(node, dict):
            _migrate_container(node)
            for value in node.values():
                _walk(value)

    _walk(tree)
    return freeze(tree) if was_frozen else tree
