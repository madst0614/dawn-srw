"""
DAWN-SRW v4.1.5.9: SRW neurons and Residual State Transition.

This file contains the active v4.1.5.9 implementation. DAWN neurons are
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

    rhat_i = r_i / ||r_i||
    what_i = w_i / ||w_i||
    O_i^RW(x) = <x, rhat_i> what_i
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
    activation    = clip(raw / activation_threshold, 0, 1)
    intensity     = min(max(scores, 0), max_intensity)
    gate          = activation * intensity

    rhat_i = r_i / ||r_i||
    what_i = w_i / ||w_i||
    O_i^RW(x) = <x, rhat_i> what_i
    out = sum_i gate_i * O_i^RW(x) / max(sum_i gate_i, 1.0)

Implementation notes
--------------------
* Routing is score-based in route space: scores = h @ emb.T.
* v4.1.5.9 can forward-normalize route embeddings inside the sharded
  SRW closures (default on for stability); stored signature parameters
  remain raw and their norms remain diagnostics.
* v4.1.5.9 uses forward unit-normalized read/write vectors: stored read/write
  parameters remain raw, but SRW execution uses their directions.
* Raw read/write parameter norms and norm-product stats remain diagnostics.
* The denominator remains activation-weighted intensity: sum(gate).
* v4.1.5.9 uses a clipped activation transition over raw / activation_width, optionally shaped by activation_power
  and uses pure score-based intensity; tau controls activation only
  and no longer appears in the intensity branch.
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

def _pool_scale_mode_is_learned(mode) -> bool:
    return str(mode).lower() in (
        'learned', 'learnable', 'trainable', 'param', 'parameter')


def _model_cfg_uses_fixed_depth_pool_scale(model_cfg) -> bool:
    # Default policy: fixed depth-scaled pool scales.
    # To opt back into learned sqrt(d_model)-initialized pool scales, set:
    #   model.pool_scale_mode: learned
    # or explicitly set:
    #   model.fixed_depth_pool_scale: false
    if not isinstance(model_cfg, dict):
        return True
    if 'fixed_depth_pool_scale' in model_cfg:
        return bool(model_cfg['fixed_depth_pool_scale'])
    mode = model_cfg.get('pool_scale_mode', 'fixed_depth')
    if _pool_scale_mode_is_learned(mode):
        return False
    return True


def _fixed_depth_pool_output_scales(d_model, n_layers):
    dm = jnp.asarray(d_model, dtype=jnp.float32)
    nl = jnp.asarray(n_layers, dtype=jnp.float32)
    qk_scale = jnp.sqrt(dm / nl)
    v_scale = jnp.sqrt(dm / nl)
    rst_scale = jnp.sqrt(dm / nl)
    return (
        jax.lax.stop_gradient(qk_scale),
        jax.lax.stop_gradient(v_scale),
        jax.lax.stop_gradient(rst_scale),
    )


def _effective_pool_output_scales(pool_params, d_model, n_layers,
                                  fixed_depth_pool_scale=False):
    if fixed_depth_pool_scale:
        return _fixed_depth_pool_output_scales(d_model, n_layers)
    return (
        pool_params['attn_qk_scale'],
        pool_params['attn_v_scale'],
        pool_params['rst_scale'],
    )


# ================================================================
# V4.1 physical constants (defaults; overridable via config).
#
#   margin        = (score - tau) - activation_width
#   z             = clip(raw / activation_width, 0, 1)
#   activation    = z ** activation_power
#   intensity     = min(max(score, 0), MAX_INTENSITY)
#   gate          = activation * intensity
#   den           = max(sum(gate), 1.0)
# ================================================================

SHARPNESS = 500.0              # legacy/API only; v4.1.5.9 activation ignores this
ACTIVATION_THRESHOLD = 0.5     # activation transition length; raw >= this gives activation=1
ACTIVATION_CUTOFF = 0.01       # intensity starts margin beyond this point
EPSILON = 1e-4                 # minimum intensity floor
MAX_INTENSITY = 10.0           # intensity cap (drift safety)
SCAN_SCALE = 0.01              # max absolute scan movement before /std
SCAN_STD_FLOOR = 0.5           # caps low-std scan amplification
DEFAULT_D_ROUTE = 64
RW_FORWARD_NORM_EPS = 1e-6     # forward-only read/write direction floor


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


def _forward_unit_direction(x):
    return x / (jnp.linalg.norm(x, axis=-1, keepdims=True)
                + RW_FORWARD_NORM_EPS)


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


LOCAL_SPIKE_METRIC_COUNT = 11
LOCAL_SPIKE_TOP1_COUNT = 17
ATTN_LOCAL_METRIC_COUNT = 7


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
                     analysis=False,
                     prune_enabled=False,
                     prune_activation_threshold=None,
                     prune_scope="all",
                     prune_denominator="pruned",
                     return_prune_stats=False,
                     local_diagnostics=False,
                     route_emb_forward_norm=False,
                     intensity_route_dim=0,
                     intensity_beta=0.5,
                     intensity_squash="tanh",
                     intensity_width=1.0,
                     activation_power=1.0):
    """Create fused shard_map'd gate+srw. Gate never materialised full.

    2-pass chunked inside shard_map:
      Pass 1: scores stats -> static tau (psum for cross-chip mean/std)
      Pass 2: gate+srw fused per chunk (gate computed and consumed per chunk)

    v4.1 gate:
        raw           = score - tau
        margin        = raw - activation_width
        z             = clip(raw / activation_width, 0, 1)
        activation    = z ** activation_power
        intensity     = min(max(score, 0), max_intensity)
        gate          = activation * intensity
        den           = max(sum(gate), 1.0)

    scan_offset = scan_scale * tanh(raw_scan_offset)
    tau = s_mean + tau_offset * s_std - scan_offset / max(s_std, scan_std_floor).
    All v4.1 constants are closure-baked.

    `analysis=False` (default, train path): returns the SLIM tuple plus
    four gate-concentration diagnostics, and skips distribution-shape stats
    (skew/kurt), boundary/entropy counters and intensity-cap fraction.
    XLA DCE's the unused work.
    `local_diagnostics=True` appends a lightweight, scalar-only local-spike
    summary to either path. It is independent of `analysis=True` and is
    collected inline during the existing chunk scan.
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
    _prune_enabled = bool(prune_enabled)
    if _prune_enabled and prune_activation_threshold is None:
        raise ValueError(
            "prune_activation_threshold must be set when prune_enabled=True")
    if prune_denominator not in ("pruned", "retained", "full"):
        raise ValueError(
            "prune_denominator must be 'pruned', 'retained', or 'full', "
            f"got {prune_denominator!r}")
    # prune_scope is resolved by the caller that builds pool-specific closures.
    # It is accepted here so paper-eval configs can carry one canonical option set.
    _ = prune_scope
    _prune_thr = jnp.float32(
        0.0 if prune_activation_threshold is None
        else prune_activation_threshold)
    _denominator_is_full = (prune_denominator == "full")
    _return_prune_stats = bool(return_prune_stats)
    _local_diagnostics = bool(local_diagnostics)
    # Default is raw route signatures. Forward-normalization is kept as an
    # optional ablation only; it can reduce routing-intensity freedom and may
    # limit expressivity.
    # Default is raw route signatures. Forward-normalization is kept as an
    # optional ablation only; it can reduce routing-intensity freedom and may
    # limit expressivity.
    _route_emb_forward_norm = bool(route_emb_forward_norm)
    _intensity_route_dim = int(intensity_route_dim or 0)
    _two_channel_intensity = _intensity_route_dim > 0
    _intensity_beta = jnp.float32(intensity_beta)
    _intensity_squash = str(intensity_squash).lower()
    _intensity_width = jnp.float32(intensity_width)
    _activation_power = jnp.float32(activation_power)
    _int_cap_thresh = _eps + _max_int - jnp.float32(1e-3)

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
    _prune_extra_specs = (
        P(),                     # kept_count_mean scalar
        P(),                     # kept_frac_mean scalar
        P(),                     # full_gate_sum_mean scalar
        P(),                     # kept_gate_sum_mean scalar
        P(),                     # retained_gate_mass_mean scalar
        P(),                     # int_cap_frac scalar
        P(),                     # gate_max_mean scalar
    )
    _local_diag_specs = (
        P(),                     # local_spike_values [1, metric]
        P(),                     # local_spike_locs [1, metric, b/t/neuron]
        P(),                     # top1_breakdown_values [1, field]
        P(),                     # top1_breakdown_locs [1, b/t/neuron]
    )
    _out_specs = (_slim_out_specs + _conc_out_specs + _analysis_extra_specs
                  if analysis else _slim_out_specs + _conc_out_specs)
    if _return_prune_stats:
        _out_specs = _out_specs + _prune_extra_specs
    if _local_diagnostics:
        _out_specs = _out_specs + _local_diag_specs

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
        diag_neg_inf = jnp.float32(-1.0e30)
        diag_vals_init = jnp.full(
            (1, LOCAL_SPIKE_METRIC_COUNT), diag_neg_inf)

        def route_emb_chunk(start):
            ec = jax.lax.dynamic_slice_in_dim(emb_bf, start, cs, axis=0)
            if _route_emb_forward_norm:
                ec_f = ec.astype(jnp.float32)
                ec = _forward_unit_direction(ec_f).astype(jnp.bfloat16)
            return ec

        def route_relation_and_intensity(h_in, route):
            # Two-channel mode: split the existing route dimension into
            # selection/address dims and intensity/mixture dims.  This keeps
            # the dense routing dot cost approximately unchanged:
            #   d_select + d_intensity == d_route.
            if _two_channel_intensity:
                d_total = h_in.shape[-1]
                d_str = min(max(_intensity_route_dim, 1), d_total - 1)
                d_sel = d_total - d_str
                h_sel = h_in[..., :d_sel].astype(jnp.float32)
                route_sel = route[:, :d_sel].astype(jnp.float32)
                h_sel = _forward_unit_direction(h_sel).astype(jnp.bfloat16)
                route_sel = _forward_unit_direction(route_sel).astype(jnp.bfloat16)
                relation = (h_sel @ route_sel.T).astype(jnp.float32)

                h_str = h_in[..., d_sel:].astype(jnp.bfloat16)
                route_str = route[:, d_sel:].astype(jnp.bfloat16)
                intensity_raw = (h_str @ route_str.T).astype(jnp.float32)
                z = intensity_raw / jnp.maximum(_intensity_width, jnp.float32(1e-6))
                if _intensity_squash == "softsign":
                    intensity_log = _intensity_beta * (z / (jnp.float32(1.0) + jnp.abs(z)))
                else:
                    intensity_log = _intensity_beta * jnp.tanh(z)
                intensity = jnp.exp(intensity_log)
                return relation, intensity

            scores = (h_in @ route.T).astype(jnp.float32)
            intensity = jnp.minimum(jnp.maximum(scores, 0.0), _max_int)
            return scores, intensity

        def route_rw_chunk(start):
            ec = route_emb_chunk(start)
            rc = jax.lax.dynamic_slice_in_dim(read_bf, start, cs, axis=0)
            wc = jax.lax.dynamic_slice_in_dim(write_bf, start, cs, axis=0)
            rc_f = rc.astype(jnp.float32)
            wc_f = wc.astype(jnp.float32)
            # v4.1.5.9: stored read/write params stay raw, but SRW
            # execution uses their directions.
            rc_dir = _forward_unit_direction(rc_f)
            wc_dir = _forward_unit_direction(wc_f)
            return ec, rc_dir.astype(jnp.bfloat16), wc_dir.astype(jnp.bfloat16)

        # --- Pass 1: exact stats over ALL chunks (scan + checkpoint) ---
        if analysis:
            @jax.checkpoint
            def stats_step(carry, i):
                s_sum, sq_sum, cube_sum, quad_sum, ns_sum, ns_sq = carry
                s = i * cs
                route = route_emb_chunk(s)
                scores_f, _ = route_relation_and_intensity(h_bf, route)
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
                route = route_emb_chunk(s)
                scores_f, _ = route_relation_and_intensity(h_bf, route)
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
        s_var = jnp.maximum(global_sq / N_total - s_mean ** 2, 0.0)
        s_std = jnp.sqrt(s_var) + 1e-8
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
        var_score = jnp.maximum(global_ns_sq / N_total - mean_score ** 2, 0.0)
        score_lb = var_score / (mean_score ** 2 + var_score + 1e-2)

        # --- Pass 2: gate + srw fused (scan + checkpoint) ---
        # v4.1 diagnostic: ceiling on intensity relative to cap (1e-3 below).
        if analysis:
            @jax.checkpoint
            def gate_srw_step(carry, i):
                (out, total_weighted_cost, total_gate_sq, total_gate_max, total_active,
                 total_strong, total_phi_binary, total_den_cost,
                 total_activation_cost, total_current_cost,
                 total_z_lt_075, total_z_lt_030, total_g_log_g,
                 total_dead_penalty, total_dead_count,
                 total_int_max, total_full_gate, total_kept_count,
                 total_int_cap_count, diag_vals) = carry
                s = i * cs
                route, rc, wc = route_rw_chunk(s)
                scores_f, intensity = route_relation_and_intensity(h_bf, route)
                raw = scores_f - tau
                margin = raw - _act_thr
                z_act = jnp.clip(raw / _act_thr, 0.0, 1.0)
                activation = jnp.power(z_act, _activation_power)
                base_gate = activation * intensity
                if _prune_enabled:
                    keep = activation > _prune_thr
                    gate = jnp.where(keep, base_gate, 0.0)
                    chunk_kept = keep.astype(jnp.float32).sum(
                        axis=-1, keepdims=True)
                else:
                    gate = base_gate
                    chunk_kept = jnp.full((B, S, 1), cs, dtype=jnp.float32)
                chunk_int_max = intensity.max()
                chunk_int_cap_count = (intensity >= _int_cap_thresh
                                        ).astype(jnp.float32).sum()
                xr = x_bf @ rc.T
                xr_f = xr.astype(jnp.float32)
                a = gate * xr_f
                c_out = (a.astype(jnp.bfloat16) @ wc).astype(jnp.float32)
                chunk_weighted = gate.sum(axis=-1, keepdims=True)
                chunk_gate_sq = jnp.square(gate).sum(axis=-1, keepdims=True)
                chunk_full_gate = base_gate.sum(axis=-1, keepdims=True)
                chunk_den_cost = (chunk_full_gate if _denominator_is_full
                                  else chunk_weighted)
                if _local_diagnostics:
                    write_norm = jnp.linalg.norm(
                        wc.astype(jnp.float32), axis=-1)
                    contrib_proxy = (
                        jnp.abs(jax.lax.stop_gradient(gate * xr_f))
                        * write_norm[None, None, :])
                    diag_chunk = jnp.full_like(diag_vals, diag_neg_inf)
                    diag_chunk = diag_chunk.at[:, 1].set(
                        jnp.max(jax.lax.stop_gradient(margin)))
                    diag_chunk = diag_chunk.at[:, 4].set(
                        jnp.max(jax.lax.stop_gradient(intensity)))
                    diag_chunk = diag_chunk.at[:, 5].set(
                        jnp.max(jnp.abs(jax.lax.stop_gradient(xr_f))))
                    diag_chunk = diag_chunk.at[:, 6].set(
                        jnp.max(contrib_proxy))
                    diag_chunk = diag_chunk.at[:, 10].set(
                        jnp.max(jnp.linalg.norm(
                            jax.lax.stop_gradient(route.astype(jnp.float32)),
                            axis=-1)))
                    diag_vals = jnp.maximum(diag_vals, diag_chunk)
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
                        total_den_cost + chunk_den_cost,
                        total_activation_cost,
                        total_current_cost,
                        total_z_lt_075 + chunk_z_lt_075,
                        total_z_lt_030 + chunk_z_lt_030,
                        total_g_log_g + chunk_g_log_g,
                        total_dead_penalty + chunk_dead_penalty,
                        total_dead_count + chunk_dead_count,
                        jnp.maximum(total_int_max, chunk_int_max),
                        total_full_gate + chunk_full_gate,
                        total_kept_count + chunk_kept,
                        total_int_cap_count + chunk_int_cap_count,
                        diag_vals), None

            (raw_out, total_weighted_cost, total_gate_sq, total_gate_max, total_active, total_strong,
             total_phi_binary, total_den_cost, total_activation_cost,
             total_current_cost, total_z_lt_075, total_z_lt_030,
             total_g_log_g, total_dead_penalty, total_dead_count,
             total_int_max, total_full_gate, total_kept_count,
             total_int_cap_count, diag_vals), _ = jax.lax.scan(
                gate_srw_step,
                (jnp.zeros((B, S, D), dtype=jnp.float32),
                 z1, z1, jnp.full((B, S, 1), -1e9), z1, z1, z1, z1, z1, z1, z1, z1, z1,
                 jnp.float32(0.0), jnp.float32(0.0),
                 jnp.float32(0.0), z1, z1, jnp.float32(0.0),
                 diag_vals_init),
                jnp.arange(nc))
        else:
            @jax.checkpoint
            def gate_srw_step(carry, i):
                (out, total_weighted_cost, total_gate_sq, total_gate_max, total_active,
                 total_strong, total_den_cost,
                 total_activation_cost, total_current_cost,
                 total_dead_penalty, total_dead_count,
                 total_int_max, total_full_gate, total_kept_count,
                 total_int_cap_count, diag_vals) = carry
                s = i * cs
                route, rc, wc = route_rw_chunk(s)
                scores_f, intensity = route_relation_and_intensity(h_bf, route)
                raw = scores_f - tau
                margin = raw - _act_thr
                z_act = jnp.clip(raw / _act_thr, 0.0, 1.0)
                activation = jnp.power(z_act, _activation_power)
                base_gate = activation * intensity
                if _prune_enabled:
                    keep = activation > _prune_thr
                    gate = jnp.where(keep, base_gate, 0.0)
                    chunk_kept = keep.astype(jnp.float32).sum(
                        axis=-1, keepdims=True)
                else:
                    gate = base_gate
                    chunk_kept = jnp.full((B, S, 1), cs, dtype=jnp.float32)
                chunk_int_max = intensity.max()
                chunk_int_cap_count = (intensity >= _int_cap_thresh
                                        ).astype(jnp.float32).sum()
                xr = x_bf @ rc.T
                xr_f = xr.astype(jnp.float32)
                a = gate * xr_f
                c_out = (a.astype(jnp.bfloat16) @ wc).astype(jnp.float32)
                chunk_weighted = gate.sum(axis=-1, keepdims=True)
                chunk_gate_sq = jnp.square(gate).sum(axis=-1, keepdims=True)
                chunk_full_gate = (
                    base_gate.sum(axis=-1, keepdims=True)
                    if (_return_prune_stats or _denominator_is_full)
                    else chunk_weighted)
                chunk_den_cost = (chunk_full_gate if _denominator_is_full
                                  else chunk_weighted)
                if _local_diagnostics:
                    write_norm = jnp.linalg.norm(
                        wc.astype(jnp.float32), axis=-1)
                    contrib_proxy = (
                        jnp.abs(jax.lax.stop_gradient(gate * xr_f))
                        * write_norm[None, None, :])
                    diag_chunk = jnp.full_like(diag_vals, diag_neg_inf)
                    diag_chunk = diag_chunk.at[:, 1].set(
                        jnp.max(jax.lax.stop_gradient(margin)))
                    diag_chunk = diag_chunk.at[:, 4].set(
                        jnp.max(jax.lax.stop_gradient(intensity)))
                    diag_chunk = diag_chunk.at[:, 5].set(
                        jnp.max(jnp.abs(jax.lax.stop_gradient(xr_f))))
                    diag_chunk = diag_chunk.at[:, 6].set(
                        jnp.max(contrib_proxy))
                    diag_chunk = diag_chunk.at[:, 10].set(
                        jnp.max(jnp.linalg.norm(
                            jax.lax.stop_gradient(route.astype(jnp.float32)),
                            axis=-1)))
                    diag_vals = jnp.maximum(diag_vals, diag_chunk)
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
                        total_den_cost + chunk_den_cost,
                        total_activation_cost,
                        total_current_cost,
                        total_dead_penalty + chunk_dead_penalty,
                        total_dead_count + chunk_dead_count,
                        jnp.maximum(total_int_max, chunk_int_max),
                        total_full_gate + chunk_full_gate,
                        total_kept_count + chunk_kept,
                        total_int_cap_count + chunk_int_cap_count,
                        diag_vals), None

            (raw_out, total_weighted_cost, total_gate_sq, total_gate_max, total_active, total_strong,
             total_den_cost, total_activation_cost, total_current_cost,
             total_dead_penalty, total_dead_count,
             total_int_max, total_full_gate, total_kept_count,
             total_int_cap_count, diag_vals), _ = jax.lax.scan(
                gate_srw_step,
                (jnp.zeros((B, S, D), dtype=jnp.float32),
                 z1, z1, jnp.full((B, S, 1), -1e9), z1, z1, z1, z1, z1,
                 jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0),
                 z1, z1, jnp.float32(0.0), diag_vals_init),
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
        # Measurement path: detached copies for diagnostics / feedback refs.
        # Action path above keeps global_den_cost/global_weighted_cost live for
        # the SRW denominator and output gradient.
        global_weighted_cost_m = jax.lax.stop_gradient(global_weighted_cost)
        global_gate_sq_m = jax.lax.stop_gradient(global_gate_sq)
        global_den_cost_m = jax.lax.stop_gradient(global_den_cost)
        global_activation_cost_m = jax.lax.stop_gradient(global_activation_cost)
        global_current_cost_m = jax.lax.stop_gradient(global_current_cost)
        global_active_m = jax.lax.stop_gradient(global_active)
        global_strong_m = jax.lax.stop_gradient(
            jax.lax.psum(total_strong, 'model'))
        global_gate_max_m = jax.lax.stop_gradient(global_gate_max)
        active_frac = global_active_m / N_total
        strong_frac = global_strong_m / N_total
        z_mean_active = global_weighted_cost_m / (global_active_m + 1e-8)

        score_std_out = jax.lax.stop_gradient(s_std.mean())
        es_out = global_weighted_cost_m.mean()          # sum(gate), observational
        active_n_mean = global_active_m.mean()
        gate_eff_n = ((global_weighted_cost_m ** 2)
                      / (global_gate_sq_m + 1e-8))
        gate_eff_ratio = gate_eff_n / jnp.maximum(global_active_m, 1.0)
        top1_gate_frac = global_gate_max_m / jnp.maximum(
            global_weighted_cost_m, 1e-8)
        tau_abs_mean = jax.lax.stop_gradient(tau).mean()
        dead_penalty_out = jax.lax.psum(total_dead_penalty, 'model')
        dead_count_out = jax.lax.stop_gradient(
            jax.lax.psum(total_dead_count, 'model'))
        # pmax has no VJP; wrap the input in stop_gradient.
        int_max_out = jax.lax.pmax(
            jax.lax.stop_gradient(total_int_max), 'model')

        den_cost_mean = global_den_cost_m.mean()
        activation_cost_mean = global_activation_cost_m.mean()
        current_cost_mean = global_current_cost_m.mean()

        slim_out = (out.astype(jnp.float32), active_frac, global_gate_max, score_lb,
                    score_std_out, es_out, active_n_mean, strong_frac, z_mean_active,
                    tau_abs_mean, dead_penalty_out, dead_count_out, int_max_out,
                    den_cost_mean, activation_cost_mean, current_cost_mean)
        conc_out = (gate_eff_n.mean(), gate_eff_ratio.mean(),
                    top1_gate_frac.mean(), top1_gate_frac.max())
        prune_out = ()
        if _return_prune_stats:
            global_full_gate = jax.lax.psum(total_full_gate, 'model')
            global_kept_count = jax.lax.psum(total_kept_count, 'model')
            global_full_gate_m = jax.lax.stop_gradient(global_full_gate)
            global_kept_count_m = jax.lax.stop_gradient(global_kept_count)
            retained_gate_mass = (
                global_weighted_cost_m / jnp.maximum(global_full_gate_m, 1e-8))
            int_cap_frac_out = jax.lax.stop_gradient(
                jax.lax.psum(total_int_cap_count, 'model')
                / jnp.float32(B * S * N_total))
            prune_out = (
                global_kept_count_m.mean(),
                (global_kept_count_m / N_total).mean(),
                global_full_gate_m.mean(),
                global_weighted_cost_m.mean(),
                retained_gate_mass.mean(),
                int_cap_frac_out,
                global_gate_max_m.mean(),
            )
        local_diag_out = ()
        if _local_diagnostics:
            tau_abs_max = jnp.max(jnp.abs(jax.lax.stop_gradient(tau_offset)))
            top1_share_max = jnp.max(
                global_gate_max_m / jnp.maximum(global_den_cost_m, 1e-8))
            gate_den_sum_max = jnp.max(global_den_cost_m)
            local_out_norm_max = jnp.max(jnp.linalg.norm(
                jax.lax.stop_gradient(out), axis=-1))
            residual_norm_max = jnp.max(jnp.linalg.norm(
                jax.lax.stop_gradient(x), axis=-1))
            h_norm_max = jnp.max(jnp.linalg.norm(
                jax.lax.stop_gradient(h), axis=-1))
            token_vals = jnp.stack([
                tau_abs_max, top1_share_max, gate_den_sum_max,
                local_out_norm_max, residual_norm_max, h_norm_max,
            ]).reshape((1, 6))
            token_slots = jnp.array([0, 2, 3, 7, 8, 9], dtype=jnp.int32)
            metric_vals = diag_vals.at[:, token_slots].set(token_vals)
            metric_vals = jax.lax.stop_gradient(metric_vals)
            metric_vals = jax.lax.pmax(metric_vals, 'model')
            metric_vals = jax.lax.pmax(metric_vals, 'data')
            metric_locs = jnp.full(
                (1, LOCAL_SPIKE_METRIC_COUNT, 3), -1, dtype=jnp.int32)
            top1_details = jnp.zeros(
                (1, LOCAL_SPIKE_TOP1_COUNT), dtype=jnp.float32)
            top1_details = top1_details.at[:, 0].set(metric_vals[:, 2])
            top1_details = top1_details.at[:, 4].set(metric_vals[:, 1])
            top1_details = top1_details.at[:, 6].set(metric_vals[:, 3])
            top1_details = top1_details.at[:, 7].set(metric_vals[:, 4])
            top1_details = top1_details.at[:, 8].set(metric_vals[:, 5])
            top1_details = top1_details.at[:, 12].set(metric_vals[:, 6])
            top1_details = top1_details.at[:, 13].set(metric_vals[:, 7])
            top1_details = top1_details.at[:, 15].set(metric_vals[:, 9])
            top1_details = top1_details.at[:, 16].set(metric_vals[:, 10])
            top1_locs = jnp.full((1, 3), -1, dtype=jnp.int32)
            local_diag_out = (
                metric_vals.astype(jnp.float32), metric_locs,
                top1_details, top1_locs)
        if not analysis:
            return slim_out + conc_out + prune_out + local_diag_out

        # --- Analysis-only extras ---
        phi_binary_frac = jax.lax.psum(total_phi_binary, 'model') / N_total
        phi_binary_frac = jax.lax.stop_gradient(phi_binary_frac)
        # Safety floor: active can collapse to 0 at init; clamp to 1.0.
        _active_denom = jnp.maximum(global_active_m, 1.0)
        z_lt_075_frac = jax.lax.stop_gradient(
            (jax.lax.psum(total_z_lt_075, 'model') / _active_denom).mean())
        z_lt_030_frac = jax.lax.stop_gradient(
            (jax.lax.psum(total_z_lt_030, 'model') / _active_denom).mean())
        active_per_token_std = global_active_m.std()
        global_g_log_g = jax.lax.stop_gradient(
            jax.lax.psum(total_g_log_g, 'model'))
        gate_sum_eps = jnp.maximum(global_weighted_cost_m, 1e-6)
        safe_glogg = jnp.where(
            global_weighted_cost_m > 1e-6, global_g_log_g, 0.0)
        entropy_per_token = -safe_glogg / gate_sum_eps + jnp.log(gate_sum_eps)
        entropy_per_token = jnp.where(
            jnp.isfinite(entropy_per_token), entropy_per_token, 0.0)
        gate_entropy = entropy_per_token.mean()
        den_cost_out = global_den_cost_m.mean()
        activation_cost_out = global_activation_cost_m.mean()
        current_cost_out = global_current_cost_m.mean()
        int_cap_frac_out = jax.lax.stop_gradient(
            jax.lax.psum(total_int_cap_count, 'model')
            / jnp.float32(B * S * N_total))
        # local_diag_out is collected inline in pass 2 above; do not run a
        # diagnostics-only chunk replay here.
        if False and _local_diagnostics:
            data_axis = jax.lax.axis_index('data')
            model_axis = jax.lax.axis_index('model')
            global_b_base = data_axis * B
            global_n_base = model_axis * N_local
            neg_inf = jnp.float32(-1.0e30)

            def _globalize(value, loc):
                value = jax.lax.stop_gradient(value)
                loc = jax.lax.stop_gradient(loc.astype(jnp.float32))
                for axis in ('model', 'data'):
                    gvalue = jax.lax.pmax(value, axis)
                    mask = (value == gvalue).astype(jnp.float32)
                    denom = jax.lax.psum(mask, axis)
                    loc = (
                        jax.lax.psum(loc * mask[..., None], axis)
                        / jnp.maximum(denom[..., None], 1.0))
                    value = gvalue
                return value, loc

            def _globalize_top1(value, loc, details):
                value = jax.lax.stop_gradient(value)
                loc = jax.lax.stop_gradient(loc.astype(jnp.float32))
                details = jax.lax.stop_gradient(details.astype(jnp.float32))
                for axis in ('model', 'data'):
                    gvalue = jax.lax.pmax(value, axis)
                    mask = (value == gvalue).astype(jnp.float32)
                    denom = jax.lax.psum(mask, axis)
                    loc = (
                        jax.lax.psum(loc * mask[..., None], axis)
                        / jnp.maximum(denom[..., None], 1.0))
                    details = (
                        jax.lax.psum(details * mask[..., None], axis)
                        / jnp.maximum(denom[..., None], 1.0))
                    value = gvalue
                return value, loc, details

            def _token_route_max(vals):
                # vals [B,S,1]; returns [1], [1,3].
                flat = vals[:, :, 0].reshape((1, B * S))
                arg = jnp.argmax(flat, axis=1).astype(jnp.int32)
                val = jnp.take_along_axis(flat, arg[:, None], axis=1)[:, 0]
                b = arg // S + global_b_base
                t = arg % S
                n = jnp.full_like(b, -1)
                return val, jnp.stack([b, t, n], axis=-1)

            def _chunk_route_max(vals, start):
                # vals [B,S,1,cs]; returns [1], [1,3].
                flat = vals[:, :, 0, :].reshape((1, B * S * cs))
                arg = jnp.argmax(flat, axis=1).astype(jnp.int32)
                val = jnp.take_along_axis(flat, arg[:, None], axis=1)[:, 0]
                b = arg // (S * cs) + global_b_base
                rem = arg % (S * cs)
                t = rem // cs
                n = rem % cs + start + global_n_base
                return val, jnp.stack([b, t, n], axis=-1)

            def _chunk_route_take(vals, arg):
                flat = vals[:, :, 0, :].reshape((1, B * S * cs))
                return jnp.take_along_axis(flat, arg[:, None], axis=1)[:, 0]

            def _local_diag_step(carry, i):
                (metric_vals, metric_locs, top1_vals, top1_locs,
                 top1_details) = carry
                s = i * cs
                route, rc, wc = route_rw_chunk(s)
                scores_base, intensity_base = route_relation_and_intensity(h_bf, route)
                scores_f = scores_base[:, :, None, :]
                intensity = intensity_base[:, :, None, :]
                tau_r = tau[:, :, None, :]
                raw = scores_f - tau_r
                gate_raw = raw - _act_thr
                z_act = jnp.clip(raw / _act_thr, 0.0, 1.0)
                activation = jnp.power(z_act, _activation_power)
                base_gate = activation * intensity
                if _prune_enabled:
                    keep = activation > _prune_thr
                    gate = jnp.where(keep, base_gate, 0.0)
                else:
                    gate = base_gate
                xr = (x_bf @ rc.T).astype(jnp.float32)[:, :, None, :]
                read_norm = jnp.linalg.norm(rc.astype(jnp.float32), axis=-1)
                write_norm = jnp.linalg.norm(wc.astype(jnp.float32), axis=-1)
                read_norm_b = read_norm[None, None, None, :]
                write_norm_b = write_norm[None, None, None, :]
                op_gain = read_norm_b * write_norm_b
                gate_den = jnp.maximum(global_weighted_cost_m[:, :, None, :], 1e-8)
                out_den = jnp.maximum(global_den_cost_m[:, :, None, :], 1.0)
                top1_share = gate / gate_den
                contrib_norm = jnp.abs((gate / out_den) * xr) * write_norm_b

                chunk_gate_raw, chunk_gate_raw_loc = _chunk_route_max(raw, s)
                chunk_top1, chunk_top1_loc = _chunk_route_max(top1_share, s)
                chunk_int, chunk_int_loc = _chunk_route_max(intensity, s)
                chunk_read, chunk_read_loc = _chunk_route_max(jnp.abs(xr), s)
                chunk_contrib, chunk_contrib_loc = _chunk_route_max(contrib_norm, s)
                chunk_vals = jnp.stack([
                    chunk_gate_raw, chunk_top1, chunk_int,
                    chunk_read, chunk_contrib,
                ], axis=1)
                chunk_locs = jnp.stack([
                    chunk_gate_raw_loc, chunk_top1_loc, chunk_int_loc,
                    chunk_read_loc, chunk_contrib_loc,
                ], axis=1)
                slots = jnp.array([1, 2, 4, 5, 6], dtype=jnp.int32)
                old_vals = metric_vals[:, slots]
                old_locs = metric_locs[:, slots, :]
                take = chunk_vals > old_vals
                metric_vals = metric_vals.at[:, slots].set(
                    jnp.where(take, chunk_vals, old_vals))
                metric_locs = metric_locs.at[:, slots, :].set(
                    jnp.where(take[..., None], chunk_locs, old_locs))

                flat_top1 = top1_share[:, :, 0, :].reshape((1, B * S * cs))
                arg = jnp.argmax(flat_top1, axis=1).astype(jnp.int32)
                chunk_top1_val = jnp.take_along_axis(
                    flat_top1, arg[:, None], axis=1)[:, 0]
                chunk_score = _chunk_route_take(scores_f, arg)
                chunk_tau = _chunk_route_take(tau_r + jnp.zeros_like(scores_f), arg)
                chunk_margin = chunk_score - chunk_tau
                chunk_gate_raw = _chunk_route_take(gate_raw, arg)
                chunk_gate = _chunk_route_take(gate, arg)
                chunk_gate_den = _chunk_route_take(
                    gate_den + jnp.zeros_like(gate), arg)
                chunk_intensity = _chunk_route_take(intensity, arg)
                chunk_xr = _chunk_route_take(xr, arg)
                chunk_write_norm = _chunk_route_take(
                    write_norm_b + jnp.zeros_like(gate), arg)
                chunk_read_norm = _chunk_route_take(
                    read_norm_b + jnp.zeros_like(gate), arg)
                chunk_op_gain = _chunk_route_take(op_gain + jnp.zeros_like(gate), arg)
                chunk_contrib = _chunk_route_take(contrib_norm, arg)
                out_norm_r = jnp.linalg.norm(out[:, :, None, :], axis=-1)
                chunk_out_norm = _chunk_route_take(
                    out_norm_r[..., None] + jnp.zeros_like(gate), arg)
                chunk_contrib_frac = chunk_contrib / jnp.maximum(chunk_out_norm, 1e-8)
                chunk_details = jnp.stack([
                    chunk_top1_val, chunk_score, chunk_tau, chunk_margin,
                    chunk_gate_raw, chunk_gate / jnp.maximum(chunk_gate_den, 1e-8),
                    chunk_gate_den, chunk_intensity, chunk_xr,
                    chunk_write_norm, chunk_read_norm, chunk_op_gain,
                    chunk_contrib, chunk_out_norm, chunk_contrib_frac,
                ], axis=1)
                take_top1 = chunk_top1_val > top1_vals
                top1_vals = jnp.where(take_top1, chunk_top1_val, top1_vals)
                top1_locs = jnp.where(take_top1[:, None],
                                      chunk_top1_loc, top1_locs)
                top1_details = jnp.where(take_top1[:, None],
                                         chunk_details, top1_details)
                return (metric_vals, metric_locs, top1_vals, top1_locs,
                        top1_details), None

            metric_vals0 = jnp.full((1, LOCAL_SPIKE_METRIC_COUNT), neg_inf)
            metric_locs0 = jnp.zeros((1, LOCAL_SPIKE_METRIC_COUNT, 3),
                                     dtype=jnp.int32)
            top1_vals0 = jnp.full((1,), neg_inf)
            top1_locs0 = jnp.zeros((1, 3), dtype=jnp.int32)
            top1_details0 = jnp.zeros((1, LOCAL_SPIKE_TOP1_COUNT),
                                      dtype=jnp.float32)
            (metric_vals, metric_locs, top1_vals, top1_locs,
             top1_details), _ = jax.lax.scan(
                _local_diag_step,
                (metric_vals0, metric_locs0, top1_vals0, top1_locs0,
                 top1_details0),
                jnp.arange(nc))

            tau_val, tau_loc = _token_route_max(jnp.abs(tau_offset[:, :, None, 0]))
            den_val, den_loc = _token_route_max(global_den_cost_m[:, :, None, 0])
            out_val, out_loc = _token_route_max(
                jnp.linalg.norm(out[:, :, None, :], axis=-1))
            resid_val, resid_loc = _token_route_max(
                jnp.linalg.norm(x, axis=-1)[:, :, None])
            token_vals = jnp.stack([tau_val, den_val, out_val, resid_val], axis=1)
            token_locs = jnp.stack([tau_loc, den_loc, out_loc, resid_loc], axis=1)
            token_slots = jnp.array([0, 3, 7, 8], dtype=jnp.int32)
            metric_vals = metric_vals.at[:, token_slots].set(token_vals)
            metric_locs = metric_locs.at[:, token_slots, :].set(token_locs)

            metric_vals, metric_locs = _globalize(metric_vals, metric_locs)
            top1_vals, top1_locs, top1_details = _globalize_top1(
                top1_vals, top1_locs, top1_details)
            local_diag_out = (metric_vals, metric_locs,
                              top1_details, top1_locs)
        return slim_out + conc_out + (phi_binary_frac, z_lt_075_frac, z_lt_030_frac,
                           score_skew, active_per_token_std, gate_entropy,
                           den_cost_out, activation_cost_out, current_cost_out,
                           score_kurt, int_cap_frac_out) + prune_out + local_diag_out

    return fused_gate_srw


def make_sharded_srw_paired(mesh, max_chunk_size=2048, dead_threshold=0.01,
                            sharpness=SHARPNESS,
                            activation_threshold=ACTIVATION_THRESHOLD,
                            activation_cutoff=ACTIVATION_CUTOFF,
                            epsilon=EPSILON,
                            max_intensity=MAX_INTENSITY,
                            scan_scale=SCAN_SCALE,
                            scan_std_floor=SCAN_STD_FLOOR,
                            analysis=False,
                            prune_enabled=False,
                            prune_activation_threshold=None,
                            prune_scope="all",
                            prune_denominator="pruned",
                            return_prune_stats=False,
                            local_diagnostics=False,
                            route_emb_forward_norm=False,
                            intensity_route_dim=0,
                            intensity_beta=0.5,
                            intensity_squash="tanh",
                            intensity_width=1.0,
                            activation_power=1.0):
    """Fused Q+K shard_map: two routes sharing same pool in one shard_map call.

    h is [B,S,2,d_route] (h_Q, h_K stacked on axis=2).
    tau_offset and raw_scan_offset are [B,S,2,1].
    x @ read.T computed once (shared by both routes).
    Scores stats computed independently per route.
    Returns out [B,S,2,D], active [B,S,1], gate_max [B,S,1].

    v4.1 gate: activation * intensity (see make_sharded_srw docstring).
    analysis/local_diagnostics: see make_sharded_srw docstring.
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
    _prune_enabled = bool(prune_enabled)
    if _prune_enabled and prune_activation_threshold is None:
        raise ValueError(
            "prune_activation_threshold must be set when prune_enabled=True")
    if prune_denominator not in ("pruned", "retained", "full"):
        raise ValueError(
            "prune_denominator must be 'pruned', 'retained', or 'full', "
            f"got {prune_denominator!r}")
    # Scope is resolved by the caller that chooses which pool closures prune.
    _ = prune_scope
    _prune_thr = jnp.float32(
        0.0 if prune_activation_threshold is None
        else prune_activation_threshold)
    _denominator_is_full = (prune_denominator == "full")
    _return_prune_stats = bool(return_prune_stats)
    _local_diagnostics = bool(local_diagnostics)
    _route_emb_forward_norm = bool(route_emb_forward_norm)
    _intensity_route_dim = int(intensity_route_dim or 0)
    _two_channel_intensity = _intensity_route_dim > 0
    _intensity_beta = jnp.float32(intensity_beta)
    _intensity_squash = str(intensity_squash).lower()
    _intensity_width = jnp.float32(intensity_width)
    _activation_power = jnp.float32(activation_power)
    _int_cap_thresh_paired = _eps + _max_int - jnp.float32(1e-3)

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
    _prune_extra_specs = (
        P(),                          # kept_count_mean scalar
        P(),                          # kept_frac_mean scalar
        P(),                          # full_gate_sum_mean scalar
        P(),                          # kept_gate_sum_mean scalar
        P(),                          # retained_gate_mass_mean scalar
        P(),                          # int_cap_frac scalar
        P(),                          # gate_max_mean scalar
    )
    _local_diag_specs = (
        P(),                          # local_spike_values [2, metric]
        P(),                          # local_spike_locs [2, metric, b/t/neuron]
        P(),                          # top1_breakdown_values [2, field]
        P(),                          # top1_breakdown_locs [2, b/t/neuron]
    )
    _out_specs = (_slim_out_specs + _conc_out_specs + _analysis_extra_specs
                  if analysis else _slim_out_specs + _conc_out_specs)
    if _return_prune_stats:
        _out_specs = _out_specs + _prune_extra_specs
    if _local_diagnostics:
        _out_specs = _out_specs + _local_diag_specs

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
        diag_neg_inf = jnp.float32(-1.0e30)
        diag_vals_init = jnp.full(
            (2, LOCAL_SPIKE_METRIC_COUNT), diag_neg_inf)

        def route_emb_chunk(start):
            ec = jax.lax.dynamic_slice_in_dim(emb_bf, start, cs, axis=0)
            if _route_emb_forward_norm:
                ec_f = ec.astype(jnp.float32)
                ec = _forward_unit_direction(ec_f).astype(jnp.bfloat16)
            return ec

        def route_relation_and_intensity(h_in, route):
            # Two-channel mode for paired Q/K routes.  Selection uses a
            # normalized relation space; intensity is a positive mixture
            # reweighting computed only from the reserved intensity dims.
            if _two_channel_intensity:
                d_total = h_in.shape[-1]
                d_str = min(max(_intensity_route_dim, 1), d_total - 1)
                d_sel = d_total - d_str
                h_sel = h_in[..., :d_sel].astype(jnp.float32)
                route_sel = route[:, :d_sel].astype(jnp.float32)
                h_sel = _forward_unit_direction(h_sel).astype(jnp.bfloat16)
                route_sel = _forward_unit_direction(route_sel).astype(jnp.bfloat16)
                relation = jnp.einsum('bsrd,nd->bsrn', h_sel, route_sel).astype(jnp.float32)

                h_str = h_in[..., d_sel:].astype(jnp.bfloat16)
                route_str = route[:, d_sel:].astype(jnp.bfloat16)
                intensity_raw = jnp.einsum('bsrd,nd->bsrn', h_str, route_str).astype(jnp.float32)
                z = intensity_raw / jnp.maximum(_intensity_width, jnp.float32(1e-6))
                if _intensity_squash == "softsign":
                    intensity_log = _intensity_beta * (z / (jnp.float32(1.0) + jnp.abs(z)))
                else:
                    intensity_log = _intensity_beta * jnp.tanh(z)
                intensity = jnp.exp(intensity_log)
                return relation, intensity

            scores = jnp.einsum('bsrd,nd->bsrn', h_in, route).astype(jnp.float32)
            intensity = jnp.minimum(jnp.maximum(scores, 0.0), _max_int)
            return scores, intensity

        def route_rw_chunk(start):
            ec = route_emb_chunk(start)
            rc = jax.lax.dynamic_slice_in_dim(read_bf, start, cs, axis=0)
            wc = jax.lax.dynamic_slice_in_dim(write_bf, start, cs, axis=0)
            rc_f = rc.astype(jnp.float32)
            wc_f = wc.astype(jnp.float32)
            # v4.1.5.9: stored read/write params stay raw, but SRW
            # execution uses their directions.
            rc_dir = _forward_unit_direction(rc_f)
            wc_dir = _forward_unit_direction(wc_f)
            return ec, rc_dir.astype(jnp.bfloat16), wc_dir.astype(jnp.bfloat16)

        # --- Pass 1: exact stats over ALL chunks (scan + checkpoint) ---
        if analysis:
            @jax.checkpoint
            def stats_step(carry, i):
                s_sum, sq_sum, cube_sum, quad_sum, ns_sum, ns_sq = carry
                s = i * cs
                route = route_emb_chunk(s)
                scores_f, _ = route_relation_and_intensity(h_bf, route)
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
                route = route_emb_chunk(s)
                scores_f, _ = route_relation_and_intensity(h_bf, route)
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
        s_var = jnp.maximum(global_sq / N_total - s_mean ** 2, 0.0)
        s_std = jnp.sqrt(s_var) + 1e-8
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
        var_score = jnp.maximum(global_ns_sq / N_total - mean_score ** 2, 0.0)
        score_lb = var_score / (mean_score ** 2 + var_score + 1e-2)

        # --- Pass 2: gate + srw fused ---
        if analysis:
            @jax.checkpoint
            def gate_srw_step(carry, i):
                (out, total_weighted_cost, total_gate_sq, total_gate_max, total_active,
                 total_strong, total_phi_binary, total_den_cost,
                 total_activation_cost, total_current_cost,
                 total_z_lt_075, total_z_lt_030, total_g_log_g,
                 total_dead_penalty, total_dead_count,
                 total_int_max, total_full_gate, total_kept_count,
                 total_int_cap_count, diag_vals) = carry
                s = i * cs
                route, rc, wc = route_rw_chunk(s)
                scores_f, intensity = route_relation_and_intensity(h_bf, route)
                raw = scores_f - tau
                margin = raw - _act_thr
                z_act = jnp.clip(raw / _act_thr, 0.0, 1.0)
                activation = jnp.power(z_act, _activation_power)
                base_gate = activation * intensity
                if _prune_enabled:
                    keep = activation > _prune_thr
                    gate = jnp.where(keep, base_gate, 0.0)
                    chunk_kept = keep.astype(jnp.float32).sum(
                        axis=-1, keepdims=True)
                else:
                    gate = base_gate
                    chunk_kept = jnp.full((B, S, 2, 1), cs, dtype=jnp.float32)
                chunk_int_max = intensity.max()
                chunk_int_cap_count = (intensity >= _int_cap_thresh_paired
                                        ).astype(jnp.float32).sum()
                xr = x_bf @ rc.T  # [B,S,N]
                xr_f = xr.astype(jnp.float32)
                a = gate * xr_f[:, :, None, :]
                c_out = jnp.einsum('bsrn,nd->bsrd', a.astype(jnp.bfloat16), wc).astype(jnp.float32)
                chunk_weighted = gate.sum(axis=-1, keepdims=True)           # [B,S,2,1]
                chunk_gate_sq = jnp.square(gate).sum(axis=-1, keepdims=True)
                chunk_full_gate = (
                    base_gate.sum(axis=-1, keepdims=True)
                    if (_return_prune_stats or _denominator_is_full)
                    else chunk_weighted)
                chunk_den_cost = (chunk_full_gate if _denominator_is_full
                                  else chunk_weighted)
                if _local_diagnostics:
                    write_norm = jnp.linalg.norm(
                        wc.astype(jnp.float32), axis=-1)
                    xr_r = xr_f[:, :, None, :]
                    contrib_proxy = (
                        jnp.abs(jax.lax.stop_gradient(gate * xr_r))
                        * write_norm[None, None, None, :])
                    diag_chunk = jnp.full_like(diag_vals, diag_neg_inf)
                    diag_chunk = diag_chunk.at[:, 1].set(
                        jnp.max(jax.lax.stop_gradient(margin),
                                axis=(0, 1, 3)))
                    diag_chunk = diag_chunk.at[:, 4].set(
                        jnp.max(jax.lax.stop_gradient(intensity),
                                axis=(0, 1, 3)))
                    diag_chunk = diag_chunk.at[:, 5].set(
                        jnp.max(jnp.abs(jax.lax.stop_gradient(
                            xr_r + jnp.zeros_like(gate))), axis=(0, 1, 3)))
                    diag_chunk = diag_chunk.at[:, 6].set(
                        jnp.max(contrib_proxy, axis=(0, 1, 3)))
                    _route_norm_max = jnp.max(jnp.linalg.norm(
                        jax.lax.stop_gradient(route.astype(jnp.float32)),
                        axis=-1))
                    diag_chunk = diag_chunk.at[:, 10].set(
                        jnp.repeat(_route_norm_max, 2))
                    diag_vals = jnp.maximum(diag_vals, diag_chunk)
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
                        total_den_cost + chunk_den_cost,
                        total_activation_cost,
                        total_current_cost,
                        total_z_lt_075 + chunk_z_lt_075,
                        total_z_lt_030 + chunk_z_lt_030,
                        total_g_log_g + chunk_g_log_g,
                        total_dead_penalty + chunk_dead_penalty,
                        total_dead_count + chunk_dead_count,
                        jnp.maximum(total_int_max, chunk_int_max),
                        total_full_gate + chunk_full_gate,
                        total_kept_count + chunk_kept,
                        total_int_cap_count + chunk_int_cap_count,
                        diag_vals), None

            (raw_out, total_weighted_cost, total_gate_sq, total_gate_max, total_active, total_strong,
             total_phi_binary, total_den_cost, total_activation_cost,
             total_current_cost, total_z_lt_075, total_z_lt_030,
             total_g_log_g, total_dead_penalty, total_dead_count,
             total_int_max, total_full_gate, total_kept_count,
             total_int_cap_count, diag_vals), _ = jax.lax.scan(
                gate_srw_step,
                (jnp.zeros((B, S, 2, D), dtype=jnp.float32),
                 z1_r, z1_r, jnp.full((B, S, 2, 1), -1e9),
                 z1_r, z1_r, z1_r, z1_r, z1_r, z1_r, z1_r, z1_r, z1_r,
                 jnp.float32(0.0), jnp.float32(0.0),
                 jnp.float32(0.0), z1_r, z1_r, jnp.float32(0.0),
                 diag_vals_init),
                jnp.arange(nc))
        else:
            @jax.checkpoint
            def gate_srw_step(carry, i):
                (out, total_weighted_cost, total_gate_sq, total_gate_max, total_active,
                 total_strong, total_den_cost,
                 total_activation_cost, total_current_cost,
                 total_dead_penalty, total_dead_count,
                 total_int_max, total_full_gate, total_kept_count,
                 total_int_cap_count, diag_vals) = carry
                s = i * cs
                route, rc, wc = route_rw_chunk(s)
                scores_f, intensity = route_relation_and_intensity(h_bf, route)
                raw = scores_f - tau
                margin = raw - _act_thr
                z_act = jnp.clip(raw / _act_thr, 0.0, 1.0)
                activation = jnp.power(z_act, _activation_power)
                base_gate = activation * intensity
                if _prune_enabled:
                    keep = activation > _prune_thr
                    gate = jnp.where(keep, base_gate, 0.0)
                    chunk_kept = keep.astype(jnp.float32).sum(
                        axis=-1, keepdims=True)
                else:
                    gate = base_gate
                    chunk_kept = jnp.full((B, S, 2, 1), cs, dtype=jnp.float32)
                chunk_int_max = intensity.max()
                chunk_int_cap_count = (intensity >= _int_cap_thresh_paired
                                        ).astype(jnp.float32).sum()
                xr = x_bf @ rc.T
                xr_f = xr.astype(jnp.float32)
                a = gate * xr_f[:, :, None, :]
                c_out = jnp.einsum('bsrn,nd->bsrd', a.astype(jnp.bfloat16), wc).astype(jnp.float32)
                chunk_weighted = gate.sum(axis=-1, keepdims=True)
                chunk_gate_sq = jnp.square(gate).sum(axis=-1, keepdims=True)
                chunk_full_gate = base_gate.sum(axis=-1, keepdims=True)
                chunk_den_cost = (chunk_full_gate if _denominator_is_full
                                  else chunk_weighted)
                if _local_diagnostics:
                    write_norm = jnp.linalg.norm(
                        wc.astype(jnp.float32), axis=-1)
                    xr_r = xr_f[:, :, None, :]
                    contrib_proxy = (
                        jnp.abs(jax.lax.stop_gradient(gate * xr_r))
                        * write_norm[None, None, None, :])
                    diag_chunk = jnp.full_like(diag_vals, diag_neg_inf)
                    diag_chunk = diag_chunk.at[:, 1].set(
                        jnp.max(jax.lax.stop_gradient(margin),
                                axis=(0, 1, 3)))
                    diag_chunk = diag_chunk.at[:, 4].set(
                        jnp.max(jax.lax.stop_gradient(intensity),
                                axis=(0, 1, 3)))
                    diag_chunk = diag_chunk.at[:, 5].set(
                        jnp.max(jnp.abs(jax.lax.stop_gradient(
                            xr_r + jnp.zeros_like(gate))), axis=(0, 1, 3)))
                    diag_chunk = diag_chunk.at[:, 6].set(
                        jnp.max(contrib_proxy, axis=(0, 1, 3)))
                    _route_norm_max = jnp.max(jnp.linalg.norm(
                        jax.lax.stop_gradient(route.astype(jnp.float32)),
                        axis=-1))
                    diag_chunk = diag_chunk.at[:, 10].set(
                        jnp.repeat(_route_norm_max, 2))
                    diag_vals = jnp.maximum(diag_vals, diag_chunk)
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
                        total_den_cost + chunk_den_cost,
                        total_activation_cost,
                        total_current_cost,
                        total_dead_penalty + chunk_dead_penalty,
                        total_dead_count + chunk_dead_count,
                        jnp.maximum(total_int_max, chunk_int_max),
                        total_full_gate + chunk_full_gate,
                        total_kept_count + chunk_kept,
                        total_int_cap_count + chunk_int_cap_count,
                        diag_vals), None

            (raw_out, total_weighted_cost, total_gate_sq, total_gate_max, total_active, total_strong,
             total_den_cost, total_activation_cost, total_current_cost,
             total_dead_penalty, total_dead_count,
             total_int_max, total_full_gate, total_kept_count,
             total_int_cap_count, diag_vals), _ = jax.lax.scan(
                gate_srw_step,
                (jnp.zeros((B, S, 2, D), dtype=jnp.float32),
                 z1_r, z1_r, jnp.full((B, S, 2, 1), -1e9),
                 z1_r, z1_r, z1_r, z1_r, z1_r,
                 jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0),
                 z1_r, z1_r, jnp.float32(0.0), diag_vals_init),
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
        # Measurement path: detached copies for diagnostics / feedback refs.
        # Action path above keeps global_den_cost/global_weighted_cost live for
        # the SRW denominator and output gradient.
        global_weighted_cost_m = jax.lax.stop_gradient(global_weighted_cost)
        global_gate_sq_m = jax.lax.stop_gradient(global_gate_sq)
        global_den_cost_m = jax.lax.stop_gradient(global_den_cost)
        global_activation_cost_m = jax.lax.stop_gradient(global_activation_cost)
        global_current_cost_m = jax.lax.stop_gradient(global_current_cost)
        global_active_m = jax.lax.stop_gradient(global_active)
        global_strong_m = jax.lax.stop_gradient(
            jax.lax.psum(total_strong, 'model'))
        global_gate_max_m = jax.lax.stop_gradient(global_gate_max)
        active_frac = global_active_m / N_total
        active_frac_mean = active_frac.mean(axis=2)
        strong_frac = global_strong_m / N_total
        strong_frac_mean = strong_frac.mean(axis=2)
        z_mean_active = global_weighted_cost_m / (global_active_m + 1e-8)
        z_mean_active_mean = z_mean_active.mean(axis=2)
        raw_gate_max_mean = global_gate_max_m.mean(axis=2)

        score_std_out = jax.lax.stop_gradient(s_std.mean())
        es_out = global_weighted_cost_m.mean()
        active_n_mean = global_active_m.mean()
        gate_eff_n = ((global_weighted_cost_m ** 2)
                      / (global_gate_sq_m + 1e-8))
        gate_eff_ratio = gate_eff_n / jnp.maximum(global_active_m, 1.0)
        top1_gate_frac = global_gate_max_m / jnp.maximum(
            global_weighted_cost_m, 1e-8)
        tau_abs_mean = jax.lax.stop_gradient(tau).mean()
        dead_penalty_out = jax.lax.psum(total_dead_penalty, 'model')
        dead_count_out = jax.lax.stop_gradient(
            jax.lax.psum(total_dead_count, 'model'))
        int_max_out = jax.lax.pmax(
            jax.lax.stop_gradient(total_int_max), 'model')

        den_cost_mean = global_den_cost_m.mean()
        activation_cost_mean = global_activation_cost_m.mean()
        current_cost_mean = global_current_cost_m.mean()

        slim_out = (out.astype(jnp.float32), active_frac_mean, raw_gate_max_mean, score_lb,
                    score_std_out, es_out, active_n_mean, strong_frac_mean,
                    z_mean_active_mean, tau_abs_mean, dead_penalty_out, dead_count_out,
                    int_max_out, den_cost_mean, activation_cost_mean, current_cost_mean)
        conc_out = (gate_eff_n.mean(), gate_eff_ratio.mean(),
                    top1_gate_frac.mean(), top1_gate_frac.max())
        prune_out = ()
        if _return_prune_stats:
            global_full_gate = jax.lax.psum(total_full_gate, 'model')
            global_kept_count = jax.lax.psum(total_kept_count, 'model')
            global_full_gate_m = jax.lax.stop_gradient(global_full_gate)
            global_kept_count_m = jax.lax.stop_gradient(global_kept_count)
            retained_gate_mass = (
                global_weighted_cost_m / jnp.maximum(global_full_gate_m, 1e-8))
            int_cap_frac_out = jax.lax.stop_gradient(
                jax.lax.psum(total_int_cap_count, 'model')
                / jnp.float32(B * S * 2 * N_total))
            prune_out = (
                global_kept_count_m.mean(),
                (global_kept_count_m / N_total).mean(),
                global_full_gate_m.mean(),
                global_weighted_cost_m.mean(),
                retained_gate_mass.mean(),
                int_cap_frac_out,
                global_gate_max_m.mean(),
            )
        local_diag_out = ()
        if _local_diagnostics:
            tau_abs_max = jnp.max(
                jnp.abs(jax.lax.stop_gradient(tau_offset[..., 0])),
                axis=(0, 1))
            top1_share_max = jnp.max(
                global_gate_max_m / jnp.maximum(global_den_cost_m, 1e-8),
                axis=(0, 1, 3))
            gate_den_sum_max = jnp.max(global_den_cost_m, axis=(0, 1, 3))
            local_out_norm_max = jnp.max(
                jnp.linalg.norm(jax.lax.stop_gradient(out), axis=-1),
                axis=(0, 1))
            residual_norm_max = jnp.repeat(
                jnp.max(jnp.linalg.norm(
                    jax.lax.stop_gradient(x), axis=-1)),
                2)
            h_norm_max = jnp.max(jnp.linalg.norm(
                jax.lax.stop_gradient(h), axis=-1), axis=(0, 1))
            token_vals = jnp.stack([
                tau_abs_max, top1_share_max, gate_den_sum_max,
                local_out_norm_max, residual_norm_max, h_norm_max,
            ], axis=1)
            token_slots = jnp.array([0, 2, 3, 7, 8, 9], dtype=jnp.int32)
            metric_vals = diag_vals.at[:, token_slots].set(token_vals)
            metric_vals = jax.lax.stop_gradient(metric_vals)
            metric_vals = jax.lax.pmax(metric_vals, 'model')
            metric_vals = jax.lax.pmax(metric_vals, 'data')
            metric_locs = jnp.full(
                (2, LOCAL_SPIKE_METRIC_COUNT, 3), -1, dtype=jnp.int32)
            top1_details = jnp.zeros(
                (2, LOCAL_SPIKE_TOP1_COUNT), dtype=jnp.float32)
            top1_details = top1_details.at[:, 0].set(metric_vals[:, 2])
            top1_details = top1_details.at[:, 4].set(metric_vals[:, 1])
            top1_details = top1_details.at[:, 6].set(metric_vals[:, 3])
            top1_details = top1_details.at[:, 7].set(metric_vals[:, 4])
            top1_details = top1_details.at[:, 8].set(metric_vals[:, 5])
            top1_details = top1_details.at[:, 12].set(metric_vals[:, 6])
            top1_details = top1_details.at[:, 13].set(metric_vals[:, 7])
            top1_details = top1_details.at[:, 15].set(metric_vals[:, 9])
            top1_details = top1_details.at[:, 16].set(metric_vals[:, 10])
            top1_locs = jnp.full((2, 3), -1, dtype=jnp.int32)
            local_diag_out = (
                metric_vals.astype(jnp.float32), metric_locs,
                top1_details, top1_locs)
        if not analysis:
            return slim_out + conc_out + prune_out + local_diag_out

        # --- Analysis-only extras ---
        phi_binary_frac = jax.lax.psum(total_phi_binary, 'model') / N_total
        phi_binary_frac_mean = jax.lax.stop_gradient(phi_binary_frac).mean(axis=2)
        _active_denom = jnp.maximum(global_active_m, 1.0)
        z_lt_075_frac = jax.lax.stop_gradient(
            (jax.lax.psum(total_z_lt_075, 'model') / _active_denom).mean())
        z_lt_030_frac = jax.lax.stop_gradient(
            (jax.lax.psum(total_z_lt_030, 'model') / _active_denom).mean())
        active_per_token_std = global_active_m.std()
        global_g_log_g = jax.lax.stop_gradient(
            jax.lax.psum(total_g_log_g, 'model'))
        gate_sum_eps = jnp.maximum(global_weighted_cost_m, 1e-6)
        safe_glogg = jnp.where(
            global_weighted_cost_m > 1e-6, global_g_log_g, 0.0)
        entropy_per_token = -safe_glogg / gate_sum_eps + jnp.log(gate_sum_eps)
        entropy_per_token = jnp.where(
            jnp.isfinite(entropy_per_token), entropy_per_token, 0.0)
        gate_entropy = entropy_per_token.mean()
        den_cost_out = global_den_cost_m.mean()
        activation_cost_out = global_activation_cost_m.mean()
        current_cost_out = global_current_cost_m.mean()
        int_cap_frac_out = jax.lax.stop_gradient(
            jax.lax.psum(total_int_cap_count, 'model')
            / jnp.float32(B * S * 2 * N_total))
        # local_diag_out is collected inline in pass 2 above; do not run a
        # diagnostics-only chunk replay here.
        if False and _local_diagnostics:
            data_axis = jax.lax.axis_index('data')
            model_axis = jax.lax.axis_index('model')
            global_b_base = data_axis * B
            global_n_base = model_axis * N_local
            neg_inf = jnp.float32(-1.0e30)

            def _globalize(value, loc):
                value = jax.lax.stop_gradient(value)
                loc = jax.lax.stop_gradient(loc.astype(jnp.float32))
                for axis in ('model', 'data'):
                    gvalue = jax.lax.pmax(value, axis)
                    mask = (value == gvalue).astype(jnp.float32)
                    denom = jax.lax.psum(mask, axis)
                    loc = (
                        jax.lax.psum(loc * mask[..., None], axis)
                        / jnp.maximum(denom[..., None], 1.0))
                    value = gvalue
                return value, loc

            def _globalize_top1(value, loc, details):
                value = jax.lax.stop_gradient(value)
                loc = jax.lax.stop_gradient(loc.astype(jnp.float32))
                details = jax.lax.stop_gradient(details.astype(jnp.float32))
                for axis in ('model', 'data'):
                    gvalue = jax.lax.pmax(value, axis)
                    mask = (value == gvalue).astype(jnp.float32)
                    denom = jax.lax.psum(mask, axis)
                    loc = (
                        jax.lax.psum(loc * mask[..., None], axis)
                        / jnp.maximum(denom[..., None], 1.0))
                    details = (
                        jax.lax.psum(details * mask[..., None], axis)
                        / jnp.maximum(denom[..., None], 1.0))
                    value = gvalue
                return value, loc, details

            def _token_route_max(vals):
                # vals [B,S,2]; returns [2], [2,3].
                flat = vals.transpose(2, 0, 1).reshape((2, B * S))
                arg = jnp.argmax(flat, axis=1).astype(jnp.int32)
                val = jnp.take_along_axis(flat, arg[:, None], axis=1)[:, 0]
                b = arg // S + global_b_base
                t = arg % S
                n = jnp.full_like(b, -1)
                return val, jnp.stack([b, t, n], axis=-1)

            def _chunk_route_max(vals, start):
                # vals [B,S,2,cs]; returns [2], [2,3].
                flat = vals.transpose(2, 0, 1, 3).reshape((2, B * S * cs))
                arg = jnp.argmax(flat, axis=1).astype(jnp.int32)
                val = jnp.take_along_axis(flat, arg[:, None], axis=1)[:, 0]
                b = arg // (S * cs) + global_b_base
                rem = arg % (S * cs)
                t = rem // cs
                n = rem % cs + start + global_n_base
                return val, jnp.stack([b, t, n], axis=-1)

            def _chunk_route_take(vals, arg):
                flat = vals.transpose(2, 0, 1, 3).reshape((2, B * S * cs))
                return jnp.take_along_axis(flat, arg[:, None], axis=1)[:, 0]

            def _local_diag_step(carry, i):
                (metric_vals, metric_locs, top1_vals, top1_locs,
                 top1_details) = carry
                s = i * cs
                route, rc, wc = route_rw_chunk(s)
                scores_f, intensity = route_relation_and_intensity(h_bf, route)
                raw = scores_f - tau
                gate_raw = raw - _act_thr
                z_act = jnp.clip(raw / _act_thr, 0.0, 1.0)
                activation = jnp.power(z_act, _activation_power)
                base_gate = activation * intensity
                if _prune_enabled:
                    keep = activation > _prune_thr
                    gate = jnp.where(keep, base_gate, 0.0)
                else:
                    gate = base_gate
                xr = (x_bf @ rc.T).astype(jnp.float32)
                xr_r = xr[:, :, None, :]
                read_norm = jnp.linalg.norm(rc.astype(jnp.float32), axis=-1)
                write_norm = jnp.linalg.norm(wc.astype(jnp.float32), axis=-1)
                read_norm_b = read_norm[None, None, None, :]
                write_norm_b = write_norm[None, None, None, :]
                op_gain = read_norm_b * write_norm_b
                gate_den = jnp.maximum(global_weighted_cost_m, 1e-8)
                out_den = jnp.maximum(global_den_cost_m, 1.0)
                top1_share = gate / gate_den
                contrib_norm = jnp.abs((gate / out_den) * xr_r) * write_norm_b

                chunk_gate_raw, chunk_gate_raw_loc = _chunk_route_max(raw, s)
                chunk_top1, chunk_top1_loc = _chunk_route_max(top1_share, s)
                chunk_int, chunk_int_loc = _chunk_route_max(intensity, s)
                chunk_read, chunk_read_loc = _chunk_route_max(
                    jnp.abs(xr_r + jnp.zeros_like(gate)), s)
                chunk_contrib, chunk_contrib_loc = _chunk_route_max(contrib_norm, s)
                chunk_vals = jnp.stack([
                    chunk_gate_raw, chunk_top1, chunk_int,
                    chunk_read, chunk_contrib,
                ], axis=1)
                chunk_locs = jnp.stack([
                    chunk_gate_raw_loc, chunk_top1_loc, chunk_int_loc,
                    chunk_read_loc, chunk_contrib_loc,
                ], axis=1)
                slots = jnp.array([1, 2, 4, 5, 6], dtype=jnp.int32)
                old_vals = metric_vals[:, slots]
                old_locs = metric_locs[:, slots, :]
                take = chunk_vals > old_vals
                metric_vals = metric_vals.at[:, slots].set(
                    jnp.where(take, chunk_vals, old_vals))
                metric_locs = metric_locs.at[:, slots, :].set(
                    jnp.where(take[..., None], chunk_locs, old_locs))

                flat_top1 = top1_share.transpose(2, 0, 1, 3).reshape(
                    (2, B * S * cs))
                arg = jnp.argmax(flat_top1, axis=1).astype(jnp.int32)
                chunk_top1_val = jnp.take_along_axis(
                    flat_top1, arg[:, None], axis=1)[:, 0]
                chunk_score = _chunk_route_take(scores_f, arg)
                chunk_tau = _chunk_route_take(tau + jnp.zeros_like(scores_f), arg)
                chunk_margin = chunk_score - chunk_tau
                chunk_gate_raw = _chunk_route_take(gate_raw, arg)
                chunk_gate = _chunk_route_take(gate, arg)
                chunk_gate_den = _chunk_route_take(
                    gate_den + jnp.zeros_like(gate), arg)
                chunk_intensity = _chunk_route_take(intensity, arg)
                chunk_xr = _chunk_route_take(xr_r + jnp.zeros_like(gate), arg)
                chunk_write_norm = _chunk_route_take(
                    write_norm_b + jnp.zeros_like(gate), arg)
                chunk_read_norm = _chunk_route_take(
                    read_norm_b + jnp.zeros_like(gate), arg)
                chunk_op_gain = _chunk_route_take(op_gain + jnp.zeros_like(gate), arg)
                chunk_contrib = _chunk_route_take(contrib_norm, arg)
                out_norm_r = jnp.linalg.norm(out, axis=-1)
                chunk_out_norm = _chunk_route_take(
                    out_norm_r[..., None] + jnp.zeros_like(gate), arg)
                chunk_contrib_frac = chunk_contrib / jnp.maximum(chunk_out_norm, 1e-8)
                chunk_details = jnp.stack([
                    chunk_top1_val, chunk_score, chunk_tau, chunk_margin,
                    chunk_gate_raw, chunk_gate / jnp.maximum(chunk_gate_den, 1e-8),
                    chunk_gate_den, chunk_intensity, chunk_xr,
                    chunk_write_norm, chunk_read_norm, chunk_op_gain,
                    chunk_contrib, chunk_out_norm, chunk_contrib_frac,
                ], axis=1)
                take_top1 = chunk_top1_val > top1_vals
                top1_vals = jnp.where(take_top1, chunk_top1_val, top1_vals)
                top1_locs = jnp.where(take_top1[:, None],
                                      chunk_top1_loc, top1_locs)
                top1_details = jnp.where(take_top1[:, None],
                                         chunk_details, top1_details)
                return (metric_vals, metric_locs, top1_vals, top1_locs,
                        top1_details), None

            metric_vals0 = jnp.full((2, LOCAL_SPIKE_METRIC_COUNT), neg_inf)
            metric_locs0 = jnp.zeros((2, LOCAL_SPIKE_METRIC_COUNT, 3),
                                     dtype=jnp.int32)
            top1_vals0 = jnp.full((2,), neg_inf)
            top1_locs0 = jnp.zeros((2, 3), dtype=jnp.int32)
            top1_details0 = jnp.zeros((2, LOCAL_SPIKE_TOP1_COUNT),
                                      dtype=jnp.float32)
            (metric_vals, metric_locs, top1_vals, top1_locs,
             top1_details), _ = jax.lax.scan(
                _local_diag_step,
                (metric_vals0, metric_locs0, top1_vals0, top1_locs0,
                 top1_details0),
                jnp.arange(nc))

            tau_val, tau_loc = _token_route_max(jnp.abs(tau_offset[..., 0]))
            den_val, den_loc = _token_route_max(global_den_cost_m[..., 0])
            out_val, out_loc = _token_route_max(jnp.linalg.norm(out, axis=-1))
            resid_val, resid_loc = _token_route_max(
                jnp.repeat(jnp.linalg.norm(x, axis=-1)[:, :, None], 2, axis=2))
            token_vals = jnp.stack([tau_val, den_val, out_val, resid_val], axis=1)
            token_locs = jnp.stack([tau_loc, den_loc, out_loc, resid_loc], axis=1)
            token_slots = jnp.array([0, 3, 7, 8], dtype=jnp.int32)
            metric_vals = metric_vals.at[:, token_slots].set(token_vals)
            metric_locs = metric_locs.at[:, token_slots, :].set(token_locs)

            metric_vals, metric_locs = _globalize(metric_vals, metric_locs)
            top1_vals, top1_locs, top1_details = _globalize_top1(
                top1_vals, top1_locs, top1_details)
            local_diag_out = (metric_vals, metric_locs,
                              top1_details, top1_locs)
        return slim_out + conc_out + (phi_binary_frac_mean, z_lt_075_frac, z_lt_030_frac,
                           score_skew, active_per_token_std, gate_entropy,
                           den_cost_out, activation_cost_out, current_cost_out,
                           score_kurt, int_cap_frac_out) + prune_out + local_diag_out

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

        # Learned route embeddings keep norm as a routing-intensity DoF.
        self.attn_qk_emb = self.param('attn_qk_emb', unit_norm_init(), (self.n_qk, db))
        self.attn_v_emb = self.param('attn_v_emb', unit_norm_init(), (self.n_v, db))
        self.rst_emb = self.param('rst_emb', unit_norm_init(), (n_rst_eff, db))

        # Read vectors define what each neuron extracts from x.
        # Stored vectors remain raw; SRW forward uses their directions.
        self.attn_qk_read = self.param('attn_qk_read', unit_norm_init(), (self.n_qk, dm))
        self.attn_v_read = self.param('attn_v_read', unit_norm_init(), (self.n_v, dm))
        self.rst_read = self.param('rst_read', unit_norm_init(), (n_rst_eff, dm))

        # Write vectors define the output direction for each neuron.
        # Raw parameter norms are still observable diagnostics.
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
    tau_offset_init: float = -0.5
    tau_offset_init_attn: Optional[float] = None
    tau_offset_init_rst: Optional[float] = None

    def setup(self):
        db = self.d_route
        tau_attn_init = float(
            self.tau_offset_init if self.tau_offset_init_attn is None
            else self.tau_offset_init_attn)
        tau_rst_init = float(
            self.tau_offset_init if self.tau_offset_init_rst is None
            else self.tau_offset_init_rst)
        self.proj_attn = nn.Dense(db * 3, name='proj_attn')
        self.proj_rst = nn.Dense(db, name='proj_rst')
        self.tau_attn = nn.Dense(3, name='tau_attn',
            kernel_init=nn.initializers.zeros,
            bias_init=lambda k, s, d: jnp.full(s, tau_attn_init, d))
        self.tau_rst = nn.Dense(1, name='tau_rst',
            kernel_init=nn.initializers.zeros,
            bias_init=lambda k, s, d: jnp.full(s, tau_rst_init, d))
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
                  n_heads, d_model, n_layers,
                  router_dropout, dropout_rate, deterministic,
                  sharded_fns, analysis=False, return_prune_stats=False,
                  local_diagnostics=False, tau_offset_clip=None,
                  fixed_depth_pool_scale=False):
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

    # Raw signature params are passed into the sharded SRW closure.
    # The closure can forward-normalize them for routing stability while
    # retaining raw parameter norms as diagnostics.
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
    if tau_offset_clip is not None and tau_offset_clip > 0:
        tau_offset_clip_f = jnp.float32(tau_offset_clip)
        tau_all = jnp.clip(tau_all, -tau_offset_clip_f, tau_offset_clip_f)
    raw_scan_offset_all = x @ router_params['raw_scan_offset_attn']['kernel'] + router_params['raw_scan_offset_attn']['bias']
    if analysis:
        _tau_all_sg = jax.lax.stop_gradient(tau_all)
        attn_tau_std = _tau_all_sg.std(axis=(0, 1))  # [3] Q/K/V
        attn_tau_kernel_norm = jnp.sqrt(
            jnp.sum(jax.lax.stop_gradient(router_params['tau_attn']['kernel']) ** 2) + 1e-12)

    qk_scale, v_scale, _ = _effective_pool_output_scales(
        pool_params, d_model, n_layers, fixed_depth_pool_scale)

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
    if return_prune_stats:
        (qk_kept_count, qk_kept_frac, qk_full_gate_sum, qk_kept_gate_sum,
         qk_retained_gate_mass, qk_int_cap_frac,
         qk_gate_max_mean) = qk_ret[20:27]
    if analysis:
        (qk_phi_bin, qk_z075, qk_z030, qk_skew, qk_apt_std, qk_entropy,
         qk_den_cost, qk_activation_cost, qk_current_cost,
         qk_kurt, qk_int_cap) = qk_ret[20:31]
        qk_raw_norm = jnp.linalg.norm(QK_out, axis=-1).mean()
    if local_diagnostics:
        (qk_local_values, qk_local_locs,
         qk_top1_values, qk_top1_locs) = qk_ret[-4:]
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
    if return_prune_stats:
        (v_kept_count, v_kept_frac, v_full_gate_sum, v_kept_gate_sum,
         v_retained_gate_mass, v_int_cap_frac,
         v_gate_max_mean) = v_ret[20:27]
    if analysis:
        (v_phi_bin, v_z075, v_z030, v_skew, v_apt_std, v_entropy,
         v_den_cost, v_activation_cost, v_current_cost,
         v_kurt, v_int_cap) = v_ret[20:31]
        v_raw_norm = jnp.linalg.norm(V, axis=-1).mean()
    if local_diagnostics:
        (v_local_values, v_local_locs,
         v_top1_values, v_top1_locs) = v_ret[-4:]
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
        if analysis:
            scores_sg = jax.lax.stop_gradient(attn_scores.astype(jnp.float32))
            score_floor = jnp.finfo(scores_sg.dtype).min
            causal_4d = causal[None, None, :, :]
            causal_f = causal_4d.astype(jnp.float32)
            valid_count = causal_f.sum() * jnp.float32(B * n_heads)
            valid_scores = jnp.where(causal_4d, scores_sg, 0.0)
            attn_logit_mean = valid_scores.sum() / valid_count
            attn_logit_var = (
                jnp.where(causal_4d, scores_sg - attn_logit_mean, 0.0) ** 2
            ).sum() / valid_count
            attn_logit_std = jnp.sqrt(attn_logit_var + 1e-12)

            masked_scores = jnp.where(causal_4d, scores_sg, score_floor)
            attn_logit_max_dbg = jnp.max(masked_scores)

            attn_w_sg = jax.lax.stop_gradient(attn_w.astype(jnp.float32))
            softmax_top1 = jnp.max(attn_w_sg, axis=-1)
            softmax_top1_mean = softmax_top1.mean()
            softmax_top1_max = softmax_top1.max()

            top1_logits = jnp.max(masked_scores, axis=-1)
            top1_idx = jnp.argmax(masked_scores, axis=-1)
            attn_idx = jnp.arange(S)
            second_scores = jnp.where(
                attn_idx[None, None, None, :] == top1_idx[..., None],
                score_floor,
                masked_scores)
            top2_logits = jnp.max(second_scores, axis=-1)
            has_top2 = (jnp.arange(S) + 1) > 1
            top2_logits = jnp.where(
                has_top2[None, None, :], top2_logits, top1_logits)
            logit_gap = top1_logits - top2_logits
            logit_gap_mean = logit_gap.mean()
            logit_gap_max = logit_gap.max()

            entropy_terms = jnp.where(
                attn_w_sg > 0.0,
                attn_w_sg * jnp.log(jnp.maximum(attn_w_sg, 1e-30)),
                0.0)
            softmax_entropy = -jnp.sum(entropy_terms, axis=-1)
            softmax_entropy_mean = softmax_entropy.mean()
            softmax_entropy_min = softmax_entropy.min()
        elif local_diagnostics:
            attn_logit_max_dbg = jnp.max(attn_scores)
            softmax_top1_max = jnp.max(attn_w)
        attn_w = safe_dropout(attn_w, dropout_rate, deterministic, rng_drop)
        out_dbg = jnp.einsum('bhst,bhtd->bhsd', attn_w, V)
        if analysis:
            return (
                out_dbg,
                attn_logit_mean, attn_logit_std,
                attn_logit_max_dbg,
                softmax_top1_mean, softmax_top1_max,
                logit_gap_mean, logit_gap_max,
                softmax_entropy_mean, softmax_entropy_min,
            )
        if local_diagnostics:
            return out_dbg, attn_logit_max_dbg, softmax_top1_max
        return out_dbg

    if analysis or local_diagnostics:
        q_norms_dbg = jnp.linalg.norm(Q, axis=-1)
        k_norms_dbg = jnp.linalg.norm(K, axis=-1)
        v_norms_dbg = jnp.linalg.norm(V, axis=-1)
    if analysis:
        q_norm = q_norms_dbg.mean()
        q_norm_std = q_norms_dbg.std()
        q_norm_max = q_norms_dbg.max()
        k_norm = k_norms_dbg.mean()
        k_norm_std = k_norms_dbg.std()
        k_norm_max = k_norms_dbg.max()
        v_norm_dbg = v_norms_dbg.mean()

    if analysis:
        (out,
         attn_logit_mean, attn_logit_std, attn_logit_max_actual,
         softmax_top1_mean, attn_softmax_top1_max,
         logit_gap_mean, logit_gap_max,
         softmax_entropy_mean, softmax_entropy_min) = _attn_scores(
            Q, K, V, rng_attn_drop)
    elif local_diagnostics:
        out, attn_logit_max_actual, attn_softmax_top1_max = _attn_scores(
            Q, K, V, rng_attn_drop)
    else:
        out = _attn_scores(Q, K, V, rng_attn_drop)
    if analysis or local_diagnostics:
        o_input_norm = jnp.linalg.norm(out, axis=-1).mean()
        if local_diagnostics and not analysis:
            q_norm_max = q_norms_dbg.max()
            k_norm_max = k_norms_dbg.max()
        v_norm_max = v_norms_dbg.max()
        o_input_norm_max = jnp.linalg.norm(out, axis=-1).max()
    out = out.transpose(0, 2, 1, 3).reshape(B, S, D)
    out = out @ expand_O_kernel
    attn_out_norm = jnp.linalg.norm(out, axis=-1).mean()
    if analysis or local_diagnostics:
        o_out_norm_max = jnp.linalg.norm(out, axis=-1).max()
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
        ret = slim_ret
        if return_prune_stats:
            ret = ret + (
                qk_kept_count, qk_kept_frac, qk_full_gate_sum,
                qk_kept_gate_sum, qk_retained_gate_mass,
                qk_int_cap_frac, qk_gate_max_mean,
                v_kept_count, v_kept_frac, v_full_gate_sum,
                v_kept_gate_sum, v_retained_gate_mass,
                v_int_cap_frac, v_gate_max_mean,
            )
        if local_diagnostics:
            attn_local_layer_values = jnp.stack([
                q_norm_max, k_norm_max, v_norm_max,
                attn_logit_max_actual, attn_softmax_top1_max,
                o_input_norm_max, o_out_norm_max,
            ])
            attn_local_values = jnp.concatenate(
                [qk_local_values, v_local_values], axis=0)
            attn_local_locs = jnp.concatenate(
                [qk_local_locs, v_local_locs], axis=0)
            attn_top1_values = jnp.concatenate(
                [qk_top1_values, v_top1_values], axis=0)
            attn_top1_locs = jnp.concatenate(
                [qk_top1_locs, v_top1_locs], axis=0)
            ret = ret + (
                attn_local_layer_values,
                attn_local_values, attn_local_locs,
                attn_top1_values, attn_top1_locs,
            )
        return ret

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
    analysis_ret = slim_ret + (
        qk_raw_norm, v_raw_norm,
        q_norm, k_norm, v_norm_dbg, attn_logit_max_actual, o_input_norm,
        attn_qk_phi_binary, attn_v_phi_binary,
        attn_tau_std, attn_tau_kernel_norm,
        attn_z_lt_075_frac, attn_z_lt_030_frac,
        attn_score_skew, attn_active_per_token_std, attn_gate_entropy,
        attn_den_cost,
        attn_activation_cost, attn_current_cost,
        attn_qk_emb_norm_max, attn_v_emb_norm_max,
        attn_score_kurt,
        attn_int_cap_frac,
        q_norm_std, q_norm_max,
        k_norm_std, k_norm_max,
        attn_logit_mean, attn_logit_std,
        softmax_top1_mean, attn_softmax_top1_max,
        logit_gap_mean, logit_gap_max,
        softmax_entropy_mean, softmax_entropy_min,
        o_input_norm_max, o_out_norm_max,
    )
    if local_diagnostics:
        attn_local_layer_values = jnp.stack([
            q_norm_max, k_norm_max, v_norm_max,
            attn_logit_max_actual, attn_softmax_top1_max,
            o_input_norm_max, o_out_norm_max,
        ])
        attn_local_values = jnp.concatenate([qk_local_values, v_local_values], axis=0)
        attn_local_locs = jnp.concatenate([qk_local_locs, v_local_locs], axis=0)
        attn_top1_values = jnp.concatenate([qk_top1_values, v_top1_values], axis=0)
        attn_top1_locs = jnp.concatenate([qk_top1_locs, v_top1_locs], axis=0)
        analysis_ret = analysis_ret + (
            attn_local_layer_values,
            attn_local_values, attn_local_locs,
            attn_top1_values, attn_top1_locs,
        )
    return analysis_ret


def _rst_forward(x, pool_params, router_params, rng,
                  router_dropout, dropout_rate, deterministic,
                  sharded_fns, analysis=False, return_prune_stats=False,
                  local_diagnostics=False, tau_offset_clip=None,
                  d_model=None, n_layers=None,
                  fixed_depth_pool_scale=False):
    """v4.1: sharded-only. sharded_fns=(fused_single, fused_paired) required.

    `analysis` see _attn_forward docstring.
    """
    rst_emb = pool_params['rst_emb']
    rst_read = pool_params['rst_read']
    rst_write = pool_params['rst_write']

    rng, rng_drop = jax.random.split(rng)
    h = x @ router_params['proj_rst']['kernel'] + router_params['proj_rst']['bias']
    h = safe_dropout(h, router_dropout, deterministic, rng_drop)

    # Raw signature params are passed into the sharded SRW closure.
    # The closure can forward-normalize them for routing stability while
    # retaining raw parameter norms as diagnostics.
    rst_emb_unit = rst_emb
    tau = x @ router_params['tau_rst']['kernel'] + router_params['tau_rst']['bias']
    if tau_offset_clip is not None and tau_offset_clip > 0:
        tau_offset_clip_f = jnp.float32(tau_offset_clip)
        tau = jnp.clip(tau, -tau_offset_clip_f, tau_offset_clip_f)
    raw_scan_offset = x @ router_params['raw_scan_offset_rst']['kernel'] + router_params['raw_scan_offset_rst']['bias']
    if analysis:
        rst_tau_std = jax.lax.stop_gradient(tau).std()
        rst_tau_kernel_norm = jnp.sqrt(
            jnp.sum(jax.lax.stop_gradient(router_params['tau_rst']['kernel']) ** 2) + 1e-12)

    if fixed_depth_pool_scale:
        if d_model is None or n_layers is None:
            raise ValueError(
                "fixed_depth_pool_scale requires d_model and n_layers.")
        _, _, rst_scale = _fixed_depth_pool_output_scales(d_model, n_layers)
    else:
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
    if return_prune_stats:
        (rst_kept_count, rst_kept_frac, rst_full_gate_sum,
         rst_kept_gate_sum, rst_retained_gate_mass, rst_int_cap_frac,
         rst_gate_max_mean) = rst_ret[20:27]
    if analysis:
        (phi_binary_frac, rst_z_lt_075_frac, rst_z_lt_030_frac,
         rst_score_skew, rst_active_per_token_std, rst_gate_entropy,
         rst_den_cost, rst_activation_cost, rst_current_cost,
         rst_score_kurt, rst_int_cap_frac) = rst_ret[20:31]
        rst_raw_out_norm = jnp.linalg.norm(out, axis=-1).mean()
    if local_diagnostics:
        (rst_local_values, rst_local_locs,
         rst_top1_values, rst_top1_locs) = rst_ret[-4:]
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
        ret = slim_ret
        if return_prune_stats:
            ret = ret + (
                rst_kept_count, rst_kept_frac, rst_full_gate_sum,
                rst_kept_gate_sum, rst_retained_gate_mass,
                rst_int_cap_frac, rst_gate_max_mean,
            )
        if local_diagnostics:
            ret = ret + (
                rst_local_values, rst_local_locs,
                rst_top1_values, rst_top1_locs,
            )
        return ret

    rst_phi_binary = phi_binary_frac.mean()
    analysis_ret = slim_ret + (
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
    if local_diagnostics:
        analysis_ret = analysis_ret + (
            rst_local_values, rst_local_locs,
            rst_top1_values, rst_top1_locs,
        )
    return analysis_ret


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
    """DAWN-SRW v4.1.5.9 with Attention Layers and RST Layers."""
    __version__ = "spatial-r1-v4.1.5.9"

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
    # Optional safety bound on the router-produced tau offsets. This is a
    # forward-time clamp, not a new parameter, so old checkpoints remain
    # compatible. Set to None/0 to preserve exact v4.1.5.9 behavior.
    tau_offset_clip: Optional[float] = None
    # Initial bias for tau_attn/tau_rst relative threshold offsets.
    tau_offset_init: float = -0.5
    # Optional pool-specific tau init overrides.
    tau_offset_init_attn: Optional[float] = None
    tau_offset_init_rst: Optional[float] = None
    # Checkpoint-compatible ablation: keep scale params in NeuronPool, but
    # ignore them in forward/inference and use depth-scaled constants.
    fixed_depth_pool_scale: bool = True

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
            router_dropout=self.router_dropout,
            tau_offset_init=self.tau_offset_init,
            tau_offset_init_attn=self.tau_offset_init_attn,
            tau_offset_init_rst=self.tau_offset_init_rst)
        self.layers = [
            DAWNBlock(d_model=self.d_model, n_heads=self.n_heads,
                      dropout_rate=self.dropout_rate, name=f'block_{i}')
            for i in range(self.n_layers)]
        self.norm = nn.LayerNorm()

    def __call__(self, input_ids, labels=None, attention_mask=None,
                 deterministic=False, sharded_fns=None, analysis=False,
                 return_prune_stats=False, local_diagnostics=False):
        """Run the shared-pool SRW Transformer forward pass.

        analysis=False is the train/eval path and returns only regular
        training metrics.  analysis=True enables extra observational stats
        such as distribution shape, boundary fraction, entropy, tau stats,
        raw norms, and debug norms.
        """
        if analysis and return_prune_stats:
            raise ValueError(
                "return_prune_stats is paper-eval-only and cannot be "
                "combined with analysis=True")
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
                    self.n_heads, self.d_model, self.n_layers,
                    self.router_dropout, self.dropout_rate, deterministic,
                    sharded_fns=_sharded, analysis=analysis,
                    return_prune_stats=return_prune_stats,
                    local_diagnostics=local_diagnostics,
                    tau_offset_clip=self.tau_offset_clip,
                    fixed_depth_pool_scale=self.fixed_depth_pool_scale)
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
                if return_prune_stats:
                    (a_qk_kept_count, a_qk_kept_frac, a_qk_full_gate_sum,
                     a_qk_kept_gate_sum, a_qk_retained_gate_mass,
                     a_qk_int_cap_frac, a_qk_gate_max_mean,
                     a_v_kept_count, a_v_kept_frac, a_v_full_gate_sum,
                     a_v_kept_gate_sum, a_v_retained_gate_mass,
                     a_v_int_cap_frac, a_v_gate_max_mean) = attn_ret[33:47]
                if analysis:
                    (a_qk_raw_norm, a_v_raw_norm,
                     a_q_norm, a_k_norm, a_v_norm_dbg, a_logit_max, a_o_input_norm,
                     a_qk_phi_bin, a_v_phi_bin,
                     a_tau_std, a_tau_kernel_norm,
                     a_z075, a_z030,
                     a_skew, a_apt_std, a_entropy,
                     a_den_cost, a_activation_cost, a_current_cost,
                     a_qk_emb_n_max, a_v_emb_n_max,
                     a_score_kurt, a_int_cap_frac,
                     a_q_norm_std, a_q_norm_max,
                     a_k_norm_std, a_k_norm_max,
                     a_logit_mean, a_logit_std,
                     a_softmax_top1_mean, a_softmax_top1_max,
                     a_logit_gap_mean, a_logit_gap_max,
                     a_softmax_entropy_mean, a_softmax_entropy_min,
                     a_o_input_norm_max, a_o_out_norm_max) = attn_ret[33:70]
                if local_diagnostics:
                    (a_attn_local_layer_values,
                     a_attn_local_values, a_attn_local_locs,
                     a_attn_top1_values, a_attn_top1_locs) = attn_ret[-5:]
                x = x + attn_out

                normed = _layer_norm(
                    x, bp['norm2']['scale'], bp['norm2']['bias'])
                rst_ret = _rst_forward(
                    normed, pool_params, router_params, rng_rst,
                    self.router_dropout, self.dropout_rate, deterministic,
                    sharded_fns=_sharded, analysis=analysis,
                    return_prune_stats=return_prune_stats,
                    local_diagnostics=local_diagnostics,
                    tau_offset_clip=self.tau_offset_clip,
                    d_model=self.d_model, n_layers=self.n_layers,
                    fixed_depth_pool_scale=self.fixed_depth_pool_scale)
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
                if return_prune_stats:
                    (k_kept_count, k_kept_frac, k_full_gate_sum,
                     k_kept_gate_sum, k_retained_gate_mass,
                     k_int_cap_frac, k_gate_max_mean) = rst_ret[28:35]
                if analysis:
                    (k_raw_out_norm,
                     k_tau_std, k_tau_kernel_norm,
                     k_z075, k_z030,
                     k_skew, k_apt_std, k_entropy,
                     k_den_cost, k_activation_cost, k_current_cost,
                     k_emb_n_max, k_score_kurt, k_phi_bin,
                     k_int_cap_frac) = rst_ret[28:43]
                if local_diagnostics:
                    (k_local_values, k_local_locs,
                     k_top1_values, k_top1_locs) = rst_ret[-4:]
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
                if return_prune_stats:
                    slim_ys = slim_ys + (
                        a_qk_kept_count, a_qk_kept_frac,
                        a_qk_full_gate_sum, a_qk_kept_gate_sum,
                        a_qk_retained_gate_mass, a_qk_int_cap_frac,
                        a_qk_gate_max_mean,
                        a_v_kept_count, a_v_kept_frac,
                        a_v_full_gate_sum, a_v_kept_gate_sum,
                        a_v_retained_gate_mass, a_v_int_cap_frac,
                        a_v_gate_max_mean,
                        k_kept_count, k_kept_frac,
                        k_full_gate_sum, k_kept_gate_sum,
                        k_retained_gate_mass, k_int_cap_frac,
                        k_gate_max_mean,
                    )
                if not analysis:
                    if local_diagnostics:
                        slim_ys = slim_ys + (
                            a_attn_local_layer_values,
                            a_attn_local_values, a_attn_local_locs,
                            a_attn_top1_values, a_attn_top1_locs,
                            k_local_values, k_local_locs,
                            k_top1_values, k_top1_locs,
                        )
                    return x, slim_ys
                analysis_ys = slim_ys + (
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
                    a_q_norm_std, a_q_norm_max,
                    a_k_norm_std, a_k_norm_max,
                    a_logit_mean, a_logit_std,
                    a_softmax_top1_mean, a_softmax_top1_max,
                    a_logit_gap_mean, a_logit_gap_max,
                    a_softmax_entropy_mean, a_softmax_entropy_min,
                    a_o_input_norm_max, a_o_out_norm_max,
                )
                if local_diagnostics:
                    analysis_ys = analysis_ys + (
                        a_attn_local_layer_values,
                        a_attn_local_values, a_attn_local_locs,
                        a_attn_top1_values, a_attn_top1_locs,
                        k_local_values, k_local_locs,
                        k_top1_values, k_top1_locs,
                    )
                return x, analysis_ys

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
            _scan_offset = 59
            if return_prune_stats:
                (attn_qk_kept_count_all, attn_qk_kept_frac_all,
                 attn_qk_full_gate_sum_all, attn_qk_kept_gate_sum_all,
                 attn_qk_retained_gate_mass_all, attn_qk_int_cap_frac_all,
                 attn_qk_gate_max_mean_all,
                 attn_v_kept_count_all, attn_v_kept_frac_all,
                 attn_v_full_gate_sum_all, attn_v_kept_gate_sum_all,
                 attn_v_retained_gate_mass_all, attn_v_int_cap_frac_all,
                 attn_v_gate_max_mean_all,
                 rst_kept_count_all, rst_kept_frac_all,
                 rst_full_gate_sum_all, rst_kept_gate_sum_all,
                 rst_retained_gate_mass_all, rst_int_cap_frac_prune_all,
                 rst_gate_max_mean_all) = scan_ys[
                    _scan_offset:_scan_offset + 21]
                _scan_offset += 21
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
                 attn_int_cap_frac_all, rst_int_cap_frac_all,
                 attn_q_norm_std_all, attn_q_norm_max_all,
                 attn_k_norm_std_all, attn_k_norm_max_all,
                 attn_logit_mean_all, attn_logit_std_all,
                 attn_softmax_top1_mean_all, attn_softmax_top1_max_all,
                 attn_logit_gap_mean_all, attn_logit_gap_max_all,
                 attn_softmax_entropy_mean_all,
                 attn_softmax_entropy_min_all,
                 attn_o_input_norm_max_all,
                 attn_o_output_norm_max_all) = scan_ys[
                    _scan_offset:_scan_offset + 52]
                _scan_offset += 52
            if local_diagnostics:
                (attn_local_layer_values_all,
                 attn_local_values_all, attn_local_locs_all,
                 attn_top1_values_all, attn_top1_locs_all,
                 rst_local_values_all, rst_local_locs_all,
                 rst_top1_values_all, rst_top1_locs_all) = scan_ys[
                    _scan_offset:_scan_offset + 9]
                _scan_offset += 9
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
        if return_prune_stats and not self.is_initializing():
            result.update({
                'attn_qk_kept_count_mean': attn_qk_kept_count_all.mean(),
                'attn_qk_kept_frac_mean': attn_qk_kept_frac_all.mean(),
                'attn_qk_full_gate_sum_mean': attn_qk_full_gate_sum_all.mean(),
                'attn_qk_kept_gate_sum_mean': attn_qk_kept_gate_sum_all.mean(),
                'attn_qk_retained_gate_mass': attn_qk_retained_gate_mass_all.mean(),
                'attn_qk_int_cap_frac': attn_qk_int_cap_frac_all.mean(),
                'attn_qk_gate_max_mean': attn_qk_gate_max_mean_all.mean(),
                'attn_v_kept_count_mean': attn_v_kept_count_all.mean(),
                'attn_v_kept_frac_mean': attn_v_kept_frac_all.mean(),
                'attn_v_full_gate_sum_mean': attn_v_full_gate_sum_all.mean(),
                'attn_v_kept_gate_sum_mean': attn_v_kept_gate_sum_all.mean(),
                'attn_v_retained_gate_mass': attn_v_retained_gate_mass_all.mean(),
                'attn_v_int_cap_frac': attn_v_int_cap_frac_all.mean(),
                'attn_v_gate_max_mean': attn_v_gate_max_mean_all.mean(),
                'rst_kept_count_mean': rst_kept_count_all.mean(),
                'rst_kept_frac_mean': rst_kept_frac_all.mean(),
                'rst_full_gate_sum_mean': rst_full_gate_sum_all.mean(),
                'rst_kept_gate_sum_mean': rst_kept_gate_sum_all.mean(),
                'rst_retained_gate_mass': rst_retained_gate_mass_all.mean(),
                'rst_int_cap_frac': rst_int_cap_frac_prune_all.mean(),
                'rst_gate_max_mean': rst_gate_max_mean_all.mean(),
            })
        if analysis and not self.is_initializing():
            _residual_norm = jnp.linalg.norm(x, axis=-1).mean()
            _emb_norm = jnp.linalg.norm(self.token_emb.embedding, axis=-1).mean()
            _o_proj_norm = jnp.linalg.norm(
                stacked['attn']['expand_O']['kernel'], axis=(-2, -1)).mean()
            _attn_logit_max_layer = jnp.argmax(attn_logit_max_all)
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
                'attn_q_norm_mean': attn_q_norm_all.mean(),
                'attn_q_norm_std': attn_q_norm_std_all.mean(),
                'attn_q_norm_max': attn_q_norm_max_all.max(),
                'attn_k_norm_mean': attn_k_norm_all.mean(),
                'attn_k_norm_std': attn_k_norm_std_all.mean(),
                'attn_k_norm_max': attn_k_norm_max_all.max(),
                'attn_logit_mean': attn_logit_mean_all.mean(),
                'attn_logit_std': attn_logit_std_all.mean(),
                'attn_logit_max': attn_logit_max_all.max(),
                'attn_logit_max_layer': _attn_logit_max_layer,
                'attn_softmax_top1_mean': attn_softmax_top1_mean_all.mean(),
                'attn_softmax_top1_max': attn_softmax_top1_max_all.max(),
                'attn_logit_gap_top1_top2_mean': attn_logit_gap_mean_all.mean(),
                'attn_logit_gap_top1_top2_max': attn_logit_gap_max_all.max(),
                'attn_softmax_entropy_mean': attn_softmax_entropy_mean_all.mean(),
                'attn_softmax_entropy_min': attn_softmax_entropy_min_all.min(),
                'attn_o_input_norm_mean': attn_o_input_norm_all.mean(),
                'attn_o_input_norm_max': attn_o_input_norm_max_all.max(),
                'attn_o_output_norm_mean': attn_out_norm_all.mean(),
                'attn_o_output_norm_max': attn_o_output_norm_max_all.max(),
                'per_layer_attn_q_norm_mean': attn_q_norm_all,
                'per_layer_attn_q_norm_std': attn_q_norm_std_all,
                'per_layer_attn_q_norm_max': attn_q_norm_max_all,
                'per_layer_attn_k_norm_mean': attn_k_norm_all,
                'per_layer_attn_k_norm_std': attn_k_norm_std_all,
                'per_layer_attn_k_norm_max': attn_k_norm_max_all,
                'per_layer_attn_logit_mean': attn_logit_mean_all,
                'per_layer_attn_logit_std': attn_logit_std_all,
                'per_layer_attn_logit_max': attn_logit_max_all,
                'per_layer_attn_softmax_top1_mean': attn_softmax_top1_mean_all,
                'per_layer_attn_softmax_top1_max': attn_softmax_top1_max_all,
                'per_layer_attn_logit_gap_top1_top2_mean': attn_logit_gap_mean_all,
                'per_layer_attn_logit_gap_top1_top2_max': attn_logit_gap_max_all,
                'per_layer_attn_softmax_entropy_mean': attn_softmax_entropy_mean_all,
                'per_layer_attn_softmax_entropy_min': attn_softmax_entropy_min_all,
                'per_layer_attn_o_input_norm_mean': attn_o_input_norm_all,
                'per_layer_attn_o_input_norm_max': attn_o_input_norm_max_all,
                'per_layer_attn_o_output_norm_mean': attn_out_norm_all,
                'per_layer_attn_o_output_norm_max': attn_o_output_norm_max_all,
                'attn_int_cap_frac': attn_int_cap_frac_all.mean(),
                'rst_int_cap_frac': rst_int_cap_frac_all.mean(),
            })
        if local_diagnostics and not self.is_initializing():
            result.update({
                'attn_local_layer_values': attn_local_layer_values_all,
                'local_spike_values': jnp.concatenate([
                    attn_local_values_all,
                    rst_local_values_all,
                ], axis=1),
                'local_spike_locs': jnp.concatenate([
                    attn_local_locs_all,
                    rst_local_locs_all,
                ], axis=1),
                'local_top1_values': jnp.concatenate([
                    attn_top1_values_all,
                    rst_top1_values_all,
                ], axis=1),
                'local_top1_locs': jnp.concatenate([
                    attn_top1_locs_all,
                    rst_top1_locs_all,
                ], axis=1),
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
            'fixed_depth_pool_scale': self.fixed_depth_pool_scale,
        }

    def get_model_info(self):
        n_rst_eff = self.n_rst if self.n_rst is not None else (
            self.n_know if self.n_know is not None else 25200)
        qk_scale, v_scale, rst_scale = _fixed_depth_pool_output_scales(
            self.d_model, self.n_layers)
        return [
            f"DAWN-SRW ({self.__version__})",
            f"  d_model={self.d_model}, d_route={self.d_route}, "
            f"n_layers={self.n_layers}, n_heads={self.n_heads}",
            f"  Attention-QK: {self.n_qk}, Attention-V: {self.n_v}, RST: {n_rst_eff}",
            f"  Route: learned d_route embedding [{self.d_route}]",
            "  Pool scales: fixed depth-scaled "
            f"(qk={float(qk_scale):.6g}, v={float(v_scale):.6g}, "
            f"rst={float(rst_scale):.6g})" if self.fixed_depth_pool_scale
            else "  Pool scales: learned sqrt(d_model)-initialized params",
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
    # v4.1.5.9: inference uses read/write directions; params stay raw.
    r_n = _forward_unit_direction(w_read.astype(jnp.float32))
    w_n = _forward_unit_direction(w_write.astype(jnp.float32))
    scores = h @ emb.T
    scores_f32 = scores.astype(jnp.float32)
    s_mean = scores_f32.mean(axis=-1, keepdims=True)
    s_std = jnp.sqrt(jnp.mean(jnp.square(scores_f32 - s_mean),
                               axis=-1, keepdims=True)) + 1e-8
    scan_offset = SCAN_SCALE * jnp.tanh(raw_scan_offset)
    tau = s_mean + tau_offset * s_std - scan_offset / jnp.maximum(s_std, SCAN_STD_FLOOR)

    raw = scores_f32 - tau
    margin = raw - ACTIVATION_THRESHOLD
    activation = jnp.clip(raw.astype(jnp.float32) / ACTIVATION_THRESHOLD, 0.0, 1.0)
    intensity = jnp.minimum(jnp.maximum(scores_f32, 0.0), MAX_INTENSITY)
    gate = activation * intensity

    xr = x.astype(jnp.float32) @ r_n.T
    a = gate * xr
    raw_out = a @ w_n
    den = jnp.maximum(gate.sum(axis=-1, keepdims=True), 1.0)
    out = raw_out.astype(jnp.float32) / den
    return out.astype(jnp.float32)


def _srw_inference_with_gates(x, h, emb, tau_offset, raw_scan_offset, w_read, w_write):
    """Like _srw_inference but also returns raw and normalized gate for analysis."""
    # v4.1.5.9: analysis inference uses read/write directions; params stay raw.
    r_n = _forward_unit_direction(w_read.astype(jnp.float32))
    w_n = _forward_unit_direction(w_write.astype(jnp.float32))
    scores = h @ emb.T
    scores_f32 = scores.astype(jnp.float32)
    s_mean = scores_f32.mean(axis=-1, keepdims=True)
    s_std = jnp.sqrt(jnp.mean(jnp.square(scores_f32 - s_mean),
                               axis=-1, keepdims=True)) + 1e-8
    scan_offset = SCAN_SCALE * jnp.tanh(raw_scan_offset)
    tau = s_mean + tau_offset * s_std - scan_offset / jnp.maximum(s_std, SCAN_STD_FLOOR)

    raw = scores_f32 - tau
    margin = raw - ACTIVATION_THRESHOLD
    activation = jnp.clip(raw.astype(jnp.float32) / ACTIVATION_THRESHOLD, 0.0, 1.0)
    intensity = jnp.minimum(jnp.maximum(scores_f32, 0.0), MAX_INTENSITY)
    gate = activation * intensity
    gate_norm = gate / jnp.maximum(gate.sum(axis=-1, keepdims=True), 1e-8)

    xr = x.astype(jnp.float32) @ r_n.T
    a = gate * xr
    raw_out = a @ w_n
    den = jnp.maximum(gate.sum(axis=-1, keepdims=True), 1.0)
    out = raw_out.astype(jnp.float32) / den
    return out.astype(jnp.float32), gate, gate_norm



def _attn_forward_cached(x, pool_params, router_params, expand_O_kernel,
                         n_heads, d_model, n_layers,
                         cache_K, cache_V, cache_len,
                         fixed_depth_pool_scale=False):
    """Cached attention decode step. x: [B, 1, D]."""
    B = x.shape[0]
    d_head = d_model // n_heads

    # Route embeddings are used as-is, matching the training path.
    qk_norm = _forward_unit_direction(pool_params['attn_qk_emb'].astype(jnp.float32))
    v_norm = _forward_unit_direction(pool_params['attn_v_emb'].astype(jnp.float32))
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
    _qk_s, _v_s, _ = _effective_pool_output_scales(
        pool_params, d_model, n_layers, fixed_depth_pool_scale)
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


def _rst_forward_inference(x, pool_params, router_params,
                           d_model=None, n_layers=None,
                           fixed_depth_pool_scale=False):
    """Inference-only RST Layer forward. No chunking, no LB, no dropout."""
    # emb used as-is (matches training path).
    rst_norm = _forward_unit_direction(pool_params['rst_emb'].astype(jnp.float32))
    h = x @ router_params['proj_rst']['kernel'] + router_params['proj_rst']['bias']
    tau = x @ router_params['tau_rst']['kernel'] + router_params['tau_rst']['bias']
    raw_scan_offset = x @ router_params['raw_scan_offset_rst']['kernel'] + router_params['raw_scan_offset_rst']['bias']
    out = _srw_inference(x, h, rst_norm, tau, raw_scan_offset,
                         pool_params['rst_read'], pool_params['rst_write'])
    if fixed_depth_pool_scale:
        if d_model is None or n_layers is None:
            raise ValueError(
                "fixed_depth_pool_scale requires d_model and n_layers.")
        _, _, rst_scale = _fixed_depth_pool_output_scales(d_model, n_layers)
    else:
        rst_scale = pool_params['rst_scale']
    return out * rst_scale




def prefill(params, model_cfg, input_ids):
    """Run full forward on prompt, populate KV cache.

    Returns: logits [B,S,vocab], cache_K, cache_V [n_layers,B,H,max_seq,d_head], cache_len
    """
    params = _squeeze_params(params)
    B, S = input_ids.shape
    d_model = model_cfg['d_model']
    n_layers = model_cfg['n_layers']
    n_heads = model_cfg['n_heads']
    fixed_depth_pool_scale = _model_cfg_uses_fixed_depth_pool_scale(model_cfg)
    max_seq = model_cfg['max_seq_len']
    d_head = d_model // n_heads

    pool_params = params['neuron_pool']
    router_params = params['router']
    qk_scale_eff, v_scale_eff, _ = _effective_pool_output_scales(
        pool_params, d_model, n_layers, fixed_depth_pool_scale)

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
        _qk_s = qk_scale_eff
        _v_s = v_scale_eff
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
        rst_out = _rst_forward_inference(
            normed, pool_params, router_params,
            d_model=d_model, n_layers=n_layers,
            fixed_depth_pool_scale=fixed_depth_pool_scale)
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
    fixed_depth_pool_scale = _model_cfg_uses_fixed_depth_pool_scale(model_cfg)

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
            n_heads, d_model, n_layers,
            cK[layer_idx], cV[layer_idx], pos,
            fixed_depth_pool_scale=fixed_depth_pool_scale)
        cK = cK.at[layer_idx].set(new_cK)
        cV = cV.at[layer_idx].set(new_cV)
        x = x + attn_out

        normed = _layer_norm(x, bp['norm2']['scale'], bp['norm2']['bias'])
        rst_out = _rst_forward_inference(
            normed, pool_params, router_params,
            d_model=d_model, n_layers=n_layers,
            fixed_depth_pool_scale=fixed_depth_pool_scale)
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
    fixed_depth_pool_scale = _model_cfg_uses_fixed_depth_pool_scale(model_cfg)

    pool_params = params['neuron_pool']
    router_params = params['router']
    qk_scale_eff, v_scale_eff, rst_scale_eff = _effective_pool_output_scales(
        pool_params, d_model, n_layers, fixed_depth_pool_scale)
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
            _qk_s = qk_scale_eff
            _v_s = v_scale_eff
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
            x = x + rst_out * rst_scale_eff
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
    fixed_depth_pool_scale = _model_cfg_uses_fixed_depth_pool_scale(model_cfg)

    pool_params = params['neuron_pool']
    router_params = params['router']
    qk_scale_eff, v_scale_eff, rst_scale_eff = _effective_pool_output_scales(
        pool_params, d_model, n_layers, fixed_depth_pool_scale)

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
        _qk_s = qk_scale_eff
        _v_s = v_scale_eff
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
        rst_out = rst_out * rst_scale_eff
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
        # v4.1.5.9: suppressed forward uses read/write directions.
        r_n = _forward_unit_direction(w_read.astype(jnp.float32))
        w_n = _forward_unit_direction(w_write.astype(jnp.float32))
        scores = h @ emb.T
        sf = scores.astype(jnp.float32)
        s_mean = sf.mean(axis=-1, keepdims=True)
        s_std = jnp.sqrt(jnp.mean(jnp.square(sf - s_mean), axis=-1, keepdims=True)) + 1e-8
        scan_offset = SCAN_SCALE * jnp.tanh(raw_scan_offset)
        tau = s_mean + tau_off * s_std - scan_offset / jnp.maximum(s_std, SCAN_STD_FLOOR)
        raw = scores - tau.astype(scores.dtype)
        margin = raw.astype(jnp.float32) - ACTIVATION_THRESHOLD
        activation = jnp.clip(raw.astype(jnp.float32) / ACTIVATION_THRESHOLD, 0.0, 1.0)
        intensity = jnp.minimum(jnp.maximum(sf, 0.0), MAX_INTENSITY)
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
        fixed_depth_pool_scale = _model_cfg_uses_fixed_depth_pool_scale(model_cfg)
        pp = params['neuron_pool']
        rp = params['router']
        qk_scale_eff, v_scale_eff, rst_scale_eff = _effective_pool_output_scales(
            pp, d_model, n_layers, fixed_depth_pool_scale)

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
            _qk_s = qk_scale_eff
            _v_s = v_scale_eff
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
            x = x + _srw_sup(normed, h_k, kn_n, tau_k, raw_scan_offset_k, pp['rst_read'], pp['rst_write'], rst_mult) * rst_scale_eff

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
