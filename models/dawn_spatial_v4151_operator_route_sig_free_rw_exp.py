"""
DAWN-SRW v4.1.5.1: operator routing with free-norm read/write

## Design philosophy

DAWN represents neurons as positions in a shared sense space. Each
neuron is a rank-1 atomic unit (read-write vector pair) positioned by
its embedding. v4.1 extends this principle down to the gate level: the
gate itself is factored into two stages mirroring biological neuron
behaviour.

## Structure

### Neurons (rank-1 atomic units)
  - Embedding/tag: route-normalized after signature concat
  - Read / Write: free norm in forward

  Neurons represent only direction. Importance is expressed through gate
  magnitude, not embedding norm.

### Tokens
  - H: free norm (learnable Dense projection of x)
  - Token strength is expressed via ||h||
  - Score = h 쨌 emb_unit = ||h|| 횞 cos(h, emb_unit)

### Two-stage gate

**Stage 1 ??Activation (binary-like selection)**:
    margin     = (score - tau) - ACTIVATION_THRESHOLD
    activation = sigmoid(SHARPNESS 횞 margin)

  With SHARPNESS=500 and ACTIVATION_THRESHOLD=0.5, activation is
  near-binary: activation ??[0.5, 0.993] over margin ??[0, 0.01] (??%
  of the intensity range). Neurons either clearly fire (margin > 0.01)
  or clearly do not (margin < 0).
  Biological analogy: action-potential firing.

**Stage 2 ??Intensity (linear, bounded)**:
    active_margin = max(margin - ACTIVATION_CUTOFF, 0)
    intensity     = EPSILON + min(active_margin, MAX_INTENSITY)

  intensity = EPSILON (effectively off) for every margin
  ??ACTIVATION_CUTOFF ??precisely the sigmoid transition zone. Past the
  cutoff it grows linearly from EPSILON to EPSILON + MAX_INTENSITY.
  Biological analogy: firing rate / synaptic output strength.

**Gate**:
    gate = activation 횞 intensity

  * Boundary (margin ??[0, 0.01]): activation ??[0.5, 0.993],
    intensity = EPSILON ??gate ??0.  No contribution.
  * Confirmed-active (margin > 0.01): activation ??1; intensity
    ??[EPSILON, EPSILON + MAX_INTENSITY].  Full dynamic range lives here.

### Denominator
    route_i = normalize(concat(normalize(tag_i),
                               normalize(unit_read_i @ Rr + unit_write_i @ Rw)))
    a_i = activation_i * intensity_i * <x, unit_read_i>
    den = max(sum(activation_i * intensity_i), 1.0)

  v4.1.5 routes on a small learned tag plus a fixed functional signature
  derived from the neuron's unit read/write operation. tag_dim,
  read_sig_dim, and write_sig_dim are config-driven; routing signature
  dimension is d_route = tag_dim + read_sig_dim + write_sig_dim.


### Scan-bias tau
    scan = scan_scale 횞 tanh(scan_bias)
    tau  = s_mean + tau_offset 횞 s_std - scan / max(s_std, scan_std_floor)

  tau_offset gives the relative threshold axis. scan_bias gives a bounded
  low-std scan/recruit axis; positive scan lowers tau and opens more neurons.

### Dead-only penalty (structure unchanged, threshold bumped)
  Neurons with max_gate < DEAD_THRESHOLD (0.01, up from 1e-4) over a
  batch get their mean_score pushed toward zero (mean h direction).
  Asymmetric ??active neurons are untouched.

## Changes from v4.0.6

1. Emb forward-normalised (unit) ??v4.0.3-style restored.
2. Gate = activation 횞 intensity with ACTIVATION_CUTOFF shift.
3. Dynamic tau removed; relative tau remains, now with bounded scan_bias.
4. Den = 誇 intensity (was 誇 z+).
5. Dead threshold 1e-4 ??0.01.
6. Z computation / s_std normalisation of the gate removed entirely.
7. NeuronPool loses raw_alpha_{qk,v,know}; DAWN loses tau_alpha_init,
   max_alpha.  Model no longer has a learnable dynamic-tau slope.

## Key values

  SHARPNESS            = 500.0   activation transition window ??0.01 margin
  ACTIVATION_THRESHOLD = 0.5     raw must exceed tau by 0.5 to enter zone
  ACTIVATION_CUTOFF    = 0.01    intensity starts margin beyond this point
  EPSILON              = 1e-4    floor intensity (effectively "off")
  MAX_INTENSITY        = 10.0    cap (prevents drift)

## Terminology

  raw           = score - tau                          (static tau)
  margin        = raw - ACTIVATION_THRESHOLD           (distance to activation)
  activation    = sigmoid(SHARPNESS 횞 margin)          (stage 1)
  active_margin = max(margin - ACTIVATION_CUTOFF, 0)   (supra-confirmation)
  intensity     = EPSILON + min(active_margin, MAX_INT)
  gate          = activation 횞 intensity

Architecture:
  NeuronPool           -- emb[N,d_bn] + w_read[N,D] + w_write[N,D]
  Router               -- proj + tau. Uses pool emb for routing.
  make_sharded_srw / _paired  -- shard_map gate+srw (dynamic_slice, bf16, 2-pass)
  _attn_forward        -- paired QK + single V -> self-attn
  _know_forward        -- single Know
  DAWN                 -- embedding + jax.lax.scan + weight-tied lm_head

Changelog:
  v4.1.5 operator-derived routing signature:
    routing signature = concat(small learned tag, fixed projection of unit read/write);
    tag_dim, read_sig_dim, and write_sig_dim are config-driven; routing signature dimension is d_route = tag_dim + read_sig_dim + write_sig_dim;
    denominator restored to activation-weighted intensity normalization;
    tests whether operation-derived signatures recover 4.0.2-style automatic dense-to-sparse differentiation while preserving separate read/write operations.

  spatial-r1-v4.1 (2026-04-21; emb forward-norm reverted 2026-04-23;
                   read/write forward-norm removed 2026-04-23):
    - Two-stage gate: activation 횞 intensity with ACTIVATION_CUTOFF.
    - Dynamic tau + raw_alpha params removed entirely.
    - Bounded scan-bias added:
        scan = scan_scale * tanh(scan_bias)
        tau = s_mean + tau_offset * s_std - scan / max(s_std, scan_std_floor)
      scan_bias is zero-init and gives a context scan/recruit axis.
    - Den = 誇 intensity; dead threshold 1e-4 ??0.01.
    - Emb forward-norm: originally restored here, then reverted in-place
      on 2026-04-23 back to v4.0.6-style (no forward unit-norm). ||emb||
      is a learnable DoF again ??WD + task loss shape it so per-neuron
      magnitude can drive competitive sparsity (qk active regime ~5-6%,
      vs ~28% under forward unit-norm). read / write still
      forward-normalise per chunk; init still unit_norm_init.
    - 2026-04-23 (free-norm): read/write forward unit-norm also removed.
      All three pool tensors (emb / read / write) are now free-norm ??      each neuron has three independent magnitude axes:
        emb_norm   ??routing influence
        read_norm  ??input sensitivity
        write_norm ??output strength
      Regulated by per-group weight decay in the optimizer
      (pool_weight_decay ??base weight_decay; see train_jax.py). CE
      self-regulates: output-saturating neurons raise loss, so
      magnitudes don't run away without the forward-norm guard.

  spatial-r1-v4.1.4 competitive denominator wiring (2026-04-27):
    - spatial-r1-v4.1.4 trains from dawn_spatial_v414_competitive_den_exp.py.
    - Keeps v4.1.2 bounded scan-bias tau and margin intensity.
    - Uses unit read/write directions in the SRW contribution:
        r_i = read_i / ||read_i||
        w_i = write_i / ||write_i||
        a_i = activation_i * intensity_i * <x, r_i>
        out = sum_i a_i w_i
              / (1 + sum_i activation_i
                   + sum_i (sqrt(a_i^2 + epsilon^2) - epsilon))

  spatial-r1-v4.1.2 scan-bias wiring (2026-04-26):
    - spatial-r1-v4.1.2 trains from dawn_spatial_v412_scan_bias_exp.py.
    - scan_bias has no extra L2 term; it only follows the existing AdamW
      masks (bias excluded, kernel base WD).
    - Analysis/suppression helpers use:
        scan = scan_scale * tanh(scan_bias)
        tau = s_mean + tau_offset * s_std - scan / max(s_std, scan_std_floor)

  spatial-r1-v4.0.6 (2026-04-20):
    - Gate confidence: 過(z) (normal CDF) ??sigmoid(scores - tau).
      gate = where(z > 0, z 쨌 sigmoid(scores - tau), 0).
      Sharper transition past threshold; scores-tau reuses the
      already-computed (scores - tau) term so no extra reduction.
    - Dynamic tau: per-token shift based on the per-token score std.
        tau_effective = tau - alpha 쨌 (1 / (s_std + 1e-6))
      Harder tokens (low s_std) get lower tau ??more neurons recruited.
      Per-pool learnable alpha via softplus (qk / v / know); raw params
      live in NeuronPool, initial value softplus_inverse(0.1).
    - Dead-only penalty: for each pool, identify dead neurons (max gate
      across the step-batch below 1e-4) and add
          sum_j max(-mean_score_j, 0) 쨌 1[dead_j]
      as a scalar aux loss. Pressure to raise the *mean score* of
      neurons that never fire, without touching active neurons.
    - v4.0.4 / v4.0.5 reverse and bidirectional-gate dropouts removed.

  spatial-r1-v4.0.3 (2026-04-20):
    - Denominator redefined from 誇 gate_j to 誇 z_j^+ (positive part of z).
      Numerator still uses gate = z쨌過(z). Numerator/denominator asymmetry
      penalises borderline joiners (z < 0.75, where gate/z < 1) so tau
      learning is a structured vote weighted by neuron contribution.
      Floor of 1.0 kept (prevents tau runaway).
    - z_sum logged alongside gsum (誇 gate, now observational only).
    - Drop emb forward normalisation (emb_unit = emb / ||emb||).
      init still uses unit_norm_init so all emb rows start on the unit
      sphere; during training WD + task loss are the only forces on
      ||emb||. Per-neuron "strength" degree of freedom: specialists can
      grow large emb norms, generalists can shrink. read / write stay
      unit-normalised in the forward (size-dominance guard).

  spatial-r1-v4.0.1 (2026-04-12):
    - GELU gate (z 횞 過(z)) ??confidence 횞 intensity structure.
      過(z) = normal CDF provides statistical confidence from s_std-normalized z.
    - den floor=1.0 from v3.9.4 (prevents tau runaway to dense regime).

  spatial-r1-v3.9.9 (2026-04-12):
    - Gate function: sigmoid(z) ??GELU-like z*sigmoid(z) with z>0 mask
    - Separate confidence (sigmoid) and intensity (z) gradient paths
    - Same denominator structure as v3.9.4 (gate sum)
    - Unbounded gate range restores neuron importance differentiation
    - Active count threshold: gate > 0 (replaces sigmoid > 0.5)

  spatial-r1-v3.9.8.1 (2026-04-12):
    - Remove xr짼 from denominator: den = 誇 gate (pure confidence sum)
    - Motivation: xr짼 in denominator compensated when large-xr neurons
      were dropped, making pruning "free" ??over-sparse
    - den = 誇 sigmoid gives "confidence-weighted average" structure
    - Same cancellation property as v3.9.4 (gate in both num and den)

  spatial-r1-v3.9.8 (2026-04-12):
    - Sigmoid gate replaces binary+STE
    - gate = sigmoid(raw/s_std): bounded [0,1], continuous, no STE needed
    - Numerator and denominator share same gate ??structural cancellation
    - den = 誇 gate 횞 xr짼 (no stop_gradient on xr짼)
    - Bounded gate separates routing confidence from contribution magnitude
    - Linear denominator (no ?? + learnable per-pool scale (init=?쉊_model, WD excluded)

  spatial-r1-v3.9.7.1 (2026-04-12):
    - Remove ??from denominator: den = 誇 sigmoid(raw/s_std) 횞 sg(xr짼) (linear)
    - Replace fixed output scale with per-pool learnable output_scale (init=1.0)
      - know_scale, qk_scale, v_scale: scalar params, WD excluded
    - Linear den restores bidirectional feedback:
      fewer neurons ??smaller den ??larger output ??loss pressure

  spatial-r1-v3.9.7 (2026-04-12):
    - Binary gate + xr짼-weighted soft denominator
    - numerator: pure binary gate (STE via sigmoid(raw/s_std))
    - denominator: ??誇 sigmoid(raw/s_std) 횞 xr짼) ??xr-weighted smooth count
    - Remove stop_gradient from xr짼 in denominator
    - Allows denominator to provide self-regulation gradient to read
    - Prevents runaway xr growth (positive feedback loop)
    - sigmoid scale = 1/s_std (adaptive, no hyperparameter)
    - STE changed from ReLU-based to sigmoid-based
    - den floor = 1e-3

  spatial-r1-v3.9.6 (2026-04-11):
    - STE binary gate + soft denominator
    - numerator: gate_hard * xr @ wc (binary selection, STE gradient)
    - denominator: soft_gate_sum = 誇 ReLU(raw) (continuous, gradient flows)
    - out = raw_out / max(soft_gate_sum, 1.0)
    - Output scale ?쉊_model unchanged from v3.9.4
    - gate_norm_mode removed (single mode only)

  spatial-r1-v3.9.5 (2026-04-11):
    - STE binary gate: forward 0/1, backward continuous gradient
    - gate = gate_hard + gate_soft - stop_gradient(gate_soft)  (STE trick)
    - gate_hard = (raw > 0).astype(dtype), gate_soft = ReLU(raw)
    - gate_norm_mode config: "sqrt_active" or "active_n"
      - sqrt_active: out = raw_out / ??active_N + 1)
      - active_n: out = raw_out / max(active_N, 1)
    - clip(0, 10) removed (binary gate, unnecessary)
    - gate_concentration logging replaced by active_n_mean

  spatial-r1-v3.9.4 (2026-04-08):
    - Remove tanh(gate_max) heuristic from all output paths
    - gate_sum normalize (ratio) + fixed ?쉊_model scale only
    - x쨌read naturally modulates per-token output magnitude
    - No learnable strength parameters, no gate_strength variable
    - gate_sum floor=1.0: backward gradient 1/gate_sum짼 ??컻 諛⑹?

  spatial-r1-v3.9.1 (2026-04-05):
    - LB loss: gate-based ??score-based (pre-ReLU)
    - All neurons receive LB gradient (no ReLU barrier)
    - Naturally adaptive: weak when scores uniform, strong when biased
    - gate LB (ng_sum/ng_sq) removed from pass 2
    - read/write: forward normalize (unit direction), init unit_norm_init
    - score_lb: CV짼 with adaptive epsilon (spread-invariant, stable at mean??)
    - gate_strength: pmax across model shards (global max)
    - Fixed output_scale = ?쉊_model (not learnable, no WD issues)
    - Attn aux /3, layer .mean() (N/layer/pool invariant)

  spatial-r1-v3.9.0 (2026-04-05):
    - Gate: exp(gate)-1 ??ReLU (linear gate). No dead neuron gradient.
    - Gate: ?-normalization removed (raw = scores - tau, no /std)
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
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Dict
from functools import partial
from jax.sharding import PartitionSpec as P
from jax.experimental.shard_map import shard_map


# ================================================================
# V4.1 physical constants (defaults; overridable via config).
#
#   margin        = (score - tau) - ACTIVATION_THRESHOLD
#   activation    = sigmoid(SHARPNESS 횞 margin)
#   active_margin = max(margin - ACTIVATION_CUTOFF, 0)
#   intensity     = EPSILON + min(active_margin, MAX_INTENSITY)
#   gate          = activation 횞 intensity
#   den           = max(誇 intensity, 1.0)
# ================================================================

SHARPNESS = 500.0              # activation sigmoid sharpness (near-binary)
ACTIVATION_THRESHOLD = 0.5     # raw must exceed tau by this to enter zone
ACTIVATION_CUTOFF = 0.01       # intensity starts margin beyond this point
EPSILON = 1e-4                 # minimum intensity floor
MAX_INTENSITY = 10.0           # intensity cap (drift safety)
SCAN_SCALE = 0.01              # max absolute scan movement before /std
SCAN_STD_FLOOR = 0.5           # caps low-std scan amplification
DEFAULT_TAG_DIM = 16
DEFAULT_READ_SIG_DIM = 24
DEFAULT_WRITE_SIG_DIM = 24
DEFAULT_READ_NORM_SIG_DIM = 1
DEFAULT_WRITE_NORM_SIG_DIM = 1


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


def fixed_sig_proj_init():
    """Fixed column-orthogonal operator-signature projection."""
    def init(key, shape, dtype=jnp.float32):
        x = jax.random.normal(key, shape, dtype)
        q, _ = jnp.linalg.qr(x)
        return q.astype(dtype)
    return init


def _norm_signature(x, dim):
    if dim <= 0:
        return jnp.zeros(x.shape[:-1] + (0,), dtype=jnp.float32)
    n = jnp.linalg.norm(x.astype(jnp.float32), axis=-1, keepdims=True)
    z = jnp.tanh(jnp.log(n + 1e-8))
    powers = jnp.arange(1, dim + 1, dtype=jnp.float32)
    return z ** powers


def build_route_signature(tag, read_unit, write_unit, read_sig_proj, write_sig_proj,
                          read_norm_sig_dim=DEFAULT_READ_NORM_SIG_DIM,
                          write_norm_sig_dim=DEFAULT_WRITE_NORM_SIG_DIM):
    tag_f = tag.astype(jnp.float32)
    tag_unit = tag_f / (jnp.linalg.norm(tag_f, axis=-1, keepdims=True) + 1e-8)
    rproj = jax.lax.stop_gradient(read_sig_proj.astype(jnp.float32))
    wproj = jax.lax.stop_gradient(write_sig_proj.astype(jnp.float32))
    read_sig = read_unit.astype(jnp.float32) @ rproj
    write_sig = write_unit.astype(jnp.float32) @ wproj
    read_sig = read_sig / (jnp.linalg.norm(read_sig, axis=-1, keepdims=True) + 1e-8)
    write_sig = write_sig / (jnp.linalg.norm(write_sig, axis=-1, keepdims=True) + 1e-8)
    read_norm_sig = _norm_signature(read_unit, read_norm_sig_dim)
    write_norm_sig = _norm_signature(write_unit, write_norm_sig_dim)
    route = jnp.concatenate(
        [tag_unit, read_sig, write_sig, read_norm_sig, write_norm_sig],
        axis=-1)
    return route / (jnp.linalg.norm(route, axis=-1, keepdims=True) + 1e-8)


def read_write_sig_norm_stats(read, write, read_sig_proj, write_sig_proj):
    r = read.astype(jnp.float32)
    w = write.astype(jnp.float32)
    r = r / (jnp.linalg.norm(r, axis=-1, keepdims=True) + 1e-8)
    w = w / (jnp.linalg.norm(w, axis=-1, keepdims=True) + 1e-8)
    read_sig = r @ jax.lax.stop_gradient(read_sig_proj.astype(jnp.float32))
    write_sig = w @ jax.lax.stop_gradient(write_sig_proj.astype(jnp.float32))
    read_norms = jax.lax.stop_gradient(jnp.linalg.norm(read_sig, axis=-1))
    write_norms = jax.lax.stop_gradient(jnp.linalg.norm(write_sig, axis=-1))
    return read_norms.mean(), read_norms.std(), write_norms.mean(), write_norms.std()


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
                     read_norm_sig_dim=DEFAULT_READ_NORM_SIG_DIM,
                     write_norm_sig_dim=DEFAULT_WRITE_NORM_SIG_DIM,
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
        den           = max(誇 intensity, 1.0)

    scan = scan_scale * tanh(scan_bias)
    tau = s_mean + tau_offset * s_std - scan / max(s_std, scan_std_floor).
    All v4.1 constants are closure-baked.

    `analysis=False` (default, train path): returns a SLIM 13-tuple that
    skips distribution-shape stats (skew/kurt), boundary/entropy
    counters and intensity-cap fraction. XLA DCE's the unused work.
    `analysis=True`: returns the SLIM tuple followed by 11 extra
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
        P(),                     # gate_sum scalar (誇gate, observational)
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
    _out_specs = (_slim_out_specs + _analysis_extra_specs
                  if analysis else _slim_out_specs)

    @partial(shard_map, mesh=mesh,
             in_specs=(P('data', None, None),    # x [B,S,D]
                       P('data', None, None),    # h [B,S,d_bn]
                       P('model', None),          # tag [N_local, tag_dim]
                       P('data', None, None),    # tau_offset [B,S,1]
                       P('data', None, None),    # scan_bias [B,S,1]
                       P('model', None),          # read [N_local, D]
                       P('model', None),          # write [N_local, D]
                       P(None, None),             # read_sig_proj [D,read_sig_dim]
                       P(None, None)),            # write_sig_proj [D,write_sig_dim]
             out_specs=_out_specs,
             check_rep=False)
    def fused_gate_srw(x, h, tag_local, tau_offset, scan_bias,
                       read_local, write_local, read_sig_proj, write_sig_proj):
        N_local = tag_local.shape[0]
        nc = max(1, N_local // max_chunk_size)
        while N_local % nc != 0 and nc < N_local:
            nc += 1
        cs = N_local // nc

        B, S, D = x.shape
        h_bf = h.astype(jnp.bfloat16)
        x_bf = x.astype(jnp.bfloat16)
        tag_bf = tag_local.astype(jnp.bfloat16)
        read_bf = read_local.astype(jnp.bfloat16)
        write_bf = write_local.astype(jnp.bfloat16)
        z1 = jnp.zeros((B, S, 1))

        def route_chunk(start):
            tc = jax.lax.dynamic_slice_in_dim(tag_bf, start, cs, axis=0)
            rc = jax.lax.dynamic_slice_in_dim(read_bf, start, cs, axis=0)
            wc = jax.lax.dynamic_slice_in_dim(write_bf, start, cs, axis=0)
            rc_f = rc.astype(jnp.float32)
            wc_f = wc.astype(jnp.float32)
            route = build_route_signature(
                tc, rc_f, wc_f, read_sig_proj, write_sig_proj,
                read_norm_sig_dim, write_norm_sig_dim)
            return route.astype(jnp.bfloat16), rc_f.astype(jnp.bfloat16), wc_f.astype(jnp.bfloat16)

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
        scan = _scan_scale * jnp.tanh(scan_bias)
        tau = s_mean + tau_offset * s_std - scan / jnp.maximum(s_std, _scan_std_floor)

        if analysis:
            global_cube = jax.lax.psum(local_cube, 'model')
            global_quad = jax.lax.psum(local_quad, 'model')
            # Skewness via E[(X-關)^3] = E[X^3] - 3關?짼 - 關쨀
            cube_mean = global_cube / N_total
            central_third = cube_mean - 3.0 * s_mean * (s_std ** 2) - s_mean ** 3
            score_skew = jax.lax.stop_gradient((central_third / (s_std ** 3 + 1e-8)).mean())
            # Kurtosis via E[(X-關)^4] = E[X^4] - 4關E[X^3] + 6關짼?짼 + 3關??            quad_mean = global_quad / N_total
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

        # --- Pass 2: gate + srw fused (scan + checkpoint) ---
        # v4.1 diagnostic: ceiling on intensity relative to cap (1e-3 below).
        if analysis:
            _int_cap_thresh = _eps + _max_int - jnp.float32(1e-3)

            @jax.checkpoint
            def gate_srw_step(carry, i):
                (out, total_weighted_cost, total_gate_max, total_active,
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

            (raw_out, total_weighted_cost, total_gate_max, total_active, total_strong,
             total_phi_binary, total_den_cost, total_activation_cost,
             total_current_cost, total_z_lt_075, total_z_lt_030,
             total_g_log_g, total_dead_penalty, total_dead_count,
             total_int_max, total_int_cap_count), _ = jax.lax.scan(
                gate_srw_step,
                (jnp.zeros((B, S, D), dtype=jnp.float32),
                 z1, jnp.full((B, S, 1), -1e9), z1, z1, z1, z1, z1, z1, z1, z1, z1,
                 jnp.float32(0.0), jnp.float32(0.0),
                 jnp.float32(0.0), jnp.float32(0.0)),
                jnp.arange(nc))
        else:
            @jax.checkpoint
            def gate_srw_step(carry, i):
                (out, total_weighted_cost, total_gate_max, total_active,
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
                chunk_intensity = gate.sum(axis=-1, keepdims=True)
                chunk_active = (activation > 0.5).astype(jnp.float32).sum(axis=-1, keepdims=True)
                chunk_strong = (activation > 0.9).astype(jnp.float32).sum(axis=-1, keepdims=True)
                # 誇intensity feeds den (consumed after scan, not returned).
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
                        jnp.maximum(total_gate_max, gate.max(axis=-1, keepdims=True)),
                        total_active + chunk_active,
                        total_strong + chunk_strong,
                        total_den_cost + chunk_intensity,
                        total_activation_cost,
                        total_current_cost,
                        total_dead_penalty + chunk_dead_penalty,
                        total_dead_count + chunk_dead_count,
                        jnp.maximum(total_int_max, chunk_int_max)), None

            (raw_out, total_weighted_cost, total_gate_max, total_active, total_strong,
             total_den_cost, total_activation_cost, total_current_cost,
             total_dead_penalty, total_dead_count,
             total_int_max), _ = jax.lax.scan(
                gate_srw_step,
                (jnp.zeros((B, S, D), dtype=jnp.float32),
                 z1, jnp.full((B, S, 1), -1e9), z1, z1, z1, z1, z1,
                 jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0)),
                jnp.arange(nc))

        global_weighted_cost = jax.lax.psum(total_weighted_cost, 'model')  # 誇gate
        # v4.1 den = 誇intensity (boundary neurons get EPSILON den penalty
        # without matching numerator ??structural bimodality pressure).
        # Effective v4.1.5 denominator: max(sum(activation * intensity), 1.0).
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
        es_out = global_weighted_cost.mean()          # 誇gate observational
        active_n_mean = global_active.mean()
        tau_abs_mean = jax.lax.stop_gradient(tau).mean()
        dead_penalty_out = jax.lax.psum(total_dead_penalty, 'model')
        dead_count_out = jax.lax.stop_gradient(
            jax.lax.psum(total_dead_count, 'model'))
        # pmax has no VJP ??wrap the input in stop_gradient.
        int_max_out = jax.lax.pmax(
            jax.lax.stop_gradient(total_int_max), 'model')

        den_cost_mean = global_den_cost.mean()
        activation_cost_mean = global_activation_cost.mean()
        current_cost_mean = global_current_cost.mean()

        slim_out = (out.astype(jnp.float32), active_frac, global_gate_max, score_lb,
                    score_std_out, es_out, active_n_mean, strong_frac, z_mean_active,
                    tau_abs_mean, dead_penalty_out, dead_count_out, int_max_out,
                    den_cost_mean, activation_cost_mean, current_cost_mean)
        if not analysis:
            return slim_out

        # --- Analysis-only extras ---
        phi_binary_frac = jax.lax.psum(total_phi_binary, 'model') / N_total
        # Safety floor ??active can collapse to 0 at init; clamp to 1.0.
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
        return slim_out + (phi_binary_frac, z_lt_075_frac, z_lt_030_frac,
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
                            read_norm_sig_dim=DEFAULT_READ_NORM_SIG_DIM,
                            write_norm_sig_dim=DEFAULT_WRITE_NORM_SIG_DIM,
                            analysis=False):
    """Fused Q+K shard_map: two routes sharing same pool in one shard_map call.

    h is [B,S,2,d_bn] (h_Q, h_K stacked on axis=2).
    tau_offset and scan_bias are [B,S,2,1].
    x @ read.T computed once (shared by both routes).
    Scores stats computed independently per route.
    Returns out [B,S,2,D], active [B,S,1], gate_max [B,S,1].

    v4.1 gate: activation 횞 intensity (see make_sharded_srw docstring).
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
    _out_specs = (_slim_out_specs + _analysis_extra_specs
                  if analysis else _slim_out_specs)

    @partial(shard_map, mesh=mesh,
             in_specs=(P('data', None, None),        # x [B,S,D]
                       P('data', None, None, None),  # h [B,S,2,d_bn]
                       P('model', None),              # tag [N_local, tag_dim]
                       P('data', None, None, None),  # tau_offset [B,S,2,1]
                       P('data', None, None, None),  # scan_bias [B,S,2,1]
                       P('model', None),              # read [N_local, D]
                       P('model', None),              # write [N_local, D]
                       P(None, None),                 # read_sig_proj [D,read_sig_dim]
                       P(None, None)),                # write_sig_proj [D,write_sig_dim]
             out_specs=_out_specs,
             check_rep=False)
    def fused_gate_srw_paired(x, h, tag_local, tau_offset, scan_bias,
                              read_local, write_local, read_sig_proj, write_sig_proj):
        N_local = tag_local.shape[0]
        nc = max(1, N_local // max_chunk_size)
        while N_local % nc != 0 and nc < N_local:
            nc += 1
        cs = N_local // nc

        B, S, D = x.shape
        # h: [B,S,2,d_route], tau_offset/scan_bias: [B,S,2,1]
        h_bf = h.astype(jnp.bfloat16)
        x_bf = x.astype(jnp.bfloat16)
        tag_bf = tag_local.astype(jnp.bfloat16)
        read_bf = read_local.astype(jnp.bfloat16)
        write_bf = write_local.astype(jnp.bfloat16)
        z1_r = jnp.zeros((B, S, 2, 1))

        def route_chunk(start):
            tc = jax.lax.dynamic_slice_in_dim(tag_bf, start, cs, axis=0)
            rc = jax.lax.dynamic_slice_in_dim(read_bf, start, cs, axis=0)
            wc = jax.lax.dynamic_slice_in_dim(write_bf, start, cs, axis=0)
            rc_f = rc.astype(jnp.float32)
            wc_f = wc.astype(jnp.float32)
            route = build_route_signature(
                tc, rc_f, wc_f, read_sig_proj, write_sig_proj,
                read_norm_sig_dim, write_norm_sig_dim)
            return route.astype(jnp.bfloat16), rc_f.astype(jnp.bfloat16), wc_f.astype(jnp.bfloat16)

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
        scan = _scan_scale * jnp.tanh(scan_bias)
        tau = s_mean + tau_offset * s_std - scan / jnp.maximum(s_std, _scan_std_floor)

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
                (out, total_weighted_cost, total_gate_max, total_active,
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

            (raw_out, total_weighted_cost, total_gate_max, total_active, total_strong,
             total_phi_binary, total_den_cost, total_activation_cost,
             total_current_cost, total_z_lt_075, total_z_lt_030,
             total_g_log_g, total_dead_penalty, total_dead_count,
             total_int_max, total_int_cap_count), _ = jax.lax.scan(
                gate_srw_step,
                (jnp.zeros((B, S, 2, D), dtype=jnp.float32),
                 z1_r, jnp.full((B, S, 2, 1), -1e9),
                 z1_r, z1_r, z1_r, z1_r, z1_r, z1_r, z1_r, z1_r, z1_r,
                 jnp.float32(0.0), jnp.float32(0.0),
                 jnp.float32(0.0), jnp.float32(0.0)),
                jnp.arange(nc))
        else:
            @jax.checkpoint
            def gate_srw_step(carry, i):
                (out, total_weighted_cost, total_gate_max, total_active,
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
                        jnp.maximum(total_gate_max, gate.max(axis=-1, keepdims=True)),
                        total_active + chunk_active,
                        total_strong + chunk_strong,
                        total_den_cost + chunk_intensity,
                        total_activation_cost,
                        total_current_cost,
                        total_dead_penalty + chunk_dead_penalty,
                        total_dead_count + chunk_dead_count,
                        jnp.maximum(total_int_max, chunk_int_max)), None

            (raw_out, total_weighted_cost, total_gate_max, total_active, total_strong,
             total_den_cost, total_activation_cost, total_current_cost,
             total_dead_penalty, total_dead_count,
             total_int_max), _ = jax.lax.scan(
                gate_srw_step,
                (jnp.zeros((B, S, 2, D), dtype=jnp.float32),
                 z1_r, jnp.full((B, S, 2, 1), -1e9),
                 z1_r, z1_r, z1_r, z1_r, z1_r,
                 jnp.float32(0.0), jnp.float32(0.0), jnp.float32(0.0)),
                jnp.arange(nc))

        # Normalize per route independently
        global_weighted_cost = jax.lax.psum(total_weighted_cost, 'model')   # 誇gate (log)
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
        if not analysis:
            return slim_out

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
        return slim_out + (phi_binary_frac_mean, z_lt_075_frac, z_lt_030_frac,
                           score_skew, active_per_token_std, gate_entropy,
                           den_cost_out, activation_cost_out, current_cost_out,
                           score_kurt, int_cap_frac_out)

    return fused_gate_srw_paired


# ================================================================
# 4. NeuronPool -- emb + w_read + w_write
# ================================================================

class NeuronPool(nn.Module):
    n_qk: int
    n_v: int
    n_know: int
    d_model: int
    d_route: int
    tag_dim: int = DEFAULT_TAG_DIM
    read_sig_dim: int = DEFAULT_READ_SIG_DIM
    write_sig_dim: int = DEFAULT_WRITE_SIG_DIM
    read_norm_sig_dim: int = DEFAULT_READ_NORM_SIG_DIM
    write_norm_sig_dim: int = DEFAULT_WRITE_NORM_SIG_DIM

    def setup(self):
        expected_route = (self.tag_dim + self.read_sig_dim + self.write_sig_dim
                          + self.read_norm_sig_dim + self.write_norm_sig_dim)
        if expected_route != self.d_route:
            raise ValueError(
                f"d_route must equal tag_dim + read_sig_dim + write_sig_dim "
                f"+ read_norm_sig_dim + write_norm_sig_dim, got "
                f"d_route={self.d_route}, tag_dim={self.tag_dim}, "
                f"read_sig_dim={self.read_sig_dim}, write_sig_dim={self.write_sig_dim}, "
                f"read_norm_sig_dim={self.read_norm_sig_dim}, "
                f"write_norm_sig_dim={self.write_norm_sig_dim}")
        db = self.tag_dim
        dm = self.d_model

        # Small learned route tags. The full route is concat(tag,
        # fixed read/write operator signature) and is forward-normalized.
        self.qk_emb = self.param('qk_emb', unit_norm_init(), (self.n_qk, db))
        self.v_emb = self.param('v_emb', unit_norm_init(), (self.n_v, db))
        self.know_emb = self.param('know_emb', unit_norm_init(), (self.n_know, db))

        # Read (what to extract from x): free norm in v4.1.5.1
        self.qk_read = self.param('qk_read', unit_norm_init(), (self.n_qk, dm))
        self.v_read = self.param('v_read', unit_norm_init(), (self.n_v, dm))
        self.know_read = self.param('know_read', unit_norm_init(), (self.n_know, dm))

        # Write (direction to push): free norm in v4.1.5.1
        self.qk_write = self.param('qk_write', unit_norm_init(), (self.n_qk, dm))
        self.v_write = self.param('v_write', unit_norm_init(), (self.n_v, dm))
        self.know_write = self.param('know_write', unit_norm_init(), (self.n_know, dm))

        # Fixed random projections for operation-derived routing signatures.
        # Stored as params for checkpoint portability; forward uses stop_gradient
        # and train_jax.py excludes *_sig_proj from optimizer weight decay.
        self.qk_read_sig_proj = self.param(
            'qk_read_sig_proj', fixed_sig_proj_init(), (dm, self.read_sig_dim))
        self.qk_write_sig_proj = self.param(
            'qk_write_sig_proj', fixed_sig_proj_init(), (dm, self.write_sig_dim))
        self.v_read_sig_proj = self.param(
            'v_read_sig_proj', fixed_sig_proj_init(), (dm, self.read_sig_dim))
        self.v_write_sig_proj = self.param(
            'v_write_sig_proj', fixed_sig_proj_init(), (dm, self.write_sig_dim))
        self.know_read_sig_proj = self.param(
            'know_read_sig_proj', fixed_sig_proj_init(), (dm, self.read_sig_dim))
        self.know_write_sig_proj = self.param(
            'know_write_sig_proj', fixed_sig_proj_init(), (dm, self.write_sig_dim))

        # Per-pool learnable output scale (init=?쉊_model, WD excluded)
        self.qk_scale = self.param('qk_scale',
            lambda k, s, d: jnp.full(s, jnp.sqrt(d)), (1,), self.d_model)
        self.v_scale = self.param('v_scale',
            lambda k, s, d: jnp.full(s, jnp.sqrt(d)), (1,), self.d_model)
        self.know_scale = self.param('know_scale',
            lambda k, s, d: jnp.full(s, jnp.sqrt(d)), (1,), self.d_model)

        # v4.1: dynamic-tau alpha params removed. tau = tau_base only.


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
        # Bounded scan axis. Zero-init preserves old behavior at step 0.
        self.scan_bias_attn = nn.Dense(3, name='scan_bias_attn',
            kernel_init=nn.initializers.zeros,
            bias_init=nn.initializers.zeros)
        self.scan_bias_know = nn.Dense(1, name='scan_bias_know',
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
    qk_emb = pool_params['qk_emb']
    qk_read = pool_params['qk_read']
    qk_write = pool_params['qk_write']
    qk_read_sig_proj = pool_params['qk_read_sig_proj']
    qk_write_sig_proj = pool_params['qk_write_sig_proj']
    v_emb = pool_params['v_emb']
    v_read = pool_params['v_read']
    v_write = pool_params['v_write']
    v_read_sig_proj = pool_params['v_read_sig_proj']
    v_write_sig_proj = pool_params['v_write_sig_proj']

    # emb used as-is; *_unit names kept for downstream readability.
    qk_emb_unit = qk_emb
    v_emb_unit = v_emb

    # Emb-norm monitoring (observational ??stop_gradient to avoid VJP hazards).
    _qk_emb_norms = jax.lax.stop_gradient(jnp.linalg.norm(qk_emb, axis=-1))
    qk_emb_norm_mean = _qk_emb_norms.mean()
    qk_emb_norm_min = _qk_emb_norms.min()
    qk_emb_norm_std = _qk_emb_norms.std()
    _v_emb_norms = jax.lax.stop_gradient(jnp.linalg.norm(v_emb, axis=-1))
    v_emb_norm_mean = _v_emb_norms.mean()
    v_emb_norm_min = _v_emb_norms.min()
    v_emb_norm_std = _v_emb_norms.std()
    if analysis:
        qk_emb_norm_max = _qk_emb_norms.max()
        v_emb_norm_max = _v_emb_norms.max()

    rng, rng_drop = jax.random.split(rng)
    h_all = x @ router_params['proj_attn']['kernel'] + router_params['proj_attn']['bias']
    h_all = safe_dropout(h_all, router_dropout, deterministic, rng_drop)
    h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)

    tau_all = x @ router_params['tau_attn']['kernel'] + router_params['tau_attn']['bias']
    scan_bias_all = x @ router_params['scan_bias_attn']['kernel'] + router_params['scan_bias_attn']['bias']
    if analysis:
        _tau_all_sg = jax.lax.stop_gradient(tau_all)
        attn_tau_std = _tau_all_sg.std(axis=(0, 1))  # [3] Q/K/V
        attn_tau_kernel_norm = jnp.sqrt(
            jnp.sum(jax.lax.stop_gradient(router_params['tau_attn']['kernel']) ** 2) + 1e-12)

    qk_scale = pool_params['qk_scale']
    v_scale = pool_params['v_scale']

    fused_single, fused_paired = sharded_fns
    h_QK = jnp.stack([h_Q, h_K], axis=2)
    tau_QK = jnp.stack([tau_all[:, :, 0:1], tau_all[:, :, 1:2]], axis=2)
    scan_bias_QK = jnp.stack([scan_bias_all[:, :, 0:1], scan_bias_all[:, :, 1:2]], axis=2)
    qk_ret = fused_paired(x, h_QK, qk_emb_unit, tau_QK, scan_bias_QK,
                           qk_read, qk_write, qk_read_sig_proj, qk_write_sig_proj)
    (QK_out, qk_active, qk_raw_gmax, qk_lb, qk_sstd, qk_es, qk_anm,
     qk_strong, qk_z_act, qk_tau_abs,
     qk_dead_pen, qk_dead_cnt, qk_int_max,
     qk_den_cost_mean, qk_activation_cost_mean, qk_current_cost_mean) = qk_ret[:16]
    if analysis:
        (qk_phi_bin, qk_z075, qk_z030, qk_skew, qk_apt_std, qk_entropy,
         qk_den_cost, qk_activation_cost, qk_current_cost,
         qk_kurt, qk_int_cap) = qk_ret[16:]
        qk_raw_norm = jnp.linalg.norm(QK_out, axis=-1).mean()
    Q = QK_out[:, :, 0, :] * qk_scale
    K = QK_out[:, :, 1, :] * qk_scale
    v_ret = fused_single(x, h_V, v_emb_unit, tau_all[:, :, 2:3],
                         scan_bias_all[:, :, 2:3], v_read, v_write,
                         v_read_sig_proj, v_write_sig_proj)
    (V, v_active, v_raw_gmax, v_lb, v_sstd, v_es, v_anm,
     v_strong, v_z_act, v_tau_abs,
     v_dead_pen, v_dead_cnt, v_int_max,
     v_den_cost_mean, v_activation_cost_mean, v_current_cost_mean) = v_ret[:16]
    if analysis:
        (v_phi_bin, v_z075, v_z030, v_skew, v_apt_std, v_entropy,
         v_den_cost, v_activation_cost, v_current_cost,
         v_kurt, v_int_cap) = v_ret[16:]
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

    # Load balance loss from gate distributions + tau regularization
    tau_reg = jnp.maximum(tau_all, 0.0).mean() * 0.01
    # TODO(v4.2?): /3 洹쇨굅 遺덈텇紐?(v3.9.1 ?붿쟻). ?꾩옱 pool 2媛?qk, v) ?⑹쓣 3?쇰줈 ?섎닎.
    # ?섎룄媛 Q/K/V 3-route ?됯퇏?대㈃ paired ?대??먯꽌 Q/K lb 遺꾨━ ?꾩슂.
    # ?섎룄媛 ?됯퇏?대㈃ /2媛 留욎쓬. lb_weight ???꾩뿉 寃곗젙 ?꾩슂.
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
    # v4.1 explore: tau_offset [B, S, 3] passed to train_step.
    attn_tau_offset = tau_all
    slim_ret = (out, aux, qk_active.mean(), v_active.mean(), attn_raw_gmax,
                attn_score_std, attn_gate_sum, attn_active_n_mean,
                attn_out_norm, attn_tau_mean,
                attn_strong,
                qk_strong.mean(), v_strong.mean(),
                attn_qk_z_mean_active, attn_v_z_mean_active,
                attn_tau_abs_mean,
                qk_emb_norm_mean, v_emb_norm_mean,
                qk_emb_norm_min, qk_emb_norm_std,
                v_emb_norm_min, v_emb_norm_std,
                attn_dead_penalty, attn_dead_count,
                attn_tau_offset,
                attn_int_max,
                attn_den_cost_mean, attn_activation_cost_mean,
                attn_current_cost_mean)
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
        qk_emb_norm_max, v_emb_norm_max,
        attn_score_kurt,
        attn_int_cap_frac,
    )


def _know_forward(x, pool_params, router_params, rng,
                  router_dropout, dropout_rate, deterministic,
                  sharded_fns, analysis=False):
    """v4.1: sharded-only. sharded_fns=(fused_single, fused_paired) required.

    `analysis` see _attn_forward docstring.
    """
    know_emb = pool_params['know_emb']
    know_read = pool_params['know_read']
    know_write = pool_params['know_write']
    know_read_sig_proj = pool_params['know_read_sig_proj']
    know_write_sig_proj = pool_params['know_write_sig_proj']

    rng, rng_drop = jax.random.split(rng)
    h = x @ router_params['proj_know']['kernel'] + router_params['proj_know']['bias']
    h = safe_dropout(h, router_dropout, deterministic, rng_drop)

    # emb used as-is; *_unit name kept for downstream readability.
    know_emb_unit = know_emb
    tau = x @ router_params['tau_know']['kernel'] + router_params['tau_know']['bias']
    scan_bias = x @ router_params['scan_bias_know']['kernel'] + router_params['scan_bias_know']['bias']
    if analysis:
        know_tau_std = jax.lax.stop_gradient(tau).std()
        know_tau_kernel_norm = jnp.sqrt(
            jnp.sum(jax.lax.stop_gradient(router_params['tau_know']['kernel']) ** 2) + 1e-12)

    know_scale = pool_params['know_scale']

    fused_single, _ = sharded_fns
    know_ret = fused_single(x, h, know_emb_unit, tau, scan_bias,
                            know_read, know_write,
                            know_read_sig_proj, know_write_sig_proj)
    (out, active_frac, raw_gate_max, lb_loss, score_std, gate_sum, active_n_mean,
     strong_frac, z_mean_act, know_tau_abs_mean,
     know_dead_penalty, know_dead_count, know_int_max,
     know_den_cost_mean, know_activation_cost_mean, know_current_cost_mean) = know_ret[:16]
    if analysis:
        (phi_binary_frac, know_z_lt_075_frac, know_z_lt_030_frac,
         know_score_skew, know_active_per_token_std, know_gate_entropy,
         know_den_cost, know_activation_cost, know_current_cost,
         know_score_kurt, know_int_cap_frac) = know_ret[16:]
        know_raw_out_norm = jnp.linalg.norm(out, axis=-1).mean()
    out = out * know_scale
    know_out_norm = jnp.linalg.norm(out, axis=-1).mean()
    rng, rng_out = jax.random.split(rng)
    out = safe_dropout(out, dropout_rate, deterministic, rng_out)

    tau_reg = jnp.maximum(tau, 0.0).mean() * 0.01
    aux = lb_loss + tau_reg
    _know_emb_norms = jax.lax.stop_gradient(jnp.linalg.norm(know_emb, axis=-1))
    emb_norm_val = _know_emb_norms.mean()
    know_emb_norm_min = _know_emb_norms.min()
    know_emb_norm_std = _know_emb_norms.std()
    if analysis:
        know_emb_norm_max = _know_emb_norms.max()
    read_norm_val = jnp.linalg.norm(know_read, axis=-1).mean()
    write_norm_val = jnp.linalg.norm(know_write, axis=-1).mean()
    know_tau_mean = tau.mean()
    know_strong = strong_frac.mean()
    know_z_mean_active = z_mean_act.mean()
    slim_ret = (out, aux, active_frac, raw_gate_max, score_std, gate_sum, active_n_mean,
                emb_norm_val, read_norm_val, write_norm_val, know_out_norm,
                know_tau_mean, know_strong, know_z_mean_active,
                know_tau_abs_mean,
                know_emb_norm_min, know_emb_norm_std,
                know_dead_penalty, know_dead_count,
                tau,
                know_int_max,
                know_den_cost_mean, know_activation_cost_mean,
                know_current_cost_mean)
    if not analysis:
        return slim_ret

    know_phi_binary = phi_binary_frac.mean()
    return slim_ret + (
        know_raw_out_norm,
        know_tau_std, know_tau_kernel_norm,
        know_z_lt_075_frac, know_z_lt_030_frac,
        know_score_skew, know_active_per_token_std, know_gate_entropy,
        know_den_cost,
        know_activation_cost, know_current_cost,
        know_emb_norm_max,
        know_score_kurt,
        know_phi_binary,
        know_int_cap_frac,
    )


# ================================================================
# 7. Flax modules (init path only)
# ================================================================

class AttentionCircuit(nn.Module):
    """Container for expand_O; real forward path is _attn_forward()."""
    d_model: int
    n_heads: int
    dropout_rate: float = 0.1

    def setup(self):
        self.expand_O = nn.Dense(
            self.d_model, use_bias=False, kernel_init=scaled_normal(0.02))


class DAWNBlock(nn.Module):
    """Container for per-layer norms + attn (expand_O) submodules.
    The real forward path is scan_body in DAWN.__call__."""
    d_model: int
    n_heads: int
    dropout_rate: float = 0.1

    def setup(self):
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()
        self.attn = AttentionCircuit(
            d_model=self.d_model, n_heads=self.n_heads,
            dropout_rate=self.dropout_rate)


# ================================================================
# 8. DAWN Model
# ================================================================

class DAWN(nn.Module):
    """DAWN-Spatial v3.8: Sense-Read-Write."""
    __version__ = "spatial-r1-v4.1.5"

    vocab_size: int = 30000
    d_model: int = 384
    n_layers: int = 12
    n_heads: int = 6
    max_seq_len: int = 512
    dropout_rate: float = 0.1
    gradient_checkpointing: bool = False

    d_route: int = (DEFAULT_TAG_DIM + DEFAULT_READ_SIG_DIM + DEFAULT_WRITE_SIG_DIM
                    + DEFAULT_READ_NORM_SIG_DIM + DEFAULT_WRITE_NORM_SIG_DIM)
    tag_dim: int = DEFAULT_TAG_DIM
    read_sig_dim: int = DEFAULT_READ_SIG_DIM
    write_sig_dim: int = DEFAULT_WRITE_SIG_DIM
    read_norm_sig_dim: int = DEFAULT_READ_NORM_SIG_DIM
    write_norm_sig_dim: int = DEFAULT_WRITE_NORM_SIG_DIM
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
        expected_route = (self.tag_dim + self.read_sig_dim + self.write_sig_dim
                          + self.read_norm_sig_dim + self.write_norm_sig_dim)
        if expected_route != self.d_route:
            raise ValueError(
                f"d_route must equal tag_dim + read_sig_dim + write_sig_dim "
                f"+ read_norm_sig_dim + write_norm_sig_dim, got "
                f"d_route={self.d_route}, tag_dim={self.tag_dim}, "
                f"read_sig_dim={self.read_sig_dim}, write_sig_dim={self.write_sig_dim}, "
                f"read_norm_sig_dim={self.read_norm_sig_dim}, "
                f"write_norm_sig_dim={self.write_norm_sig_dim}")
        self.token_emb = nn.Embed(
            self.vocab_size, self.d_model, embedding_init=scaled_normal(0.02))
        self.pos_emb = nn.Embed(
            self.max_seq_len, self.d_model, embedding_init=scaled_normal(0.02))
        self.neuron_pool = NeuronPool(
            n_qk=self.n_qk, n_v=self.n_v, n_know=self.n_know,
            d_model=self.d_model, d_route=self.d_route,
            tag_dim=self.tag_dim, read_sig_dim=self.read_sig_dim,
            write_sig_dim=self.write_sig_dim,
            read_norm_sig_dim=self.read_norm_sig_dim,
            write_norm_sig_dim=self.write_norm_sig_dim)
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
                 deterministic=False, sharded_fns=None, analysis=False):
        """v4.1 forward.

        `analysis=False` (train path): result dict contains only the
        fields the train loop and REGULAR logging consume ??skew/kurt,
        phi_binary, z_lt_*, gate_entropy, apt_std, int_cap_frac, debug
        norms, raw norms, tau_std / tau_kernel_norm, emb_norm_max are
        NOT computed (XLA DCE's the unused work from the fused kernels
        and forward helpers). Caller must pass sharded_fns built with
        `analysis=False`.
        `analysis=True`: result dict adds all of the above observational
        fields. Used by analysis_step at val time. Caller must pass
        sharded_fns built with `analysis=True`.
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
            attn_qk_strong_all = _z
            attn_v_strong_all = _z
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
            attn_tau_abs_all = _z
            know_tau_abs_all = _z
            attn_qk_emb_n_mean_all = _z
            attn_v_emb_n_mean_all = _z
            know_emb_n_std_all = _z
            attn_qk_emb_n_min_all = _z
            attn_qk_emb_n_std_all = _z
            attn_v_emb_n_min_all = _z
            attn_v_emb_n_std_all = _z
            know_emb_n_min_all = _z
            attn_dead_penalty_all = _z
            know_dead_penalty_all = _z
            attn_dead_count_all = _z
            know_dead_count_all = _z
            attn_tau_offset_all = _z
            know_tau_offset_all = _z
            attn_int_max_all = _z
            know_int_max_all = _z
            attn_den_cost_mean_all = _z
            know_den_cost_mean_all = _z
            attn_activation_cost_mean_all = _z
            know_activation_cost_mean_all = _z
            attn_current_cost_mean_all = _z
            know_current_cost_mean_all = _z
            # Trigger Flax param realization for all submodules (init-only).
            # The real forward runs through scan_body in the else branch and
            # accesses params by path, not via these module calls.
            _ = self.neuron_pool.qk_emb  # triggers NeuronPool.setup ??all pool params
            _ = self.router.proj_attn(x)
            _ = self.router.proj_know(x)
            _ = self.router.tau_attn(x)
            _ = self.router.tau_know(x)
            _ = self.router.scan_bias_attn(x)
            _ = self.router.scan_bias_know(x)
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
                rng, rng_attn, rng_know = jax.random.split(rng, 3)

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
                 a_current_cost_mean) = attn_ret[:29]
                if analysis:
                    (a_qk_raw_norm, a_v_raw_norm,
                     a_q_norm, a_k_norm, a_v_norm_dbg, a_logit_max, a_o_input_norm,
                     a_qk_phi_bin, a_v_phi_bin,
                     a_tau_std, a_tau_kernel_norm,
                     a_z075, a_z030,
                     a_skew, a_apt_std, a_entropy,
                     a_den_cost, a_activation_cost, a_current_cost,
                     a_qk_emb_n_max, a_v_emb_n_max,
                     a_score_kurt, a_int_cap_frac) = attn_ret[29:]
                x = x + attn_out

                normed = _layer_norm(
                    x, bp['norm2']['scale'], bp['norm2']['bias'])
                know_ret = _know_forward(
                    normed, pool_params, router_params, rng_know,
                    self.router_dropout, self.dropout_rate, deterministic,
                    sharded_fns=_sharded, analysis=analysis)
                (know_out, know_aux, k_active, k_raw_gmax, k_sstd, k_gsum,
                 k_active_n_mean, k_emb_n, k_read_n, k_write_n, k_out_norm,
                 k_tau_mean, k_strong, k_z_act, k_tau_abs,
                 k_emb_n_min, k_emb_n_std,
                 k_dead_penalty, k_dead_count,
                 k_tau_offset,
                 k_int_max,
                 k_den_cost_mean, k_activation_cost_mean,
                 k_current_cost_mean) = know_ret[:24]
                if analysis:
                    (k_raw_out_norm,
                     k_tau_std, k_tau_kernel_norm,
                     k_z075, k_z030,
                     k_skew, k_apt_std, k_entropy,
                     k_den_cost, k_activation_cost, k_current_cost,
                     k_emb_n_max, k_score_kurt, k_phi_bin,
                     k_int_cap_frac) = know_ret[24:]
                x = x + know_out

                slim_ys = (attn_aux, know_aux,
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

            (attn_auxes, know_auxes,
             know_active_all, know_raw_gmax_all, know_sstd_all, know_gsum_all, know_active_n_mean_all,
             attn_qk_active_all, attn_v_active_all, attn_raw_gmax_all, attn_sstd_all, attn_gsum_all, attn_active_n_mean_all,
             k_emb_n_all, k_read_n_all, k_write_n_all,
             know_out_norm_all,
             attn_out_norm_all, attn_tau_mean_all, know_tau_mean_all,
             know_strong_all, attn_strong_all,
             attn_qk_strong_all, attn_v_strong_all,
             know_z_act_all, attn_qk_z_act_all, attn_v_z_act_all,
             attn_tau_abs_all, know_tau_abs_all,
             attn_qk_emb_n_mean_all, attn_v_emb_n_mean_all,
             know_emb_n_std_all,
             attn_qk_emb_n_min_all, attn_qk_emb_n_std_all,
             attn_v_emb_n_min_all, attn_v_emb_n_std_all,
             know_emb_n_min_all,
            attn_dead_penalty_all, know_dead_penalty_all,
            attn_dead_count_all, know_dead_count_all,
            attn_tau_offset_all, know_tau_offset_all,
            attn_int_max_all, know_int_max_all,
            attn_den_cost_mean_all, know_den_cost_mean_all,
            attn_activation_cost_mean_all, know_activation_cost_mean_all,
            attn_current_cost_mean_all, know_current_cost_mean_all) = scan_ys[:51]
            if analysis:
                (attn_qk_raw_norm_all, attn_v_raw_norm_all, know_raw_out_norm_all,
                 attn_q_norm_all, attn_k_norm_all, attn_v_norm_dbg_all,
                 attn_logit_max_all, attn_o_input_norm_all,
                 know_phi_bin_all, attn_qk_phi_bin_all, attn_v_phi_bin_all,
                 attn_tau_std_all, know_tau_std_all,
                 attn_tau_kernel_norm_all, know_tau_kernel_norm_all,
                 attn_z075_all, know_z075_all,
                 attn_z030_all, know_z030_all,
                 attn_skew_all, know_skew_all,
                 attn_apt_std_all, know_apt_std_all,
                 attn_entropy_all, know_entropy_all,
                 attn_den_cost_all, know_den_cost_all,
                 attn_activation_cost_all, know_activation_cost_all,
                 attn_current_cost_all, know_current_cost_all,
                 attn_qk_emb_n_max_all, attn_v_emb_n_max_all,
                 know_emb_n_max_all,
                 attn_score_kurt_all, know_score_kurt_all,
                 attn_int_cap_frac_all, know_int_cap_frac_all) = scan_ys[51:]
            # TODO(v4.2?): attn_aux???대? /3, know_aux???앹쭨 ??layer ?됯퇏 ??load balance 媛以묒튂媛
            # pool蹂?鍮꾨?移?QK/V??/3L, Know??/L, know媛 3諛?媛뺥븿). ?섎룄 ?뺤씤 ?꾩슂.
            total_aux = (attn_auxes + know_auxes).mean()

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
            'know_z_mean_active': know_z_act_all.mean(),

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

            'know_emb_norm': k_emb_n_all.mean(),
            'know_read_norm': k_read_n_all.mean(),
            'know_write_norm': k_write_n_all.mean(),

            'know_out_norm': know_out_norm_all.mean(),
            'attn_out_norm': attn_out_norm_all.mean(),
            'attn_tau_mean': attn_tau_mean_all.mean(),
            'know_tau_mean': know_tau_mean_all.mean(),
            'attn_tau_abs_mean': attn_tau_abs_all.mean(),
            'know_tau_abs_mean': know_tau_abs_all.mean(),
            'qk_emb_norm_mean': attn_qk_emb_n_mean_all.mean(),
            'qk_emb_norm_min': attn_qk_emb_n_min_all.min(),
            'qk_emb_norm_std': attn_qk_emb_n_std_all.mean(),
            'v_emb_norm_mean': attn_v_emb_n_mean_all.mean(),
            'v_emb_norm_min': attn_v_emb_n_min_all.min(),
            'v_emb_norm_std': attn_v_emb_n_std_all.mean(),
            'know_emb_norm_min': know_emb_n_min_all.min(),
            'know_emb_norm_std': know_emb_n_std_all.mean(),

            # v4.0.6: dead-only penalty (separate from aux; weighted in train loop).
            # Mean across layers so the training weight is layer-count-agnostic.
            'attn_dead_penalty': attn_dead_penalty_all.mean(),
            'know_dead_penalty': know_dead_penalty_all.mean(),
            'dead_penalty': (attn_dead_penalty_all.mean()
                             + know_dead_penalty_all.mean()),
            'attn_dead_count': attn_dead_count_all.mean(),
            'know_dead_count': know_dead_count_all.mean(),

            'per_layer_attn_out_norm': attn_out_norm_all,
            'per_layer_know_out_norm': know_out_norm_all,
            # v4.1 explore: per-layer tau_offset stacks for RPE exploration loss.
            # Shapes: attn [L, B, S, 3], know [L, B, S, 1].
            'attn_tau_offset': attn_tau_offset_all,
            'know_tau_offset': know_tau_offset_all,
            # v4.1.5 denominator diagnostic: sum(activation * intensity).
            'attn_int_max': attn_int_max_all.max(),
            'know_int_max': know_int_max_all.max(),
            'attn_gate_den_sum_mean': attn_den_cost_mean_all.mean(),
            'know_gate_den_sum_mean': know_den_cost_mean_all.mean(),
        }
        if not self.is_initializing():
            qk_read_m, qk_read_s, qk_write_m, qk_write_s = read_write_sig_norm_stats(
                pool_params['qk_read'], pool_params['qk_write'],
                pool_params['qk_read_sig_proj'], pool_params['qk_write_sig_proj'])
            v_read_m, v_read_s, v_write_m, v_write_s = read_write_sig_norm_stats(
                pool_params['v_read'], pool_params['v_write'],
                pool_params['v_read_sig_proj'], pool_params['v_write_sig_proj'])
            know_read_m, know_read_s, know_write_m, know_write_s = read_write_sig_norm_stats(
                pool_params['know_read'], pool_params['know_write'],
                pool_params['know_read_sig_proj'], pool_params['know_write_sig_proj'])
            result.update({
                'qk_read_sig_norm_mean': qk_read_m,
                'qk_read_sig_norm_std': qk_read_s,
                'qk_write_sig_norm_mean': qk_write_m,
                'qk_write_sig_norm_std': qk_write_s,
                'v_read_sig_norm_mean': v_read_m,
                'v_read_sig_norm_std': v_read_s,
                'v_write_sig_norm_mean': v_write_m,
                'v_write_sig_norm_std': v_write_s,
                'know_read_sig_norm_mean': know_read_m,
                'know_read_sig_norm_std': know_read_s,
                'know_write_sig_norm_mean': know_write_m,
                'know_write_sig_norm_std': know_write_s,
            })

        if analysis and not self.is_initializing():
            _residual_norm = jnp.linalg.norm(x, axis=-1).mean()
            _emb_norm = jnp.linalg.norm(self.token_emb.embedding, axis=-1).mean()
            _o_proj_norm = jnp.linalg.norm(
                stacked['attn']['expand_O']['kernel'], axis=(-2, -1)).mean()
            result.update({
                'know_phi_binary': know_phi_bin_all.mean(),
                'attn_qk_phi_binary': attn_qk_phi_bin_all.mean(),
                'attn_v_phi_binary': attn_v_phi_bin_all.mean(),
                'attn_tau_std': attn_tau_std_all.mean(axis=0),
                'know_tau_std': know_tau_std_all.mean(),
                'attn_tau_kernel_norm': attn_tau_kernel_norm_all.mean(),
                'know_tau_kernel_norm': know_tau_kernel_norm_all.mean(),
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
                'attn_gate_den_sum': attn_den_cost_all.mean(),
                'know_gate_den_sum': know_den_cost_all.mean(),
                'qk_emb_norm_max': attn_qk_emb_n_max_all.max(),
                'v_emb_norm_max': attn_v_emb_n_max_all.max(),
                'know_emb_norm_max': know_emb_n_max_all.max(),
                'attn_score_kurt': attn_score_kurt_all.mean(),
                'know_score_kurt': know_score_kurt_all.mean(),
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
                'attn_int_cap_frac': attn_int_cap_frac_all.mean(),
                'know_int_cap_frac': know_int_cap_frac_all.mean(),
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
        return {
            'model_version': self.__version__,
            'vocab_size': self.vocab_size, 'd_model': self.d_model,
            'n_layers': self.n_layers, 'n_heads': self.n_heads,
            'max_seq_len': self.max_seq_len,
            'd_route': self.d_route,
            'n_qk': self.n_qk, 'n_v': self.n_v, 'n_know': self.n_know,
            'tag_dim': self.tag_dim,
            'read_sig_dim': self.read_sig_dim,
            'write_sig_dim': self.write_sig_dim,
            'read_norm_sig_dim': self.read_norm_sig_dim,
            'write_norm_sig_dim': self.write_norm_sig_dim,
        }

    def get_model_info(self):
        return [
            f"DAWN v{self.__version__}: Sense-Read-Write",
            f"  d_model={self.d_model}, d_route={self.d_route}, "
            f"tag_dim={self.tag_dim}, read_sig_dim={self.read_sig_dim}, "
            f"write_sig_dim={self.write_sig_dim}, "
            f"read_norm_sig_dim={self.read_norm_sig_dim}, "
            f"write_norm_sig_dim={self.write_norm_sig_dim}, "
            f"n_layers={self.n_layers}, n_heads={self.n_heads}",
            f"  QK: {self.n_qk}, V: {self.n_v}, Know: {self.n_know}",
            f"  Route: tag[{self.tag_dim}] + read_sig[{self.read_sig_dim}] "
            f"+ write_sig[{self.write_sig_dim}] "
            f"+ read_norm[{self.read_norm_sig_dim}] "
            f"+ write_norm[{self.write_norm_sig_dim}]",
        ]


# ================================================================
# 9. INFERENCE API ??KV-cache prefill + decode
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


def _srw_inference(x, h, tag, tau_offset, scan_bias, w_read, w_write,
                   read_sig_proj, write_sig_proj):
    """Non-chunked SRW for inference with v4.1.5 route signatures."""
    r_n = w_read.astype(jnp.float32)
    w_n = w_write.astype(jnp.float32)
    route = build_route_signature(tag, r_n, w_n, read_sig_proj, write_sig_proj)
    scores = h @ route.T
    scores_f32 = scores.astype(jnp.float32)
    s_mean = scores_f32.mean(axis=-1, keepdims=True)
    s_std = jnp.sqrt(jnp.mean(jnp.square(scores_f32 - s_mean),
                               axis=-1, keepdims=True)) + 1e-8
    scan = SCAN_SCALE * jnp.tanh(scan_bias)
    tau = s_mean + tau_offset * s_std - scan / jnp.maximum(s_std, SCAN_STD_FLOOR)

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


def _srw_inference_with_gates(x, h, tag, tau_offset, scan_bias, w_read, w_write,
                              read_sig_proj, write_sig_proj):
    """Like _srw_inference but also returns raw and normalized gate for analysis."""
    r_n = w_read.astype(jnp.float32)
    w_n = w_write.astype(jnp.float32)
    route = build_route_signature(tag, r_n, w_n, read_sig_proj, write_sig_proj)
    scores = h @ route.T
    scores_f32 = scores.astype(jnp.float32)
    s_mean = scores_f32.mean(axis=-1, keepdims=True)
    s_std = jnp.sqrt(jnp.mean(jnp.square(scores_f32 - s_mean),
                               axis=-1, keepdims=True)) + 1e-8
    scan = SCAN_SCALE * jnp.tanh(scan_bias)
    tau = s_mean + tau_offset * s_std - scan / jnp.maximum(s_std, SCAN_STD_FLOOR)

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

    # emb used as-is (matches training path ??v4.0.6-style, no
    # forward unit-norm).
    qk_norm = pool_params['qk_emb']
    v_norm = pool_params['v_emb']
    qk_rp = pool_params['qk_read_sig_proj']
    qk_wp = pool_params['qk_write_sig_proj']
    v_rp = pool_params['v_read_sig_proj']
    v_wp = pool_params['v_write_sig_proj']
    h_all = x @ router_params['proj_attn']['kernel'] + router_params['proj_attn']['bias']
    h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)
    tau_all = x @ router_params['tau_attn']['kernel'] + router_params['tau_attn']['bias']
    scan_bias_all = x @ router_params['scan_bias_attn']['kernel'] + router_params['scan_bias_attn']['bias']

    Q = _srw_inference(x, h_Q, qk_norm, tau_all[:, :, 0:1], scan_bias_all[:, :, 0:1],
                       pool_params['qk_read'], pool_params['qk_write'], qk_rp, qk_wp)
    K_new = _srw_inference(x, h_K, qk_norm, tau_all[:, :, 1:2], scan_bias_all[:, :, 1:2],
                           pool_params['qk_read'], pool_params['qk_write'], qk_rp, qk_wp)
    V_new = _srw_inference(x, h_V, v_norm, tau_all[:, :, 2:3], scan_bias_all[:, :, 2:3],
                           pool_params['v_read'], pool_params['v_write'], v_rp, v_wp)
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
    # emb used as-is (matches training path).
    know_norm = pool_params['know_emb']
    know_rp = pool_params['know_read_sig_proj']
    know_wp = pool_params['know_write_sig_proj']
    h = x @ router_params['proj_know']['kernel'] + router_params['proj_know']['bias']
    tau = x @ router_params['tau_know']['kernel'] + router_params['tau_know']['bias']
    scan_bias = x @ router_params['scan_bias_know']['kernel'] + router_params['scan_bias_know']['bias']
    out = _srw_inference(x, h, know_norm, tau, scan_bias,
                         pool_params['know_read'], pool_params['know_write'],
                         know_rp, know_wp)
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

    # v4.0.3: no forward emb normalisation.
    qk_norm = pool_params['qk_emb']
    v_norm = pool_params['v_emb']
    qk_rp = pool_params['qk_read_sig_proj']
    qk_wp = pool_params['qk_write_sig_proj']
    v_rp = pool_params['v_read_sig_proj']
    v_wp = pool_params['v_write_sig_proj']

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
        scan_bias_all = normed @ router_params['scan_bias_attn']['kernel'] + router_params['scan_bias_attn']['bias']

        Q = _srw_inference(normed, h_Q, qk_norm, tau_all[:, :, 0:1], scan_bias_all[:, :, 0:1],
                           pool_params['qk_read'], pool_params['qk_write'], qk_rp, qk_wp)
        K_val = _srw_inference(normed, h_K, qk_norm, tau_all[:, :, 1:2], scan_bias_all[:, :, 1:2],
                               pool_params['qk_read'], pool_params['qk_write'], qk_rp, qk_wp)
        V_val = _srw_inference(normed, h_V, v_norm, tau_all[:, :, 2:3], scan_bias_all[:, :, 2:3],
                               pool_params['v_read'], pool_params['v_write'], v_rp, v_wp)
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
    Returns: (avg_loss, ppl, accuracy, total_valid) ??all jnp scalars.
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

    # v4.0.3: no forward emb normalisation.
    qk_norm = pool_params['qk_emb']
    v_norm = pool_params['v_emb']
    know_norm = pool_params['know_emb']
    qk_rp = pool_params['qk_read_sig_proj']
    qk_wp = pool_params['qk_write_sig_proj']
    v_rp = pool_params['v_read_sig_proj']
    v_wp = pool_params['v_write_sig_proj']
    know_rp = pool_params['know_read_sig_proj']
    know_wp = pool_params['know_write_sig_proj']

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
            scan_bias_all = normed @ router_params['scan_bias_attn']['kernel'] + router_params['scan_bias_attn']['bias']

            Q = _srw_inference(normed, h_Q, qk_norm, tau_all[:, :, 0:1], scan_bias_all[:, :, 0:1],
                               pool_params['qk_read'], pool_params['qk_write'], qk_rp, qk_wp)
            K = _srw_inference(normed, h_K, qk_norm, tau_all[:, :, 1:2], scan_bias_all[:, :, 1:2],
                               pool_params['qk_read'], pool_params['qk_write'], qk_rp, qk_wp)
            V = _srw_inference(normed, h_V, v_norm, tau_all[:, :, 2:3], scan_bias_all[:, :, 2:3],
                               pool_params['v_read'], pool_params['v_write'], v_rp, v_wp)
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
            scan_bias_k = normed @ router_params['scan_bias_know']['kernel'] + router_params['scan_bias_know']['bias']
            know_out = _srw_inference(normed, h_k, know_norm, tau_k, scan_bias_k,
                                     pool_params['know_read'], pool_params['know_write'],
                                     know_rp, know_wp)
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
    mode='light': returns gate_norm only (R.2, D6 ??half the memory)

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

    # v4.0.3: no forward emb normalisation.
    qk_norm = pool_params['qk_emb']
    v_norm = pool_params['v_emb']
    know_norm_w = pool_params['know_emb']
    qk_rp = pool_params['qk_read_sig_proj']
    qk_wp = pool_params['qk_write_sig_proj']
    v_rp = pool_params['v_read_sig_proj']
    v_wp = pool_params['v_write_sig_proj']
    know_rp = pool_params['know_read_sig_proj']
    know_wp = pool_params['know_write_sig_proj']

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
        scan_bias_all = normed @ router_params['scan_bias_attn']['kernel'] + router_params['scan_bias_attn']['bias']

        Q, gate_Q_raw, gate_Q = _srw_inference_with_gates(
            normed, h_Q, qk_norm, tau_all[:, :, 0:1], scan_bias_all[:, :, 0:1],
            pool_params['qk_read'], pool_params['qk_write'], qk_rp, qk_wp)
        K, gate_K_raw, gate_K = _srw_inference_with_gates(
            normed, h_K, qk_norm, tau_all[:, :, 1:2], scan_bias_all[:, :, 1:2],
            pool_params['qk_read'], pool_params['qk_write'], qk_rp, qk_wp)
        V, gate_V_raw, gate_V = _srw_inference_with_gates(
            normed, h_V, v_norm, tau_all[:, :, 2:3], scan_bias_all[:, :, 2:3],
            pool_params['v_read'], pool_params['v_write'], v_rp, v_wp)
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
        scan_bias_k = normed @ router_params['scan_bias_know']['kernel'] + router_params['scan_bias_know']['bias']
        know_out, gate_Know_raw, gate_Know = _srw_inference_with_gates(
            normed, h_k, know_norm_w, tau_k, scan_bias_k,
            pool_params['know_read'], pool_params['know_write'], know_rp, know_wp)
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

    def _srw_sup(x, h, tag, tau_off, scan_bias, w_read, w_write,
                 read_sig_proj, write_sig_proj, mult):
        """SRW with optional gate suppression. v4.0.3: denominator = 誇z^+ (floor 1.0)."""
        r_n = w_read.astype(jnp.float32)
        w_n = w_write.astype(jnp.float32)
        route = build_route_signature(tag, r_n, w_n, read_sig_proj, write_sig_proj)
        scores = h @ route.T
        sf = scores.astype(jnp.float32)
        s_mean = sf.mean(axis=-1, keepdims=True)
        s_std = jnp.sqrt(jnp.mean(jnp.square(sf - s_mean), axis=-1, keepdims=True)) + 1e-8
        scan = SCAN_SCALE * jnp.tanh(scan_bias)
        tau = s_mean + tau_off * s_std - scan / jnp.maximum(s_std, SCAN_STD_FLOOR)
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
        # v4.0.3: no forward emb normalisation.
        qk_n = pp['qk_emb']
        v_n = pp['v_emb']
        kn_n = pp['know_emb']
        qk_rp = pp['qk_read_sig_proj']
        qk_wp = pp['qk_write_sig_proj']
        v_rp = pp['v_read_sig_proj']
        v_wp = pp['v_write_sig_proj']
        kn_rp = pp['know_read_sig_proj']
        kn_wp = pp['know_write_sig_proj']

        for i in range(n_layers):
            bp = params[f'block_{i}']
            normed = _layer_norm(x, bp['norm1']['scale'], bp['norm1']['bias'])
            h_all = normed @ rp['proj_attn']['kernel'] + rp['proj_attn']['bias']
            h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)
            tau_all = normed @ rp['tau_attn']['kernel'] + rp['tau_attn']['bias']
            scan_bias_all = normed @ rp['scan_bias_attn']['kernel'] + rp['scan_bias_attn']['bias']

            Q = _srw_sup(normed, h_Q, qk_n, tau_all[:,:,0:1], scan_bias_all[:,:,0:1], pp['qk_read'], pp['qk_write'], qk_rp, qk_wp, qk_mult)
            K = _srw_sup(normed, h_K, qk_n, tau_all[:,:,1:2], scan_bias_all[:,:,1:2], pp['qk_read'], pp['qk_write'], qk_rp, qk_wp, qk_mult)
            V = _srw_sup(normed, h_V, v_n, tau_all[:,:,2:3], scan_bias_all[:,:,2:3], pp['v_read'], pp['v_write'], v_rp, v_wp, v_mult)
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
            scan_bias_k = normed @ rp['scan_bias_know']['kernel'] + rp['scan_bias_know']['bias']
            x = x + _srw_sup(normed, h_k, kn_n, tau_k, scan_bias_k, pp['know_read'], pp['know_write'], kn_rp, kn_wp, know_mult) * pp['know_scale']

        norm_p = params['norm']
        x = _layer_norm(x, norm_p['scale'], norm_p['bias'])
        return x @ params['token_emb']['embedding'].T

    return forward_fn

