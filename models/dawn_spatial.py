"""
DAWN-Spatial: Rank-1 Neuron Architecture (JAX/Flax)

Key concept:
  - Every neuron is a single vector v_i [D] (rank-1).
  - Sense: activation_i = v_i · x  (scalar)
  - Fire:  gate decides which neurons fire (threshold tau)
  - Emit:  out = Σ gate_i * activation_i * v_i
  - Complex transforms emerge from multi-layer residual, not single neuron.

Reference:
  - model_v17_1_jax.py: scan, checkpoint, cached forward, weight tying
  - model_v18_5.py: threshold/tau gate mechanism
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Dict


# ================================================================
# Trace-safe dropout (from v17.1 — compatible with nn.remat)
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
# Pure functional LayerNorm (from v17.1)
# ================================================================

def _layer_norm(x, scale, bias, eps=1e-6):
    """Pure functional LayerNorm (matches Flax nn.LayerNorm)."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * scale + bias


# ================================================================
# Initializers
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
# Threshold gate (v18.5 tau mechanism → JAX)
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
