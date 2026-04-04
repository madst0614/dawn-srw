"""
DAWN-Spatial v3.8: Sense-Read-Write (JAX/Flax)

Changelog:
  spatial-r1-v3.8.2 (2026-04-01):
    - d_bottleneck 64->128 (routing resolution improvement)
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
    """v18.5 style relative tau. All [B,S,N] tensors in bf16.
    Only mean/std/tau (scalar/[B,S,1]) computed in f32.
    """
    # mean/std in f32 (reduce results are [B,S,1], tiny)
    scores_f32 = scores.astype(jnp.float32)
    s_mean = scores_f32.mean(axis=-1, keepdims=True)
    s_std = jnp.sqrt(jnp.mean(jnp.square(scores_f32 - s_mean), axis=-1, keepdims=True)) + 1e-8
    tau = s_mean + tau_offset * s_std

    # gate in bf16 ([B,S,N] tensors)
    raw = scores - tau.astype(scores.dtype)
    gate = jnp.where(raw > 0, raw,
                     (1e-6 * jnp.exp(jnp.clip(raw.astype(jnp.float32), -10.0, 0.0))).astype(scores.dtype))

    gate = jnp.clip(gate, 0.0, 10.0)
    exp_gate = (jnp.exp(gate.astype(jnp.float32)) - 1.0).astype(scores.dtype)

    # sum/max reduce → [B,S,1], then normalize in same dtype
    exp_sum = exp_gate.sum(axis=-1, keepdims=True) + 1e-8
    ratio = exp_gate / exp_sum
    gate_strength = jnp.tanh(
        exp_gate.max(axis=-1, keepdims=True).astype(jnp.float32)
    ).astype(scores.dtype)

    return ratio * gate_strength


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
                        P(),                     # gs_mean scalar
                        P()),                    # es_mean scalar
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

        # --- Pass 1: stats from first chunk only (checkpointed) ---
        @jax.checkpoint
        def estimate_stats(h, emb, nc_val):
            ec0 = jax.lax.dynamic_slice_in_dim(emb, 0, cs, axis=0)
            sc0 = h @ ec0.T
            s_sum = sc0.sum(axis=-1, keepdims=True).astype(jnp.float32) * nc_val
            sq_sum = (sc0 ** 2).sum(axis=-1, keepdims=True).astype(jnp.float32) * nc_val
            return s_sum, sq_sum

        local_sum, local_sq = estimate_stats(h_bf, emb_bf, nc)

        global_sum = jax.lax.psum(local_sum, 'model')
        global_sq = jax.lax.psum(local_sq, 'model')
        N_total = N_local * _model_axis_size

        s_mean = global_sum / N_total
        s_std = jnp.sqrt(global_sq / N_total - s_mean ** 2) + 1e-8
        tau = s_mean + tau_offset * s_std
        tau_bf = tau.astype(jnp.bfloat16)

        # --- Pass 2: gate + srw fused (scan + checkpoint) ---
        # lb_sq_sum: accumulate per-chunk LB contribution (scalar)
        @jax.checkpoint
        def gate_srw_step(carry, i):
            out, exp_sum, exp_max, active, lb_sum, lb_sq = carry
            s = i * cs
            ec = jax.lax.dynamic_slice_in_dim(emb_bf, s, cs, axis=0)
            rc = jax.lax.dynamic_slice_in_dim(read_bf, s, cs, axis=0)
            wc = jax.lax.dynamic_slice_in_dim(write_bf, s, cs, axis=0)
            sc = h_bf @ ec.T
            raw = sc - tau_bf
            gc = jnp.where(raw > 0, raw,
                            (1e-6 * jnp.exp(jnp.clip(raw.astype(jnp.float32), -10.0, 0.0))).astype(jnp.bfloat16))
            gc = jnp.clip(gc, 0.0, 10.0)
            eg = (jnp.exp(gc.astype(jnp.float32)) - 1.0).astype(jnp.bfloat16)
            ef = eg.astype(jnp.float32)
            xr = x_bf @ rc.T
            c_out = ((eg * xr) @ wc).astype(jnp.float32)
            # LB: per-neuron mean gate for this chunk's neurons [cs]
            gn = ef.mean(axis=(0, 1))  # [cs] avg gate per neuron
            lb_sum = lb_sum + gn.sum()
            lb_sq = lb_sq + (gn ** 2).sum()
            return (out + c_out, exp_sum + ef.sum(axis=-1, keepdims=True),
                    jnp.maximum(exp_max, ef.max(axis=-1, keepdims=True)),
                    active + (raw > 0).sum(axis=-1, keepdims=True).astype(jnp.float32),
                    lb_sum, lb_sq), None

        z_scalar = jnp.float32(0.0)
        (raw_out, total_es, total_em, total_ac, lb_sum, lb_sq), _ = jax.lax.scan(
            gate_srw_step,
            (jnp.zeros((B, S, D), dtype=jnp.float32),
             z1, jnp.full((B, S, 1), -1e9), z1,
             z_scalar, z_scalar),
            jnp.arange(nc))

        global_exp_sum = jax.lax.psum(total_es, 'model') + 1e-8
        gate_strength = jnp.tanh(total_em)
        out = raw_out / global_exp_sum * gate_strength
        # bf16 before psum: 640MB → 320MB per all-reduce
        out = jax.lax.psum(out.astype(jnp.bfloat16), 'model')

        active = jax.lax.psum(total_ac, 'model')

        # Load balance loss on normalized gate: exp_gate / exp_sum
        lb_sum = jax.lax.psum(lb_sum, 'data') / _data_axis_size
        lb_sq = jax.lax.psum(lb_sq, 'data') / _data_axis_size
        global_lb_sum = jax.lax.psum(lb_sum, 'model')
        global_lb_sq = jax.lax.psum(lb_sq, 'model')
        exp_sum_scalar = global_exp_sum.mean() + 1e-8
        norm_lb_sum = global_lb_sum / exp_sum_scalar
        norm_lb_sq = global_lb_sq / (exp_sum_scalar ** 2)
        lb_loss = norm_lb_sq * N_total - 2.0 * norm_lb_sum + 1.0

        gs_mean = jnp.tanh(total_em).mean()
        es_mean = global_exp_sum.mean()
        return out.astype(jnp.float32), active / N_total, total_em, lb_loss, gs_mean, es_mean

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
                        P(),                         # gs_mean scalar
                        P()),                        # es_mean scalar
             check_rep=False)
    def fused_gate_srw_paired(x, h, emb_local, tau_offset, read_local, write_local):
        N_local = emb_local.shape[0]
        nc = max(1, N_local // max_chunk_size)
        while N_local % nc != 0 and nc < N_local:
            nc += 1
        cs = N_local // nc

        B, S, D = x.shape
        # h: [B,S,2,d_bn], tau_offset: [B,S,2,1]
        h_bf = h.astype(jnp.bfloat16)
        x_bf = x.astype(jnp.bfloat16)
        emb_bf = emb_local.astype(jnp.bfloat16)
        read_bf = read_local.astype(jnp.bfloat16)
        write_bf = write_local.astype(jnp.bfloat16)
        z1_r = jnp.zeros((B, S, 2, 1))  # per-route accumulators

        # --- Pass 1: stats from first chunk only (checkpointed) ---
        @jax.checkpoint
        def estimate_stats(h, emb, nc_val):
            ec0 = jax.lax.dynamic_slice_in_dim(emb, 0, cs, axis=0)
            sc0 = jnp.einsum('bsrd,nd->bsrn', h, ec0)
            s_sum = sc0.sum(axis=-1, keepdims=True).astype(jnp.float32) * nc_val
            sq_sum = (sc0 ** 2).sum(axis=-1, keepdims=True).astype(jnp.float32) * nc_val
            return s_sum, sq_sum

        local_sum, local_sq = estimate_stats(h_bf, emb_bf, nc)

        global_sum = jax.lax.psum(local_sum, 'model')  # [B,S,2,1]
        global_sq = jax.lax.psum(local_sq, 'model')
        N_total = N_local * _model_axis_size

        s_mean = global_sum / N_total      # [B,S,2,1]
        s_std = jnp.sqrt(global_sq / N_total - s_mean ** 2) + 1e-8
        tau = s_mean + tau_offset * s_std   # [B,S,2,1]
        tau_bf = tau.astype(jnp.bfloat16)

        # --- Pass 2: gate + srw fused ---
        @jax.checkpoint
        def gate_srw_step(carry, i):
            out, exp_sum, exp_max, active, lb_sum, lb_sq = carry
            s = i * cs
            ec = jax.lax.dynamic_slice_in_dim(emb_bf, s, cs, axis=0)
            rc = jax.lax.dynamic_slice_in_dim(read_bf, s, cs, axis=0)
            wc = jax.lax.dynamic_slice_in_dim(write_bf, s, cs, axis=0)
            sc = jnp.einsum('bsrd,nd->bsrn', h_bf, ec)
            raw = sc - tau_bf
            gc = jnp.where(raw > 0, raw,
                            (1e-6 * jnp.exp(jnp.clip(raw.astype(jnp.float32), -10.0, 0.0))).astype(jnp.bfloat16))
            gc = jnp.clip(gc, 0.0, 10.0)
            eg = (jnp.exp(gc.astype(jnp.float32)) - 1.0).astype(jnp.bfloat16)
            ef = eg.astype(jnp.float32)
            xr = x_bf @ rc.T
            c_out = jnp.einsum('bsrn,nd->bsrd', eg * xr[:, :, None, :], wc).astype(jnp.float32)
            # LB: average over route dim → per-neuron mean [cs]
            gn = ef.mean(axis=(0, 1, 2))  # [cs] avg gate per neuron (across B,S,2)
            lb_sum = lb_sum + gn.sum()
            lb_sq = lb_sq + (gn ** 2).sum()
            return (out + c_out,
                    exp_sum + ef.sum(axis=-1, keepdims=True),
                    jnp.maximum(exp_max, ef.max(axis=-1, keepdims=True)),
                    active + (raw > 0).sum(axis=-1, keepdims=True).astype(jnp.float32),
                    lb_sum, lb_sq), None

        z_scalar = jnp.float32(0.0)
        (raw_out, total_es, total_em, total_ac, lb_sum, lb_sq), _ = jax.lax.scan(
            gate_srw_step,
            (jnp.zeros((B, S, 2, D), dtype=jnp.float32),
             z1_r, jnp.full((B, S, 2, 1), -1e9), z1_r,
             z_scalar, z_scalar),
            jnp.arange(nc))

        # Normalize per route independently
        global_exp_sum = jax.lax.psum(total_es, 'model') + 1e-8
        gate_strength = jnp.tanh(total_em)
        out = raw_out / global_exp_sum * gate_strength
        out = jax.lax.psum(out.astype(jnp.bfloat16), 'model')

        active = jax.lax.psum(total_ac, 'model')
        active_mean = active.mean(axis=2)
        gate_max_mean = total_em.mean(axis=2)

        # Load balance loss on normalized gate
        lb_sum = jax.lax.psum(lb_sum, 'data') / _data_axis_size
        lb_sq = jax.lax.psum(lb_sq, 'data') / _data_axis_size
        global_lb_sum = jax.lax.psum(lb_sum, 'model')
        global_lb_sq = jax.lax.psum(lb_sq, 'model')
        exp_sum_scalar = global_exp_sum.mean() + 1e-8
        norm_lb_sum = global_lb_sum / exp_sum_scalar
        norm_lb_sq = global_lb_sq / (exp_sum_scalar ** 2)
        lb_loss = norm_lb_sq * N_total - 2.0 * norm_lb_sum + 1.0

        gs_mean = jnp.tanh(total_em).mean()
        es_mean = global_exp_sum.mean()
        return out.astype(jnp.float32), active_mean / N_total, gate_max_mean, lb_loss, gs_mean, es_mean

    return fused_gate_srw_paired


def _srw_chunked(x, h, emb_norm, tau_offset, w_read, w_write, n_chunks):
    """Fallback: non-sharded chunked srw (for mesh_model=1 or 40M scale).

    Uses fori_loop + checkpoint. No shard_map.
    """
    B, S, D = x.shape
    N = emb_norm.shape[0]
    cs = N // n_chunks

    h_bf = h.astype(jnp.bfloat16)
    x_bf = x.astype(jnp.bfloat16)
    emb_bf = emb_norm.astype(jnp.bfloat16)
    read_bf = w_read.astype(jnp.bfloat16)
    write_bf = w_write.astype(jnp.bfloat16)

    z1 = jnp.zeros((B, S, 1))

    # --- Stats from first chunk only (checkpointed) ---
    @jax.checkpoint
    def estimate_stats(h, emb, nc_val):
        ec0 = jax.lax.dynamic_slice_in_dim(emb, 0, cs, axis=0)
        sc0 = h @ ec0.T
        s_sum = sc0.sum(axis=-1, keepdims=True).astype(jnp.float32) * nc_val
        sq_sum = (sc0 ** 2).sum(axis=-1, keepdims=True).astype(jnp.float32) * nc_val
        return s_sum, sq_sum

    s_sum, sq_sum = estimate_stats(h_bf, emb_bf, n_chunks)
    s_mean = s_sum / N
    s_std = jnp.sqrt(sq_sum / N - s_mean**2) + 1e-8
    tau = s_mean + tau_offset * s_std
    tau_bf = tau.astype(jnp.bfloat16)

    @jax.checkpoint
    def gsrw_step(carry, i):
        out, es, em, ac, lbs, lbq = carry
        s = i * cs
        ec = jax.lax.dynamic_slice_in_dim(emb_bf, s, cs, axis=0)
        rc = jax.lax.dynamic_slice_in_dim(read_bf, s, cs, axis=0)
        wc = jax.lax.dynamic_slice_in_dim(write_bf, s, cs, axis=0)
        sc = h_bf @ ec.T
        raw = sc - tau_bf
        gc = jnp.where(raw > 0, raw,
                        (1e-6 * jnp.exp(jnp.clip(raw.astype(jnp.float32), -10.0, 0.0))).astype(jnp.bfloat16))
        gc = jnp.clip(gc, 0.0, 10.0)
        eg = (jnp.exp(gc.astype(jnp.float32))-1.0).astype(jnp.bfloat16)
        ef = eg.astype(jnp.float32)
        xr = x_bf @ rc.T
        co = ((eg*xr)@wc).astype(jnp.float32)
        gn = ef.mean(axis=(0, 1))  # [cs]
        lbs = lbs + gn.sum()
        lbq = lbq + (gn ** 2).sum()
        return (out+co, es+ef.sum(axis=-1,keepdims=True),
                jnp.maximum(em, ef.max(axis=-1,keepdims=True)),
                ac+(raw>0).sum(axis=-1,keepdims=True).astype(jnp.float32),
                lbs, lbq), None

    z_scalar = jnp.float32(0.0)
    (raw_out, tes, tem, tac, lb_sum, lb_sq), _ = jax.lax.scan(
        gsrw_step,
        (jnp.zeros((B,S,D), dtype=jnp.float32), z1, jnp.full((B,S,1),-1e9), z1,
         z_scalar, z_scalar),
        jnp.arange(n_chunks))

    inv_es = (1.0/(tes+1e-8)).astype(jnp.bfloat16)
    gs = jnp.tanh(tem).astype(jnp.bfloat16)
    out = raw_out * inv_es * gs

    # Load balance on normalized gate
    exp_sum_scalar = tes.mean() + 1e-8
    norm_lb_sum = lb_sum / exp_sum_scalar
    norm_lb_sq = lb_sq / (exp_sum_scalar ** 2)
    lb_loss = norm_lb_sq * N - 2.0 * norm_lb_sum + 1.0

    gs_mean = jnp.tanh(tem).mean()
    es_mean = tes.mean()
    return out.astype(jnp.float32), tac / N, tem, lb_loss, gs_mean, es_mean


# ================================================================
# 4. NeuronPool -- emb + w_read + w_write
# ================================================================

class NeuronPool(nn.Module):
    n_qk: int
    n_v: int
    n_know: int
    d_model: int
    d_bottleneck: int

    def setup(self):
        db = self.d_bottleneck
        dm = self.d_model

        # Sense (routing, low-dim)
        self.qk_emb = self.param('qk_emb', unit_norm_init(), (self.n_qk, db))
        self.v_emb = self.param('v_emb', unit_norm_init(), (self.n_v, db))
        self.know_emb = self.param('know_emb', unit_norm_init(), (self.n_know, db))

        # Read (what to extract from x)
        self.qk_read = self.param('qk_read', scaled_normal(0.02), (self.n_qk, dm))
        self.v_read = self.param('v_read', scaled_normal(0.02), (self.n_v, dm))
        self.know_read = self.param('know_read', scaled_normal(0.02), (self.n_know, dm))

        # Write (direction to push)
        self.qk_write = self.param('qk_write', scaled_normal(0.02), (self.n_qk, dm))
        self.v_write = self.param('v_write', scaled_normal(0.02), (self.n_v, dm))
        self.know_write = self.param('know_write', scaled_normal(0.02), (self.n_know, dm))


# ================================================================
# 5. Router -- proj + tau (unchanged)
# ================================================================

class Router(nn.Module):
    d_model: int
    d_bottleneck: int
    n_qk: int
    n_v: int
    n_know: int
    max_k_qk: int
    max_k_v: int
    max_k_know: int
    router_dropout: float = 0.1

    def setup(self):
        db = self.d_bottleneck
        self.proj_attn = nn.Dense(db * 3, name='proj_attn')
        self.proj_know = nn.Dense(db, name='proj_know')
        self.tau_attn = nn.Dense(3, name='tau_attn',
            kernel_init=nn.initializers.zeros,
            bias_init=lambda k, s, d: jnp.full(s, -0.5))
        self.tau_know = nn.Dense(1, name='tau_know',
            kernel_init=nn.initializers.zeros,
            bias_init=lambda k, s, d: jnp.full(s, -0.5))

    def get_attention_gates(self, x, neuron_pool, deterministic, rng):
        qk_norm = neuron_pool.qk_emb / (
            jnp.linalg.norm(neuron_pool.qk_emb, axis=-1, keepdims=True) + 1e-8)
        v_norm = neuron_pool.v_emb / (
            jnp.linalg.norm(neuron_pool.v_emb, axis=-1, keepdims=True) + 1e-8)

        rng, rng_drop = jax.random.split(rng)
        h_all = self.proj_attn(x)
        h_all = safe_dropout(h_all, self.router_dropout, deterministic, rng_drop)
        h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)

        tau_all = self.tau_attn(x)
        g_Q = threshold_gate(h_Q @ qk_norm.T, tau_all[:, :, 0:1])
        g_K = threshold_gate(h_K @ qk_norm.T, tau_all[:, :, 1:2])
        g_V = threshold_gate(h_V @ v_norm.T, tau_all[:, :, 2:3])

        t_qk = 1.0 / self.n_qk
        t_v = 1.0 / self.n_v
        aux = (
            ((g_Q.mean(axis=(0, 1)) - t_qk) ** 2).sum() * self.n_qk +
            ((g_K.mean(axis=(0, 1)) - t_qk) ** 2).sum() * self.n_qk +
            ((g_V.mean(axis=(0, 1)) - t_v) ** 2).sum() * self.n_v
        )
        return g_Q, g_K, g_V, aux

    def get_knowledge_gates(self, x, neuron_pool, deterministic, rng):
        know_norm = neuron_pool.know_emb / (
            jnp.linalg.norm(neuron_pool.know_emb, axis=-1, keepdims=True) + 1e-8)

        rng, rng_drop = jax.random.split(rng)
        h = self.proj_know(x)
        h = safe_dropout(h, self.router_dropout, deterministic, rng_drop)

        tau = self.tau_know(x)
        gate = threshold_gate(h @ know_norm.T, tau)

        t = 1.0 / self.n_know
        aux = ((gate.mean(axis=(0, 1)) - t) ** 2).sum() * self.n_know
        return gate, aux


# ================================================================
# 6. Pure functions for scan body
# ================================================================

def _attn_forward(x, pool_params, router_params, expand_O_kernel, rng,
                  n_qk, n_v,
                  max_k_qk, max_k_v, n_heads, d_model,
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

    qk_norm = qk_emb / (jnp.linalg.norm(qk_emb, axis=-1, keepdims=True) + 1e-8)
    v_norm = v_emb / (jnp.linalg.norm(v_emb, axis=-1, keepdims=True) + 1e-8)

    rng, rng_drop = jax.random.split(rng)
    h_all = x @ router_params['proj_attn']['kernel'] + router_params['proj_attn']['bias']
    h_all = safe_dropout(h_all, router_dropout, deterministic, rng_drop)
    h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)

    tau_all = x @ router_params['tau_attn']['kernel'] + router_params['tau_attn']['bias']

    if sharded_fns is not None:
        fused_single, fused_paired = sharded_fns
        h_QK = jnp.stack([h_Q, h_K], axis=2)  # [B,S,2,d_bn]
        tau_QK = jnp.stack([tau_all[:, :, 0:1], tau_all[:, :, 1:2]], axis=2)  # [B,S,2,1]
        QK_out, _, _, qk_lb, qk_gs, qk_es = fused_paired(x, h_QK, qk_norm, tau_QK, qk_read, qk_write)
        Q = QK_out[:, :, 0, :]  # [B,S,D]
        K = QK_out[:, :, 1, :]  # [B,S,D]
        V, _, _, v_lb, v_gs, v_es = fused_single(x, h_V, v_norm, tau_all[:, :, 2:3], v_read, v_write)
    else:
        Q, _, _, q_lb, q_gs, q_es = _srw_chunked(x, h_Q, qk_norm, tau_all[:, :, 0:1],
                                qk_read, qk_write, n_chunks_qk)
        K, _, _, k_lb, k_gs2, k_es2 = _srw_chunked(x, h_K, qk_norm, tau_all[:, :, 1:2],
                                qk_read, qk_write, n_chunks_qk)
        V, _, _, v_lb, v_gs, v_es = _srw_chunked(x, h_V, v_norm, tau_all[:, :, 2:3],
                                v_read, v_write, n_chunks_v)
        qk_lb = q_lb + k_lb
        qk_gs = (q_gs + k_gs2) / 2
        qk_es = (q_es + k_es2) / 2

    d_head = d_model // n_heads
    Q = Q.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
    K = K.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
    V = V.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)

    scale = jnp.sqrt(jnp.float32(d_head))

    @jax.checkpoint
    def _attn_scores(Q, K, V):
        attn_scores = jnp.einsum('bhsd,bhtd->bhst', Q, K) / scale
        causal = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
        attn_scores = jnp.where(causal, attn_scores,
                                jnp.finfo(attn_scores.dtype).min)
        attn_w = jax.nn.softmax(attn_scores, axis=-1)
        return jnp.einsum('bhst,bhtd->bhsd', attn_w, V)

    out = _attn_scores(Q, K, V)
    out = out.transpose(0, 2, 1, 3).reshape(B, S, D)
    out = out @ expand_O_kernel
    rng, rng_out = jax.random.split(rng)
    out = safe_dropout(out, dropout_rate, deterministic, rng_out)

    # Load balance loss from gate distributions + tau regularization
    tau_reg = jnp.maximum(tau_all, 0.0).mean() * 0.01
    aux = qk_lb + v_lb + tau_reg
    attn_gs = (qk_gs + v_gs) / 2
    attn_es = (qk_es + v_es) / 2
    return out, aux, attn_gs, attn_es


def _know_forward(x, pool_params, router_params, rng,
                  max_k_know,
                  router_dropout, dropout_rate, deterministic,
                  n_chunks_know=1, sharded_fns=None):
    know_emb = pool_params['know_emb']
    know_read = pool_params['know_read']
    know_write = pool_params['know_write']

    rng, rng_drop = jax.random.split(rng)
    h = x @ router_params['proj_know']['kernel'] + router_params['proj_know']['bias']
    h = safe_dropout(h, router_dropout, deterministic, rng_drop)

    know_norm = know_emb / (jnp.linalg.norm(know_emb, axis=-1, keepdims=True) + 1e-8)
    tau = x @ router_params['tau_know']['kernel'] + router_params['tau_know']['bias']

    if sharded_fns is not None:
        fused_single, fused_paired = sharded_fns
        out, active_count, gate_max_val, lb_loss, gs_mean, es_mean = fused_single(
            x, h, know_norm, tau, know_read, know_write)
    else:
        out, active_count, gate_max_val, lb_loss, gs_mean, es_mean = _srw_chunked(
            x, h, know_norm, tau, know_read, know_write, n_chunks_know)

    rng, rng_out = jax.random.split(rng)
    out = safe_dropout(out, dropout_rate, deterministic, rng_out)

    # Load balance loss from gate distribution + tau regularization
    tau_reg = jnp.maximum(tau, 0.0).mean() * 0.01
    aux = lb_loss + tau_reg
    emb_norm_val = jnp.linalg.norm(know_emb, axis=-1).mean()
    read_norm_val = jnp.linalg.norm(know_read, axis=-1).mean()
    write_norm_val = jnp.linalg.norm(know_write, axis=-1).mean()
    return out, aux, active_count, gate_max_val, gs_mean, es_mean, emb_norm_val, read_norm_val, write_norm_val


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
    __version__ = "spatial-r1-v3.8.2"

    vocab_size: int = 30000
    d_model: int = 384
    n_layers: int = 12
    n_heads: int = 6
    max_seq_len: int = 512
    dropout_rate: float = 0.1
    gradient_checkpointing: bool = False

    d_bottleneck: int = 128
    n_qk: int = 1580
    n_v: int = 2600
    n_know: int = 25200
    max_k_qk: int = 158
    max_k_v: int = 260
    max_k_know: int = 1810
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
            d_model=self.d_model, d_bottleneck=self.d_bottleneck)
        self.router = Router(
            d_model=self.d_model, d_bottleneck=self.d_bottleneck,
            n_qk=self.n_qk, n_v=self.n_v, n_know=self.n_know,
            max_k_qk=self.max_k_qk, max_k_v=self.max_k_v,
            max_k_know=self.max_k_know, router_dropout=self.router_dropout)
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
            total_aux = jnp.float32(0.0)
            attn_auxes = jnp.float32(0.0)
            know_auxes = jnp.float32(0.0)
            know_actives = jnp.float32(0.0)
            know_gmaxes = jnp.float32(0.0)
            know_gs_all = jnp.float32(0.0)
            know_es_all = jnp.float32(0.0)
            for layer in self.layers:
                x, aux = layer(x, self.neuron_pool, self.router,
                               attention_mask, deterministic)
                total_aux = total_aux + aux
        else:
            all_params = self.variables['params']
            pool_params = all_params['neuron_pool']
            router_params = all_params['router']

            _sharded = sharded_fns  # None for non-sharded, tuple for shard_map

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
                attn_out, attn_aux, attn_gs, attn_es = _attn_forward(
                    normed, pool_params, router_params,
                    bp['attn']['expand_O']['kernel'], rng_attn,
                    self.n_qk, self.n_v,
                    self.max_k_qk, self.max_k_v,
                    self.n_heads, self.d_model,
                    self.router_dropout, self.dropout_rate, deterministic,
                    self.n_chunks_qk, self.n_chunks_v,
                    sharded_fns=_sharded)
                x = x + attn_out

                normed = _layer_norm(
                    x, bp['norm2']['scale'], bp['norm2']['bias'])
                know_out, know_aux, know_active, know_gmax, know_gs, know_es, k_emb_n, k_read_n, k_write_n = _know_forward(
                    normed, pool_params, router_params, rng_know,
                    self.max_k_know,
                    self.router_dropout, self.dropout_rate, deterministic,
                    self.n_chunks_know, sharded_fns=_sharded)
                x = x + know_out
                return x, (attn_aux, know_aux, know_active, know_gmax, know_gs, know_es, k_emb_n, k_read_n, k_write_n)

            if self.gradient_checkpointing:
                scan_body = jax.checkpoint(scan_body)

            xs = {'params': stacked, 'rng': layer_rngs}
            x, (attn_auxes, know_auxes, know_actives, know_gmaxes,
                know_gs_all, know_es_all, k_emb_n_all, k_read_n_all, k_write_n_all) = jax.lax.scan(
                scan_body, x, xs)
            total_aux = (attn_auxes + know_auxes).sum()

        x = self.norm(x)
        result = {
            'aux_loss': total_aux,
            'attn_aux': attn_auxes.mean(),
            'know_aux': know_auxes.mean(),
            'know_active': know_actives.mean(),
            'know_gate_max': know_gmaxes.mean(),
            'know_gs': know_gs_all.mean(),
            'know_es': know_es_all.mean(),
            'know_emb_norm': k_emb_n_all.mean(),
            'know_read_norm': k_read_n_all.mean(),
            'know_write_norm': k_write_n_all.mean(),
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

    def diversity_loss(self):
        def _div(neurons, max_sample=4096):
            N = neurons.shape[0]
            if N > max_sample:
                stride = N // max_sample
                neurons = neurons[::stride][:max_sample]
            n = neurons / (jnp.linalg.norm(neurons, axis=-1, keepdims=True) + 1e-8)
            sim = n @ n.T
            mask = ~jnp.eye(sim.shape[0], dtype=jnp.bool_)
            return jnp.abs(sim * mask).sum() / mask.sum()
        pool = self.neuron_pool
        return (_div(pool.qk_emb) + _div(pool.qk_read) + _div(pool.qk_write) +
                _div(pool.v_emb) + _div(pool.v_read) + _div(pool.v_write) +
                _div(pool.know_emb) + _div(pool.know_read) + _div(pool.know_write)) / 9

    def get_auxiliary_losses(self):
        return {'neuron_diversity': self.diversity_loss()}

    def get_config(self):
        return {
            'model_version': self.__version__,
            'vocab_size': self.vocab_size, 'd_model': self.d_model,
            'n_layers': self.n_layers, 'n_heads': self.n_heads,
            'max_seq_len': self.max_seq_len,
            'd_bottleneck': self.d_bottleneck,
            'n_qk': self.n_qk, 'n_v': self.n_v, 'n_know': self.n_know,
            'max_k_qk': self.max_k_qk, 'max_k_v': self.max_k_v,
            'max_k_know': self.max_k_know,
        }

    def get_model_info(self):
        return [
            f"DAWN v{self.__version__}: Sense-Read-Write",
            f"  d_model={self.d_model}, d_bottleneck={self.d_bottleneck}, "
            f"n_layers={self.n_layers}, n_heads={self.n_heads}",
            f"  QK: {self.n_qk}, V: {self.n_v}, Know: {self.n_know}",
            f"  Per neuron: emb[{self.d_bottleneck}] + read[{self.d_model}] "
            f"+ write[{self.d_model}]",
        ]


# ================================================================
# 9. INFERENCE API — KV-cache prefill + decode
#    Pure functions only. Training code above is untouched.
# ================================================================

def _srw_inference(x, h, emb_norm, tau_offset, w_read, w_write):
    """Non-chunked SRW for inference (S=1 typically).
    No checkpoint, no LB loss, no bf16 casting.
    """
    scores = h @ emb_norm.T
    scores_f32 = scores.astype(jnp.float32)
    s_mean = scores_f32.mean(axis=-1, keepdims=True)
    s_std = jnp.sqrt(jnp.mean(jnp.square(scores_f32 - s_mean),
                               axis=-1, keepdims=True)) + 1e-8
    tau = s_mean + tau_offset * s_std

    raw = scores - tau.astype(scores.dtype)
    gc = jnp.where(raw > 0, raw,
                   (1e-6 * jnp.exp(jnp.clip(
                       raw.astype(jnp.float32), -10.0, 0.0)
                   )).astype(scores.dtype))
    gc = jnp.clip(gc, 0.0, 10.0)
    eg = (jnp.exp(gc.astype(jnp.float32)) - 1.0).astype(scores.dtype)

    exp_sum = eg.sum(axis=-1, keepdims=True).astype(jnp.float32) + 1e-8
    gate_strength = jnp.tanh(
        eg.max(axis=-1, keepdims=True).astype(jnp.float32))

    xr = x @ w_read.T
    raw_out = (eg * xr) @ w_write
    out = raw_out.astype(jnp.float32) / exp_sum * gate_strength
    return out.astype(jnp.float32)


def _srw_inference_with_gates(x, h, emb_norm, tau_offset, w_read, w_write):
    """Like _srw_inference but also returns normalized gate [B,S,N] for analysis."""
    scores = h @ emb_norm.T
    scores_f32 = scores.astype(jnp.float32)
    s_mean = scores_f32.mean(axis=-1, keepdims=True)
    s_std = jnp.sqrt(jnp.mean(jnp.square(scores_f32 - s_mean),
                               axis=-1, keepdims=True)) + 1e-8
    tau = s_mean + tau_offset * s_std

    raw = scores - tau.astype(scores.dtype)
    gc = jnp.where(raw > 0, raw,
                   (1e-6 * jnp.exp(jnp.clip(
                       raw.astype(jnp.float32), -10.0, 0.0)
                   )).astype(scores.dtype))
    gc = jnp.clip(gc, 0.0, 10.0)
    eg = (jnp.exp(gc.astype(jnp.float32)) - 1.0).astype(scores.dtype)

    exp_sum = eg.sum(axis=-1, keepdims=True).astype(jnp.float32) + 1e-8
    gate_strength = jnp.tanh(
        eg.max(axis=-1, keepdims=True).astype(jnp.float32))
    gate_norm = (eg.astype(jnp.float32) / exp_sum * gate_strength)

    xr = x @ w_read.T
    raw_out = (eg * xr) @ w_write
    out = raw_out.astype(jnp.float32) / exp_sum * gate_strength
    return out.astype(jnp.float32), gate_norm


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
    return _srw_inference(x, h, know_norm, tau,
                          pool_params['know_read'], pool_params['know_write'])


def prefill(params, model_cfg, input_ids):
    """Run full forward on prompt, populate KV cache.

    Returns: logits [B,S,vocab], cache_K, cache_V [n_layers,B,H,max_seq,d_head], cache_len
    """
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
    token_id = token_id.reshape(-1, 1)
    B = token_id.shape[0]
    d_model = model_cfg['d_model']
    n_layers = model_cfg['n_layers']
    n_heads = model_cfg['n_heads']

    pool_params = params['neuron_pool']
    router_params = params['router']

    x = (params['token_emb']['embedding'][token_id]
         + params['pos_emb']['embedding'][[cache_len]])

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
    n_seqs = all_tokens.shape[0]
    n_batches = n_seqs // batch_size
    tokens = all_tokens[:n_batches * batch_size].reshape(n_batches, batch_size, -1)

    d_model = model_cfg['d_model']
    n_layers = model_cfg['n_layers']
    n_heads = model_cfg['n_heads']
    max_seq = model_cfg['max_seq_len']

    pool_params = params['neuron_pool']
    router_params = params['router']
    norm_params = params['norm']
    emb_matrix = params['token_emb']['embedding']
    pos_matrix = params['pos_emb']['embedding']

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
        x = emb_matrix[input_ids] + pos_matrix[positions]

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
            x = x + know_out
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


def analysis_forward(params, model_cfg, input_ids):
    """Forward returning per-layer gate distributions + output norms.

    For D.1 (Q/K specialization), D.4 (layer balance), D.3/D.5 (knowledge neurons).

    Returns:
        logits: [B, S, vocab]
        layer_info: dict with stacked arrays:
            gate_Q: [n_layers, B, S, n_qk]
            gate_K: [n_layers, B, S, n_qk]
            gate_V: [n_layers, B, S, n_v]
            gate_Know: [n_layers, B, S, n_know]
            attn_out_norm: [n_layers]
            know_out_norm: [n_layers]
    """
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

    def analysis_layer(carry, xs):
        x = carry
        bp = xs['params']

        normed = _layer_norm(x, bp['norm1']['scale'], bp['norm1']['bias'])
        h_all = normed @ router_params['proj_attn']['kernel'] + router_params['proj_attn']['bias']
        h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)
        tau_all = normed @ router_params['tau_attn']['kernel'] + router_params['tau_attn']['bias']

        Q, gate_Q = _srw_inference_with_gates(
            normed, h_Q, qk_norm, tau_all[:, :, 0:1],
            pool_params['qk_read'], pool_params['qk_write'])
        K, gate_K = _srw_inference_with_gates(
            normed, h_K, qk_norm, tau_all[:, :, 1:2],
            pool_params['qk_read'], pool_params['qk_write'])
        V, gate_V = _srw_inference_with_gates(
            normed, h_V, v_norm, tau_all[:, :, 2:3],
            pool_params['v_read'], pool_params['v_write'])

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
        know_out, gate_Know = _srw_inference_with_gates(
            normed, h_k, know_norm_w, tau_k,
            pool_params['know_read'], pool_params['know_write'])
        know_out_norm = jnp.linalg.norm(know_out, axis=-1).mean()
        x = x + know_out

        return x, {
            'gate_Q': gate_Q, 'gate_K': gate_K,
            'gate_V': gate_V, 'gate_Know': gate_Know,
            'attn_out_norm': attn_out_norm,
            'know_out_norm': know_out_norm,
        }

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
    qk_mult = jnp.where(suppress_masks.get('qk', jnp.zeros(1, dtype=bool)), 0.0, 1.0) \
        if 'qk' in suppress_masks else None
    v_mult = jnp.where(suppress_masks.get('v', jnp.zeros(1, dtype=bool)), 0.0, 1.0) \
        if 'v' in suppress_masks else None
    know_mult = jnp.where(suppress_masks.get('know', jnp.zeros(1, dtype=bool)), 0.0, 1.0) \
        if 'know' in suppress_masks else None

    def _srw_sup(x, h, emb_n, tau_off, w_read, w_write, mult):
        """SRW with optional gate suppression."""
        scores = h @ emb_n.T
        sf = scores.astype(jnp.float32)
        s_mean = sf.mean(axis=-1, keepdims=True)
        s_std = jnp.sqrt(jnp.mean(jnp.square(sf - s_mean), axis=-1, keepdims=True)) + 1e-8
        tau = s_mean + tau_off * s_std
        raw = scores - tau.astype(scores.dtype)
        gc = jnp.where(raw > 0, raw,
                       (1e-6 * jnp.exp(jnp.clip(raw.astype(jnp.float32), -10.0, 0.0))).astype(scores.dtype))
        gc = jnp.clip(gc, 0.0, 10.0)
        eg = (jnp.exp(gc.astype(jnp.float32)) - 1.0).astype(scores.dtype)
        if mult is not None:
            eg = eg * mult[None, None, :]
        exp_sum = eg.sum(axis=-1, keepdims=True).astype(jnp.float32) + 1e-8
        gs = jnp.tanh(eg.max(axis=-1, keepdims=True).astype(jnp.float32))
        xr = x @ w_read.T
        out = (eg * xr) @ w_write
        return (out.astype(jnp.float32) / exp_sum * gs).astype(jnp.float32)

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
            x = x + _srw_sup(normed, h_k, kn_n, tau_k, pp['know_read'], pp['know_write'], know_mult)

        norm_p = params['norm']
        x = _layer_norm(x, norm_p['scale'], norm_p['bias'])
        return x @ params['token_emb']['embedding'].T

    return forward_fn
