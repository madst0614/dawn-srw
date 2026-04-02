"""
DAWN-Spatial v3.8: Sense-Read-Write (JAX/Flax)

Changelog:
  spatial-r1-v3.8.2 (2026-04-01):
    - d_bottleneck 64->128 (routing resolution improvement)
    - threshold_gate clamp (NaN prevention)
    - Neuron counts adjusted for param budget

  spatial-r1-v3.8.1 (2026-04-01):
    - threshold_gate: v18.5 relative tau (mean + offset*std)
    - Dead neuron gradient flow (1e-8 * exp(raw))
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
                     (1e-4 * jax.nn.softplus(raw.astype(jnp.float32))).astype(scores.dtype))

    gate = jnp.clip(gate, 0.0, 10.0)
    exp_gate = (jnp.exp(gate.astype(jnp.float32)) - 1.0).astype(scores.dtype)

    # sum/max reduce → [B,S,1], then normalize in same dtype
    exp_sum = exp_gate.sum(axis=-1, keepdims=True) + 1e-4
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

    @partial(shard_map, mesh=mesh,
             in_specs=(P('data', None, None),    # x [B,S,D]
                       P('data', None, None),    # h [B,S,d_bn]
                       P('model', None),          # emb_norm [N_local, d_bn]
                       P('data', None, None),    # tau_offset [B,S,1]
                       P('model', None),          # read [N_local, D]
                       P('model', None)),         # write [N_local, D]
             out_specs=(P('data', None, None),   # out [B,S,D]
                        P('data', None, None),   # active [B,S,1]
                        P('data', None, None)),  # gate_max [B,S,1]
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

        # --- Pass 1: scores stats -> tau (scan + checkpoint) ---
        @jax.checkpoint
        def stats_step(carry, i):
            s_sum, sq_sum = carry
            s = i * cs
            ec = jax.lax.dynamic_slice_in_dim(emb_bf, s, cs, axis=0)
            sc = h_bf @ ec.T  # bf16 [B,S,cs]
            # Reduce in bf16, accumulate in f32 — no [B,S,cs] f32 copy
            return (s_sum + sc.sum(axis=-1, keepdims=True).astype(jnp.float32),
                    sq_sum + (sc ** 2).sum(axis=-1, keepdims=True).astype(jnp.float32)), None

        (local_sum, local_sq), _ = jax.lax.scan(
            stats_step, (z1, z1), jnp.arange(nc))
        global_sum = jax.lax.psum(local_sum, 'model')
        global_sq = jax.lax.psum(local_sq, 'model')
        model_size = jax.lax.psum(jnp.int32(1), 'model')
        N_total = N_local * model_size

        s_mean = global_sum / N_total
        s_std = jnp.sqrt(global_sq / N_total - s_mean ** 2) + 1e-8
        tau = s_mean + tau_offset * s_std
        tau_bf = tau.astype(jnp.bfloat16)

        # --- Pass 2: gate + srw fused (scan + checkpoint) ---
        @jax.checkpoint
        def gate_srw_step(carry, i):
            out, exp_sum, exp_max, active = carry
            s = i * cs
            ec = jax.lax.dynamic_slice_in_dim(emb_bf, s, cs, axis=0)
            rc = jax.lax.dynamic_slice_in_dim(read_bf, s, cs, axis=0)
            wc = jax.lax.dynamic_slice_in_dim(write_bf, s, cs, axis=0)
            sc = h_bf @ ec.T
            raw = sc - tau_bf
            gc = jnp.where(raw > 0, raw,
                            (1e-4 * jax.nn.softplus(
                                raw.astype(jnp.float32))).astype(jnp.bfloat16))
            gc = jnp.clip(gc, 0.0, 10.0)
            eg = (jnp.exp(gc.astype(jnp.float32)) - 1.0).astype(jnp.bfloat16)
            ef = eg.astype(jnp.float32)
            xr = x_bf @ rc.T
            c_out = ((eg * xr) @ wc).astype(jnp.float32)
            return (out + c_out, exp_sum + ef.sum(axis=-1, keepdims=True),
                    jnp.maximum(exp_max, ef.max(axis=-1, keepdims=True)),
                    active + (raw > 0).sum(axis=-1, keepdims=True).astype(jnp.float32)), None

        (raw_out, total_es, total_em, total_ac), _ = jax.lax.scan(
            gate_srw_step,
            (jnp.zeros((B, S, D), dtype=jnp.float32),
             z1, jnp.full((B, S, 1), -1e9), z1),
            jnp.arange(nc))

        global_exp_sum = jax.lax.psum(total_es, 'model') + 1e-4
        gate_strength = jax.lax.stop_gradient(jnp.tanh(total_em))
        out = raw_out / global_exp_sum * gate_strength
        # bf16 before psum: 640MB → 320MB per all-reduce
        out = jax.lax.psum(out.astype(jnp.bfloat16), 'model')

        active = jax.lax.psum(total_ac, 'model')
        return out.astype(jnp.float32), active / N_total, total_em

    return fused_gate_srw



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

    @jax.checkpoint
    def stats_step(carry, i):
        ss, sq = carry
        s = i * cs
        ec = jax.lax.dynamic_slice_in_dim(emb_bf, s, cs, axis=0)
        sc = h_bf @ ec.T  # bf16
        return (ss + sc.sum(axis=-1, keepdims=True).astype(jnp.float32),
                sq + (sc**2).sum(axis=-1, keepdims=True).astype(jnp.float32)), None

    (s_sum, sq_sum), _ = jax.lax.scan(stats_step, (z1, z1), jnp.arange(n_chunks))
    s_mean = s_sum / N
    s_std = jnp.sqrt(sq_sum / N - s_mean**2) + 1e-8
    tau = s_mean + tau_offset * s_std
    tau_bf = tau.astype(jnp.bfloat16)

    @jax.checkpoint
    def gsrw_step(carry, i):
        out, es, em, ac = carry
        s = i * cs
        ec = jax.lax.dynamic_slice_in_dim(emb_bf, s, cs, axis=0)
        rc = jax.lax.dynamic_slice_in_dim(read_bf, s, cs, axis=0)
        wc = jax.lax.dynamic_slice_in_dim(write_bf, s, cs, axis=0)
        sc = h_bf @ ec.T
        raw = sc - tau_bf
        gc = jnp.where(raw > 0, raw,
                        (1e-4*jax.nn.softplus(raw.astype(jnp.float32))).astype(jnp.bfloat16))
        gc = jnp.clip(gc, 0.0, 10.0)
        eg = (jnp.exp(gc.astype(jnp.float32))-1.0).astype(jnp.bfloat16)
        ef = eg.astype(jnp.float32)
        xr = x_bf @ rc.T
        co = ((eg*xr)@wc).astype(jnp.float32)
        return (out+co, es+ef.sum(axis=-1,keepdims=True),
                jnp.maximum(em, ef.max(axis=-1,keepdims=True)),
                ac+(raw>0).sum(axis=-1,keepdims=True).astype(jnp.float32)), None

    (raw_out, tes, tem, tac), _ = jax.lax.scan(
        gsrw_step,
        (jnp.zeros((B,S,D), dtype=jnp.float32), z1, jnp.full((B,S,1),-1e9), z1),
        jnp.arange(n_chunks))

    inv_es = (1.0/(tes+1e-4)).astype(jnp.bfloat16)
    gs = jnp.tanh(tem).astype(jnp.bfloat16)
    out = raw_out * inv_es * gs
    return out.astype(jnp.float32), tac / N, tem


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
        fused_fn = sharded_fns
        Q, _, _ = fused_fn(x, h_Q, qk_norm, tau_all[:, :, 0:1], qk_read, qk_write)
        K, _, _ = fused_fn(x, h_K, qk_norm, tau_all[:, :, 1:2], qk_read, qk_write)
        V, _, _ = fused_fn(x, h_V, v_norm, tau_all[:, :, 2:3], v_read, v_write)
    else:
        Q, _, _ = _srw_chunked(x, h_Q, qk_norm, tau_all[:, :, 0:1],
                                qk_read, qk_write, n_chunks_qk)
        K, _, _ = _srw_chunked(x, h_K, qk_norm, tau_all[:, :, 1:2],
                                qk_read, qk_write, n_chunks_qk)
        V, _, _ = _srw_chunked(x, h_V, v_norm, tau_all[:, :, 2:3],
                                v_read, v_write, n_chunks_v)

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

    # Load balance: use tau_reg only (no full gate available from chunked)
    tau_reg = jnp.maximum(tau_all, 0.0).mean() * 0.01
    aux = tau_reg
    return out, aux


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
        fused_fn = sharded_fns
        out, active_count, gate_max_val = fused_fn(
            x, h, know_norm, tau, know_read, know_write)
    else:
        out, active_count, gate_max_val = _srw_chunked(
            x, h, know_norm, tau, know_read, know_write, n_chunks_know)

    rng, rng_out = jax.random.split(rng)
    out = safe_dropout(out, dropout_rate, deterministic, rng_out)

    # Load balance from active_count (no full gate tensor available)
    tau_reg = jnp.maximum(tau, 0.0).mean() * 0.01
    aux = tau_reg
    return out, aux, active_count, gate_max_val


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
            know_actives = jnp.float32(0.0)
            know_gmaxes = jnp.float32(0.0)
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
                attn_out, attn_aux = _attn_forward(
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
                know_out, know_aux, know_active, know_gmax = _know_forward(
                    normed, pool_params, router_params, rng_know,
                    self.max_k_know,
                    self.router_dropout, self.dropout_rate, deterministic,
                    self.n_chunks_know, sharded_fns=_sharded)
                x = x + know_out
                return x, (attn_aux + know_aux, know_active, know_gmax)

            if self.gradient_checkpointing:
                scan_body = jax.checkpoint(scan_body)

            xs = {'params': stacked, 'rng': layer_rngs}
            x, (aux_losses, know_actives, know_gmaxes) = jax.lax.scan(
                scan_body, x, xs)
            total_aux = aux_losses.sum()

        x = self.norm(x)
        result = {
            'aux_loss': total_aux,
            'know_active': know_actives.mean(),
            'know_gate_max': know_gmaxes.mean(),
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
