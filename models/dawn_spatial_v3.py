"""
DAWN-Spatial v3.4: Low-dim Sense + Bottleneck Emit + Fast Gate (JAX/Flax)

Changelog:
  spatial-r1-v3.4.1 (2026-04-01):
    - threshold_gate_fast: top_k first, gate on k items only (not N)
    - emit_bottleneck_sparse: gather w_enc[64] (0.6GB) + matmul
    - 98.6% of Knowledge time was threshold_gate on N=21000
    - Now: top_k(N) + gate(k=1536) + gather(k,64) -> ~10x faster

  spatial-r1-v3.4.0 (2026-04-01):
    - Bottleneck emit: emb[64] + w_enc[64] + w_dec[64,384]

Architecture:
  NeuronPool             -- emb[N,d_bn] + w_enc[N,d_bn] + w_dec[d_bn,D]
  Router                 -- proj + tau. Uses pool emb for routing.
  threshold_gate_fast    -- top_k first, gate on k items only
  emit_bottleneck_sparse -- gather w_enc[k,d_bn] + gate @ sel_w_enc @ w_dec
  _attn_forward          -- fast gate -> sparse emit -> self-attn
  _know_forward          -- fast gate -> sparse emit
  DAWN                   -- embedding + jax.lax.scan + weight-tied lm_head
"""

import jax
import jax.numpy as jnp
import flax.linen as nn
from typing import Optional, Dict


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
# 2. Fast gate: top_k first, then gate on k items only
# ================================================================

def threshold_gate_fast(scores, tau, max_k):
    """top_k on raw scores first, then gate computation on k items only.

    scores: [B, S, N]
    tau:    [B, S, 1]
    max_k:  int

    Returns:
        gate:    [B, S, k]  -- dense gate values for top_k items
        top_idx: [B, S, k]  -- indices of selected neurons
    """
    effective_k = min(max_k, scores.shape[-1])

    # top_k on raw scores (N -> k)
    topk_scores, topk_idx = jax.lax.top_k(scores, effective_k)

    # Gate computation on k items only (not N!)
    raw_gate = topk_scores - tau
    gate = jnp.where(raw_gate > 0, raw_gate, 1e-8 * jnp.exp(raw_gate))
    exp_gate = jnp.exp(gate) - 1.0

    gate_sum = exp_gate.sum(axis=-1, keepdims=True) + 1e-8
    gate_strength = jnp.tanh(exp_gate.max(axis=-1, keepdims=True))
    scaled = (exp_gate / gate_sum) * gate_strength

    return scaled, topk_idx


# ================================================================
# 3. Sparse bottleneck emit (gather w_enc[k, d_bn])
# ================================================================

@jax.checkpoint
def emit_bottleneck_sparse(gate, w_enc, w_dec, topk_idx):
    """Gather top_k w_enc, then gate @ sel_w_enc @ w_dec.

    gate:     [B, S, k]
    w_enc:    [N, d_bn]
    w_dec:    [d_bn, D]
    topk_idx: [B, S, k]

    gather: [B,S,k,d_bn] = [32,512,1536,64] = 0.6GB (safe)
    """
    sel_w_enc = w_enc[topk_idx]                          # [B, S, k, d_bn]
    h = jnp.einsum('bsk,bskd->bsd', gate, sel_w_enc)    # [B, S, d_bn]
    return h @ w_dec                                      # [B, S, D]


# ================================================================
# 4. NeuronPool -- emb[d_bn] + w_enc[d_bn] + w_dec[d_bn, D]
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
        self.qk_emb = self.param('qk_emb', unit_norm_init(), (self.n_qk, db))
        self.v_emb = self.param('v_emb', unit_norm_init(), (self.n_v, db))
        self.know_emb = self.param('know_emb', unit_norm_init(), (self.n_know, db))
        self.qk_w_enc = self.param('qk_w_enc', unit_norm_init(), (self.n_qk, db))
        self.v_w_enc = self.param('v_w_enc', unit_norm_init(), (self.n_v, db))
        self.know_w_enc = self.param('know_w_enc', unit_norm_init(), (self.n_know, db))
        self.qk_w_dec = self.param('qk_w_dec', scaled_normal(0.02), (db, dm))
        self.v_w_dec = self.param('v_w_dec', scaled_normal(0.02), (db, dm))
        self.know_w_dec = self.param('know_w_dec', scaled_normal(0.02), (db, dm))


# ================================================================
# 5. Router -- proj + tau (uses pool emb directly)
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
        self.tau_attn = nn.Dense(3, name='tau_attn')
        self.tau_know = nn.Dense(1, name='tau_know')

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
        g_Q, i_Q = threshold_gate_fast(h_Q @ qk_norm.T, tau_all[:, :, 0:1], self.max_k_qk)
        g_K, i_K = threshold_gate_fast(h_K @ qk_norm.T, tau_all[:, :, 1:2], self.max_k_qk)
        g_V, i_V = threshold_gate_fast(h_V @ v_norm.T, tau_all[:, :, 2:3], self.max_k_v)

        # Load balance (softmax on full scores)
        scores_qk = h_Q @ qk_norm.T
        scores_v = h_V @ v_norm.T
        usage_qk = jax.nn.softmax(scores_qk, axis=-1).mean(axis=(0, 1))
        usage_v = jax.nn.softmax(scores_v, axis=-1).mean(axis=(0, 1))
        t_qk = 1.0 / self.n_qk
        t_v = 1.0 / self.n_v
        aux = (
            ((usage_qk - t_qk) ** 2).sum() * self.n_qk * 3 +
            ((usage_v - t_v) ** 2).sum() * self.n_v
        )
        return g_Q, i_Q, g_K, i_K, g_V, i_V, aux

    def get_knowledge_gates(self, x, neuron_pool, deterministic, rng):
        know_norm = neuron_pool.know_emb / (
            jnp.linalg.norm(neuron_pool.know_emb, axis=-1, keepdims=True) + 1e-8)

        rng, rng_drop = jax.random.split(rng)
        h = self.proj_know(x)
        h = safe_dropout(h, self.router_dropout, deterministic, rng_drop)
        scores = h @ know_norm.T

        tau = self.tau_know(x)
        gate, idx = threshold_gate_fast(scores, tau, self.max_k_know)

        usage = jax.nn.softmax(scores, axis=-1).mean(axis=(0, 1))
        t = 1.0 / self.n_know
        aux = ((usage - t) ** 2).sum() * self.n_know
        return gate, idx, aux


# ================================================================
# 6. Pure functions for scan body
# ================================================================

def _attn_forward(x, pool_params, router_params, expand_O_kernel, rng,
                  n_qk, n_v,
                  max_k_qk, max_k_v, n_heads, d_model,
                  router_dropout, dropout_rate, deterministic):
    B, S, D = x.shape
    qk_emb = pool_params['qk_emb']
    qk_w_enc = pool_params['qk_w_enc']
    qk_w_dec = pool_params['qk_w_dec']
    v_emb = pool_params['v_emb']
    v_w_enc = pool_params['v_w_enc']
    v_w_dec = pool_params['v_w_dec']

    qk_norm = qk_emb / (jnp.linalg.norm(qk_emb, axis=-1, keepdims=True) + 1e-8)
    v_norm = v_emb / (jnp.linalg.norm(v_emb, axis=-1, keepdims=True) + 1e-8)

    rng, rng_drop = jax.random.split(rng)
    h_all = x @ router_params['proj_attn']['kernel'] + router_params['proj_attn']['bias']
    h_all = safe_dropout(h_all, router_dropout, deterministic, rng_drop)
    h_Q, h_K, h_V = jnp.split(h_all, 3, axis=-1)

    scores_qk = h_Q @ qk_norm.T
    scores_v = h_V @ v_norm.T

    tau_all = x @ router_params['tau_attn']['kernel'] + router_params['tau_attn']['bias']

    g_Q, i_Q = threshold_gate_fast(scores_qk, tau_all[:, :, 0:1], max_k_qk)
    g_K, i_K = threshold_gate_fast(h_K @ qk_norm.T, tau_all[:, :, 1:2], max_k_qk)
    g_V, i_V = threshold_gate_fast(scores_v, tau_all[:, :, 2:3], max_k_v)

    Q = emit_bottleneck_sparse(g_Q, qk_w_enc, qk_w_dec, i_Q)
    K = emit_bottleneck_sparse(g_K, qk_w_enc, qk_w_dec, i_K)
    V = emit_bottleneck_sparse(g_V, v_w_enc, v_w_dec, i_V)

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

    rng, rng_attn, rng_out = jax.random.split(rng, 3)
    attn_w = safe_dropout(attn_w, dropout_rate, deterministic, rng_attn)

    out = jnp.einsum('bhst,bhtd->bhsd', attn_w, V)
    out = out.transpose(0, 2, 1, 3).reshape(B, S, D)
    out = out @ expand_O_kernel
    out = safe_dropout(out, dropout_rate, deterministic, rng_out)

    # Load balance
    usage_qk = jax.nn.softmax(scores_qk, axis=-1).mean(axis=(0, 1))
    usage_v = jax.nn.softmax(scores_v, axis=-1).mean(axis=(0, 1))
    t_qk = 1.0 / n_qk
    t_v = 1.0 / n_v
    aux = (
        ((usage_qk - t_qk) ** 2).sum() * n_qk * 3 +
        ((usage_v - t_v) ** 2).sum() * n_v
    )
    return out, aux


def _know_forward(x, pool_params, router_params, rng,
                  max_k_know,
                  router_dropout, dropout_rate, deterministic):
    know_emb = pool_params['know_emb']
    know_w_enc = pool_params['know_w_enc']
    know_w_dec = pool_params['know_w_dec']

    know_norm = know_emb / (jnp.linalg.norm(know_emb, axis=-1, keepdims=True) + 1e-8)

    rng, rng_drop = jax.random.split(rng)
    h = x @ router_params['proj_know']['kernel'] + router_params['proj_know']['bias']
    h = safe_dropout(h, router_dropout, deterministic, rng_drop)
    scores = h @ know_norm.T

    tau = x @ router_params['tau_know']['kernel'] + router_params['tau_know']['bias']
    gate, topk_idx = threshold_gate_fast(scores, tau, max_k_know)

    out = emit_bottleneck_sparse(gate, know_w_enc, know_w_dec, topk_idx)

    rng, rng_out = jax.random.split(rng)
    out = safe_dropout(out, dropout_rate, deterministic, rng_out)

    usage = jax.nn.softmax(scores, axis=-1).mean(axis=(0, 1))
    t = 1.0 / know_emb.shape[0]
    aux = ((usage - t) ** 2).sum() * know_emb.shape[0]
    return out, aux


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

        g_Q, i_Q, g_K, i_K, g_V, i_V, aux = router.get_attention_gates(
            x, neuron_pool, deterministic, rng_r)

        Q = emit_bottleneck_sparse(g_Q, neuron_pool.qk_w_enc, neuron_pool.qk_w_dec, i_Q)
        K = emit_bottleneck_sparse(g_K, neuron_pool.qk_w_enc, neuron_pool.qk_w_dec, i_K)
        V = emit_bottleneck_sparse(g_V, neuron_pool.v_w_enc, neuron_pool.v_w_dec, i_V)

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
        gate, idx, aux = router.get_knowledge_gates(
            x, neuron_pool, deterministic, rng_r)
        out = emit_bottleneck_sparse(
            gate, neuron_pool.know_w_enc, neuron_pool.know_w_dec, idx)
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
    """DAWN-Spatial v3.4.1: Fast Gate + Sparse Bottleneck Emit."""
    __version__ = "spatial-r1-v3.4.1"

    vocab_size: int = 30000
    d_model: int = 384
    n_layers: int = 12
    n_heads: int = 6
    max_seq_len: int = 512
    dropout_rate: float = 0.1
    gradient_checkpointing: bool = False

    d_bottleneck: int = 64
    n_qk: int = 1570
    n_v: int = 2620
    n_know: int = 21000
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
                 deterministic=False):
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

                normed = _layer_norm(
                    x, bp['norm1']['scale'], bp['norm1']['bias'])
                attn_out, attn_aux = _attn_forward(
                    normed, pool_params, router_params,
                    bp['attn']['expand_O']['kernel'], rng_attn,
                    self.n_qk, self.n_v,
                    self.max_k_qk, self.max_k_v,
                    self.n_heads, self.d_model,
                    self.router_dropout, self.dropout_rate, deterministic)
                x = x + attn_out

                normed = _layer_norm(
                    x, bp['norm2']['scale'], bp['norm2']['bias'])
                know_out, know_aux = _know_forward(
                    normed, pool_params, router_params, rng_know,
                    self.max_k_know,
                    self.router_dropout, self.dropout_rate, deterministic)
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
        return (_div(pool.qk_emb) + _div(pool.qk_w_enc) +
                _div(pool.v_emb) + _div(pool.v_w_enc) +
                _div(pool.know_emb) + _div(pool.know_w_enc)) / 6

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
            f"DAWN v{self.__version__}: Fast Gate + Sparse Bottleneck Emit",
            f"  d_model={self.d_model}, d_bottleneck={self.d_bottleneck}, "
            f"n_layers={self.n_layers}, n_heads={self.n_heads}",
            f"  QK: {self.n_qk}, V: {self.n_v}, Know: {self.n_know}",
            f"  max_k: qk={self.max_k_qk}, v={self.max_k_v}, "
            f"know={self.max_k_know}",
        ]
