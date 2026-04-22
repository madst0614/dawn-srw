"""
Vanilla Transformer Baseline — JAX/Flax

DAWN과 공정한 TPU 비교를 위한 Standard Transformer 구현.
train_jax.py와 동일한 인터페이스: dict 반환 (loss, aux_loss, correct, valid_count).
"""

import jax
import jax.numpy as jnp
import flax.linen as nn


def scaled_normal(scale=0.02):
    def init(key, shape, dtype=jnp.float32):
        return jax.random.normal(key, shape, dtype) * scale
    return init


class StandardAttention(nn.Module):
    d_model: int
    n_heads: int
    dropout_rate: float = 0.1

    def setup(self):
        self.d_head = self.d_model // self.n_heads
        self.q_proj = nn.Dense(self.d_model)
        self.k_proj = nn.Dense(self.d_model)
        self.v_proj = nn.Dense(self.d_model)
        self.o_proj = nn.Dense(self.d_model, use_bias=False)
        self.attn_dropout = nn.Dropout(self.dropout_rate)

    def __call__(self, x, deterministic=False):
        B, S, D = x.shape
        q = self.q_proj(x).reshape(B, S, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        k = self.k_proj(x).reshape(B, S, self.n_heads, self.d_head).transpose(0, 2, 1, 3)
        v = self.v_proj(x).reshape(B, S, self.n_heads, self.d_head).transpose(0, 2, 1, 3)

        scale = jnp.sqrt(jnp.float32(self.d_head))
        scores = jnp.einsum('bhsd,bhtd->bhst', q, k) / scale
        causal_mask = jnp.tril(jnp.ones((S, S), dtype=jnp.bool_))
        scores = jnp.where(causal_mask, scores, jnp.finfo(scores.dtype).min)
        attn_weights = jax.nn.softmax(scores, axis=-1)

        attn_weights = self.attn_dropout(attn_weights, deterministic=deterministic)

        out = jnp.einsum('bhst,bhtd->bhsd', attn_weights, v)
        out = out.transpose(0, 2, 1, 3).reshape(B, S, D)
        return self.o_proj(out)


class StandardFFN(nn.Module):
    d_model: int
    d_ff: int
    dropout_rate: float = 0.1

    @nn.compact
    def __call__(self, x, deterministic=False):
        h = nn.Dense(self.d_ff)(x)
        h = nn.gelu(h)
        h = nn.Dropout(self.dropout_rate)(h, deterministic=deterministic)
        h = nn.Dense(self.d_model)(h)
        return h


class TransformerLayer(nn.Module):
    d_model: int
    n_heads: int
    d_ff: int
    dropout_rate: float = 0.1

    def setup(self):
        self.attn = StandardAttention(self.d_model, self.n_heads, self.dropout_rate)
        self.ffn = StandardFFN(self.d_model, self.d_ff, self.dropout_rate)
        self.norm1 = nn.LayerNorm()
        self.norm2 = nn.LayerNorm()

    def __call__(self, x, deterministic=False):
        normed = self.norm1(x)
        x = x + self.attn(normed, deterministic)
        normed = self.norm2(x)
        x = x + self.ffn(normed, deterministic)
        return x


class VanillaTransformer(nn.Module):
    """Vanilla Transformer for Language Modeling — JAX/Flax

    DAWN과 동일한 인터페이스. train_jax.py에서 그대로 사용 가능.
    aux_loss=0, orthogonality_loss=0, knowledge_diversity_loss=0.
    """
    __version__ = "baseline-JAX"

    vocab_size: int = 30522
    d_model: int = 384
    d_ff: int = 1536
    n_layers: int = 12
    n_heads: int = 6
    max_seq_len: int = 512
    dropout_rate: float = 0.1
    gradient_checkpointing: bool = False

    def setup(self):
        self.token_emb = nn.Embed(self.vocab_size, self.d_model,
                                  embedding_init=scaled_normal(0.02))
        self.pos_emb = nn.Embed(self.max_seq_len, self.d_model,
                                embedding_init=scaled_normal(0.02))
        LayerCls = nn.remat(TransformerLayer) if self.gradient_checkpointing else TransformerLayer
        self.layers = [
            LayerCls(self.d_model, self.n_heads, self.d_ff,
                     self.dropout_rate, name=f'layer_{i}')
            for i in range(self.n_layers)
        ]
        self.norm = nn.LayerNorm()
        self.emb_dropout = nn.Dropout(self.dropout_rate)

    def __call__(self, input_ids, labels=None, attention_mask=None,
                 deterministic=False):
        B, S = input_ids.shape
        positions = jnp.arange(S)[jnp.newaxis, :]
        x = self.token_emb(input_ids) + self.pos_emb(positions)

        x = self.emb_dropout(x, deterministic=deterministic)

        for layer in self.layers:
            x = layer(x, deterministic)

        x = self.norm(x)

        result = {
            'aux_loss': jnp.float32(0.0),
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

    def orthogonality_loss(self):
        return jnp.float32(0.0)

    def knowledge_diversity_loss(self):
        return jnp.float32(0.0)

    def get_config(self):
        return {
            'model_version': self.__version__,
            'vocab_size': self.vocab_size, 'd_model': self.d_model,
            'd_ff': self.d_ff,
            'n_layers': self.n_layers, 'n_heads': self.n_heads,
            'max_seq_len': self.max_seq_len,
        }

    def get_model_info(self):
        return [
            f"  Model: VanillaTransformer (baseline-JAX)",
            f"  d_model={self.d_model}, d_ff={self.d_ff}, n_layers={self.n_layers}, n_heads={self.n_heads}",
            f"  max_seq_len={self.max_seq_len}, dropout={self.dropout_rate}",
            f"  gradient_checkpointing={self.gradient_checkpointing}",
        ]


# ================================================================
# KV-Cached Inference  (for fast autoregressive generation)
# ================================================================

def _layer_norm(x, scale, bias, eps=1e-6):
    """Pure functional LayerNorm."""
    mean = jnp.mean(x, axis=-1, keepdims=True)
    var = jnp.mean(jnp.square(x - mean), axis=-1, keepdims=True)
    return (x - mean) / jnp.sqrt(var + eps) * scale + bias


def _vanilla_attention_cached(x, attn_params, n_heads, d_model,
                               kv_cache_k, kv_cache_v, cache_index):
    """Standard attention with KV cache.

    Args:
        x:           [B, S, D]
        attn_params: dict with q_proj, k_proj, v_proj, o_proj
        kv_cache_k:  [B, H, max_len, d_head]
        kv_cache_v:  [B, H, max_len, d_head]
        cache_index: scalar int

    Returns:
        (output [B,S,D], updated_kv_cache_k, updated_kv_cache_v)
    """
    B, S, D = x.shape
    d_head = d_model // n_heads

    Q     = x @ attn_params['q_proj']['kernel'] + attn_params['q_proj']['bias']
    K_new = x @ attn_params['k_proj']['kernel'] + attn_params['k_proj']['bias']
    V_new = x @ attn_params['v_proj']['kernel'] + attn_params['v_proj']['bias']

    Q     = Q.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
    K_new = K_new.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)
    V_new = V_new.reshape(B, S, n_heads, d_head).transpose(0, 2, 1, 3)

    kv_cache_k = jax.lax.dynamic_update_slice(
        kv_cache_k, K_new, (0, 0, cache_index, 0))
    kv_cache_v = jax.lax.dynamic_update_slice(
        kv_cache_v, V_new, (0, 0, cache_index, 0))

    scale = jnp.sqrt(jnp.float32(d_head))
    scores = jnp.einsum('bhsd,bhtd->bhst', Q, kv_cache_k) / scale

    max_len = kv_cache_k.shape[2]
    q_positions = cache_index + jnp.arange(S)
    cache_positions = jnp.arange(max_len)
    causal = cache_positions[None, :] <= q_positions[:, None]
    scores = jnp.where(causal[None, None, :, :], scores,
                        jnp.finfo(scores.dtype).min)

    attn_weights = jax.nn.softmax(scores, axis=-1)
    attn_out = jnp.einsum('bhst,bhtd->bhsd', attn_weights, kv_cache_v)
    attn_out = attn_out.transpose(0, 2, 1, 3).reshape(B, S, D)

    output = attn_out @ attn_params['o_proj']['kernel']
    return output, kv_cache_k, kv_cache_v


def vanilla_init_kv_cache(config, batch_size=1):
    """Create zero-initialised KV caches for all layers.

    Returns:
        (all_k, all_v)  each  [n_layers, B, H, max_seq_len, d_head]
    """
    n_layers = config['n_layers']
    n_heads  = config.get('n_heads', 6)
    d_model  = config.get('d_model', 384)
    max_len  = config.get('max_seq_len', 512)
    d_head   = d_model // n_heads
    shape = (n_layers, batch_size, n_heads, max_len, d_head)
    return jnp.zeros(shape), jnp.zeros(shape)


def vanilla_cached_forward(params, config, input_ids,
                            kv_caches_k, kv_caches_v, cache_index):
    """Full forward pass with KV cache for VanillaTransformer.

    Pure function suitable for ``jax.jit``.

    Args:
        params:       param dict (may have outer 'params' key)
        config:       model config dict
        input_ids:    [B, S]
        kv_caches_k:  [n_layers, B, H, max_len, d_head]
        kv_caches_v:  [n_layers, B, H, max_len, d_head]
        cache_index:  scalar int

    Returns:
        (logits [B,S,V], updated_kv_caches_k, updated_kv_caches_v)
    """
    n_layers = config['n_layers']
    d_model  = config.get('d_model', 384)
    n_heads  = config.get('n_heads', 6)

    p = params
    if hasattr(p, 'get') and 'params' in p:
        p = p['params']

    token_emb   = p['token_emb']['embedding']
    pos_emb     = p['pos_emb']['embedding']
    norm_params = p['norm']

    B, S = input_ids.shape
    x = token_emb[input_ids]
    positions = jnp.arange(S) + cache_index
    x = x + pos_emb[positions][None, :]

    # Stack per-layer params
    layer_params_list = [p[f'layer_{i}'] for i in range(n_layers)]
    stacked_lp = jax.tree.map(lambda *a: jnp.stack(a), *layer_params_list)

    def scan_body(carry, xs):
        x = carry
        lp   = xs['params']
        kv_k = xs['kv_k']
        kv_v = xs['kv_v']

        # Attention
        normed = _layer_norm(x, lp['norm1']['scale'], lp['norm1']['bias'])
        attn_out, kv_k, kv_v = _vanilla_attention_cached(
            normed, lp['attn'], n_heads, d_model,
            kv_k, kv_v, cache_index)
        x = x + attn_out

        # FFN
        normed = _layer_norm(x, lp['norm2']['scale'], lp['norm2']['bias'])
        h = normed @ lp['ffn']['Dense_0']['kernel'] + lp['ffn']['Dense_0']['bias']
        h = jax.nn.gelu(h)
        ffn_out = h @ lp['ffn']['Dense_1']['kernel'] + lp['ffn']['Dense_1']['bias']
        x = x + ffn_out

        return x, {'kv_k': kv_k, 'kv_v': kv_v}

    xs = {
        'params': stacked_lp,
        'kv_k':   kv_caches_k,
        'kv_v':   kv_caches_v,
    }
    x, outputs = jax.lax.scan(scan_body, x, xs)

    x = _layer_norm(x, norm_params['scale'], norm_params['bias'])
    logits = x @ token_emb.T

    return logits, outputs['kv_k'], outputs['kv_v']
