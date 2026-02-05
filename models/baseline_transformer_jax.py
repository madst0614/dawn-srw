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

        if self.dropout_rate > 0.0:
            rng = self.make_rng('dropout')
            keep = 1.0 - self.dropout_rate
            mask = jax.random.bernoulli(rng, keep, attn_weights.shape)
            mask = jnp.where(deterministic, jnp.ones_like(mask), mask)
            attn_weights = jnp.where(mask, attn_weights / keep, 0.0)

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
        if self.dropout_rate > 0.0:
            rng = self.make_rng('dropout')
            keep = 1.0 - self.dropout_rate
            mask = jax.random.bernoulli(rng, keep, h.shape)
            mask = jnp.where(deterministic, jnp.ones_like(mask), mask)
            h = jnp.where(mask, h / keep, 0.0)
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
        self.layers = [
            TransformerLayer(self.d_model, self.n_heads, self.d_ff,
                             self.dropout_rate, name=f'layer_{i}')
            for i in range(self.n_layers)
        ]
        self.norm = nn.LayerNorm()

    def __call__(self, input_ids, labels=None, attention_mask=None,
                 deterministic=False):
        B, S = input_ids.shape
        positions = jnp.arange(S)[jnp.newaxis, :]
        x = self.token_emb(input_ids) + self.pos_emb(positions)

        if self.dropout_rate > 0.0:
            rng = self.make_rng('dropout')
            keep = 1.0 - self.dropout_rate
            mask = jax.random.bernoulli(rng, keep, x.shape)
            mask = jnp.where(deterministic, jnp.ones_like(mask), mask)
            x = jnp.where(mask, x / keep, 0.0)

        for layer in self.layers:
            if self.gradient_checkpointing and not self.is_initializing():
                x = jax.checkpoint(layer)(x, deterministic)
            else:
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
