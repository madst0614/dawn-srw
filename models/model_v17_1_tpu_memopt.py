"""
DAWN v17.1-TPU-MemOpt: Memory-Optimized TPU Version

Base: model_v17_1_tpu.py (dense all-neurons-first computation)
Optimization: Full Recompute — Feature와 Restore 모두 중간 텐서 재계산

메모리 절감 원리:
  TPU naive:  all_h = einsum('bsd,ndr->bsnr', x, neurons)  → [B,S,N,R] 저장
              all_r = einsum('bsr,nrd->bsnd', h, neurons)  → [B,S,N,D] 저장
  MemOpt:     XLA 디바이스에서는 직접 계산 (XLA 컴파일러 자체 rematerialization 활용)
              비-XLA에서는 torch.utils.checkpoint로 중간 텐서 재계산

  XLA 컴파일러는 XLA_FLAGS='--xla_auto_spilling=true' 설정 시
  HBM 초과 시 자동으로 중간 텐서를 host memory로 spill하거나 재계산함.

변경 파일: AttentionCircuit, KnowledgeCircuit만 수정
나머지 (Router, SharedNeurons, DAWN, DAWNBlock 등)는 동일
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# XLA-compatible checkpoint: XLA에서는 직접 계산, 비-XLA에서는 torch checkpoint
try:
    from torch.utils.checkpoint import checkpoint as _torch_checkpoint
except ImportError:
    _torch_checkpoint = None


# ================================================================
# Recompute helpers (XLA-compatible)
# ================================================================

def _feature_fn(x, neurons, weights):
    """Feature stage: x [B,S,D] → h [B,S,R] via neuron pool"""
    all_h = torch.einsum('bsd,ndr->bsnr', x, neurons)
    return torch.einsum('bsnr,bsn->bsr', all_h, weights)


def feature_recompute(x, neurons, weights):
    """Feature with recompute — all_h [B,S,N,R] not saved"""
    if x.device.type == 'xla':
        # XLA: 직접 계산. XLA 컴파일러가 자체 rematerialization 처리
        return _feature_fn(x, neurons, weights)
    else:
        return _torch_checkpoint(_feature_fn, x, neurons, weights, use_reentrant=False)


def _restore_fn(h, neurons, weights):
    """Restore stage: h [B,S,R] → output [B,S,D] via neuron pool"""
    all_out = torch.einsum('bsr,nrd->bsnd', h, neurons)
    return torch.einsum('bsnd,bsn->bsd', all_out, weights)


def restore_recompute(h, neurons, weights):
    """Restore with recompute — all_out [B,S,N,D] not saved"""
    if h.device.type == 'xla':
        # XLA: 직접 계산. XLA 컴파일러가 자체 rematerialization 처리
        return _restore_fn(h, neurons, weights)
    else:
        return _torch_checkpoint(_restore_fn, h, neurons, weights, use_reentrant=False)


# ================================================================
# Unchanged modules (same as model_v17_1_tpu.py)
# ================================================================

class UnifiedNeuronRouter(nn.Module):
    """
    v17.1-TPU: 6 attention projections + 2 knowledge projections
    (Identical to model_v17_1_tpu.py)
    """
    def __init__(self, d_model, n_feature_qk, n_feature_v, n_restore_qk, n_restore_v,
                 n_feature_know, n_restore_know,
                 d_space=64, dropout=0.1, **kwargs):
        super().__init__()
        self.n_feature_qk = n_feature_qk
        self.n_feature_v = n_feature_v
        self.n_restore_qk = n_restore_qk
        self.n_restore_v = n_restore_v
        self.n_feature_know = n_feature_know
        self.n_restore_know = n_restore_know
        self.d_space = d_space
        self.ema_alpha = kwargs.get('excitability_ema_alpha', 0.01)

        total_neurons = (n_feature_qk + n_feature_v + n_restore_qk + n_restore_v +
                        n_feature_know + n_restore_know)
        self.total_neurons = total_neurons

        self.feature_qk_end = n_feature_qk
        self.feature_v_end = n_feature_qk + n_feature_v
        self.restore_qk_end = n_feature_qk + n_feature_v + n_restore_qk
        self.restore_v_end = n_feature_qk + n_feature_v + n_restore_qk + n_restore_v
        self.feature_know_end = self.restore_v_end + n_feature_know

        self.proj_all = nn.Linear(d_model, d_space * 6)
        self.proj_feature_know = nn.Linear(d_model, d_space)
        self.proj_restore_know = nn.Linear(d_model, d_space)
        self.dropout = nn.Dropout(dropout)

        self.neuron_emb = nn.Parameter(torch.randn(total_neurons, d_space) * 0.02)

        self.register_buffer('usage_ema_feature_q', torch.zeros(n_feature_qk))
        self.register_buffer('usage_ema_feature_k', torch.zeros(n_feature_qk))
        self.register_buffer('usage_ema_feature_v', torch.zeros(n_feature_v))
        self.register_buffer('usage_ema_restore_q', torch.zeros(n_restore_qk))
        self.register_buffer('usage_ema_restore_k', torch.zeros(n_restore_qk))
        self.register_buffer('usage_ema_restore_v', torch.zeros(n_restore_v))
        self.register_buffer('usage_ema_feature_know', torch.zeros(n_feature_know))
        self.register_buffer('usage_ema_restore_know', torch.zeros(n_restore_know))

    def get_knowledge_logits(self, x):
        emb_norm = F.normalize(self.neuron_emb, dim=-1)
        h_feature_know = self.dropout(self.proj_feature_know(x))
        emb_feature_know = emb_norm[self.restore_v_end:self.feature_know_end]
        logits_feature_know = torch.einsum('bsd,nd->bsn', h_feature_know, emb_feature_know)
        h_restore_know = self.dropout(self.proj_restore_know(x))
        emb_restore_know = emb_norm[self.feature_know_end:]
        logits_restore_know = torch.einsum('bsd,nd->bsn', h_restore_know, emb_restore_know)
        return logits_feature_know, logits_restore_know

    def get_all_logits(self, x):
        emb_norm = F.normalize(self.neuron_emb, dim=-1)
        all_proj = self.dropout(self.proj_all(x))
        h_fqk_Q, h_fqk_K, h_fv, h_rqk_Q, h_rqk_K, h_rv = all_proj.chunk(6, dim=-1)
        fqk_emb = emb_norm[:self.feature_qk_end]
        fv_emb = emb_norm[self.feature_qk_end:self.feature_v_end]
        rqk_emb = emb_norm[self.feature_v_end:self.restore_qk_end]
        rv_emb = emb_norm[self.restore_qk_end:self.restore_v_end]
        logits_fqk_Q = torch.einsum('bsd,nd->bsn', h_fqk_Q, fqk_emb)
        logits_fqk_K = torch.einsum('bsd,nd->bsn', h_fqk_K, fqk_emb)
        logits_fv = torch.einsum('bsd,nd->bsn', h_fv, fv_emb)
        logits_rqk_Q = torch.einsum('bsd,nd->bsn', h_rqk_Q, rqk_emb)
        logits_rqk_K = torch.einsum('bsd,nd->bsn', h_rqk_K, rqk_emb)
        logits_rv = torch.einsum('bsd,nd->bsn', h_rv, rv_emb)
        return logits_fqk_Q, logits_fqk_K, logits_fv, logits_rqk_Q, logits_rqk_K, logits_rv

    def update_usage(self, weights, neuron_type, attention_mask=None):
        if not self.training:
            return
        if weights.dim() == 3:
            active = (weights > 0).float()
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                active = active * mask
                count = mask.sum() + 1e-8
                usage = active.sum(dim=[0, 1]) / count
            else:
                usage = active.mean(dim=[0, 1])
        else:
            usage = (weights > 0).float().mean(dim=0)
        decay = 1 - self.ema_alpha
        buf = getattr(self, f'usage_ema_{neuron_type}')
        buf.mul_(decay).add_(usage, alpha=self.ema_alpha)


class SharedNeurons(nn.Module):
    """Identical to model_v17_1_tpu.py"""
    def __init__(self, d_model, rank, n_feature_qk, n_feature_v, n_restore_qk, n_restore_v,
                 n_feature_know, n_restore_know, knowledge_rank=128):
        super().__init__()
        self.d_model = d_model
        self.rank = rank
        self.knowledge_rank = knowledge_rank
        self.n_feature_qk = n_feature_qk
        self.n_feature_v = n_feature_v
        self.n_restore_qk = n_restore_qk
        self.n_restore_v = n_restore_v
        self.n_feature_know = n_feature_know
        self.n_restore_know = n_restore_know

        self.f_neurons = nn.Parameter(torch.zeros(n_feature_qk + n_feature_v, d_model, rank))
        self.r_neurons = nn.Parameter(torch.zeros(n_restore_qk + n_restore_v, rank, d_model))
        self.feature_know = nn.Parameter(torch.zeros(n_feature_know, d_model, knowledge_rank))
        self.restore_know = nn.Parameter(torch.zeros(n_restore_know, knowledge_rank, d_model))
        self._init_parameters()

    @property
    def feature_qk_neurons(self):
        return self.f_neurons[:self.n_feature_qk]

    @property
    def feature_v_neurons(self):
        return self.f_neurons[self.n_feature_qk:]

    @property
    def restore_qk_neurons(self):
        return self.r_neurons[:self.n_restore_qk]

    @property
    def restore_v_neurons(self):
        return self.r_neurons[self.n_restore_qk:]

    def _init_parameters(self):
        for i in range(self.n_feature_qk + self.n_feature_v):
            nn.init.orthogonal_(self.f_neurons.data[i])
        for i in range(self.n_restore_qk + self.n_restore_v):
            nn.init.orthogonal_(self.r_neurons.data[i])
        for i in range(self.n_feature_know):
            nn.init.orthogonal_(self.feature_know.data[i])
        for i in range(self.n_restore_know):
            nn.init.orthogonal_(self.restore_know.data[i])


class GlobalRouters(nn.Module):
    """Identical to model_v17_1_tpu.py"""
    def __init__(self, d_model, n_feature_qk, n_feature_v, n_restore_qk, n_restore_v,
                 n_feature_know, n_restore_know,
                 top_k_feature_qk=8, top_k_feature_v=8,
                 top_k_restore_qk=8, top_k_restore_v=8,
                 top_k_feature_know=4, top_k_restore_know=4,
                 d_space=64, router_dropout=0.1, **kwargs):
        super().__init__()
        self.d_model = d_model
        self.n_feature_qk = n_feature_qk
        self.n_feature_v = n_feature_v
        self.n_restore_qk = n_restore_qk
        self.n_restore_v = n_restore_v
        self.n_feature_know = n_feature_know
        self.n_restore_know = n_restore_know
        self.top_k_feature_qk = top_k_feature_qk
        self.top_k_feature_v = top_k_feature_v
        self.top_k_restore_qk = top_k_restore_qk
        self.top_k_restore_v = top_k_restore_v
        self.top_k_feature_know = top_k_feature_know
        self.top_k_restore_know = top_k_restore_know
        self.neuron_router = UnifiedNeuronRouter(
            d_model, n_feature_qk, n_feature_v, n_restore_qk, n_restore_v,
            n_feature_know, n_restore_know,
            d_space=d_space, dropout=router_dropout, **kwargs
        )

    def _topk_sparsify(self, weights, k):
        topk_vals, topk_idx = torch.topk(weights, k, dim=-1)
        sparse_weights = torch.zeros_like(weights)
        sparse_weights.scatter_(-1, topk_idx, topk_vals)
        sparse_weights = sparse_weights / (sparse_weights.sum(dim=-1, keepdim=True) + 1e-8)
        return sparse_weights, topk_idx

    def get_attention_weights(self, x, attention_mask=None, return_routing_info=False):
        (fqk_logits_Q, fqk_logits_K, fv_logits,
         rqk_logits_Q, rqk_logits_K, rv_logits) = self.neuron_router.get_all_logits(x)
        fqk_pref_Q = F.softmax(fqk_logits_Q, dim=-1)
        fqk_pref_K = F.softmax(fqk_logits_K, dim=-1)
        fv_pref = F.softmax(fv_logits, dim=-1)
        rqk_pref_Q = F.softmax(rqk_logits_Q, dim=-1)
        rqk_pref_K = F.softmax(rqk_logits_K, dim=-1)
        rv_pref = F.softmax(rv_logits, dim=-1)

        aux_loss = 0.0
        if self.training:
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                count = mask.sum() + 1e-8
                usage_fqk_Q = (fqk_pref_Q * mask).sum(dim=(0, 1)) / count
                usage_fqk_K = (fqk_pref_K * mask).sum(dim=(0, 1)) / count
                usage_fv = (fv_pref * mask).sum(dim=(0, 1)) / count
                usage_rqk_Q = (rqk_pref_Q * mask).sum(dim=(0, 1)) / count
                usage_rqk_K = (rqk_pref_K * mask).sum(dim=(0, 1)) / count
                usage_rv = (rv_pref * mask).sum(dim=(0, 1)) / count
            else:
                usage_fqk_Q = fqk_pref_Q.mean(dim=(0, 1))
                usage_fqk_K = fqk_pref_K.mean(dim=(0, 1))
                usage_fv = fv_pref.mean(dim=(0, 1))
                usage_rqk_Q = rqk_pref_Q.mean(dim=(0, 1))
                usage_rqk_K = rqk_pref_K.mean(dim=(0, 1))
                usage_rv = rv_pref.mean(dim=(0, 1))
            target_fqk = 1.0 / self.n_feature_qk
            target_fv = 1.0 / self.n_feature_v
            target_rqk = 1.0 / self.n_restore_qk
            target_rv = 1.0 / self.n_restore_v
            aux_loss += ((usage_fqk_Q - target_fqk) ** 2).sum() * self.n_feature_qk
            aux_loss += ((usage_fqk_K - target_fqk) ** 2).sum() * self.n_feature_qk
            aux_loss += ((usage_fv - target_fv) ** 2).sum() * self.n_feature_v
            aux_loss += ((usage_rqk_Q - target_rqk) ** 2).sum() * self.n_restore_qk
            aux_loss += ((usage_rqk_K - target_rqk) ** 2).sum() * self.n_restore_qk
            aux_loss += ((usage_rv - target_rv) ** 2).sum() * self.n_restore_v

        fqk_weights_Q, _ = self._topk_sparsify(fqk_pref_Q, self.top_k_feature_qk)
        fqk_weights_K, _ = self._topk_sparsify(fqk_pref_K, self.top_k_feature_qk)
        fv_weights, _ = self._topk_sparsify(fv_pref, self.top_k_feature_v)
        rqk_weights_Q, _ = self._topk_sparsify(rqk_pref_Q, self.top_k_restore_qk)
        rqk_weights_K, _ = self._topk_sparsify(rqk_pref_K, self.top_k_restore_qk)
        rv_weights, _ = self._topk_sparsify(rv_pref, self.top_k_restore_v)

        if self.training:
            self.neuron_router.update_usage(fqk_weights_Q, 'feature_q', attention_mask)
            self.neuron_router.update_usage(fqk_weights_K, 'feature_k', attention_mask)
            self.neuron_router.update_usage(fv_weights, 'feature_v', attention_mask)
            self.neuron_router.update_usage(rqk_weights_Q, 'restore_q', attention_mask)
            self.neuron_router.update_usage(rqk_weights_K, 'restore_k', attention_mask)
            self.neuron_router.update_usage(rv_weights, 'restore_v', attention_mask)

        if return_routing_info:
            routing_info = {
                'fqk_weights_Q': fqk_weights_Q.detach(),
                'fqk_weights_K': fqk_weights_K.detach(),
                'fv_weights': fv_weights.detach(),
                'rqk_weights_Q': rqk_weights_Q.detach(),
                'rqk_weights_K': rqk_weights_K.detach(),
                'rv_weights': rv_weights.detach(),
                'fqk_q_pref': fqk_pref_Q.detach(),
                'fqk_k_pref': fqk_pref_K.detach(),
                'fv_pref': fv_pref.detach(),
                'rqk_q_pref': rqk_pref_Q.detach(),
                'rqk_k_pref': rqk_pref_K.detach(),
                'rv_pref': rv_pref.detach(),
                'token_routing': True,
            }
        else:
            routing_info = None
        return fqk_weights_Q, fqk_weights_K, fv_weights, rqk_weights_Q, rqk_weights_K, rv_weights, routing_info, aux_loss

    def get_knowledge_weights(self, x, attention_mask=None, return_routing_info=False):
        logits_f, logits_r = self.neuron_router.get_knowledge_logits(x)
        pref_f = F.softmax(logits_f, dim=-1)
        pref_r = F.softmax(logits_r, dim=-1)

        aux_loss = 0.0
        if self.training:
            if attention_mask is not None:
                mask = attention_mask.unsqueeze(-1).float()
                count = mask.sum() + 1e-8
                usage_f = (pref_f * mask).sum(dim=(0, 1)) / count
                usage_r = (pref_r * mask).sum(dim=(0, 1)) / count
            else:
                usage_f = pref_f.mean(dim=(0, 1))
                usage_r = pref_r.mean(dim=(0, 1))
            target_f = 1.0 / self.n_feature_know
            target_r = 1.0 / self.n_restore_know
            aux_loss += ((usage_f - target_f) ** 2).sum() * self.n_feature_know
            aux_loss += ((usage_r - target_r) ** 2).sum() * self.n_restore_know

        feature_know_w, _ = self._topk_sparsify(pref_f, self.top_k_feature_know)
        restore_know_w, _ = self._topk_sparsify(pref_r, self.top_k_restore_know)
        if self.training:
            self.neuron_router.update_usage(feature_know_w, 'feature_know', attention_mask)
            self.neuron_router.update_usage(restore_know_w, 'restore_know', attention_mask)
        if return_routing_info:
            routing_info = {
                'feature_know_w': feature_know_w.detach(),
                'restore_know_w': restore_know_w.detach(),
                'feature_know_pref': pref_f.detach(),
                'restore_know_pref': pref_r.detach(),
            }
        else:
            routing_info = None
        return feature_know_w, restore_know_w, routing_info, aux_loss


# ================================================================
# Modified modules: AttentionCircuit, KnowledgeCircuit
# ================================================================

class AttentionCircuit(nn.Module):
    """
    v17.1-TPU-MemOpt: Full Recompute 적용 (Feature + Restore)

    변경점 (vs model_v17_1_tpu.py):
    - Feature: feature_recompute() 사용 → all_h_* 텐서 저장 안 함
    - Restore: restore_recompute() 사용 → all_r_* 텐서 저장 안 함

    주의: QK가 f_qk를 공유하는데, Full recompute에서는 x @ f_qk를 Q, K 각각 계산 (2번).
         Forward에서 1번 추가 계산되지만, all_h_qk [B,S,N_fqk,R] 저장 안 해서 메모리 절약.
    """
    def __init__(self, shared_neurons, d_model, n_heads, rank, dropout=0.1):
        super().__init__()
        self.shared_neurons = shared_neurons
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.rank = rank
        self.expand_O = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.out_dropout = nn.Dropout(dropout)

    def forward(self, x, fqk_weights_Q, fqk_weights_K, fv_weights,
                rqk_weights_Q, rqk_weights_K, rv_weights, attention_mask=None):
        B, S, D = x.shape

        # === Feature: recompute 적용 ===
        # QK는 Q, K 두 번 사용 → 개별 recompute (x @ f_qk 2번 계산)
        f_qk = self.shared_neurons.feature_qk_neurons           # [N_fqk, D, R]
        h_q = feature_recompute(x, f_qk, fqk_weights_Q)         # [B, S, R]
        h_k = feature_recompute(x, f_qk, fqk_weights_K)         # [B, S, R]

        f_v = self.shared_neurons.feature_v_neurons              # [N_fv, D, R]
        h_v = feature_recompute(x, f_v, fv_weights)              # [B, S, R]

        # === Restore: recompute 적용 ===
        r_qk = self.shared_neurons.restore_qk_neurons            # [N_rqk, R, D]
        Q = restore_recompute(h_q, r_qk, rqk_weights_Q)         # [B, S, D]
        K = restore_recompute(h_k, r_qk, rqk_weights_K)         # [B, S, D]

        r_v = self.shared_neurons.restore_v_neurons              # [N_rv, R, D]
        V = restore_recompute(h_v, r_v, rv_weights)              # [B, S, D]

        # Multi-head attention (변경 없음)
        Q = Q.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        K = K.view(B, S, self.n_heads, self.d_head).transpose(1, 2)
        V = V.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        dropout_p = self.attn_dropout.p if self.training else 0.0
        attn_out = F.scaled_dot_product_attention(
            Q, K, V, attn_mask=None, is_causal=True, dropout_p=dropout_p,
        )
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, D)
        output = self.expand_O(attn_out)
        output = self.out_dropout(output)
        return output


class KnowledgeCircuit(nn.Module):
    """
    v17.1-TPU-MemOpt: Full Recompute 적용 (Feature + Restore)

    변경점:
    - Feature: feature_recompute() 사용 → all_h 텐서 저장 안 함
    - Restore: restore_recompute() 사용 → 중간 텐서 저장 안 함
    """
    def __init__(self, shared_neurons, d_model, n_feature_know, n_restore_know,
                 knowledge_rank, top_k_feature_know=4, top_k_restore_know=4, dropout=0.1):
        super().__init__()
        self.shared_neurons = shared_neurons
        self.d_model = d_model
        self.n_feature_know = n_feature_know
        self.n_restore_know = n_restore_know
        self.knowledge_rank = knowledge_rank
        self.top_k_feature_know = top_k_feature_know
        self.top_k_restore_know = top_k_restore_know
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, feature_know_w, restore_know_w, attention_mask=None):
        B, S, D = x.shape

        # Feature: recompute 적용
        f_know = self.shared_neurons.feature_know                # [N_f, D, KR]
        h = feature_recompute(x, f_know, feature_know_w)         # [B, S, KR]

        # Restore: recompute 적용
        r_know = self.shared_neurons.restore_know                # [N_r, KR, D]
        output = restore_recompute(h, r_know, restore_know_w)    # [B, S, D]

        return self.dropout(output)


# ================================================================
# DAWNBlock, DAWN: 거의 동일, import만 달라짐
# ================================================================

class DAWNBlock(nn.Module):
    """Identical structure to model_v17_1_tpu.py"""
    def __init__(self, shared_neurons, d_model, n_heads, rank,
                 n_feature_know, n_restore_know, knowledge_rank=128,
                 top_k_feature_know=4, top_k_restore_know=4, dropout=0.1):
        super().__init__()
        self.attn = AttentionCircuit(shared_neurons, d_model, n_heads, rank, dropout)
        self.knowledge = KnowledgeCircuit(
            shared_neurons, d_model, n_feature_know, n_restore_know,
            knowledge_rank=knowledge_rank,
            top_k_feature_know=top_k_feature_know,
            top_k_restore_know=top_k_restore_know,
            dropout=dropout
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, router, attention_mask=None, return_routing_info=False):
        normed_x = self.norm1(x)
        fqk_w_Q, fqk_w_K, fv_w, rqk_w_Q, rqk_w_K, rv_w, attn_routing_info, aux_loss = router.get_attention_weights(
            normed_x, attention_mask, return_routing_info
        )
        attn_out = self.attn(normed_x, fqk_w_Q, fqk_w_K, fv_w, rqk_w_Q, rqk_w_K, rv_w, attention_mask)
        x = x + attn_out

        normed_x = self.norm2(x)
        feature_know_w, restore_know_w, know_routing_info, know_aux_loss = router.get_knowledge_weights(
            normed_x, attention_mask, return_routing_info
        )
        know_out = self.knowledge(normed_x, feature_know_w, restore_know_w, attention_mask)
        x = x + know_out

        if return_routing_info:
            routing_info = {
                'attention': attn_routing_info,
                'knowledge': know_routing_info,
                'attn_out_norm': attn_out.norm(dim=-1).mean().detach(),
                'know_out_norm': know_out.norm(dim=-1).mean().detach(),
            }
        else:
            routing_info = None
        return x, routing_info, aux_loss + know_aux_loss


class DAWN(nn.Module):
    """
    DAWN v17.1-TPU-MemOpt

    Identical to v17.1-TPU except:
    - AttentionCircuit uses feature_recompute + restore_recompute (full recompute)
    - KnowledgeCircuit uses feature_recompute + restore_recompute (full recompute)

    Trade-off: Forward에서 QK feature 계산이 2배 (Q, K 각각)
              하지만 [B,S,N,R] + [B,S,N,D] 중간 텐서를 저장 안 함
    """
    __version__ = "17.1-TPU-MemOpt"

    def __init__(self, vocab_size=30000, d_model=320, n_layers=4, n_heads=8,
                 rank=64, max_seq_len=512, d_space=64,
                 n_feature_qk=56, n_feature_v=24,
                 top_k_feature_qk=16, top_k_feature_v=6,
                 n_restore_qk=56, n_restore_v=24,
                 top_k_restore_qk=16, top_k_restore_v=6,
                 n_feature_know=24, n_restore_know=24,
                 top_k_feature_know=4, top_k_restore_know=4,
                 knowledge_rank=128,
                 dropout=0.1, router_dropout=0.1,
                 gradient_checkpointing=False, **kwargs):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by n_heads ({n_heads})")

        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.n_heads = n_heads
        self.rank = rank
        self.knowledge_rank = knowledge_rank
        self.max_seq_len = max_seq_len
        self.d_space = d_space
        self.dropout_rate = dropout
        self.router_dropout = router_dropout
        self.gradient_checkpointing = gradient_checkpointing

        self.n_feature_qk = n_feature_qk
        self.n_feature_v = n_feature_v
        self.top_k_feature_qk = top_k_feature_qk
        self.top_k_feature_v = top_k_feature_v
        self.n_restore_qk = n_restore_qk
        self.n_restore_v = n_restore_v
        self.top_k_restore_qk = top_k_restore_qk
        self.top_k_restore_v = top_k_restore_v
        self.n_feature_know = n_feature_know
        self.n_restore_know = n_restore_know
        self.top_k_feature_know = top_k_feature_know
        self.top_k_restore_know = top_k_restore_know

        # v15 compat
        self.n_feature = n_feature_qk
        self.n_neurons = n_feature_qk
        self.basis_rank = rank

        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_seq_len, d_model)
        self.emb_dropout = nn.Dropout(dropout)

        self.shared_neurons = SharedNeurons(
            d_model=d_model, rank=rank,
            n_feature_qk=n_feature_qk, n_feature_v=n_feature_v,
            n_restore_qk=n_restore_qk, n_restore_v=n_restore_v,
            n_feature_know=n_feature_know, n_restore_know=n_restore_know,
            knowledge_rank=knowledge_rank,
        )

        self.router = GlobalRouters(
            d_model=d_model,
            n_feature_qk=n_feature_qk, n_feature_v=n_feature_v,
            n_restore_qk=n_restore_qk, n_restore_v=n_restore_v,
            n_feature_know=n_feature_know, n_restore_know=n_restore_know,
            top_k_feature_qk=top_k_feature_qk, top_k_feature_v=top_k_feature_v,
            top_k_restore_qk=top_k_restore_qk, top_k_restore_v=top_k_restore_v,
            top_k_feature_know=top_k_feature_know, top_k_restore_know=top_k_restore_know,
            d_space=d_space, router_dropout=router_dropout,
        )

        self.layers = nn.ModuleList([
            DAWNBlock(
                shared_neurons=self.shared_neurons, d_model=d_model, n_heads=n_heads,
                rank=rank, n_feature_know=n_feature_know, n_restore_know=n_restore_know,
                knowledge_rank=knowledge_rank,
                top_k_feature_know=top_k_feature_know, top_k_restore_know=top_k_restore_know,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)
        self.lm_head.weight = self.token_emb.weight
        self.aux_loss = 0.0
        self._init_weights()

    def _init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, input_ids, labels=None, attention_mask=None, return_routing_info=False):
        B, S = input_ids.shape
        device = input_ids.device
        if S > self.max_seq_len:
            raise ValueError(f"Sequence length {S} exceeds max_seq_len {self.max_seq_len}")

        positions = torch.arange(S, device=device).unsqueeze(0).expand(B, -1)
        x = self.emb_dropout(self.token_emb(input_ids) + self.pos_emb(positions))

        routing_infos = [] if return_routing_info else None
        total_aux_loss = 0.0

        for layer in self.layers:
            if self.gradient_checkpointing and self.training:
                x, routing_info, aux_loss = checkpoint(
                    layer, x, self.router, attention_mask, False,
                    use_reentrant=False
                )
            else:
                x, routing_info, aux_loss = layer(x, self.router, attention_mask, return_routing_info)
            if routing_infos is not None:
                routing_infos.append(routing_info)
            total_aux_loss += aux_loss

        self.aux_loss = total_aux_loss
        x = self.norm(x)
        logits = self.lm_head(x)

        if labels is not None:
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous().long()
            loss = F.cross_entropy(shift_logits.view(-1, self.vocab_size), shift_labels.view(-1), ignore_index=-100)
            if return_routing_info:
                return loss, logits, routing_infos
            return loss, logits

        if return_routing_info:
            return logits, routing_infos
        return logits

    def get_config(self):
        return {
            'model_version': self.__version__,
            'vocab_size': self.vocab_size, 'd_model': self.d_model,
            'n_layers': self.n_layers, 'n_heads': self.n_heads,
            'rank': self.rank, 'knowledge_rank': self.knowledge_rank,
            'max_seq_len': self.max_seq_len,
            'n_feature_qk': self.n_feature_qk, 'n_feature_v': self.n_feature_v,
            'top_k_feature_qk': self.top_k_feature_qk, 'top_k_feature_v': self.top_k_feature_v,
            'n_restore_qk': self.n_restore_qk, 'n_restore_v': self.n_restore_v,
            'top_k_restore_qk': self.top_k_restore_qk, 'top_k_restore_v': self.top_k_restore_v,
            'n_feature_know': self.n_feature_know, 'n_restore_know': self.n_restore_know,
            'top_k_feature_know': self.top_k_feature_know, 'top_k_restore_know': self.top_k_restore_know,
            'd_space': self.d_space,
        }

    def get_model_info(self):
        return [
            f"DAWN v{self.__version__}: Q/K Shared + Knowledge Feature-Restore (TPU-MemOpt)",
            f"  d_model={self.d_model}, n_layers={self.n_layers}, n_heads={self.n_heads}",
            f"  rank={self.rank}, knowledge_rank={self.knowledge_rank}",
            f"  max_seq_len={self.max_seq_len}, dropout={self.dropout_rate}",
            f"",
            f"  [Attention - Q/K Shared Pool]",
            f"  Feature_QK: {self.n_feature_qk} × {self.d_model} × {self.rank} (top-k={self.top_k_feature_qk})",
            f"  Feature_V: {self.n_feature_v} × {self.d_model} × {self.rank} (top-k={self.top_k_feature_v})",
            f"  Restore_QK: {self.n_restore_qk} × {self.rank} × {self.d_model} (top-k={self.top_k_restore_qk})",
            f"  Restore_V: {self.n_restore_v} × {self.rank} × {self.d_model} (top-k={self.top_k_restore_v})",
            f"",
            f"  [Knowledge - Feature-Restore]",
            f"  Feature_Know: {self.n_feature_know} × {self.d_model} × {self.knowledge_rank} (top-k={self.top_k_feature_know})",
            f"  Restore_Know: {self.n_restore_know} × {self.knowledge_rank} × {self.d_model} (top-k={self.top_k_restore_know})",
            f"",
            f"  [Memory Optimization]",
            f"  Full Recompute: ON (Feature + Restore 모두 중간 텐서 재계산)",
            f"  Feature [B,S,N,R] + Restore [B,S,N,D] 저장 안 함",
            f"",
            f"  [Router]",
            f"  d_space={self.d_space}, router_dropout={self.router_dropout}",
            f"  token_routing=True (SSM removed)",
            f"",
            f"  [Other]",
            f"  gradient_checkpointing={self.gradient_checkpointing}",
        ]

    def knowledge_diversity_loss(self):
        feat_know = self.shared_neurons.feature_know
        rest_know = self.shared_neurons.restore_know
        feat_flat = feat_know.view(feat_know.size(0), -1)
        feat_norm = F.normalize(feat_flat, dim=-1)
        feat_sim = torch.matmul(feat_norm, feat_norm.T)
        mask_f = ~torch.eye(feat_sim.size(0), dtype=torch.bool, device=feat_sim.device)
        feat_loss = feat_sim[mask_f].abs().mean()
        rest_flat = rest_know.view(rest_know.size(0), -1)
        rest_norm = F.normalize(rest_flat, dim=-1)
        rest_sim = torch.matmul(rest_norm, rest_norm.T)
        mask_r = ~torch.eye(rest_sim.size(0), dtype=torch.bool, device=rest_sim.device)
        rest_loss = rest_sim[mask_r].abs().mean()
        return (feat_loss + rest_loss) / 2

    def orthogonality_loss(self):
        I = torch.eye(self.rank, device=self.shared_neurons.f_neurons.device).unsqueeze(0)
        W_fqk = self.shared_neurons.feature_qk_neurons
        WtW_fqk = torch.bmm(W_fqk.transpose(1, 2), W_fqk)
        loss_fqk = ((WtW_fqk - I) ** 2).mean()
        W_fv = self.shared_neurons.feature_v_neurons
        WtW_fv = torch.bmm(W_fv.transpose(1, 2), W_fv)
        loss_fv = ((WtW_fv - I) ** 2).mean()
        W_rqk = self.shared_neurons.restore_qk_neurons
        WWt_rqk = torch.bmm(W_rqk, W_rqk.transpose(1, 2))
        loss_rqk = ((WWt_rqk - I) ** 2).mean()
        W_rv = self.shared_neurons.restore_v_neurons
        WWt_rv = torch.bmm(W_rv, W_rv.transpose(1, 2))
        loss_rv = ((WWt_rv - I) ** 2).mean()
        I_know = torch.eye(self.knowledge_rank, device=self.shared_neurons.feature_know.device).unsqueeze(0)
        W_fknow = self.shared_neurons.feature_know
        WtW_fknow = torch.bmm(W_fknow.transpose(1, 2), W_fknow)
        loss_fknow = ((WtW_fknow - I_know) ** 2).mean()
        W_rknow = self.shared_neurons.restore_know
        WWt_rknow = torch.bmm(W_rknow, W_rknow.transpose(1, 2))
        loss_rknow = ((WWt_rknow - I_know) ** 2).mean()
        return (loss_fqk + loss_fv + loss_rqk + loss_rv + loss_fknow + loss_rknow) / 6

    def get_auxiliary_losses(self):
        return {
            'orth_total': self.orthogonality_loss(),
            'knowledge_div': self.knowledge_diversity_loss(),
        }

    def __repr__(self):
        params = sum(p.numel() for p in self.parameters()) / 1e6
        attn_neurons = self.n_feature_qk + self.n_feature_v + self.n_restore_qk + self.n_restore_v
        know_neurons = self.n_feature_know + self.n_restore_know
        return (
            f"DAWN v{self.__version__}: Q/K Shared + Knowledge Feature-Restore (TPU-MemOpt)\n"
            f"  Params: {params:.1f}M\n"
            f"  Attention: Feature_QK={self.n_feature_qk} (k={self.top_k_feature_qk}), "
            f"Feature_V={self.n_feature_v} (k={self.top_k_feature_v})\n"
            f"            Restore_QK={self.n_restore_qk} (k={self.top_k_restore_qk}), "
            f"Restore_V={self.n_restore_v} (k={self.top_k_restore_v})\n"
            f"  Knowledge: Feature={self.n_feature_know} (k={self.top_k_feature_know}), "
            f"Restore={self.n_restore_know} (k={self.top_k_restore_know})\n"
            f"  Total neurons: {attn_neurons} (attn) + {know_neurons} (know) = {attn_neurons + know_neurons}\n"
            f"  Memory: Full Recompute ON (Feature + Restore)"
        )
