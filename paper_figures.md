# DAWN Paper Figures & Tables - Data Source Documentation

이 문서는 DAWN 논문에 포함되는 모든 Figure와 Table의 데이터 소스, 계산 로직, 관련 코드를 정리합니다.

---

## 목차

1. [Figures](#figures)
   - [Fig 3: Q/K Specialization](#fig-3-qk-specialization)
   - [Fig 4: POS Neuron Specialization](#fig-4-pos-neuron-specialization)
   - [Fig 5: Semantic Coherence](#fig-5-semantic-coherence-factual-heatmap)
   - [Fig 6: Training Dynamics](#fig-6-training-dynamics)
   - [Fig 7: Layer-wise Contribution](#fig-7-layer-wise-contribution)
2. [Tables](#tables)
   - [Table 1: Model Statistics](#table-1-model-statistics)
   - [Table 2: Neuron Utilization](#table-2-neuron-utilization)
3. [Text Outputs](#text-outputs)
4. [Data Flow Summary](#data-flow-summary)
5. [Review Checklist](#review-checklist)

---

## Figures

### Fig 3: Q/K Specialization

**목적**: 같은 QK pool에서 Q와 K가 다른 뉴런을 선택하는지 보여줌

#### 관련 파일
```
scripts/analysis/
├── routing.py
│   ├── analyze_qk_overlap() [line 329-426]  # Jaccard similarity
│   └── analyze_qk_usage() [line 428-581]    # Per-neuron counts & correlation
│
├── paper_figures.py
│   └── generate_figure3() [line 401-427]
│
└── visualizers/qk_specialization.py
    └── plot_qk_specialization()
```

#### 계산 로직
```python
# routing.py - analyze_qk_usage()

# 1. 각 뉴런별 Q/K 선택 횟수 카운트 (line 502-510)
selected_q = (w_q > 0).float().sum(dim=[0, 1])  # top-k weight > 0 = 선택됨
selected_k = (w_k > 0).float().sum(dim=[0, 1])

# 2. 모든 배치/레이어 합산 (line 527-530)
for lidx, (q_counts, k_counts, _) in layer_data.items():
    total_q += q_counts
    total_k += k_counts

# 3. Pearson correlation 계산 (line 549-552)
q_np = total_q.cpu().numpy()
k_np = total_k.cpu().numpy()
corr = np.corrcoef(q_np, k_np)[0, 1]
```

#### Measurement Method (논문용)

**Primary Metric: Pearson Correlation**
```
For each neuron i in the shared Q/K pool, we count:
- c_i^Q: number of times selected for Q projection
- c_i^K: number of times selected for K projection

Correlation: r = pearson(c^Q, c^K) across all neurons
Interpretation: r < 0 → neurons specialize (Q-popular ≠ K-popular)
```

**Secondary Metric: Categorical Classification (ratio 기반 권장)**
```python
total = q_np + k_np
q_ratio = q_np / (total + 1e-8)  # 0~1

# Active 기준: 하위 10% 제외
min_usage = np.percentile(total, 10)
active = total > min_usage

# Specialization 기준: 70/30 split
q_specialized = (active & (q_ratio > 0.7)).sum()
k_specialized = (active & (q_ratio < 0.3)).sum()
shared = (active & (q_ratio >= 0.3) & (q_ratio <= 0.7)).sum()
inactive = (~active).sum()
```

**70/30 threshold justification:**
- 명확한 majority preference (>2:1 ratio)
- Q/K에 symmetric한 기준
- Sensitivity analysis로 robustness 검증 (60-80% 범위)

#### 검증 완료 ✅
| 항목 | 상태 | 비고 |
|------|------|------|
| `w > 0` 체크 | ✅ | top-k sparsity와 정합 (선택된 뉴런만 양수) |
| Weight 소스 | ✅ | `fqk_weights_Q` 사용 (top-k 후, softmax 전 logits 아님) |
| 집계 범위 | ✅ | 모든 레이어/배치 합산 |
| Correlation 계산 | ✅ | `np.corrcoef()` 정확 |
| Specialization | ✅ | ratio 기반으로 수정 완료 (commit e4e1246) |

#### 구현 완료 ✅ (2025-01-21)
```python
# routing.py line 554-603 - 수정된 코드
total_usage = q_np + k_np
q_ratio = np.zeros_like(q_np, dtype=float)
valid_mask = total_usage > 0
q_ratio[valid_mask] = q_np[valid_mask] / total_usage[valid_mask]

# Inactive: 하위 10% by total usage
usage_threshold = np.percentile(total_usage, 10)
inactive_mask = total_usage <= usage_threshold

# Ratio 기반 분류
active_mask = ~inactive_mask
q_specialized = int(((q_ratio > 0.7) & active_mask).sum())
k_specialized = int(((q_ratio < 0.3) & active_mask).sum())
shared = int(((q_ratio >= 0.3) & (q_ratio <= 0.7) & active_mask).sum())
inactive = int(inactive_mask.sum())

# Histogram용 데이터
q_ratio_active = q_ratio[active_mask].tolist()
```

**출력 데이터:**
- `q_ratio`: 뉴런별 Q 비율 (scatter plot용)
- `q_ratio_active`: 활성 뉴런만 (histogram용)
- `specialization_thresholds`: 사용된 임계값 기록
- `sensitivity_analysis`: 여러 threshold별 결과 (논문 robustness 검증용)

```python
# sensitivity_analysis 구조
{
    '0.6': {'q_specialized': N, 'k_specialized': M, 'shared': S, 'total_active': T},
    '0.65': {...},
    '0.7': {...},
    '0.75': {...},
    '0.8': {...},
}
```

#### 데이터 소스
- `self.results['routing']['qk_usage']` (analyze_qk_usage 결과)
- 측정 범위: n_batches × batch_size × seq_len × n_layers

#### 출력
- (a) Q vs K scatter plot + correlation coefficient
- (b) Specialization 분포 histogram (Q-only, K-only, Shared, Inactive)

---

### Fig 4: POS Neuron Specialization

**목적**: 뉴런별 POS 선호 패턴을 보여줌 (selectivity heatmap)

#### 관련 파일
```
scripts/analysis/
├── pos_neuron.py
│   └── NeuronFeatureAnalyzer
│       ├── _build_matrices() [line 3604-3685]      # activation matrix 구성
│       ├── _compute_neuron_features() [line 3687-3732]
│       ├── compute_selectivity_matrix() [NEW]      # selectivity score 계산
│       ├── detect_specialized_neurons() [line 3830-3920]  # 80%+ threshold (appendix용)
│       └── run_full_analysis()
│
├── paper_figures.py
│   └── generate_figure4() [line 428-472]
│
└── visualizers/pos_neurons.py
    ├── plot_pos_selectivity_heatmap() [NEW]        # Main figure
    ├── plot_pos_selectivity_clustered() [NEW]      # Clustered version
    └── plot_pos_specialization_from_features()     # Appendix용 (80% threshold)
```

#### Main Figure: Selectivity Heatmap

**Selectivity Score 정의:**
```python
# compute_selectivity_matrix()

Selectivity(neuron, POS) = P(neuron active | POS) / P(neuron active)

# 해석:
# > 1: 뉴런이 해당 POS를 선호 (baseline보다 더 자주 활성화)
# = 1: 뉴런이 해당 POS에 무관
# < 1: 뉴런이 해당 POS를 회피 (baseline보다 덜 활성화)
```

**계산 로직:**
```python
# pos_neuron.py - compute_selectivity_matrix()

total_tokens = len(token_data)
pos_totals = pos_matrix.sum(axis=0)  # 각 POS의 총 토큰 수

for neuron_idx in active_neurons:
    p_neuron = neuron_totals[neuron_idx] / total_tokens  # P(neuron active)

    for pos_idx in range(n_pos):
        # P(neuron active | POS)
        p_neuron_given_pos = neuron_pos_counts[neuron_idx, pos_idx] / pos_totals[pos_idx]

        selectivity[neuron_idx, pos_idx] = p_neuron_given_pos / p_neuron
```

**Heatmap 시각화:**
```python
# visualizers/pos_neurons.py - plot_pos_selectivity_heatmap()

# Log2 scale for visualization (centered at 0 = selectivity 1)
log_selectivity = np.log2(selectivity)

# Color: Red = prefer (>1), Blue = avoid (<1), White = neutral (=1)
# Range: -2 to +2 (0.25x to 4x)
```

#### Appendix: 80% Threshold Specialization

**Concentration Metric (기존 방식):**
```
concentration(neuron, POS) = activations_on_POS / total_activations_of_neuron

Specialized if: max(concentration) >= 80%
```

**참고**: 80% threshold에서 5/1103 뉴런만 specialized (0.5%) →
Selectivity heatmap이 더 informative함

#### 검증 완료 ✅
| 항목 | 상태 | 비고 |
|------|------|------|
| 활성화 판정 | ✅ | 우선 model mask, fallback weight > 0 |
| Concentration 계산 | ✅ | count_for_POS / total * 100 |
| 80% threshold | ✅ | Reasonable - overwhelming majority |
| 모든 pool 포함 | ✅ | 6 attention + 2 knowledge pools concatenated |

#### 주의사항 ⚠️
- v17.1 모델은 routing_info에 mask 저장 안 함 → fallback 사용
- fallback은 `weight > 0` 사용 (top-k sparsity 후라 non-zero = active)

#### 데이터 소스
- Universal Dependencies English Web Treebank (UD EWT)
- POS tags: 17개 UPOS categories
- `NeuronFeatureAnalyzer.run_full_analysis()` 결과

#### 출력 데이터 구조

**Main: Selectivity (paper_results.json)**
```python
'fig4_pos_specialization': {
    'selectivity': {
        'top_selective_per_pos': {
            'NOUN': [{'neuron': 45, 'selectivity': 3.2}, ...],
            'VERB': [...],
            ...
        },
        'mean_selectivity_by_pos': {'NOUN': 1.2, 'VERB': 0.9, ...},
        'selectivity_range': {'mean': 0.8, 'std': 0.3, 'max': 2.5},
        'n_active_neurons': 1000,
    },
    # Appendix용 (80% threshold)
    'specialized_neurons_80pct': 5,
    'specialized_ratio_80pct': 0.005,
    ...
}
```

**Full Selectivity Matrix (별도 파일)**
```python
# neuron_features/selectivity_matrix.npy
selectivity_matrix: np.ndarray  # [n_neurons, n_pos]
```

**Appendix: Specialization Summary**
```python
'specialization_summary': {
    'total_neurons': N,
    'specialized_count': {
        '60%': {'pos': X1, 'position': Y1, ...},
        '70%': {'pos': X2, 'position': Y2, ...},
        '80%': {'pos': X3, 'position': Y3, ...},
    },
    'specialized_ratio': {
        '60%': {'pos': X1/N, ...},
        '70%': {'pos': X2/N, ...},
        '80%': {'pos': X3/N, ...},
    },
    'by_pool': {
        'fqk_q': {'total': N1, 'specialized_60': S1, 'specialized_70': S2, 'specialized_80': S3},
        'fqk_k': {...},
        'fv': {...},
        'rqk_q': {...},
        'rqk_k': {...},
        'rv': {...},
        'fknow': {...},
        'rknow': {...},
    }
}
```

#### 출력
- (a) Top 20 POS-specialized neurons (concentration ≥80%)
- (b) Number of specialized neurons per POS category

---

### Fig 5: Semantic Coherence (Factual Heatmap)

**목적**: 관련된 사실(예: 수도 이름들)이 유사한 뉴런 패턴을 공유함을 보여줌

#### 논문 Caption
```
Figure 5: Factual knowledge neuron activation in F-Know pool.
Left section shows neurons consistently activated across all factual queries (shared).
Middle section shows capital-city-specific neurons.
Right section shows neurons specific to color queries.
Dashed line separates geographic (Paris, London, Tokyo) from non-geographic (Blue) targets.
```

#### 관련 파일
```
scripts/analysis/
├── behavioral.py
│   └── analyze_factual_neurons() [line 518-767]
│       ├── Generation loop [line 599-706]
│       ├── Neuron collection [line 643-666]
│       └── Result computation [line 726-760]
│
├── paper_figures.py
│   └── generate_figure5() [line 474-527]
│
└── visualizers/factual_heatmap.py
    └── plot_factual_heatmap() [line 31+]
```

#### 계산 로직
```python
# behavioral.py - analyze_factual_neurons()

# 1. Generation loop (line 599-706)
while successful_runs < min_target_count and total_runs < max_runs:
    generated = base_input_ids.clone()  # 매 run fresh start

    for step in range(max_tokens_per_run):
        outputs = self.model(generated, return_routing_info=True)
        routing = self.extractor.extract(outputs)

        # 2. 활성 뉴런 추출 - 모든 레이어에서 (line 644-666)
        step_neurons = set()  # SHARED POOL: set으로 중복 제거
        for layer_idx, layer in enumerate(routing):
            m = layer.get_mask(pool_type)       # 우선: binary mask
            if m is None:
                w = layer.get_weight(pool_type)  # fallback: weight > 0
                active = (w > 0).nonzero()
            else:
                active = m.nonzero()
            step_neurons.update(active)  # neuron index = identity

        # 3. Target 발견 시 뉴런 카운트 (line 674-685)
        if target_lower in token_text.strip().lower():
            for n in step_neurons:
                target_neuron_counts[n] += 1
            successful_runs += 1
            break
        else:
            # Baseline: target 아닌 step들
            for n in step_neurons:
                baseline_neuron_counts[n] += 1

# 4. 결과 계산 (line 726-760)
target_freq = {n: count / successful_runs for n, count in target_neuron_counts.items()}
common_neurons_100 = [n for n, f in target_freq.items() if f >= 1.0]
common_neurons_80 = [n for n, f in target_freq.items() if f >= 0.8]

# 5. Contrastive score (target-specific 뉴런 찾기)
contrastive_score[n] = target_freq[n] - baseline_freq[n]
```

#### Measurement Method (논문용)

**Approach: Independent Generation Runs**
```
1. Start from prompt (e.g., "The capital of France is")
2. Generate tokens until target appears (e.g., "Paris")
3. Record active neurons at target generation step
4. Repeat for min_target_count runs (default: 100)
5. Compute frequency: how often each neuron activates when generating target
```

**Shared Pool Design:**
```
DAWN uses shared neuron pool across layers:
- Neuron index i in Layer 0 = same physical neuron as index i in Layer 5
- Therefore: collect unique neuron indices via set(), not layer-specific
- This enables cross-layer neuron reuse analysis
```

**Contrastive Score:**
```
For each neuron n:
  contrastive_score[n] = target_freq[n] - baseline_freq[n]

High positive score → neuron is target-specific (활성이 target에 집중)
```

#### 검증 완료 ✅
| 항목 | 상태 | 비고 |
|------|------|------|
| Shared pool 처리 | ✅ | `set()`으로 중복 제거, index = neuron identity |
| Mask 우선 사용 | ✅ | `get_mask()` 우선, fallback `weight > 0` |
| Target step만 기록 | ✅ | target 발견 step의 뉴런만 카운트 |
| Contrastive score | ✅ | baseline 대비 target-specific 뉴런 식별 |

#### 주의사항 ⚠️
- v17.1은 routing_info에 mask 없음 → fallback 사용
- top-k 후 `weight > 0` 사용 (non-zero = active)
- `pool_type` 파라미터로 분석 대상 pool 선택 가능 (기본: 'fv')

#### 데이터 소스
- `self.results['factual']`
- Example prompts: "The capital of France is" → "Paris"

#### 출력
- Heatmap: Target × Neuron (activation frequency %)
- Common neurons shared across semantic category (e.g., capitals)

---

### Fig 6: Training Dynamics

**목적**: DAWN vs Vanilla Transformer의 학습 곡선 비교

#### 관련 파일
```
scripts/analysis/
├── paper_figures.py
│   └── generate_figure6() [line 528-625]
│
└── visualizers/training_dynamics.py
    └── plot_training_dynamics()
```

#### 계산 로직
```python
# paper_figures.py - training_log.txt 파싱
# 형식: "Step 1000: loss=3.5432, val_loss=3.6789, ..."
for line in log_file:
    match = re.search(r'Step (\d+).*val_loss=([0-9.]+)', line)
    if match:
        steps.append(int(match.group(1)))
        losses.append(float(match.group(2)))
```

#### 검증 포인트
| 항목 | 상태 | 비고 |
|------|------|------|
| 로그 파싱 | ✅ | regex로 step, val_loss 추출 |
| 비교 조건 | ⚠️ | 두 모델이 같은 조건인지 확인 필요 |

#### 확인 필요 ⚠️
- [ ] 같은 데이터셋으로 학습했는가?
- [ ] 같은 step 수 학습했는가?
- [ ] 같은 evaluation 방식인가?
- [ ] Parameter 수가 comparable한가?

#### 데이터 소스
- `{checkpoint_path}/training_log.txt`
- DAWN checkpoint + Vanilla checkpoint

#### 출력
- Validation loss curves (DAWN vs Vanilla)
- X축: Training steps, Y축: Validation loss

---

### Fig 7: Layer-wise Contribution

**목적**: 각 레이어에서 Attention vs Knowledge circuit의 기여도 비율

#### 관련 파일
```
scripts/analysis/
├── routing.py
│   └── analyze_layer_contribution() [line 687-766]
│
├── utils.py
│   └── LayerRoutingData.get_output_norms() [line 906-924]
│
├── paper_figures.py
│   └── generate_figure7() [line 626-675]
│
└── visualizers/layer_contribution.py
    └── plot_routing_stats()
```

#### 계산 로직
```python
# 1. 모델이 forward에서 저장 (model_v17_1.py line 711-718)
attn_out_norm = attn_out.norm(dim=-1).mean().detach()
know_out_norm = know_out.norm(dim=-1).mean().detach()
routing_info = {
    'attn_out_norm': attn_out_norm,
    'know_out_norm': know_out_norm,
}

# 2. 분석 코드에서 추출 (routing.py line 726-732)
for layer in routing:
    norms = layer.get_output_norms()
    layer_norms[layer_idx]['attn'].append(norms['attn_out_norm'])
    layer_norms[layer_idx]['know'].append(norms['know_out_norm'])

# 3. Attention ratio 계산 (routing.py line 750)
attention_ratio = attn_norm / (attn_norm + know_norm) * 100
```

#### Measurement Method (논문용)
```
Output Norm Based Measurement:
- attn_out_norm = ||attention_circuit_output||₂ averaged over tokens
- know_out_norm = ||knowledge_circuit_output||₂ averaged over tokens
- attention_ratio = attn_norm / (attn_norm + know_norm)

This measures actual output magnitude contribution, not routing weights.
```

#### 검증 완료 ✅
| 항목 | 상태 | 비고 |
|------|------|------|
| Output norm 저장 | ✅ | 모델이 forward에서 저장 (line 711-718) |
| Norm 추출 | ✅ | `layer.get_output_norms()` 사용 |
| Ratio 계산 | ✅ | `attn / (attn + know) * 100` |
| 레이어별 집계 | ✅ | 모든 레이어 개별 분석 |

#### 데이터 소스
- `self.results['routing']['layer_contribution']`
- 모델이 forward pass에서 저장하는 `attn_out_norm`, `know_out_norm`

#### 출력
- Stacked bar chart: Attention vs Knowledge per layer
- Layer 0~N의 기여도 비율

---

## Tables

### Table 1: Model Statistics

**목적**: DAWN vs Vanilla 모델 비교 (파라미터, 성능)

#### 관련 파일
```
scripts/analysis/analyze_all.py
├── analyze_model_info() [line 200-265]
├── analyze_performance() [line 267-380]
├── _generate_tables() [line 1717-1833]
│   └── model_stats.csv / model_stats.tex
└── _generate_paper_results_json() [line 2360-2470]
    └── paper_results.json['table1_model_stats']

scripts/evaluation/evaluate.py
└── estimate_flops() [line 50-95]
```

#### 데이터 항목
| Metric | Source |
|--------|--------|
| parameters_M | `model_info['total_M']` |
| flops_G | `model_info['flops_G']` |
| perplexity | `performance['validation']['perplexity']` |
| accuracy | `performance['validation']['accuracy']` |
| tokens_per_sec | `performance['speed']['tokens_per_sec']` |

#### 파라미터 계산 로직 (CRITICAL)

```python
# analyze_all.py - analyze_model_info() [line 209-223]

# 1. 파라미터 카운트 (실제 로드된 PyTorch 모델에서)
total_params = sum(p.numel() for p in self.model.parameters())
trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

# 2. FLOPs 추정 (evaluate.py - estimate_flops)
from scripts.evaluation.evaluate import estimate_flops
flops = estimate_flops(self.model, seq_len=512)

# 3. 결과 구조
params_info = {
    'total': total_params,           # 전체 파라미터 수
    'trainable': trainable_params,   # 학습 가능 파라미터 수
    'total_M': total_params / 1e6,   # 백만 단위
    'flops': flops,                  # 총 FLOPs
    'flops_G': flops / 1e9,          # 10억 단위
}
```

#### FLOPs 추정 로직

```python
# evaluate.py - estimate_flops() [line 50-95]

def estimate_flops(model, seq_len=512):
    d_model = getattr(model, 'd_model', 512)
    n_layers = getattr(model, 'n_layers', 12)
    vocab_size = getattr(model, 'vocab_size', 30000)

    if hasattr(model, 'shared_neurons'):
        # DAWN model
        rank = getattr(model, 'rank', 64)
        top_k_qk = getattr(model, 'top_k_feature_qk', 20)
        top_k_v = getattr(model, 'top_k_feature_v', 6)
        knowledge_rank = getattr(model, 'knowledge_rank', 128)
        top_k_know = getattr(model, 'top_k_feature_know', 4)

        # Per layer FLOPs:
        attn_proj = 2 * (top_k_qk * 2 + top_k_v) * d_model * rank * seq_len
        attn_scores = 2 * seq_len * seq_len * d_model
        knowledge = 2 * 2 * top_k_know * d_model * knowledge_rank * seq_len
        per_layer = attn_proj + attn_scores + knowledge
    else:
        # Vanilla transformer
        # 4 * d_model^2 * seq_len (QKV + O projections)
        # + 2 * seq_len^2 * d_model (attention)
        # + 8 * d_model^2 * seq_len (FFN, assuming 4x expansion)
        per_layer = 4 * d_model * d_model * seq_len + \
                    2 * seq_len * seq_len * d_model + \
                    8 * d_model * d_model * seq_len
```

#### 주의사항 ⚠️

**Table 1 vs Fig 6 일관성**:
- Table 1: `model_info.get('total_M')` - 실제 로드된 모델에서 계산
- Fig 6: 동일한 `model_info` 사용 (수정 후)
- **절대로 checkpoint state_dict에서 별도 카운트하지 않음**

```python
# paper_results.json 생성 시 (analyze_all.py line 2578-2583)
paper_results['fig6_training_dynamics'] = {
    'dawn': {
        'params_M': round(model_info.get('total_M', 0), 2),  # Table 1과 동일 소스
        ...
    }
}
```

#### 데이터 소스
- `self.results['model_info']` ← `analyze_model_info()`
- `self.results['performance']` ← `analyze_performance()`

---

### Table 2: Neuron Utilization

**목적**: 각 Pool별 뉴런 활용 현황 (Active/Dead)

#### 관련 파일
```
scripts/analysis/analyze_all.py
└── _generate_tables() [line 1834-1860]
    └── neuron_utilization.csv / neuron_utilization.tex
```

#### 데이터 항목
| Column | Description |
|--------|-------------|
| pool | F-Q, F-K, F-V, R-Q, R-K, R-V, F-Know, R-Know |
| total | 해당 pool의 총 뉴런 수 |
| active | EMA > threshold인 뉴런 수 |
| dead | EMA ≈ 0인 뉴런 수 |
| active_ratio | active / total |
| gini | Gini coefficient (불균등도) |

#### 데이터 소스
- `self.results['health']['ema_distribution']` ← `analyze_health()`

---

## Text Outputs

### Generation Comparison (generation_comparison.txt)

**목적**: DAWN vs Baseline 생성 품질 비교

#### 관련 파일
```
scripts/analysis/analyze_all.py
└── _generate_comparison_samples() [line 1862-1970]
```

#### Prompt Categories
- **Factual Knowledge**: "The capital of France is", ...
- **Common Sense**: "If you drop a glass on the floor, it will", ...
- **Narrative**: "Once upon a time, in a small village,", ...
- **Technical**: "In machine learning, gradient descent is", ...

#### 데이터 소스
- 실시간 생성 (각 모델의 `model.generate()`)

---

## Data Flow Summary

| Output | Analysis Function | Data Key | Visualizer |
|--------|-------------------|----------|------------|
| **Fig 3** | `routing.analyze_qk_usage()` | `routing.qk_usage` | `qk_specialization.py` |
| **Fig 4** | `NeuronFeatureAnalyzer.run_full_analysis()` | `neuron_features` | `pos_neurons.py` |
| **Fig 5** | `behavioral.analyze_factual_neurons()` | `factual` | `factual_heatmap.py` |
| **Fig 6** | (파일 파싱) | `training_log.txt` | `training_dynamics.py` |
| **Fig 7** | `routing.analyze_layer_contribution()` | `routing.layer_contribution` | `layer_contribution.py` |
| **Table 1** | `analyze_model_info()` + `analyze_performance()` | `model_info`, `performance` | CSV/LaTeX |
| **Table 2** | `analyze_health()` | `health.ema_distribution` | CSV/LaTeX |
| **Text** | `_generate_comparison_samples()` | 실시간 생성 | TXT |

---

## Paper Results JSON Output

Paper mode 실행 시 `paper/paper_results.json`에 모든 수치 데이터가 저장됩니다.

### 출력 구조
```json
{
  "table1_model_stats": {
    "parameters_M": 24.5,
    "flops_G": 1.23,
    "perplexity": 45.67,
    "accuracy": 32.1,
    "tokens_per_sec": 15000
  },
  "table2_neuron_util": {
    "F-Q": {"total": 2048, "active": 1856, "dead": 192, "active_ratio": 0.906, "gini": 0.234}
  },
  "fig3_qk_specialization": {
    "feature_qk": {
      "correlation": -0.45,
      "q_specialized": 450,
      "k_specialized": 380,
      "shared": 1018,
      "inactive": 200,
      "q_counts": [...],
      "k_counts": [...],
      "q_ratio": [...],
      "q_ratio_active": [...]
    }
  },
  "fig4_pos_specialization": {
    "pos_neuron_counts": {"NOUN": 45, "VERB": 38, ...},
    "top_specialized": [{"neuron_id": 123, "pos": "NOUN", "concentration": 92.5}, ...]
  },
  "fig5_factual": {
    "common_neurons_100": [12, 45, 78],
    "common_neurons_80": [12, 45, 78, 123],
    "contrastive_top50": [...],
    "per_target": {...}
  },
  "fig7_layer_contribution": {
    "per_layer": {"L0": {"attention_ratio": 65.2, "knowledge_ratio": 34.8}, ...},
    "summary": {"attention_ratio_mean": 62.3, "knowledge_ratio_mean": 37.7}
  }
}
```

---

## Review Checklist (Updated 2025-01-21)

| Item | Logic | Status | Notes |
|------|-------|--------|-------|
| Fig 3 Q/K correlation | `np.corrcoef(q_counts, k_counts)` | ✅ Verified | Uses top-k weights correctly |
| Fig 3 Specialization | ratio-based (70/30) | ✅ Fixed | `q_ratio > 0.7` (commit e4e1246) |
| Fig 4 POS concentration | `count[pos] / total * 100` | ✅ Verified | 80% threshold OK |
| Fig 4 Layer aggregation | `any()` (union) | ✅ Fixed | Changed from majority vote |
| Fig 4 Mask source | `get_mask()` → `weight > 0` | ✅ Verified | Fallback works with top-k |
| Fig 5 Factual neurons | shared pool, set dedup | ✅ Verified | Correct DAWN semantics |
| Fig 5 Contrastive | `target_freq - baseline_freq` | ✅ Verified | |
| Fig 6 Training dynamics | log 파일 파싱 | ⚠️ Check | 두 모델 조건 동일 확인 필요 |
| Fig 7 Layer contribution | `attn_norm / (attn + know)` | ✅ Verified | Model stores norms |
| Table 1 Model stats | model_info + performance | ✅ OK | |
| Table 2 Neuron util | health.ema_distribution | ✅ OK | |
| paper_results.json | `_generate_paper_results_json()` | ✅ Added | commit e4e1246 |

### Completed ✅
1. **Fig 3**: Specialization threshold를 ratio 기반으로 수정 완료 (commit e4e1246)
   - `q_ratio = q_count / (q_count + k_count)`
   - Q-specialized: `q_ratio > 0.7`
   - K-specialized: `q_ratio < 0.3`
   - Inactive: 하위 10% usage

2. **paper_results.json**: 모든 수치 데이터 JSON 출력 추가 (commit e4e1246)

3. **Fig 4**: 레이어 합산 방식을 union으로 수정 완료
   - 변경 전: `combined_mask = all_masks.mean(axis=0) > 0.5` (majority vote)
   - 변경 후: `combined_mask = all_masks.any(axis=0)` (union)
   - 이유: DAWN shared pool에서 뉴런이 어느 레이어에서든 선택되면 해당 토큰 처리에 관여

### Action Items (남은 작업)
1. **Fig 6**: DAWN vs Vanilla 비교 조건 동일 확인

### Key Verification Points
- [x] Top-k sparsity: `weight > 0` correctly identifies selected neurons
- [x] WEIGHT_KEY_MAP: `fqk_q` → `fqk_weights_Q` (top-k applied)
- [x] Shared pool: neuron index = identity across layers
- [x] Output norms: model saves `attn_out_norm`, `know_out_norm`
- [x] Fig 4 레이어 합산: union 방식으로 수정 완료

---

## 실행 방법

```bash
# Paper 모드로 전체 분석 + Figure/Table 생성
python scripts/analysis/analyze_all.py \
    --checkpoint /path/to/dawn/checkpoint \
    --val_data /path/to/val_data.pt \
    --output /path/to/output \
    --compare_checkpoint /path/to/vanilla/checkpoint \
    --only paper

# 개별 Figure 생성
python scripts/analysis/paper_figures.py \
    --checkpoint /path/to/checkpoint \
    --val_data /path/to/val_data.pt \
    --output /path/to/figures \
    --figures 3,4,5,6,7
```

---

*Last updated: 2025-01-21*
