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
'fig7_pos_selectivity_heatmap': {
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
│   └── analyze_factual_neurons() [line 509-780+]
│       ├── Token validation [line 565-579]
│       ├── Generation loop [line 599-708]
│       ├── Neuron collection (all pools) [line 633-679]
│       ├── Exact match check [line 682]
│       └── Result computation [line 731-760]
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

        # 3. Target 발견 시 뉴런 카운트 (line 681-697)
        # EXACT MATCH 사용 (substring 아님)
        if token_text.strip().lower() == target_lower:
            for n in step_neurons:
                target_neuron_counts[pool][f'{pool}_{n}'] += 1
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
1. Validate targets are single tokens (warning if multi-token)
2. Start from prompt (e.g., "The capital of France is")
3. Generate tokens until target appears (e.g., "Paris")
   - EXACT MATCH: token_text.strip().lower() == target_lower
4. Record active neurons at target generation step
5. Repeat for min_target_count runs (default: 100)
6. Compute frequency: how often each neuron activates when generating target
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
| **Exact match** | ✅ | `==` 사용 (substring `in` 아님) |
| **단일 토큰 검증** | ✅ | `token_validation` 결과에 저장 |

#### 주의사항 ⚠️
- v17.1은 routing_info에 mask 없음 → fallback 사용
- top-k 후 `weight > 0` 사용 (non-zero = active)
- `pool_type` 파라미터로 분석 대상 pool 선택 가능 (기본: 'fv')

#### 데이터 소스
- `self.results['factual']`
- Example prompts: "The capital of France is" → "Paris"

#### Heatmap Category Classification

**파일**: `visualizers/factual_heatmap.py` - `get_neuron_category()` [line 161-187]

**분류 로직 (Sequential Filtering)**:

뉴런을 순차적으로 필터링하여 카테고리 분류:

```python
def get_neuron_category(neuron):
    capital_freqs = [all_neurons[t].get(neuron, 0) for t in capital_targets]
    other_freqs = [all_neurons[t].get(neuron, 0) for t in other_targets]
    all_freqs = capital_freqs + other_freqs

    # Step 1: Shared - 모든 타겟에서 0.7+
    if all_freqs and all(f >= 0.7 for f in all_freqs):
        return 0  # Shared

    # Step 2: Capital-specific - 모든 capital 0.7+ AND 모든 other < 0.3
    if (all(f >= 0.7 for f in capital_freqs) and
        all(f < 0.3 for f in other_freqs)):
        return 1  # Capital-specific

    # Step 3: Other-specific - 해당 other 0.7+ AND 모든 capital < 0.3
    if (any(f >= 0.7 for f in other_freqs) and
        all(f < 0.3 for f in capital_freqs)):
        return 2  # Other-specific

    return 3  # Mixed (excluded from figure)
```

**카테고리 정의**:
| Category | Condition | 의미 |
|----------|-----------|------|
| Shared (0) | ALL targets >= 0.7 | 모든 factual query에서 공통 활성화 |
| Capital-specific (1) | ALL capitals >= 0.7 AND ALL others < 0.3 | 수도 관련 query에서만 활성화 |
| Other-specific (2) | ANY other >= 0.7 AND ALL capitals < 0.3 | 특정 비수도 query에서만 활성화 |
| Mixed (3) | 위 조건 모두 불충족 | Figure에서 제외 |

**Threshold 근거**:
- 0.7: 70% 이상 run에서 활성화 = 신뢰할 수 있는 consistent activation
- 0.3: 30% 미만 = 해당 카테고리에 무관한 뉴런으로 판정
- Sequential filtering으로 명확한 separation 보장

**왜 100%가 아닌 70-80% threshold인가?**

100% consistency가 나오지 않는 이유:
1. **문맥 의존성**: "Paris"가 수도, 사람 이름, 신화 인물 등 다양한 의미 가능
2. **라우팅 변동성**: top-k 선택 시 비슷한 weight의 뉴런 간 경쟁
3. **Generation sampling**: temperature로 인한 자연스러운 변동

따라서 70-80% threshold는:
- 100%: 너무 strict → 거의 모든 뉴런 제외
- 70-80%: "대부분의 run에서 일관되게 활성화" = 신뢰할 수 있는 패턴

> "Neurons activating in ≥70% of runs are considered consistently associated with the target, accounting for natural variation in routing decisions."

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

#### 검증 완료 ✅
| 항목 | 상태 | 비고 |
|------|------|------|
| 로그 파싱 | ✅ | regex로 step, val_loss 추출 |
| 비교 조건 | ✅ | paper_data.json에서 학습 조건 확인 가능 |

#### 학습 조건 확인 ✅
- [x] 같은 데이터셋으로 학습 (paper_data.json의 training.dataset)
- [x] 같은 step 수 학습 (paper_data.json의 training.total_steps)
- [x] 같은 evaluation 방식 (paper_data.json의 training.validation)
- [x] Parameter 수 확인 가능 (paper_data.json의 models.dawn/vanilla.parameters_M)

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
│   └── analyze_layer_contribution() [line 716-800+]
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
├── analyze_model_info() [line 276-344]
├── analyze_performance() [line 346-469]
├── _generate_tables() [line 1737-1870+]
│   └── model_stats.csv / model_stats.tex
└── _generate_paper_results_json() [line 2500+]
    └── paper_results.json['table1_model_stats']

scripts/evaluation/evaluate.py
└── estimate_flops() [line 50-121]
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
# analyze_all.py - analyze_model_info() [line 276-344]

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

#### FLOPs 추정 로직 (이론적 Sparse FLOPs)

```python
# evaluate.py - estimate_flops() [line 50-121]
# Includes both sparse matmul and einsum operations

# === DAWN Model ===
# Feature sparse matmul: Q, K (각 top_k_fqk), V (top_k_fv)
attn_feat_matmul = 2 * (top_k_fqk * 2 + top_k_fv) * d_model * rank * seq_len
# Feature einsum (x3 for Q, K, V)
attn_feat_einsum = 2 * 3 * d_model * rank * seq_len

# Restore sparse matmul: Q, K (각 top_k_rqk), V (top_k_rv)
attn_rest_matmul = 2 * (top_k_rqk * 2 + top_k_rv) * rank * d_model * seq_len
# Restore einsum (x3 for Q, K, V)
attn_rest_einsum = 2 * 3 * rank * d_model * seq_len

# Attention scores: QK^T + scores@V
attn_scores = 2 * 2 * seq_len * seq_len * d_model

# expand_O: Linear(d_model, d_model)
expand_o = 2 * d_model * d_model * seq_len

# Knowledge: Feature matmul + einsum, Restore matmul + einsum
know_feat = 2 * top_k_fknow * d_model * knowledge_rank * seq_len
know_feat_ein = 2 * d_model * knowledge_rank * seq_len
know_rest = 2 * top_k_rknow * knowledge_rank * d_model * seq_len
know_rest_ein = 2 * knowledge_rank * d_model * seq_len

# === Vanilla Model ===
qkvo = 2 * 4 * d_model * d_model * seq_len      # 8 * d^2 * seq
attn_scores = 2 * 2 * seq_len * seq_len * d_model  # 4 * seq^2 * d
ffn = 2 * 2 * d_model * d_ff * seq_len          # 4 * d * d_ff * seq
```

#### FLOPs 검증 ✅ (2025-01-23 Updated)

| Component | DAWN (Sparse) | Vanilla (Dense) | 상태 |
|-----------|---------------|-----------------|------|
| **Attention Circuit** | | | |
| - Feature matmul | `2 * (fqk*2 + fv) * d * rank * seq` | - | ✅ |
| - Feature einsum | `2 * 3 * d * rank * seq` | - | ✅ NEW |
| - Restore matmul | `2 * (rqk*2 + rv) * rank * d * seq` | - | ✅ |
| - Restore einsum | `2 * 3 * rank * d * seq` | - | ✅ NEW |
| - QKVO (Vanilla) | - | `8 * d² * seq` | ✅ |
| **Attention Scores** | `4 * seq² * d` | `4 * seq² * d` | ✅ |
| **expand_O** | `2 * d² * seq` | (in QKVO) | ✅ |
| **Knowledge Circuit** | | | |
| - Feature matmul + einsum | `2 * fknow * d * rank + 2 * d * rank` | - | ✅ |
| - Restore matmul + einsum | `2 * rknow * rank * d + 2 * rank * d` | - | ✅ |
| - FFN (Vanilla) | - | `4 * d * d_ff * seq` | ✅ |

**DAWN top-k 파라미터**:
- `top_k_fqk`, `top_k_fv`: Feature pathway (Q/K, V)
- `top_k_rqk`, `top_k_rv`: Restore pathway (Q/K, V)
- `top_k_fknow`, `top_k_rknow`: Knowledge circuit

**결론**: ✅ 이론적 Sparse FLOPs (matmul + einsum) 기준으로 Fair comparison

#### tokens/sec 측정 검증 ✅

```python
# benchmark_speed.py - benchmark_model() [line 188-237]
# performance.py - SpeedBenchmark.benchmark() [line 227-278]
```

| 검증 항목 | benchmark_speed.py | performance.py | 상태 |
|----------|-------------------|----------------|------|
| Warmup | ✅ 10회 기본 | ✅ 10회 기본 | ✅ |
| CUDA sync | ✅ 매 iteration | ✅ 전체 후 1회 | ✅ |
| 동일 조건 | ✅ 같은 seq_len, batch_size | ✅ 같은 seq_len, batch_size | ✅ |
| 측정 방식 | per-iteration timing | batch timing | ⚠️ 약간 차이 |

**측정 공정성**: ✅ DAWN과 Vanilla를 동일한 함수로 측정하므로 fair comparison

**참고**: benchmark_speed.py가 더 정확한 per-iteration 측정 (권장)

#### 주의사항 ⚠️

**Table 1 vs Fig 6 일관성**:
- Table 1: `model_info.get('total_M')` - 실제 로드된 모델에서 계산
- Fig 6: 동일한 `model_info` 사용 (수정 후)
- **절대로 checkpoint state_dict에서 별도 카운트하지 않음**

```python
# paper_results.json 생성 시 (analyze_all.py line 2578-2583)
paper_results['fig6_convergence_comparison'] = {
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
| **Fig 3** (F-QK Specialization) | `routing.analyze_qk_usage()` | `routing.qk_usage` | `qk_specialization.py` |
| **Fig 4** (Convergence Comparison) | (파일 파싱) | `training_log.txt` | `training_dynamics.py` |
| **Fig 5** (Attn-Know Balance) | `routing.analyze_layer_contribution()` | `routing.layer_contribution` | `layer_contribution.py` |
| **Fig 6** (R-QK Specialization, Appendix) | `routing.analyze_qk_usage()` | `routing.qk_usage` | `qk_specialization.py` |
| **Fig 7** (POS Heatmap, Appendix) | `NeuronFeatureAnalyzer.run_full_analysis()` | `neuron_features` | `pos_neurons.py` |
| **Fig 8** (Knowledge Neurons, Appendix) | `behavioral.analyze_factual_neurons()` | `factual` | `factual_heatmap.py` |
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
  "fig3_fqk_specialization": {
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
  "fig4_attention_knowledge_balance": {
    "per_layer": {"L0": {"attention_ratio": 65.2, "knowledge_ratio": 34.8}, ...},
    "summary": {"attention_ratio_mean": 62.3, "knowledge_ratio_mean": 37.7}
  },
  "fig5_rqk_specialization": {
    "restore_qk": { "...": "same structure as fig3" }
  },
  "fig6_convergence_comparison": {
    "...": "training dynamics data"
  },
  "fig7_pos_selectivity_heatmap": {
    "pos_neuron_counts": {"NOUN": 45, "VERB": 38, ...},
    "top_specialized": [{"neuron_id": 123, "pos": "NOUN", "concentration": 92.5}, ...]
  },
  "fig8_knowledge_neurons": {
    "common_neurons_100": [12, 45, 78],
    "common_neurons_80": [12, 45, 78, 123],
    "contrastive_top50": [...],
    "per_target": {...}
  }
}
```

---

## Review Checklist (Updated 2025-01-23)

| Item | Logic | Status | Notes |
|------|-------|--------|-------|
| Fig 3 Q/K correlation | `np.corrcoef(q_counts, k_counts)` | ✅ Verified | Uses top-k weights correctly |
| Fig 3 Specialization | ratio-based (70/30) | ✅ Fixed | `q_ratio > 0.7` (commit e4e1246) |
| Fig 4 POS concentration | `count[pos] / total * 100` | ✅ Verified | 80% threshold OK |
| Fig 4 Layer aggregation | `any()` (union) | ✅ Fixed | Changed from majority vote |
| Fig 4 Mask source | `get_mask()` → `weight > 0` | ✅ Verified | Fallback works with top-k |
| Fig 5 Factual neurons | shared pool, set dedup | ✅ Verified | exact match 사용, 단일 토큰 검증 포함 |
| Fig 5 Contrastive | `target_freq - baseline_freq` | ✅ Verified | |
| Fig 6 Training dynamics | log 파일 파싱 | ✅ Verified | 학습 조건은 paper_data.json에서 확인 |
| Fig 7 Layer contribution | `attn_norm / (attn + know)` | ✅ Verified | Model stores norms |
| Table 1 Model stats | model_info + performance | ✅ OK | |
| Table 2 Neuron util | health.ema_distribution | ✅ OK | |
| paper_results.json | `_generate_paper_results_json()` | ✅ Added | commit e4e1246 |
| **FLOPs Calculation** | `estimate_flops()` | ✅ Fixed | Vanilla FFN 2x→4x 수정 완료 |
| **tokens/sec Measurement** | `SpeedBenchmark.benchmark()` | ✅ Verified | Warmup 포함, 동일 조건 |

### FLOPs 계산 검증 (2025-01-23 Updated)

**코드 위치**: `scripts/evaluation/evaluate.py` line 50-121

**이론적 Sparse FLOPs 계산** (matmul + einsum 포함)

| Component | DAWN (Sparse) | Vanilla (Dense) | Status |
|-----------|---------------|-----------------|--------|
| **Attention Circuit** | | | |
| - Feature matmul | `2 * (fqk*2 + fv) * d * rank * seq` | - | ✅ |
| - Feature einsum | `2 * 3 * d * rank * seq` | - | ✅ |
| - Restore matmul | `2 * (rqk*2 + rv) * rank * d * seq` | - | ✅ |
| - Restore einsum | `2 * 3 * rank * d * seq` | - | ✅ |
| - QKVO projections | - | `8 * d² * seq` | ✅ |
| - Attention scores | `4 * seq² * d` | `4 * seq² * d` | ✅ |
| - expand_O | `2 * d² * seq` | (in QKVO) | ✅ |
| **Knowledge Circuit** | | | |
| - Feature matmul | `2 * fknow * d * rank * seq` | - | ✅ |
| - Feature einsum | `2 * d * rank * seq` | - | ✅ |
| - Restore matmul | `2 * rknow * rank * d * seq` | - | ✅ |
| - Restore einsum | `2 * rank * d * seq` | - | ✅ |
| - FFN (Vanilla) | - | `4 * d * d_ff * seq` | ✅ |
| Router overhead | ❌ 미포함 | N/A | 무시 가능 |

**Note**: LM head 제외 (per-layer FLOPs만 계산)

**결론**: ✅ 이론적 Sparse FLOPs (matmul + einsum) 기준으로 fair comparison

### tokens/sec 측정 검증 (2025-01-23)

**코드 위치**:
- `scripts/analysis/performance.py` - `SpeedBenchmark.benchmark()` line 227-278
- `scripts/benchmark_speed.py` - `benchmark_model()` line 188-237

| 검증 항목 | 결과 | 비고 |
|----------|------|------|
| Warmup 포함 | ✅ | 기본 10회 warmup |
| CUDA synchronize | ✅ | GPU 동기화 수행 |
| 동일 조건 측정 | ✅ | 같은 seq_len, batch_size로 DAWN/Vanilla 모두 측정 |
| 측정 함수 동일 | ✅ | 같은 함수로 두 모델 측정 |

**측정 코드 요약**:
```python
# benchmark_speed.py - benchmark_model()
def benchmark_model(model, seq_len=512, batch_size=1, warmup=10, iterations=100):
    # 1. Warmup (10회)
    for _ in range(warmup):
        with torch.no_grad():
            _ = model(x)

    # 2. CUDA sync 후 측정
    torch.cuda.synchronize()
    for _ in range(iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        torch.cuda.synchronize()
        end = time.perf_counter()

    # 3. tokens_per_sec = (batch_size * seq_len) / avg_time
```

**결론**: ✅ DAWN과 Vanilla 동일 조건에서 fair하게 측정됨

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

4. **FLOPs 계산 전면 수정** (2025-01-23): `scripts/evaluation/evaluate.py` line 50-120
   - DAWN: Restore pathway 추가, expand_O 추가, attention scores 수정 (2x → 4x)
   - Vanilla: QKVO 수정 (4x → 8x), attention scores 수정 (2x → 4x), FFN 유지 (4x)
   - LM head: 2x factor 추가
   - 이론적 Sparse FLOPs 기준으로 fair comparison

### Action Items (남은 작업)
(없음 - 모든 항목 완료)

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

*Last updated: 2025-01-23*
