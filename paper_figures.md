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
│   └── analyze_qk_overlap() [line 430-577]
│
├── paper_figures.py
│   └── generate_figure3() [line 401-427]
│
└── visualizers/qk_specialization.py
    └── plot_qk_specialization()
```

#### 계산 로직
```python
# routing.py:500-550

# 1. 각 뉴런별 Q/K 선택 횟수 카운트
selected_q = (w_q > 0).float().sum(dim=[0, 1])  # weight > 0 = 선택됨
selected_k = (w_k > 0).float().sum(dim=[0, 1])

# 2. 모든 배치/레이어 합산
for layer in routing:
    total_q += selected_q
    total_k += selected_k

# 3. Pearson correlation 계산
corr = np.corrcoef(total_q, total_k)[0, 1]  # 예: r = -0.75
```

#### 데이터 소스
- `self.results['routing']['qk_overlap']`
- 측정 데이터: 25 batches × 16 samples × 512 tokens × 12 layers

#### 출력
- (a) Q vs K scatter plot + correlation coefficient
- (b) Specialization 분포 (Q-only, K-only, Shared, Inactive)

---

### Fig 4: POS Neuron Specialization

**목적**: 특정 POS(품사)에 특화된 뉴런이 존재함을 보여줌

#### 관련 파일
```
scripts/analysis/
├── pos_neuron.py
│   ├── TokenCombinationAnalyzer.analyze_dataset()
│   └── NeuronFeatureAnalyzer
│       ├── build_neuron_profiles() [line 3700+]
│       ├── detect_specialized_neurons() [line 3830+]
│       └── run_full_analysis()
│
├── paper_figures.py
│   └── generate_figure4() [line 428-472]
│
└── visualizers/pos_neurons.py
    └── plot_pos_specialization_from_features() [line 373+]
```

#### 계산 로직
```python
# pos_neuron.py:3838, 3867

# Concentration (절대적 비율) - Fig 4에서 사용
top_pos_pct = (activations_for_top_POS / total_activations) * 100

# Specialized neuron 판정
if top_pos_pct >= 80:  # threshold = 0.8
    specialized['pos'].append(neuron)
```

**참고: Selectivity (상대적 선호도) - 내부 분석용**
```python
# pos_neuron.py:801-818
selectivity[pos, neuron] = mean_weight[pos, neuron] / neuron_avg[neuron]
# selectivity > 2.0 = "Specialist"
```

#### 데이터 소스
- UD Treebank (English EWT)로 POS 태깅
- `NeuronFeatureAnalyzer.run_full_analysis()` 결과

#### 출력
- (a) Top 20 POS-specialized neurons (concentration ≥80%)
- (b) Number of specialized neurons per POS category

---

### Fig 5: Semantic Coherence (Factual Heatmap)

**목적**: 관련된 사실(예: 수도 이름들)이 유사한 뉴런 패턴을 공유함을 보여줌

#### 관련 파일
```
scripts/analysis/
├── behavioral.py
│   └── analyze_factual_neurons() [line 518-770]
│
├── paper_figures.py
│   └── generate_figure5() [line 474-527]
│
└── visualizers/factual_heatmap.py
    └── plot_factual_heatmap() [line 31+]
```

#### 계산 로직
```python
# behavioral.py:643-686

# 1. Target 토큰 생성까지 반복
while successful_runs < min_target_count and total_runs < max_runs:
    for step in range(max_tokens_per_run):
        # 2. 활성 뉴런 추출 (모든 레이어에서 - SHARED POOL)
        step_neurons = set()
        for layer in routing:
            active = (weight > 0.01).nonzero()
            step_neurons.update(active)  # 같은 뉴런이면 중복 제거

        # 3. Target 발견 시 뉴런 카운트
        if target in generated_token:
            for n in step_neurons:
                target_neuron_counts[n] += 1

# 4. Common neurons 계산
target_freq = {n: count / successful_runs for n, count in counts.items()}
common_neurons_100 = [n for n, f in target_freq.items() if f >= 1.0]  # 100%
common_neurons_80 = [n for n, f in target_freq.items() if f >= 0.8]   # 80%+

# 5. Contrastive score (target-specific 뉴런 찾기)
contrastive_score = target_freq - baseline_freq
```

#### 핵심 설계 (DAWN 특성)
```
DAWN은 Shared Neuron Pool 사용:
- Layer 0의 neuron 42 = Layer 5의 neuron 42 = 같은 뉴런
- 따라서 neuron index만 저장, set()으로 중복 제거가 올바른 로직
```

#### 데이터 소스
- `self.results['factual']`
- Prompts: "The capital of France is" → "Paris" 등

#### 출력
- Heatmap: Target × Neuron (activation frequency)
- Neurons grouped by semantic category

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
# paper_figures.py:560-600

# training_log.txt 파싱
# 형식: "Step 1000: loss=3.5432, val_loss=3.6789, ..."
for line in log_file:
    match = re.search(r'Step (\d+).*val_loss=([0-9.]+)', line)
    if match:
        steps.append(int(match.group(1)))
        losses.append(float(match.group(2)))
```

#### 데이터 소스
- `{checkpoint_path}/training_log.txt`
- DAWN checkpoint + Vanilla checkpoint

#### 출력
- Validation loss curves (DAWN vs Vanilla)
- X축: Training steps, Y축: Validation loss

#### 검토 필요
- 두 모델이 같은 조건(데이터, steps, evaluation)인지 확인

---

### Fig 7: Layer-wise Contribution

**목적**: 각 레이어에서 Attention vs Knowledge circuit의 기여도 비율

#### 관련 파일
```
scripts/analysis/
├── routing.py
│   └── analyze_layer_contribution() [line 687-764]
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
# routing.py:725-752

# 1. 모델이 저장한 output norm 가져오기
norms = layer.get_output_norms()
# → {'attn_out_norm': 0.42, 'know_out_norm': 0.58}

# 2. Attention ratio 계산
attention_ratio = attn_norm / (attn_norm + know_norm) * 100

# "Attention > 50%" = attention_ratio > 50
# = Attention circuit이 Knowledge circuit보다 기여도가 큼
```

```python
# utils.py:918-923 - Output norm 추출
def get_output_norms(self) -> Dict[str, float]:
    norms = {}
    if 'attn_out_norm' in self.raw:
        norms['attn_out_norm'] = self.raw['attn_out_norm']
    if 'know_out_norm' in self.raw:
        norms['know_out_norm'] = self.raw['know_out_norm']
    return norms
```

#### 데이터 소스
- `self.results['routing']['layer_contribution']`
- 모델이 forward pass에서 저장하는 `attn_out_norm`, `know_out_norm`

#### 출력
- Stacked bar chart: Attention vs Knowledge per layer
- Layer 0~11의 기여도 비율

---

## Tables

### Table 1: Model Statistics

**목적**: DAWN vs Vanilla 모델 비교 (파라미터, 성능)

#### 관련 파일
```
scripts/analysis/analyze_all.py
└── _generate_tables() [line 1717-1833]
    └── model_stats.csv / model_stats.tex
```

#### 데이터 항목
| Metric | Source |
|--------|--------|
| parameters_M | `model_info['total_M']` |
| flops_G | `model_info['flops_G']` |
| perplexity | `performance['validation']['perplexity']` |
| accuracy | `performance['validation']['accuracy']` |
| tokens_per_sec | `performance['speed']['tokens_per_sec']` |

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
| **Fig 3** | `routing.analyze_qk_overlap()` | `routing.qk_overlap` | `qk_specialization.py` |
| **Fig 4** | `NeuronFeatureAnalyzer.run_full_analysis()` | `neuron_features` | `pos_neurons.py` |
| **Fig 5** | `behavioral.analyze_factual_neurons()` | `factual` | `factual_heatmap.py` |
| **Fig 6** | (파일 파싱) | `training_log.txt` | `training_dynamics.py` |
| **Fig 7** | `routing.analyze_layer_contribution()` | `routing.layer_contribution` | `layer_contribution.py` |
| **Table 1** | `analyze_model_info()` + `analyze_performance()` | `model_info`, `performance` | CSV/LaTeX |
| **Table 2** | `analyze_health()` | `health.ema_distribution` | CSV/LaTeX |
| **Text** | `_generate_comparison_samples()` | 실시간 생성 | TXT |

---

## Review Checklist

| Item | Logic | Status |
|------|-------|--------|
| Fig 3 Q/K correlation | `np.corrcoef(q_counts, k_counts)` | ✅ OK |
| Fig 4 POS concentration | `count[pos] / total * 100` | ✅ OK |
| Fig 5 Factual neurons | shared pool, set dedup | ✅ OK |
| Fig 6 Training dynamics | log 파일 파싱 | ⚠️ 조건 동일 확인 필요 |
| Fig 7 Layer contribution | `attn_norm / (attn + know)` | ✅ OK |
| Table 1 Model stats | model_info + performance | ✅ OK |
| Table 2 Neuron util | health.ema_distribution | ✅ OK |

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
