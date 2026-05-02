# Model History

This document preserves model history that should not live inside active model
source files. Active source files describe current behavior only.

## dawn_srw

- Active implementation: `models/dawn_srw.py`
- Compatibility implementation path: `models/dawn_spatial_v4155.py`
- Training entry point: `scripts/train_jax.py`
- `dawn_srw` is the official model name. The former
  `spatial-r1-v4.1.5.5` key is retained only as a legacy config/checkpoint
  alias.
- Terminology/API refactor:
  - DAWN-SRW is the active model name; the implementation is described in
    terms of SRW neurons, signature embeddings, RW operators, Attention Layer,
    RST Layer, model decisions, and scan offsets.
  - Pool params were renamed:
    `qk_* -> attn_qk_*`, `v_* -> attn_v_*`, `know_* -> rst_*`.
  - Router params were renamed:
    `proj_know -> proj_rst`, `tau_know -> tau_rst`,
    `scan_bias_attn -> raw_scan_offset_attn`, and
    `scan_bias_know -> raw_scan_offset_rst`.
  - Forward helpers were renamed:
    `AttentionCircuit -> AttentionLayer`,
    `_know_forward -> _rst_forward`, and
    `_know_forward_inference -> _rst_forward_inference`.
  - Public metrics now use primary `rst_*`, `attn_qk_*`, and `attn_v_*`
    names; temporary legacy aliases are returned for `train_jax.py` logging.
  - Legacy checkpoint migration is exposed as
    `migrate_legacy_v4155_params(params)` and preserves already-migrated
    values when old and new keys both exist.
- Functional change from v4.1.5.2: read/write vectors are used raw in SRW;
  their magnitudes are natural vector magnitudes.
- Gate and denominator are unchanged from v4.1.5.2:
  `gate = activation * intensity`, `den = max(sum(gate), 1.0)`.
- Weight decay: `attn_qk_emb`, `attn_v_emb`, `rst_emb`, and raw read/write
  vectors all receive `pool_weight_decay`; only v4.1.5.2 excludes read/write
  WD.

## spatial-r1-v4.1.5.4

- Archived implementation:
  `models/legacy/dawn_spatial_v4154_activation_den_exp.py`
- Archived configs:
  `configs/legacy/train_config_spatial_r1_v4_1_5_4_*.yaml`
- Functional change from v4.1.5.2: SRW denominator used activation mass only:
  `den = max(sum(activation), 1.0)`.
- Numerator was unchanged:
  `gate = activation * intensity`,
  `a_i = gate_i * <x, normalize(read_i)>`.
- Read/write vectors remained forward-normalized inside SRW.
- Removed from the active training registry; kept only as a historical
  experiment.

## spatial-r1-v4.1.5.3

- Archived implementation:
  `models/legacy/dawn_spatial_v4153_factorized_routing_exp.py`
- Archived config:
  `configs/legacy/train_config_spatial_r1_v4_1_5_3_40M_c4_5B_tag0_sigr64_sigw64_factorized.yaml`
- Factorized read/write signature routing experiment.
- Removed learned route tags in the archived config:
  `tag_dim=0`, `read_sig_dim=64`, `write_sig_dim=64`, `d_route=128`.
- Used `routing_type="factorized"` with separate read/write operator
  signature factors.
- Kept the v4.1 two-stage gate and gate denominator:
  `gate = activation * intensity`,
  `den = max(sum(activation * intensity), 1.0)`.

## spatial-r1-v4.1.5.2

- Active implementation: `models/dawn_spatial_v4152.py`
- Training entry point: `scripts/train_jax.py`
- Routing score: `scores = h @ emb.T`.
- Read/write vectors are used by the SRW operator and are normalized in the
  forward path.
- Gate: `activation * intensity`.
- SRW denominator: `max(sum(gate), 1.0)`.
- Neuron pool and router are shared across layers.  Per-layer parameters are
  LayerNorms and attention output projections.
- Older checkpoints with previous route split fields are migrated in
  `load_checkpoint` before Flax state restoration.

### Cleanup Notes

- Moved historical notes out of active source files.
- Removed implementation-facing references to experimental route splits from
  the active model.
- Kept checkpoint compatibility in the loader rather than carrying obsolete
  route parameters in the model.

## spatial-r1-v4.1.5

- Introduced operator-derived routing signatures in the experimental branch.
- Route vector was formed from a learned tag plus fixed projections of unit
  read/write vectors.
- Split dimensions were configured by `tag_dim`, `read_sig_dim`, and
  `write_sig_dim`.
- Denominator used activation-weighted intensity normalization.
- Goal was to test whether operation-derived route structure could recover
  dense-to-sparse differentiation while preserving separate read/write
  operators.

## spatial-r1-v4.1

- Introduced two-stage gate:
  `activation = sigmoid(sharpness * margin)`,
  `intensity = epsilon + min(max(margin - cutoff, 0), max_intensity)`,
  `gate = activation * intensity`.
- Removed dynamic tau alpha parameters.
- Added bounded scan bias:
  `scan = scan_scale * tanh(scan_bias)`.
- Relative tau remained:
  `tau = s_mean + tau_offset * s_std - scan / max(s_std, scan_std_floor)`.
- Dead threshold was raised from `1e-4` to `0.01`.
- Removed normalized-z gate computation.
- Route embedding norm was kept as a learnable degree of freedom.
- Read/write handling varied across experiments; later variants normalized
  read/write inside the SRW forward path.

## spatial-r1-v4.1.4

- Competitive denominator experiment.
- Kept v4.1.2 bounded scan-bias tau and margin intensity.
- Used unit read/write directions in SRW:
  `r_i = read_i / ||read_i||`, `w_i = write_i / ||write_i||`.
- Contribution:
  `a_i = activation_i * intensity_i * <x, r_i>`.
- Output denominator combined activation and contribution magnitude:
  `1 + sum(activation_i) + sum(sqrt(a_i^2 + epsilon^2) - epsilon)`.

## spatial-r1-v4.1.2

- Scan-bias wiring experiment.
- Trained from `models/legacy/dawn_spatial_v412_scan_bias_exp.py`.
- Added `scan_bias` without an extra L2 term.
- `scan_bias` followed the existing AdamW masks.
- Analysis and suppression helpers used:
  `scan = scan_scale * tanh(scan_bias)`,
  `tau = s_mean + tau_offset * s_std - scan / max(s_std, scan_std_floor)`.
- Routing score remained `h @ emb.T`.
- Denominator used the intensity sum in that implementation.

## spatial-r1-v4.0.6

- Gate confidence changed toward a sharper confidence term around
  `scores - tau`.
- Dynamic tau used per-token score std:
  `tau_effective = tau - alpha * (1 / (s_std + 1e-6))`.
- Harder tokens with lower score std recruited more neurons.
- Per-pool dynamic tau alpha parameters lived in `NeuronPool`.
- Dead-only penalty identified neurons whose max gate over a batch was below
  `1e-4` and pushed their mean score upward.
- Reverse and bidirectional gate dropouts from v4.0.4/v4.0.5 were removed.

## spatial-r1-v4.0.3

- Denominator changed from gate sum to positive intensity sum.
- Numerator still used a confidence-weighted intensity gate.
- The numerator/denominator asymmetry penalized borderline joiners.
- Denominator floor of `1.0` was kept to prevent tau runaway.
- Route embedding forward normalization was removed.
- Embedding rows still started from unit-norm initialization, but training
  could change their norms.
- Read/write stayed forward-normalized as a size-dominance guard.

## spatial-r1-v4.0.1

- Introduced GELU-like gate as confidence times intensity.
- Normal-CDF confidence provided a statistical interpretation from
  score-standardized z.
- Kept denominator floor at `1.0`.

## spatial-r1-v3.9.9

- Replaced sigmoid gate with GELU-like `z * sigmoid(z)` under a positive mask.
- Separated confidence and intensity gradient paths.
- Restored unbounded gate range for neuron importance differentiation.
- Active count threshold moved from sigmoid confidence to positive gate.

## spatial-r1-v3.9.8.1

- Removed `xr` from denominator.
- Motivation: large-`xr` neurons in the denominator made pruning too cheap.
- Denominator became a pure confidence/gate sum again.

## spatial-r1-v3.9.8

- Replaced binary/STE gate with continuous sigmoid gate.
- Gate became bounded in `[0, 1]`.
- Numerator and denominator shared the same gate for structural cancellation.
- Denominator included gate-weighted `xr`.
- Added learnable per-pool scale.

## spatial-r1-v3.9.7.1

- Removed a constant term from the denominator.
- Added learnable per-pool output scales:
  `know_scale`, `qk_scale`, and `v_scale`.
- Linear denominator restored loss pressure when fewer neurons were active.

## spatial-r1-v3.9.7

- Used binary gate in the forward pass with a smooth denominator.
- Numerator used hard binary selection with STE gradients.
- Denominator used a smooth sigmoid/ReLU-style count weighted by `xr`.
- Removed stop-gradient from `xr` in denominator.
- Goal was to provide self-regulating gradients to read vectors and prevent
  runaway read magnitude.

## spatial-r1-v3.9.6

- Used STE binary gate with soft denominator.
- Numerator: hard gate times `xr @ write`.
- Denominator: continuous soft gate sum.
- Removed `gate_norm_mode`.

## spatial-r1-v3.9.5

- Introduced STE binary gate:
  `gate = gate_hard + gate_soft - stop_gradient(gate_soft)`.
- `gate_hard = raw > 0`, `gate_soft = relu(raw)`.
- Supported `sqrt_active` and `active_n` normalization modes.
- Removed hard clip at `10`.
- Replaced gate concentration logging with active count logging.

## spatial-r1-v3.9.4

- Implementation lives in `models/legacy/dawn_spatial_v394_exp.py`.
- Training registry entries import the legacy module directly.
- Removed `tanh(gate_max)` output heuristic.
- Used gate-sum normalization with fixed output scale.
- Let `x @ read` naturally modulate per-token output magnitude.
- Removed learnable strength parameters.
- Kept denominator floor at `1.0`.

## spatial-r1-v3.9.1

- Changed load-balance loss from gate-based to pre-activation score-based.
- Allowed all neurons to receive load-balance gradients.
- Removed gate load-balance computation from pass 2.
- Forward-normalized read/write vectors.
- Used coefficient-of-variation style score load balancing.
- Used global `pmax` for gate strength.
- Fixed output scale at `sqrt(d_model)`.

## spatial-r1-v3.9.0

- Used linear ReLU-style gate from `exp(gate) - 1`.
- Removed score standardization from the gate.
- Used uniform-target MSE style load balancing.
- Restored attention dropout inside checkpointed attention.
- Added score std and gate concentration logging.
- Updated inference APIs to match the linear gate.

## spatial-r1-v3.8.2

- Increased routing dimension from `64` to `128`.
- Added threshold gate clamp for NaN prevention.
- Adjusted neuron counts for parameter budget.

## spatial-r1-v3.8.1

- Used relative tau as `mean + offset * std`.
- Added tiny dead-neuron gradient flow.
- Added exponential scaling and gate strength.
- Tau bias initialized at `-0.5`.

## spatial-r1-v3.8.0

- Introduced Sense-Read-Write rank-1 neuron form:
  each neuron has route embedding, read vector, and write vector.
- Output:
  `sum_i gate_i * (x @ read_i) * write_i`.
- Token state directly participates in output computation.
- Computation is matmul plus elementwise operations.

## spatial-r1-v3.7.0

- Earlier sense/direct-emit variant using gate over write vectors.
