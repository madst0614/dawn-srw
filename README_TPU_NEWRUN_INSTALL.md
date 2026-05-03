# DAWN/Transformer 400M TPU init-newrun package

이 패키지는 기존 checkpoint를 읽어서(params only 기본값) 새 checkpoint_dir/run_* 폴더에 새로 저장하는 TPU pod 학습용 파일 묶음입니다.
원본 `scripts/train_jax.py`는 건드리지 않습니다.

## 들어있는 파일

```text
scripts/train_jax_init_newrun.py
scripts/setup_and_run_tpu_pod_init_newrun.sh
scripts/launch_tpu_pod_init_newrun.sh
configs/train_config_baseline_tpu_400M_c4_40B_v4_32_stable.yaml
configs/train_config_spatial_r1_v3.9.4_400M_c4_40B_v4_32_stable.yaml
downstream_tools/...
```

## repo에 넣는 법

repo 루트에서 그대로 풀면 됩니다.

```bash
cd ~/dawn-spatial
unzip -o /path/to/dawn_tpu_newrun_all_files.zip -d .
chmod +x scripts/setup_and_run_tpu_pod_init_newrun.sh
chmod +x scripts/launch_tpu_pod_init_newrun.sh
chmod +x downstream_tools/run_downstream_suite.sh
```

중요: `launch_tpu_pod_init_newrun.sh`는 worker들에서 GitHub branch를 clone/pull한 뒤 실행합니다. 따라서 로컬에만 파일을 두면 worker에 반영되지 않습니다. launch를 쓰려면 반드시 commit/push 하세요.

```bash
git add scripts/train_jax_init_newrun.py \
        scripts/setup_and_run_tpu_pod_init_newrun.sh \
        scripts/launch_tpu_pod_init_newrun.sh \
        configs/train_config_baseline_tpu_400M_c4_40B_v4_32_stable.yaml \
        configs/train_config_spatial_r1_v3.9.4_400M_c4_40B_v4_32_stable.yaml \
        downstream_tools/

git commit -m "add init-newrun TPU training scripts and stable 400M configs"
git push origin <YOUR_BRANCH>
```

## 실행 예시

### Transformer baseline: 기존 checkpoint에서 params만 읽고 새 run 생성

```bash
bash scripts/launch_tpu_pod_init_newrun.sh \
  --tpu dawn-400m-v4-32 \
  --zone us-central2-b \
  --project dawn-486218 \
  --branch <YOUR_BRANCH> \
  --config configs/train_config_baseline_tpu_400M_c4_40B_v4_32_stable.yaml \
  --init-from gs://dawn-tpu-data-c4/checkpoints/<BASELINE_EXISTING_RUN_OR_CKPT>
```

### DAWN v3.9.4: 기존 checkpoint에서 params만 읽고 새 run 생성

```bash
bash scripts/launch_tpu_pod_init_newrun.sh \
  --tpu dawn-400m-v4-32 \
  --zone us-central2-b \
  --project dawn-486218 \
  --branch <YOUR_BRANCH> \
  --config configs/train_config_spatial_r1_v3.9.4_400M_c4_40B_v4_32_stable.yaml \
  --init-from gs://dawn-tpu-data-c4/checkpoints/<DAWN394_EXISTING_RUN_OR_CKPT>
```

`--init-from`은 run folder나 `.flax` 파일 둘 다 가능합니다.

```text
gs://.../run_xxx
gs://.../run_xxx/checkpoint_step50000.flax
gs://.../run_xxx/best_model.flax
```

기본값은 params만 로드하고 optimizer/step은 새로 시작합니다.
기존 optimizer까지 이어가려면 아래 옵션을 붙입니다.

```bash
--load-opt-state --preserve-step
```

## 기존 checkpoint를 안 건드리는 방식

- `--init-from`: 읽기 전용 source
- `checkpoint_dir` in config: 새 저장 위치

현재 stable config의 새 저장 위치:

```text
Transformer:
gs://dawn-tpu-data-c4/checkpoints/baseline_400M_c4_40B_v4_32_stable

DAWN v3.9.4:
gs://dawn-tpu-data-c4/checkpoints/dawn_spatial_r1v3.9.4_400M_c4_40B_v4_32_stable
```

## 현재 안정화 설정

```yaml
val_interval: 2000
checkpoint_interval: 5000
log_interval: 100
```

DAWN config에는 추가로:

```yaml
log_analysis_multiplier: 50
heavy_geometry_multiplier: 100
```
