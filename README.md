# MetaCam

계산광학 시뮬레이션/복원 실험용 레포를 표준 프로젝트 구조(`src`, `configs`, `assets`, `outputs`)로 정리했습니다.

## 폴더 구조

```text
.
├─ src/
│  └─ metacam/                # Canonical source package
│     ├─ physics/             # ASM/SASM, propagation, MetaOperator
│     ├─ data/                # .mat 로더, 증강 유틸
│     ├─ ops/                 # torch/numpy 수치 유틸
│     ├─ metrics/             # loss, correlation
│     ├─ vision/              # 시각화/이미지 처리
│     ├─ patterns/            # 타깃 패턴 인코딩
│     └─ nn/                  # 향후 neural reconstruction 스캐폴드
├─ configs/
│  ├─ paths.yaml
│  ├─ simulation/base.yaml
│  └─ training/base.yaml
├─ assets/
│  └─ data/                   # 기존 Data 내용 이동
├─ outputs/
│  ├─ checkpoints/
│  ├─ experiments/
│  └─ figures/
├─ notebooks/
│  └─ PhaseCam_Simul_...ipynb
├─ scripts/
│  └─ smoke_check.py
├─ tests/
│  └─ test_imports.py
├─ Library/                   # Legacy compatibility wrappers
├─ fieldprop/                 # Legacy compatibility wrappers
└─ metacam/                   # src-layout compatibility shim
```

## 호환성 정책

- 기존 코드의 `Library.*`, `fieldprop.*` import는 유지됩니다.
- 기존 루트 경로 참조를 깨지 않기 위해 아래 호환 링크를 유지합니다.
  - `Data -> assets/data`

## 권장 import (신규 코드)

```python
from metacam.physics import MetaOperator, asm_master_alltorch
from metacam.ops.torch_ops import torch_pad_center, normxcorr2_fft
from metacam.metrics import NPCCloss, tv_loss
```

## 빠른 검증

```bash
python scripts/smoke_check.py
```

Real-scale Adam+SASM 실행:

```bash
python scripts/run_phasecam_realscale_test.py
```

옵션 예시:

```bash
python scripts/run_phasecam_realscale_test.py --iterations 20
python scripts/run_phasecam_realscale_test.py --widthmap-file 0.6NA_random_70_1_300_1mm_mapped_width.mat
```

## 향후 NN 복원 확장 포인트

- forward optics: `src/metacam/physics/`
- trainable inverse model: `src/metacam/nn/`
- experiment config: `configs/training/*.yaml`, `configs/simulation/*.yaml`
- run artifacts: `outputs/`

## Reduced 512 Testbed

신규 reduced 테스트베드는 기존 real-scale 경로에서 실제로 쓰이던 파라미터를 기준으로 줄였습니다.

- baseline source:
  - `src/metacam/nn/phasecam_realscale.py`
  - `scripts/run_phasecam_realscale_test.py`
- baseline values:
  - simulation grid: `5713`
  - simulation pixel pitch: `350 nm`
  - camera pixel pitch: `1.85 um`
  - wavelength: `532 nm`
  - aperture width: `1.0 mm`
  - meta-to-sensor distance: `6.3 mm`
  - object-to-meta distance: `0.4 mm`
  - width-map size: `2856 x 2856`

Reduced config는 `configs/simulation/test512.yaml`에 정리되어 있습니다.

- reduced simulation grid: `512 x 512`
- camera pixel pitch: unchanged
- object support / aperture / meta-pixel count: `256`
- reduced active width: `89.6 um`
- reduced meta-to-sensor distance: `564.6 um`
- reduced object-to-meta distance: `35.8 um`
- full camera-sampled reduced sensor window: `463 x 463`

Scaling rule은 `lambda * z / D`를 거의 보존하도록 lateral dimensions와 `z`를 함께 `s = 512 / 5713`로 줄였고, discrete sampling consistency를 위해 active support와 meta-pixel count는 `256`으로 snap했습니다.

## Neural Reconstruction

추가된 구성요소:

- differentiable forward model: `src/metacam/physics/phasecam_forward.py`
- synthetic phase dataset: `src/metacam/data/synthetic_phase_dataset.py`
- direct baseline: `src/metacam/nn/baselines.py`
- physics-guided unrolled model: `src/metacam/nn/physics_unrolled.py`
- training / eval helpers: `src/metacam/nn/train_utils.py`

학습 loss:

- periodic phase loss: `mean(|exp(i phi_pred) - exp(i phi_gt)|^2)`
- measurement consistency loss: `MSE(I(phi_pred), I_meas)`
- TV regularizer

평가 지표:

- wrapped phase MAE / RMSE
- complex correlation
- intensity consistency error
- runtime per sample

## Commands

Random metasurface mask 생성:

```bash
.venv/bin/python scripts/generate_metacam_mask.py --simulation-config configs/simulation/test512.yaml
```

Baseline U-Net 학습:

```bash
.venv/bin/python scripts/train_phasecam_unet.py --config configs/training/unet512.yaml
```

Physics-guided unrolled model 학습:

```bash
.venv/bin/python scripts/train_phasecam_unrolled.py --config configs/training/unrolled512.yaml
```

빠른 smoke run:

```bash
.venv/bin/python scripts/train_phasecam_unet.py --config configs/training/unet512.yaml --quick
.venv/bin/python scripts/train_phasecam_unrolled.py --config configs/training/unrolled512.yaml --quick
```

단일 모델 평가:

```bash
.venv/bin/python scripts/eval_phasecam_nn.py \
  --config configs/training/unrolled512.yaml \
  --checkpoint outputs/checkpoints/unrolled512/best.pt
```

Benchmark:

```bash
.venv/bin/python scripts/benchmark_reconstruction.py \
  --unet-checkpoint outputs/checkpoints/unet512/best.pt \
  --unrolled-checkpoint outputs/checkpoints/unrolled512/best.pt
```
