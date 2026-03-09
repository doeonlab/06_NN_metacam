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
