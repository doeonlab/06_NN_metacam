"""Quick import/runtime smoke checks for MetaCam project layout."""

import importlib
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

MODULES = [
    "metacam",
    "metacam.physics.propagation",
    "metacam.physics.meta_operator",
    "metacam.data.io",
    "metacam.ops.torch_ops",
    "metacam.metrics.losses",
    "metacam.metrics.npcc",
    "metacam.vision.phasecam",
    "Library.fieldprop",
    "Library.torch_matfun",
    "fieldprop.fieldprop",
]


def main() -> None:
    for name in MODULES:
        importlib.import_module(name)

    from metacam.metrics.npcc import NPCCloss
    from metacam.physics.propagation import asm_master_alltorch
    from metacam.ops.torch_ops import torch_crop_center, torch_pad_center

    x = torch.ones(1, 1, 8, 8)
    xp = torch_pad_center(x, (10, 10))
    _ = torch_crop_center(xp, 8)

    asmt = asm_master_alltorch(sim_fov_target=8e-6, sim_px=1e-6, device=torch.device("cpu"))
    wl = torch.tensor([[[[550e-9]]]])
    z = torch.tensor([[[[1e-3]]]])
    kz = asmt.get_kz(wl)
    kernel = asmt.get_kernel(wl, kz, z)
    out = asmt.prop_w_kernel(torch.ones(1, 1, asmt.ydim, asmt.xdim, dtype=torch.complex64), kernel)
    _ = NPCCloss(torch.rand(1, 1, 8, 8), torch.rand(1, 1, 8, 8), ch_size=1, batch_size=1)

    print("SMOKE_OK", out.shape)


if __name__ == "__main__":
    main()
