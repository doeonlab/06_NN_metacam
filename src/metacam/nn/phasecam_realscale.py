"""Real-scale PhaseCam Adam+SASM reconstruction runner.

This module reproduces the flow in
`PhaseCam_Simul_RealScale_Adam_SASM_5700px_240915.ipynb` as scriptable code.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
from pathlib import Path
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.transforms.functional as TF
from torchvision.io import read_image

from metacam.data.io import load_mat_file
from metacam.metrics.losses import tv_loss
from metacam.metrics.npcc import NPCCloss
from metacam.ops.torch_ops import normxcorr2_fft, torch_crop_center, torch_pad_center
from metacam.physics.propagation import asm_master_alltorch


@dataclass
class PhaseCamRealScaleConfig:
    """Configuration for notebook-equivalent real-scale Adam+SASM optimization."""

    project_root: Path

    sim_px_m: float = 350e-9
    sim_n: int = 5713
    cam_pixel_pitch_m: float = 1.85e-6
    test_window_px: int = 3000
    aperture_width_m: float = 1e-3

    wavelength_ref_nm: int = 532
    z_meta_to_sensor_m: float = 6.3e-3
    z_object_to_meta_m: float = 0.4e-3
    sasm_blfactor: float = 0.5

    widthmap_file: str = "0.6NA_random_70_1_300_1mm_mapped_width.mat"
    lut_file: str = "bayesLUT_MSE_v6.3_nonoverlap.mat"
    target_file: str = "usafimage.png"

    iterations: int = 100
    lr: float = 0.05
    adam_beta1: float = 0.8
    adam_beta2: float = 0.9
    tv_weight: float = 0.5
    progress_every: int = 10

    fab_dir: Path = field(default_factory=lambda: Path("Data") / "Fab" / "B17")
    target_dir: Path = field(default_factory=lambda: Path("Data") / "Target")


@dataclass
class PhaseCamRealScaleResult:
    """Summary metrics from real-scale reconstruction."""

    device: str
    use_antialias: bool
    runtime_sec: float
    iterations: int
    aperture_width_px: int
    masm: float

    initial_loss_mse: float
    final_loss_total: float
    final_loss_mse: float
    final_loss_tv: float
    final_corr_speckle: float

    final_corr_crop500: float
    final_corr_crop500_index: tuple[int, int]
    final_npcc_crop500: float

    loss_history: list[float]
    corr_history: list[float]


@dataclass
class PhaseCamRealScaleTensors:
    """Optional tensors for visualization/debugging."""

    u_obj: torch.Tensor
    u_obj_fake: torch.Tensor
    spk_real: torch.Tensor
    spk_sim: torch.Tensor
    corr_crop500: torch.Tensor


def select_device(device_name: str = "auto") -> torch.device:
    if device_name == "auto":
        if torch.backends.mps.is_available():
            return torch.device("mps")
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("cpu")
    return torch.device(device_name)


def _wl_to_idx(wavelength_nm: int) -> np.uint16:
    return np.uint16((wavelength_nm - 440) * 10)


def _load_target_usaf(target_path: Path, device: torch.device) -> torch.Tensor:
    usaf = torch.sum(read_image(target_path), dim=0)
    usaf = usaf.to(device).unsqueeze(0).unsqueeze(0).to(torch.float32)
    usaf = usaf - usaf.min()
    usaf = usaf / usaf.max()
    return usaf


def _prepare_model(config: PhaseCamRealScaleConfig, device: torch.device, use_antialias: bool) -> dict[str, Any]:
    sim_fov = config.sim_px_m * config.sim_n
    asmt = asm_master_alltorch(sim_fov, config.sim_px_m, device)

    aperture_width_px = int(np.fix(config.aperture_width_m / config.sim_px_m))
    aperture = torch.ones(1, 1, aperture_width_px, aperture_width_px)
    aperture = torch_pad_center(aperture, (asmt.ydim, asmt.xdim)).to(device)
    aperture_obj = aperture

    fab_dir = config.project_root / config.fab_dir
    target_dir = config.project_root / config.target_dir

    wmap_path = fab_dir / config.widthmap_file
    if not wmap_path.exists():
        raise FileNotFoundError(f"Widthmap file not found: {wmap_path}")
    wmap = load_mat_file(wmap_path)["mapped_width"]
    meta_wmap_idx = torch.tensor(wmap - 60, dtype=torch.int, device=device).unsqueeze(0).unsqueeze(0)
    meta_wmap_idx_1 = meta_wmap_idx

    lut_path = fab_dir / config.lut_file
    if not lut_path.exists():
        raise FileNotFoundError(f"LUT file not found: {lut_path}")
    lut_data = load_mat_file(lut_path)
    lut_opt_interp = lut_data["lut_opt_interp"]
    ldim = lut_opt_interp.shape[0]

    target_path = target_dir / config.target_file
    if not target_path.exists():
        raise FileNotFoundError(f"Target image not found: {target_path}")
    usaf = _load_target_usaf(target_path, device)

    ampimg_tensor = usaf * 0.5 + 0.5
    ampimg_pad = torch_pad_center(ampimg_tensor, (asmt.ydim, asmt.xdim), padval=1)
    phimg_pad = 1 - usaf
    phimg_pad = torch_pad_center(phimg_pad, (asmt.ydim, asmt.xdim), padval=1)
    u_obj = aperture_obj * ampimg_pad * torch.exp(1j * phimg_pad) + torch.logical_not(aperture_obj)

    c = config.sim_n // 2
    u_obj[0, 0, c : c + 5, c : c + 5] = 1 + 0.5j
    u_obj[0, 0, c + 6 : c + 11, c : c + 5] = 1 + 0.5j
    u_obj[0, 0, c : c + 5, c + 6 : c + 11] = 1 + 0.5j
    u_obj[0, 0, c + 6 : c + 11, c + 6 : c + 11] = 1 + 0.5j

    lut_est = torch.tensor(
        lut_opt_interp[:, _wl_to_idx(config.wavelength_ref_nm)],
        dtype=torch.float32,
        device=device,
    ).reshape(1, 1, 1, ldim)
    meta_phmap = lut_est.squeeze()[meta_wmap_idx_1]
    meta_phmap_simgrid = torch_pad_center(meta_phmap, (asmt.ydim, asmt.xdim))
    u_s1 = aperture * torch.exp(1j * meta_phmap_simgrid)

    wl = torch.tensor([[config.wavelength_ref_nm * 1e-9]], device=device).reshape(1, 1, 1, 1)
    z1 = torch.tensor([config.z_meta_to_sensor_m], device=device).reshape(1, 1, 1, 1)
    z2 = torch.tensor([config.z_object_to_meta_m], device=device).reshape(1, 1, 1, 1)

    kz1 = asmt.get_kz(wl)
    kernel1 = asmt.get_kernel(wl, kz=kz1, z=z1)

    kz2 = asmt.get_kz(wl)
    kernel2 = asmt.get_kernel(wl, kz=kz2, z=z2)

    delta_h, q_1, sfov = asmt.sasm_get_kernel(wl=wl, z=z1, BLfactor=config.sasm_blfactor)
    masm = sfov.item() / sim_fov

    u_obj_prop = asmt.prop_w_kernel(U=u_obj, ASM_kernel=kernel2)
    u_s1_prop = asmt.sasm_prop_w_kernel(U=u_s1 * u_obj_prop, delta_H=delta_h, Q_1=q_1)
    spk_real = torch.abs(u_s1_prop)
    spk_real = F.interpolate(
        spk_real,
        scale_factor=1 / ((config.cam_pixel_pitch_m / config.sim_px_m) / masm),
        mode="bilinear",
        antialias=use_antialias,
    )
    spk_real = torch_crop_center(spk_real, config.test_window_px)

    # Extra comparison metric from notebook (ASM vs SASM).
    spk_sasm = torch.abs(u_s1_prop)
    spk_asm = asmt.prop_w_kernel(U=u_s1 * u_obj_prop, ASM_kernel=kernel1).abs()
    spk_asm_downsampled = F.interpolate(spk_asm, scale_factor=1 / masm, mode="bilinear", antialias=use_antialias)
    corr_asm_sasm = torch.amax(
        normxcorr2_fft(torch_crop_center(spk_asm_downsampled, 400), torch_crop_center(spk_sasm, 400), norm=False),
        dim=(2, 3),
        keepdim=True,
    ).item()

    return {
        "asmt": asmt,
        "aperture_width_px": aperture_width_px,
        "u_obj": u_obj,
        "u_s1": u_s1,
        "spk_real": spk_real,
        "kernel2": kernel2,
        "delta_h": delta_h,
        "q_1": q_1,
        "masm": masm,
        "corr_asm_sasm": corr_asm_sasm,
    }


def run_phasecam_realscale(
    config: PhaseCamRealScaleConfig,
    device_name: str = "auto",
    return_tensors: bool = False,
) -> tuple[PhaseCamRealScaleResult, PhaseCamRealScaleTensors | None]:
    """Run notebook-equivalent real-scale Adam+SASM optimization."""

    device = select_device(device_name)
    use_antialias = device.type != "mps"
    t0 = time.time()

    prepared = _prepare_model(config, device, use_antialias)
    asmt = prepared["asmt"]
    aperture_width_px = prepared["aperture_width_px"]
    u_obj = prepared["u_obj"]
    u_s1 = prepared["u_s1"]
    spk_real = prepared["spk_real"]
    kernel2 = prepared["kernel2"]
    delta_h = prepared["delta_h"]
    q_1 = prepared["q_1"]
    masm = prepared["masm"]

    obj_dim = asmt.xdim
    obj_fake = torch.exp(1j * torch.ones(1, 1, obj_dim, obj_dim, device=device)).detach().clone().requires_grad_(True)

    optimizer = optim.Adam([obj_fake], lr=config.lr, betas=(config.adam_beta1, config.adam_beta2))
    criterion_mse = nn.MSELoss()

    loss_history: list[float] = []
    corr_history: list[float] = []
    loss_i0 = None
    spk_sim = torch.zeros_like(spk_real)
    last_loss_mse = float("inf")
    last_loss_tv = float("inf")

    for i in range(config.iterations):
        optimizer.zero_grad()

        u_obj_fake = torch_pad_center(obj_fake, (asmt.ydim, asmt.xdim), padval=0)
        u_prop = asmt.prop_w_kernel(U=u_obj_fake, ASM_kernel=kernel2)
        u_temp = asmt.sasm_prop_w_kernel(U=u_s1 * u_prop, delta_H=delta_h, Q_1=q_1)

        spk_sim = torch.abs(u_temp)
        spk_sim = F.interpolate(
            spk_sim,
            scale_factor=1 / ((config.cam_pixel_pitch_m / config.sim_px_m) / masm),
            mode="bilinear",
            antialias=use_antialias,
        )
        spk_sim = torch_crop_center(spk_sim, dim=config.test_window_px)

        corr = torch.amax(normxcorr2_fft(spk_sim, spk_real, norm=False), dim=(2, 3), keepdim=True)
        corr_history.append(corr.item())

        loss_mse = criterion_mse(spk_real, spk_sim)
        if i == 0:
            loss_i0 = loss_mse.detach()
        loss_mse_norm = loss_mse / loss_i0
        loss_tv_term = tv_loss(obj_fake, order=1)
        loss = loss_mse_norm + loss_tv_term * config.tv_weight
        loss_history.append(loss.item())

        loss.backward(retain_graph=False)
        optimizer.step()

        last_loss_mse = float(loss_mse_norm.detach().cpu())
        last_loss_tv = float(loss_tv_term.detach().cpu())

        if config.progress_every > 0 and i % config.progress_every == 0:
            print("----------------------------")
            print(f"Iteration {i}, Spk del Loss: {loss.item()}")

    # Recompute final tensors with the latest obj_fake state.
    u_obj_fake = torch_pad_center(obj_fake, (asmt.ydim, asmt.xdim), padval=0)
    u_prop = asmt.prop_w_kernel(U=u_obj_fake, ASM_kernel=kernel2)
    u_temp = asmt.sasm_prop_w_kernel(U=u_s1 * u_prop, delta_H=delta_h, Q_1=q_1)
    spk_sim = torch.abs(u_temp)
    spk_sim = F.interpolate(
        spk_sim,
        scale_factor=1 / ((config.cam_pixel_pitch_m / config.sim_px_m) / masm),
        mode="bilinear",
        antialias=use_antialias,
    )
    spk_sim = torch_crop_center(spk_sim, dim=config.test_window_px)

    corr_crop500 = normxcorr2_fft(torch_crop_center(u_obj.abs(), 500), torch_crop_center(u_obj_fake.abs(), 500))
    corr_crop500_max = torch.amax(corr_crop500, dim=(2, 3), keepdim=True).item()
    corr_crop500_idx = torch.unravel_index(torch.argmax(corr_crop500), corr_crop500.shape[-2:])
    npcc_crop500 = NPCCloss(torch_crop_center(u_obj.abs(), 500), torch_crop_center(u_obj_fake.abs(), 500)).item()

    result = PhaseCamRealScaleResult(
        device=str(device),
        use_antialias=use_antialias,
        runtime_sec=time.time() - t0,
        iterations=config.iterations,
        aperture_width_px=aperture_width_px,
        masm=masm,
        initial_loss_mse=float(loss_i0.detach().cpu()) if loss_i0 is not None else float("inf"),
        final_loss_total=loss_history[-1] if loss_history else float("inf"),
        final_loss_mse=last_loss_mse,
        final_loss_tv=last_loss_tv,
        final_corr_speckle=corr_history[-1] if corr_history else float("nan"),
        final_corr_crop500=corr_crop500_max,
        final_corr_crop500_index=(int(corr_crop500_idx[0]), int(corr_crop500_idx[1])),
        final_npcc_crop500=npcc_crop500,
        loss_history=loss_history,
        corr_history=corr_history,
    )

    if not return_tensors:
        return result, None

    tensors = PhaseCamRealScaleTensors(
        u_obj=u_obj.detach().cpu(),
        u_obj_fake=u_obj_fake.detach().cpu(),
        spk_real=spk_real.detach().cpu(),
        spk_sim=spk_sim.detach().cpu(),
        corr_crop500=corr_crop500.detach().cpu(),
    )
    return result, tensors


def save_phasecam_realscale_outputs(
    result: PhaseCamRealScaleResult,
    save_dir: Path,
    tensors: PhaseCamRealScaleTensors | None = None,
    save_plots: bool = True,
) -> None:
    """Save metrics and summary figure for real-scale runs."""

    save_dir.mkdir(parents=True, exist_ok=True)
    with (save_dir / "metrics.json").open("w") as f:
        json.dump(asdict(result), f, indent=2)

    if tensors is None or not save_plots:
        return

    vis_size = min(result.aperture_width_px, tensors.u_obj.shape[-1])
    gt_amp = torch_crop_center(tensors.u_obj, vis_size).abs()[0, 0].numpy()
    gt_phase = torch_crop_center(tensors.u_obj, vis_size).angle()[0, 0].numpy()
    recon_amp = torch_crop_center(tensors.u_obj_fake, vis_size).abs()[0, 0].numpy()
    recon_phase = torch_crop_center(tensors.u_obj_fake, vis_size).angle()[0, 0].numpy()

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.ravel()

    im = axes[0].imshow(gt_amp, cmap="gray")
    axes[0].set_title("GT amp")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    im = axes[1].imshow(gt_phase, cmap="hot")
    axes[1].set_title("GT phase")
    fig.colorbar(im, ax=axes[1], fraction=0.046, pad=0.04)

    im = axes[2].imshow(recon_amp, cmap="gray")
    axes[2].set_title("Recon. amp")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    im = axes[3].imshow(recon_phase, cmap="hot")
    axes[3].set_title("Recon. phase")
    fig.colorbar(im, ax=axes[3], fraction=0.046, pad=0.04)

    axes[4].plot(result.loss_history)
    axes[4].set_title("Loss curve")
    axes[4].set_xlabel("Iteration")
    axes[4].grid(True, alpha=0.2)

    axes[5].plot(result.corr_history)
    axes[5].set_title("Correlation curve")
    axes[5].set_xlabel("Iteration")
    axes[5].grid(True, alpha=0.2)

    im = axes[6].imshow(tensors.spk_real[0, 0].numpy(), cmap="hot")
    axes[6].set_title("Spk intensity GT")
    fig.colorbar(im, ax=axes[6], fraction=0.046, pad=0.04)

    im = axes[7].imshow(tensors.spk_sim[0, 0].numpy(), cmap="hot")
    axes[7].set_title("Spk estimated intensity")
    fig.colorbar(im, ax=axes[7], fraction=0.046, pad=0.04)

    for idx in [0, 1, 2, 3, 6, 7]:
        axes[idx].set_xticks([])
        axes[idx].set_yticks([])

    plt.tight_layout()
    plt.savefig(save_dir / "reconstruction_summary.png", dpi=150)
    plt.close()
