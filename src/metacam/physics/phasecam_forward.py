"""Reduced-scale PhaseCam forward-model helpers.

This module keeps the repo's existing ASM/SASM propagation path and adds a
compact configuration/IO layer that is suitable for synthetic neural
reconstruction experiments.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
import warnings
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from metacam.data.io import load_mat_file
from metacam.ops.torch_ops import torch_crop_center, torch_pad_center
from metacam.physics.propagation import asm_master_alltorch


def _snap_px(value: float, multiple: int, lower_bound: int = 1) -> int:
    if multiple <= 1:
        return max(lower_bound, int(round(value)))
    return max(lower_bound, int(round(value / multiple)) * multiple)


def _asm_target_fov_m(sim_grid_size_px: int, sim_pixel_pitch_m: float) -> float:
    if sim_grid_size_px % 2 == 0:
        return (sim_grid_size_px + 0.1) * sim_pixel_pitch_m
    return sim_grid_size_px * sim_pixel_pitch_m


def _wrap_phase(phase: torch.Tensor) -> torch.Tensor:
    return torch.remainder(phase, 2 * math.pi)


def _bounded_phase(phase: torch.Tensor) -> torch.Tensor:
    return math.pi * torch.tanh(phase / math.pi)


def _resize_wrapped_phase(phase: torch.Tensor, size: tuple[int, int]) -> torch.Tensor:
    complex_phase = torch.polar(torch.ones_like(phase), phase)
    real = F.interpolate(complex_phase.real, size=size, mode="bilinear", antialias=True)
    imag = F.interpolate(complex_phase.imag, size=size, mode="bilinear", antialias=True)
    resized = torch.complex(real, imag)
    return torch.angle(resized)


@dataclass(frozen=True)
class PhaseCamRealScaleBaseline:
    """Real-scale parameters extracted from the existing repo."""

    sim_grid_size_px: int = 5713
    sim_pixel_pitch_m: float = 350e-9
    camera_pixel_pitch_m: float = 1.85e-6
    wavelength_m: float = 532e-9
    aperture_width_m: float = 1e-3
    z_meta_to_sensor_m: float = 6.3e-3
    z_object_to_meta_m: float = 0.4e-3
    sasm_bandlimit_factor: float = 0.5
    widthmap_file: str = "assets/data/Fab/B17/0.6NA_random_70_1_300_1mm_mapped_width.mat"
    lut_file: str = "assets/data/Fab/B17/bayesLUT_MSE_v6.3_nonoverlap.mat"
    target_file: str = "assets/data/Target/usafimage.png"
    measurement_crop_px: int = 3000
    meta_pixel_count_px: int = 2856
    object_support_px: int = 2857

    @property
    def sim_fov_m(self) -> float:
        return self.sim_grid_size_px * self.sim_pixel_pitch_m

    @property
    def aperture_width_px(self) -> int:
        return int(np.fix(self.aperture_width_m / self.sim_pixel_pitch_m))

    @property
    def meta_pixel_pitch_m(self) -> float:
        return self.meta_active_width_m / self.meta_pixel_count_px

    @property
    def meta_active_width_m(self) -> float:
        return self.meta_pixel_count_px * self.sim_pixel_pitch_m

    @property
    def object_support_width_m(self) -> float:
        return self.object_support_px * self.sim_pixel_pitch_m


@dataclass(frozen=True)
class ReducedPhaseCamConfig:
    """Reduced neural-training configuration derived from the real-scale setup."""

    name: str
    scale_factor: float
    sim_grid_size_px: int
    sim_pixel_pitch_m: float
    camera_pixel_pitch_m: float
    wavelength_m: float
    aperture_width_px: int
    aperture_width_m: float
    object_support_px: int
    object_support_width_m: float
    meta_pixel_count_px: int
    meta_active_width_m: float
    z_meta_to_sensor_m: float
    z_object_to_meta_m: float
    sasm_bandlimit_factor: float
    sensor_full_size_px: int
    measurement_window_px: int
    sensor_crop_strategy: str = "full_downsampled"
    phase_file: str | None = None
    widthmap_file: str | None = None
    lut_file: str | None = None
    meta_phase_seed: int = 1234
    mask_mean_rad: float = math.pi
    mask_std_rad: float = math.pi / 2

    @property
    def sim_fov_m(self) -> float:
        return self.sim_grid_size_px * self.sim_pixel_pitch_m

    @property
    def resize_ratio(self) -> float:
        return self.camera_pixel_pitch_m / self.sim_pixel_pitch_m

    @property
    def meta_pixel_pitch_m(self) -> float:
        return self.meta_active_width_m / self.meta_pixel_count_px

    @property
    def object_pixel_pitch_m(self) -> float:
        return self.object_support_width_m / self.object_support_px

    @property
    def speckle_proxy_m(self) -> float:
        return self.wavelength_m * self.z_meta_to_sensor_m / self.aperture_width_m

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SamplingDiagnostics:
    scale_factor: float
    camera_pixel_pitch_ratio: float
    meta_pixel_pitch_ratio: float
    object_pixel_pitch_ratio: float
    speckle_proxy_ratio: float
    sensor_scale_factor: float
    sensor_full_size_px: int
    measurement_window_px: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def derive_reduced_phasecam_config(
    baseline: PhaseCamRealScaleBaseline | None = None,
    target_sim_grid_px: int = 512,
    snap_multiple_px: int = 16,
    measurement_window_strategy: str = "full_downsampled",
    measurement_window_px: int | None = None,
    phase_file: str | None = "assets/data/Fab/generated/test512_gaussian_phase_seed1234.npz",
    meta_phase_seed: int = 1234,
) -> tuple[ReducedPhaseCamConfig, SamplingDiagnostics]:
    """Derive a reduced configuration while preserving sensor sampling trends."""

    if baseline is None:
        baseline = PhaseCamRealScaleBaseline()

    scale_factor = target_sim_grid_px / baseline.sim_grid_size_px
    aperture_width_px = _snap_px(baseline.aperture_width_px * scale_factor, snap_multiple_px)
    object_support_px = _snap_px(baseline.object_support_px * scale_factor, snap_multiple_px)
    meta_pixel_count_px = _snap_px(baseline.meta_pixel_count_px * scale_factor, snap_multiple_px)

    aperture_width_px = min(aperture_width_px, target_sim_grid_px)
    object_support_px = min(object_support_px, aperture_width_px)
    meta_pixel_count_px = min(meta_pixel_count_px, aperture_width_px)

    aperture_width_m = aperture_width_px * baseline.sim_pixel_pitch_m
    object_support_width_m = object_support_px * baseline.sim_pixel_pitch_m
    meta_active_width_m = meta_pixel_count_px * baseline.sim_pixel_pitch_m

    z_meta_to_sensor_m = baseline.z_meta_to_sensor_m * scale_factor
    z_object_to_meta_m = baseline.z_object_to_meta_m * scale_factor

    masm = (baseline.wavelength_m * z_meta_to_sensor_m / baseline.sim_pixel_pitch_m) / (target_sim_grid_px * baseline.sim_pixel_pitch_m)
    sensor_scale_factor = masm / (baseline.camera_pixel_pitch_m / baseline.sim_pixel_pitch_m)
    sensor_full_size_px = int(
        F.interpolate(
            torch.zeros(1, 1, target_sim_grid_px, target_sim_grid_px),
            scale_factor=sensor_scale_factor,
            mode="bilinear",
            antialias=True,
        ).shape[-1]
    )

    if measurement_window_strategy == "full_downsampled":
        measurement_window_px = sensor_full_size_px if measurement_window_px is None else measurement_window_px
    elif measurement_window_strategy == "scaled_baseline_crop":
        scaled_crop = max(1, int(round(baseline.measurement_crop_px * scale_factor)))
        measurement_window_px = scaled_crop if measurement_window_px is None else measurement_window_px
    else:
        raise ValueError(f"Unsupported measurement_window_strategy: {measurement_window_strategy}")

    measurement_window_px = min(measurement_window_px, sensor_full_size_px)

    reduced = ReducedPhaseCamConfig(
        name=f"test{target_sim_grid_px}",
        scale_factor=scale_factor,
        sim_grid_size_px=target_sim_grid_px,
        sim_pixel_pitch_m=baseline.sim_pixel_pitch_m,
        camera_pixel_pitch_m=baseline.camera_pixel_pitch_m,
        wavelength_m=baseline.wavelength_m,
        aperture_width_px=aperture_width_px,
        aperture_width_m=aperture_width_m,
        object_support_px=object_support_px,
        object_support_width_m=object_support_width_m,
        meta_pixel_count_px=meta_pixel_count_px,
        meta_active_width_m=meta_active_width_m,
        z_meta_to_sensor_m=z_meta_to_sensor_m,
        z_object_to_meta_m=z_object_to_meta_m,
        sasm_bandlimit_factor=baseline.sasm_bandlimit_factor,
        sensor_full_size_px=sensor_full_size_px,
        measurement_window_px=measurement_window_px,
        sensor_crop_strategy=measurement_window_strategy,
        phase_file=phase_file,
        widthmap_file=baseline.widthmap_file,
        lut_file=baseline.lut_file,
        meta_phase_seed=meta_phase_seed,
    )
    diagnostics = evaluate_sampling_consistency(baseline, reduced)
    _warn_on_sampling_deviation(diagnostics)
    return reduced, diagnostics


def evaluate_sampling_consistency(
    baseline: PhaseCamRealScaleBaseline,
    reduced: ReducedPhaseCamConfig,
) -> SamplingDiagnostics:
    return SamplingDiagnostics(
        scale_factor=reduced.scale_factor,
        camera_pixel_pitch_ratio=reduced.camera_pixel_pitch_m / baseline.camera_pixel_pitch_m,
        meta_pixel_pitch_ratio=reduced.meta_pixel_pitch_m / baseline.meta_pixel_pitch_m,
        object_pixel_pitch_ratio=reduced.object_pixel_pitch_m / baseline.sim_pixel_pitch_m,
        speckle_proxy_ratio=reduced.speckle_proxy_m / (baseline.wavelength_m * baseline.z_meta_to_sensor_m / baseline.aperture_width_m),
        sensor_scale_factor=reduced.sensor_full_size_px / reduced.sim_grid_size_px,
        sensor_full_size_px=reduced.sensor_full_size_px,
        measurement_window_px=reduced.measurement_window_px,
    )


def _warn_on_sampling_deviation(diagnostics: SamplingDiagnostics, tol: float = 0.05) -> None:
    ratios = {
        "camera_pixel_pitch": diagnostics.camera_pixel_pitch_ratio,
        "meta_pixel_pitch": diagnostics.meta_pixel_pitch_ratio,
        "object_pixel_pitch": diagnostics.object_pixel_pitch_ratio,
        "speckle_proxy": diagnostics.speckle_proxy_ratio,
    }
    for key, value in ratios.items():
        if abs(value - 1.0) > tol:
            warnings.warn(f"{key} changed by more than {tol:.0%}: ratio={value:.4f}", stacklevel=2)


def gaussian_wrapped_phase_pattern(
    shape: tuple[int, int],
    seed: int,
    mean_rad: float = math.pi,
    std_rad: float = math.pi / 2,
    device: torch.device | None = None,
) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    samples = torch.randn(shape, generator=generator, dtype=torch.float32)
    phase = mean_rad + std_rad * samples
    phase = _wrap_phase(phase)
    if device is not None:
        phase = phase.to(device)
    return phase


def save_phase_pattern(
    phase: torch.Tensor,
    path: str | Path,
    metadata: dict[str, Any] | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "phase_wrapped": phase.detach().cpu().numpy().astype(np.float32),
        "metadata_json": json.dumps(metadata or {}, indent=2),
    }
    np.savez(path, **payload)
    return path


def load_phase_pattern(path: str | Path) -> tuple[torch.Tensor, dict[str, Any]]:
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix == ".npz":
        data = np.load(path, allow_pickle=True)
        phase = torch.from_numpy(data["phase_wrapped"]).to(torch.float32)
        metadata_raw = data.get("metadata_json", "{}")
        if isinstance(metadata_raw, np.ndarray):
            metadata_raw = metadata_raw.item()
        metadata = json.loads(metadata_raw)
        return phase, metadata

    if suffix == ".npy":
        return torch.from_numpy(np.load(path)).to(torch.float32), {}

    if suffix == ".pt":
        payload = torch.load(path, map_location="cpu")
        if isinstance(payload, dict) and "phase_wrapped" in payload:
            phase = payload["phase_wrapped"]
            metadata = payload.get("metadata", {})
        else:
            phase = payload
            metadata = {}
        return phase.to(torch.float32), metadata

    if suffix == ".mat":
        data = load_mat_file(path)
        for key in ("phase_wrapped", "meta_phase", "phase"):
            if data is not None and key in data:
                return torch.from_numpy(np.asarray(data[key])).to(torch.float32), {}
        raise KeyError(f"No supported phase key found in {path}")

    raise ValueError(f"Unsupported metasurface phase file: {path}")


def _wl_to_idx(wavelength_nm: float) -> np.uint16:
    return np.uint16((wavelength_nm - 440) * 10)


def build_phase_from_widthmap(
    widthmap_path: str | Path,
    lut_path: str | Path,
    wavelength_m: float,
    target_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    widthmap_data = load_mat_file(widthmap_path)
    if widthmap_data is None or "mapped_width" not in widthmap_data:
        raise FileNotFoundError(f"Could not load mapped_width from {widthmap_path}")
    mapped_width = torch.from_numpy(np.asarray(widthmap_data["mapped_width"], dtype=np.int32))

    lut_data = load_mat_file(lut_path)
    if lut_data is None or "lut_opt_interp" not in lut_data:
        raise FileNotFoundError(f"Could not load lut_opt_interp from {lut_path}")
    lut = np.asarray(lut_data["lut_opt_interp"])
    lut_tensor = torch.from_numpy(lut[:, _wl_to_idx(wavelength_m * 1e9)].astype(np.float32))

    meta_idx = torch.clamp(mapped_width.to(torch.long) - 60, min=0, max=lut_tensor.numel() - 1)
    phase = lut_tensor[meta_idx]
    if target_size is not None and tuple(phase.shape[-2:]) != tuple(target_size):
        phase = _resize_wrapped_phase(phase.unsqueeze(0).unsqueeze(0), size=target_size)[0, 0]
    return phase.to(torch.float32)


def resolve_metasurface_phase(
    config: ReducedPhaseCamConfig,
    project_root: str | Path,
) -> tuple[torch.Tensor, dict[str, Any]]:
    project_root = Path(project_root)
    target_size = (config.meta_pixel_count_px, config.meta_pixel_count_px)

    if config.phase_file is not None:
        phase_path = project_root / config.phase_file
        if phase_path.exists():
            phase, metadata = load_phase_pattern(phase_path)
            if tuple(phase.shape[-2:]) != target_size:
                phase = _resize_wrapped_phase(phase.unsqueeze(0).unsqueeze(0), size=target_size)[0, 0]
            return _wrap_phase(phase), metadata

    if config.widthmap_file is not None and config.lut_file is not None:
        phase = build_phase_from_widthmap(
            widthmap_path=project_root / config.widthmap_file,
            lut_path=project_root / config.lut_file,
            wavelength_m=config.wavelength_m,
            target_size=target_size,
        )
        return _wrap_phase(phase), {"source": "widthmap_lut"}

    phase = gaussian_wrapped_phase_pattern(
        shape=target_size,
        seed=config.meta_phase_seed,
        mean_rad=config.mask_mean_rad,
        std_rad=config.mask_std_rad,
    )
    return phase, {"source": "generated", "seed": config.meta_phase_seed}


class PhaseCamForwardModel(nn.Module):
    """Differentiable phase-only PhaseCam forward model."""

    def __init__(
        self,
        config: ReducedPhaseCamConfig,
        project_root: str | Path,
        meta_phase: torch.Tensor | None = None,
        device: str | torch.device = "cpu",
    ) -> None:
        super().__init__()
        self.config = config
        self.project_root = Path(project_root)
        self.device_hint = torch.device(device)
        self.use_antialias = self.device_hint.type != "mps"

        if meta_phase is None:
            meta_phase, metadata = resolve_metasurface_phase(config, self.project_root)
        else:
            metadata = {"source": "provided"}
        self.meta_phase_metadata = metadata

        self.asmt = asm_master_alltorch(
            _asm_target_fov_m(config.sim_grid_size_px, config.sim_pixel_pitch_m),
            config.sim_pixel_pitch_m,
            self.device_hint,
        )
        self.resize_ratio = config.camera_pixel_pitch_m / config.sim_pixel_pitch_m

        aperture = torch.ones(1, 1, config.aperture_width_px, config.aperture_width_px, dtype=torch.float32)
        aperture = torch_pad_center(aperture, (self.asmt.ydim, self.asmt.xdim)).to(self.device_hint)
        self.register_buffer("aperture_obj", aperture)
        self.register_buffer("aperture_meta", aperture.clone())

        meta_phase = meta_phase.to(torch.float32)
        if tuple(meta_phase.shape[-2:]) != (config.meta_pixel_count_px, config.meta_pixel_count_px):
            meta_phase = _resize_wrapped_phase(
                meta_phase.unsqueeze(0).unsqueeze(0),
                size=(config.meta_pixel_count_px, config.meta_pixel_count_px),
            )[0, 0]
        meta_phase_pad = torch_pad_center(
            meta_phase.unsqueeze(0).unsqueeze(0),
            (self.asmt.ydim, self.asmt.xdim),
        ).to(self.device_hint)
        self.register_buffer("meta_phase", meta_phase_pad)
        self.register_buffer("u_s1", self.aperture_meta * torch.exp(1j * meta_phase_pad))

        wl = torch.tensor([[config.wavelength_m]], device=self.device_hint).reshape(1, 1, 1, 1)
        z_meta = torch.tensor([config.z_meta_to_sensor_m], device=self.device_hint).reshape(1, 1, 1, 1)
        z_obj = torch.tensor([config.z_object_to_meta_m], device=self.device_hint).reshape(1, 1, 1, 1)

        kz_meta = self.asmt.get_kz(wl)
        kz_obj = self.asmt.get_kz(wl)
        kernel_obj = self.asmt.get_kernel(wl, kz=kz_obj, z=z_obj)
        delta_h, q_1, sfov = self.asmt.sasm_get_kernel(wl=wl, z=z_meta, BLfactor=config.sasm_bandlimit_factor)
        masm = sfov.item() / config.sim_fov_m

        self.register_buffer("kernel_obj", kernel_obj)
        self.register_buffer("delta_h", delta_h)
        self.register_buffer("q_1", q_1)
        self.register_buffer("wl", wl)
        self.masm = masm
        self.sensor_scale_factor = self.masm / self.resize_ratio
        self.sensor_full_size_px = int(
            F.interpolate(
                torch.zeros(1, 1, self.asmt.ydim, self.asmt.xdim, device=self.device_hint),
                scale_factor=self.sensor_scale_factor,
                mode="bilinear",
                antialias=self.use_antialias,
            ).shape[-1]
        )
        if self.sensor_full_size_px != config.sensor_full_size_px:
            warnings.warn(
                f"Configured sensor_full_size_px={config.sensor_full_size_px} but actual interpolation gives "
                f"{self.sensor_full_size_px}. Using actual value.",
                stacklevel=2,
            )

    def sensor_summary(self) -> dict[str, Any]:
        return {
            "masm": self.masm,
            "sensor_scale_factor": self.sensor_scale_factor,
            "sensor_full_size_px": self.sensor_full_size_px,
            "measurement_window_px": self.config.measurement_window_px,
            "meta_phase_metadata": self.meta_phase_metadata,
        }

    def build_object_field(self, phase: torch.Tensor) -> torch.Tensor:
        phase = _bounded_phase(phase.to(self.aperture_obj.device))
        phase_pad = torch_pad_center(
            phase,
            (self.asmt.ydim, self.asmt.xdim),
            padval=0.0,
        )
        return self.aperture_obj * torch.exp(1j * phase_pad) + torch.logical_not(self.aperture_obj)

    def forward_sensor_field(self, phase: torch.Tensor) -> torch.Tensor:
        u_obj = self.build_object_field(phase)
        u_prop = self.asmt.prop_w_kernel(U=u_obj, ASM_kernel=self.kernel_obj)
        return self.asmt.sasm_prop_w_kernel(U=self.u_s1 * u_prop, delta_H=self.delta_h, Q_1=self.q_1)

    def downsample_sensor(self, sensor_field: torch.Tensor) -> torch.Tensor:
        amplitude = sensor_field.abs()
        amplitude = F.interpolate(
            amplitude,
            scale_factor=self.sensor_scale_factor,
            mode="bilinear",
            antialias=self.use_antialias,
        )
        if self.config.measurement_window_px < amplitude.shape[-1]:
            amplitude = torch_crop_center(amplitude, self.config.measurement_window_px)
        return amplitude

    def downsample_sensor_complex(self, sensor_field: torch.Tensor) -> torch.Tensor:
        real = F.interpolate(
            sensor_field.real,
            scale_factor=self.sensor_scale_factor,
            mode="bilinear",
            antialias=self.use_antialias,
        )
        imag = F.interpolate(
            sensor_field.imag,
            scale_factor=self.sensor_scale_factor,
            mode="bilinear",
            antialias=self.use_antialias,
        )
        camera_field = torch.complex(real, imag)
        if self.config.measurement_window_px < camera_field.shape[-1]:
            camera_field = torch_crop_center(camera_field, self.config.measurement_window_px)
        return camera_field

    def build_camera_field(self, camera_amplitude: torch.Tensor, camera_phase: torch.Tensor) -> torch.Tensor:
        return torch.polar(camera_amplitude.to(self.aperture_obj.device), camera_phase.to(self.aperture_obj.device))

    def upsample_camera_field(self, camera_field: torch.Tensor) -> torch.Tensor:
        if camera_field.shape[-1] != self.sensor_full_size_px or camera_field.shape[-2] != self.sensor_full_size_px:
            camera_field = torch_pad_center(
                camera_field,
                (self.sensor_full_size_px, self.sensor_full_size_px),
                padval=0.0,
            )
        real = F.interpolate(
            camera_field.real,
            size=(self.asmt.ydim, self.asmt.xdim),
            mode="bilinear",
            antialias=self.use_antialias,
        )
        imag = F.interpolate(
            camera_field.imag,
            size=(self.asmt.ydim, self.asmt.xdim),
            mode="bilinear",
            antialias=self.use_antialias,
        )
        return torch.complex(real, imag)

    def adjoint_sasm_sensor(self, sensor_field: torch.Tensor) -> torch.Tensor:
        sensor_field = torch.fft.ifftshift(sensor_field, dim=(-2, -1))
        sensor_field = torch.fft.ifft2(sensor_field, norm="ortho")
        sensor_field = torch.fft.fftshift(sensor_field, dim=(-2, -1))
        sensor_field = sensor_field * torch.conj(self.q_1)
        sensor_field = torch.fft.fft2(sensor_field, norm="ortho")
        sensor_field = sensor_field * torch.conj(self.delta_h)
        sensor_field = torch.fft.ifft2(sensor_field, norm="ortho")
        return sensor_field

    def backproject_camera_field(
        self,
        camera_field: torch.Tensor,
        crop_to_object: bool = True,
    ) -> torch.Tensor:
        sensor_field = self.upsample_camera_field(camera_field)
        meta_plane = self.adjoint_sasm_sensor(sensor_field)
        obj_prop = torch.conj(self.u_s1) * meta_plane
        obj_field = self.asmt.prop_w_kernel(U=obj_prop, ASM_kernel=torch.conj(self.kernel_obj))
        if crop_to_object:
            return torch_crop_center(obj_field, self.config.object_support_px)
        return obj_field

    def resize_measurement_to_object(self, measurement: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            measurement,
            size=(self.config.object_support_px, self.config.object_support_px),
            mode="bilinear",
            antialias=self.use_antialias,
        )

    def forward(self, phase: torch.Tensor) -> torch.Tensor:
        return self.downsample_sensor(self.forward_sensor_field(phase))
