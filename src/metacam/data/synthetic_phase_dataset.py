"""Synthetic phase-only dataset generation for reduced PhaseCam experiments."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from metacam.physics.phasecam_forward import PhaseCamForwardModel


def _complex_angle(field: torch.Tensor) -> torch.Tensor:
    return torch.atan2(field.imag, field.real)


def _gaussian_kernel(kernel_size: int, sigma: float) -> torch.Tensor:
    coords = torch.arange(kernel_size, dtype=torch.float32) - (kernel_size - 1) / 2
    yy, xx = torch.meshgrid(coords, coords, indexing="ij")
    kernel = torch.exp(-(xx**2 + yy**2) / (2 * sigma**2))
    kernel = kernel / kernel.sum()
    return kernel.view(1, 1, kernel_size, kernel_size)


def _smooth_field(noise: torch.Tensor, sigma: float) -> torch.Tensor:
    kernel_size = max(5, int(round(6 * sigma)) | 1)
    kernel = _gaussian_kernel(kernel_size, sigma).to(noise.device)
    padding = kernel_size // 2
    return F.conv2d(noise, kernel, padding=padding)


def _normalize_phase_map(phase: torch.Tensor, phase_range: float) -> torch.Tensor:
    phase = phase - phase.mean(dim=(-2, -1), keepdim=True)
    phase = phase / phase.abs().amax(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return phase * phase_range


@dataclass(frozen=True)
class SyntheticPhaseDatasetConfig:
    num_samples: int
    object_size_px: int
    phase_range_rad: float = math.pi
    seed: int = 42
    pattern_types: tuple[str, ...] = ("gaussian_field", "blobs", "bandlimited", "edges")
    use_local_images: bool = True
    local_image_dir: str | None = "assets/data/Target"
    materialize_measurements: bool = True
    cache_in_memory: bool = False
    noise_mode: str = "none"
    poisson_peak_count: float = 50_000.0
    gaussian_noise_std: float = 0.0


class SyntheticPhaseDataset(Dataset):
    """Deterministic synthetic phase-only dataset."""

    def __init__(
        self,
        config: SyntheticPhaseDatasetConfig,
        project_root: str | Path,
        forward_model: PhaseCamForwardModel | None = None,
    ) -> None:
        self.config = config
        self.project_root = Path(project_root)
        self.forward_model = forward_model
        self._image_paths = self._discover_local_images()
        self._cache: list[dict[str, Any]] | None = None

        if self.config.cache_in_memory:
            self._cache = [self._make_sample(idx) for idx in range(self.config.num_samples)]

    def _discover_local_images(self) -> list[Path]:
        if not self.config.use_local_images or self.config.local_image_dir is None:
            return []
        image_dir = self.project_root / self.config.local_image_dir
        if not image_dir.exists():
            return []
        paths = [path for path in sorted(image_dir.iterdir()) if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}]
        return paths

    def __len__(self) -> int:
        return self.config.num_samples

    def __getitem__(self, index: int) -> dict[str, Any]:
        if self._cache is not None:
            cached = self._cache[index]
            return {key: value.clone() if isinstance(value, torch.Tensor) else value for key, value in cached.items()}
        return self._make_sample(index)

    def _make_generator(self, index: int) -> torch.Generator:
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.config.seed + index)
        return generator

    def _select_kind(self, index: int) -> str:
        kinds = list(self.config.pattern_types)
        if self._image_paths and "local_images" not in kinds:
            kinds.append("local_images")
        return kinds[index % len(kinds)]

    def _make_sample(self, index: int) -> dict[str, Any]:
        generator = self._make_generator(index)
        kind = self._select_kind(index)
        phase = self._generate_phase(kind, generator, index)
        object_field = torch.polar(torch.ones_like(phase), phase)

        intensity = torch.empty(0)
        camera_phase = torch.empty(0)
        camera_real = torch.empty(0)
        camera_imag = torch.empty(0)
        if self.config.materialize_measurements:
            if self.forward_model is None:
                raise ValueError("forward_model is required when materialize_measurements=True")
            with torch.no_grad():
                phase_batch = phase.unsqueeze(0).unsqueeze(0).to(self.forward_model.aperture_obj.device)
                sensor_field = self.forward_model.forward_sensor_field(phase_batch)
                intensity = self.forward_model.downsample_sensor(sensor_field)[0].cpu()
                camera_field = self.forward_model.downsample_sensor_complex(sensor_field)[0].cpu()
                camera_phase = _complex_angle(camera_field)
                camera_real = camera_field.real
                camera_imag = camera_field.imag
            intensity = self._apply_noise(intensity, generator)

        return {
            "phase": phase.unsqueeze(0),
            "object_real": object_field.real.unsqueeze(0),
            "object_imag": object_field.imag.unsqueeze(0),
            "intensity": intensity,
            "camera_phase": camera_phase,
            "camera_real": camera_real,
            "camera_imag": camera_imag,
            "metadata": {"seed": self.config.seed + index, "kind": kind},
        }

    def _generate_phase(self, kind: str, generator: torch.Generator, index: int) -> torch.Tensor:
        size = self.config.object_size_px
        if kind == "gaussian_field":
            noise = torch.randn(1, 1, size, size, generator=generator)
            sigma = float(torch.randint(low=max(3, size // 32), high=max(4, size // 8), size=(1,), generator=generator).item())
            phase = _smooth_field(noise, sigma=sigma)[0, 0]
        elif kind == "blobs":
            phase = self._blob_phase(size=size, generator=generator)
        elif kind == "bandlimited":
            phase = self._bandlimited_phase(size=size, generator=generator)
        elif kind == "edges":
            phase = self._edge_phase(size=size, generator=generator)
        elif kind == "local_images":
            phase = self._local_image_phase(size=size, generator=generator, index=index)
        else:
            raise ValueError(f"Unsupported phase pattern type: {kind}")
        return _normalize_phase_map(phase.unsqueeze(0).unsqueeze(0), self.config.phase_range_rad)[0, 0]

    def _blob_phase(self, size: int, generator: torch.Generator) -> torch.Tensor:
        yy, xx = torch.meshgrid(
            torch.linspace(-1.0, 1.0, size),
            torch.linspace(-1.0, 1.0, size),
            indexing="ij",
        )
        field = torch.zeros(size, size)
        num_blobs = int(torch.randint(4, 10, size=(1,), generator=generator).item())
        for _ in range(num_blobs):
            cx = torch.empty(1).uniform_(-0.8, 0.8, generator=generator).item()
            cy = torch.empty(1).uniform_(-0.8, 0.8, generator=generator).item()
            sx = torch.empty(1).uniform_(0.05, 0.25, generator=generator).item()
            sy = torch.empty(1).uniform_(0.05, 0.25, generator=generator).item()
            amp = torch.empty(1).uniform_(-1.0, 1.0, generator=generator).item()
            field = field + amp * torch.exp(-((xx - cx) ** 2 / (2 * sx**2) + (yy - cy) ** 2 / (2 * sy**2)))
        return field

    def _bandlimited_phase(self, size: int, generator: torch.Generator) -> torch.Tensor:
        freq = torch.randn(size, size, generator=generator) + 1j * torch.randn(size, size, generator=generator)
        ky = torch.fft.fftfreq(size).view(-1, 1)
        kx = torch.fft.fftfreq(size).view(1, -1)
        cutoff = float(torch.empty(1).uniform_(0.05, 0.18, generator=generator).item())
        mask = (kx**2 + ky**2) <= cutoff**2
        filtered = freq * mask
        field = torch.fft.ifft2(filtered).real
        return field

    def _edge_phase(self, size: int, generator: torch.Generator) -> torch.Tensor:
        coarse = max(8, size // 8)
        noise = torch.rand(1, 1, coarse, coarse, generator=generator)
        threshold = float(torch.empty(1).uniform_(0.35, 0.65, generator=generator).item())
        binary = (noise > threshold).float()
        upsampled = F.interpolate(binary, size=(size, size), mode="nearest")
        sigma = float(torch.randint(low=max(2, size // 64), high=max(3, size // 16), size=(1,), generator=generator).item())
        return _smooth_field(upsampled, sigma=sigma)[0, 0]

    def _local_image_phase(self, size: int, generator: torch.Generator, index: int) -> torch.Tensor:
        if not self._image_paths:
            return self._bandlimited_phase(size=size, generator=generator)
        path = self._image_paths[index % len(self._image_paths)]
        image = Image.open(path).convert("L")
        array = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0)
        array = array.unsqueeze(0).unsqueeze(0)
        array = F.interpolate(array, size=(size, size), mode="bilinear", antialias=True)
        return array[0, 0] * 2 - 1

    def _apply_noise(self, intensity: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
        if self.config.noise_mode == "none":
            return intensity
        if self.config.noise_mode != "poisson_gaussian":
            raise ValueError(f"Unsupported noise mode: {self.config.noise_mode}")

        peak = max(self.config.poisson_peak_count, 1.0)
        scaled = intensity / intensity.amax().clamp_min(1e-6) * peak
        poisson = torch.poisson(scaled)
        noisy = poisson / peak
        if self.config.gaussian_noise_std > 0:
            noisy = noisy + torch.randn_like(noisy, generator=generator) * self.config.gaussian_noise_std
        return noisy.clamp_min(0.0)
