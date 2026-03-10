"""Direct neural reconstruction baselines."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from metacam.nn.reconstruction import ReconstructionModel
from metacam.nn.unet_small import UNetSmall


def normalize_intensity_input(
    measurement: torch.Tensor,
    output_size: tuple[int, int] | None = None,
) -> torch.Tensor:
    x = torch.log1p(measurement.clamp_min(0.0))
    mean = x.mean(dim=(-2, -1), keepdim=True)
    std = x.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    x = (x - mean) / std
    if output_size is not None and tuple(x.shape[-2:]) != tuple(output_size):
        x = F.interpolate(
            x,
            size=output_size,
            mode="bilinear",
            antialias=x.device.type != "mps",
        )
    return x


class PhaseIntensityUNet(ReconstructionModel):
    """Simple U-Net baseline from measured intensity to wrapped phase."""

    def __init__(
        self,
        object_size_px: int,
        base_channels: int = 32,
        depth: int = 4,
        phase_range_rad: float = math.pi,
    ) -> None:
        super().__init__()
        self.object_size_px = object_size_px
        self.phase_range_rad = phase_range_rad
        self.backbone = UNetSmall(
            in_channels=1,
            out_channels=1,
            base_channels=base_channels,
            depth=depth,
        )

    def preprocess(self, measurement: torch.Tensor) -> torch.Tensor:
        return normalize_intensity_input(
            measurement,
            output_size=(self.object_size_px, self.object_size_px),
        )

    def forward(self, measurement: torch.Tensor) -> torch.Tensor:
        x = self.preprocess(measurement)
        phase = self.backbone(x)
        return self.phase_range_rad * torch.tanh(phase / self.phase_range_rad)
