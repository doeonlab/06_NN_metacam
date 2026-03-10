"""Direct inference model that preserves the full speckle measurement."""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from metacam.nn.reconstruction import ReconstructionModel
from metacam.nn.unet_small import UNetSmall
from metacam.physics.phasecam_forward import PhaseCamForwardModel


def _zscore(x: torch.Tensor) -> torch.Tensor:
    mean = x.mean(dim=(-2, -1), keepdim=True)
    std = x.std(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return (x - mean) / std


def _complex_angle(field: torch.Tensor) -> torch.Tensor:
    return torch.atan2(field.imag, field.real)


class FullSpeckleDirectPhaseNet(ReconstructionModel):
    """Full-speckle direct network with sensor-phase prediction and one-shot backprop."""

    def __init__(
        self,
        forward_model: PhaseCamForwardModel,
        sensor_base_channels: int = 24,
        sensor_depth: int = 4,
        object_base_channels: int = 32,
        object_depth: int = 4,
    ) -> None:
        super().__init__()
        self.forward_model = forward_model
        self.sensor_encoder = UNetSmall(
            in_channels=4,
            out_channels=2,
            base_channels=sensor_base_channels,
            depth=sensor_depth,
        )
        self.object_decoder = UNetSmall(
            in_channels=4,
            out_channels=1,
            base_channels=object_base_channels,
            depth=object_depth,
        )

    def measurement_features(self, measurement: torch.Tensor) -> torch.Tensor:
        measurement = measurement.clamp_min(0.0)
        mean_norm = measurement / measurement.mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
        features = [
            _zscore(measurement),
            _zscore(torch.log1p(measurement)),
            _zscore(torch.sqrt(measurement + 1e-6)),
            _zscore(mean_norm),
        ]
        return torch.cat(features, dim=1)

    def sensor_phase_from_measurement(self, measurement: torch.Tensor) -> torch.Tensor:
        sensor_code = self.sensor_encoder(self.measurement_features(measurement))
        sensor_code = sensor_code / sensor_code.norm(dim=1, keepdim=True).clamp_min(1e-6)
        return torch.atan2(sensor_code[:, 1:2], sensor_code[:, 0:1])

    def forward_with_aux(self, measurement: torch.Tensor) -> dict[str, torch.Tensor]:
        measurement = measurement.to(self.forward_model.aperture_obj.device)
        sensor_phase = self.sensor_phase_from_measurement(measurement)
        camera_field = self.forward_model.build_camera_field(measurement, sensor_phase)
        object_field = self.forward_model.backproject_camera_field(camera_field, crop_to_object=True)
        object_features = torch.cat(
            [
                object_field.real,
                object_field.imag,
                object_field.abs(),
                _complex_angle(object_field),
            ],
            dim=1,
        )
        phase = math.pi * torch.tanh(self.object_decoder(object_features) / math.pi)
        return {
            "phase": phase,
            "sensor_phase": sensor_phase,
            "camera_field": camera_field,
            "object_field": object_field,
        }

    def forward(self, measurement: torch.Tensor) -> torch.Tensor:
        return self.forward_with_aux(measurement)["phase"]
