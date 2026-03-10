"""Physics-guided unrolled phase reconstruction models."""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from metacam.nn.baselines import PhaseIntensityUNet, normalize_intensity_input
from metacam.nn.reconstruction import ReconstructionModel
from metacam.physics.phasecam_forward import PhaseCamForwardModel


def _normalize_feature(x: torch.Tensor) -> torch.Tensor:
    scale = x.abs().mean(dim=(-2, -1), keepdim=True).clamp_min(1e-6)
    return x / scale


def _bound_phase(x: torch.Tensor) -> torch.Tensor:
    return math.pi * torch.tanh(x / math.pi)


class ResidualPhaseBlock(nn.Module):
    def __init__(self, in_channels: int, hidden_channels: int = 32) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 1, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class PhysicsGuidedUnrolledNet(ReconstructionModel):
    """Small unrolled phase-retrieval network with learned proximal blocks."""

    def __init__(
        self,
        forward_model: PhaseCamForwardModel,
        num_stages: int = 4,
        init_base_channels: int = 32,
        init_depth: int = 4,
        proximal_channels: int = 32,
        step_size_init: float = 0.25,
        proximal_scale_init: float = 0.1,
    ) -> None:
        super().__init__()
        self.forward_model = forward_model
        self.num_stages = num_stages
        self.object_size_px = forward_model.config.object_support_px
        self.initializer = PhaseIntensityUNet(
            object_size_px=self.object_size_px,
            base_channels=init_base_channels,
            depth=init_depth,
        )
        self.prox_blocks = nn.ModuleList([ResidualPhaseBlock(in_channels=3, hidden_channels=proximal_channels) for _ in range(num_stages)])
        self.log_step_sizes = nn.Parameter(torch.full((num_stages,), math.log(math.expm1(step_size_init))))
        self.log_prox_scales = nn.Parameter(torch.full((num_stages,), math.log(math.expm1(proximal_scale_init))))

    def _step_size(self, idx: int) -> torch.Tensor:
        return F.softplus(self.log_step_sizes[idx]) + 1e-4

    def _prox_scale(self, idx: int) -> torch.Tensor:
        return F.softplus(self.log_prox_scales[idx])

    def forward(self, measurement: torch.Tensor) -> torch.Tensor:
        measurement = measurement.to(self.forward_model.aperture_obj.device)
        phase_state = self.initializer(measurement)

        for stage_idx in range(self.num_stages):
            phase_state = _bound_phase(phase_state)
            phase_state.requires_grad_(True)
            predicted = self.forward_model(phase_state)
            residual = predicted - measurement
            measurement_loss = 0.5 * residual.square().mean()
            gradient = torch.autograd.grad(
                measurement_loss,
                phase_state,
                create_graph=self.training,
                retain_graph=True,
            )[0]

            residual_obj = self.forward_model.resize_measurement_to_object(residual)
            features = torch.cat(
                [
                    phase_state / math.pi,
                    _normalize_feature(gradient),
                    _normalize_feature(residual_obj),
                ],
                dim=1,
            )
            learned_update = self.prox_blocks[stage_idx](features)
            phase_state = phase_state - self._step_size(stage_idx) * gradient
            phase_state = phase_state + self._prox_scale(stage_idx) * learned_update

        return _bound_phase(phase_state)

    def warm_start(self, measurement: torch.Tensor) -> torch.Tensor:
        return self.initializer(measurement)

    def preprocess_measurement(self, measurement: torch.Tensor) -> torch.Tensor:
        return normalize_intensity_input(
            measurement,
            output_size=(self.object_size_px, self.object_size_px),
        )
