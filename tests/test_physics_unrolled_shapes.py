from dataclasses import replace
from pathlib import Path

import torch

from metacam.nn.physics_unrolled import PhysicsGuidedUnrolledNet
from metacam.physics.phasecam_forward import (
    PhaseCamForwardModel,
    PhaseCamRealScaleBaseline,
    derive_reduced_phasecam_config,
    gaussian_wrapped_phase_pattern,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_unrolled_model_forward_and_backward():
    reduced, _ = derive_reduced_phasecam_config(PhaseCamRealScaleBaseline(), target_sim_grid_px=128)
    reduced = replace(reduced, phase_file=None)
    meta_phase = gaussian_wrapped_phase_pattern((reduced.meta_pixel_count_px, reduced.meta_pixel_count_px), seed=13)
    forward_model = PhaseCamForwardModel(reduced, project_root=PROJECT_ROOT, meta_phase=meta_phase, device="cpu")
    model = PhysicsGuidedUnrolledNet(forward_model=forward_model, num_stages=2, init_base_channels=16, init_depth=3, proximal_channels=16)

    target_phase = torch.randn(1, 1, reduced.object_support_px, reduced.object_support_px) * 0.2
    measurement = forward_model(target_phase)
    prediction = model(measurement)
    loss = (prediction - target_phase).pow(2).mean()
    loss.backward()

    assert prediction.shape == target_phase.shape
    assert any(parameter.grad is not None for parameter in model.parameters())
