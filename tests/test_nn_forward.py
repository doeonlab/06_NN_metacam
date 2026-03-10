from dataclasses import replace
from pathlib import Path

import torch

from metacam.nn.baselines import PhaseIntensityUNet
from metacam.physics.phasecam_forward import (
    PhaseCamForwardModel,
    PhaseCamRealScaleBaseline,
    derive_reduced_phasecam_config,
    gaussian_wrapped_phase_pattern,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_unet_forward_shape_matches_object_support():
    reduced, _ = derive_reduced_phasecam_config(PhaseCamRealScaleBaseline(), target_sim_grid_px=128)
    reduced = replace(reduced, phase_file=None)
    meta_phase = gaussian_wrapped_phase_pattern((reduced.meta_pixel_count_px, reduced.meta_pixel_count_px), seed=11)
    forward_model = PhaseCamForwardModel(reduced, project_root=PROJECT_ROOT, meta_phase=meta_phase, device="cpu")
    model = PhaseIntensityUNet(object_size_px=reduced.object_support_px, base_channels=16, depth=3)

    phase = torch.zeros(2, 1, reduced.object_support_px, reduced.object_support_px)
    measurement = forward_model(phase)
    prediction = model(measurement)

    assert prediction.shape == phase.shape
    assert prediction.abs().max() <= torch.pi + 1e-5
