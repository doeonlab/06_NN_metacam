from dataclasses import replace
from pathlib import Path

import torch

from metacam.nn.direct_fullspeckle import FullSpeckleDirectPhaseNet
from metacam.physics.phasecam_forward import (
    PhaseCamForwardModel,
    PhaseCamRealScaleBaseline,
    derive_reduced_phasecam_config,
    gaussian_wrapped_phase_pattern,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_direct_fullspeckle_model_uses_full_measurement_grid():
    reduced, _ = derive_reduced_phasecam_config(PhaseCamRealScaleBaseline(), target_sim_grid_px=128)
    reduced = replace(reduced, phase_file=None)
    meta_phase = gaussian_wrapped_phase_pattern((reduced.meta_pixel_count_px, reduced.meta_pixel_count_px), seed=19)
    forward_model = PhaseCamForwardModel(reduced, project_root=PROJECT_ROOT, meta_phase=meta_phase, device="cpu")
    model = FullSpeckleDirectPhaseNet(
        forward_model=forward_model,
        sensor_base_channels=8,
        sensor_depth=3,
        object_base_channels=8,
        object_depth=3,
    )

    target_phase = torch.randn(2, 1, reduced.object_support_px, reduced.object_support_px) * 0.15
    measurement = forward_model(target_phase)
    outputs = model.forward_with_aux(measurement)

    assert outputs["phase"].shape == target_phase.shape
    assert outputs["sensor_phase"].shape[-2:] == measurement.shape[-2:]
    assert outputs["camera_field"].shape[-2:] == measurement.shape[-2:]
    assert outputs["object_field"].shape[-2:] == (reduced.object_support_px, reduced.object_support_px)
