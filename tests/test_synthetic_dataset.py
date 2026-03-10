from dataclasses import replace
from pathlib import Path

import torch

from metacam.data.synthetic_phase_dataset import SyntheticPhaseDataset, SyntheticPhaseDatasetConfig
from metacam.physics.phasecam_forward import (
    PhaseCamForwardModel,
    PhaseCamRealScaleBaseline,
    derive_reduced_phasecam_config,
    gaussian_wrapped_phase_pattern,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_dataset_is_deterministic_and_shapes_match():
    reduced, _ = derive_reduced_phasecam_config(PhaseCamRealScaleBaseline(), target_sim_grid_px=128)
    reduced = replace(reduced, phase_file=None)
    meta_phase = gaussian_wrapped_phase_pattern((reduced.meta_pixel_count_px, reduced.meta_pixel_count_px), seed=7)
    forward_model = PhaseCamForwardModel(reduced, project_root=PROJECT_ROOT, meta_phase=meta_phase, device="cpu")

    dataset = SyntheticPhaseDataset(
        config=SyntheticPhaseDatasetConfig(
            num_samples=4,
            object_size_px=reduced.object_support_px,
            seed=99,
            cache_in_memory=False,
            materialize_measurements=True,
        ),
        project_root=PROJECT_ROOT,
        forward_model=forward_model,
    )

    sample_a = dataset[0]
    sample_b = dataset[0]
    assert sample_a["phase"].shape == (1, reduced.object_support_px, reduced.object_support_px)
    assert sample_a["intensity"].shape == (1, reduced.measurement_window_px, reduced.measurement_window_px)
    assert torch.allclose(sample_a["phase"], sample_b["phase"])
    assert torch.allclose(sample_a["intensity"], sample_b["intensity"])
