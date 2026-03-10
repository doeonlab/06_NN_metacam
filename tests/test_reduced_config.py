from metacam.physics.phasecam_forward import PhaseCamRealScaleBaseline, derive_reduced_phasecam_config


def test_reduced_config_preserves_sampling_rules():
    baseline = PhaseCamRealScaleBaseline()
    reduced, diagnostics = derive_reduced_phasecam_config(baseline=baseline, target_sim_grid_px=512)

    assert reduced.sim_grid_size_px == 512
    assert reduced.aperture_width_px == 256
    assert reduced.object_support_px == 256
    assert reduced.meta_pixel_count_px == 256
    assert reduced.camera_pixel_pitch_m == baseline.camera_pixel_pitch_m
    assert reduced.sensor_full_size_px == 463
    assert abs(diagnostics.meta_pixel_pitch_ratio - 1.0) < 1e-6
    assert abs(diagnostics.object_pixel_pitch_ratio - 1.0) < 1e-6
    assert abs(diagnostics.speckle_proxy_ratio - 1.0) < 5e-4
