"""Physics operators (ASM/SASM propagation and metalens operators)."""

from metacam.physics.meta_operator import MetaOperator
from metacam.physics.phasecam_forward import (
    PhaseCamForwardModel,
    PhaseCamRealScaleBaseline,
    ReducedPhaseCamConfig,
    SamplingDiagnostics,
    derive_reduced_phasecam_config,
    evaluate_sampling_consistency,
    gaussian_wrapped_phase_pattern,
    load_phase_pattern,
    resolve_metasurface_phase,
    save_phase_pattern,
)
from metacam.physics.propagation import (
    FieldPropagator,
    asm,
    asm_master_alltorch,
    circ_mask,
    elip_mask,
    fft_conv2d,
    lensphase,
    sasm,
    torch_fft,
    torch_ifft,
)

__all__ = [
    "MetaOperator",
    "PhaseCamForwardModel",
    "PhaseCamRealScaleBaseline",
    "ReducedPhaseCamConfig",
    "SamplingDiagnostics",
    "FieldPropagator",
    "asm",
    "asm_master_alltorch",
    "circ_mask",
    "derive_reduced_phasecam_config",
    "elip_mask",
    "evaluate_sampling_consistency",
    "fft_conv2d",
    "gaussian_wrapped_phase_pattern",
    "lensphase",
    "load_phase_pattern",
    "resolve_metasurface_phase",
    "sasm",
    "save_phase_pattern",
    "torch_fft",
    "torch_ifft",
]
