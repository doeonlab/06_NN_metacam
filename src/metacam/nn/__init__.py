"""Neural reconstruction scaffold modules."""

from metacam.nn.baselines import PhaseIntensityUNet
from metacam.nn.direct_fullspeckle import FullSpeckleDirectPhaseNet
from metacam.nn.phasecam_realscale import (
    PhaseCamRealScaleConfig,
    PhaseCamRealScaleResult,
    PhaseCamRealScaleTensors,
    run_phasecam_realscale,
    save_phasecam_realscale_outputs,
)
from metacam.nn.physics_unrolled import PhysicsGuidedUnrolledNet
from metacam.nn.reconstruction import ReconstructionBatch, ReconstructionModel
from metacam.nn.train_utils import (
    apply_quick_run,
    benchmark_models,
    build_forward_model,
    build_model,
    ensure_metasurface_phase_file,
    evaluate_iterative_baseline,
    evaluate_model,
    load_checkpoint_model,
    load_simulation_config,
    load_yaml_config,
    make_dataloaders,
    run_iterative_phase_reconstruction,
    save_qualitative_phase_figure,
    train_model,
    wait_for_pid_exit,
)

__all__ = [
    "FullSpeckleDirectPhaseNet",
    "PhaseIntensityUNet",
    "PhaseCamRealScaleConfig",
    "PhaseCamRealScaleResult",
    "PhaseCamRealScaleTensors",
    "PhysicsGuidedUnrolledNet",
    "ReconstructionBatch",
    "ReconstructionModel",
    "apply_quick_run",
    "benchmark_models",
    "build_forward_model",
    "build_model",
    "ensure_metasurface_phase_file",
    "evaluate_iterative_baseline",
    "evaluate_model",
    "load_checkpoint_model",
    "load_simulation_config",
    "load_yaml_config",
    "make_dataloaders",
    "run_phasecam_realscale",
    "run_iterative_phase_reconstruction",
    "save_qualitative_phase_figure",
    "save_phasecam_realscale_outputs",
    "train_model",
    "wait_for_pid_exit",
]
