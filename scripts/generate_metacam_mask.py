"""Generate a reproducible reduced-scale metasurface phase mask."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metacam.nn.train_utils import load_simulation_config
from metacam.physics.phasecam_forward import gaussian_wrapped_phase_pattern, save_phase_pattern


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate a Gaussian-wrapped metasurface phase mask.")
    parser.add_argument("--simulation-config", default="configs/simulation/test512.yaml")
    parser.add_argument("--output", default=None, help="Override output .npz path")
    parser.add_argument("--seed", type=int, default=None, help="Override mask seed")
    parser.add_argument("--mean-rad", type=float, default=None, help="Gaussian mean in radians")
    parser.add_argument("--std-rad", type=float, default=None, help="Gaussian std in radians")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    simulation_config, _, diagnostics = load_simulation_config(PROJECT_ROOT, args.simulation_config)
    seed = simulation_config.meta_phase_seed if args.seed is None else args.seed
    mean_rad = simulation_config.mask_mean_rad if args.mean_rad is None else args.mean_rad
    std_rad = simulation_config.mask_std_rad if args.std_rad is None else args.std_rad

    output = args.output or simulation_config.phase_file
    if output is None:
        raise ValueError("No output phase-file path is available.")

    phase = gaussian_wrapped_phase_pattern(
        shape=(simulation_config.meta_pixel_count_px, simulation_config.meta_pixel_count_px),
        seed=seed,
        mean_rad=mean_rad,
        std_rad=std_rad,
    )
    metadata = {
        "simulation_config": simulation_config.to_dict(),
        "sampling_diagnostics": diagnostics,
        "distribution": {"mean_rad": mean_rad, "std_rad": std_rad},
        "seed": seed,
    }
    saved = save_phase_pattern(phase, PROJECT_ROOT / output, metadata=metadata)
    print(f"saved_mask: {saved}")
    print(f"shape: {tuple(phase.shape)}")
    print(f"seed: {seed}")
    print(f"mean_rad: {mean_rad}")
    print(f"std_rad: {std_rad}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
