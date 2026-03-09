"""Run the real-scale PhaseCam Adam+SASM reconstruction test.

Examples:
  python scripts/run_phasecam_realscale_test.py
  python scripts/run_phasecam_realscale_test.py --iterations 10
"""

from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metacam.nn.phasecam_realscale import (
    PhaseCamRealScaleConfig,
    run_phasecam_realscale,
    save_phasecam_realscale_outputs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="PhaseCam real-scale Adam+SASM test runner.")
    parser.add_argument("--device", default="auto", help="Device: auto|cpu|cuda|mps")
    parser.add_argument("--iterations", type=int, default=100, help="Outer optimization iterations (notebook default: 100)")
    parser.add_argument("--progress-every", type=int, default=10, help="Progress print interval")
    parser.add_argument("--test-window", type=int, default=3000, help="Sensor crop width in pixels")
    parser.add_argument("--lr", type=float, default=0.05, help="Adam learning rate")
    parser.add_argument("--tv-weight", type=float, default=0.5, help="TV loss weight")
    parser.add_argument("--widthmap-file", default="0.6NA_random_70_1_300_1mm_mapped_width.mat", help="Widthmap .mat filename in Data/Fab/B17")
    parser.add_argument("--target-file", default="usafimage.png", help="Target image filename in Data/Target")
    parser.add_argument(
        "--save-plots",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Save summary figure (default: enabled)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "outputs" / "experiments" / "phasecam_realscale_test",
        help="Base output directory",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    config = PhaseCamRealScaleConfig(
        project_root=PROJECT_ROOT,
        iterations=args.iterations,
        progress_every=args.progress_every,
        test_window_px=args.test_window,
        lr=args.lr,
        tv_weight=args.tv_weight,
        widthmap_file=args.widthmap_file,
        target_file=args.target_file,
    )

    result, tensors = run_phasecam_realscale(
        config=config,
        device_name=args.device,
        return_tensors=args.save_plots,
    )

    run_dir = args.output_dir / datetime.now().strftime("%Y%m%d_%H%M%S")
    save_phasecam_realscale_outputs(
        result=result,
        save_dir=run_dir,
        tensors=tensors,
        save_plots=args.save_plots,
    )

    print("======================================")
    print("PhaseCam real-scale Adam+SASM test done.")
    print(f"device: {result.device}")
    print(f"use_antialias: {result.use_antialias}")
    print(f"runtime_sec: {result.runtime_sec:.3f}")
    print(f"iterations: {result.iterations}")
    print(f"MASM: {result.masm:.6f}")
    print(f"final_loss_total: {result.final_loss_total:.6f}")
    print(f"final_loss_mse: {result.final_loss_mse:.6f}")
    print(f"final_loss_tv: {result.final_loss_tv:.6f}")
    print(f"final_corr_speckle: {result.final_corr_speckle:.6f}")
    print(f"final_corr_crop500: {result.final_corr_crop500:.6f}")
    print(f"final_corr_crop500_index: {result.final_corr_crop500_index}")
    print(f"final_npcc_crop500: {result.final_npcc_crop500:.6f}")
    print(f"saved: {run_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
