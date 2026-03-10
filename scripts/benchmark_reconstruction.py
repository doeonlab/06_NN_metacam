"""Benchmark iterative and neural PhaseCam reconstruction methods."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metacam.nn.train_utils import (
    benchmark_models,
    build_forward_model,
    ensure_metasurface_phase_file,
    load_checkpoint_model,
    load_simulation_config,
    load_yaml_config,
    make_dataloaders,
    save_json,
    select_training_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark iterative and NN PhaseCam reconstruction.")
    parser.add_argument("--unrolled-config", default="configs/training/unrolled512.yaml")
    parser.add_argument("--unrolled-checkpoint", default="outputs/checkpoints/unrolled512/best.pt")
    parser.add_argument("--unet-config", default="configs/training/unet512.yaml")
    parser.add_argument("--unet-checkpoint", default="outputs/checkpoints/unet512/best.pt")
    parser.add_argument("--device", default="auto", help="Device: auto|cpu|cuda|mps")
    parser.add_argument("--output", default="outputs/experiments/benchmark_reconstruction.json")
    parser.add_argument("--refinement-steps", type=int, default=None, help="Override refinement steps")
    parser.add_argument("--test-samples", type=int, default=None, help="Optional override for benchmark test-set size")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    unrolled_cfg = load_yaml_config(PROJECT_ROOT / args.unrolled_config)
    unet_cfg = load_yaml_config(PROJECT_ROOT / args.unet_config)
    if args.test_samples is not None:
        unrolled_cfg["data"]["test_samples"] = args.test_samples
        unet_cfg["data"]["test_samples"] = args.test_samples

    simulation_config, _, _ = load_simulation_config(PROJECT_ROOT, unrolled_cfg["experiment"]["simulation_config"])
    ensure_metasurface_phase_file(PROJECT_ROOT, simulation_config)
    device = select_training_device(args.device, priority=unrolled_cfg.get("device", {}).get("priority"))
    forward_model = build_forward_model(PROJECT_ROOT, simulation_config, device)
    _, _, test_loader = make_dataloaders(PROJECT_ROOT, simulation_config, unrolled_cfg["data"])

    unrolled_model = load_checkpoint_model(args.unrolled_checkpoint, "unrolled", unrolled_cfg["model"], forward_model, device)
    unet_model = load_checkpoint_model(args.unet_checkpoint, "unet", unet_cfg["model"], forward_model, device)

    evaluation_cfg = unrolled_cfg.get("evaluation", {})
    refinement_steps = evaluation_cfg.get("refinement_steps", 0) if args.refinement_steps is None else args.refinement_steps
    results = benchmark_models(
        forward_model=forward_model,
        data_loader=test_loader,
        loss_config=unrolled_cfg["loss"],
        device=device,
        unet_model=unet_model,
        unrolled_model=unrolled_model,
        iterative_steps=int(evaluation_cfg.get("iterative_steps", 40)),
        iterative_lr=float(evaluation_cfg.get("iterative_lr", 0.05)),
        iterative_tv_weight=float(evaluation_cfg.get("iterative_tv_weight", 1e-4)),
        refinement_steps=int(refinement_steps),
        refinement_lr=float(evaluation_cfg.get("refinement_lr", 0.01)),
    )

    output_path = PROJECT_ROOT / args.output
    save_json(output_path, results)
    for method, metrics in results.items():
        print(f"[{method}]")
        for key, value in metrics.items():
            print(f"  {key}: {value:.6f}")
    print(f"saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
