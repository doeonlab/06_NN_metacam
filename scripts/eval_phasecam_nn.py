"""Evaluate a trained PhaseCam neural reconstructor."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metacam.nn.train_utils import (
    build_forward_model,
    ensure_metasurface_phase_file,
    evaluate_model,
    load_checkpoint_model,
    load_simulation_config,
    load_yaml_config,
    make_dataloaders,
    save_json,
    select_training_device,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a trained PhaseCam neural model.")
    parser.add_argument("--config", required=True, help="Training config used for the checkpoint")
    parser.add_argument("--checkpoint", required=True, help="Path to the saved checkpoint")
    parser.add_argument("--device", default="auto", help="Device: auto|cpu|cuda|mps")
    parser.add_argument("--refinement-steps", type=int, default=0, help="Optional Adam refinement steps after inference")
    parser.add_argument("--refinement-lr", type=float, default=0.01, help="Optional Adam refinement learning rate")
    parser.add_argument("--test-samples", type=int, default=None, help="Optional override for the number of test samples")
    parser.add_argument("--output", default=None, help="Optional JSON output path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml_config(PROJECT_ROOT / args.config)
    if args.test_samples is not None:
        config["data"]["test_samples"] = args.test_samples
    simulation_config, _, _ = load_simulation_config(PROJECT_ROOT, config["experiment"]["simulation_config"])
    ensure_metasurface_phase_file(PROJECT_ROOT, simulation_config)
    device = select_training_device(args.device, priority=config.get("device", {}).get("priority"))
    forward_model = build_forward_model(PROJECT_ROOT, simulation_config, device)
    _, _, test_loader = make_dataloaders(PROJECT_ROOT, simulation_config, config["data"])

    model_type = config["experiment"]["model_type"]
    model = load_checkpoint_model(args.checkpoint, model_type, config["model"], forward_model, device)
    metrics = evaluate_model(
        model,
        test_loader,
        forward_model,
        config["loss"],
        device=device,
        refinement_steps=args.refinement_steps,
        refinement_lr=args.refinement_lr,
        refinement_tv_weight=float(config["evaluation"].get("iterative_tv_weight", 1e-4)),
    )

    output_path = Path(args.output) if args.output else PROJECT_ROOT / config["experiment"]["output_dir"] / f"eval_{model_type}.json"
    save_json(output_path, metrics)
    for key, value in metrics.items():
        print(f"{key}: {value:.6f}")
    print(f"saved: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
