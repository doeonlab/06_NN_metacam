"""Train the full-speckle direct-inference PhaseCam model."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from metacam.nn.train_utils import (
    apply_quick_run,
    build_forward_model,
    build_model,
    ensure_metasurface_phase_file,
    evaluate_model,
    load_checkpoint_model,
    load_simulation_config,
    load_yaml_config,
    make_dataloaders,
    save_json,
    seed_everything,
    select_training_device,
    train_model,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the full-speckle direct PhaseCam model.")
    parser.add_argument("--config", default="configs/training/direct_fullspeckle512.yaml")
    parser.add_argument("--device", default="auto", help="Device: auto|cpu|cuda|mps")
    parser.add_argument("--quick", action="store_true", help="Use quick-run dataset and epoch settings")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_yaml_config(PROJECT_ROOT / args.config)
    if args.quick:
        config = apply_quick_run(config)

    seed_everything(int(config["experiment"].get("seed", 42)))
    simulation_config, baseline, diagnostics = load_simulation_config(PROJECT_ROOT, config["experiment"]["simulation_config"])
    ensure_metasurface_phase_file(PROJECT_ROOT, simulation_config)

    device = select_training_device(args.device, priority=config.get("device", {}).get("priority"))
    forward_model = build_forward_model(PROJECT_ROOT, simulation_config, device)
    train_loader, val_loader, test_loader = make_dataloaders(PROJECT_ROOT, simulation_config, config["data"])
    model = build_model(config["experiment"]["model_type"], config["model"], forward_model)

    checkpoint_dir = PROJECT_ROOT / config["experiment"]["checkpoint_dir"]
    output_dir = PROJECT_ROOT / config["experiment"]["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    history_path = output_dir / "history.json"
    save_json(
        output_dir / "config_snapshot.json",
        {"train_config": config, "simulation_config": simulation_config.to_dict(), "baseline": baseline.__dict__, "diagnostics": diagnostics},
    )

    summary = train_model(
        model=model,
        model_type=config["experiment"]["model_type"],
        train_loader=train_loader,
        val_loader=val_loader,
        forward_model=forward_model,
        train_config=config,
        device=device,
        checkpoint_dir=checkpoint_dir,
        history_path=history_path,
    )

    best_model = load_checkpoint_model(checkpoint_dir / "best.pt", config["experiment"]["model_type"], config["model"], forward_model, device)
    test_metrics = evaluate_model(best_model, test_loader, forward_model, config["loss"], device=device)
    save_json(output_dir / "test_metrics.json", test_metrics)

    print(f"device: {device}")
    print(f"checkpoint_dir: {checkpoint_dir}")
    print(f"output_dir: {output_dir}")
    print(f"best_val_loss: {summary['best_val_loss']:.6f}")
    print(f"test_wrapped_phase_mae: {test_metrics['wrapped_phase_mae']:.6f}")
    print(f"test_runtime_sec_per_sample: {test_metrics['runtime_sec_per_sample']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
