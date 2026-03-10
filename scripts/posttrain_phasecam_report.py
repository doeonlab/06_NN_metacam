"""Wait for training to finish, then run evaluation and save qualitative figures."""

from __future__ import annotations

import argparse
from pathlib import Path
import time
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
    save_qualitative_phase_figure,
    select_training_device,
    wait_for_pid_exit,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run post-training inference and qualitative reporting for PhaseCam.")
    parser.add_argument("--config", default="configs/training/unrolled512.yaml")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint path. Defaults to <checkpoint_dir>/best.pt from the config.")
    parser.add_argument("--device", default="auto", help="Device: auto|cpu|cuda|mps")
    parser.add_argument("--wait-pid", type=int, default=None, help="Optional PID to wait for before running inference.")
    parser.add_argument("--poll-seconds", type=float, default=60.0, help="PID polling interval when --wait-pid is used.")
    parser.add_argument("--require-newer-than", type=float, default=None, help="Require checkpoint/history files to be newer than this unix timestamp.")
    parser.add_argument("--test-samples", type=int, default=None, help="Optional override for test split size.")
    parser.add_argument("--num-figure-samples", type=int, default=6, help="Number of qualitative examples to render.")
    parser.add_argument("--refinement-steps", type=int, default=0, help="Optional extra Adam refinement after network inference.")
    parser.add_argument("--refinement-lr", type=float, default=0.01, help="Learning rate for optional refinement.")
    parser.add_argument("--output-dir", default=None, help="Output directory for the report.")
    return parser.parse_args()


def _assert_fresh(path: Path, unix_time: float, label: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{label} missing: {path}")
    if path.stat().st_mtime <= unix_time:
        raise RuntimeError(f"{label} was not updated after the required timestamp: {path}")


def main() -> int:
    args = parse_args()
    config = load_yaml_config(PROJECT_ROOT / args.config)
    if args.test_samples is not None:
        config["data"]["test_samples"] = args.test_samples

    if args.wait_pid is not None:
        print(f"waiting_for_pid: {args.wait_pid}", flush=True)
        wait_for_pid_exit(args.wait_pid, poll_seconds=args.poll_seconds)
        print(f"pid_finished: {args.wait_pid}", flush=True)

    simulation_config, _, _ = load_simulation_config(PROJECT_ROOT, config["experiment"]["simulation_config"])
    ensure_metasurface_phase_file(PROJECT_ROOT, simulation_config)

    checkpoint_path = Path(args.checkpoint) if args.checkpoint else PROJECT_ROOT / config["experiment"]["checkpoint_dir"] / "best.pt"
    history_path = PROJECT_ROOT / config["experiment"]["output_dir"] / "history.json"
    if args.require_newer_than is not None:
        _assert_fresh(checkpoint_path, args.require_newer_than, "checkpoint")
        _assert_fresh(history_path, args.require_newer_than, "history")

    device = select_training_device(args.device, priority=config.get("device", {}).get("priority"))
    forward_model = build_forward_model(PROJECT_ROOT, simulation_config, device)
    _, _, test_loader = make_dataloaders(PROJECT_ROOT, simulation_config, config["data"])
    model = load_checkpoint_model(checkpoint_path, config["experiment"]["model_type"], config["model"], forward_model, device)

    output_dir = Path(args.output_dir) if args.output_dir else PROJECT_ROOT / config["experiment"]["output_dir"] / "posttrain_report"
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics = evaluate_model(
        model,
        test_loader,
        forward_model,
        config["loss"],
        device=device,
        refinement_steps=args.refinement_steps,
        refinement_lr=args.refinement_lr,
        refinement_tv_weight=float(config.get("evaluation", {}).get("iterative_tv_weight", 1e-4)),
    )
    save_json(output_dir / "eval_metrics.json", metrics)

    figure_result = save_qualitative_phase_figure(
        model=model,
        data_loader=test_loader,
        forward_model=forward_model,
        device=device,
        output_path=output_dir / "qualitative_examples.png",
        num_examples=args.num_figure_samples,
        refinement_steps=args.refinement_steps,
        refinement_lr=args.refinement_lr,
        refinement_tv_weight=float(config.get("evaluation", {}).get("iterative_tv_weight", 1e-4)),
    )

    summary = {
        "finished_at_unix": time.time(),
        "checkpoint": str(checkpoint_path),
        "metrics_path": str(output_dir / "eval_metrics.json"),
        **figure_result,
    }
    save_json(output_dir / "report_summary.json", summary)

    print(f"device: {device}")
    print(f"checkpoint: {checkpoint_path}")
    print(f"metrics_saved: {output_dir / 'eval_metrics.json'}")
    print(f"figure_saved: {figure_result['figure_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
