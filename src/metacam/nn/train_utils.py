"""Training, evaluation, and benchmarking helpers for MetaCam NN models."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
import math
import os
from pathlib import Path
import random
import time
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.utils.data import DataLoader
import yaml

from metacam.data.synthetic_phase_dataset import SyntheticPhaseDataset, SyntheticPhaseDatasetConfig
from metacam.metrics.losses import tv_loss
from metacam.nn.baselines import PhaseIntensityUNet
from metacam.nn.physics_unrolled import PhysicsGuidedUnrolledNet
from metacam.physics.phasecam_forward import (
    PhaseCamForwardModel,
    PhaseCamRealScaleBaseline,
    ReducedPhaseCamConfig,
    derive_reduced_phasecam_config,
    evaluate_sampling_consistency,
    gaussian_wrapped_phase_pattern,
    save_phase_pattern,
)


def load_yaml_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def merge_nested_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge_nested_dict(merged[key], value)
        else:
            merged[key] = value
    return merged


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_training_device(device_name: str = "auto", priority: list[str] | None = None) -> torch.device:
    if device_name != "auto":
        return torch.device(device_name)
    for candidate in priority or ["mps", "cuda", "cpu"]:
        if candidate == "cuda" and torch.cuda.is_available():
            return torch.device("cuda")
        if candidate == "mps" and torch.backends.mps.is_available():
            return torch.device("mps")
        if candidate == "cpu":
            return torch.device("cpu")
    return torch.device("cpu")


def _replace_reduced_config(config: ReducedPhaseCamConfig, **updates: Any) -> ReducedPhaseCamConfig:
    return replace(config, **updates)


def load_simulation_config(
    project_root: str | Path,
    config_path: str | Path,
) -> tuple[ReducedPhaseCamConfig, PhaseCamRealScaleBaseline, dict[str, Any]]:
    project_root = Path(project_root)
    payload = load_yaml_config(project_root / config_path if not Path(config_path).is_absolute() else config_path)
    baseline = PhaseCamRealScaleBaseline(**payload.get("baseline", {}))

    reduced_payload = payload.get("reduced")
    if reduced_payload is None:
        reduced, diagnostics = derive_reduced_phasecam_config(baseline=baseline)
    else:
        reduced = ReducedPhaseCamConfig(**reduced_payload)
        diagnostics = evaluate_sampling_consistency(baseline, reduced)

    mask_payload = payload.get("mask", {})
    if mask_payload:
        updates = {}
        if "phase_file" in mask_payload:
            updates["phase_file"] = mask_payload["phase_file"]
        if "seed" in mask_payload:
            updates["meta_phase_seed"] = mask_payload["seed"]
        distribution = mask_payload.get("distribution", {})
        if "mean_rad" in distribution:
            updates["mask_mean_rad"] = distribution["mean_rad"]
        if "std_rad" in distribution:
            updates["mask_std_rad"] = distribution["std_rad"]
        if updates:
            reduced = _replace_reduced_config(reduced, **updates)

    return reduced, baseline, diagnostics.to_dict()


def ensure_metasurface_phase_file(
    project_root: str | Path,
    simulation_config: ReducedPhaseCamConfig,
) -> Path | None:
    if simulation_config.phase_file is None:
        return None
    phase_path = Path(project_root) / simulation_config.phase_file
    if phase_path.exists():
        return phase_path

    phase = gaussian_wrapped_phase_pattern(
        shape=(simulation_config.meta_pixel_count_px, simulation_config.meta_pixel_count_px),
        seed=simulation_config.meta_phase_seed,
        mean_rad=simulation_config.mask_mean_rad,
        std_rad=simulation_config.mask_std_rad,
    )
    metadata = {
        "seed": simulation_config.meta_phase_seed,
        "distribution": {"mean_rad": simulation_config.mask_mean_rad, "std_rad": simulation_config.mask_std_rad},
        "config": asdict(simulation_config),
    }
    save_phase_pattern(phase, phase_path, metadata=metadata)
    return phase_path


def build_forward_model(
    project_root: str | Path,
    simulation_config: ReducedPhaseCamConfig,
    device: torch.device,
) -> PhaseCamForwardModel:
    return PhaseCamForwardModel(
        config=simulation_config,
        project_root=project_root,
        device=device,
    )


def build_model(
    model_type: str,
    model_config: dict[str, Any],
    forward_model: PhaseCamForwardModel,
) -> nn.Module:
    if model_type == "unet":
        return PhaseIntensityUNet(
            object_size_px=forward_model.config.object_support_px,
            base_channels=model_config.get("base_channels", 32),
            depth=model_config.get("depth", 4),
        )
    if model_type == "unrolled":
        return PhysicsGuidedUnrolledNet(
            forward_model=forward_model,
            num_stages=model_config.get("num_stages", 4),
            init_base_channels=model_config.get("base_channels", 32),
            init_depth=model_config.get("depth", 4),
            proximal_channels=model_config.get("proximal_channels", 32),
            step_size_init=model_config.get("step_size_init", 0.25),
            proximal_scale_init=model_config.get("proximal_scale_init", 0.1),
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def make_dataset(
    project_root: str | Path,
    simulation_config: ReducedPhaseCamConfig,
    dataset_config: dict[str, Any],
    split: str,
) -> SyntheticPhaseDataset:
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unsupported split: {split}")

    split_key = f"{split}_samples"
    split_offset = {"train": 0, "val": 100_000, "test": 200_000}[split]
    materialize = dataset_config.get("materialize_measurements", True)

    dataset_forward = build_forward_model(project_root, simulation_config, torch.device("cpu")) if materialize else None
    config = SyntheticPhaseDatasetConfig(
        num_samples=int(dataset_config[split_key]),
        object_size_px=simulation_config.object_support_px,
        phase_range_rad=dataset_config.get("phase_range_rad", math.pi),
        seed=int(dataset_config.get("seed", 42)) + split_offset,
        pattern_types=tuple(dataset_config.get("pattern_types", ["gaussian_field", "blobs", "bandlimited", "edges"])),
        use_local_images=bool(dataset_config.get("use_local_images", True)),
        local_image_dir=dataset_config.get("local_image_dir", "assets/data/Target"),
        materialize_measurements=materialize,
        cache_in_memory=bool(dataset_config.get(f"cache_{split}", False)),
        noise_mode=dataset_config.get("noise_mode", "none"),
        poisson_peak_count=float(dataset_config.get("poisson_peak_count", 50_000.0)),
        gaussian_noise_std=float(dataset_config.get("gaussian_noise_std", 0.0)),
    )
    return SyntheticPhaseDataset(config=config, project_root=project_root, forward_model=dataset_forward)


def make_dataloaders(
    project_root: str | Path,
    simulation_config: ReducedPhaseCamConfig,
    data_config: dict[str, Any],
) -> tuple[DataLoader, DataLoader, DataLoader]:
    train_dataset = make_dataset(project_root, simulation_config, data_config, split="train")
    val_dataset = make_dataset(project_root, simulation_config, data_config, split="val")
    test_dataset = make_dataset(project_root, simulation_config, data_config, split="test")

    batch_size = int(data_config.get("batch_size", 1))
    eval_batch_size = int(data_config.get("eval_batch_size", 1))
    num_workers = int(data_config.get("num_workers", 0))

    return (
        DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers),
        DataLoader(val_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers),
        DataLoader(test_dataset, batch_size=eval_batch_size, shuffle=False, num_workers=num_workers),
    )


def wrapped_phase_delta(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.atan2(torch.sin(prediction - target), torch.cos(prediction - target))


def periodic_phase_loss(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred_complex = torch.polar(torch.ones_like(prediction), prediction)
    target_complex = torch.polar(torch.ones_like(target), target)
    return (pred_complex - target_complex).abs().pow(2).mean()


def phase_metrics(prediction: torch.Tensor, target: torch.Tensor) -> dict[str, float]:
    delta = wrapped_phase_delta(prediction, target)
    mae = delta.abs().mean().item()
    rmse = torch.sqrt(delta.pow(2).mean()).item()
    pred_complex = torch.polar(torch.ones_like(prediction), prediction)
    target_complex = torch.polar(torch.ones_like(target), target)
    numerator = torch.mean(pred_complex * torch.conj(target_complex))
    denominator = torch.sqrt(torch.mean(pred_complex.abs().pow(2)) * torch.mean(target_complex.abs().pow(2))).clamp_min(1e-6)
    corr = (numerator.abs() / denominator).item()
    return {
        "wrapped_phase_mae": mae,
        "wrapped_phase_rmse": rmse,
        "complex_correlation": corr,
    }


def compute_loss_terms(
    prediction: torch.Tensor,
    target: torch.Tensor,
    measurement: torch.Tensor,
    forward_model: PhaseCamForwardModel,
    loss_config: dict[str, Any],
) -> tuple[torch.Tensor, dict[str, torch.Tensor], torch.Tensor]:
    measurement_prediction = forward_model(prediction)
    phase_loss = periodic_phase_loss(prediction, target)
    meas_loss = F.mse_loss(measurement_prediction, measurement)
    tv_term = tv_loss(prediction, order=1)
    total = (
        float(loss_config.get("phase", 1.0)) * phase_loss
        + float(loss_config.get("measurement", 0.25)) * meas_loss
        + float(loss_config.get("tv", 1e-4)) * tv_term
    )
    terms = {
        "phase_loss": phase_loss,
        "measurement_loss": meas_loss,
        "tv_loss": tv_term,
    }
    return total, terms, measurement_prediction


def _sync_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def train_model(
    model: nn.Module,
    model_type: str,
    train_loader: DataLoader,
    val_loader: DataLoader,
    forward_model: PhaseCamForwardModel,
    train_config: dict[str, Any],
    device: torch.device,
    checkpoint_dir: str | Path,
    history_path: str | Path,
) -> dict[str, Any]:
    model.to(device)
    optimizer = AdamW(
        model.parameters(),
        lr=float(train_config["optimization"].get("learning_rate", 2e-4)),
        weight_decay=float(train_config["optimization"].get("weight_decay", 1e-4)),
    )
    epochs = int(train_config["optimization"].get("epochs", 10))
    grad_clip_norm = float(train_config["optimization"].get("grad_clip_norm", 0.0))
    patience = int(train_config["optimization"].get("early_stopping_patience", max(epochs, 1)))
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    history: list[dict[str, Any]] = []
    best_val = float("inf")
    best_epoch = 0
    bad_epochs = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_total = 0.0
        train_count = 0
        for batch in train_loader:
            measurement = batch["intensity"].to(device)
            target = batch["phase"].to(device)

            optimizer.zero_grad(set_to_none=True)
            prediction = model(measurement)
            loss, terms, _ = compute_loss_terms(prediction, target, measurement, forward_model, train_config["loss"])
            loss.backward()
            if grad_clip_norm > 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip_norm)
            optimizer.step()

            train_total += float(loss.detach().cpu()) * measurement.size(0)
            train_count += measurement.size(0)

        val_metrics = evaluate_model(model, val_loader, forward_model, train_config["loss"], device=device)
        epoch_payload = {
            "epoch": epoch,
            "train_loss": train_total / max(train_count, 1),
            **val_metrics,
        }
        history.append(epoch_payload)
        print(
            f"epoch={epoch} train_loss={epoch_payload['train_loss']:.6f} "
            f"val_loss={epoch_payload['loss']:.6f} "
            f"val_wrapped_phase_mae={epoch_payload['wrapped_phase_mae']:.6f}",
            flush=True,
        )
        save_json(history_path, {"model_type": model_type, "history": history})

        checkpoint_payload = {
            "model_type": model_type,
            "epoch": epoch,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
            "train_config": train_config,
        }
        torch.save(checkpoint_payload, checkpoint_dir / "last.pt")

        if val_metrics["loss"] < best_val:
            best_val = val_metrics["loss"]
            best_epoch = epoch
            bad_epochs = 0
            torch.save(checkpoint_payload, checkpoint_dir / "best.pt")
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                break

    summary = {
        "best_val_loss": best_val,
        "best_epoch": best_epoch,
        "checkpoint_dir": str(checkpoint_dir),
        "history_path": str(history_path),
    }
    save_json(Path(history_path).with_name("summary.json"), summary)
    return summary


def evaluate_model(
    model: nn.Module,
    data_loader: DataLoader,
    forward_model: PhaseCamForwardModel,
    loss_config: dict[str, Any],
    device: torch.device,
    refinement_steps: int = 0,
    refinement_lr: float = 0.01,
    refinement_tv_weight: float = 1e-4,
) -> dict[str, float]:
    model.eval()
    accum = {
        "loss": 0.0,
        "phase_loss": 0.0,
        "measurement_loss": 0.0,
        "tv_loss": 0.0,
        "wrapped_phase_mae": 0.0,
        "wrapped_phase_rmse": 0.0,
        "complex_correlation": 0.0,
        "intensity_consistency_error": 0.0,
        "runtime_sec_per_sample": 0.0,
    }
    count = 0

    for batch in data_loader:
        measurement = batch["intensity"].to(device)
        target = batch["phase"].to(device)

        _sync_device(device)
        t0 = time.perf_counter()
        with torch.enable_grad():
            prediction = model(measurement)
            if refinement_steps > 0:
                prediction, _ = run_iterative_phase_reconstruction(
                    forward_model=forward_model,
                    measurement=measurement,
                    iterations=refinement_steps,
                    lr=refinement_lr,
                    tv_weight=refinement_tv_weight,
                    init_phase=prediction.detach(),
                )
        _sync_device(device)
        runtime = time.perf_counter() - t0

        prediction = prediction.detach()
        total, terms, measurement_prediction = compute_loss_terms(prediction, target, measurement, forward_model, loss_config)
        metrics = phase_metrics(prediction, target)
        batch_size = measurement.size(0)
        count += batch_size

        accum["loss"] += float(total.detach().cpu()) * batch_size
        accum["phase_loss"] += float(terms["phase_loss"].detach().cpu()) * batch_size
        accum["measurement_loss"] += float(terms["measurement_loss"].detach().cpu()) * batch_size
        accum["tv_loss"] += float(terms["tv_loss"].detach().cpu()) * batch_size
        accum["wrapped_phase_mae"] += metrics["wrapped_phase_mae"] * batch_size
        accum["wrapped_phase_rmse"] += metrics["wrapped_phase_rmse"] * batch_size
        accum["complex_correlation"] += metrics["complex_correlation"] * batch_size
        accum["intensity_consistency_error"] += float(F.mse_loss(measurement_prediction, measurement).detach().cpu()) * batch_size
        accum["runtime_sec_per_sample"] += runtime

    if count == 0:
        return {key: float("nan") for key in accum}
    return {key: value / count for key, value in accum.items()}


def run_iterative_phase_reconstruction(
    forward_model: PhaseCamForwardModel,
    measurement: torch.Tensor,
    iterations: int = 40,
    lr: float = 0.05,
    tv_weight: float = 1e-4,
    init_phase: torch.Tensor | None = None,
) -> tuple[torch.Tensor, float]:
    device = measurement.device
    if init_phase is None:
        phase = torch.zeros(
            measurement.size(0),
            1,
            forward_model.config.object_support_px,
            forward_model.config.object_support_px,
            device=device,
        )
    else:
        phase = init_phase.detach().clone().to(device)

    phase_param = nn.Parameter(phase)
    optimizer = torch.optim.Adam([phase_param], lr=lr)
    _sync_device(device)
    t0 = time.perf_counter()
    for _ in range(iterations):
        optimizer.zero_grad(set_to_none=True)
        prediction = math.pi * torch.tanh(phase_param / math.pi)
        measurement_prediction = forward_model(prediction)
        loss = F.mse_loss(measurement_prediction, measurement) + tv_weight * tv_loss(prediction, order=1)
        loss.backward()
        optimizer.step()
    _sync_device(device)
    runtime = time.perf_counter() - t0
    with torch.no_grad():
        prediction = math.pi * torch.tanh(phase_param / math.pi)
    return prediction.detach(), runtime


def evaluate_iterative_baseline(
    data_loader: DataLoader,
    forward_model: PhaseCamForwardModel,
    loss_config: dict[str, Any],
    iterative_steps: int,
    iterative_lr: float,
    iterative_tv_weight: float,
    device: torch.device,
    init_model: nn.Module | None = None,
) -> dict[str, float]:
    accum = {
        "loss": 0.0,
        "wrapped_phase_mae": 0.0,
        "wrapped_phase_rmse": 0.0,
        "complex_correlation": 0.0,
        "intensity_consistency_error": 0.0,
        "runtime_sec_per_sample": 0.0,
    }
    count = 0

    if init_model is not None:
        init_model.eval()

    for batch in data_loader:
        measurement = batch["intensity"].to(device)
        target = batch["phase"].to(device)
        init_phase = None
        if init_model is not None:
            with torch.enable_grad():
                init_phase = init_model(measurement).detach()
        prediction, runtime = run_iterative_phase_reconstruction(
            forward_model=forward_model,
            measurement=measurement,
            iterations=iterative_steps,
            lr=iterative_lr,
            tv_weight=iterative_tv_weight,
            init_phase=init_phase,
        )
        total, _, measurement_prediction = compute_loss_terms(prediction, target, measurement, forward_model, loss_config)
        metrics = phase_metrics(prediction, target)
        batch_size = measurement.size(0)
        count += batch_size
        accum["loss"] += float(total.detach().cpu()) * batch_size
        accum["wrapped_phase_mae"] += metrics["wrapped_phase_mae"] * batch_size
        accum["wrapped_phase_rmse"] += metrics["wrapped_phase_rmse"] * batch_size
        accum["complex_correlation"] += metrics["complex_correlation"] * batch_size
        accum["intensity_consistency_error"] += float(F.mse_loss(measurement_prediction, measurement).detach().cpu()) * batch_size
        accum["runtime_sec_per_sample"] += runtime

    return {key: value / max(count, 1) for key, value in accum.items()}


def load_checkpoint_model(
    checkpoint_path: str | Path,
    model_type: str | None,
    model_config: dict[str, Any] | None,
    forward_model: PhaseCamForwardModel,
    device: torch.device,
) -> nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    checkpoint_model_type = checkpoint.get("model_type", model_type)
    checkpoint_model_config = checkpoint.get("train_config", {}).get("model", model_config)
    if checkpoint_model_type is None or checkpoint_model_config is None:
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain enough model metadata to rebuild the model.")
    model = build_model(checkpoint_model_type, checkpoint_model_config, forward_model).to(device)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return model


def apply_quick_run(train_config: dict[str, Any]) -> dict[str, Any]:
    quick = train_config.get("quick_run", {})
    if not quick:
        return train_config
    updated = merge_nested_dict(train_config, {"data": {}, "optimization": {}})
    data_updates = {key: quick[key] for key in ("train_samples", "val_samples", "test_samples") if key in quick}
    opt_updates = {key: quick[key] for key in ("epochs",) if key in quick}
    updated["data"] = merge_nested_dict(updated["data"], data_updates)
    updated["optimization"] = merge_nested_dict(updated["optimization"], opt_updates)
    return updated


def benchmark_models(
    forward_model: PhaseCamForwardModel,
    data_loader: DataLoader,
    loss_config: dict[str, Any],
    device: torch.device,
    unet_model: nn.Module | None = None,
    unrolled_model: nn.Module | None = None,
    iterative_steps: int = 40,
    iterative_lr: float = 0.05,
    iterative_tv_weight: float = 1e-4,
    refinement_steps: int = 0,
    refinement_lr: float = 0.01,
) -> dict[str, dict[str, float]]:
    results: dict[str, dict[str, float]] = {}
    if unet_model is not None:
        results["unet"] = evaluate_model(unet_model, data_loader, forward_model, loss_config, device=device)
    if unrolled_model is not None:
        results["unrolled"] = evaluate_model(unrolled_model, data_loader, forward_model, loss_config, device=device)
        if refinement_steps > 0:
            results["unrolled_refined"] = evaluate_model(
                unrolled_model,
                data_loader,
                forward_model,
                loss_config,
                device=device,
                refinement_steps=refinement_steps,
                refinement_lr=refinement_lr,
                refinement_tv_weight=iterative_tv_weight,
            )
    results["iterative_adam"] = evaluate_iterative_baseline(
        data_loader=data_loader,
        forward_model=forward_model,
        loss_config=loss_config,
        iterative_steps=iterative_steps,
        iterative_lr=iterative_lr,
        iterative_tv_weight=iterative_tv_weight,
        device=device,
    )
    return results


def _run_model_inference(
    model: nn.Module,
    measurement: torch.Tensor,
    forward_model: PhaseCamForwardModel,
    refinement_steps: int = 0,
    refinement_lr: float = 0.01,
    refinement_tv_weight: float = 1e-4,
) -> tuple[torch.Tensor, torch.Tensor]:
    with torch.enable_grad():
        prediction = model(measurement)
        if refinement_steps > 0:
            prediction, _ = run_iterative_phase_reconstruction(
                forward_model=forward_model,
                measurement=measurement,
                iterations=refinement_steps,
                lr=refinement_lr,
                tv_weight=refinement_tv_weight,
                init_phase=prediction.detach(),
            )
    predicted_measurement = forward_model(prediction)
    return prediction.detach(), predicted_measurement.detach()


def save_qualitative_phase_figure(
    model: nn.Module,
    data_loader: DataLoader,
    forward_model: PhaseCamForwardModel,
    device: torch.device,
    output_path: str | Path,
    num_examples: int = 6,
    refinement_steps: int = 0,
    refinement_lr: float = 0.01,
    refinement_tv_weight: float = 1e-4,
) -> dict[str, Any]:
    model.eval()
    rows: list[dict[str, Any]] = []
    example_metrics: list[dict[str, float]] = []

    for batch in data_loader:
        measurement = batch["intensity"].to(device)
        target = batch["phase"].to(device)
        prediction, predicted_measurement = _run_model_inference(
            model=model,
            measurement=measurement,
            forward_model=forward_model,
            refinement_steps=refinement_steps,
            refinement_lr=refinement_lr,
            refinement_tv_weight=refinement_tv_weight,
        )
        batch_size = measurement.size(0)
        for idx in range(batch_size):
            target_i = target[idx : idx + 1]
            prediction_i = prediction[idx : idx + 1]
            measurement_i = measurement[idx : idx + 1]
            predicted_measurement_i = predicted_measurement[idx : idx + 1]
            wrapped_error = wrapped_phase_delta(prediction_i, target_i).abs()
            metrics = phase_metrics(prediction_i, target_i)
            metrics["intensity_consistency_error"] = float(F.mse_loss(predicted_measurement_i, measurement_i).detach().cpu())
            example_metrics.append(metrics)
            rows.append(
                {
                    "measurement": measurement_i[0, 0].detach().cpu().numpy(),
                    "target_phase": target_i[0, 0].detach().cpu().numpy(),
                    "prediction_phase": prediction_i[0, 0].detach().cpu().numpy(),
                    "wrapped_error": wrapped_error[0, 0].detach().cpu().numpy(),
                    "predicted_measurement": predicted_measurement_i[0, 0].detach().cpu().numpy(),
                }
            )
            if len(rows) >= num_examples:
                break
        if len(rows) >= num_examples:
            break

    if not rows:
        raise ValueError("No qualitative examples were collected.")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    num_rows = len(rows)
    fig, axes = plt.subplots(num_rows, 5, figsize=(18, 3.5 * num_rows), squeeze=False)
    titles = [
        "Input intensity",
        "GT phase",
        "Predicted phase",
        "|Wrapped error|",
        "Predicted intensity",
    ]

    phase_vmin, phase_vmax = -math.pi, math.pi
    for row_idx, row in enumerate(rows):
        panels = [
            (row["measurement"], "inferno", None, None),
            (row["target_phase"], "twilight", phase_vmin, phase_vmax),
            (row["prediction_phase"], "twilight", phase_vmin, phase_vmax),
            (row["wrapped_error"], "magma", 0.0, math.pi),
            (row["predicted_measurement"], "inferno", None, None),
        ]
        for col_idx, (image, cmap, vmin, vmax) in enumerate(panels):
            ax = axes[row_idx, col_idx]
            im = ax.imshow(image, cmap=cmap, vmin=vmin, vmax=vmax)
            if row_idx == 0:
                ax.set_title(titles[col_idx])
            ax.set_xticks([])
            ax.set_yticks([])
            if col_idx == 0:
                metrics = example_metrics[row_idx]
                ax.set_ylabel(
                    f"sample {row_idx + 1}\nMAE={metrics['wrapped_phase_mae']:.3f}\nI-MSE={metrics['intensity_consistency_error']:.4f}",
                    rotation=0,
                    labelpad=42,
                    va="center",
                )
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)

    metrics_path = output_path.with_suffix(".json")
    save_json(metrics_path, {"num_examples": num_rows, "examples": example_metrics})
    return {"figure_path": str(output_path), "metrics_path": str(metrics_path), "num_examples": num_rows}


def wait_for_pid_exit(pid: int, poll_seconds: float = 60.0) -> None:
    while True:
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return
        time.sleep(poll_seconds)
