from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rul.config import load_config, save_config
from rul.losses import build_loss
from rul.metrics import regression_metrics
from rul.models import build_model
from rul.ncmapss import NCMAPSSPrepared, load_prepared_cache
from rul.preprocessing import RULWindowDataset, WindowData
from rul.training import training_objective, validation_risk_score
from rul.utils import count_parameters, get_device, measure_inference_time, save_json, set_seed
from run_formal_benchmark import FORMAL_BENCHMARK_MODELS, FORMAL_BENCHMARK_SEEDS


DEFAULT_CACHE = PROJECT_ROOT / "data" / "external" / "processed" / "ncmapss_ds02_smp100_win50.npz"


def make_loader(
    data: WindowData,
    *,
    sensor_indices: list[int],
    batch_size: int,
    shuffle: bool,
    augmentation: bool,
    seed: int,
) -> tuple[RULWindowDataset, DataLoader]:
    dataset = RULWindowDataset(
        data,
        sensor_indices=sensor_indices,
        augmentation_profile="missing_noise_drift" if augmentation else "none",
        noise_std=0.01 if augmentation else 0.0,
        missing_rate=0.10 if augmentation else 0.0,
        block_missing_rate=0.08 if augmentation else 0.0,
        drift_rate=0.03 if augmentation else 0.0,
        append_missing_mask=True,
        base_seed=seed if augmentation else None,
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=0)
    return dataset, loader


@torch.no_grad()
def predict(model: torch.nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    rows = []
    for x, _ in loader:
        rows.append(model(x.to(device)).cpu().numpy())
    return np.maximum(np.concatenate(rows), 0.0)


def unit_macro_metrics(data: WindowData, predictions: np.ndarray) -> dict[str, float]:
    rows = []
    for unit in np.unique(data.unit_ids):
        mask = data.unit_ids == unit
        rows.append(regression_metrics(data.y[mask], predictions[mask]))
    return {
        "unit_macro_rmse": float(np.mean([row["rmse"] for row in rows])),
        "unit_macro_mae": float(np.mean([row["mae"] for row in rows])),
        "unit_macro_nasa_per_window": float(
            np.mean([row["nasa_score"] / int(np.sum(data.unit_ids == unit)) for row, unit in zip(rows, np.unique(data.unit_ids))])
        ),
        "unit_macro_lpr30": float(np.mean([row["critical_30_late_prediction_ratio"] for row in rows])),
    }


def checkpoint_is_valid(run_dir: Path) -> bool:
    metrics_path = run_dir / "metrics.json"
    checkpoint_path = run_dir / "best_model.pt"
    if not metrics_path.exists() or not checkpoint_path.exists():
        return False
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001
        return False
    return (
        metrics.get("result_status") in {"external_validation", "exploratory_dev_unit_rotation"}
        and metrics.get("dataset") == "N-CMAPSS_DS02"
        and int(metrics.get("planned_epochs", 0)) >= 80
        and isinstance(checkpoint, dict)
        and "model_state" in checkpoint
    )


def train_one(
    prepared: NCMAPSSPrepared,
    *,
    model_name: str,
    seed: int,
    run_dir: Path,
    epochs: int,
    patience: int,
    batch_size: int,
    device: torch.device,
    protocol_config: dict[str, object],
    experiment_name: str,
    result_status: str = "external_validation",
) -> dict[str, object]:
    set_seed(seed)
    run_dir.mkdir(parents=True, exist_ok=True)
    train_dataset, train_loader = make_loader(
        prepared.train,
        sensor_indices=prepared.sensor_indices,
        batch_size=batch_size,
        shuffle=True,
        augmentation=True,
        seed=seed,
    )
    _, val_loader = make_loader(
        prepared.val,
        sensor_indices=prepared.sensor_indices,
        batch_size=batch_size * 2,
        shuffle=False,
        augmentation=False,
        seed=seed,
    )
    _, test_loader = make_loader(
        prepared.test,
        sensor_indices=prepared.sensor_indices,
        batch_size=batch_size * 2,
        shuffle=False,
        augmentation=False,
        seed=seed,
    )
    input_features = len(prepared.feature_names) + len(prepared.sensor_indices)
    model_cfg = protocol_config["model"]
    train_cfg = protocol_config["training"]
    model = build_model(
        model_name,
        input_features=input_features,
        window_size=int(prepared.metadata["window_size"]),
        hidden_size=int(model_cfg["hidden_size"]),
        tcn_channels=list(model_cfg["tcn_channels"]),
        kernel_size=int(model_cfg["kernel_size"]),
        dropout=float(model_cfg["dropout"]),
        use_uncertainty_head=bool(model_cfg.get("use_uncertainty_head", True)),
        mask_feature_count=len(prepared.sensor_indices),
    ).to(device)
    criterion = build_loss(
        str(train_cfg["loss"]),
        float(train_cfg["huber_delta"]),
        float(train_cfg["late_life_threshold"]),
        float(train_cfg["late_life_weight"]),
        float(train_cfg["late_over_weight"]),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    checkpoint_path = run_dir / "best_model.pt"
    history = []
    best_score = float("inf")
    best_epoch = 0
    stale = 0
    completed = 0

    for epoch in range(1, epochs + 1):
        completed = epoch
        train_dataset.set_epoch(epoch - 1)
        model.train()
        total_loss = 0.0
        total_samples = 0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, _ = training_objective(model, x, y, criterion, train_cfg, prepared.sensor_indices)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(y)
            total_samples += len(y)
        val_predictions = predict(model, val_loader, device)
        val_metrics = regression_metrics(prepared.val.y, val_predictions)
        val_rmse = float(val_metrics["rmse"])
        val_score = validation_risk_score(val_metrics, train_cfg, 100.0)
        history.append(
            {
                "epoch": epoch,
                "train_loss": total_loss / max(total_samples, 1),
                "val_rmse": val_rmse,
                "val_lpr30": val_metrics["critical_30_late_prediction_ratio"],
                "val_slpr30_10": val_metrics["critical_30_severe_late_10_ratio"],
                "val_risk_score": val_score,
            }
        )
        print(
            f"[N-CMAPSS] model={model_name} seed={seed} epoch={epoch}/{epochs} "
            f"train_loss={history[-1]['train_loss']:.4f} val_rmse={val_rmse:.4f} "
            f"val_risk={val_score:.4f} best={best_score:.4f}",
            flush=True,
        )
        if val_score < best_score - 1e-8:
            best_score = val_score
            best_epoch = epoch
            stale = 0
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "best_epoch": best_epoch,
                    "best_val_rmse": val_rmse,
                    "best_val_risk_score": best_score,
                    "model": model_name,
                    "seed": seed,
                },
                checkpoint_path,
            )
        else:
            stale += 1
            if stale >= patience:
                break

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    val_predictions = predict(model, val_loader, device)
    test_predictions = predict(model, test_loader, device)
    val_metrics = regression_metrics(prepared.val.y, val_predictions)
    test_metrics = regression_metrics(prepared.test.y, test_predictions)
    macro = unit_macro_metrics(prepared.test, test_predictions)
    sample = next(iter(test_loader))[0]
    gpu_ms = measure_inference_time(model, sample[:1], device, warmup=5, repeats=30)

    metrics: dict[str, object] = {
        "dataset": "N-CMAPSS_DS02",
        "experiment_name": experiment_name,
        "subset": "DS02",
        "model": model_name,
        "seed": seed,
        "result_status": result_status,
        "allowed_for_paper": True,
        "planned_epochs": epochs,
        "completed_epochs": completed,
        "best_epoch": best_epoch,
        "checkpoint_selection_metric": "val_risk_score",
        "best_val_rmse": checkpoint["best_val_rmse"],
        "best_val_risk_score": best_score,
        "training_complete_reason": "early_stopped" if completed < epochs else "max_epochs",
        "device": str(device),
        "parameters": count_parameters(model),
        "single_sample_inference_ms": gpu_ms,
        "sampling": prepared.metadata["sampling"],
        "window_size": prepared.metadata["window_size"],
        "stride": prepared.metadata["stride"],
        "condition_clusters": prepared.metadata["condition_clusters"],
        "train_units": prepared.metadata["train_units"],
        "validation_units": prepared.metadata["validation_units"],
        "test_units": prepared.metadata["test_units"],
        "n_train_windows": len(prepared.train.y),
        "n_validation_windows": len(prepared.val.y),
        "n_test_windows": len(prepared.test.y),
        "mask_channels": True,
        "augmentation_profile": "missing_noise_drift",
        "loss": "symmetric_late_life_weighted_huber" if float(train_cfg["late_over_weight"]) == 1.0 else "asymmetric_weighted_huber",
        "late_life_weight": float(train_cfg["late_life_weight"]),
        "late_over_weight": float(train_cfg["late_over_weight"]),
        "protocol_name": (
            "paper_main_v3_external"
            if result_status == "external_validation"
            else "round12_ncmapss_computational_proxy"
        ),
        "protocol_version": "3.0",
        **{f"val_{key}": value for key, value in val_metrics.items()},
        **{f"test_{key}": value for key, value in test_metrics.items()},
        "test_nasa_score_per_window": float(test_metrics["nasa_score"] / len(prepared.test.y)),
        **macro,
    }
    save_json(metrics, run_dir / "metrics.json")
    save_config(protocol_config, run_dir / "run_config.yaml")
    pd.DataFrame(history).to_csv(run_dir / "epoch_history.csv", index=False)
    pd.DataFrame(
        {
            "unit_id": prepared.test.unit_ids,
            "end_index": prepared.test.end_cycles,
            "true_rul": prepared.test.y,
            "pred_rul": test_predictions,
        }
    ).to_csv(run_dir / "test_predictions.csv", index=False)
    return metrics


def write_summary(rows: list[dict[str, object]], output: Path, *, prefix: str = "ncmapss_ds02") -> None:
    frame = pd.DataFrame(rows)
    frame.to_csv(output / f"{prefix}_runs.csv", index=False)
    numeric = [
        "test_rmse",
        "test_mae",
        "test_nasa_score_per_window",
        "test_critical_30_late_prediction_ratio",
        "unit_macro_rmse",
        "unit_macro_mae",
        "unit_macro_nasa_per_window",
        "unit_macro_lpr30",
        "parameters",
        "single_sample_inference_ms",
    ]
    summary = frame.groupby("model", as_index=False).agg(
        seed_count=("seed", "nunique"),
        **{f"{column}_mean": (column, "mean") for column in numeric},
        **{f"{column}_std": (column, lambda x: x.std(ddof=1)) for column in numeric},
    )
    summary.to_csv(output / f"{prefix}_summary.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the locked model suite on the exploratory N-CMAPSS DS02 task.")
    parser.add_argument("--cache", default=str(DEFAULT_CACHE))
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "paper_main.yaml"))
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--models", nargs="+", default=FORMAL_BENCHMARK_MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=FORMAL_BENCHMARK_SEEDS)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--experiment-prefix", default="ncmapss_ds02")
    parser.add_argument("--output-prefix", default="ncmapss_ds02")
    parser.add_argument("--late-over-weight", type=float)
    parser.add_argument("--rerun-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    total = len(args.models) * len(args.seeds)
    print(f"[N-CMAPSS GRID] models={len(args.models)} seeds={len(args.seeds)} total={total}")
    if args.dry_run:
        print("NCMAPSS_GRID_DRY_RUN_PASS")
        return
    cache_path = Path(args.cache)
    if not cache_path.exists():
        raise FileNotFoundError(f"Missing prepared cache: {cache_path}. Run scripts/prepare_ncmapss_cache.py first.")
    prepared = load_prepared_cache(cache_path)
    protocol_config = copy.deepcopy(load_config(args.config))
    if args.late_over_weight is not None:
        protocol_config["training"]["late_over_weight"] = float(args.late_over_weight)
    device = get_device(args.device)
    results_root = Path(args.results_dir)
    rows = []
    index = 0
    for seed in args.seeds:
        for model_name in args.models:
            index += 1
            experiment_name = f"{args.experiment_prefix}_seed{seed}"
            run_dir = results_root / experiment_name / "DS02" / model_name
            if not args.rerun_complete and args.epochs == 80 and checkpoint_is_valid(run_dir):
                rows.append(json.loads((run_dir / "metrics.json").read_text(encoding="utf-8-sig")))
                print(f"[N-CMAPSS SKIP {index}/{total}] model={model_name} seed={seed}")
                continue
            print(f"[N-CMAPSS RUN {index}/{total}] model={model_name} seed={seed}", flush=True)
            rows.append(
                train_one(
                    prepared,
                    model_name=model_name,
                    seed=seed,
                    run_dir=run_dir,
                    epochs=args.epochs,
                    patience=args.patience,
                    batch_size=args.batch_size,
                    device=device,
                    protocol_config=protocol_config,
                    experiment_name=experiment_name,
                )
            )
    output = PROJECT_ROOT / "paper_outputs" / "advanced_evidence"
    output.mkdir(parents=True, exist_ok=True)
    write_summary(rows, output, prefix=args.output_prefix)
    print(f"NCMAPSS_BENCHMARK_READY {output}")


if __name__ == "__main__":
    main()
