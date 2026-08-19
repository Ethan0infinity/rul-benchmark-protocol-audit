from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from .augmentations import (
    apply_block_missing_with_mask,
    apply_burst_noise,
    apply_correlated_group_missing_with_mask,
    apply_gaussian_noise,
    apply_sensor_bias_shift,
    apply_sensor_drift,
    apply_sensor_missing_with_mask,
    apply_stuck_at_fault,
    apply_window_random_missing_with_mask,
)
from .config import load_config
from .metrics import regression_metrics
from .models import build_model
from .preprocessing import WindowData
from .training import evaluate_window_data, prepare_data, train_model
from .utils import get_device, load_json, save_json


def run_experiment_grid(config: dict[str, Any], subsets: list[str], models: list[str]) -> list[dict[str, Any]]:
    rows = []
    for subset in subsets:
        for model_name in models:
            rows.append(train_model(config, subset=subset, model_name=model_name))
    return rows


def _load_run(run_dir: str | Path):
    run_dir = Path(run_dir)
    config = load_config(run_dir / "run_config.yaml")
    metrics = load_json(run_dir / "metrics.json")
    subset = metrics["subset"]
    model_name = metrics["model"]
    prepared = prepare_data(config, subset, run_dir=None)
    device = get_device(config["training"].get("device", "auto"))
    model_cfg = config["model"]
    model = build_model(
        model_name,
        input_features=len(prepared.input_features),
        window_size=int(config["data"].get("window_size", 30)),
        hidden_size=int(model_cfg.get("hidden_size", 64)),
        tcn_channels=list(model_cfg.get("tcn_channels", [48, 64])),
        kernel_size=int(model_cfg.get("kernel_size", 3)),
        dropout=float(model_cfg.get("dropout", 0.1)),
        use_reliability_gate=bool(model_cfg.get("use_reliability_gate", True)),
        use_channel_gate=bool(model_cfg.get("use_channel_gate", True)),
        use_local_trend=bool(model_cfg.get("use_local_trend", True)),
        use_temporal_attention=bool(model_cfg.get("use_temporal_attention", True)),
        mask_feature_count=len(prepared.sensor_indices) if bool(config.get("data", {}).get("append_missing_mask", False)) else 0,
    ).to(device)
    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    model.eval()
    return config, metrics, prepared, model, device


def add_relative_degradation(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    clean = next((row for row in rows if row.get("scenario") == "clean"), None)
    if clean is None:
        return rows
    clean_rmse = float(clean.get("rmse", 0.0) or 0.0)
    clean_nasa = float(clean.get("nasa_score", 0.0) or 0.0)
    enriched = []
    for row in rows:
        item = dict(row)
        rmse = float(item.get("rmse", 0.0) or 0.0)
        nasa = float(item.get("nasa_score", 0.0) or 0.0)
        item["relative_rmse_increase"] = (rmse - clean_rmse) / clean_rmse if clean_rmse else 0.0
        item["relative_nasa_score_increase"] = (nasa - clean_nasa) / clean_nasa if clean_nasa else 0.0
        enriched.append(item)
    return enriched


def evaluate_robustness(
    run_dir: str | Path,
    *,
    perturbation_seed: int = 42,
    save: bool = True,
    engine_rows: list[dict[str, Any]] | None = None,
    robustness_override: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    run_dir = Path(run_dir)
    config, metrics, prepared, model, device = _load_run(run_dir)
    batch_size = int(config["training"].get("batch_size", 128))
    robust_cfg = robustness_override if robustness_override is not None else config.get("robustness", {})
    append_missing_mask = bool(config.get("data", {}).get("append_missing_mask", False))
    rows: list[dict[str, Any]] = []

    def append_engine_rows(scenario: str, level: float, data: WindowData, predictions: np.ndarray) -> None:
        if engine_rows is None:
            return
        for unit_id, true_rul, pred_rul in zip(data.unit_ids, data.y, predictions):
            error = float(pred_rul - true_rul)
            engine_rows.append(
                {
                    "scenario": scenario,
                    "level": float(level),
                    "training_seed": int(metrics.get("seed", config.get("project", {}).get("seed", 42))),
                    "perturbation_seed": int(perturbation_seed),
                    "subset": metrics["subset"],
                    "model": metrics["model"],
                    "unit_id": int(unit_id),
                    "true_rul": float(true_rul),
                    "pred_rul": float(pred_rul),
                    "error": error,
                    "squared_error": error * error,
                    "critical_30": int(float(true_rul) <= 30.0),
                    "late_indicator_30": int(float(true_rul) <= 30.0 and error > 0.0),
                }
            )

    base_metrics, base_predictions = evaluate_window_data(
        model,
        prepared.test,
        device,
        batch_size=batch_size,
        sensor_indices=prepared.sensor_indices,
        append_missing_mask=append_missing_mask,
    )
    metadata = {
        "training_seed": int(metrics.get("seed", config.get("project", {}).get("seed", 42))),
        "perturbation_seed": int(perturbation_seed),
        "subset": metrics["subset"],
        "model": metrics["model"],
    }
    rows.append({"scenario": "clean", "level": 0.0, **metadata, **base_metrics})
    append_engine_rows("clean", 0.0, prepared.test, base_predictions)

    for level in robust_cfg.get("noise_levels", []):
        level = float(level)
        if level <= 0:
            continue
        x = apply_gaussian_noise(prepared.test.x, level, prepared.sensor_indices, seed=perturbation_seed)
        wd = WindowData(x=x, y=prepared.test.y, unit_ids=prepared.test.unit_ids, end_cycles=prepared.test.end_cycles)
        row_metrics, predictions = evaluate_window_data(
            model,
            wd,
            device,
            batch_size=batch_size,
            sensor_indices=prepared.sensor_indices,
            append_missing_mask=append_missing_mask,
        )
        rows.append({"scenario": "gaussian_noise", "level": level, **metadata, **row_metrics})
        append_engine_rows("gaussian_noise", level, wd, predictions)

    for level in robust_cfg.get("missing_rates", []):
        level = float(level)
        if level <= 0:
            continue
        x, observed_mask = apply_sensor_missing_with_mask(
            prepared.test.x, level, prepared.sensor_indices, seed=perturbation_seed
        )
        wd = WindowData(
            x=x,
            y=prepared.test.y,
            unit_ids=prepared.test.unit_ids,
            end_cycles=prepared.test.end_cycles,
            observed_mask=observed_mask,
        )
        row_metrics, predictions = evaluate_window_data(
            model,
            wd,
            device,
            batch_size=batch_size,
            sensor_indices=prepared.sensor_indices,
            append_missing_mask=append_missing_mask,
        )
        rows.append({"scenario": "global_sensor_missing", "level": level, **metadata, **row_metrics})
        append_engine_rows("global_sensor_missing", level, wd, predictions)

        x, observed_mask = apply_window_random_missing_with_mask(
            prepared.test.x, level, prepared.sensor_indices, seed=perturbation_seed
        )
        wd = WindowData(
            x=x,
            y=prepared.test.y,
            unit_ids=prepared.test.unit_ids,
            end_cycles=prepared.test.end_cycles,
            observed_mask=observed_mask,
        )
        row_metrics, predictions = evaluate_window_data(
            model,
            wd,
            device,
            batch_size=batch_size,
            sensor_indices=prepared.sensor_indices,
            append_missing_mask=append_missing_mask,
        )
        rows.append({"scenario": "window_random_missing", "level": level, **metadata, **row_metrics})
        append_engine_rows("window_random_missing", level, wd, predictions)

    for level in robust_cfg.get("block_missing_rates", []):
        level = float(level)
        if level <= 0:
            continue
        x, observed_mask = apply_block_missing_with_mask(
            prepared.test.x, level, prepared.sensor_indices, seed=perturbation_seed
        )
        wd = WindowData(
            x=x,
            y=prepared.test.y,
            unit_ids=prepared.test.unit_ids,
            end_cycles=prepared.test.end_cycles,
            observed_mask=observed_mask,
        )
        row_metrics, predictions = evaluate_window_data(
            model,
            wd,
            device,
            batch_size=batch_size,
            sensor_indices=prepared.sensor_indices,
            append_missing_mask=append_missing_mask,
        )
        rows.append({"scenario": "block_missing", "level": level, **metadata, **row_metrics})
        append_engine_rows("block_missing", level, wd, predictions)

    for level in robust_cfg.get("drift_rates", []):
        level = float(level)
        if level <= 0:
            continue
        x = apply_sensor_drift(prepared.test.x, level, prepared.sensor_indices, seed=perturbation_seed)
        wd = WindowData(x=x, y=prepared.test.y, unit_ids=prepared.test.unit_ids, end_cycles=prepared.test.end_cycles)
        row_metrics, predictions = evaluate_window_data(
            model,
            wd,
            device,
            batch_size=batch_size,
            sensor_indices=prepared.sensor_indices,
            append_missing_mask=append_missing_mask,
        )
        rows.append({"scenario": "sensor_drift", "level": level, **metadata, **row_metrics})
        append_engine_rows("sensor_drift", level, wd, predictions)

    def evaluate_array(scenario: str, level: float, x: np.ndarray, observed_mask: np.ndarray | None = None) -> None:
        wd = WindowData(
            x=x,
            y=prepared.test.y,
            unit_ids=prepared.test.unit_ids,
            end_cycles=prepared.test.end_cycles,
            observed_mask=observed_mask,
        )
        row_metrics, predictions = evaluate_window_data(
            model,
            wd,
            device,
            batch_size=batch_size,
            sensor_indices=prepared.sensor_indices,
            append_missing_mask=append_missing_mask,
        )
        rows.append({"scenario": scenario, "level": float(level), **metadata, **row_metrics})
        append_engine_rows(scenario, level, wd, predictions)

    for level in robust_cfg.get("bias_levels", []):
        level = float(level)
        if level > 0:
            evaluate_array(
                "sensor_bias_shift",
                level,
                apply_sensor_bias_shift(prepared.test.x, level, prepared.sensor_indices, seed=perturbation_seed),
            )

    for level in robust_cfg.get("stuck_at_rates", []):
        level = float(level)
        if level > 0:
            evaluate_array(
                "stuck_at_fault",
                level,
                apply_stuck_at_fault(prepared.test.x, level, prepared.sensor_indices, seed=perturbation_seed),
            )

    for level in robust_cfg.get("burst_noise_levels", []):
        level = float(level)
        if level > 0:
            evaluate_array(
                "burst_noise",
                level,
                apply_burst_noise(prepared.test.x, level, prepared.sensor_indices, seed=perturbation_seed),
            )

    for level in robust_cfg.get("correlated_missing_rates", []):
        level = float(level)
        if level <= 0:
            continue
        x, observed_mask = apply_correlated_group_missing_with_mask(
            prepared.test.x,
            level,
            prepared.sensor_indices,
            seed=perturbation_seed,
        )
        evaluate_array("correlated_group_missing", level, x, observed_mask)

    rows = add_relative_degradation(rows)
    if save:
        suffix = "" if perturbation_seed == 42 else f"_seed{perturbation_seed}"
        save_json(
            {
                "run": str(run_dir),
                "subset": metrics["subset"],
                "model": metrics["model"],
                "perturbation_seed": perturbation_seed,
                "rows": rows,
            },
            run_dir / f"robustness{suffix}.json",
        )
    return rows


def metrics_rows_to_csv(rows: list[dict[str, Any]], path: str | Path) -> None:
    import csv

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = sorted({key for row in rows for key in row.keys()})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)
