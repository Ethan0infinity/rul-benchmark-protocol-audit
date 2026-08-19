from __future__ import annotations

import csv
import copy
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .cmapss import load_subset
from .config import project_path, save_config
from .features import feature_summary, select_features, sensor_indices_in_features
from .losses import SmoothLateRiskLoss, build_loss, pinball_loss
from .metrics import monotonicity_metrics, regression_metrics
from .models import build_model
from .preprocessing import (
    RULWindowDataset,
    WindowData,
    fit_transform_train_val_test,
    make_simulated_last_windows,
    make_validation_endpoint_windows,
    make_windows,
    split_units,
)
from .progress import (
    RunProgressContext,
    append_epoch_history_csv,
    format_epoch_status,
    format_run_finish,
    format_run_start,
    write_progress_json,
)
from .protocol import classify_result_status, load_protocol_lock
from .utils import count_parameters, ensure_dir, get_device, measure_inference_time, profile_forward_flops, save_json, set_seed


@dataclass
class PreparedData:
    train: WindowData
    val: WindowData
    val_last: WindowData
    test: WindowData
    features: list[str]
    input_features: list[str]
    selected_sensors: list[str]
    sensor_indices: list[int]
    scaler: object
    feature_scores: pd.DataFrame
    validation_endpoint_sets: dict[str, WindowData] | None = None


def _resolve_data_root(config: dict[str, Any]) -> Path:
    data_root = Path(config["data"]["root"])
    if data_root.is_absolute():
        return data_root
    return project_path(config, *data_root.parts)


def checkpoint_score_from_metrics(metrics: dict[str, float], selection_metric: str) -> float:
    metric = selection_metric.lower()
    aliases = {
        "val_loss_all_windows": "val_loss",
        "val_all_loss": "val_loss",
        "val_last_loss": "val_last_loss",
        "val_last_rmse": "val_last_rmse",
        "val_last_nasa_score": "val_last_nasa_score",
    }
    key = aliases.get(metric, selection_metric)
    if key not in metrics:
        raise KeyError(f"Selection metric '{selection_metric}' was not computed. Available keys: {sorted(metrics)}")
    return float(metrics[key])


def validation_risk_score(metrics: dict[str, float], train_cfg: dict[str, Any], rul_cap: float) -> float:
    """Predeclared dimensionless accuracy-risk checkpoint criterion."""
    rmse_term = float(metrics["rmse"]) / max(float(rul_cap), 1.0)
    lpr_term = float(metrics["critical_30_late_prediction_ratio"])
    severe_term = float(metrics["critical_30_severe_late_10_ratio"])
    return (
        rmse_term
        + float(train_cfg.get("selection_lpr_weight", 0.25)) * lpr_term
        + float(train_cfg.get("selection_severe_late_weight", 0.25)) * severe_term
    )


def model_feature_metadata(model_name: str, model_cfg: dict[str, Any], append_missing_mask: bool) -> dict[str, bool]:
    """Report architecture features from the selected model, not global config defaults."""
    name = model_name.lower()
    proposed = name in {"rast_gru_v2", "rast_gru_pp", "ocm_rast_gru"}
    reliability = name in {"rast_gru", "ra_tcn_gru", "rast_tcn_gru"} or (
        proposed and bool(model_cfg.get("use_reliability_gate", True))
    )
    channel_gate = name in {
        "rs_tcn_gru",
        "rast_gru",
        "ra_tcn_gru",
        "rast_tcn_gru",
        "dual_attention_tcn",
        "da_tcn",
        "regime_dual_attention_cnn_gru",
        "regime_da_cnn_gru",
    } or (proposed and bool(model_cfg.get("use_channel_gate", True)))
    temporal_attention = name in {
        "attention_gru",
        "att_gru",
        "bigru_attention",
        "attention_bigru",
        "bi_gru_attention",
        "transformer_lite",
        "lite_transformer",
        "dual_attention_tcn",
        "da_tcn",
        "regime_dual_attention_cnn_gru",
        "regime_da_cnn_gru",
        "sensor_graph_gru",
        "sg_gru",
        "quantile_gru",
        "probabilistic_gru",
    } or (proposed and bool(model_cfg.get("use_temporal_attention", True)))
    return {
        "model_use_reliability_gate": reliability,
        "model_use_mask_aware_reliability_gate": proposed and reliability and append_missing_mask,
        "model_use_channel_gate": channel_gate,
        "model_use_local_trend": proposed and bool(model_cfg.get("use_local_trend", True)),
        "model_use_temporal_attention": temporal_attention,
        "model_use_uncertainty_head": name in {"quantile_gru", "probabilistic_gru"}
        or (proposed and bool(model_cfg.get("use_uncertainty_head", True))),
    }


def training_objective(
    model: torch.nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    criterion: torch.nn.Module,
    train_cfg: dict[str, Any],
    sensor_indices: list[int],
) -> tuple[torch.Tensor, dict[str, float]]:
    if hasattr(model, "forward_quantiles"):
        lower, pred, upper = model.forward_quantiles(x)
    else:
        pred = model(x)
        lower = upper = None

    base = criterion(pred, y)
    total = base
    components = {"base": float(base.detach())}

    if bool(getattr(model, "supports_risk_regularization", False)) or bool(
        train_cfg.get("allow_shared_risk_regularization", False)
    ):
        risk_weight = float(train_cfg.get("smooth_late_risk_weight", 0.0))
        if risk_weight > 0:
            risk = SmoothLateRiskLoss(
                threshold=float(train_cfg.get("late_life_threshold", 30.0)),
                temperature=float(train_cfg.get("smooth_late_risk_temperature", 2.0)),
            )(pred, y)
            total = total + risk_weight * risk
            components["smooth_late_risk"] = float(risk.detach())

        reliability_weight = float(train_cfg.get("reliability_supervision_weight", 0.0))
        if reliability_weight > 0 and hasattr(model, "reliability_supervision_loss"):
            reliability = model.reliability_supervision_loss(x)
            total = total + reliability_weight * reliability
            components["reliability_supervision"] = float(reliability.detach())

        consistency_weight = float(train_cfg.get("consistency_weight", 0.0))
        consistency_noise_std = float(train_cfg.get("consistency_noise_std", 0.0))
        if consistency_weight > 0 and consistency_noise_std > 0 and sensor_indices:
            second_view = x.clone()
            second_view[:, :, sensor_indices] = second_view[:, :, sensor_indices] + torch.randn_like(
                second_view[:, :, sensor_indices]
            ) * consistency_noise_std
            second_pred = model(second_view)
            consistency = torch.nn.functional.smooth_l1_loss(second_pred, pred.detach())
            total = total + consistency_weight * consistency
            components["consistency"] = float(consistency.detach())

    quantile_weight = float(train_cfg.get("quantile_calibration_weight", 0.0))
    if quantile_weight > 0 and lower is not None and upper is not None:
        calibration = pinball_loss(lower, y, 0.1) + pinball_loss(upper, y, 0.9)
        total = total + quantile_weight * calibration
        components["quantile_calibration"] = float(calibration.detach())

    components["total"] = float(total.detach())
    return total, components


def prepare_data(config: dict[str, Any], subset: str, run_dir: str | Path | None = None) -> PreparedData:
    data_cfg = config["data"]
    train_df, test_df = load_subset(_resolve_data_root(config), subset, data_cfg.get("rul_cap", 125))
    project_seed = int(config["project"].get("seed", 42))
    seed = int(config["project"].get("split_seed", project_seed))
    train_units, val_units = split_units(
        train_df["unit_id"].to_numpy(),
        float(data_cfg.get("validation_split", 0.2)),
        seed,
    )
    train_core_df = train_df[train_df["unit_id"].isin(train_units)].copy()
    val_df = train_df[train_df["unit_id"].isin(val_units)].copy()

    score_path = Path(run_dir) / "feature_scores.csv" if run_dir is not None else None
    selected = select_features(
        train_core_df,
        feature_mode=data_cfg.get("feature_mode", "selected"),
        max_selected_sensors=int(data_cfg.get("max_selected_sensors", 14)),
        include_settings=bool(data_cfg.get("include_settings", True)),
        output_csv=score_path,
    )
    train_scaled, val_scaled, test_scaled, scaler = fit_transform_train_val_test(
        train_core_df,
        val_df,
        test_df,
        selected.features,
        scaler_name=data_cfg.get("scaler", "standard"),
        subset=subset,
        condition_clusters=data_cfg.get("condition_clusters"),
        random_state=seed,
    )
    train_windows = make_windows(
        train_scaled,
        selected.features,
        int(data_cfg.get("window_size", 30)),
        int(data_cfg.get("stride", 1)),
        last_only=False,
    )
    val_windows = make_windows(
        val_scaled,
        selected.features,
        int(data_cfg.get("window_size", 30)),
        int(data_cfg.get("stride", 1)),
        last_only=False,
    )
    validation_last_strategy = str(data_cfg.get("validation_last_strategy", "simulated")).lower()
    if validation_last_strategy in {"failure", "run_to_failure", "physical_last"}:
        val_last_windows = make_windows(
            val_scaled,
            selected.features,
            int(data_cfg.get("window_size", 30)),
            int(data_cfg.get("stride", 1)),
            last_only=True,
        )
    else:
        val_last_windows = make_simulated_last_windows(
            val_scaled,
            selected.features,
            int(data_cfg.get("window_size", 30)),
            seed=seed,
            min_rul=float(data_cfg.get("validation_last_min_rul", 1.0)),
            max_rul=data_cfg.get("validation_last_max_rul", data_cfg.get("rul_cap", 125)),
        )
    test_windows = make_windows(
        test_scaled,
        selected.features,
        int(data_cfg.get("window_size", 30)),
        int(data_cfg.get("stride", 1)),
        last_only=True,
    )
    sensor_indices = sensor_indices_in_features(selected.features)
    append_mask = bool(data_cfg.get("append_missing_mask", False))
    input_features = list(selected.features)
    if append_mask:
        input_features.extend(f"{selected.features[i]}_observed" for i in sensor_indices)
    validation_endpoint_sets = None
    if bool(data_cfg.get("validation_endpoint_confirmation", False)):
        validation_endpoint_sets = {
            "uniform_single": val_last_windows,
            "fixed_multi": make_validation_endpoint_windows(
                val_scaled,
                selected.features,
                int(data_cfg.get("window_size", 30)),
                strategy="fixed_multi",
                seed=seed,
                rul_cap=float(data_cfg.get("rul_cap", 125.0)),
            ),
            "critical_stratified": make_validation_endpoint_windows(
                val_scaled,
                selected.features,
                int(data_cfg.get("window_size", 30)),
                strategy="critical_stratified",
                seed=seed,
                rul_cap=float(data_cfg.get("rul_cap", 125.0)),
            ),
        }
    return PreparedData(
        train=train_windows,
        val=val_windows,
        val_last=val_last_windows,
        test=test_windows,
        features=selected.features,
        input_features=input_features,
        selected_sensors=selected.selected_sensors,
        sensor_indices=sensor_indices,
        scaler=scaler,
        feature_scores=selected.scores,
        validation_endpoint_sets=validation_endpoint_sets,
    )


def _make_loader(
    data: WindowData,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    sensor_indices: list[int],
    sensor_dropout_prob: float = 0.0,
    sensor_dropout_fraction: float = 0.0,
    noise_std: float = 0.0,
    augmentation_profile: str = "none",
    missing_rate: float = 0.0,
    block_missing_rate: float = 0.0,
    drift_rate: float = 0.0,
    mechanism_apply_probability: float = 1.0,
    corrupted_window_probability: float = 1.0,
    append_missing_mask: bool = False,
    base_seed: int | None = None,
    shuffle_seed: int | None = None,
) -> DataLoader:
    dataset = RULWindowDataset(
        data,
        sensor_indices=sensor_indices,
        sensor_dropout_prob=sensor_dropout_prob,
        sensor_dropout_fraction=sensor_dropout_fraction,
        noise_std=noise_std,
        augmentation_profile=augmentation_profile,
        missing_rate=missing_rate,
        block_missing_rate=block_missing_rate,
        drift_rate=drift_rate,
        mechanism_apply_probability=mechanism_apply_probability,
        corrupted_window_probability=corrupted_window_probability,
        append_missing_mask=append_missing_mask,
        base_seed=base_seed,
    )
    generator = None
    if shuffle and shuffle_seed is not None:
        generator = torch.Generator()
        generator.manual_seed(int(shuffle_seed))
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
    )


@torch.no_grad()
def predict(
    model: torch.nn.Module,
    data: WindowData,
    device: torch.device,
    batch_size: int = 256,
    sensor_indices: list[int] | None = None,
    append_missing_mask: bool = False,
) -> np.ndarray:
    loader = DataLoader(
        RULWindowDataset(data, sensor_indices=sensor_indices, append_missing_mask=append_missing_mask),
        batch_size=batch_size,
        shuffle=False,
    )
    preds = []
    model.eval()
    for x, _ in loader:
        out = model(x.to(device)).detach().cpu().numpy()
        preds.append(out)
    return np.concatenate(preds)


def evaluate_window_data(
    model: torch.nn.Module,
    data: WindowData,
    device: torch.device,
    batch_size: int = 256,
    sensor_indices: list[int] | None = None,
    append_missing_mask: bool = False,
) -> tuple[dict[str, float], np.ndarray]:
    preds = predict(
        model,
        data,
        device,
        batch_size=batch_size,
        sensor_indices=sensor_indices,
        append_missing_mask=append_missing_mask,
    )
    preds = np.maximum(preds, 0.0)
    return regression_metrics(data.y, preds), preds


@torch.no_grad()
def evaluate_prediction_intervals(
    model: torch.nn.Module,
    data: WindowData,
    device: torch.device,
    batch_size: int,
    sensor_indices: list[int],
    append_missing_mask: bool,
) -> tuple[dict[str, float], np.ndarray] | None:
    if not hasattr(model, "forward_quantiles"):
        return None
    loader = DataLoader(
        RULWindowDataset(data, sensor_indices=sensor_indices, append_missing_mask=append_missing_mask),
        batch_size=batch_size,
        shuffle=False,
    )
    rows = []
    model.eval()
    for x, _ in loader:
        lower, median, upper = model.forward_quantiles(x.to(device))
        rows.append(torch.stack([lower, median, upper], dim=-1).detach().cpu().numpy())
    intervals = np.concatenate(rows, axis=0)
    intervals[:, 0] = np.maximum(intervals[:, 0], 0.0)
    intervals[:, 1] = np.maximum(intervals[:, 1], 0.0)
    intervals[:, 2] = np.maximum(intervals[:, 2], intervals[:, 1])
    covered = (data.y >= intervals[:, 0]) & (data.y <= intervals[:, 2])
    metrics = {
        "interval_80_coverage": float(np.mean(covered)),
        "interval_80_mean_width": float(np.mean(intervals[:, 2] - intervals[:, 0])),
        "interval_80_lower_miss_rate": float(np.mean(data.y < intervals[:, 0])),
        "interval_80_upper_miss_rate": float(np.mean(data.y > intervals[:, 2])),
    }
    return metrics, intervals


def _write_predictions(path: Path, data: WindowData, preds: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["unit_id", "end_cycle", "true_rul", "pred_rul"])
        for unit, cycle, true, pred in zip(data.unit_ids, data.end_cycles, data.y, preds):
            writer.writerow([int(unit), int(cycle), float(true), float(pred)])


def _write_interval_predictions(path: Path, data: WindowData, intervals: np.ndarray) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["unit_id", "end_cycle", "true_rul", "q10_rul", "q50_rul", "q90_rul"])
        for unit, cycle, true, row in zip(data.unit_ids, data.end_cycles, data.y, intervals):
            writer.writerow([int(unit), int(cycle), float(true), *[float(value) for value in row]])


def train_model(
    config: dict[str, Any],
    subset: str | None = None,
    model_name: str | None = None,
    run_index: int | None = None,
    total_runs: int | None = None,
) -> dict[str, Any]:
    subset = subset or config["data"].get("subset", "FD001")
    model_name = model_name or config["model"].get("name", "rs_tcn_gru")
    project_seed = int(config["project"].get("seed", 42))
    split_seed = int(config["project"].get("split_seed", project_seed))
    initialization_seed = int(config["project"].get("initialization_seed", project_seed))
    shuffle_seed_value = config["project"].get("shuffle_seed")
    shuffle_seed = None if shuffle_seed_value is None else int(shuffle_seed_value)
    set_seed(initialization_seed)
    deterministic_cudnn = bool(config.get("training", {}).get("deterministic_cudnn", True))
    torch.backends.cudnn.deterministic = deterministic_cudnn
    torch.backends.cudnn.benchmark = not deterministic_cudnn

    results_root = project_path(config, config["project"].get("results_dir", "results"))
    experiment_name = config.get("project", {}).get("experiment_name")
    run_dir = ensure_dir(results_root / experiment_name / subset / model_name) if experiment_name else ensure_dir(results_root / subset / model_name)
    protocol_lock = load_protocol_lock()
    protocol_metadata = classify_result_status(experiment_name or "default", run_dir, protocol_lock)
    preprocessing_start = time.perf_counter()
    prepared = prepare_data(config, subset, run_dir=run_dir)
    preprocessing_elapsed_sec = time.perf_counter() - preprocessing_start

    data_cfg = config["data"]
    train_cfg = config["training"]
    aug_cfg = config.get("augmentation", {})
    device = get_device(train_cfg.get("device", "auto"))
    batch_size = int(train_cfg.get("batch_size", 128))
    num_workers = int(train_cfg.get("num_workers", 0))
    total_epochs = int(train_cfg.get("epochs", 80))
    progress_cfg = config.get("progress", {})
    show_console_progress = bool(progress_cfg.get("console", True))
    show_batch_bar = bool(progress_cfg.get("batch_bar", True))
    append_missing_mask = bool(data_cfg.get("append_missing_mask", False))
    augmentation_seed = int(aug_cfg.get("seed", project_seed))
    deterministic_training = bool(aug_cfg.get("deterministic", True))
    selection_metric = str(train_cfg.get("selection_metric", "val_last_risk_score"))

    train_loader = _make_loader(
        prepared.train,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        sensor_indices=prepared.sensor_indices,
        sensor_dropout_prob=float(aug_cfg.get("train_sensor_dropout_prob", 0.0)),
        sensor_dropout_fraction=float(aug_cfg.get("train_sensor_dropout_fraction", 0.0)),
        noise_std=float(aug_cfg.get("train_noise_std", 0.0)),
        augmentation_profile=aug_cfg.get("profile", "none"),
        missing_rate=float(aug_cfg.get("train_missing_rate", 0.0)),
        block_missing_rate=float(aug_cfg.get("train_block_missing_rate", 0.0)),
        drift_rate=float(aug_cfg.get("train_drift_rate", 0.0)),
        mechanism_apply_probability=float(aug_cfg.get("mechanism_apply_probability", 1.0)),
        corrupted_window_probability=float(aug_cfg.get("corrupted_window_probability", 1.0)),
        append_missing_mask=append_missing_mask,
        base_seed=augmentation_seed if deterministic_training else None,
        shuffle_seed=shuffle_seed,
    )
    val_loader = _make_loader(
        prepared.val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        sensor_indices=prepared.sensor_indices,
        append_missing_mask=append_missing_mask,
    )
    val_last_loader = _make_loader(
        prepared.val_last,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        sensor_indices=prepared.sensor_indices,
        append_missing_mask=append_missing_mask,
    )
    endpoint_confirmation_sets = prepared.validation_endpoint_sets or {}
    endpoint_confirmation_best: dict[str, dict[str, Any]] = {
        name: {"score": float("inf"), "epoch": -1, "state": None, "validation_metrics": None}
        for name in endpoint_confirmation_sets
    }

    history_csv_path = run_dir / "epoch_history.csv"
    progress_json_path = run_dir / "progress.json"
    if history_csv_path.exists():
        history_csv_path.unlink()

    context = RunProgressContext(
        experiment_name=experiment_name or "default",
        subset=subset,
        model=model_name,
        run_dir=run_dir,
        device=str(device),
        total_epochs=total_epochs,
        batch_size=batch_size,
        train_batches=len(train_loader),
        n_train_windows=int(len(prepared.train.y)),
        n_val_units=int(len(np.unique(prepared.val.unit_ids))),
        n_test_units=int(len(np.unique(prepared.test.unit_ids))),
        n_features=len(prepared.input_features),
        n_sensors=len(prepared.selected_sensors),
    )
    if show_console_progress:
        print(format_run_start(context, run_index=run_index, total_runs=total_runs), flush=True)
    write_progress_json(
        progress_json_path,
        {
            "status": "starting",
            "experiment_name": experiment_name or "default",
            "subset": subset,
            "model": model_name,
            "run_dir": str(run_dir),
            "device": str(device),
            "current_epoch": 0,
            "total_epochs": total_epochs,
            "selection_metric": selection_metric,
        },
    )

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
        use_uncertainty_head=bool(model_cfg.get("use_uncertainty_head", True)),
        mask_feature_count=len(prepared.sensor_indices) if append_missing_mask else 0,
    ).to(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    criterion = build_loss(
        train_cfg.get("loss", "weighted_huber"),
        delta=float(train_cfg.get("huber_delta", 1.0)),
        late_life_threshold=float(train_cfg.get("late_life_threshold", 30.0)),
        late_life_weight=float(train_cfg.get("late_life_weight", 1.5)),
        late_over_weight=float(train_cfg.get("late_over_weight", 1.0)),
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 1e-4)),
    )

    best_score = float("inf")
    best_epoch_val_loss = float("inf")
    best_epoch_val_last_rmse = float("inf")
    best_epoch_val_last_nasa_score = float("inf")
    best_epoch = -1
    patience = int(train_cfg.get("patience", 12))
    history: list[dict[str, float]] = []
    checkpoint_path = run_dir / "best_model.pt"

    training_start = time.perf_counter()
    for epoch in range(1, total_epochs + 1):
        epoch_start = time.perf_counter()
        model.train()
        if hasattr(train_loader.dataset, "set_epoch"):
            train_loader.dataset.set_epoch(epoch)
        train_losses = []
        training_components: dict[str, list[float]] = {}
        iterator = tqdm(
            train_loader,
            desc=f"{subset} {model_name} epoch {epoch}/{total_epochs}",
            leave=False,
            dynamic_ncols=True,
            mininterval=0.5,
            file=sys.stdout,
            disable=not show_batch_bar,
        )
        for x, y in iterator:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss, components = training_objective(
                model,
                x,
                y,
                criterion,
                train_cfg,
                prepared.sensor_indices,
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            optimizer.step()
            loss_value = float(loss.detach().cpu())
            train_losses.append(loss_value)
            for key, value in components.items():
                training_components.setdefault(key, []).append(value)
            iterator.set_postfix(loss=f"{loss_value:.4f}")

        model.eval()
        val_losses = []
        for x, y in val_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            val_losses.append(float(criterion(pred, y).detach().cpu()))
        val_last_losses = []
        for x, y in val_last_loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            val_last_losses.append(float(criterion(pred, y).detach().cpu()))
        train_loss = float(np.mean(train_losses))
        val_loss = float(np.mean(val_losses))
        val_last_loss = float(np.mean(val_last_losses))
        val_last_epoch_metrics, _ = evaluate_window_data(
            model,
            prepared.val_last,
            device,
            batch_size=batch_size,
            sensor_indices=prepared.sensor_indices,
            append_missing_mask=append_missing_mask,
        )
        epoch_selection_metrics = {
            "val_loss": val_loss,
            "val_last_loss": val_last_loss,
            **{f"val_last_{key}": value for key, value in val_last_epoch_metrics.items()},
        }
        epoch_selection_metrics["val_last_risk_score"] = validation_risk_score(
            val_last_epoch_metrics,
            train_cfg,
            float(data_cfg.get("rul_cap", 125.0)),
        )
        selection_score = checkpoint_score_from_metrics(epoch_selection_metrics, selection_metric)
        if endpoint_confirmation_sets:
            for strategy, endpoint_data in endpoint_confirmation_sets.items():
                if strategy == "uniform_single":
                    strategy_metrics = val_last_epoch_metrics
                else:
                    strategy_metrics, _ = evaluate_window_data(
                        model,
                        endpoint_data,
                        device,
                        batch_size=batch_size,
                        sensor_indices=prepared.sensor_indices,
                        append_missing_mask=append_missing_mask,
                    )
                strategy_score = validation_risk_score(
                    strategy_metrics,
                    train_cfg,
                    float(data_cfg.get("rul_cap", 125.0)),
                )
                if strategy_score < float(endpoint_confirmation_best[strategy]["score"]):
                    endpoint_confirmation_best[strategy] = {
                        "score": float(strategy_score),
                        "epoch": int(epoch),
                        "state": {
                            key: value.detach().cpu().clone()
                            for key, value in model.state_dict().items()
                        },
                        "validation_metrics": dict(strategy_metrics),
                    }
        learning_rate = float(optimizer.param_groups[0]["lr"])

        if selection_score < best_score:
            best_score = selection_score
            best_epoch = epoch
            best_epoch_val_loss = val_loss
            best_epoch_val_last_rmse = float(val_last_epoch_metrics["rmse"])
            best_epoch_val_last_nasa_score = float(val_last_epoch_metrics["nasa_score"])
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "features": prepared.features,
                    "input_features": prepared.input_features,
                    "config": config,
                },
                checkpoint_path,
            )
        early_stop_best_epoch = best_epoch
        if endpoint_confirmation_best:
            early_stop_best_epoch = max(
                early_stop_best_epoch,
                *(int(item["epoch"]) for item in endpoint_confirmation_best.values()),
            )
        patience_wait = max(0, epoch - early_stop_best_epoch)
        epoch_elapsed = time.perf_counter() - epoch_start
        epoch_row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_last_loss": val_last_loss,
            "val_last_rmse": float(val_last_epoch_metrics["rmse"]),
            "val_last_nasa_score": float(val_last_epoch_metrics["nasa_score"]),
            "val_last_lpr30": float(val_last_epoch_metrics["critical_30_late_prediction_ratio"]),
            "val_last_slpr30_10": float(val_last_epoch_metrics["critical_30_severe_late_10_ratio"]),
            "val_last_risk_score": float(epoch_selection_metrics["val_last_risk_score"]),
            "selection_score": selection_score,
            "best_selection_score": best_score,
            "best_checkpoint_score": best_score,
            "best_val_loss": best_epoch_val_loss,
            "best_epoch_val_all_loss": best_epoch_val_loss,
            "best_epoch_val_last_rmse": best_epoch_val_last_rmse,
            "best_epoch_val_last_nasa_score": best_epoch_val_last_nasa_score,
            "best_epoch": best_epoch,
            "patience_wait": patience_wait,
            "learning_rate": learning_rate,
            "elapsed_sec": epoch_elapsed,
            **{
                f"train_{key}_component": float(np.mean(values))
                for key, values in training_components.items()
                if values
            },
        }
        history.append(epoch_row)
        append_epoch_history_csv(history_csv_path, epoch_row)
        write_progress_json(
            progress_json_path,
            {
                "status": "running",
                "experiment_name": experiment_name or "default",
                "subset": subset,
                "model": model_name,
                "run_dir": str(run_dir),
                "device": str(device),
                "current_epoch": epoch,
                "total_epochs": total_epochs,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_last_loss": val_last_loss,
                "val_last_rmse": float(val_last_epoch_metrics["rmse"]),
                "val_last_nasa_score": float(val_last_epoch_metrics["nasa_score"]),
                "selection_metric": selection_metric,
                "selection_score": selection_score,
                "best_selection_score": best_score,
                "best_checkpoint_score": best_score,
                "best_val_loss": best_epoch_val_loss,
                "best_epoch_val_all_loss": best_epoch_val_loss,
                "best_epoch_val_last_rmse": best_epoch_val_last_rmse,
                "best_epoch_val_last_nasa_score": best_epoch_val_last_nasa_score,
                "best_epoch": best_epoch,
                "patience_wait": patience_wait,
                "patience": patience,
            },
        )
        if show_console_progress:
            print(
                format_epoch_status(
                    epoch=epoch,
                    total_epochs=total_epochs,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    best_val_loss=best_score,
                    best_epoch=best_epoch,
                    patience_wait=patience_wait,
                    patience=patience,
                    elapsed_sec=epoch_elapsed,
                    learning_rate=learning_rate,
                ),
                flush=True,
            )
        if patience_wait >= patience:
            if show_console_progress:
                print(f"[EARLY STOP] no validation improvement for {patience_wait} epochs", flush=True)
            break

    training_elapsed_sec = time.perf_counter() - training_start
    peak_gpu_memory_mb = (
        float(torch.cuda.max_memory_allocated(device) / (1024**2)) if device.type == "cuda" else 0.0
    )
    if endpoint_confirmation_sets:
        confirmation_rows = []
        for strategy, endpoint_data in endpoint_confirmation_sets.items():
            selected = endpoint_confirmation_best[strategy]
            if selected["state"] is None:
                raise RuntimeError(f"No checkpoint selected for endpoint strategy={strategy}")
            model.load_state_dict(selected["state"])
            strategy_test_metrics, strategy_test_preds = evaluate_window_data(
                model,
                prepared.test,
                device,
                batch_size=batch_size,
                sensor_indices=prepared.sensor_indices,
                append_missing_mask=append_missing_mask,
            )
            validation_metrics = selected["validation_metrics"] or {}
            confirmation_rows.append(
                {
                    "subset": subset,
                    "model": model_name,
                    "seed": project_seed,
                    "endpoint_strategy": strategy,
                    "best_epoch": int(selected["epoch"]),
                    "best_validation_risk_score": float(selected["score"]),
                    "validation_endpoint_count": int(len(endpoint_data.y)),
                    "validation_engine_count": int(len(np.unique(endpoint_data.unit_ids))),
                    "validation_critical_30_count": int(np.sum(endpoint_data.y <= 30.0)),
                    "validation_rul_mean": float(np.mean(endpoint_data.y)),
                    **{f"validation_{key}": value for key, value in validation_metrics.items()},
                    **{f"test_{key}": value for key, value in strategy_test_metrics.items()},
                    "test_nasa_per_engine": float(strategy_test_metrics["nasa_score"] / len(prepared.test.y)),
                }
            )
            _write_predictions(
                run_dir / f"test_predictions_endpoint_{strategy}.csv",
                prepared.test,
                strategy_test_preds,
            )
            torch.save(
                {
                    "model_state": selected["state"],
                    "features": prepared.features,
                    "input_features": prepared.input_features,
                    "config": config,
                    "endpoint_strategy": strategy,
                    "best_epoch": int(selected["epoch"]),
                },
                run_dir / f"best_model_endpoint_{strategy}.pt",
            )
        pd.DataFrame(confirmation_rows).to_csv(
            run_dir / "checkpoint_endpoint_confirmation.csv", index=False
        )
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"])
    val_metrics, val_preds = evaluate_window_data(
        model,
        prepared.val,
        device,
        batch_size=batch_size,
        sensor_indices=prepared.sensor_indices,
        append_missing_mask=append_missing_mask,
    )
    val_last_metrics, val_last_preds = evaluate_window_data(
        model,
        prepared.val_last,
        device,
        batch_size=batch_size,
        sensor_indices=prepared.sensor_indices,
        append_missing_mask=append_missing_mask,
    )
    test_metrics, test_preds = evaluate_window_data(
        model,
        prepared.test,
        device,
        batch_size=batch_size,
        sensor_indices=prepared.sensor_indices,
        append_missing_mask=append_missing_mask,
    )
    interval_result = evaluate_prediction_intervals(
        model,
        prepared.test,
        device,
        batch_size=batch_size,
        sensor_indices=prepared.sensor_indices,
        append_missing_mask=append_missing_mask,
    )
    interval_metrics: dict[str, float] = {}
    test_intervals: np.ndarray | None = None
    if interval_result is not None:
        interval_metrics, test_intervals = interval_result

    sample_data = RULWindowDataset(
        prepared.test,
        sensor_indices=prepared.sensor_indices,
        append_missing_mask=append_missing_mask,
    )
    sample_x = [sample_data[i][0].numpy() for i in range(min(batch_size, len(prepared.test.x)))]
    sample = torch.from_numpy(np.stack(sample_x)).float()
    inference_ms = measure_inference_time(model, sample, device, warmup=5, repeats=20)
    single_sample_inference_ms = measure_inference_time(model, sample[:1], device, warmup=5, repeats=20)
    cpu_model = copy.deepcopy(model).to(torch.device("cpu"))
    cpu_single_sample_inference_ms = measure_inference_time(cpu_model, sample[:1], torch.device("cpu"), warmup=3, repeats=10)
    profiled_flops_batch1 = profile_forward_flops(copy.deepcopy(cpu_model), sample[:1])

    metrics = {
        "subset": subset,
        "model": model_name,
        "experiment_name": experiment_name or "default",
        "run_dir": str(run_dir),
        "seed": project_seed,
        "split_seed": split_seed,
        "initialization_seed": initialization_seed,
        "shuffle_seed": shuffle_seed,
        "created_time": datetime.now().astimezone().isoformat(timespec="seconds"),
        **protocol_metadata,
        "best_epoch": best_epoch,
        "completed_epochs": int(history[-1]["epoch"]) if history else 0,
        "planned_epochs": total_epochs,
        "training_complete_reason": "early_stopped" if history and int(history[-1]["patience_wait"]) >= patience else "max_epochs",
        "best_val_loss": best_epoch_val_loss,
        "best_epoch_val_all_loss": best_epoch_val_loss,
        "best_selection_score": best_score,
        "best_checkpoint_score": best_score,
        "checkpoint_selection_metric": selection_metric,
        "best_epoch_val_last_rmse": best_epoch_val_last_rmse,
        "best_epoch_val_last_nasa_score": best_epoch_val_last_nasa_score,
        "selection_metric": selection_metric,
        "device": str(device),
        "parameters": count_parameters(model),
        "profiled_flops_batch1": profiled_flops_batch1,
        "preprocessing_elapsed_sec": preprocessing_elapsed_sec,
        "training_elapsed_sec": training_elapsed_sec,
        "peak_gpu_memory_mb": peak_gpu_memory_mb,
        "batch_inference_ms": inference_ms,
        "single_sample_inference_ms": single_sample_inference_ms,
        "cpu_single_sample_inference_ms": cpu_single_sample_inference_ms,
        "n_train_windows": int(len(prepared.train.y)),
        "n_val_windows": int(len(prepared.val.y)),
        "n_val_last_windows": int(len(prepared.val_last.y)),
        "n_val_units": int(len(np.unique(prepared.val.unit_ids))),
        "n_test_units": int(len(np.unique(prepared.test.unit_ids))),
        "append_missing_mask": append_missing_mask,
        "scaler": data_cfg.get("scaler", "standard"),
        "validation_last_strategy": data_cfg.get("validation_last_strategy", "simulated"),
        "augmentation_seed": augmentation_seed,
        "deterministic_training": deterministic_training,
        "val_rul_min": float(np.min(prepared.val.y)),
        "val_rul_max": float(np.max(prepared.val.y)),
        "val_rul_unique": int(len(np.unique(prepared.val.y))),
        "val_last_rul_min": float(np.min(prepared.val_last.y)),
        "val_last_rul_max": float(np.max(prepared.val_last.y)),
        "val_last_rul_mean": float(np.mean(prepared.val_last.y)),
        "val_last_rul_median": float(np.median(prepared.val_last.y)),
        "val_last_rul_unique": int(len(np.unique(prepared.val_last.y))),
        "mask_feature_count": len(prepared.sensor_indices) if append_missing_mask else 0,
        **model_feature_metadata(model_name, model_cfg, append_missing_mask),
        "smooth_late_risk_weight": float(train_cfg.get("smooth_late_risk_weight", 0.0)),
        "reliability_supervision_weight": float(train_cfg.get("reliability_supervision_weight", 0.0)),
        "consistency_weight": float(train_cfg.get("consistency_weight", 0.0)),
        "quantile_calibration_weight": float(train_cfg.get("quantile_calibration_weight", 0.0)),
        "selection_lpr_weight": float(train_cfg.get("selection_lpr_weight", 0.25)),
        "selection_severe_late_weight": float(train_cfg.get("selection_severe_late_weight", 0.25)),
        **{f"val_{k}": v for k, v in val_metrics.items()},
        **{f"val_{k}": v for k, v in monotonicity_metrics(prepared.val, val_preds).items()},
        **{f"val_last_{k}": v for k, v in val_last_metrics.items()},
        **{f"test_{k}": v for k, v in test_metrics.items()},
        **{f"test_{k}": v for k, v in interval_metrics.items()},
        **feature_summary(prepared.input_features),
    }

    hist_counts, hist_edges = np.histogram(prepared.val_last.y, bins=min(10, max(1, len(np.unique(prepared.val_last.y)))))
    val_last_summary = {
        "strategy": data_cfg.get("validation_last_strategy", "simulated"),
        "count": int(len(prepared.val_last.y)),
        "rul_min": metrics["val_last_rul_min"],
        "rul_max": metrics["val_last_rul_max"],
        "rul_mean": metrics["val_last_rul_mean"],
        "rul_median": metrics["val_last_rul_median"],
        "rul_unique": metrics["val_last_rul_unique"],
        "histogram_counts": [int(v) for v in hist_counts.tolist()],
        "histogram_edges": [float(v) for v in hist_edges.tolist()],
    }
    save_json(metrics, run_dir / "metrics.json")
    save_json(val_last_summary, run_dir / "val_last_cutoff_summary.json")
    try:
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(5, 3))
        ax.hist(prepared.val_last.y, bins=min(10, max(1, len(np.unique(prepared.val_last.y)))))
        ax.set_xlabel("Simulated validation RUL")
        ax.set_ylabel("Engine windows")
        ax.set_title(f"{subset} validation cutoff distribution")
        fig.tight_layout()
        fig.savefig(run_dir / "val_last_rul_histogram.png", dpi=160)
        plt.close(fig)
    except Exception:
        pass
    save_json({"history": history}, run_dir / "history.json")
    save_json(
        {
            "features": prepared.features,
            "input_features": prepared.input_features,
            "selected_sensors": prepared.selected_sensors,
            "append_missing_mask": append_missing_mask,
        },
        run_dir / "features.json",
    )
    save_config(config, run_dir / "run_config.yaml")
    _write_predictions(run_dir / "val_predictions.csv", prepared.val, val_preds)
    _write_predictions(run_dir / "val_last_predictions.csv", prepared.val_last, val_last_preds)
    _write_predictions(run_dir / "test_predictions.csv", prepared.test, test_preds)
    if test_intervals is not None:
        _write_interval_predictions(run_dir / "test_interval_predictions.csv", prepared.test, test_intervals)
    write_progress_json(
        progress_json_path,
        {
            "status": "finished",
            "experiment_name": experiment_name or "default",
            "subset": subset,
            "model": model_name,
            "run_dir": str(run_dir),
            "device": str(device),
            "current_epoch": int(history[-1]["epoch"]) if history else 0,
            "total_epochs": total_epochs,
            "best_epoch": best_epoch,
            "selection_metric": selection_metric,
            "best_selection_score": best_score,
            "best_checkpoint_score": best_score,
            "best_val_loss": best_epoch_val_loss,
            "best_epoch_val_all_loss": best_epoch_val_loss,
            "test_rmse": metrics.get("test_rmse"),
            "test_mae": metrics.get("test_mae"),
            "test_nasa_score": metrics.get("test_nasa_score"),
        },
    )
    if show_console_progress:
        print(format_run_finish(metrics), flush=True)

    return metrics
