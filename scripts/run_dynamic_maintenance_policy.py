"""Evaluate a dynamic RUL-trigger maintenance policy on complete test trajectories.

This script is an exploratory decision study, not a prospective maintenance
trial. The policy observes only the current and previous predictions. True
failure time is used after the policy path is fixed to score preventive versus
corrective replacement.

The stored C-MAPSS test files contain trajectories up to an observation cutoff.
The public RUL file supplies the remaining time at that cutoff. The default
evaluation is therefore an observed-prefix analysis: an engine with no trigger
before the cutoff is marked unresolved rather than being assigned a later
action that cannot be observed. The declared final RUL is used only to score a
trigger that occurred within the observed prefix.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rul.cmapss import load_subset
from rul.config import load_config
from rul.features import select_features, sensor_indices_in_features
from rul.models import build_model
from rul.preprocessing import RULWindowDataset, fit_transform_train_val_test, make_windows, split_units
from rul.training import _resolve_data_root
from rul.utils import get_device, set_seed


DEFAULT_SUBSETS = ("FD001", "FD002", "FD003", "FD004")
DEFAULT_MODELS = ("rast_gru", "rast_gru_v2")
DEFAULT_THRESHOLDS = (10.0, 20.0, 30.0, 40.0)
DEFAULT_LEAD_TIMES = (0.0, 5.0, 10.0)
DEFAULT_PERSISTENCE = (1, 2, 3)
DEFAULT_CORRECTIVE_RATIOS = (5.0, 10.0, 20.0)
DEFAULT_AGE_TRIGGER = 100.0


def prepare_full_test_windows(config: dict[str, Any], subset: str, run_dir: Path):
    data_cfg = config["data"]
    train_df, test_df = load_subset(_resolve_data_root(config), subset, data_cfg.get("rul_cap", 125))
    split_seed = int(config["project"].get("split_seed", config["project"].get("seed", 42)))
    train_units, val_units = split_units(
        train_df["unit_id"].to_numpy(),
        float(data_cfg.get("validation_split", 0.2)),
        split_seed,
    )
    train_core = train_df[train_df["unit_id"].isin(train_units)].copy()
    val = train_df[train_df["unit_id"].isin(val_units)].copy()
    selected = select_features(
        train_core,
        feature_mode=data_cfg.get("feature_mode", "selected"),
        max_selected_sensors=int(data_cfg.get("max_selected_sensors", 14)),
        include_settings=bool(data_cfg.get("include_settings", True)),
        output_csv=run_dir / "policy_feature_scores.csv",
    )
    _, _, test_scaled, _ = fit_transform_train_val_test(
        train_core,
        val,
        test_df,
        selected.features,
        scaler_name=data_cfg.get("scaler", "standard"),
        subset=subset,
        condition_clusters=data_cfg.get("condition_clusters"),
        random_state=split_seed,
    )
    windows = make_windows(
        test_scaled,
        selected.features,
        int(data_cfg.get("window_size", 30)),
        int(data_cfg.get("stride", 1)),
        last_only=False,
    )
    sensor_indices = sensor_indices_in_features(selected.features)
    append_mask = bool(data_cfg.get("append_missing_mask", False))
    input_features = list(selected.features)
    if append_mask:
        input_features.extend(f"{selected.features[i]}_observed" for i in sensor_indices)
    return windows, sensor_indices, append_mask, len(input_features), selected.features


def predict_all(model: torch.nn.Module, windows, sensor_indices, append_mask, device, batch_size):
    loader = DataLoader(
        RULWindowDataset(windows, sensor_indices=sensor_indices, append_missing_mask=append_mask),
        batch_size=batch_size,
        shuffle=False,
    )
    predictions = []
    model.eval()
    with torch.no_grad():
        for x, _ in loader:
            predictions.append(model(x.to(device)).detach().cpu().numpy())
    return np.concatenate(predictions).astype(float)


def load_checkpoint_model(config: dict[str, Any], model_name: str, input_features: int, run_dir: Path, device):
    model_cfg = config["model"]
    model = build_model(
        model_name,
        input_features=input_features,
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
        mask_feature_count=14 if bool(config["data"].get("append_missing_mask", False)) else 0,
    ).to(device)
    checkpoint = torch.load(run_dir / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state"])
    return model


def simulate_policy(group: pd.DataFrame, threshold: float, lead_time: float, persistence: int, corrective_ratio: float) -> dict[str, Any]:
    group = group.sort_values("end_cycle").reset_index(drop=True)
    pred = group["pred_rul"].to_numpy(float)
    cycles = group["end_cycle"].to_numpy(float)
    failure_cycle = float(cycles[-1] + group["true_rul"].iloc[-1])
    alarm = pred <= float(threshold)
    trigger_idx = None
    streak = 0
    for idx, is_alarm in enumerate(alarm):
        streak = streak + 1 if bool(is_alarm) else 0
        if streak >= int(persistence):
            trigger_idx = idx - int(persistence) + 1
            break
    if trigger_idx is None:
        return {
            "triggered": 0,
            "preventive": 0,
            "resolved": 0,
            "failure_before_service": np.nan,
            "unresolved_beyond_observation": 1,
            "trigger_cycle": np.nan,
            "service_cycle": np.nan,
            "failure_cycle": failure_cycle,
            "remaining_life_at_trigger": np.nan,
            "cost": np.nan,
        }
    trigger_cycle = float(cycles[trigger_idx])
    service_cycle = trigger_cycle + float(lead_time)
    preventive = service_cycle < failure_cycle
    remaining_at_trigger = failure_cycle - trigger_cycle
    if preventive:
        # Normalized units: preventive replacement=1; unused-life cost is
        # reported separately so it cannot be mistaken for a calibrated money.
        cost = 1.0 + 0.01 * max(remaining_at_trigger, 0.0)
    else:
        cost = float(corrective_ratio)
    return {
        "triggered": 1,
        "preventive": int(preventive),
        "resolved": 1,
        "failure_before_service": int(not preventive),
        "unresolved_beyond_observation": 0,
        "trigger_cycle": trigger_cycle,
        "service_cycle": service_cycle,
        "failure_cycle": failure_cycle,
        "remaining_life_at_trigger": remaining_at_trigger,
        "cost": float(cost),
    }


def simulate_fixed_age_policy(
    group: pd.DataFrame,
    age_trigger: float,
    lead_time: float,
    corrective_ratio: float,
) -> dict[str, Any]:
    """A model-free calendar-age comparator with a fixed trigger."""
    group = group.sort_values("end_cycle").reset_index(drop=True)
    cycles = group["end_cycle"].to_numpy(float)
    failure_cycle = float(cycles[-1] + group["true_rul"].iloc[-1])
    candidates = np.flatnonzero(cycles >= float(age_trigger))
    if len(candidates) == 0:
        return {
            "triggered": 0,
            "preventive": 0,
            "resolved": 0,
            "failure_before_service": np.nan,
            "unresolved_beyond_observation": 1,
            "trigger_cycle": np.nan,
            "service_cycle": np.nan,
            "failure_cycle": failure_cycle,
            "remaining_life_at_trigger": np.nan,
            "cost": np.nan,
        }
    trigger_cycle = float(cycles[int(candidates[0])])
    service_cycle = trigger_cycle + float(lead_time)
    preventive = service_cycle < failure_cycle
    remaining_at_trigger = failure_cycle - trigger_cycle
    return {
        "triggered": 1,
        "preventive": int(preventive),
        "resolved": 1,
        "failure_before_service": int(not preventive),
        "unresolved_beyond_observation": 0,
        "trigger_cycle": trigger_cycle,
        "service_cycle": service_cycle,
        "failure_cycle": failure_cycle,
        "remaining_life_at_trigger": remaining_at_trigger,
        "cost": float(1.0 + 0.01 * max(remaining_at_trigger, 0.0)) if preventive else float(corrective_ratio),
    }


def build_paired_model_comparisons(cell: pd.DataFrame) -> pd.DataFrame:
    """Compare model policies on the same engines and policy cells.

    Costs are compared only when both model policies resolve before the
    observation cutoff and both costs are finite. Resolution discordance is
    retained as a separate outcome rather than hidden by conditional means.
    """
    keys = [
        "subset",
        "split_level",
        "training_stream",
        "unit_id",
        "rul_trigger_threshold",
        "lead_time_cycles",
        "persistence_windows",
        "corrective_cost_ratio",
    ]
    group_keys = [key for key in keys if key != "unit_id"]
    columns = group_keys + [
        "eligible_engine_count",
        "reference_model",
        "comparison_model",
        "reference_resolved_count",
        "comparison_resolved_count",
        "both_resolved_count",
        "reference_only_resolved_count",
        "comparison_only_resolved_count",
        "neither_resolved_count",
        "mean_cost_delta_comparison_minus_reference_among_both_resolved",
        "median_cost_delta_comparison_minus_reference_among_both_resolved",
        "paired_cost_support",
    ]
    if cell.empty:
        return pd.DataFrame(columns=columns)

    model_rows = cell[cell["policy_type"].eq("model_threshold")].copy()
    reference_model = "rast_gru"
    comparison_model = "rast_gru_v2"
    reference = model_rows[model_rows["model"].eq(reference_model)]
    comparison = model_rows[model_rows["model"].eq(comparison_model)]
    if reference.empty or comparison.empty:
        return pd.DataFrame(columns=columns)

    merged = reference.merge(
        comparison,
        on=keys,
        how="inner",
        suffixes=("_reference", "_comparison"),
    )
    if merged.empty:
        return pd.DataFrame(columns=columns)

    rows: list[dict[str, Any]] = []
    for group_values, group in merged.groupby(group_keys, dropna=False):
        if not isinstance(group_values, tuple):
            group_values = (group_values,)
        row = dict(zip(group_keys, group_values))
        ref_resolved = group["resolved_reference"].astype(int).eq(1)
        cmp_resolved = group["resolved_comparison"].astype(int).eq(1)
        both = ref_resolved & cmp_resolved
        ref_only = ref_resolved & ~cmp_resolved
        cmp_only = cmp_resolved & ~ref_resolved
        neither = ~ref_resolved & ~cmp_resolved
        delta = (
            pd.to_numeric(group.loc[both, "cost_comparison"], errors="coerce")
            - pd.to_numeric(group.loc[both, "cost_reference"], errors="coerce")
        ).dropna()
        row.update(
            {
                "eligible_engine_count": int(len(group)),
                "reference_model": reference_model,
                "comparison_model": comparison_model,
                "reference_resolved_count": int(ref_resolved.sum()),
                "comparison_resolved_count": int(cmp_resolved.sum()),
                "both_resolved_count": int(both.sum()),
                "reference_only_resolved_count": int(ref_only.sum()),
                "comparison_only_resolved_count": int(cmp_only.sum()),
                "neither_resolved_count": int(neither.sum()),
                "mean_cost_delta_comparison_minus_reference_among_both_resolved": float(delta.mean()) if not delta.empty else np.nan,
                "median_cost_delta_comparison_minus_reference_among_both_resolved": float(delta.median()) if not delta.empty else np.nan,
                "paired_cost_support": int(len(delta)),
            }
        )
        rows.append(row)
    return pd.DataFrame(rows, columns=columns)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "paper_main.yaml"))
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--experiment-name", default="paper_main_v3_seed42")
    parser.add_argument(
        "--run-manifest",
        help="Optional full_5x10 manifest. When supplied, evaluate each listed completed run.",
    )
    parser.add_argument("--subsets", nargs="+", default=list(DEFAULT_SUBSETS))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--thresholds", nargs="+", type=float, default=list(DEFAULT_THRESHOLDS))
    parser.add_argument("--lead-times", nargs="+", type=float, default=list(DEFAULT_LEAD_TIMES))
    parser.add_argument("--persistence", nargs="+", type=int, default=list(DEFAULT_PERSISTENCE))
    parser.add_argument("--corrective-ratios", nargs="+", type=float, default=list(DEFAULT_CORRECTIVE_RATIOS))
    parser.add_argument("--age-trigger-cycle", type=float, default=DEFAULT_AGE_TRIGGER)
    parser.add_argument("--device", default="cpu", choices=("cpu", "cuda", "auto"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "paper_outputs" / "dynamic_maintenance_policy"))
    args = parser.parse_args()

    base = load_config(args.config)
    base["project"]["results_dir"] = args.results_dir
    device = get_device(args.device)
    set_seed(int(base["project"].get("seed", 42)))
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    protocol = {
        "evidence_role": "post-result exploratory dynamic policy evaluation",
        "policy_observation": "current and previous model predictions only",
        "true_rul_use": "evaluation cost only; never used to trigger an action",
        "preventive_cost": "1 normalized unit",
        "early_replacement_penalty": "0.01 normalized units per remaining cycle at trigger",
        "corrective_cost_ratio_grid": list(args.corrective_ratios),
        "threshold_grid": list(args.thresholds),
        "lead_time_grid": list(args.lead_times),
        "persistence_grid": list(args.persistence),
        "policy_scope": "C-MAPSS observed-prefix trajectories; not field maintenance validation",
        "unresolved_rule": "no in-prefix trigger remains unresolved; it is not recoded as corrective failure",
        "cost_denominator_rule": "model-based cost is conditional on resolved engines",
        "paired_cost_rule": "OCM-versus-RAST cost contrasts use jointly resolved engines only",
        "model_free_comparator": f"fixed age trigger at cycle {args.age_trigger_cycle}; always corrective baseline",
        "device": str(device),
    }
    (output_dir / "dynamic_policy_protocol.json").write_text(json.dumps(protocol, indent=2), encoding="utf-8")

    if args.run_manifest:
        manifest_frame = pd.read_csv(args.run_manifest)
        if manifest_frame.empty:
            raise ValueError(f"Run manifest is empty: {args.run_manifest}")
        run_specs = [
            (
                str(row.subset),
                str(row.model),
                PROJECT_ROOT / str(row.run_dir),
                int(row.split_level),
                int(row.training_stream),
            )
            for row in manifest_frame.itertuples(index=False)
            if str(row.status) in {"completed", "reused"}
        ]
    else:
        run_specs = [
            (
                subset,
                model_name,
                PROJECT_ROOT / args.results_dir / args.experiment_name / subset / model_name,
                int(base["project"].get("split_seed", base["project"].get("seed", 42))),
                int(base["project"].get("seed", 42)),
            )
            for subset in args.subsets
            for model_name in args.models
        ]
    baseline_source_model = run_specs[0][1] if run_specs else None
    for subset, model_name, run_dir, split_level, training_stream in run_specs:
            config_path = run_dir / "run_config.yaml"
            if not config_path.exists():
                raise FileNotFoundError(config_path)
            config = load_config(config_path)
            windows, sensor_indices, append_mask, input_features, _ = prepare_full_test_windows(config, subset, run_dir)
            model = load_checkpoint_model(config, model_name, input_features, run_dir, device)
            pred = predict_all(model, windows, sensor_indices, append_mask, device, int(config["training"].get("batch_size", 128)))
            frame = pd.DataFrame({
                "unit_id": windows.unit_ids.astype(int),
                "end_cycle": windows.end_cycles.astype(int),
                "true_rul": windows.y.astype(float),
                "pred_rul": np.maximum(pred, 0.0),
            })
            prediction_path = output_dir / (
                f"{subset}_{model_name}_split{split_level}_stream{training_stream}_"
                "trajectory_predictions.csv"
            )
            frame.to_csv(prediction_path, index=False)
            for threshold in args.thresholds:
                for lead_time in args.lead_times:
                    for persistence in args.persistence:
                        for corrective_ratio in args.corrective_ratios:
                            for unit_id, group in frame.groupby("unit_id"):
                                result = simulate_policy(group, threshold, lead_time, persistence, corrective_ratio)
                                all_rows.append({
                                    "subset": subset,
                                    "model": model_name,
                                    "split_level": split_level,
                                    "training_stream": training_stream,
                                    "policy_type": "model_threshold",
                                    "unit_id": int(unit_id),
                                    "rul_trigger_threshold": threshold,
                                    "lead_time_cycles": lead_time,
                                    "persistence_windows": persistence,
                                    "corrective_cost_ratio": corrective_ratio,
                                    **result,
                                })
                            if model_name != baseline_source_model:
                                continue
                            for baseline_name, baseline_result in (
                                ("always_corrective", {
                                         "triggered": 0,
                                         "preventive": 0,
                                         "resolved": 1,
                                         "failure_before_service": 1,
                                }),
                            ):
                                for unit_id, group in frame.groupby("unit_id"):
                                    failure_cycle = float(group.sort_values("end_cycle")["end_cycle"].iloc[-1] + group.sort_values("end_cycle")["true_rul"].iloc[-1])
                                    all_rows.append({
                                         "subset": subset,
                                         "model": baseline_name,
                                         "split_level": split_level,
                                         "training_stream": training_stream,
                                         "policy_type": baseline_name,
                                        "unit_id": int(unit_id),
                                        "rul_trigger_threshold": threshold,
                                        "lead_time_cycles": lead_time,
                                        "persistence_windows": persistence,
                                         "corrective_cost_ratio": corrective_ratio,
                                         "triggered": 0,
                                         "preventive": 0,
                                         "resolved": 1,
                                         "failure_before_service": 1,
                                         "unresolved_beyond_observation": 0,
                                        "trigger_cycle": np.nan,
                                        "service_cycle": np.nan,
                                        "failure_cycle": failure_cycle,
                                        "remaining_life_at_trigger": np.nan,
                                        "cost": float(corrective_ratio),
                                    })
                                    age_result = simulate_fixed_age_policy(group, args.age_trigger_cycle, lead_time, corrective_ratio)
                                    all_rows.append({
                                         "subset": subset,
                                         "model": "fixed_age",
                                         "split_level": split_level,
                                         "training_stream": training_stream,
                                         "policy_type": "fixed_age",
                                        "unit_id": int(unit_id),
                                        "rul_trigger_threshold": threshold,
                                        "lead_time_cycles": lead_time,
                                        "persistence_windows": persistence,
                                        "corrective_cost_ratio": corrective_ratio,
                                        **age_result,
                                    })
    cell = pd.DataFrame(all_rows)
    cell.to_csv(output_dir / "dynamic_policy_engine_level.csv", index=False)
    summary = cell.groupby(
        [
            "subset",
            "model",
            "split_level",
            "training_stream",
            "policy_type",
            "rul_trigger_threshold",
            "lead_time_cycles",
            "persistence_windows",
            "corrective_cost_ratio",
        ],
        as_index=False,
    ).agg(
        mean_cost_among_resolved=("cost", "mean"),
        preventive_rate=("preventive", "mean"),
        trigger_rate=("triggered", "mean"),
        resolved_count=("resolved", "sum"),
        resolution_rate=("resolved", "mean"),
        failure_before_service_rate=("failure_before_service", "mean"),
        unresolved_rate=("unresolved_beyond_observation", "mean"),
        mean_remaining_life_at_trigger=("remaining_life_at_trigger", "mean"),
        engine_count=("unit_id", "nunique"),
    )
    summary.to_csv(output_dir / "dynamic_policy_summary.csv", index=False)

    paired = build_paired_model_comparisons(cell)
    paired.to_csv(output_dir / "dynamic_policy_paired_model_comparison.csv", index=False)
    print(f"DYNAMIC_POLICY_READY rows={len(cell)} cells={len(summary)} output={output_dir}")


if __name__ == "__main__":
    main()
