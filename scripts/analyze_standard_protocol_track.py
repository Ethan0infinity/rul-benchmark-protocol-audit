from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_advanced_evidence import nasa_contribution, prediction_metrics
from run_formal_benchmark import FORMAL_BENCHMARK_SEEDS


DEFAULT_MODELS = (
    "rast_gru",
    "cnn_lstm",
    "attention_gru",
    "sensor_graph_gru",
    "transformer_lite",
    "rast_gru_v2",
)


def validate_track_protocol(
    metadata: dict,
    config: dict,
    track: str,
    run_dir: Path,
) -> None:
    training = config["training"]
    augmentation = config["augmentation"]
    if track == "common_risk_protocol":
        expected = {
            "scaler": "condition_standard",
            "append_missing_mask": True,
            "checkpoint_selection_metric": "val_last_risk_score",
        }
        config_expected = {
            "loss": "asymmetric_weighted_huber",
            "late_over_weight": 1.5,
            "smooth_late_risk_weight": 0.1,
            "selection_metric": "val_last_risk_score",
            "selection_lpr_weight": 0.25,
            "selection_severe_late_weight": 0.25,
        }
        augmentation_profile = "missing_noise_drift"
    else:
        expected = {
            "scaler": "standard",
            "append_missing_mask": False,
            "checkpoint_selection_metric": "val_last_rmse",
        }
        config_expected = {
            "loss": "huber",
            "late_over_weight": 1.0,
            "smooth_late_risk_weight": 0.0,
            "selection_metric": "val_last_rmse",
            "selection_lpr_weight": 0.0,
            "selection_severe_late_weight": 0.0,
        }
        augmentation_profile = "none"
    mismatches = []
    for key, value in expected.items():
        if metadata.get(key) != value:
            mismatches.append(f"metrics.{key}={metadata.get(key)!r}, expected {value!r}")
    for key, value in config_expected.items():
        if training.get(key) != value:
            mismatches.append(f"training.{key}={training.get(key)!r}, expected {value!r}")
    if augmentation.get("profile") != augmentation_profile:
        mismatches.append(
            f"augmentation.profile={augmentation.get('profile')!r}, expected {augmentation_profile!r}"
        )
    if mismatches:
        raise ValueError(f"Track protocol mismatch in {run_dir}: " + "; ".join(mismatches))


def paired_metric(error: np.ndarray, true_rul: np.ndarray, name: str) -> float:
    if name == "rmse":
        return float(np.sqrt(np.mean(error**2)))
    if name == "nasa_per_engine":
        return float(np.mean(nasa_contribution(error)))
    if name == "lpr30":
        critical = true_rul <= 30.0
        return float(np.mean(error[critical] > 0.0))
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare FD004 common-risk and standardized conventional training tracks.")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "paper_outputs" / "advanced_evidence"))
    parser.add_argument("--models", nargs="+", default=list(DEFAULT_MODELS))
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    args = parser.parse_args()

    rows = []
    prediction_frames: dict[tuple[int, str, str], pd.DataFrame] = {}
    root = Path(args.results_dir)
    for seed in FORMAL_BENCHMARK_SEEDS:
        for model in args.models:
            for track in ("common_risk_protocol", "standardized_conventional_protocol"):
                if track == "common_risk_protocol":
                    run_dir = root / f"paper_main_v3_seed{seed}" / "FD004" / model
                else:
                    run_dir = (
                        root
                        / "standard_protocol_track"
                        / f"standard_protocol_fd004_seed{seed}"
                        / "FD004"
                        / model
                    )
                pred_path = run_dir / "test_predictions.csv"
                metrics_path = run_dir / "metrics.json"
                if not pred_path.exists() or not metrics_path.exists():
                    raise FileNotFoundError(run_dir)
                pred = pd.read_csv(pred_path)
                prediction_frames[(seed, model, track)] = pred
                metadata = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
                config = yaml.safe_load((run_dir / "run_config.yaml").read_text(encoding="utf-8"))
                validate_track_protocol(metadata, config, track, run_dir)
                row = {
                    "seed": seed,
                    "model": model,
                    "track": track,
                    "checkpoint_selection_metric": metadata["checkpoint_selection_metric"],
                    "scaler": metadata["scaler"],
                    "append_missing_mask": metadata["append_missing_mask"],
                    "best_epoch": metadata["best_epoch"],
                    "parameters": metadata["parameters"],
                }
                row.update(prediction_metrics(pred))
                rows.append(row)

    long = pd.DataFrame(rows)
    metrics = ("rmse", "mae", "nasa_per_engine", "lpr30", "mle30", "late_cvar95_30")
    summary = long.groupby(["model", "track"], as_index=False).agg(
        seed_count=("seed", "nunique"),
        **{f"{metric}_mean": (metric, "mean") for metric in metrics},
        **{f"{metric}_std": (metric, lambda x: x.std(ddof=1)) for metric in metrics},
    )
    pivot = summary.pivot(index="model", columns="track", values=[f"{metric}_mean" for metric in metrics])
    comparison_rows = []
    for model in args.models:
        row = {"model": model}
        for metric in metrics:
            conventional = float(pivot.loc[model, (f"{metric}_mean", "standardized_conventional_protocol")])
            common = float(pivot.loc[model, (f"{metric}_mean", "common_risk_protocol")])
            row[f"{metric}_common"] = common
            row[f"{metric}_conventional"] = conventional
            row[f"{metric}_conventional_minus_common"] = conventional - common
        comparison_rows.append(row)

    rng = np.random.default_rng(20260715)
    bootstrap_rows = []
    seeds = list(FORMAL_BENCHMARK_SEEDS)
    for model in args.models:
        paired = {}
        for seed in seeds:
            common = prediction_frames[(seed, model, "common_risk_protocol")]
            conventional = prediction_frames[(seed, model, "standardized_conventional_protocol")]
            merged = common.merge(
                conventional,
                on=["unit_id", "true_rul"],
                suffixes=("_common", "_conventional"),
                validate="one_to_one",
            )
            true = merged["true_rul"].to_numpy(float)
            paired[seed] = (
                true,
                merged["pred_rul_common"].to_numpy(float) - true,
                merged["pred_rul_conventional"].to_numpy(float) - true,
            )
        for metric_name in ("rmse", "nasa_per_engine", "lpr30"):
            observed = float(
                np.mean(
                    [
                        paired_metric(common_error, true, metric_name)
                        - paired_metric(conventional_error, true, metric_name)
                        for true, common_error, conventional_error in paired.values()
                    ]
                )
            )
            samples = np.empty(args.bootstrap_reps, dtype=float)
            for rep in range(args.bootstrap_reps):
                values = []
                for sampled_seed in rng.choice(seeds, size=len(seeds), replace=True):
                    true, common_error, conventional_error = paired[int(sampled_seed)]
                    indices = rng.integers(0, len(true), size=len(true))
                    values.append(
                        paired_metric(common_error[indices], true[indices], metric_name)
                        - paired_metric(conventional_error[indices], true[indices], metric_name)
                    )
                samples[rep] = float(np.mean(values))
            bootstrap_rows.append(
                {
                    "model": model,
                    "metric": metric_name,
                    "mean_difference_common_minus_conventional": observed,
                    "percentile_ci95_low": float(np.quantile(samples, 0.025)),
                    "percentile_ci95_high": float(np.quantile(samples, 0.975)),
                    "probability_conventional_lower": float(np.mean(samples > 0.0)),
                    "probability_tie": float(np.mean(np.isclose(samples, 0.0, atol=1e-12))),
                    "probability_common_lower": float(np.mean(samples < 0.0)),
                    "bootstrap_repetitions": args.bootstrap_reps,
                }
            )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    long.to_csv(output / "standard_protocol_track_seed_metrics.csv", index=False)
    summary.to_csv(output / "standard_protocol_track_summary.csv", index=False)
    pd.DataFrame(comparison_rows).to_csv(output / "standard_protocol_track_comparison.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(output / "standard_protocol_track_paired_bootstrap.csv", index=False)
    print(
        f"STANDARD_PROTOCOL_TRACK_READY rows={len(long)} bootstrap_rows={len(bootstrap_rows)} output={output}"
    )


if __name__ == "__main__":
    main()
