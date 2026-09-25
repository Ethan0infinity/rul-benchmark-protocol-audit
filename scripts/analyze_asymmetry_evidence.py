from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_advanced_evidence import nasa_contribution, prediction_metrics
from calibrate_prediction_intervals import calibrate_run, summarize as summarize_conformal
from run_formal_benchmark import FORMAL_BENCHMARK_SEEDS


VARIANTS = ("asymmetric_full", "symmetric_candidate")


def run_dir(results_dir: Path, seed: int, variant: str) -> Path:
    if variant == "asymmetric_full":
        return results_dir / f"paper_main_v3_seed{seed}" / "FD004" / "rast_gru_v2"
    return results_dir / f"ablation_fd004_seed{seed}_weighted_huber_no_asymmetry" / "FD004" / "rast_gru_v2"


def interval_metrics(frame: pd.DataFrame, alpha: float = 0.20) -> dict[str, float]:
    y = frame["true_rul"].to_numpy(float)
    lower = frame["q10_rul"].to_numpy(float)
    median = frame["q50_rul"].to_numpy(float)
    upper = frame["q90_rul"].to_numpy(float)
    covered = (y >= lower) & (y <= upper)
    width = upper - lower
    score = width + (2.0 / alpha) * np.maximum(lower - y, 0.0) + (2.0 / alpha) * np.maximum(y - upper, 0.0)
    critical = y <= 30.0
    return {
        "interval_coverage": float(np.mean(covered)),
        "interval_width": float(np.mean(width)),
        "interval_score": float(np.mean(score)),
        "critical_interval_coverage": float(np.mean(covered[critical])),
        "quantile_crossing_rate": float(np.mean((lower > median) | (median > upper))),
    }


def metric_value(frame: pd.DataFrame, metric: str) -> float:
    error = frame["pred_rul"].to_numpy(float) - frame["true_rul"].to_numpy(float)
    return metric_from_arrays(error, frame["true_rul"].to_numpy(float), metric)


def metric_from_arrays(error: np.ndarray, true_rul: np.ndarray, metric: str) -> float:
    critical = true_rul <= 30.0
    if metric == "rmse":
        return float(np.sqrt(np.mean(error**2)))
    if metric == "nasa_per_engine":
        return float(np.mean(nasa_contribution(error)))
    if metric == "lpr30":
        return float(np.mean(error[critical] > 0.0))
    if metric == "slpr30_10":
        return float(np.mean(error[critical] > 10.0))
    if metric.startswith("decision_cost_"):
        late_cost = float(metric.rsplit("_", 1)[1])
        return float(np.mean(late_cost * np.maximum(error, 0.0) + np.maximum(-error, 0.0)))
    raise ValueError(metric)


def paired_bootstrap(
    paired: dict[int, pd.DataFrame], *, metric: str, reps: int, rng: np.random.Generator
) -> dict[str, float | int | str]:
    seeds = sorted(paired)
    arrays = {
        seed: (
            frame["true_rul"].to_numpy(float),
            frame["pred_rul_full"].to_numpy(float) - frame["true_rul"].to_numpy(float),
            frame["pred_rul_symmetric"].to_numpy(float) - frame["true_rul"].to_numpy(float),
        )
        for seed, frame in paired.items()
    }
    observed = np.mean(
        [metric_from_arrays(full, true, metric) - metric_from_arrays(symmetric, true, metric) for true, full, symmetric in arrays.values()]
    )
    samples = np.empty(reps, dtype=float)
    for rep in range(reps):
        differences = []
        for seed in rng.choice(seeds, size=len(seeds), replace=True):
            true, full, symmetric = arrays[int(seed)]
            indices = rng.integers(0, len(true), size=len(true))
            differences.append(
                metric_from_arrays(full[indices], true[indices], metric)
                - metric_from_arrays(symmetric[indices], true[indices], metric)
            )
        samples[rep] = float(np.mean(differences))
    return {
        "metric": metric,
        "mean_difference_full_minus_symmetric": float(observed),
        "ci95_low": float(np.quantile(samples, 0.025)),
        "ci95_high": float(np.quantile(samples, 0.975)),
        "probability_symmetric_lower": float(np.mean(samples > 0.0)),
        "probability_tie": float(np.mean(np.isclose(samples, 0.0, atol=1e-12))),
        "probability_symmetric_higher": float(np.mean(samples < 0.0)),
        "seed_count": len(seeds),
        "bootstrap_repetitions": reps,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit the FD004 asymmetric-loss component against its symmetric candidate.")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "paper_outputs" / "advanced_evidence"))
    parser.add_argument("--seeds", nargs="+", type=int, default=FORMAL_BENCHMARK_SEEDS)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    paired: dict[int, pd.DataFrame] = {}
    conformal_rows: list[dict[str, object]] = []
    conformal_cache_path = output_dir / "asymmetry_fd004_conformal_seed.csv"
    conformal_cache = pd.read_csv(conformal_cache_path) if conformal_cache_path.exists() else pd.DataFrame()

    for seed in args.seeds:
        predictions: dict[str, pd.DataFrame] = {}
        for variant in VARIANTS:
            directory = run_dir(results_dir, seed, variant)
            pred_path = directory / "test_predictions.csv"
            interval_path = directory / "test_interval_predictions.csv"
            if not pred_path.exists() or not interval_path.exists():
                raise FileNotFoundError(f"Missing asymmetry-audit artifact: {directory}")
            pred = pd.read_csv(pred_path)
            interval = pd.read_csv(interval_path)
            predictions[variant] = pred
            row: dict[str, object] = {"seed": seed, "variant": variant}
            row.update(prediction_metrics(pred))
            critical_error = (
                pred.loc[pred["true_rul"].to_numpy(float) <= 30.0, "pred_rul"].to_numpy(float)
                - pred.loc[pred["true_rul"].to_numpy(float) <= 30.0, "true_rul"].to_numpy(float)
            )
            row["slpr30_10"] = float(np.mean(critical_error > 10.0))
            row.update(interval_metrics(interval))
            error = pred["pred_rul"].to_numpy(float) - pred["true_rul"].to_numpy(float)
            for ratio in (1.0, 2.0, 5.0, 10.0):
                row[f"decision_cost_{ratio:g}"] = float(
                    np.mean(ratio * np.maximum(error, 0.0) + np.maximum(-error, 0.0))
                )
            rows.append(row)
            cached = conformal_cache.loc[
                (conformal_cache.get("seed", pd.Series(dtype=int)) == seed)
                & (conformal_cache.get("model", pd.Series(dtype=str)) == variant)
            ]
            if len(cached) == 1:
                conformal_row = cached.iloc[0].to_dict()
            else:
                conformal_row, _ = calibrate_run(
                    directory,
                    seed=seed,
                    subset="FD004",
                    model_name=variant,
                    alpha=0.20,
                )
            conformal_rows.append(conformal_row)

        paired[seed] = predictions["asymmetric_full"].merge(
            predictions["symmetric_candidate"],
            on=["unit_id", "true_rul"],
            suffixes=("_full", "_symmetric"),
            validate="one_to_one",
        )

    seed_level = pd.DataFrame(rows)
    numeric = [column for column in seed_level.columns if column not in {"seed", "variant"}]
    summary = seed_level.groupby("variant", as_index=False).agg(
        seed_count=("seed", "nunique"),
        **{f"{column}_mean": (column, "mean") for column in numeric},
        **{f"{column}_std": (column, lambda x: x.std(ddof=1)) for column in numeric},
    )
    rng = np.random.default_rng(20260715)
    bootstrap = pd.DataFrame(
        [
            paired_bootstrap(paired, metric=metric, reps=args.bootstrap_reps, rng=rng)
            for metric in (
                "rmse",
                "nasa_per_engine",
                "lpr30",
                "slpr30_10",
                "decision_cost_1",
                "decision_cost_2",
                "decision_cost_5",
                "decision_cost_10",
            )
        ]
    )
    conformal = pd.DataFrame(conformal_rows)
    conformal_summary = summarize_conformal(conformal)

    seed_level.to_csv(output_dir / "asymmetry_fd004_seed_metrics.csv", index=False)
    summary.to_csv(output_dir / "asymmetry_fd004_summary.csv", index=False)
    bootstrap.to_csv(output_dir / "asymmetry_fd004_paired_bootstrap.csv", index=False)
    conformal.to_csv(output_dir / "asymmetry_fd004_conformal_seed.csv", index=False)
    conformal_summary.to_csv(output_dir / "asymmetry_fd004_conformal_summary.csv", index=False)

    indexed = summary.set_index("variant")
    core = ["rmse_mean", "nasa_per_engine_mean", "lpr30_mean", "slpr30_10_mean"]
    dominates = all(
        float(indexed.loc["symmetric_candidate", metric]) < float(indexed.loc["asymmetric_full", metric])
        for metric in core
    )
    verdict = {
        "decision_source": "post-hoc FD004 ablation; exploratory only",
        "symmetric_candidate_strictly_dominates_core_fd004_means": dominates,
        "confirmation_required_on_independent_task": True,
        "core_metrics": core,
        "bootstrap_repetitions": args.bootstrap_reps,
    }
    (output_dir / "asymmetry_fd004_verdict.json").write_text(
        json.dumps(verdict, indent=2), encoding="utf-8", newline="\n"
    )
    print(f"ASYMMETRY_EVIDENCE_READY dominates={dominates} output={output_dir}")


if __name__ == "__main__":
    main()
