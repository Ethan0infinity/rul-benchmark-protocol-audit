from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_advanced_evidence import nasa_contribution, prediction_metrics
from run_formal_benchmark import FORMAL_BENCHMARK_SEEDS


def directory(results_dir: Path, seed: int, variant: str) -> Path:
    if variant == "risk_score_checkpoint":
        return results_dir / f"paper_main_v3_seed{seed}" / "FD004" / "rast_gru_v2"
    return (
        results_dir
        / "checkpoint_sensitivity"
        / f"checkpoint_rmse_fd004_seed{seed}"
        / "FD004"
        / "rast_gru_v2"
    )


def metric(error: np.ndarray, true_rul: np.ndarray, name: str) -> float:
    critical = true_rul <= 30.0
    if name == "rmse":
        return float(np.sqrt(np.mean(error**2)))
    if name == "nasa_per_engine":
        return float(np.mean(nasa_contribution(error)))
    if name == "lpr30":
        return float(np.mean(error[critical] > 0.0))
    if name == "slpr30_10":
        return float(np.mean(error[critical] > 10.0))
    raise ValueError(name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare risk-score and pure-RMSE checkpoint selection on FD004.")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "paper_outputs" / "advanced_evidence"))
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    args = parser.parse_args()

    variants = ("risk_score_checkpoint", "rmse_checkpoint")
    rows = []
    pairs = {}
    for seed in FORMAL_BENCHMARK_SEEDS:
        predictions = {}
        for variant in variants:
            run_dir = directory(Path(args.results_dir), seed, variant)
            metrics_path = run_dir / "metrics.json"
            pred_path = run_dir / "test_predictions.csv"
            if not metrics_path.exists() or not pred_path.exists():
                raise FileNotFoundError(run_dir)
            metadata = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
            frame = pd.read_csv(pred_path)
            predictions[variant] = frame
            row = {
                "seed": seed,
                "checkpoint_variant": variant,
                "best_epoch": metadata["best_epoch"],
                "checkpoint_selection_metric": metadata["checkpoint_selection_metric"],
            }
            row.update(prediction_metrics(frame))
            critical = frame["true_rul"].to_numpy(float) <= 30.0
            error = frame["pred_rul"].to_numpy(float) - frame["true_rul"].to_numpy(float)
            row["slpr30_10"] = float(np.mean(error[critical] > 10.0))
            rows.append(row)
        pairs[seed] = predictions["risk_score_checkpoint"].merge(
            predictions["rmse_checkpoint"],
            on=["unit_id", "true_rul"],
            suffixes=("_risk", "_rmse"),
            validate="one_to_one",
        )

    long = pd.DataFrame(rows)
    numeric = [column for column in long.columns if column not in {"seed", "checkpoint_variant", "checkpoint_selection_metric"}]
    summary = long.groupby("checkpoint_variant", as_index=False).agg(
        seed_count=("seed", "nunique"),
        **{f"{column}_mean": (column, "mean") for column in numeric},
        **{f"{column}_std": (column, lambda x: x.std(ddof=1)) for column in numeric},
    )

    arrays = {
        seed: (
            frame["true_rul"].to_numpy(float),
            frame["pred_rul_risk"].to_numpy(float) - frame["true_rul"].to_numpy(float),
            frame["pred_rul_rmse"].to_numpy(float) - frame["true_rul"].to_numpy(float),
        )
        for seed, frame in pairs.items()
    }
    rng = np.random.default_rng(20260715)
    bootstrap_rows = []
    seeds = list(FORMAL_BENCHMARK_SEEDS)
    for name in ("rmse", "nasa_per_engine", "lpr30", "slpr30_10"):
        observed = np.mean([metric(risk, true, name) - metric(rmse, true, name) for true, risk, rmse in arrays.values()])
        samples = np.empty(args.bootstrap_reps, dtype=float)
        for rep in range(args.bootstrap_reps):
            values = []
            for sampled_seed in rng.choice(seeds, size=len(seeds), replace=True):
                true, risk, rmse = arrays[int(sampled_seed)]
                indices = rng.integers(0, len(true), size=len(true))
                values.append(metric(risk[indices], true[indices], name) - metric(rmse[indices], true[indices], name))
            samples[rep] = float(np.mean(values))
        bootstrap_rows.append(
            {
                "metric": name,
                "mean_difference_risk_minus_rmse_checkpoint": float(observed),
                "percentile_ci95_low": float(np.quantile(samples, 0.025)),
                "percentile_ci95_high": float(np.quantile(samples, 0.975)),
                "probability_rmse_checkpoint_lower": float(np.mean(samples > 0.0)),
                "probability_tie": float(np.mean(np.isclose(samples, 0.0, atol=1e-12))),
                "probability_risk_checkpoint_lower": float(np.mean(samples < 0.0)),
                "bootstrap_repetitions": args.bootstrap_reps,
            }
        )
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    long.to_csv(output / "checkpoint_sensitivity_seed_metrics.csv", index=False)
    summary.to_csv(output / "checkpoint_sensitivity_summary.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(output / "checkpoint_sensitivity_paired_bootstrap.csv", index=False)
    print(f"CHECKPOINT_SENSITIVITY_READY output={output}")


if __name__ == "__main__":
    main()
