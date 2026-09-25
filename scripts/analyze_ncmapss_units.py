from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rul.metrics import regression_metrics
from run_formal_benchmark import FORMAL_BENCHMARK_MODELS, FORMAL_BENCHMARK_SEEDS


def main() -> None:
    parser = argparse.ArgumentParser(description="Report fixed-task N-CMAPSS DS02 results by held-out test unit.")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "paper_outputs" / "advanced_evidence"))
    parser.add_argument("--experiment-prefix", default="ncmapss_ds02")
    parser.add_argument("--models", nargs="+", default=FORMAL_BENCHMARK_MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=FORMAL_BENCHMARK_SEEDS)
    parser.add_argument("--output-prefix", default="ncmapss_ds02")
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    results_dir = Path(args.results_dir)
    for seed in args.seeds:
        for model in args.models:
            path = (
                results_dir
                / f"{args.experiment_prefix}_seed{seed}"
                / "DS02"
                / model
                / "test_predictions.csv"
            )
            if not path.exists():
                raise FileNotFoundError(path)
            frame = pd.read_csv(path)
            for unit, unit_frame in frame.groupby("unit_id", sort=True):
                metrics = regression_metrics(
                    unit_frame["true_rul"].to_numpy(float), unit_frame["pred_rul"].to_numpy(float)
                )
                rows.append(
                    {
                        "seed": seed,
                        "model": model,
                        "test_unit": int(unit),
                        "window_count": len(unit_frame),
                        "rmse": metrics["rmse"],
                        "mae": metrics["mae"],
                        "nasa_per_window": metrics["nasa_score"] / len(unit_frame),
                        "lpr30": metrics["critical_30_late_prediction_ratio"],
                        "critical_window_count": metrics["critical_30_count"],
                    }
                )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    long = pd.DataFrame(rows)
    summary = long.groupby(["model", "test_unit"], as_index=False).agg(
        seed_count=("seed", "nunique"),
        window_count=("window_count", "first"),
        rmse_mean=("rmse", "mean"),
        rmse_std=("rmse", lambda x: x.std(ddof=1)),
        mae_mean=("mae", "mean"),
        mae_std=("mae", lambda x: x.std(ddof=1)),
        nasa_per_window_mean=("nasa_per_window", "mean"),
        nasa_per_window_std=("nasa_per_window", lambda x: x.std(ddof=1)),
        lpr30_mean=("lpr30", "mean"),
        lpr30_std=("lpr30", lambda x: x.std(ddof=1)),
    )
    expected = len(args.seeds) * len(args.models) * 3
    if len(long) != expected or long["test_unit"].nunique() != 3:
        raise ValueError(f"Expected {expected} seed-model-unit rows across three test units; found {len(long)}")
    long.to_csv(output_dir / f"{args.output_prefix}_per_unit_seed.csv", index=False)
    summary.to_csv(output_dir / f"{args.output_prefix}_per_unit_summary.csv", index=False)
    print(f"NCMAPSS_UNIT_ANALYSIS_READY rows={len(long)} units={sorted(long['test_unit'].unique())}")


if __name__ == "__main__":
    main()
