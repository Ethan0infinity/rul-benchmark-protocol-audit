from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rul.statistics import exact_mcnemar_test, paired_bootstrap_mean_diff, wilcoxon_signed_rank


def _prediction_path(results_dir: Path, experiment_name: str, subset: str, model: str) -> Path:
    return results_dir / experiment_name / subset / model / "test_predictions.csv"


def _prediction_metrics(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    err = df["pred_rul"] - df["true_rul"]
    df["abs_error"] = err.abs()
    df["nasa_score_contribution"] = np.where(err < 0, np.exp(-err / 13.0) - 1.0, np.exp(err / 10.0) - 1.0)
    df["critical_30_abs_error"] = np.where(df["true_rul"] <= 30.0, df["abs_error"], np.nan)
    df["critical_50_abs_error"] = np.where(df["true_rul"] <= 50.0, df["abs_error"], np.nan)
    df["late_indicator"] = (err > 0).astype(float)
    df["critical_30_late_indicator"] = np.where(df["true_rul"] <= 30.0, (err > 0).astype(float), np.nan)
    return df[
        [
            "unit_id",
            "abs_error",
            "nasa_score_contribution",
            "critical_30_abs_error",
            "critical_50_abs_error",
            "late_indicator",
            "critical_30_late_indicator",
        ]
    ]


def compare_models(
    results_dir: Path,
    experiment_name: str,
    subsets: list[str],
    proposed_model: str,
    baseline_models: list[str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for subset in subsets:
        proposed_path = _prediction_path(results_dir, experiment_name, subset, proposed_model)
        if not proposed_path.exists():
            raise FileNotFoundError(f"Missing proposed predictions: {proposed_path}")
        proposed = _prediction_metrics(proposed_path)
        for baseline_model in baseline_models:
            baseline_path = _prediction_path(results_dir, experiment_name, subset, baseline_model)
            if not baseline_path.exists():
                raise FileNotFoundError(f"Missing baseline predictions: {baseline_path}")
            baseline = _prediction_metrics(baseline_path)
            merged = baseline.merge(proposed, on="unit_id", how="inner", suffixes=("_baseline", "_proposed"))
            for metric in [
                "abs_error",
                "nasa_score_contribution",
                "critical_30_abs_error",
                "critical_50_abs_error",
                "late_indicator",
                "critical_30_late_indicator",
            ]:
                metric_rows = merged[[f"{metric}_baseline", f"{metric}_proposed"]].dropna()
                if metric_rows.empty:
                    continue
                baseline_values = metric_rows[f"{metric}_baseline"]
                proposed_values = metric_rows[f"{metric}_proposed"]
                if metric.endswith("indicator"):
                    test_name = "exact_mcnemar"
                    test = exact_mcnemar_test(baseline_values, proposed_values)
                else:
                    test_name = "wilcoxon_signed_rank"
                    test = wilcoxon_signed_rank(baseline_values, proposed_values)
                bootstrap = paired_bootstrap_mean_diff(baseline_values, proposed_values)
                rows.append(
                    {
                        "experiment_name": experiment_name,
                        "subset": subset,
                        "metric": metric,
                        "test_name": test_name,
                        "proposed_model": proposed_model,
                        "baseline_model": baseline_model,
                        "mean_baseline_metric": float(metric_rows[f"{metric}_baseline"].mean()),
                        "mean_proposed_metric": float(metric_rows[f"{metric}_proposed"].mean()),
                        "paired_mean_diff_baseline_minus_proposed": bootstrap["mean_diff"],
                        "paired_mean_diff_ci95_low": bootstrap["ci_low"],
                        "paired_mean_diff_ci95_high": bootstrap["ci_high"],
                        "paired_bootstrap_reps": bootstrap["n_bootstrap"],
                        **test,
                    }
                )
    return rows


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run Wilcoxon signed-rank tests on per-engine test errors.")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument(
        "--experiment-name",
        default="paper_main_v3_seed42",
        help="Experiment name to compare. Defaults to paper_main_v3_seed42 for PyCharm current-file runs.",
    )
    parser.add_argument(
        "--experiment-prefix",
        default=None,
        help="Run tests for every experiment directory whose name starts with this prefix.",
    )
    parser.add_argument("--subsets", nargs="+", default=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--proposed-model", default="rast_gru_v2")
    parser.add_argument("--baseline-models", nargs="+", default=["gru", "tcn_gru", "rast_gru"])
    parser.add_argument("--output", default=str(PROJECT_ROOT / "paper_outputs" / "tables" / "table_statistical_tests.csv"))
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    if args.experiment_prefix:
        experiment_names = sorted(
            path.name for path in results_dir.iterdir() if path.is_dir() and path.name.startswith(args.experiment_prefix)
        )
        if not experiment_names:
            raise FileNotFoundError(f"No experiment directories match prefix: {args.experiment_prefix}")
    else:
        experiment_names = [args.experiment_name]

    rows: list[dict[str, object]] = []
    for experiment_name in experiment_names:
        rows.extend(
            compare_models(
                results_dir,
                experiment_name,
                args.subsets,
                args.proposed_model,
                args.baseline_models,
            )
        )
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({key for row in rows for key in row.keys()}))
        writer.writeheader()
        writer.writerows(rows)
    print(f"[STATS] saved {len(rows)} paired tests to {output}")


if __name__ == "__main__":
    main()
