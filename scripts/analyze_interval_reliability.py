from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mpl_cache"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from calibrate_prediction_intervals import conformal_quantile, interval_metrics
from rul.experiments import _load_run
from rul.training import evaluate_prediction_intervals
from run_formal_benchmark import FORMAL_BENCHMARK_SEEDS, FORMAL_BENCHMARK_SUBSETS


MODELS = ("quantile_gru", "rast_gru_v2")


def evaluate_run(run_dir: Path, *, seed: int, subset: str, model: str, nominal_levels: list[float]) -> list[dict]:
    config, _, prepared, network, device = _load_run(run_dir)
    kwargs = {
        "device": device,
        "batch_size": int(config["training"].get("batch_size", 128)),
        "sensor_indices": prepared.sensor_indices,
        "append_missing_mask": bool(config["data"].get("append_missing_mask", False)),
    }
    validation_result = evaluate_prediction_intervals(network, prepared.val_last, **kwargs)
    test_result = evaluate_prediction_intervals(network, prepared.test, **kwargs)
    if validation_result is None or test_result is None:
        raise TypeError(f"No interval head: {run_dir}")
    _, validation_intervals = validation_result
    _, test_intervals = test_result
    validation_scores = np.maximum.reduce(
        [
            validation_intervals[:, 0] - prepared.val_last.y,
            prepared.val_last.y - validation_intervals[:, 2],
            np.zeros(len(prepared.val_last.y), dtype=float),
        ]
    )
    raw_lower = test_intervals[:, 0]
    raw_median = test_intervals[:, 1]
    raw_upper = test_intervals[:, 2]
    crossing_rate = float(np.mean((raw_lower > raw_median) | (raw_median > raw_upper)))
    rows = []
    for nominal in nominal_levels:
        alpha = 1.0 - nominal
        qhat, finite_level = conformal_quantile(validation_scores, alpha)
        lower = np.maximum(raw_lower - qhat, 0.0)
        upper = raw_upper + qhat
        metrics = interval_metrics(prepared.test.y, lower, upper, alpha)
        rows.append(
            {
                "seed": seed,
                "subset": subset,
                "model": model,
                "nominal_coverage": nominal,
                "finite_sample_quantile_level": finite_level,
                "calibration_engine_count": len(prepared.val_last.y),
                "test_engine_count": len(prepared.test.y),
                "qhat": qhat,
                "quantile_crossing_rate": crossing_rate,
                **metrics,
            }
        )
    return rows


def plot_reliability(summary: pd.DataFrame, output: Path) -> None:
    display = {"quantile_gru": "Quantile-GRU", "rast_gru_v2": "OCM-MST-GRU"}
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    for model, group in summary.groupby("model"):
        group = group.sort_values("nominal_coverage")
        label = display.get(model, model)
        axes[0].plot(group["nominal_coverage"], group["coverage_mean"], marker="o", linewidth=2.0, label=label)
        axes[1].plot(group["nominal_coverage"], group["mean_width_mean"], marker="o", linewidth=2.0, label=label)
    axes[0].plot([0.70, 0.95], [0.70, 0.95], linestyle="--", color="#333333", linewidth=1.2)
    axes[0].set(xlabel="Nominal target", ylabel="Empirical coverage", title="Post-selection coverage diagnostic")
    axes[1].set(xlabel="Nominal marginal coverage", ylabel="Mean interval width (cycles)", title="Coverage-width trade-off")
    for ax in axes:
        ax.grid(alpha=0.25)
    axes[0].legend(frameon=False)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build nominal 70%-95% post-selection empirical interval diagnostics."
    )
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "paper_outputs" / "advanced_evidence"))
    parser.add_argument("--models", nargs="+", default=list(MODELS))
    parser.add_argument("--seeds", nargs="+", type=int, default=FORMAL_BENCHMARK_SEEDS)
    parser.add_argument("--subsets", nargs="+", default=FORMAL_BENCHMARK_SUBSETS)
    parser.add_argument("--nominal-levels", nargs="+", type=float, default=[0.70, 0.75, 0.80, 0.85, 0.90, 0.95])
    args = parser.parse_args()

    if any(level <= 0.0 or level >= 1.0 for level in args.nominal_levels):
        raise ValueError("Nominal levels must lie strictly between zero and one.")
    rows = []
    total = len(args.models) * len(args.seeds) * len(args.subsets)
    index = 0
    for seed in args.seeds:
        for subset in args.subsets:
            for model in args.models:
                index += 1
                print(f"[INTERVAL RELIABILITY {index}/{total}] seed={seed} subset={subset} model={model}", flush=True)
                run_dir = Path(args.results_dir) / f"paper_main_v3_seed{seed}" / subset / model
                rows.extend(
                    evaluate_run(
                        run_dir,
                        seed=seed,
                        subset=subset,
                        model=model,
                        nominal_levels=args.nominal_levels,
                    )
                )
    frame = pd.DataFrame(rows)
    summary = frame.groupby(["model", "nominal_coverage"], as_index=False).agg(
        task_count=("coverage", "count"),
        coverage_mean=("coverage", "mean"),
        coverage_std=("coverage", lambda x: x.std(ddof=1)),
        critical_coverage_mean=("critical_coverage", "mean"),
        critical_coverage_std=("critical_coverage", lambda x: x.std(ddof=1)),
        mean_width_mean=("mean_width", "mean"),
        mean_width_std=("mean_width", lambda x: x.std(ddof=1)),
        interval_score_mean=("interval_score", "mean"),
        qhat_mean=("qhat", "mean"),
        calibration_engine_count_min=("calibration_engine_count", "min"),
        crossing_rate_max=("quantile_crossing_rate", "max"),
    )
    output_dir = Path(args.output_dir)
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output_dir / "interval_reliability_seed_subset.csv", index=False)
    summary.to_csv(output_dir / "interval_reliability_summary.csv", index=False)
    plot_reliability(summary, figure_dir / "fig_interval_reliability_curve")
    print(f"INTERVAL_RELIABILITY_READY rows={len(frame)} output={output_dir}")


if __name__ == "__main__":
    main()
