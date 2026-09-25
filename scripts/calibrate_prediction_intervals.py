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

from rul.experiments import _load_run
from rul.training import evaluate_prediction_intervals
from run_formal_benchmark import FORMAL_BENCHMARK_SEEDS, FORMAL_BENCHMARK_SUBSETS

plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9.5})


INTERVAL_MODELS = ["quantile_gru", "rast_gru_v2"]


def interval_metrics(y: np.ndarray, lower: np.ndarray, upper: np.ndarray, alpha: float) -> dict[str, float]:
    covered = (y >= lower) & (y <= upper)
    width = upper - lower
    score = width + (2.0 / alpha) * np.maximum(lower - y, 0.0) + (2.0 / alpha) * np.maximum(y - upper, 0.0)
    critical = y <= 30.0
    return {
        "coverage": float(np.mean(covered)),
        "absolute_coverage_error": float(abs(np.mean(covered) - (1.0 - alpha))),
        "mean_width": float(np.mean(width)),
        "interval_score": float(np.mean(score)),
        "critical_coverage": float(np.mean(covered[critical])) if np.any(critical) else float("nan"),
        "critical_mean_width": float(np.mean(width[critical])) if np.any(critical) else float("nan"),
    }


def conformal_quantile(scores: np.ndarray, alpha: float) -> tuple[float, float]:
    n = len(scores)
    if n < 2:
        raise ValueError("At least two validation-last engines are required for residual-quantile adjustment.")
    level = min(1.0, np.ceil((n + 1) * (1.0 - alpha)) / n)
    qhat = float(np.quantile(scores, level, method="higher"))
    return max(qhat, 0.0), float(level)


def calibrate_run(run_dir: Path, *, seed: int, subset: str, model_name: str, alpha: float) -> tuple[dict, pd.DataFrame]:
    config, _, prepared, model, device = _load_run(run_dir)
    batch_size = int(config["training"].get("batch_size", 128))
    kwargs = {
        "device": device,
        "batch_size": batch_size,
        "sensor_indices": prepared.sensor_indices,
        "append_missing_mask": bool(config["data"].get("append_missing_mask", False)),
    }
    validation_result = evaluate_prediction_intervals(model, prepared.val_last, **kwargs)
    test_result = evaluate_prediction_intervals(model, prepared.test, **kwargs)
    if validation_result is None or test_result is None:
        raise TypeError(f"Model does not expose quantile intervals: {run_dir}")
    _, validation_intervals = validation_result
    _, test_intervals = test_result
    validation_scores = np.maximum.reduce(
        [
            validation_intervals[:, 0] - prepared.val_last.y,
            prepared.val_last.y - validation_intervals[:, 2],
            np.zeros(len(prepared.val_last.y), dtype=float),
        ]
    )
    qhat, quantile_level = conformal_quantile(validation_scores, alpha)
    raw_lower = test_intervals[:, 0]
    raw_upper = test_intervals[:, 2]
    calibrated_lower = np.maximum(raw_lower - qhat, 0.0)
    calibrated_upper = raw_upper + qhat
    raw = interval_metrics(prepared.test.y, raw_lower, raw_upper, alpha)
    calibrated = interval_metrics(prepared.test.y, calibrated_lower, calibrated_upper, alpha)
    row = {
        "seed": seed,
        "subset": subset,
        "model": model_name,
        "nominal_coverage": 1.0 - alpha,
        "calibration_engine_count": len(prepared.val_last.y),
        "test_engine_count": len(prepared.test.y),
        "conformal_quantile_level": quantile_level,
        "conformal_qhat": qhat,
        **{f"raw_{key}": value for key, value in raw.items()},
        **{f"conformal_{key}": value for key, value in calibrated.items()},
    }
    predictions = pd.DataFrame(
        {
            "seed": seed,
            "subset": subset,
            "model": model_name,
            "unit_id": prepared.test.unit_ids,
            "true_rul": prepared.test.y,
            "raw_q10": raw_lower,
            "raw_q90": raw_upper,
            "conformal_q10": calibrated_lower,
            "conformal_q90": calibrated_upper,
            "conformal_qhat": qhat,
        }
    )
    return row, predictions


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "raw_coverage",
        "raw_absolute_coverage_error",
        "raw_mean_width",
        "raw_interval_score",
        "raw_critical_coverage",
        "conformal_coverage",
        "conformal_absolute_coverage_error",
        "conformal_mean_width",
        "conformal_interval_score",
        "conformal_critical_coverage",
        "conformal_qhat",
    ]
    seed_macro = frame.groupby(["seed", "model"], as_index=False)[metrics].mean()
    return seed_macro.groupby("model", as_index=False).agg(
        seed_count=("seed", "nunique"),
        **{f"{metric}_mean": (metric, "mean") for metric in metrics},
        **{f"{metric}_std": (metric, lambda x: x.std(ddof=1)) for metric in metrics},
    )


def plot_calibration(frame: pd.DataFrame, output: Path) -> None:
    long = []
    display = {"quantile_gru": "Quantile-GRU", "rast_gru_v2": "OCM-MST-GRU"}
    for row in frame.to_dict("records"):
        for state in ("raw", "conformal"):
            long.append(
                {
                    "model": display.get(row["model"], row["model"]),
                    "state": "Raw quantiles" if state == "raw" else "Empirical adjustment",
                    "coverage": row[f"{state}_coverage"],
                    "width": row[f"{state}_mean_width"],
                }
            )
    plot = pd.DataFrame(long)
    fig, axes = plt.subplots(1, 2, figsize=(10.8, 4.4))
    sns.pointplot(data=plot, x="state", y="coverage", hue="model", errorbar=("ci", 95), dodge=0.25, ax=axes[0])
    axes[0].axhline(0.8, color="#333333", linestyle="--", linewidth=1.0)
    axes[0].set_xlabel("")
    axes[0].set_ylabel("Empirical coverage")
    axes[0].set_title("Nominal 80% coverage")
    sns.pointplot(data=plot, x="state", y="width", hue="model", errorbar=("ci", 95), dodge=0.25, ax=axes[1])
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Mean interval width (cycles)")
    axes[1].set_title("Sharpness cost of adjustment")
    if axes[1].legend_ is not None:
        axes[1].legend_.remove()
    axes[0].legend(title="Model", frameon=False)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply a post-selection empirical residual-quantile adjustment to q10/q90 RUL intervals. "
            "The reused validation endpoints do not support a split-conformal coverage guarantee."
        )
    )
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--models", nargs="+", default=INTERVAL_MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=FORMAL_BENCHMARK_SEEDS)
    parser.add_argument("--subsets", nargs="+", default=FORMAL_BENCHMARK_SUBSETS)
    parser.add_argument("--alpha", type=float, default=0.20)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not 0.0 < args.alpha < 1.0:
        raise ValueError("alpha must be between 0 and 1")
    total = len(args.models) * len(args.seeds) * len(args.subsets)
    print(f"[INTERVAL ADJUSTMENT] evaluations={total} models={','.join(args.models)} alpha={args.alpha}")
    if args.dry_run:
        print("INTERVAL_ADJUSTMENT_DRY_RUN_PASS")
        return
    rows = []
    predictions = []
    results_root = Path(args.results_dir)
    index = 0
    for seed in args.seeds:
        for subset in args.subsets:
            for model_name in args.models:
                index += 1
                print(f"[INTERVAL ADJUSTMENT {index}/{total}] seed={seed} subset={subset} model={model_name}", flush=True)
                run_dir = results_root / f"paper_main_v3_seed{seed}" / subset / model_name
                row, prediction = calibrate_run(
                    run_dir,
                    seed=seed,
                    subset=subset,
                    model_name=model_name,
                    alpha=args.alpha,
                )
                rows.append(row)
                predictions.append(prediction)

    output = PROJECT_ROOT / "paper_outputs" / "advanced_evidence"
    figure_dir = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "conformal_interval_seed_subset.csv", index=False)
    summarize(frame).to_csv(output / "conformal_interval_summary.csv", index=False)
    pd.concat(predictions, ignore_index=True).to_csv(output / "conformal_interval_predictions.csv", index=False)
    plot_calibration(frame, figure_dir / "fig_conformal_interval_calibration")
    print(f"POST_SELECTION_INTERVAL_ADJUSTMENT_READY {output}")


if __name__ == "__main__":
    main()
