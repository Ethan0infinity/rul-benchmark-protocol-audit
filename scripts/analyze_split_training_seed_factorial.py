from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).absolute().parents[1]
POINTS = ("RAST", "Core", "Asym")
METRICS = ("rmse", "nasa_per_engine", "lpr30", "late_cvar95_30")


def late_cvar95(predictions: pd.DataFrame) -> float:
    true = predictions["true_rul"].to_numpy(float)
    error = predictions["pred_rul"].to_numpy(float) - true
    late = error[(true <= 30.0) & (error > 0.0)]
    if len(late) == 0:
        return 0.0
    threshold = float(np.quantile(late, 0.95))
    return float(np.mean(late[late >= threshold]))


def load_cells(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for item in manifest.itertuples(index=False):
        run_dir = Path(item.run_dir)
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8-sig"))
        predictions = pd.read_csv(run_dir / "test_predictions.csv")
        rows.append(
            {
                "point": item.point,
                "split_seed": int(item.split_seed),
                "training_seed": int(item.training_seed),
                "rmse": float(metrics["test_rmse"]),
                "nasa_per_engine": float(metrics["test_nasa_score"]) / float(metrics["n_test_units"]),
                "lpr30": float(metrics["test_critical_30_late_prediction_ratio"]),
                "late_cvar95_30": late_cvar95(predictions),
                "run_dir": str(run_dir),
            }
        )
    return pd.DataFrame(rows)


def decompose(cells: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for point in POINTS:
        part = cells[cells["point"] == point]
        for metric in METRICS:
            matrix = part.pivot(index="split_seed", columns="training_seed", values=metric)
            grand = float(matrix.to_numpy().mean())
            split_effects = matrix.mean(axis=1) - grand
            training_effects = matrix.mean(axis=0) - grand
            residual = matrix - grand - split_effects.to_numpy()[:, None] - training_effects.to_numpy()[None, :]
            rows.append(
                {
                    "point": point,
                    "metric": metric,
                    "grand_mean": grand,
                    "total_cell_sd": float(matrix.stack().std(ddof=1)),
                    "split_mean_sd": float(split_effects.std(ddof=1)),
                    "training_mean_sd": float(training_effects.std(ddof=1)),
                    "nonadditive_residual_rms": float(np.sqrt(np.mean(residual.to_numpy() ** 2))),
                    "cell_min": float(matrix.min().min()),
                    "cell_max": float(matrix.max().max()),
                    "split_levels": int(matrix.shape[0]),
                    "training_levels": int(matrix.shape[1]),
                }
            )
    return pd.DataFrame(rows)


def crossed_bootstrap(cells: pd.DataFrame, repetitions: int) -> pd.DataFrame:
    rows: list[dict] = []
    comparisons = (("Core", "RAST"), ("Asym", "RAST"), ("Core", "Asym"))
    split_levels = sorted(cells["split_seed"].unique())
    training_levels = sorted(cells["training_seed"].unique())
    for comparison_index, (left, right) in enumerate(comparisons):
        for metric_index, metric in enumerate(METRICS):
            left_matrix = (
                cells[cells["point"] == left]
                .pivot(index="split_seed", columns="training_seed", values=metric)
                .loc[split_levels, training_levels]
                .to_numpy(float)
            )
            right_matrix = (
                cells[cells["point"] == right]
                .pivot(index="split_seed", columns="training_seed", values=metric)
                .loc[split_levels, training_levels]
                .to_numpy(float)
            )
            difference = left_matrix - right_matrix
            rng = np.random.default_rng(20260720 + comparison_index * 101 + metric_index)
            draws = np.empty(repetitions, dtype=float)
            for rep in range(repetitions):
                split_draw = rng.integers(0, len(split_levels), size=len(split_levels))
                training_draw = rng.integers(0, len(training_levels), size=len(training_levels))
                draws[rep] = float(difference[np.ix_(split_draw, training_draw)].mean())
            low, high = np.quantile(draws, [0.025, 0.975])
            rows.append(
                {
                    "left_point": left,
                    "right_point": right,
                    "metric": metric,
                    "mean_difference_left_minus_right": float(difference.mean()),
                    "crossed_factor_bootstrap_ci95_low": float(low),
                    "crossed_factor_bootstrap_ci95_high": float(high),
                    "probability_left_lower": float(np.mean(draws < 0.0)),
                    "split_levels": len(split_levels),
                    "training_levels": len(training_levels),
                    "bootstrap_repetitions": repetitions,
                }
            )
    return pd.DataFrame(rows)


def plot_cells(cells: pd.DataFrame, output: Path) -> None:
    labels = {"rmse": "RMSE", "lpr30": "LPR@30"}
    colors = {"RAST": "#2f6f9f", "Core": "#2a9d8f", "Asym": "#c44e52"}
    markers = {"RAST": "o", "Core": "^", "Asym": "s"}
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5))
    for ax, metric in zip(axes, ("rmse", "lpr30")):
        for point in POINTS:
            part = cells[cells["point"] == point]
            grouped = part.groupby("split_seed")[metric].agg(["mean", "std"])
            x = np.arange(len(grouped))
            ax.errorbar(
                x,
                grouped["mean"],
                yerr=grouped["std"],
                color=colors[point],
                marker=markers[point],
                linewidth=1.7,
                capsize=3,
                label=point,
            )
        ax.set_xticks(x, [str(value) for value in grouped.index])
        ax.set_xlabel("Engine split seed")
        ax.set_ylabel(labels[metric])
        ax.grid(linestyle=":", alpha=0.35)
    axes[0].legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the FD004 split-by-training-seed audit.")
    parser.add_argument(
        "--manifest",
        default=str(PROJECT_ROOT / "results" / "seed_factorial_fd004" / "run_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "paper_outputs" / "advanced_evidence"),
    )
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    expected = len(POINTS) * 3 * 3
    if len(manifest) != expected:
        raise ValueError(f"Expected {expected} complete factorial cells, found {len(manifest)}")
    if manifest.groupby("point").size().to_dict() != {point: 9 for point in POINTS}:
        raise ValueError("Each point must contain a complete 3-by-3 grid.")

    output = Path(args.output_dir)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    cells = load_cells(manifest)
    cells.to_csv(output / "seed_factorial_fd004_cells.csv", index=False)
    decomposition = decompose(cells)
    decomposition.to_csv(output / "seed_factorial_fd004_decomposition.csv", index=False)
    intervals = crossed_bootstrap(cells, args.bootstrap_reps)
    intervals.to_csv(output / "seed_factorial_fd004_paired_intervals.csv", index=False)
    plot_cells(cells, figures / "fig_seed_factorial_fd004")
    print(
        f"SEED_FACTORIAL_ANALYSIS_READY cells={len(cells)} "
        f"decomposition={len(decomposition)} intervals={len(intervals)}"
    )


if __name__ == "__main__":
    main()
