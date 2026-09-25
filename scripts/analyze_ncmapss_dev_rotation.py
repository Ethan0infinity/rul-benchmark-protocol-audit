from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).absolute().parents[1]
POINTS = ("RAST", "Core", "Asym")
METRICS = ("unit_macro_rmse", "unit_macro_nasa_per_window", "unit_macro_lpr30")


def load_cells(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in manifest.itertuples(index=False):
        run_dir = Path(item.run_dir)
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8-sig"))
        rows.append(
            {
                "point": item.point,
                "test_unit": int(item.test_unit),
                "validation_unit": int(item.validation_unit),
                "training_seed": int(item.training_seed),
                **{metric: float(metrics[metric]) for metric in METRICS},
                "run_dir": run_dir.relative_to(PROJECT_ROOT).as_posix(),
            }
        )
    return pd.DataFrame(rows)


def summarize(cells: pd.DataFrame) -> pd.DataFrame:
    return cells.groupby("point", as_index=False).agg(
        unit_count=("test_unit", "nunique"),
        seed_count=("training_seed", "nunique"),
        **{f"{metric}_mean": (metric, "mean") for metric in METRICS},
        **{f"{metric}_sd": (metric, lambda x: x.std(ddof=1)) for metric in METRICS},
    )


def crossed_bootstrap(cells: pd.DataFrame, repetitions: int) -> pd.DataFrame:
    rows = []
    units = sorted(cells["test_unit"].unique())
    seeds = sorted(cells["training_seed"].unique())
    comparisons = (("Core", "RAST"), ("Asym", "RAST"), ("Core", "Asym"))
    for comparison_index, (left, right) in enumerate(comparisons):
        for metric_index, metric in enumerate(METRICS):
            matrices = []
            for point in (left, right):
                matrices.append(
                    cells[cells["point"] == point]
                    .pivot(index="test_unit", columns="training_seed", values=metric)
                    .loc[units, seeds]
                    .to_numpy(float)
                )
            difference = matrices[0] - matrices[1]
            rng = np.random.default_rng(20260720 + comparison_index * 101 + metric_index)
            draws = np.empty(repetitions, dtype=float)
            for rep in range(repetitions):
                unit_draw = rng.integers(0, len(units), size=len(units))
                seed_draw = rng.integers(0, len(seeds), size=len(seeds))
                draws[rep] = float(difference[np.ix_(unit_draw, seed_draw)].mean())
            low, high = np.quantile(draws, [0.025, 0.975])
            rows.append(
                {
                    "left_point": left,
                    "right_point": right,
                    "metric": metric,
                    "mean_difference_left_minus_right": float(difference.mean()),
                    "unit_seed_bootstrap_ci95_low": float(low),
                    "unit_seed_bootstrap_ci95_high": float(high),
                    "probability_left_lower": float(np.mean(draws < 0.0)),
                    "test_unit_count": len(units),
                    "training_seed_count": len(seeds),
                    "bootstrap_repetitions": repetitions,
                }
            )
    return pd.DataFrame(rows)


def plot_unit_results(cells: pd.DataFrame, output: Path) -> None:
    colors = {"RAST": "#2f6f9f", "Core": "#2a9d8f", "Asym": "#c44e52"}
    markers = {"RAST": "o", "Core": "^", "Asym": "s"}
    panels = (("unit_macro_rmse", "Unit RMSE"), ("unit_macro_lpr30", "Unit LPR@30"))
    fig, axes = plt.subplots(1, 2, figsize=(8.4, 3.5))
    for ax, (metric, label) in zip(axes, panels):
        for point in POINTS:
            summary = cells[cells["point"] == point].groupby("test_unit")[metric].agg(["mean", "std"])
            ax.errorbar(
                summary.index.astype(str), summary["mean"], yerr=summary["std"],
                color=colors[point], marker=markers[point], linewidth=1.7, capsize=3, label=point,
            )
        ax.set_xlabel("Held-out development unit")
        ax.set_ylabel(label)
        ax.grid(linestyle=":", alpha=0.35)
    axes[0].legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze N-CMAPSS development-unit rotation evidence.")
    parser.add_argument(
        "--manifest",
        default=str(PROJECT_ROOT / "results" / "ncmapss_dev_rotation" / "run_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "paper_outputs" / "advanced_evidence"),
    )
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    if len(manifest) != len(POINTS) * 6 * 3:
        raise ValueError(f"Expected 54 rotation cells, found {len(manifest)}")
    output = Path(args.output_dir)
    figures = output / "figures"
    figures.mkdir(parents=True, exist_ok=True)
    cells = load_cells(manifest)
    cells.to_csv(output / "ncmapss_dev_rotation_cells.csv", index=False)
    summary = summarize(cells)
    summary.to_csv(output / "ncmapss_dev_rotation_summary.csv", index=False)
    intervals = crossed_bootstrap(cells, args.bootstrap_reps)
    intervals.to_csv(output / "ncmapss_dev_rotation_paired_intervals.csv", index=False)
    plot_unit_results(cells, figures / "fig_ncmapss_dev_rotation")
    print(
        f"NCMAPSS_DEV_ROTATION_ANALYSIS_READY cells={len(cells)} "
        f"summary={len(summary)} intervals={len(intervals)}"
    )


if __name__ == "__main__":
    main()
