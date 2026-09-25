from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).absolute().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mpl_cache"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import norm, studentized_range

from rul.config import load_config
from run_formal_benchmark import FORMAL_BENCHMARK_MODELS, FORMAL_BENCHMARK_SEEDS, FORMAL_BENCHMARK_SUBSETS


plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 10.5,
        "axes.titlesize": 12.0,
        "axes.labelsize": 11.0,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9.5,
        "legend.fontsize": 9.5,
        "lines.linewidth": 1.8,
        "savefig.dpi": 400,
    }
)


MODEL_DISPLAY = {
    "gru": "GRU",
    "lstm": "LSTM",
    "tcn": "TCN",
    "tcn_gru": "TCN-GRU",
    "rast_gru": "RAST-GRU",
    "cnn_lstm": "CNN-LSTM",
    "attention_gru": "Attention-GRU",
    "bigru_attention": "BiGRU-Attention",
    "transformer_lite": "Transformer-lite",
    "dual_attention_tcn": "Dual-attention TCN",
    "sensor_graph_gru": "SensorGraph-GRU",
    "quantile_gru": "Quantile-GRU",
    "rast_gru_v2": "OCM-MST-GRU",
}

METRIC_LABELS = {
    "rmse": "RMSE",
    "mae": "MAE",
    "nasa_per_engine": "NASA/engine",
    "lpr30": "LPR@30",
}

PRACTICAL_THRESHOLDS = {
    "rmse": 0.5,
    "mae": 0.5,
    "nasa_per_engine": 1.0,
    "lpr30": 0.02,
}

INTERVAL_MODELS = ("quantile_gru", "rast_gru_v2")


def nasa_contribution(error: np.ndarray) -> np.ndarray:
    return np.where(error < 0.0, np.exp(-error / 13.0) - 1.0, np.exp(error / 10.0) - 1.0)


def prediction_metrics(frame: pd.DataFrame) -> dict[str, float]:
    error = frame["pred_rul"].to_numpy(float) - frame["true_rul"].to_numpy(float)
    critical = frame["true_rul"].to_numpy(float) <= 30.0
    critical_error = error[critical]
    positive_late_error = critical_error[critical_error > 0.0]
    late_excess = np.maximum(critical_error, 0.0)
    q95 = float(np.quantile(positive_late_error, 0.95)) if len(positive_late_error) else 0.0
    cvar95 = (
        float(np.mean(positive_late_error[positive_late_error >= q95]))
        if len(positive_late_error)
        else 0.0
    )
    return {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "nasa_per_engine": float(np.mean(nasa_contribution(error))),
        "lpr30": float(np.mean(error[critical] > 0.0)) if np.any(critical) else float("nan"),
        "mle30": float(np.mean(late_excess)) if np.any(critical) else float("nan"),
        "conditional_mle30": float(np.mean(positive_late_error)) if len(positive_late_error) else 0.0,
        "late_q90_30": float(np.quantile(positive_late_error, 0.90)) if len(positive_late_error) else 0.0,
        "late_q95_30": q95,
        "late_cvar95_30": cvar95,
        "engine_count": int(len(frame)),
        "critical_engine_count": int(np.sum(critical)),
    }


def load_prediction_grid(results_dir: Path) -> dict[tuple[int, str, str], pd.DataFrame]:
    grid: dict[tuple[int, str, str], pd.DataFrame] = {}
    missing: list[Path] = []
    for seed in FORMAL_BENCHMARK_SEEDS:
        for subset in FORMAL_BENCHMARK_SUBSETS:
            for model in FORMAL_BENCHMARK_MODELS:
                path = results_dir / f"paper_main_v3_seed{seed}" / subset / model / "test_predictions.csv"
                if not path.exists():
                    missing.append(path)
                    continue
                frame = pd.read_csv(path)
                required = {"unit_id", "true_rul", "pred_rul"}
                if not required.issubset(frame.columns):
                    raise ValueError(f"Prediction file lacks required columns: {path}")
                grid[(seed, subset, model)] = frame[list(required)].copy()
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} formal prediction files; first missing file: {missing[0]}")
    return grid


def load_interval_grid(results_dir: Path) -> dict[tuple[int, str, str], pd.DataFrame]:
    grid: dict[tuple[int, str, str], pd.DataFrame] = {}
    missing: list[Path] = []
    required = {"unit_id", "true_rul", "q10_rul", "q50_rul", "q90_rul"}
    for seed in FORMAL_BENCHMARK_SEEDS:
        for subset in FORMAL_BENCHMARK_SUBSETS:
            for model in INTERVAL_MODELS:
                path = results_dir / f"paper_main_v3_seed{seed}" / subset / model / "test_interval_predictions.csv"
                if not path.exists():
                    missing.append(path)
                    continue
                frame = pd.read_csv(path)
                if not required.issubset(frame.columns):
                    raise ValueError(f"Interval file lacks required columns: {path}")
                if np.any(frame["q10_rul"].to_numpy(float) > frame["q50_rul"].to_numpy(float)) or np.any(
                    frame["q50_rul"].to_numpy(float) > frame["q90_rul"].to_numpy(float)
                ):
                    raise ValueError(f"Crossing quantiles found in {path}")
                grid[(seed, subset, model)] = frame[list(required)].copy()
    if missing:
        raise FileNotFoundError(f"Missing {len(missing)} interval files; first missing file: {missing[0]}")
    return grid


def build_interval_calibration(
    grid: dict[tuple[int, str, str], pd.DataFrame], alpha: float = 0.20
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for (seed, subset, model), frame in grid.items():
        y = frame["true_rul"].to_numpy(float)
        lower = frame["q10_rul"].to_numpy(float)
        median = frame["q50_rul"].to_numpy(float)
        upper = frame["q90_rul"].to_numpy(float)
        width = upper - lower
        covered = (y >= lower) & (y <= upper)
        interval_score = width + (2.0 / alpha) * np.maximum(lower - y, 0.0) + (2.0 / alpha) * np.maximum(y - upper, 0.0)
        critical = y <= 30.0
        rows.append(
            {
                "seed": seed,
                "subset": subset,
                "model": model,
                "model_display": MODEL_DISPLAY[model],
                "nominal_coverage": 1.0 - alpha,
                "empirical_coverage": float(np.mean(covered)),
                "absolute_coverage_error": float(abs(np.mean(covered) - (1.0 - alpha))),
                "mean_interval_width": float(np.mean(width)),
                "normalized_mean_width": float(np.mean(width) / 125.0),
                "mean_interval_score": float(np.mean(interval_score)),
                "median_rmse": float(np.sqrt(np.mean((median - y) ** 2))),
                "critical_coverage": float(np.mean(covered[critical])) if np.any(critical) else float("nan"),
                "critical_mean_width": float(np.mean(width[critical])) if np.any(critical) else float("nan"),
                "engine_count": int(len(frame)),
                "critical_engine_count": int(np.sum(critical)),
            }
        )
    long = pd.DataFrame(rows)
    seed_macro = (
        long.groupby(["seed", "model", "model_display"], as_index=False)[
            [
                "empirical_coverage",
                "absolute_coverage_error",
                "mean_interval_width",
                "normalized_mean_width",
                "mean_interval_score",
                "median_rmse",
                "critical_coverage",
                "critical_mean_width",
            ]
        ]
        .mean()
    )
    summary = seed_macro.groupby(["model", "model_display"], as_index=False).agg(
        seed_count=("seed", "nunique"),
        **{
            f"{column}_mean": (column, "mean")
            for column in seed_macro.columns
            if column not in {"seed", "model", "model_display"}
        },
        **{
            f"{column}_std": (column, lambda x: x.std(ddof=1))
            for column in seed_macro.columns
            if column not in {"seed", "model", "model_display"}
        },
    )
    return long, summary


def build_aggregation(grid: dict[tuple[int, str, str], pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    per_subset_rows: list[dict[str, object]] = []
    seed_rows: list[dict[str, object]] = []
    for seed in FORMAL_BENCHMARK_SEEDS:
        for model in FORMAL_BENCHMARK_MODELS:
            subset_metrics = []
            pooled_parts = []
            for subset in FORMAL_BENCHMARK_SUBSETS:
                frame = grid[(seed, subset, model)]
                metrics = prediction_metrics(frame)
                subset_metrics.append(metrics)
                pooled_parts.append(frame.assign(subset=subset))
                per_subset_rows.append(
                    {"seed": seed, "subset": subset, "model": model, "model_display": MODEL_DISPLAY[model], **metrics}
                )
            pooled = prediction_metrics(pd.concat(pooled_parts, ignore_index=True))
            row: dict[str, object] = {"seed": seed, "model": model, "model_display": MODEL_DISPLAY[model]}
            for metric in [
                "rmse",
                "mae",
                "nasa_per_engine",
                "lpr30",
                "mle30",
                "conditional_mle30",
                "late_q90_30",
                "late_q95_30",
                "late_cvar95_30",
            ]:
                row[f"macro_{metric}"] = float(np.nanmean([item[metric] for item in subset_metrics]))
                row[f"pooled_{metric}"] = float(pooled[metric])
            row["pooled_engine_count"] = pooled["engine_count"]
            row["pooled_critical_engine_count"] = pooled["critical_engine_count"]
            seed_rows.append(row)

    per_subset = pd.DataFrame(per_subset_rows)
    per_seed = pd.DataFrame(seed_rows)
    summary = per_seed.groupby(["model", "model_display"], as_index=False).agg(
        **{
            f"{column}_mean": (column, "mean")
            for column in per_seed.columns
            if column.startswith("macro_") or column.startswith("pooled_") and not column.endswith("count")
        },
        **{
            f"{column}_std": (column, lambda x: x.std(ddof=1))
            for column in per_seed.columns
            if column.startswith("macro_") or column.startswith("pooled_") and not column.endswith("count")
        },
    )
    return per_subset, summary


def build_rank_tables(per_subset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    for (seed, subset), group in per_subset.groupby(["seed", "subset"]):
        for metric in ["rmse", "nasa_per_engine", "lpr30"]:
            ranked = group[["model", "model_display", metric]].copy()
            ranked["rank"] = ranked[metric].rank(method="average", ascending=True)
            for item in ranked.to_dict("records"):
                rows.append({"seed": seed, "subset": subset, "metric": metric, **item})
    long = pd.DataFrame(rows)
    summary = long.groupby(["metric", "model", "model_display"], as_index=False).agg(
        average_rank=("rank", "mean"),
        rank_std=("rank", lambda x: x.std(ddof=1)),
        task_count=("rank", "count"),
    )
    composite = summary.groupby(["model", "model_display"], as_index=False)["average_rank"].mean()
    composite = composite.rename(columns={"average_rank": "average_rank_across_metrics"})
    summary = summary.merge(composite, on=["model", "model_display"], how="left")
    return long, summary


def plot_critical_difference(rank_summary: pd.DataFrame, output: Path) -> None:
    metrics = ["rmse", "nasa_per_engine", "lpr30"]
    k = len(FORMAL_BENCHMARK_MODELS)
    n = len(FORMAL_BENCHMARK_SEEDS) * len(FORMAL_BENCHMARK_SUBSETS)
    q_alpha = float(studentized_range.ppf(0.95, k, np.inf) / math.sqrt(2.0))
    cd = q_alpha * math.sqrt(k * (k + 1) / (6.0 * n))
    fig, axes = plt.subplots(3, 1, figsize=(11.8, 9.0), sharex=True)
    for ax, metric in zip(axes, metrics):
        sub = rank_summary[rank_summary["metric"] == metric].sort_values("average_rank")
        y = np.arange(len(sub))
        ax.scatter(sub["average_rank"], y, s=55, color="#1f77b4", zorder=3)
        for yi, row in zip(y, sub.itertuples(index=False)):
            ax.text(float(row.average_rank) + 0.08, yi, str(row.model_display), va="center", fontsize=9)
        ax.set_yticks([])
        ax.set_xlim(0.8, k + 0.8)
        ax.set_title(f"{METRIC_LABELS[metric]} average rank over {n} subset-seed tasks", fontsize=12)
        ax.grid(axis="x", linestyle=":", alpha=0.35)
        ax.plot([1.0, 1.0 + cd], [-0.8, -0.8], color="black", linewidth=2.2)
        ax.plot([1.0, 1.0], [-1.0, -0.6], color="black", linewidth=1.5)
        ax.plot([1.0 + cd, 1.0 + cd], [-1.0, -0.6], color="black", linewidth=1.5)
        ax.text(1.0 + cd / 2.0, -1.15, f"Nemenyi CD={cd:.2f}", ha="center", fontsize=9)
    axes[-1].set_xlabel("Average rank (lower is better)", fontsize=11)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def build_threshold_rank_stability(threshold_path: Path) -> pd.DataFrame:
    threshold = pd.read_csv(threshold_path)
    threshold = threshold[threshold["subset"] == "FD004"].copy()
    rows = []
    for (metric_name, tau, delta), group in threshold.groupby(
        ["metric_name", "rul_threshold", "late_error_threshold"], sort=True
    ):
        group = group.copy()
        group["rank"] = pd.to_numeric(group["ratio_mean"], errors="coerce").rank(method="average")
        label = f"{metric_name}; tau={int(float(tau))}; delta={int(float(delta))}"
        for row in group.itertuples(index=False):
            rows.append(
                {
                    "threshold_definition": label,
                    "metric_name": metric_name,
                    "rul_threshold": float(tau),
                    "late_error_threshold": float(delta),
                    "model": row.model,
                    "model_display": MODEL_DISPLAY.get(row.model, row.model),
                    "ratio_mean": float(row.ratio_mean),
                    "rank": float(row.rank),
                }
            )
    return pd.DataFrame(rows)


def plot_threshold_heatmap(rank_long: pd.DataFrame, output: Path) -> None:
    pivot = rank_long.pivot(index="threshold_definition", columns="model_display", values="rank")
    ordered = [MODEL_DISPLAY[model] for model in FORMAL_BENCHMARK_MODELS]
    pivot = pivot[[column for column in ordered if column in pivot.columns]]
    fig, ax = plt.subplots(figsize=(13.5, 8.0))
    sns.heatmap(pivot, annot=True, fmt=".1f", cmap="viridis_r", vmin=1, vmax=len(ordered), ax=ax, cbar_kws={"label": "Rank"})
    ax.set_xlabel("")
    ax.set_ylabel("FD004 threshold definition")
    ax.set_title("Critical-zone rank stability across LPR/SLPR thresholds")
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_method_architecture(output: Path) -> None:
    # Final-size, vector-first schematic for a 183-mm double-column figure.
    fig, ax = plt.subplots(figsize=(183 / 25.4, 116 / 25.4))
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 66)
    ax.axis("off")

    ink = "#252525"
    neutral = "#6F7480"
    design = "#E7EEF7"
    design_edge = "#5379A5"
    model = "#E5F1EC"
    model_edge = "#39745F"
    validation = "#FFF1D6"
    validation_edge = "#B17822"
    evaluation = "#F1EAF5"
    evaluation_edge = "#7A5B89"

    def box(x: float, y: float, w: float, h: float, text: str, face: str, edge: str,
            *, fontsize: float = 7.0, weight: str = "normal") -> None:
        ax.add_patch(
            FancyBboxPatch(
                (x, y), w, h,
                boxstyle="round,pad=0.16,rounding_size=0.45",
                linewidth=0.85, edgecolor=edge, facecolor=face,
            )
        )
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fontsize, fontweight=weight, color=ink, linespacing=1.18)

    def arrow(start: tuple[float, float], end: tuple[float, float], *, color: str = neutral,
              style: str = "-|>", dashed: bool = False) -> None:
        ax.add_patch(
            FancyArrowPatch(
                start, end, arrowstyle=style, mutation_scale=8.5, linewidth=0.9,
                color=color, linestyle="--" if dashed else "-", shrinkA=2, shrinkB=2,
                connectionstyle="arc3,rad=0.0",
            )
        )

    # Stage headers establish the scientific workflow rather than a decorative model stack.
    for x, w, letter, title, color in [
        (1.0, 22.0, "a", "Study design", design_edge),
        (25.5, 44.0, "b", "Model estimation", model_edge),
        (72.0, 27.0, "c", "Fixed test evaluation", evaluation_edge),
    ]:
        ax.text(x, 63.2, letter, fontsize=8.4, fontweight="bold", color=ink, va="center")
        ax.text(x + 3.2, 63.2, title, fontsize=8.4, fontweight="bold", color=color, va="center")
        ax.plot([x, x + w], [60.8, 60.8], color=color, linewidth=1.2)

    box(1.0, 49.0, 22.0, 8.0, "C-MAPSS FD001-FD004\nN-CMAPSS DS02 fixed task", design, design_edge, weight="bold")
    box(1.0, 36.5, 22.0, 8.0, "Engine/unit split\ntraining core | validation | test", "white", design_edge)
    box(1.0, 24.0, 22.0, 8.0, "Fixed study protocol\ncomposite levels\nRUL cap | window | budget", "white", design_edge)
    arrow((12.0, 49.0), (12.0, 44.5), color=design_edge)
    arrow((12.0, 36.5), (12.0, 32.0), color=design_edge)

    # Main estimation path.
    box(26.0, 49.0, 12.0, 8.0, "Train-only\ncondition scaling", model, model_edge)
    box(41.0, 49.0, 12.0, 8.0, "Values + masks\nsummary statistics", model, model_edge)
    box(56.0, 49.0, 13.0, 8.0, "Mask-aware\ncontext weighting", model, model_edge)
    arrow((38.0, 53.0), (41.0, 53.0), color=model_edge)
    arrow((53.0, 53.0), (56.0, 53.0), color=model_edge)

    box(32.0, 36.5, 14.0, 8.0, "Local-trend path\n+ GRU encoder", "white", model_edge)
    box(50.0, 36.5, 14.0, 8.0, "Attention / last-state\nfusion", "white", model_edge)
    arrow((62.5, 49.0), (44.0, 44.5), color=model_edge)
    arrow((46.0, 40.5), (50.0, 40.5), color=model_edge)

    box(
        43.0,
        24.0,
        18.0,
        8.0,
        "RUL outputs\nlower | point | upper",
        model,
        model_edge,
        weight="bold",
    )
    arrow((57.0, 36.5), (54.0, 32.0), color=model_edge)

    # Two estimation controls: objective and validation-only selection.
    box(26.0, 9.5, 19.0, 9.0, "Training objective\nBase + Late + Cons\n+ Rel + Endpoints", "white", model_edge)
    box(49.0, 9.5, 20.0, 9.0, "One fixed endpoint / val engine\nRMSE/RULcap + 0.25 LPR\n+ 0.25 SLPR", validation, validation_edge)
    arrow((47.0, 24.0), (35.5, 18.5), color=model_edge)
    arrow((57.0, 24.0), (59.0, 18.5), color=validation_edge)
    ax.text(59.0, 6.8, "validation only", ha="center", fontsize=6.5,
            color=validation_edge, fontweight="bold")

    # Evaluation path starts only after checkpoint freezing.
    box(73.0, 49.0, 25.0, 8.0, "Freeze selected checkpoint\nno test-time refitting", evaluation, evaluation_edge, weight="bold")
    box(73.0, 36.5, 25.0, 8.0, "Clean engine endpoints\n+ matched normalized perturbations", "white", evaluation_edge)
    box(73.0, 24.0, 25.0, 8.0, "Accuracy | late-overprediction | tail error\nasymmetric error cost | intervals", "white", evaluation_edge)
    box(73.0, 9.5, 25.0, 9.0, "Paired estimation + ablation\nprotocol sensitivity + computational proxy", evaluation, evaluation_edge)
    arrow((69.0, 14.0), (76.0, 49.0), color=validation_edge)
    arrow((85.5, 49.0), (85.5, 44.5), color=evaluation_edge)
    arrow((85.5, 36.5), (85.5, 32.0), color=evaluation_edge)
    arrow((85.5, 24.0), (85.5, 18.5), color=evaluation_edge)

    # Dashed barrier makes leakage control visible and reviewable.
    ax.plot([70.7, 70.7], [6.5, 59.5], color="#9A9A9A", linewidth=0.8, linestyle=(0, (2, 2)))
    ax.text(70.7, 5.0, "test boundary", ha="center", va="center", fontsize=6.3, color=neutral)
    ax.text(49.5, 2.0,
            "Test labels and perturbation outcomes never enter fitting, checkpoint selection, or hyperparameter choice.",
            ha="center", va="center", fontsize=6.6, color=ink)

    fig.subplots_adjust(left=0.01, right=0.99, bottom=0.02, top=0.99)
    fig.savefig(output.with_suffix(".svg"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output.with_suffix(".tiff"), dpi=600, bbox_inches="tight", pad_inches=0.03)
    fig.savefig(output.with_suffix(".png"), dpi=400, bbox_inches="tight", pad_inches=0.03)
    plt.close(fig)


def is_pareto_efficient(values: np.ndarray) -> np.ndarray:
    efficient = np.ones(len(values), dtype=bool)
    for i, point in enumerate(values):
        if not efficient[i]:
            continue
        dominated_by_other = np.any(np.all(values <= point, axis=1) & np.any(values < point, axis=1))
        if dominated_by_other:
            efficient[i] = False
    return efficient


def dominated_hypervolume_3d(values: np.ndarray, reference: np.ndarray | None = None) -> float:
    """Exact union volume of minimization boxes [point, reference] in three dimensions."""
    points = np.asarray(values, dtype=float)
    ref = np.asarray(reference if reference is not None else [1.1, 1.1, 1.1], dtype=float)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Hypervolume requires an n by 3 objective matrix.")
    points = points[np.all(points < ref, axis=1)]
    if len(points) == 0:
        return 0.0

    volume = 0.0
    x_breaks = np.unique(np.r_[points[:, 0], ref[0]])
    for x0, x1 in zip(x_breaks[:-1], x_breaks[1:]):
        active_x = points[points[:, 0] <= x0 + 1e-12]
        if len(active_x) == 0:
            continue
        area = 0.0
        y_breaks = np.unique(np.r_[active_x[:, 1], ref[1]])
        for y0, y1 in zip(y_breaks[:-1], y_breaks[1:]):
            active_y = active_x[active_x[:, 1] <= y0 + 1e-12]
            if len(active_y) == 0:
                continue
            area += float(y1 - y0) * float(ref[2] - np.min(active_y[:, 2]))
        volume += float(x1 - x0) * area
    return float(max(volume, 0.0))


def build_seed_pareto_stability(per_subset: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    objective_columns = ["rmse", "nasa_per_engine", "lpr30"]
    for seed, seed_frame in per_subset.groupby("seed"):
        macro = seed_frame.groupby(["model", "model_display"], as_index=False)[objective_columns].mean()
        normalized_parts = []
        for column in objective_columns:
            values = macro[column].to_numpy(float)
            spread = float(values.max() - values.min())
            normalized_parts.append((values - values.min()) / spread if spread else np.zeros_like(values))
        objective = np.column_stack(normalized_parts)
        efficient = is_pareto_efficient(objective)
        total_hv = dominated_hypervolume_3d(objective)
        for index, item in macro.iterrows():
            without = np.delete(objective, index, axis=0)
            exclusive = max(total_hv - dominated_hypervolume_3d(without), 0.0)
            rows.append(
                {
                    "seed": int(seed),
                    "model": item["model"],
                    "model_display": item["model_display"],
                    "pareto_efficient": bool(efficient[index]),
                    "distance_to_ideal": float(np.sqrt(np.mean(objective[index] ** 2))),
                    "set_hypervolume": total_hv,
                    "exclusive_hypervolume_contribution": exclusive,
                    "hypervolume_reference": "(1.1,1.1,1.1) after within-seed min-max normalization",
                }
            )
    long = pd.DataFrame(rows)
    summary = long.groupby(["model", "model_display"], as_index=False).agg(
        seed_count=("seed", "nunique"),
        pareto_seed_frequency=("pareto_efficient", "mean"),
        distance_to_ideal_mean=("distance_to_ideal", "mean"),
        distance_to_ideal_std=("distance_to_ideal", lambda x: x.std(ddof=1)),
        exclusive_hypervolume_mean=("exclusive_hypervolume_contribution", "mean"),
        exclusive_hypervolume_std=("exclusive_hypervolume_contribution", lambda x: x.std(ddof=1)),
    )
    return long, summary


def build_pareto_analysis(aggregation: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    columns = ["macro_rmse_mean", "pooled_nasa_per_engine_mean", "macro_lpr30_mean"]
    pareto = aggregation[["model", "model_display", *columns]].copy()
    normalized = []
    for column in columns:
        values = pareto[column].to_numpy(float)
        spread = float(values.max() - values.min())
        normalized.append((values - values.min()) / spread if spread else np.zeros_like(values))
    objective = np.column_stack(normalized)
    pareto["pareto_efficient"] = is_pareto_efficient(objective)
    pareto["distance_to_ideal"] = np.sqrt(np.mean(objective**2, axis=1))

    win_counts = np.zeros(len(pareto), dtype=int)
    total = 0
    for a in range(21):
        for b in range(21 - a):
            c = 20 - a - b
            weights = np.asarray([a, b, c], dtype=float) / 20.0
            if np.allclose(weights, 0.0):
                continue
            scores = objective @ weights
            winners = np.flatnonzero(np.isclose(scores, scores.min(), atol=1e-12))
            win_counts[winners] += 1
            total += 1
    pareto["simplex_weight_optimal_share"] = win_counts / max(total, 1)
    pareto = pareto.sort_values(["pareto_efficient", "distance_to_ideal"], ascending=[False, True]).reset_index(drop=True)
    weights = pd.DataFrame(
        {
            "model": pareto["model"],
            "model_display": pareto["model_display"],
            "simplex_weight_optimal_share": pareto["simplex_weight_optimal_share"],
        }
    )
    return pareto, weights


def plot_interval_calibration(summary: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    for row in summary.to_dict("records"):
        ax.errorbar(
            row["mean_interval_width_mean"],
            row["empirical_coverage_mean"],
            xerr=row["mean_interval_width_std"],
            yerr=row["empirical_coverage_std"],
            marker="o",
            markersize=7,
            capsize=3,
            linewidth=1.5,
            label=row["model_display"],
        )
    ax.axhline(0.80, color="#555555", linestyle="--", linewidth=1.2, label="Nominal 80%")
    ax.set_xlabel("Mean q10-q90 interval width (cycles)")
    ax.set_ylabel("Empirical coverage")
    ax.set_title("Predictive RUL interval coverage diagnostic")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def build_fairness_table(results_dir: Path) -> pd.DataFrame:
    rows = []
    for model in FORMAL_BENCHMARK_MODELS:
        metrics_rows = []
        for seed in FORMAL_BENCHMARK_SEEDS:
            path = results_dir / f"paper_main_v3_seed{seed}" / "FD004" / model / "metrics.json"
            metrics_rows.append(json.loads(path.read_text(encoding="utf-8-sig")))
        first_dir = results_dir / f"paper_main_v3_seed{FORMAL_BENCHMARK_SEEDS[0]}" / "FD004" / model
        cfg = load_config(first_dir / "run_config.yaml")
        rows.append(
            {
                "model": model,
                "model_display": MODEL_DISPLAY[model],
                "parameters_fd004": int(metrics_rows[0]["parameters"]),
                "profiled_flops_batch1": int(metrics_rows[0].get("profiled_flops_batch1", 0)),
                "gpu_ms_sample_mean": float(np.mean([row["single_sample_inference_ms"] for row in metrics_rows])),
                "gpu_ms_sample_std": float(np.std([row["single_sample_inference_ms"] for row in metrics_rows], ddof=1)),
                "gpu_ms_batch128_mean": float(np.mean([row["batch_inference_ms"] for row in metrics_rows])),
                "cpu_ms_sample_mean": float(np.mean([row["cpu_single_sample_inference_ms"] for row in metrics_rows])),
                "cpu_ms_sample_std": float(np.std([row["cpu_single_sample_inference_ms"] for row in metrics_rows], ddof=1)),
                "peak_gpu_memory_mb_mean": float(np.mean([row.get("peak_gpu_memory_mb", np.nan) for row in metrics_rows])),
                "peak_gpu_memory_mb_std": float(np.std([row.get("peak_gpu_memory_mb", np.nan) for row in metrics_rows], ddof=1)),
                "training_minutes_mean": float(
                    np.mean([row.get("training_elapsed_sec", np.nan) for row in metrics_rows]) / 60.0
                ),
                "training_minutes_std": float(
                    np.std([row.get("training_elapsed_sec", np.nan) for row in metrics_rows], ddof=1) / 60.0
                ),
                "preprocessing_seconds_mean": float(np.mean([row.get("preprocessing_elapsed_sec", np.nan) for row in metrics_rows])),
                "planned_epochs": int(cfg["training"]["epochs"]),
                "completed_epochs_mean": float(np.mean([row["completed_epochs"] for row in metrics_rows])),
                "checkpoint_selection": cfg["training"]["selection_metric"],
                "mask_channels": bool(cfg["data"]["append_missing_mask"]),
                "receives_mask_channels": bool(cfg["data"]["append_missing_mask"]),
                "explicit_mask_processing_module": model == "rast_gru_v2",
                "augmentation_profile": cfg["augmentation"]["profile"],
                "uses_shared_degradation_augmentation": cfg["augmentation"]["profile"] != "none",
                "training_loss": cfg["training"]["loss"],
                "shared_base_loss": cfg["training"]["loss"],
                "smooth_risk_regularization": model == "rast_gru_v2",
                "reliability_self_supervision": model == "rast_gru_v2",
                "consistency_regularization": model == "rast_gru_v2",
                "quantile_calibration": model in {"quantile_gru", "rast_gru_v2"},
                "additional_objective_terms": (
                    "smooth late risk; reliability BCE; two-view consistency; pinball"
                    if model == "rast_gru_v2"
                    else "q10/q90 pinball"
                    if model == "quantile_gru"
                    else "none"
                ),
                "objective_code": (
                    "Base+Risk+Rel+Cons+Q"
                    if model == "rast_gru_v2"
                    else "Base+Q"
                    if model == "quantile_gru"
                    else "Base"
                ),
                "hyperparameter_budget": "one shared validation-selected training configuration; no model-specific test tuning",
                "benchmark_track": "controlled common-protocol comparison",
                "seed_interpretation": "composite seed controls engine split, initialization, shuffling, and augmentation",
            }
        )
    return pd.DataFrame(rows)


def build_late_tail_summary(per_subset: pd.DataFrame) -> pd.DataFrame:
    metrics = ["mle30", "conditional_mle30", "late_q90_30", "late_q95_30", "late_cvar95_30"]
    per_seed = per_subset.groupby(["seed", "model", "model_display"], as_index=False)[metrics].mean()
    return per_seed.groupby(["model", "model_display"], as_index=False).agg(
        seed_count=("seed", "nunique"),
        **{f"macro_{metric}_mean": (metric, "mean") for metric in metrics},
        **{f"macro_{metric}_std": (metric, lambda x: x.std(ddof=1)) for metric in metrics},
    )


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build_figure_data_manifest(output_dir: Path) -> dict[str, object]:
    mapping = {
        "fig_critical_difference.pdf": ["benchmark_rank_summary.csv"],
        "fig_fd004_threshold_rank_heatmap.pdf": ["threshold_rank_stability_fd004.csv"],
        "fig_decision_cost_sensitivity.pdf": ["decision_cost_sensitivity.csv"],
        "fig_core_asym_decision_cost.pdf": ["core_asym_decision_cost_bootstrap.csv"],
        "fig_interval_calibration.pdf": ["interval_calibration_summary.csv"],
        "fig_method_architecture.pdf": [],
        "fig_mechanism_diagnostics.pdf": [
            "reliability_fault_localization_summary.csv",
            "attention_mechanism_seed_level.csv",
            "reliability_calibration_bins.csv",
        ],
        "fig_condition_normalization_pca.pdf": ["condition_normalization_diagnostics.csv"],
        "fig_fd004_failure_cases.pdf": ["failure_case_engine_level.csv"],
        "fig_conformal_interval_calibration.pdf": ["conformal_interval_summary.csv"],
        "fig_stress_nine_family_heatmap.pdf": [
            "stress_pair_extended_auc_comparison.csv",
            "stress_pair_extended_worst_comparison.csv",
        ],
        "fig_stress_global_missing_failure.pdf": ["stress_global_missing_failure_curve.csv"],
        "fig_interval_reliability_curve.pdf": ["interval_reliability_summary.csv"],
        "fig_validation_endpoint_distribution.pdf": [
            "validation_endpoint_distribution_targets.csv",
            "validation_endpoint_distribution_summary.csv",
        ],
        "fig_cross_backbone_protocol_buildup.pdf": [
            "cross_backbone_protocol_buildup_seed_metrics.csv",
            "cross_backbone_protocol_buildup_summary.csv",
        ],
        "fig_stress_operating_points.pdf": [
            "stress_operating_points_curve_means.csv",
            "stress_operating_points_uncertainty.csv",
        ],
    }
    rows = []
    for figure_name, source_names in mapping.items():
        figure_path = output_dir / "figures" / figure_name
        rows.append(
            {
                "figure": figure_name,
                "path": str(figure_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "figure_sha256": _sha256(figure_path) if figure_path.exists() else None,
                "source_files": [
                    {
                        "name": source_name,
                        "sha256": _sha256(output_dir / source_name) if (output_dir / source_name).exists() else None,
                        "rows": int(len(pd.read_csv(output_dir / source_name)))
                        if (output_dir / source_name).exists()
                        else None,
                    }
                    for source_name in source_names
                ],
            }
        )
    for figure_name in [
        "fig_fd004_random_missing_rmse.pdf",
        "fig_fd004_random_missing_lpr30.pdf",
        "fig_fd004_global_missing_rmse.pdf",
        "fig_fd004_block_missing_lpr30.pdf",
        "fig_fd004_drift_lpr30.pdf",
        "fig_fd004_drift_nasa.pdf",
    ]:
        figure_path = PROJECT_ROOT / "paper_outputs" / "revised_figures" / figure_name
        source_path = output_dir / "stress_multiseed_summary.csv"
        rows.append(
            {
                "figure": figure_name,
                "path": str(figure_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
                "figure_sha256": _sha256(figure_path) if figure_path.exists() else None,
                "source_files": [
                    {
                        "name": source_path.name,
                        "sha256": _sha256(source_path) if source_path.exists() else None,
                        "rows": int(len(pd.read_csv(source_path))) if source_path.exists() else None,
                    }
                ],
            }
        )
    return {
        "protocol_version": "3.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "figures": rows,
    }


def build_decision_cost_sensitivity(
    grid: dict[tuple[int, str, str], pd.DataFrame],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    late_cost_ratios = [1.0, 2.0, 5.0, 10.0]
    task_rows = []
    for late_cost in late_cost_ratios:
        for seed in FORMAL_BENCHMARK_SEEDS:
            for subset in FORMAL_BENCHMARK_SUBSETS:
                for model in FORMAL_BENCHMARK_MODELS:
                    frame = grid[(seed, subset, model)]
                    error = frame["pred_rul"].to_numpy(float) - frame["true_rul"].to_numpy(float)
                    cost = late_cost * np.maximum(error, 0.0) + np.maximum(-error, 0.0)
                    task_rows.append(
                        {
                            "late_to_early_cost_ratio": late_cost,
                            "seed": seed,
                            "subset": subset,
                            "model": model,
                            "model_display": MODEL_DISPLAY[model],
                            "mean_asymmetric_cycle_cost": float(np.mean(cost)),
                        }
                    )
    tasks = pd.DataFrame(task_rows)
    tasks["task_rank"] = tasks.groupby(["late_to_early_cost_ratio", "seed", "subset"])[
        "mean_asymmetric_cycle_cost"
    ].rank(method="average")
    summary = tasks.groupby(
        ["late_to_early_cost_ratio", "model", "model_display"], as_index=False
    ).agg(
        mean_asymmetric_cycle_cost=("mean_asymmetric_cycle_cost", "mean"),
        std_asymmetric_cycle_cost=("mean_asymmetric_cycle_cost", lambda x: x.std(ddof=1)),
        average_rank=("task_rank", "mean"),
        task_count=("task_rank", "count"),
    )
    return tasks, summary


def plot_decision_cost_sensitivity(summary: pd.DataFrame, output: Path) -> None:
    pivot = summary.pivot(
        index="late_to_early_cost_ratio", columns="model_display", values="average_rank"
    )
    ordered = [MODEL_DISPLAY[model] for model in FORMAL_BENCHMARK_MODELS]
    pivot = pivot[[column for column in ordered if column in pivot.columns]]
    fig, ax = plt.subplots(figsize=(15.0, 5.2))
    sns.heatmap(
        pivot,
        annot=True,
        fmt=".1f",
        cmap="viridis_r",
        vmin=1,
        vmax=len(ordered),
        linewidths=0.4,
        ax=ax,
        cbar_kws={"label": "Average rank"},
    )
    ax.set_xlabel("")
    ax.set_ylabel("Late-to-early cycle-cost ratio")
    ax.set_title("Decision-cost rank sensitivity over subset-seed tasks")
    ax.tick_params(axis="x", rotation=35)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def paired_metric(frame: pd.DataFrame, metric: str, indices: np.ndarray | None = None) -> float:
    sample = frame.iloc[indices] if indices is not None else frame
    baseline_error = sample["pred_rul_baseline"].to_numpy(float) - sample["true_rul"].to_numpy(float)
    proposed_error = sample["pred_rul_proposed"].to_numpy(float) - sample["true_rul"].to_numpy(float)
    if metric == "rmse":
        return float(np.sqrt(np.mean(baseline_error**2)) - np.sqrt(np.mean(proposed_error**2)))
    if metric == "mae":
        return float(np.mean(np.abs(baseline_error)) - np.mean(np.abs(proposed_error)))
    if metric == "nasa_per_engine":
        return float(np.mean(nasa_contribution(baseline_error) - nasa_contribution(proposed_error)))
    if metric == "lpr30":
        zone = sample["true_rul"].to_numpy(float) <= 30.0
        return float(np.mean(baseline_error[zone] > 0.0) - np.mean(proposed_error[zone] > 0.0))
    raise ValueError(f"Unknown metric: {metric}")


def paired_metric_arrays(
    true_rul: np.ndarray,
    baseline_error: np.ndarray,
    proposed_error: np.ndarray,
    metric: str,
    indices: np.ndarray | None = None,
) -> float:
    if indices is not None:
        true_rul = true_rul[indices]
        baseline_error = baseline_error[indices]
        proposed_error = proposed_error[indices]
    if metric == "rmse":
        return float(np.sqrt(np.mean(baseline_error**2)) - np.sqrt(np.mean(proposed_error**2)))
    if metric == "mae":
        return float(np.mean(np.abs(baseline_error)) - np.mean(np.abs(proposed_error)))
    if metric == "nasa_per_engine":
        return float(np.mean(nasa_contribution(baseline_error) - nasa_contribution(proposed_error)))
    if metric == "lpr30":
        zone = true_rul <= 30.0
        return float(np.mean(baseline_error[zone] > 0.0) - np.mean(proposed_error[zone] > 0.0))
    raise ValueError(f"Unknown metric: {metric}")


def bca_interval(bootstrap: np.ndarray, point: float, jackknife: np.ndarray, confidence: float = 0.95) -> tuple[float, float]:
    eps = 1.0 / (2.0 * max(len(bootstrap), 1))
    proportion = float(np.clip(np.mean(bootstrap < point), eps, 1.0 - eps))
    z0 = float(norm.ppf(proportion))
    jack_mean = float(np.mean(jackknife))
    centered = jack_mean - jackknife
    denominator = 6.0 * float(np.sum(centered**2) ** 1.5)
    acceleration = float(np.sum(centered**3) / denominator) if denominator > 0 else 0.0
    alpha = (1.0 - confidence) / 2.0
    adjusted = []
    for probability in (alpha, 1.0 - alpha):
        z = float(norm.ppf(probability))
        denom = 1.0 - acceleration * (z0 + z)
        adjusted.append(float(norm.cdf(z0 + (z0 + z) / denom)))
    low_q, high_q = np.clip(adjusted, 0.0, 1.0)
    return float(np.quantile(bootstrap, low_q)), float(np.quantile(bootstrap, high_q))


def hierarchical_bootstrap(
    pairs: dict[tuple[str, int], pd.DataFrame],
    metric: str,
    *,
    reps: int,
    seed: int,
) -> dict[str, float]:
    subsets = list(FORMAL_BENCHMARK_SUBSETS)
    seeds = list(FORMAL_BENCHMARK_SEEDS)
    arrays = {
        key: (
            frame["true_rul"].to_numpy(float),
            frame["pred_rul_baseline"].to_numpy(float) - frame["true_rul"].to_numpy(float),
            frame["pred_rul_proposed"].to_numpy(float) - frame["true_rul"].to_numpy(float),
        )
        for key, frame in pairs.items()
    }
    cluster_values = np.asarray(
        [paired_metric_arrays(*arrays[(subset, train_seed)], metric) for subset in subsets for train_seed in seeds]
    )
    point = float(np.mean(cluster_values))
    jackknife = np.asarray(
        [float(np.mean(np.delete(cluster_values, i))) for i in range(len(cluster_values))], dtype=float
    )
    rng = np.random.default_rng(seed)
    bootstrap = np.empty(reps, dtype=float)
    for rep in range(reps):
        sampled_subset_positions = rng.integers(0, len(subsets), size=len(subsets))
        subset_values = []
        for subset_position in sampled_subset_positions:
            subset = subsets[int(subset_position)]
            seed_values = []
            sampled_seed_positions = rng.integers(0, len(seeds), size=len(seeds))
            for seed_position in sampled_seed_positions:
                train_seed = seeds[int(seed_position)]
                cell = arrays[(subset, train_seed)]
                indices = rng.integers(0, len(cell[0]), size=len(cell[0]))
                seed_values.append(paired_metric_arrays(*cell, metric, indices))
            subset_values.append(float(np.mean(seed_values)))
        bootstrap[rep] = float(np.mean(subset_values))
    ci_low, ci_high = bca_interval(bootstrap, point, jackknife)
    threshold = PRACTICAL_THRESHOLDS[metric]
    p_two_sided = float(min(1.0, 2.0 * min(np.mean(bootstrap <= 0.0), np.mean(bootstrap >= 0.0))))
    return {
        "mean_paired_difference_baseline_minus_proposed": point,
        "bca_ci95_low": ci_low,
        "bca_ci95_high": ci_high,
        "standardized_cluster_effect": float(point / np.std(cluster_values, ddof=1)) if np.std(cluster_values, ddof=1) > 0 else 0.0,
        "probability_proposed_better": float(np.mean(bootstrap > 0.0)),
        "practical_threshold": threshold,
        "probability_exceeds_practical_threshold": float(np.mean(bootstrap > threshold)),
        "bootstrap_two_sided_p": p_two_sided,
        "bootstrap_reps": reps,
        "cluster_count": int(len(cluster_values)),
    }


def holm_adjust(values: pd.Series) -> pd.Series:
    p = pd.to_numeric(values, errors="coerce").to_numpy(float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.0
    m = len(p)
    for rank, index in enumerate(order):
        running = max(running, (m - rank) * p[index])
        adjusted[index] = min(1.0, running)
    return pd.Series(adjusted, index=values.index)


def build_hierarchical_bootstrap(
    grid: dict[tuple[int, str, str], pd.DataFrame],
    *,
    reps: int,
) -> pd.DataFrame:
    rows = []
    proposed = "rast_gru_v2"
    for baseline_index, baseline in enumerate(model for model in FORMAL_BENCHMARK_MODELS if model != proposed):
        pairs: dict[tuple[str, int], pd.DataFrame] = {}
        for subset in FORMAL_BENCHMARK_SUBSETS:
            for train_seed in FORMAL_BENCHMARK_SEEDS:
                baseline_frame = grid[(train_seed, subset, baseline)].rename(columns={"pred_rul": "pred_rul_baseline"})
                proposed_frame = grid[(train_seed, subset, proposed)].rename(columns={"pred_rul": "pred_rul_proposed"})
                merged = baseline_frame[["unit_id", "true_rul", "pred_rul_baseline"]].merge(
                    proposed_frame[["unit_id", "true_rul", "pred_rul_proposed"]],
                    on=["unit_id", "true_rul"],
                    how="inner",
                    validate="one_to_one",
                )
                pairs[(subset, train_seed)] = merged
        for metric_index, metric in enumerate(["rmse", "mae", "nasa_per_engine", "lpr30"]):
            result = hierarchical_bootstrap(
                pairs,
                metric,
                reps=reps,
                seed=20260710 + baseline_index * 100 + metric_index,
            )
            rows.append(
                {
                    "baseline_model": baseline,
                    "baseline_display": MODEL_DISPLAY[baseline],
                    "proposed_model": proposed,
                    "metric": metric,
                    **result,
                }
            )
    frame = pd.DataFrame(rows)
    frame["holm_p"] = frame.groupby("metric", group_keys=False)["bootstrap_two_sided_p"].apply(holm_adjust)
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Build aggregation, rank, Pareto, fairness, and hierarchical-bootstrap evidence.")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "paper_outputs" / "advanced_evidence"))
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    expected = len(FORMAL_BENCHMARK_MODELS) * len(FORMAL_BENCHMARK_SEEDS) * len(FORMAL_BENCHMARK_SUBSETS)
    print(f"[ADVANCED EVIDENCE] loading {expected} formal prediction artifacts", flush=True)
    grid = load_prediction_grid(results_dir)
    interval_grid = load_interval_grid(results_dir)
    per_subset, aggregation = build_aggregation(grid)
    late_tail = build_late_tail_summary(per_subset)
    rank_long, rank_summary = build_rank_tables(per_subset)
    threshold_rank = build_threshold_rank_stability(PROJECT_ROOT / "paper_outputs" / "tables" / "table_threshold_sensitivity.csv")
    threshold_paired = pd.read_csv(
        PROJECT_ROOT / "paper_outputs" / "tables" / "table_threshold_paired_bootstrap.csv"
    )
    ablation_summary = pd.read_csv(
        PROJECT_ROOT / "paper_outputs" / "tables" / "table_ablation_fd004_multiseed.csv"
    )
    pareto, weight_sensitivity = build_pareto_analysis(aggregation)
    pareto_seed_long, pareto_seed_summary = build_seed_pareto_stability(per_subset)
    interval_long, interval_summary = build_interval_calibration(interval_grid)
    fairness = build_fairness_table(results_dir)
    decision_cost_tasks, decision_cost_summary = build_decision_cost_sensitivity(grid)

    per_subset.to_csv(output_dir / "subset_seed_metrics.csv", index=False)
    aggregation.to_csv(output_dir / "aggregation_summary.csv", index=False)
    late_tail.to_csv(output_dir / "late_tail_risk_summary.csv", index=False)
    rank_long.to_csv(output_dir / "benchmark_ranks_long.csv", index=False)
    rank_summary.to_csv(output_dir / "benchmark_rank_summary.csv", index=False)
    threshold_rank.to_csv(output_dir / "threshold_rank_stability_fd004.csv", index=False)
    threshold_paired.to_csv(output_dir / "threshold_paired_bootstrap.csv", index=False)
    ablation_summary.to_csv(output_dir / "ablation_fd004_multiseed_summary.csv", index=False)
    pareto.to_csv(output_dir / "pareto_analysis.csv", index=False)
    weight_sensitivity.to_csv(output_dir / "pareto_weight_sensitivity.csv", index=False)
    pareto_seed_long.to_csv(output_dir / "pareto_seed_stability_long.csv", index=False)
    pareto_seed_summary.to_csv(output_dir / "pareto_seed_stability_summary.csv", index=False)
    interval_long.to_csv(output_dir / "interval_calibration_seed_subset.csv", index=False)
    interval_summary.to_csv(output_dir / "interval_calibration_summary.csv", index=False)
    fairness.to_csv(output_dir / "benchmark_fairness.csv", index=False)
    decision_cost_tasks.to_csv(output_dir / "decision_cost_tasks.csv", index=False)
    decision_cost_summary.to_csv(output_dir / "decision_cost_sensitivity.csv", index=False)
    plot_critical_difference(rank_summary, figure_dir / "fig_critical_difference")
    plot_threshold_heatmap(threshold_rank, figure_dir / "fig_fd004_threshold_rank_heatmap")
    plot_method_architecture(figure_dir / "fig_method_architecture")
    plot_decision_cost_sensitivity(decision_cost_summary, figure_dir / "fig_decision_cost_sensitivity")
    plot_interval_calibration(interval_summary, figure_dir / "fig_interval_calibration")

    print(f"[ADVANCED EVIDENCE] hierarchical bootstrap reps={args.bootstrap_reps}", flush=True)
    bootstrap = build_hierarchical_bootstrap(grid, reps=args.bootstrap_reps)
    bootstrap.to_csv(output_dir / "hierarchical_bootstrap_bca.csv", index=False)
    manifest = {
        "protocol_name": "locked_rul_protocol_v3",
        "protocol_version": "3.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "formal_run_count": expected,
        "training_seeds": FORMAL_BENCHMARK_SEEDS,
        "subsets": FORMAL_BENCHMARK_SUBSETS,
        "models": FORMAL_BENCHMARK_MODELS,
        "interval_models": list(INTERVAL_MODELS),
        "bootstrap_repetitions": int(args.bootstrap_reps),
        "pareto_hypervolume_reference": "(1.1,1.1,1.1) after within-seed min-max normalization",
        "source_results_directory": str(results_dir.resolve()),
        "ablation_fd004_multiseed_rows": int(
            len(pd.read_csv(results_dir / "ablation_fd004_multiseed_runs.csv"))
        ),
        "supplementary_artifact_rows": {
            filename: int(len(pd.read_csv(output_dir / filename)))
            for filename in [
                "ncmapss_ds02_runs.csv",
                "stress_multiseed_seed_level.csv",
                "stress_degradation_auc_seed_level.csv",
                "reliability_fault_localization_seed_level.csv",
                "attention_mechanism_seed_level.csv",
                "design_sensitivity_seed_level.csv",
                "close_prior_fd002_runs.csv",
                "conformal_interval_seed_subset.csv",
                "failure_case_engine_level.csv",
                "condition_normalization_grouped_cv.csv",
            ]
        },
    }
    (output_dir / "advanced_evidence_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    (output_dir / "figure_data_manifest.json").write_text(
        json.dumps(build_figure_data_manifest(output_dir), indent=2), encoding="utf-8"
    )
    print(f"ADVANCED_EVIDENCE_READY {output_dir}")


if __name__ == "__main__":
    main()
