from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_advanced_evidence import prediction_metrics
from rul.config import load_config
from run_cross_backbone_protocol_buildup import MODELS, STAGES
from run_formal_benchmark import FORMAL_BENCHMARK_SEEDS


DISPLAY = {
    "global_scaling": "Global scaling",
    "condition_normalization": "+ condition norm.",
    "plus_mask_channels": "+ mask channels",
    "plus_degradation_augmentation": "+ degradation aug.",
    "plus_asymmetric_base_loss": "+ asymmetric base loss",
    "plus_risk_checkpoint": "+ risk checkpoint",
}
METRICS = ("rmse", "nasa_per_engine", "lpr30", "late_cvar95_30")


def run_dir(root: Path, model: str, seed: int, stage: str) -> Path:
    return root / f"protocol_buildup_{stage}_seed{seed}" / "FD004" / model


def validate_run_provenance(path: Path, model: str, seed: int, stage: str) -> None:
    config_path = path / "run_config.yaml"
    prediction_path = path / "test_predictions.csv"
    metrics_path = path / "metrics.json"
    for required in (config_path, prediction_path, metrics_path):
        if not required.exists() or required.stat().st_size == 0:
            raise FileNotFoundError(required)
    cfg = load_config(config_path)
    expected_experiment = f"protocol_buildup_{stage}_seed{seed}"
    checks = {
        "model": cfg["model"]["name"] == model,
        "seed": int(cfg["project"]["seed"]) == seed,
        "experiment": cfg["project"]["experiment_name"] == expected_experiment,
        "epoch_budget": int(cfg["training"]["epochs"]) == 80,
        "patience": int(cfg["training"]["patience"]) == 12,
        "fast_cudnn": cfg["training"].get("deterministic_cudnn") is False,
    }
    failed = [name for name, passed in checks.items() if not passed]
    if failed:
        raise ValueError(f"Invalid protocol build-up provenance at {path}: {', '.join(failed)}")


def bootstrap_transition(pairs: dict[int, pd.DataFrame], metric: str, reps: int, seed: int) -> tuple[float, float, float, float]:
    rng = np.random.default_rng(seed)
    ordered = [pairs[key] for key in sorted(pairs)]
    truth = np.stack([frame["true_rul"].to_numpy(float) for frame in ordered])
    left_error = np.stack(
        [frame["pred_rul_left"].to_numpy(float) - frame["true_rul"].to_numpy(float) for frame in ordered]
    )
    right_error = np.stack(
        [frame["pred_rul_right"].to_numpy(float) - frame["true_rul"].to_numpy(float) for frame in ordered]
    )

    def values(errors: np.ndarray, y_true: np.ndarray) -> np.ndarray:
        if metric == "rmse":
            return np.sqrt(np.mean(errors**2, axis=-1))
        if metric == "nasa_per_engine":
            contributions = np.where(errors < 0.0, np.exp(-errors / 13.0) - 1.0, np.exp(errors / 10.0) - 1.0)
            return np.mean(contributions, axis=-1)
        critical = y_true <= 30.0
        if metric == "lpr30":
            return np.sum(critical & (errors > 0.0), axis=-1) / np.maximum(np.sum(critical, axis=-1), 1)
        if metric == "late_cvar95_30":
            positive = np.where(critical & (errors > 0.0), errors, np.nan)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", category=RuntimeWarning)
                threshold = np.nanquantile(positive, 0.95, axis=-1)
            tail = positive >= threshold[..., None]
            count = np.sum(tail, axis=-1)
            total = np.nansum(np.where(tail, positive, np.nan), axis=-1)
            result = np.zeros_like(total, dtype=float)
            np.divide(total, count, out=result, where=count > 0)
            return result
        raise KeyError(metric)

    observed = values(right_error, truth) - values(left_error, truth)
    seed_draw = rng.integers(0, len(ordered), size=(reps, len(ordered)))
    engine_draw = rng.integers(0, truth.shape[1], size=(reps, len(ordered), truth.shape[1]))
    sampled_truth = truth[seed_draw[..., None], engine_draw]
    sampled_left = left_error[seed_draw[..., None], engine_draw]
    sampled_right = right_error[seed_draw[..., None], engine_draw]
    samples = np.mean(values(sampled_right, sampled_truth) - values(sampled_left, sampled_truth), axis=1)
    return (
        float(np.mean(observed)),
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
        float(np.mean(samples < 0.0)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the FD004 cross-backbone protocol build-up.")
    parser.add_argument("--results-root", default=str(PROJECT_ROOT / "results" / "cross_backbone_protocol_buildup"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "paper_outputs" / "advanced_evidence"))
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    args = parser.parse_args()
    root = Path(args.results_root)
    output = Path(args.output_dir)
    figures = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    rows = []
    for model in MODELS:
        for seed in FORMAL_BENCHMARK_SEEDS:
            for stage in STAGES:
                directory = run_dir(root, model, seed, stage)
                validate_run_provenance(directory, model, seed, stage)
                path = directory / "test_predictions.csv"
                rows.append({"model": model, "seed": seed, "stage": stage, **prediction_metrics(pd.read_csv(path))})
    seed_level = pd.DataFrame(rows)
    seed_level.to_csv(output / "cross_backbone_protocol_buildup_seed_metrics.csv", index=False)
    summary = seed_level.groupby(["model", "stage"], as_index=False).agg(
        seed_count=("seed", "nunique"),
        **{f"{metric}_mean": (metric, "mean") for metric in METRICS},
        **{f"{metric}_std": (metric, lambda x: x.std(ddof=1)) for metric in METRICS},
    )
    summary["stage_order"] = summary["stage"].map({stage: i for i, stage in enumerate(STAGES)})
    summary = summary.sort_values(["model", "stage_order"])
    summary.to_csv(output / "cross_backbone_protocol_buildup_summary.csv", index=False)

    transition_rows = []
    for model_index, model in enumerate(MODELS):
        for transition_index, (left, right) in enumerate(zip(STAGES[:-1], STAGES[1:])):
            pairs = {}
            for seed in FORMAL_BENCHMARK_SEEDS:
                left_frame = pd.read_csv(run_dir(root, model, seed, left) / "test_predictions.csv")
                right_frame = pd.read_csv(run_dir(root, model, seed, right) / "test_predictions.csv")
                pairs[seed] = left_frame.merge(
                    right_frame,
                    on=["unit_id", "true_rul"],
                    suffixes=("_left", "_right"),
                    validate="one_to_one",
                )
            for metric_index, metric in enumerate(METRICS):
                mean, low, high, probability = bootstrap_transition(
                    pairs,
                    metric,
                    args.bootstrap_reps,
                    20260717 + model_index * 1000 + transition_index * 100 + metric_index,
                )
                transition_rows.append(
                    {
                        "model": model,
                        "left_stage": left,
                        "right_stage": right,
                        "metric": metric,
                        "mean_difference_right_minus_left": mean,
                        "percentile_ci95_low": low,
                        "percentile_ci95_high": high,
                        "probability_right_lower": probability,
                        "bootstrap_repetitions": args.bootstrap_reps,
                        "inference_unit": "test engine nested within resampled composite seed",
                    }
                )
    transitions = pd.DataFrame(transition_rows)
    transitions.to_csv(output / "cross_backbone_protocol_buildup_transitions.csv", index=False)

    fig, axes = plt.subplots(2, 2, figsize=(12.8, 8.0), sharex=True)
    labels = {"rmse": "RMSE", "nasa_per_engine": "NASA / engine", "lpr30": "LPR@30", "late_cvar95_30": "Late CVaR95"}
    colors = {"rast_gru": "#35618D", "transformer_lite": "#B8573A"}
    for ax, metric in zip(axes.flat, METRICS):
        for model_index, model in enumerate(MODELS):
            part = summary[summary.model == model].sort_values("stage_order")
            offset = -0.06 if model_index == 0 else 0.06
            ax.plot(
                part.stage_order.to_numpy(float) + offset,
                part[f"{metric}_mean"],
                marker="o",
                linewidth=1.8,
                color=colors[model],
                label=model.replace("_", " "),
            )
            for stage_order, stage in enumerate(STAGES):
                values = seed_level[
                    (seed_level["model"] == model) & (seed_level["stage"] == stage)
                ][metric].to_numpy(float)
                jitter = np.linspace(-0.025, 0.025, len(values))
                ax.scatter(
                    np.full(len(values), stage_order + offset) + jitter,
                    values,
                    s=17,
                    facecolor="white",
                    edgecolor=colors[model],
                    linewidth=0.8,
                    alpha=0.9,
                    zorder=3,
                )
        ax.set_title(labels[metric], loc="left", fontsize=10)
        ax.grid(axis="y", linestyle=":", alpha=0.35)
    axes[0, 0].legend(frameon=False, fontsize=8.5)
    for ax in axes[-1, :]:
        ax.set_xticks(range(len(STAGES)), [DISPLAY[stage] for stage in STAGES], rotation=24, ha="right")
    fig.suptitle("Cumulative protocol build-up on two fixed backbones (FD004, five seed points)", fontsize=11)
    fig.tight_layout()
    fig.savefig(figures / "fig_cross_backbone_protocol_buildup.pdf", bbox_inches="tight")
    fig.savefig(figures / "fig_cross_backbone_protocol_buildup.png", dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"CROSS_BACKBONE_BUILDUP_ANALYSIS_READY rows={len(seed_level)} transitions={len(transitions)}")


if __name__ == "__main__":
    main()
