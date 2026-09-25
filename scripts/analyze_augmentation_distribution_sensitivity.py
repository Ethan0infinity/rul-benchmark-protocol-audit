from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_stress_pair_extension import SCENARIOS, add_bias_diagnostics, normalized_curve_mean
from rul.config import load_config
from rul.experiments import evaluate_robustness
from run_augmentation_distribution_sensitivity import CONDITIONS, POINTS
from run_multiseed_stress_tests import PERTURBATION_SEEDS


CLEAN_METRICS = {
    "rmse": "test_rmse",
    "lpr30": "test_critical_30_late_prediction_ratio",
}
STRESS_METRICS = (
    "rmse",
    "critical_30_late_prediction_ratio",
    "critical_30_signed_error_mean",
    "critical_30_decision_cost_2",
    "critical_30_decision_cost_5",
    "critical_30_decision_cost_10",
)


def load_clean_rows(manifest: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    for item in manifest.itertuples(index=False):
        run_dir = Path(item.run_dir)
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
        with (run_dir / "metrics.json").open("r", encoding="utf-8-sig") as handle:
            metrics = json.load(handle)
        predictions = pd.read_csv(run_dir / "test_predictions.csv")
        error = predictions["pred_rul"].to_numpy(float) - predictions["true_rul"].to_numpy(float)
        critical = predictions["true_rul"].to_numpy(float) <= 30.0
        rows.append(
            {
                "point": item.point,
                "condition": item.condition,
                "training_seed": int(item.training_seed),
                "clean_rmse": float(metrics[CLEAN_METRICS["rmse"]]),
                "clean_lpr30": float(metrics[CLEAN_METRICS["lpr30"]]),
                "clean_signed_bias": float(np.mean(error)),
                "clean_critical30_signed_bias": float(np.mean(error[critical])),
                "run_dir": str(run_dir),
            }
        )
    return pd.DataFrame(rows)


def collect_stress_rows(manifest: pd.DataFrame, raw_path: Path) -> pd.DataFrame:
    existing = pd.read_csv(raw_path) if raw_path.exists() else pd.DataFrame()
    done: set[tuple[str, str, int, int]] = set()
    if not existing.empty:
        done = set(
            zip(
                existing["point"], existing["condition"],
                existing["training_seed"].astype(int), existing["perturbation_seed"].astype(int),
            )
        )
    frames = [existing] if not existing.empty else []
    total = len(manifest) * len(PERTURBATION_SEEDS)
    index = 0
    robustness = load_config(PROJECT_ROOT / "configs" / "paper_main.yaml")["robustness"]
    for item in manifest.itertuples(index=False):
        run_dir = Path(item.run_dir)
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
        for perturbation_seed in PERTURBATION_SEEDS:
            index += 1
            key = (item.point, item.condition, int(item.training_seed), int(perturbation_seed))
            if key in done:
                print(f"[AUGMENTATION STRESS SKIP {index}/{total}] {key}", flush=True)
                continue
            print(f"[AUGMENTATION STRESS {index}/{total}] {key}", flush=True)
            frame = pd.DataFrame(
                evaluate_robustness(
                    run_dir,
                    perturbation_seed=int(perturbation_seed),
                    save=False,
                    robustness_override=robustness,
                )
            )
            frame = add_bias_diagnostics(frame)
            frame["point"] = item.point
            frame["condition"] = item.condition
            frame["training_seed"] = int(item.training_seed)
            frame["perturbation_seed"] = int(perturbation_seed)
            frames.append(frame)
            combined = pd.concat(frames, ignore_index=True)
            combined.to_csv(raw_path, index=False)
    return pd.concat(frames, ignore_index=True)


def stress_curve_rows(raw: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict] = []
    keys = ["point", "condition", "training_seed", "perturbation_seed"]
    for key, group in raw.groupby(keys, sort=True):
        row = dict(zip(keys, key))
        for metric in STRESS_METRICS:
            values = [normalized_curve_mean(group[(group["scenario"] == scenario) | (group["scenario"] == "clean")], metric) for scenario in SCENARIOS]
            row[f"stress_{metric}_nine_family_mean"] = float(np.mean(values))
        rows.append(row)
    return pd.DataFrame(rows)


def seed_level_rows(clean: pd.DataFrame, stress: pd.DataFrame) -> pd.DataFrame:
    stress_seed = stress.groupby(["point", "condition", "training_seed"], as_index=False).agg(
        **{
            f"stress_{metric}_nine_family_mean": (f"stress_{metric}_nine_family_mean", "mean")
            for metric in STRESS_METRICS
        }
    )
    stress_seed = stress_seed.rename(
        columns={"stress_critical_30_late_prediction_ratio_nine_family_mean": "stress_lpr30_nine_family_mean"}
    )
    return clean.merge(stress_seed, on=["point", "condition", "training_seed"], validate="one_to_one")


def summarize(seed_rows: pd.DataFrame) -> pd.DataFrame:
    metrics = [
        "clean_rmse", "clean_lpr30", "clean_signed_bias", "clean_critical30_signed_bias",
        "stress_rmse_nine_family_mean", "stress_lpr30_nine_family_mean",
        "stress_critical_30_signed_error_mean_nine_family_mean",
        "stress_critical_30_decision_cost_2_nine_family_mean",
        "stress_critical_30_decision_cost_5_nine_family_mean",
        "stress_critical_30_decision_cost_10_nine_family_mean",
    ]
    rows: list[dict] = []
    for (point, condition), group in seed_rows.groupby(["point", "condition"], sort=False):
        row = {"point": point, "condition": condition, "seed_count": int(len(group))}
        for metric in metrics:
            row[f"{metric}_mean"] = float(group[metric].mean())
            row[f"{metric}_sd"] = float(group[metric].std(ddof=1))
        rows.append(row)
    return pd.DataFrame(rows)


def paired_bootstrap(seed_rows: pd.DataFrame, repetitions: int = 10000) -> pd.DataFrame:
    metrics = [
        "clean_rmse", "clean_lpr30", "clean_signed_bias", "clean_critical30_signed_bias",
        "stress_rmse_nine_family_mean", "stress_lpr30_nine_family_mean",
        "stress_critical_30_signed_error_mean_nine_family_mean",
        "stress_critical_30_decision_cost_2_nine_family_mean",
        "stress_critical_30_decision_cost_5_nine_family_mean",
        "stress_critical_30_decision_cost_10_nine_family_mean",
    ]
    rows: list[dict] = []
    for point_index, point in enumerate(POINTS):
        point_rows = seed_rows[seed_rows["point"] == point]
        reference = point_rows[point_rows["condition"] == "p1"].set_index("training_seed")
        for condition_index, condition in enumerate(CONDITIONS):
            candidate = point_rows[point_rows["condition"] == condition].set_index("training_seed")
            shared = sorted(set(reference.index) & set(candidate.index))
            if len(shared) != 5:
                raise ValueError(f"Expected five paired seeds for {point}/{condition}; found {len(shared)}")
            rng = np.random.default_rng(20260720 + point_index * 101 + condition_index)
            draw = rng.integers(0, len(shared), size=(repetitions, len(shared)))
            for metric in metrics:
                difference = candidate.loc[shared, metric].to_numpy(float) - reference.loc[shared, metric].to_numpy(float)
                samples = difference[draw].mean(axis=1)
                low, high = np.quantile(samples, [0.025, 0.975])
                rows.append(
                    {
                        "point": point,
                        "condition": condition,
                        "reference": "p1",
                        "metric": metric,
                        "mean_difference_condition_minus_p1": float(np.mean(difference)),
                        "paired_seed_bootstrap_ci95_low": float(low),
                        "paired_seed_bootstrap_ci95_high": float(high),
                        "probability_condition_lower": float(np.mean(samples < 0.0)),
                        "seed_count": len(shared),
                        "bootstrap_repetitions": repetitions,
                    }
                )
    return pd.DataFrame(rows)


def no_overlap_audit(intervals: pd.DataFrame) -> pd.DataFrame:
    """Expose the p=0 condition as a no-training-perturbation overlap audit."""

    frame = intervals[intervals["condition"] == "p0"].copy()
    frame["training_condition"] = "no normalized-coordinate augmentation"
    frame["test_condition"] = "same declared nine-family perturbation battery"
    frame["design_role"] = (
        "no-overlap augmentation diagnostic; all perturbation families are held out "
        "jointly during training"
    )
    frame["claim_limit"] = (
        "does not isolate individual perturbation families or physical fault robustness"
    )
    return frame


def plot_summary(summary: pd.DataFrame, output: Path) -> None:
    order = list(CONDITIONS)
    labels = {"p0": "p=0", "p025": "p=0.25", "p05": "p=0.5", "p1": "p=1", "mix50": "50/50 mix"}
    metrics = [
        ("clean_rmse", "Clean RMSE"),
        ("clean_lpr30", "Clean LPR@30"),
        ("clean_critical30_signed_bias", "Critical signed bias (cycles)"),
        ("stress_lpr30_nine_family_mean", "Nine-family LPR curve mean"),
    ]
    styles = {"RAST": ("#2f6f9f", "o"), "Core": ("#2a9d8f", "^"), "Asym": ("#c44e52", "s")}
    fig, axes = plt.subplots(2, 2, figsize=(9.0, 6.8))
    x = np.arange(len(order))
    for ax, (metric, ylabel) in zip(axes.flat, metrics):
        for point in POINTS:
            group = summary[summary["point"] == point].set_index("condition").loc[order]
            color, marker = styles[point]
            ax.errorbar(
                x, group[f"{metric}_mean"], yerr=group[f"{metric}_sd"],
                label=point, color=color, marker=marker, linewidth=1.8, capsize=3,
            )
        ax.set_xticks(x, [labels[item] for item in order], rotation=18)
        ax.set_ylabel(ylabel)
        ax.grid(linestyle=":", alpha=0.35)
    axes[0, 0].legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze augmentation-distribution sensitivity results.")
    parser.add_argument(
        "--manifest",
        default=str(PROJECT_ROOT / "results" / "augmentation_distribution_sensitivity" / "run_manifest.csv"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "paper_outputs" / "advanced_evidence"),
    )
    parser.add_argument("--reuse-stress", action="store_true")
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    (output / "figures").mkdir(parents=True, exist_ok=True)
    manifest = pd.read_csv(args.manifest)
    if len(manifest) != len(POINTS) * len(CONDITIONS) * 5:
        raise ValueError(f"Expected 75 manifest rows, found {len(manifest)}")
    clean = load_clean_rows(manifest)
    clean.to_csv(output / "augmentation_distribution_clean_seed_level.csv", index=False)
    raw_path = output / "augmentation_distribution_stress_raw.csv"
    raw = pd.read_csv(raw_path) if args.reuse_stress else collect_stress_rows(manifest, raw_path)
    stress = stress_curve_rows(raw)
    stress.to_csv(output / "augmentation_distribution_stress_curve_seed_level.csv", index=False)
    seed_rows = seed_level_rows(clean, stress)
    seed_rows.to_csv(output / "augmentation_distribution_seed_level.csv", index=False)
    summary = summarize(seed_rows)
    summary.to_csv(output / "augmentation_distribution_summary.csv", index=False)
    intervals = paired_bootstrap(seed_rows)
    intervals.to_csv(output / "augmentation_distribution_paired_intervals.csv", index=False)
    no_overlap_audit(intervals).to_csv(
        output / "no_overlap_augmentation_audit.csv",
        index=False,
    )
    plot_summary(summary, output / "figures" / "fig_augmentation_distribution_sensitivity")
    print(
        f"AUGMENTATION_DISTRIBUTION_ANALYSIS_READY cells={len(seed_rows)} "
        f"summary_rows={len(summary)} intervals={len(intervals)}"
    )


if __name__ == "__main__":
    main()
