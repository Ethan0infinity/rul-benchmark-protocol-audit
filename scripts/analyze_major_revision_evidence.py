from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

SEEDS = (42, 123, 2024, 2025, 2026)
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
VARIANTS = ("core", "asym")
COMPARATORS = ("rast_gru", "transformer_lite", "official_dual_mixer")
COMPARATOR_DISPLAY = {
    "rast_gru": "RAST-GRU",
    "transformer_lite": "Transformer-lite",
    "official_dual_mixer": "Dual-Mixer",
}


def prediction_path(results: Path, seed: int, subset: str, model: str) -> Path:
    if model == "core":
        if subset == "FD004":
            return results / f"ablation_fd004_seed{seed}_weighted_huber_no_asymmetry" / subset / "rast_gru_v2" / "test_predictions.csv"
        return results / f"core_ocm_seed{seed}" / subset / "rast_gru_v2" / "test_predictions.csv"
    if model == "asym":
        return results / f"paper_main_v3_seed{seed}" / subset / "rast_gru_v2" / "test_predictions.csv"
    if model == "official_dual_mixer":
        return results / f"official_dual_mixer_seed{seed}" / subset / model / "test_predictions.csv"
    return results / f"paper_main_v3_seed{seed}" / subset / model / "test_predictions.csv"


def metric(true: np.ndarray, pred: np.ndarray, name: str) -> float:
    error = pred - true
    if name == "rmse":
        return float(np.sqrt(np.mean(error**2)))
    if name == "nasa_per_engine":
        return float(np.mean(np.where(error < 0, np.exp(-error / 13.0) - 1.0, np.exp(error / 10.0) - 1.0)))
    critical = true <= 30.0
    zone_error = error[critical]
    if name == "lpr30":
        return float(np.mean(zone_error > 0.0))
    if name.startswith("cost_"):
        rho = float(name.split("_")[1])
        return float(np.mean(np.maximum(-error, 0.0) + rho * np.maximum(error, 0.0)))
    raise ValueError(name)


def nested_paired_bootstrap(
    pairs: dict[tuple[int, str], pd.DataFrame], metric_name: str, reps: int, rng: np.random.Generator
) -> tuple[float, float, float, float, float]:
    observed_by_seed = []
    for seed in SEEDS:
        effects = []
        for subset in SUBSETS:
            frame = pairs[(seed, subset)]
            true = frame.true_rul.to_numpy(float)
            effects.append(
                metric(true, frame.pred_rul_left.to_numpy(float), metric_name)
                - metric(true, frame.pred_rul_right.to_numpy(float), metric_name)
            )
        observed_by_seed.append(float(np.mean(effects)))
    samples = np.empty(reps, dtype=float)
    for rep in range(reps):
        seed_effects = []
        for sampled_seed in rng.choice(SEEDS, len(SEEDS), replace=True):
            subset_effects = []
            for subset in SUBSETS:
                frame = pairs[(int(sampled_seed), subset)]
                indices = rng.integers(0, len(frame), len(frame))
                part = frame.iloc[indices]
                true = part.true_rul.to_numpy(float)
                subset_effects.append(
                    metric(true, part.pred_rul_left.to_numpy(float), metric_name)
                    - metric(true, part.pred_rul_right.to_numpy(float), metric_name)
                )
            seed_effects.append(float(np.mean(subset_effects)))
        samples[rep] = float(np.mean(seed_effects))
    p_two_sided = min(1.0, 2.0 * min(float(np.mean(samples <= 0.0)), float(np.mean(samples >= 0.0))))
    return (
        float(np.mean(observed_by_seed)),
        float(np.quantile(samples, 0.025)),
        float(np.quantile(samples, 0.975)),
        float(np.mean(samples < 0.0)),
        p_two_sided,
    )


def holm_adjust(frame: pd.DataFrame, family: list[str]) -> pd.Series:
    order = np.argsort(frame["bootstrap_two_sided_p"].to_numpy(float))
    raw = frame["bootstrap_two_sided_p"].to_numpy(float)[order]
    adjusted_sorted = np.maximum.accumulate(np.minimum(1.0, raw * (len(raw) - np.arange(len(raw)))))
    adjusted = np.empty_like(adjusted_sorted)
    adjusted[order] = adjusted_sorted
    return pd.Series(adjusted, index=frame.index, name="holm_adjusted_p")


def subset_core_rast_bootstrap(results: Path, reps: int) -> pd.DataFrame:
    """Estimate the locked Core/Asym minus RAST contrast inside each subset."""
    rows: list[dict] = []
    for variant_index, variant in enumerate(VARIANTS):
        for subset_index, subset in enumerate(SUBSETS):
            pairs: dict[int, dict[str, np.ndarray]] = {}
            for seed in SEEDS:
                left = pd.read_csv(prediction_path(results, seed, subset, variant))
                right = pd.read_csv(prediction_path(results, seed, subset, "rast_gru"))
                merged = left.merge(
                    right,
                    on=["unit_id", "true_rul"],
                    suffixes=("_left", "_right"),
                    validate="one_to_one",
                ).sort_values("unit_id").reset_index(drop=True)
                pairs[seed] = {
                    "unit_id": merged["unit_id"].to_numpy(int),
                    "true": merged["true_rul"].to_numpy(float),
                    "left": merged["pred_rul_left"].to_numpy(float),
                    "right": merged["pred_rul_right"].to_numpy(float),
                }
            for metric_index, metric_name in enumerate(("rmse", "nasa_per_engine", "lpr30")):
                observed = []
                for arrays in pairs.values():
                    true = arrays["true"]
                    observed.append(
                        metric(true, arrays["left"], metric_name)
                        - metric(true, arrays["right"], metric_name)
                    )
                rng = np.random.default_rng(
                    20260722 + variant_index * 1000 + subset_index * 100 + metric_index
                )
                samples = np.empty(reps, dtype=float)
                for rep in range(reps):
                    seed_effects = []
                    for sampled_seed in rng.choice(SEEDS, len(SEEDS), replace=True):
                        arrays = pairs[int(sampled_seed)]
                        unit_ids = arrays["unit_id"]
                        sampled_units = rng.choice(unit_ids, len(unit_ids), replace=True)
                        sampled_indices = np.searchsorted(unit_ids, sampled_units)
                        true = arrays["true"][sampled_indices]
                        seed_effects.append(
                            metric(true, arrays["left"][sampled_indices], metric_name)
                            - metric(true, arrays["right"][sampled_indices], metric_name)
                        )
                    samples[rep] = float(np.mean(seed_effects))
                low, high = np.quantile(samples, [0.025, 0.975])
                classification = "unresolved"
                if high < 0.0:
                    classification = f"CI supports lower {variant}"
                elif low > 0.0:
                    classification = "CI supports lower RAST"
                rows.append(
                    {
                        "ocm_variant": variant,
                        "comparator": "rast_gru",
                        "subset": subset,
                        "metric": metric_name,
                        "mean_difference_ocm_minus_rast": float(np.mean(observed)),
                        "percentile_ci95_low": float(low),
                        "percentile_ci95_high": float(high),
                        "probability_ocm_lower": float(np.mean(samples < 0.0)),
                        "ci_classification": classification,
                        "seed_count": len(SEEDS),
                        "test_engine_count": int(len(next(iter(pairs.values()))["unit_id"])),
                        "bootstrap_repetitions": reps,
                        "inference_unit": "paired test engine nested within resampled composite seed",
                        "analysis_status": "post-lock confirmation",
                    }
                )
    return pd.DataFrame(rows)


def build_comparisons(results: Path, reps: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = []
    metrics = ("rmse", "nasa_per_engine", "lpr30")
    for variant_index, variant in enumerate(VARIANTS):
        for comparator_index, comparator in enumerate(COMPARATORS):
            pairs = {}
            for seed in SEEDS:
                for subset in SUBSETS:
                    left = pd.read_csv(prediction_path(results, seed, subset, variant))
                    right = pd.read_csv(prediction_path(results, seed, subset, comparator))
                    pairs[(seed, subset)] = left.merge(
                        right,
                        on=["unit_id", "true_rul"],
                        suffixes=("_left", "_right"),
                        validate="one_to_one",
                    )
            for metric_index, metric_name in enumerate(metrics):
                mean, low, high, probability, p_value = nested_paired_bootstrap(
                    pairs,
                    metric_name,
                    reps,
                    np.random.default_rng(20260717 + variant_index * 1000 + comparator_index * 100 + metric_index),
                )
                rows.append(
                    {
                        "ocm_variant": variant,
                        "comparator": comparator,
                        "metric": metric_name,
                        "mean_difference_ocm_minus_comparator": mean,
                        "percentile_ci95_low": low,
                        "percentile_ci95_high": high,
                        "probability_ocm_lower": probability,
                        "bootstrap_two_sided_p": p_value,
                        "bootstrap_repetitions": reps,
                        "inference_unit": "test engine nested within fixed subset and resampled composite seed",
                    }
                )
    comparisons = pd.DataFrame(rows)
    comparisons["holm_adjusted_p"] = np.nan
    for variant, indices in comparisons.groupby("ocm_variant").groups.items():
        comparisons.loc[indices, "holm_adjusted_p"] = holm_adjust(comparisons.loc[indices], list(indices))
    comparisons["holm_family"] = "nine comparator-metric contrasts within each OCM operating point"

    cost_rows = []
    for variant_index, variant in enumerate(VARIANTS):
        for comparator_index, comparator in enumerate(COMPARATORS):
            pairs = {}
            for seed in SEEDS:
                for subset in SUBSETS:
                    left = pd.read_csv(prediction_path(results, seed, subset, variant))
                    right = pd.read_csv(prediction_path(results, seed, subset, comparator))
                    pairs[(seed, subset)] = left.merge(
                        right,
                        on=["unit_id", "true_rul"],
                        suffixes=("_left", "_right"),
                        validate="one_to_one",
                    )
            for ratio_index, ratio in enumerate((1, 2, 5, 10)):
                mean, low, high, probability, _ = nested_paired_bootstrap(
                    pairs,
                    f"cost_{ratio}",
                    reps,
                    np.random.default_rng(20261717 + variant_index * 1000 + comparator_index * 100 + ratio_index),
                )
                cost_rows.append(
                    {
                        "ocm_variant": variant,
                        "comparator": comparator,
                        "late_to_early_cost_ratio": ratio,
                        "mean_difference_ocm_minus_comparator": mean,
                        "percentile_ci95_low": low,
                        "percentile_ci95_high": high,
                        "probability_ocm_lower_cost": probability,
                        "bootstrap_repetitions": reps,
                        "cost_definition": "mean(max(-error,0) + rho*max(error,0)) over test-engine endpoints",
                    }
                )
    return comparisons, pd.DataFrame(cost_rows)


def endpoint_diagnostics(source: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    frame = pd.read_csv(source / "endpoint_selection_confirmation_seed_metrics.csv")
    rows = []
    for (variant, subset), group in frame.groupby(["variant", "subset"]):
        pivot = group.pivot(index="seed", columns="endpoint_strategy", values="best_epoch")
        exact_all = (pivot.nunique(axis=1) == 1)
        rows.append(
            {
                "variant": variant,
                "subset": subset,
                "seed_count": int(len(pivot)),
                "uniform_epoch_mean": float(pivot["uniform_single"].mean()),
                "fixed_multi_epoch_mean": float(pivot["fixed_multi"].mean()),
                "critical_stratified_epoch_mean": float(pivot["critical_stratified"].mean()),
                "same_epoch_all_three_count": int(exact_all.sum()),
                "same_epoch_all_three_fraction": float(exact_all.mean()),
                "max_pairwise_epoch_gap_mean": float((pivot.max(axis=1) - pivot.min(axis=1)).mean()),
            }
        )
    agreement = pd.DataFrame(rows)
    task_summary = frame.groupby(["variant", "subset", "endpoint_strategy"], as_index=False).agg(
        seed_count=("seed", "nunique"),
        best_epoch_mean=("best_epoch", "mean"),
        best_epoch_std=("best_epoch", lambda x: x.std(ddof=1)),
        rmse_mean=("test_rmse", "mean"),
        rmse_std=("test_rmse", lambda x: x.std(ddof=1)),
        lpr30_mean=("test_critical_30_late_prediction_ratio", "mean"),
        lpr30_std=("test_critical_30_late_prediction_ratio", lambda x: x.std(ddof=1)),
        slpr30_10_mean=("test_critical_30_severe_late_10_ratio", "mean"),
        slpr30_10_std=("test_critical_30_severe_late_10_ratio", lambda x: x.std(ddof=1)),
    )
    task_summary["correlation_handling"] = "one endpoint per test engine; bootstrap resamples engines within composite seeds"
    return task_summary, agreement


def plot_decision_cost_pairs(costs: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    ratios = np.array([1, 2, 5, 10], dtype=float)
    colors = {"core": "#2F6B9A", "asym": "#B44C43"}
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 4.7), sharey=False)
    for ax, comparator in zip(axes, COMPARATORS, strict=True):
        part = costs[costs["comparator"] == comparator]
        for variant in VARIANTS:
            line = part[part["ocm_variant"] == variant].sort_values("late_to_early_cost_ratio")
            mean = line["mean_difference_ocm_minus_comparator"].to_numpy(float)
            low = line["percentile_ci95_low"].to_numpy(float)
            high = line["percentile_ci95_high"].to_numpy(float)
            ax.errorbar(
                ratios,
                mean,
                yerr=np.vstack([mean - low, high - mean]),
                marker="o",
                markersize=5.5,
                capsize=3.5,
                linewidth=2.0,
                color=colors[variant],
                label=f"OCM-{variant.title()}",
            )
        ax.axhline(0.0, color="#444444", linewidth=1.1, linestyle="--")
        ax.set_xticks(ratios, [str(int(value)) for value in ratios])
        ax.set_xlabel(r"Late-to-early error weight $\kappa$")
        ax.set_title(f"Relative to {COMPARATOR_DISPLAY[comparator]}", loc="left", fontsize=11)
        ax.grid(axis="y", alpha=0.2)
        ax.set_ylabel(r"Paired $A_\kappa$ difference")
    axes[0].legend(frameon=False, fontsize=9)
    fig.suptitle("Asymmetric-error-cost sensitivity with paired 95% bootstrap intervals", fontsize=12)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build evidence requested by the controlled accuracy-risk major revision.")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "paper_outputs" / "advanced_evidence"))
    parser.add_argument(
        "--figure-dir",
        default=str(PROJECT_ROOT / "paper_outputs" / "advanced_evidence" / "figures"),
    )
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument(
        "--reuse-aggregate",
        action="store_true",
        help="Reuse existing aggregate comparator and decision-cost tables; rebuild subset Core/Asym contrasts.",
    )
    args = parser.parse_args()
    results = Path(args.results_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    if args.reuse_aggregate:
        comparisons = pd.read_csv(output / "core_asym_major_comparator_bootstrap.csv")
        costs = pd.read_csv(output / "core_asym_decision_cost_bootstrap.csv")
    else:
        comparisons, costs = build_comparisons(results, args.bootstrap_reps)
        comparisons.to_csv(output / "core_asym_major_comparator_bootstrap.csv", index=False)
        costs.to_csv(output / "core_asym_decision_cost_bootstrap.csv", index=False)
    subset_comparisons = subset_core_rast_bootstrap(results, args.bootstrap_reps)
    subset_comparisons.to_csv(output / "core_asym_rast_subset_bootstrap.csv", index=False)
    plot_decision_cost_pairs(costs, Path(args.figure_dir) / "fig_core_asym_decision_cost")
    task, agreement = endpoint_diagnostics(output)
    task.to_csv(output / "endpoint_selection_task_summary.csv", index=False)
    agreement.to_csv(output / "endpoint_selection_epoch_agreement.csv", index=False)
    print(
        "MAJOR_REVISION_EVIDENCE_READY "
        f"comparisons={len(comparisons)} subset_comparisons={len(subset_comparisons)} "
        f"costs={len(costs)} endpoint_tasks={len(task)}"
    )


if __name__ == "__main__":
    main()
