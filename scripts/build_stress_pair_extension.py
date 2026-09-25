from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap

def trapezoidal_integral(
    y: np.ndarray,
    x: np.ndarray | None = None,
    dx: float = 1.0,
    axis: int = -1,
) -> np.ndarray | np.floating:
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x=x, dx=dx, axis=axis)
    return np.trapz(y, x=x, dx=dx, axis=axis)


TRAPEZOID = trapezoidal_integral
PROJECT_ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rul.experiments import evaluate_robustness
from run_formal_benchmark import FORMAL_BENCHMARK_SEEDS
from run_multiseed_stress_tests import PERTURBATION_SEEDS, nested_seed_ci


MODELS = ("rast_gru", "rast_gru_v2")
SCENARIOS = (
    "gaussian_noise", "global_sensor_missing", "window_random_missing", "block_missing",
    "correlated_group_missing", "sensor_drift", "sensor_bias_shift", "stuck_at_fault", "burst_noise",
)
METRICS = (
    "rmse", "critical_30_late_prediction_ratio", "critical_30_mean_late_excess",
    "critical_30_late_cvar95", "critical_30_signed_error_mean",
    "critical_30_early_error_magnitude", "critical_30_late_early_event_ratio",
    "critical_30_decision_cost_1", "critical_30_decision_cost_2",
    "critical_30_decision_cost_5", "critical_30_decision_cost_10",
)
DISPLAY = {
    "rmse": "RMSE",
    "critical_30_late_prediction_ratio": "LPR@30",
    "critical_30_mean_late_excess": "MLE@30",
    "critical_30_late_cvar95": "Late CVaR95",
    "critical_30_signed_error_mean": "Signed error",
    "critical_30_early_error_magnitude": "Early-error magnitude",
    "critical_30_late_early_event_ratio": "Late/early ratio",
    "critical_30_decision_cost_1": "Decision cost (rho=1)",
    "critical_30_decision_cost_2": "Decision cost (rho=2)",
    "critical_30_decision_cost_5": "Decision cost (rho=5)",
    "critical_30_decision_cost_10": "Decision cost (rho=10)",
}
SCENARIO_DISPLAY = {
    "gaussian_noise": "Noise", "global_sensor_missing": "Global miss.",
    "window_random_missing": "Window miss.", "block_missing": "Block miss.",
    "correlated_group_missing": "Group miss.", "sensor_drift": "Drift",
    "sensor_bias_shift": "Bias", "stuck_at_fault": "Stuck-at", "burst_noise": "Burst",
}


def normalized_curve_mean(group: pd.DataFrame, metric: str) -> float:
    """Average metric value over the declared perturbation interval.

    The clean point is the zero-intensity endpoint. Trapezoidal integration is
    divided by the actual scenario-specific intensity range, so fault families
    with different physical scales remain comparable as curve averages.
    """
    clean = float(group[group.scenario == "clean"][metric].iloc[0])
    curve = group[group.scenario != "clean"].sort_values("level")
    levels = np.concatenate([[0.0], curve.level.to_numpy(float)])
    values = np.concatenate([[clean], curve[metric].to_numpy(float)])
    intensity_range = float(levels.max() - levels.min())
    if intensity_range <= 0.0:
        raise ValueError("Perturbation grid must span a positive intensity range.")
    return float(TRAPEZOID(values, levels) / intensity_range)


def add_bias_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    late = result["critical_30_late_prediction_ratio"].astype(float)
    early = result["critical_30_early_prediction_ratio"].astype(float)
    result["critical_30_signed_error_mean"] = (
        late * result["critical_30_over_error_mean"].astype(float)
        + early * result["critical_30_under_error_mean"].astype(float)
    )
    result["critical_30_early_error_magnitude"] = -result[
        "critical_30_under_error_mean"
    ].astype(float)
    result["critical_30_late_early_event_ratio"] = late / early.clip(lower=1e-6)
    return result


def family_uncertainty(auc: pd.DataFrame, reps: int = 5000) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Nested paired bootstrap over training seeds and matched perturbation seeds."""
    sensitivity_margins = {
        "rmse": (0.25, 0.50, 1.00),
        "critical_30_late_prediction_ratio": (0.01, 0.02, 0.05),
    }
    rows = []
    sensitivity_rows = []
    training_seeds = list(FORMAL_BENCHMARK_SEEDS)
    for scenario_index, scenario in enumerate(SCENARIOS):
        scenario_frame = auc[auc["scenario"] == scenario]
        for metric_index, metric in enumerate(METRICS):
            column = f"{metric}_auc"
            pivot = scenario_frame.pivot(
                index=["training_seed", "perturbation_seed"], columns="model", values=column
            )
            differences = (pivot["rast_gru_v2"] - pivot["rast_gru"]).rename("difference")
            rng = np.random.default_rng(20260716 + scenario_index * 101 + metric_index)
            matrix = differences.unstack("perturbation_seed").loc[training_seeds].to_numpy(float)
            training_draw = rng.integers(0, matrix.shape[0], size=(reps, matrix.shape[0]))
            perturbation_draw = rng.integers(
                0, matrix.shape[1], size=(reps, matrix.shape[0], matrix.shape[1])
            )
            samples = matrix[training_draw[:, :, None], perturbation_draw].mean(axis=(1, 2))
            observed = float(differences.mean())
            ci_low = float(np.quantile(samples, 0.025))
            ci_high = float(np.quantile(samples, 0.975))
            if ci_high < 0.0:
                classification = "CI supports lower OCM"
            elif ci_low > 0.0:
                classification = "CI supports lower RAST"
            else:
                classification = "unresolved"
            rows.append(
                {
                    "scenario": scenario,
                    "metric": metric,
                    "ocm_minus_rast_auc": observed,
                    "nested_bootstrap_ci95_low": ci_low,
                    "nested_bootstrap_ci95_high": ci_high,
                    "probability_ocm_lower": float(np.mean(samples < 0.0)),
                    "probability_ocm_higher": float(np.mean(samples > 0.0)),
                    "ci_classification": classification,
                    "training_seed_count": int(differences.index.get_level_values(0).nunique()),
                    "perturbation_seed_count": int(differences.index.get_level_values(1).nunique()),
                    "bootstrap_repetitions": reps,
                    "direction_note": (
                        "directional bias; lower is not intrinsically better"
                        if metric == "critical_30_signed_error_mean"
                        else "lower is better"
                    ),
                }
            )
            for margin in sensitivity_margins.get(metric, ()):
                sensitivity_rows.append(
                    {
                        "scenario": scenario,
                        "metric": metric,
                        "practical_equivalence_margin": float(margin),
                        "probability_within_margin": float(np.mean(np.abs(samples) <= margin)),
                        "nested_bootstrap_ci95_low": ci_low,
                        "nested_bootstrap_ci95_high": ci_high,
                        "bootstrap_repetitions": reps,
                        "interpretation": "threshold sensitivity only; no engineering equivalence claim",
                    }
                )
    return pd.DataFrame(rows), pd.DataFrame(sensitivity_rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paired OCM/RAST stress evidence with late-tail metrics.")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "paper_outputs" / "advanced_evidence"))
    parser.add_argument("--reuse", action="store_true", help="Reuse the extended raw CSV instead of reevaluating checkpoints.")
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    raw_path = output / "stress_pair_extended_seed_level.csv"

    if args.reuse:
        raw = pd.read_csv(raw_path)
    else:
        rows = []
        total = len(FORMAL_BENCHMARK_SEEDS) * len(MODELS) * len(PERTURBATION_SEEDS)
        index = 0
        for seed in FORMAL_BENCHMARK_SEEDS:
            for model in MODELS:
                run_dir = Path(args.results_dir) / f"paper_main_v3_seed{seed}" / "FD004" / model
                for perturb_seed in PERTURBATION_SEEDS:
                    index += 1
                    print(f"[STRESS PAIR {index}/{total}] seed={seed} model={model} perturb={perturb_seed}", flush=True)
                    rows.extend(evaluate_robustness(run_dir, perturbation_seed=perturb_seed, save=False))
        raw = pd.DataFrame(rows)
        raw = add_bias_diagnostics(raw)
        missing = [metric for metric in METRICS if metric not in raw.columns]
        if missing:
            raise ValueError(f"Extended robustness metrics missing: {missing}")
        raw.to_csv(raw_path, index=False)

    raw = add_bias_diagnostics(raw)

    auc_rows = []
    worst_rows = []
    for (seed, perturb_seed, model, scenario), group in raw[raw.scenario.isin(["clean", *SCENARIOS])].groupby(
        ["training_seed", "perturbation_seed", "model", "scenario"]
    ):
        if scenario == "clean":
            continue
        clean = raw[(raw.training_seed == seed) & (raw.perturbation_seed == perturb_seed) &
                    (raw.model == model) & (raw.scenario == "clean")]
        combined = pd.concat([clean, group], ignore_index=True)
        item = {"training_seed": seed, "perturbation_seed": perturb_seed, "model": model, "scenario": scenario}
        worst = dict(item)
        for metric in METRICS:
            item[f"{metric}_auc"] = normalized_curve_mean(combined, metric)
            worst[f"{metric}_worst"] = float(group[metric].max())
        item["intensity_min"] = float(combined["level"].min())
        item["intensity_max"] = float(combined["level"].max())
        item["intensity_point_count"] = int(combined["level"].nunique())
        item["curve_summary_definition"] = "trapezoidal integral divided by declared intensity range"
        auc_rows.append(item)
        worst_rows.append(worst)
    auc = pd.DataFrame(auc_rows)
    worst = pd.DataFrame(worst_rows)
    auc.to_csv(output / "stress_pair_extended_auc_seed_level.csv", index=False)
    worst.to_csv(output / "stress_pair_extended_worst_seed_level.csv", index=False)

    comparison_rows = []
    for scenario in SCENARIOS:
        for metric in METRICS:
            column = f"{metric}_auc"
            means = auc[auc.scenario == scenario].groupby("model")[column].mean()
            rast = float(means["rast_gru"])
            ocm = float(means["rast_gru_v2"])
            comparison_rows.append({"scenario": scenario, "metric": metric, "rast_auc": rast, "ocm_auc": ocm,
                                    "ocm_minus_rast": ocm - rast,
                                    "relative_difference": (ocm - rast) / abs(rast) if rast else np.nan})
    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(output / "stress_pair_extended_auc_comparison.csv", index=False)
    uncertainty, equivalence_sensitivity = family_uncertainty(auc)
    uncertainty.to_csv(output / "stress_family_uncertainty.csv", index=False)
    equivalence_sensitivity.to_csv(output / "stress_equivalence_margin_sensitivity.csv", index=False)

    worst_comparison_rows = []
    for scenario in SCENARIOS:
        for metric in METRICS:
            column = f"{metric}_worst"
            means = worst[worst.scenario == scenario].groupby("model")[column].mean()
            rast = float(means["rast_gru"])
            ocm = float(means["rast_gru_v2"])
            worst_comparison_rows.append(
                {
                    "scenario": scenario,
                    "metric": metric,
                    "rast_worst": rast,
                    "ocm_worst": ocm,
                    "ocm_minus_rast": ocm - rast,
                }
            )
    worst_comparison = pd.DataFrame(worst_comparison_rows)
    worst_comparison.to_csv(output / "stress_pair_extended_worst_comparison.csv", index=False)

    def heatmap_matrices(frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
        raw = frame.pivot(index="metric", columns="scenario", values="ocm_minus_rast")
        metric_order = [metric for metric in METRICS if metric in raw.index]
        raw = raw.loc[metric_order, list(SCENARIOS)]
        normalized = np.sign(raw)
        raw.index = normalized.index = [DISPLAY[item] for item in raw.index]
        raw.columns = normalized.columns = [SCENARIO_DISPLAY[item] for item in raw.columns]
        return raw, normalized

    primary_metrics = (
        "rmse", "critical_30_late_prediction_ratio", "critical_30_mean_late_excess",
        "critical_30_late_cvar95", "critical_30_decision_cost_5",
    )

    def primary(frame: pd.DataFrame) -> pd.DataFrame:
        return frame[frame["metric"].isin(primary_metrics)].copy()

    auc_raw, auc_color = heatmap_matrices(primary(comparison))
    worst_raw, worst_color = heatmap_matrices(primary(worst_comparison))
    fig, axes = plt.subplots(2, 1, figsize=(13.4, 9.0), sharex=True)
    for ax, raw_values, color_values, title in (
        (axes[0], auc_raw, auc_color, "(a) Degradation-curve mean difference"),
        (axes[1], worst_raw, worst_color, "(b) Worst evaluated level difference"),
    ):
        sns.heatmap(
            color_values,
            annot=raw_values,
            fmt="+.3f",
            center=0,
            vmin=-1,
            vmax=1,
            cmap=ListedColormap(["#4c78a8", "#f2f2f2", "#c44e52"]),
            linewidths=.6,
            cbar_kws={"label": "Direction only", "ticks": [-1, 0, 1]},
            ax=ax,
        )
        ax.collections[0].colorbar.set_ticklabels(["OCM-Asym lower", "equal", "RAST lower"])
        ax.set_xlabel("Controlled sensor-degradation family")
        ax.set_ylabel("Metric (cell text keeps original units)")
        ax.set_title(title, loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(figure_dir / "fig_stress_nine_family_heatmap.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "fig_stress_nine_family_heatmap.png", dpi=400, bbox_inches="tight")
    plt.close(fig)

    failure = raw[raw.scenario == "global_sensor_missing"].copy()
    summary_rows = []
    for (model, level), group in failure.groupby(["model", "level"]):
        low, high = nested_seed_ci(group, "critical_30_late_prediction_ratio", reps=3000,
                                   seed=20260716 + int(level * 1000) + (1 if model == "rast_gru_v2" else 0))
        summary_rows.append({"model": model, "level": level,
                             "lpr_mean": group.critical_30_late_prediction_ratio.mean(),
                             "ci95_low": low, "ci95_high": high})
    failure_summary = pd.DataFrame(summary_rows)
    failure_summary.to_csv(output / "stress_global_missing_failure_curve.csv", index=False)
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    styles = {"rast_gru": ("RAST-GRU", "#4c78a8", "o"), "rast_gru_v2": ("OCM-MST-GRU", "#d95f02", "s")}
    for model, group in failure_summary.groupby("model"):
        label, color, marker = styles[model]
        group = group.sort_values("level")
        ax.plot(group.level, group.lpr_mean, label=label, color=color, marker=marker, linewidth=2)
        ax.fill_between(group.level, group.ci95_low, group.ci95_high, color=color, alpha=.14)
    ax.set_xlabel("Global sensor-missing rate")
    ax.set_ylabel("LPR@30")
    ax.grid(linestyle=":", alpha=.4)
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig(figure_dir / "fig_stress_global_missing_failure.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "fig_stress_global_missing_failure.png", dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(
        f"STRESS_PAIR_EXTENSION_READY rows={len(raw)} "
        f"auc_comparisons={len(comparison)} worst_comparisons={len(worst_comparison)}"
    )


if __name__ == "__main__":
    main()
