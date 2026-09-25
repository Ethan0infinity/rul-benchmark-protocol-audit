from __future__ import annotations

import argparse
import math
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUT_TABLE = PROJECT_ROOT / "paper_outputs" / "revised_tables"
OUT_FIG = PROJECT_ROOT / "paper_outputs" / "revised_figures"

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "font.size": 13,
        "axes.titlesize": 15,
        "axes.labelsize": 13,
        "xtick.labelsize": 12,
        "ytick.labelsize": 12,
        "legend.fontsize": 12,
        "figure.dpi": 160,
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
ORDER = [
    "gru", "lstm", "tcn", "tcn_gru", "rast_gru", "cnn_lstm", "attention_gru",
    "bigru_attention", "transformer_lite", "dual_attention_tcn", "sensor_graph_gru", "quantile_gru", "rast_gru_v2",
]
CORE_ORDER = [
    "gru", "tcn_gru", "rast_gru", "cnn_lstm", "attention_gru", "transformer_lite",
    "dual_attention_tcn", "sensor_graph_gru", "quantile_gru", "rast_gru_v2",
]
MULTI_ORDER = ORDER.copy()
FORMAL_SEEDS = [42, 123, 2024, 2025, 2026]


def fmt_pm(mean: float, std: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} $\\pm$ {std:.{digits}f}"


def holm_adjust(p_values: list[float]) -> list[float]:
    n = len(p_values)
    order = np.argsort(p_values)
    adjusted = np.empty(n, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (n - rank) * p_values[idx]
        running = max(running, val)
        adjusted[idx] = min(running, 1.0)
    return adjusted.tolist()


def add_clean_rows_for_missing_curves(df: pd.DataFrame, scenarios: list[str]) -> pd.DataFrame:
    clean = df[df["scenario"] == "clean"].copy()
    rows = [df]
    for sc in scenarios:
        c = clean.copy()
        c["scenario"] = sc
        c["level"] = 0.0
        rows.append(c)
    return pd.concat(rows, ignore_index=True)


def build_tables() -> None:
    # Dataset summary with number of conditions/fault modes.
    ds = pd.read_csv(PROJECT_ROOT / "reports" / "dataset_summary.csv")
    ds["conditions"] = ds["subset"].map({"FD001": 1, "FD002": 6, "FD003": 1, "FD004": 6})
    ds["fault_modes"] = ds["subset"].map({"FD001": 1, "FD002": 1, "FD003": 2, "FD004": 2})
    ds = ds[["subset", "train_rows", "test_rows", "train_units", "test_units", "rul_rows", "conditions", "fault_modes"]]
    ds.to_csv(OUT_TABLE / "dataset_summary_revised.csv", index=False)

    # Single-seed table for all implemented baselines.
    main = pd.read_csv(PROJECT_ROOT / "paper_outputs" / "tables" / "table_main_accuracy.csv")
    crit = pd.read_csv(PROJECT_ROOT / "paper_outputs" / "tables" / "table_critical_zone.csv")
    single = main.merge(
        crit[["experiment_name", "subset", "model", "test_critical_30_late_prediction_ratio"]],
        on=["experiment_name", "subset", "model"],
        how="left",
    )
    single["model_display"] = single["model"].map(MODEL_DISPLAY)
    single["model_order"] = single["model"].map({m: i for i, m in enumerate(ORDER)})
    single = single.sort_values(["subset", "model_order"])
    single[[
        "subset", "model", "model_display", "test_rmse", "test_mae", "test_nasa_score",
        "test_critical_30_late_prediction_ratio", "parameters", "single_sample_inference_ms",
        "cpu_single_sample_inference_ms",
    ]].to_csv(OUT_TABLE / "single_seed_all_baselines_revised.csv", index=False)

    # Multiseed summary formatted.
    ms = pd.read_csv(PROJECT_ROOT / "paper_outputs" / "tables" / "table_multiseed_summary.csv")
    ms["model_order"] = ms["model"].map({m: i for i, m in enumerate(MULTI_ORDER)})
    ms = ms.sort_values(["subset", "model_order"])
    ms["rmse_pm"] = ms.apply(lambda r: fmt_pm(r.test_rmse_mean, r.test_rmse_std), axis=1)
    ms["mae_pm"] = ms.apply(lambda r: fmt_pm(r.test_mae_mean, r.test_mae_std), axis=1)
    ms["nasa_pm"] = ms.apply(lambda r: fmt_pm(r.test_nasa_score_mean, r.test_nasa_score_std), axis=1)
    ms["lpr30_pm"] = ms.apply(lambda r: fmt_pm(r.test_critical_30_late_prediction_ratio_mean, r.test_critical_30_late_prediction_ratio_std), axis=1)
    ms.to_csv(OUT_TABLE / "multiseed_summary_revised.csv", index=False)

    overall = pd.read_csv(PROJECT_ROOT / "paper_outputs" / "tables" / "table_multiseed_overall.csv")
    overall["model_order"] = overall["model"].map({m: i for i, m in enumerate(MULTI_ORDER)})
    overall = overall.sort_values("model_order")
    overall["rmse_pm"] = overall.apply(lambda r: fmt_pm(r.test_rmse_mean, r.test_rmse_std), axis=1)
    overall["mae_pm"] = overall.apply(lambda r: fmt_pm(r.test_mae_mean, r.test_mae_std), axis=1)
    overall["nasa_pm"] = overall.apply(lambda r: fmt_pm(r.test_nasa_score_mean, r.test_nasa_score_std), axis=1)
    overall["lpr30_pm"] = overall.apply(lambda r: fmt_pm(r.test_critical_30_late_prediction_ratio_mean, r.test_critical_30_late_prediction_ratio_std), axis=1)
    overall.to_csv(OUT_TABLE / "multiseed_overall_revised.csv", index=False)

    # Severe late prediction ratios from predictions for the completed five-seed formal grid.
    rows = []
    for seed in FORMAL_SEEDS:
        for subset in ["FD001", "FD002", "FD003", "FD004"]:
            for model in MULTI_ORDER:
                pred_path = PROJECT_ROOT / "results" / f"paper_main_v3_seed{seed}" / subset / model / "test_predictions.csv"
                if not pred_path.exists():
                    continue
                p = pd.read_csv(pred_path)
                err = p["pred_rul"] - p["true_rul"]
                zone = p["true_rul"] <= 30
                if int(zone.sum()) == 0:
                    slpr5 = slpr10 = 0.0
                else:
                    slpr5 = float(((err > 5) & zone).sum() / zone.sum())
                    slpr10 = float(((err > 10) & zone).sum() / zone.sum())
                rows.append({"seed": seed, "subset": subset, "model": model, "model_display": MODEL_DISPLAY[model], "slpr30_5": slpr5, "slpr30_10": slpr10})
    sev = pd.DataFrame(rows)
    sev_summary = sev.groupby(["subset", "model", "model_display"], as_index=False).agg(
        slpr30_5_mean=("slpr30_5", "mean"),
        slpr30_5_std=("slpr30_5", "std"),
        slpr30_10_mean=("slpr30_10", "mean"),
        slpr30_10_std=("slpr30_10", "std"),
    )
    sev_summary["model_order"] = sev_summary["model"].map({m: i for i, m in enumerate(MULTI_ORDER)})
    sev_summary = sev_summary.sort_values(["subset", "model_order"])
    sev.to_csv(OUT_TABLE / "severe_late_seed_level.csv", index=False)
    sev_summary.to_csv(OUT_TABLE / "severe_late_summary_revised.csv", index=False)

    # Holm-corrected stats.
    stats = pd.read_csv(PROJECT_ROOT / "paper_outputs" / "tables" / "table_statistical_tests.csv")
    stats["p_holm"] = np.nan
    for _, idx in stats.groupby(["subset", "metric"]).groups.items():
        p = stats.loc[idx, "p_value"].tolist()
        stats.loc[idx, "p_holm"] = holm_adjust(p)
    stats["significant_holm_0_05"] = stats["p_holm"] < 0.05
    stats.to_csv(OUT_TABLE / "statistical_tests_holm_revised.csv", index=False)
    fd004_lpr = stats[(stats["subset"] == "FD004") & (stats["metric"] == "critical_30_late_indicator")].copy()
    if not fd004_lpr.empty:
        fd004_summary = fd004_lpr.groupby("baseline_model", as_index=False).agg(
            seed_count=("experiment_name", "nunique"),
            baseline_lpr30_mean=("mean_baseline_metric", "mean"),
            proposed_lpr30_mean=("mean_proposed_metric", "mean"),
            mean_diff=("paired_mean_diff_baseline_minus_proposed", "mean"),
            mean_diff_ci95_low_envelope=("paired_mean_diff_ci95_low", "min"),
            mean_diff_ci95_high_envelope=("paired_mean_diff_ci95_high", "max"),
            baseline_only_total=("baseline_only", "sum"),
            proposed_only_total=("proposed_only", "sum"),
            discordant_pairs_total=("discordant_pairs", "sum"),
            seeds_ci_excludes_zero=(
                "paired_mean_diff_ci95_low",
                lambda x: int(
                    np.sum(
                        (x.to_numpy(dtype=float) > 0)
                        | (
                            fd004_lpr.loc[x.index, "paired_mean_diff_ci95_high"].to_numpy(dtype=float)
                            < 0
                        )
                    )
                ),
            ),
            mcnemar_p_min=("p_value", "min"),
            mcnemar_p_max=("p_value", "max"),
            holm_p_min=("p_holm", "min"),
            holm_p_max=("p_holm", "max"),
            holm_significant_seed_count=("significant_holm_0_05", "sum"),
        )
        fd004_summary["holm_p_range"] = fd004_summary.apply(
            lambda r: f"{float(r.holm_p_min):.3g}-{float(r.holm_p_max):.3g}",
            axis=1,
        )
        fd004_summary["mean_diff_ci95_envelope"] = fd004_summary.apply(
            lambda r: f"[{float(r.mean_diff_ci95_low_envelope):.3f}, {float(r.mean_diff_ci95_high_envelope):.3f}]",
            axis=1,
        )
        fd004_summary.to_csv(OUT_TABLE / "statistical_fd004_lpr30_mcnemar_summary.csv", index=False)

    # Multi-objective ablation view: lower clean error, NASA score, and LPR@30 are all preferred.
    ablation = pd.read_csv(PROJECT_ROOT / "paper_outputs" / "tables" / "table_ablation.csv")
    needed = ["ablation_variant", "test_rmse", "test_nasa_score", "test_critical_30_late_prediction_ratio"]
    if set(needed).issubset(ablation.columns):
        multi = ablation.copy()
        for metric in ["test_rmse", "test_nasa_score", "test_critical_30_late_prediction_ratio"]:
            values = pd.to_numeric(multi[metric], errors="coerce")
            span = values.max() - values.min()
            multi[f"{metric}_norm"] = 0.0 if not np.isfinite(span) or span == 0 else (values - values.min()) / span
        multi["multi_objective_tradeoff_score"] = multi[
            [
                "test_rmse_norm",
                "test_nasa_score_norm",
                "test_critical_30_late_prediction_ratio_norm",
            ]
        ].mean(axis=1)
        columns = [
            column
            for column in [
                "subset",
                "model",
                "model_display",
                "ablation_variant",
                "test_rmse",
                "test_nasa_score",
                "test_critical_30_late_prediction_ratio",
                "test_rmse_norm",
                "test_nasa_score_norm",
                "test_critical_30_late_prediction_ratio_norm",
                "multi_objective_tradeoff_score",
            ]
            if column in multi.columns
        ]
        multi = multi[columns].sort_values("multi_objective_tradeoff_score").reset_index(drop=True)
        multi.to_csv(OUT_TABLE / "ablation_multi_objective_revised.csv", index=False)

    # Robustness-oriented ablation: checks whether modules help under degradation, not only clean data.
    ablation_robustness_runs = {
        "full": PROJECT_ROOT / "results" / "ablation_full" / "FD004" / "rast_gru_v2" / "robustness.csv",
        "w/o reliability gate": PROJECT_ROOT / "results" / "ablation_no_reliability_gate" / "FD004" / "rast_gru_v2" / "robustness.csv",
        "w/o missing mask": PROJECT_ROOT / "results" / "ablation_no_missing_mask" / "FD004" / "rast_gru_v2" / "robustness.csv",
        "w/o asymmetric loss": PROJECT_ROOT / "results" / "ablation_weighted_huber_no_asymmetry" / "FD004" / "rast_gru_v2" / "robustness.csv",
        "w/o condition norm": PROJECT_ROOT / "results" / "ablation_no_condition_norm" / "FD004" / "rast_gru_v2" / "robustness.csv",
    }
    selected_scenarios = [
        ("clean", 0.0),
        ("block_missing", 0.3),
        ("sensor_drift", 0.1),
        ("global_sensor_missing", 0.2),
    ]
    robustness_rows = []
    for variant, csv_path in ablation_robustness_runs.items():
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        for scenario, level in selected_scenarios:
            row = df[(df["scenario"] == scenario) & (df["level"] == level)]
            if row.empty:
                continue
            r = row.iloc[0]
            robustness_rows.append(
                {
                    "subset": "FD004",
                    "variant": variant,
                    "scenario": scenario,
                    "level": level,
                    "rmse": round(float(r["rmse"]), 3),
                    "nasa_score": round(float(r["nasa_score"]), 3),
                    "lpr30": round(float(r["critical_30_late_prediction_ratio"]), 3),
                    "slpr30_5": round(float(r["critical_30_severe_late_5_ratio"]), 3),
                    "slpr30_10": round(float(r["critical_30_severe_late_10_ratio"]), 3),
                }
            )
    if robustness_rows:
        pd.DataFrame(robustness_rows).to_csv(OUT_TABLE / "ablation_robustness_fd004_revised.csv", index=False)

    threshold = PROJECT_ROOT / "paper_outputs" / "tables" / "table_threshold_sensitivity.csv"
    if threshold.exists():
        thresh = pd.read_csv(threshold)
        thresh.to_csv(OUT_TABLE / "threshold_sensitivity_revised.csv", index=False)
        fd004_proposed = thresh[(thresh["subset"] == "FD004") & (thresh["model"] == "rast_gru_v2")].copy()
        if not fd004_proposed.empty:
            fd004_proposed.to_csv(OUT_TABLE / "threshold_sensitivity_fd004_ocm_revised.csv", index=False)

    multi_ablation = PROJECT_ROOT / "paper_outputs" / "tables" / "table_ablation_fd004_multiseed.csv"
    if multi_ablation.exists():
        ab = pd.read_csv(multi_ablation)
        ab.to_csv(OUT_TABLE / "ablation_fd004_multiseed_revised.csv", index=False)
        if "full" in set(ab["ablation_variant"]):
            full = ab[ab["ablation_variant"] == "full"].iloc[0]
            deltas = ab.copy()
            for metric in [
                "test_rmse_mean",
                "test_nasa_score_mean",
                "test_critical_30_late_prediction_ratio_mean",
                "test_critical_30_severe_late_5_ratio_mean",
                "test_critical_30_severe_late_10_ratio_mean",
            ]:
                if metric in deltas.columns:
                    deltas[f"delta_vs_full_{metric}"] = pd.to_numeric(deltas[metric], errors="coerce") - float(full[metric])
            deltas.to_csv(OUT_TABLE / "ablation_fd004_multiseed_delta_revised.csv", index=False)


def plot_multiseed_rmse() -> None:
    ms = pd.read_csv(PROJECT_ROOT / "paper_outputs" / "tables" / "table_multiseed_summary.csv")
    subsets = ["FD001", "FD002", "FD003", "FD004"]
    if ms.empty:
        return
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.2), sharex=False)
    axes = axes.ravel()
    for ax, subset in zip(axes, subsets):
        sub = ms[ms["subset"] == subset].copy()
        sub["model_order"] = sub["model"].map({m: i for i, m in enumerate(MULTI_ORDER)})
        sub = sub.sort_values("model_order", ascending=False)
        labels = [MODEL_DISPLAY.get(m, m) for m in sub["model"]]
        y = np.arange(len(sub))
        colors = ["#2c7fb8" if model != "rast_gru_v2" else "#d95f0e" for model in sub["model"]]
        ax.barh(y, sub["test_rmse_mean"], xerr=sub["test_rmse_std"], color=colors, alpha=0.88, capsize=3)
        ax.set_yticks(y)
        ax.set_yticklabels(labels, fontsize=10)
        ax.set_xlabel("RMSE")
        ax.set_title(subset)
        ax.grid(axis="x", linestyle=":", alpha=0.35)
    fig.suptitle("Five-seed RMSE by subset", fontsize=15)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    plt.savefig(OUT_FIG / "fig_multiseed_rmse_errorbar.png", dpi=600)
    plt.savefig(OUT_FIG / "fig_multiseed_rmse_errorbar.pdf", bbox_inches="tight")
    plt.close()


def plot_fd004_pareto() -> None:
    ms = pd.read_csv(PROJECT_ROOT / "paper_outputs" / "tables" / "table_multiseed_summary.csv")
    sub = ms[ms["subset"] == "FD004"].copy()
    sub.loc[sub["model"] == "rast_gru_v2", "model_display"] = "OCM-Asym"
    core_path = PROJECT_ROOT / "paper_outputs" / "advanced_evidence" / "ocm_core_asym_subset_seed.csv"
    if core_path.exists():
        core = pd.read_csv(core_path)
        core = core[(core["variant"] == "core") & (core["subset"] == "FD004")]
        if len(core) == len(FORMAL_SEEDS):
            sub = pd.concat(
                [
                    sub,
                    pd.DataFrame(
                        [
                            {
                                "subset": "FD004",
                                "model": "rast_gru_v2_core",
                                "model_display": "OCM-Core",
                                "test_rmse_mean": core["rmse"].mean(),
                                "test_rmse_std": core["rmse"].std(ddof=1),
                                "test_critical_30_late_prediction_ratio_mean": core["lpr30"].mean(),
                                "test_critical_30_late_prediction_ratio_std": core["lpr30"].std(ddof=1),
                            }
                        ]
                    ),
                ],
                ignore_index=True,
            )
    sub["model_order"] = sub["model"].map({m: i for i, m in enumerate(MULTI_ORDER)})
    sub.loc[sub["model"] == "rast_gru_v2_core", "model_order"] = len(MULTI_ORDER)
    sub = sub.sort_values("model_order").reset_index(drop=True)
    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    neutral = "#8A8F98"
    signal = {"rast_gru": "#2F6690", "rast_gru_v2": "#C65D2E", "rast_gru_v2_core": "#2C8C6B"}
    for i, (_, r) in enumerate(sub.iterrows(), start=1):
        color = signal.get(r["model"], neutral)
        highlighted = r["model"] in signal
        ax.scatter(
            r["test_rmse_mean"],
            r["test_critical_30_late_prediction_ratio_mean"],
            s=92 if highlighted else 70,
            color=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax.text(
            r["test_rmse_mean"],
            r["test_critical_30_late_prediction_ratio_mean"],
            str(i),
            ha="center",
            va="center",
            fontsize=8.0,
            color="white",
            weight="bold",
            zorder=4,
        )
        ax.errorbar(
            r["test_rmse_mean"],
            r["test_critical_30_late_prediction_ratio_mean"],
            xerr=r["test_rmse_std"],
            yerr=r["test_critical_30_late_prediction_ratio_std"],
            capsize=3,
            fmt="none",
            ecolor=color,
            elinewidth=1.25 if highlighted else 0.85,
            alpha=0.8 if highlighted else 0.48,
            zorder=2,
        )
    legend_text = "\n".join(f"{i}. {name}" for i, name in enumerate(sub["model_display"], start=1))
    ax.text(
        1.02,
        0.98,
        legend_text,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=8.2,
        linespacing=1.18,
        bbox={"boxstyle": "square,pad=0.35", "facecolor": "white", "edgecolor": "0.82", "alpha": 0.98},
    )
    ax.set_xlabel("RMSE (lower is better)", fontsize=9)
    ax.set_ylabel("LPR@30 (lower is better)", fontsize=9)
    ax.tick_params(labelsize=8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(linestyle=":", color="0.82", linewidth=0.7)
    fig.tight_layout(rect=(0, 0, 0.82, 1))
    plt.savefig(OUT_FIG / "fig_fd004_pareto_tradeoff.png", dpi=600)
    plt.savefig(OUT_FIG / "fig_fd004_pareto_tradeoff.pdf", bbox_inches="tight")
    plt.savefig(OUT_FIG / "fig_fd004_pareto_tradeoff.svg", bbox_inches="tight")
    plt.savefig(OUT_FIG / "fig_fd004_pareto_tradeoff.tiff", dpi=600, bbox_inches="tight")
    plt.close()


def plot_predictions() -> None:
    pred = pd.read_csv(PROJECT_ROOT / "results" / "paper_main_v3_seed42" / "FD004" / "rast_gru_v2" / "test_predictions.csv")
    true = pred["true_rul"].to_numpy()
    est = pred["pred_rul"].to_numpy()
    err = est - true
    lim_max = max(true.max(), est.max()) + 5
    plt.figure(figsize=(6.2, 5.2))
    plt.scatter(true, est, s=18, alpha=0.75)
    plt.plot([0, lim_max], [0, lim_max], linestyle="--")
    plt.xlabel("True RUL")
    plt.ylabel("Predicted RUL")
    plt.title("FD004 true vs predicted RUL (OCM-MST-GRU, seed 42)")
    plt.tight_layout()
    plt.savefig(OUT_FIG / "fig_fd004_true_pred_scatter.png", dpi=600)
    plt.savefig(OUT_FIG / "fig_fd004_true_pred_scatter.pdf", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(6.2, 5.2))
    plt.scatter(true, err, s=18, alpha=0.75)
    plt.axhline(0, linestyle="--")
    plt.axvline(30, linestyle=":")
    plt.xlabel("True RUL")
    plt.ylabel("Prediction error (predicted - true)")
    plt.title("FD004 residuals and near-failure region")
    plt.tight_layout()
    plt.savefig(OUT_FIG / "fig_fd004_residuals.png", dpi=600)
    plt.savefig(OUT_FIG / "fig_fd004_residuals.pdf", bbox_inches="tight")
    plt.close()

    idx = np.argsort(true)[::-1]
    plt.figure(figsize=(8.2, 4.8))
    plt.plot(np.arange(len(idx)), true[idx], label="True RUL")
    plt.plot(np.arange(len(idx)), est[idx], label="Predicted RUL")
    plt.xlabel("Test engine sorted by true RUL")
    plt.ylabel("RUL")
    plt.title("FD004 sorted RUL diagnostic")
    plt.legend(frameon=False)
    plt.tight_layout()
    plt.savefig(OUT_FIG / "fig_fd004_sorted_rul.png", dpi=600)
    plt.savefig(OUT_FIG / "fig_fd004_sorted_rul.pdf", bbox_inches="tight")
    plt.close()

    near_err = err[true <= 30]
    plt.figure(figsize=(6.2, 4.8))
    plt.hist(near_err, bins=16)
    plt.axvline(0, linestyle="--")
    plt.xlabel("Prediction error within true RUL <= 30")
    plt.ylabel("Count")
    plt.title("Near-failure error distribution on FD004 (OCM-MST-GRU, seed 42)")
    plt.tight_layout()
    plt.savefig(OUT_FIG / "fig_fd004_near_failure_error_hist.png", dpi=600)
    plt.savefig(OUT_FIG / "fig_fd004_near_failure_error_hist.pdf", bbox_inches="tight")
    plt.close()


def plot_robustness() -> None:
    df = pd.read_csv(PROJECT_ROOT / "paper_outputs" / "advanced_evidence" / "stress_multiseed_summary.csv")
    df = add_clean_rows_for_missing_curves(df, ["window_random_missing", "global_sensor_missing", "block_missing"])
    # Predeclared display set for readable manuscript figures: a simple recurrent baseline,
    # the closest structural baseline, two stronger lightweight baselines, a compact
    # transformer baseline, and the proposed model. Full rows stay in the artifact tables.
    display_order = [
        "gru",
        "rast_gru",
        "attention_gru",
        "transformer_lite",
        "sensor_graph_gru",
        "quantile_gru",
        "rast_gru_v2",
    ]
    for scenario, metric, ylabel, fname, title in [
        ("window_random_missing", "rmse", "RMSE", "fig_fd004_random_missing_rmse.png", "FD004 random missing stress test"),
        ("window_random_missing", "critical_30_late_prediction_ratio", "LPR@30", "fig_fd004_random_missing_lpr30.png", "FD004 random missing late-risk test"),
        ("global_sensor_missing", "rmse", "RMSE", "fig_fd004_global_missing_rmse.png", "FD004 global missing stress test"),
        ("block_missing", "critical_30_late_prediction_ratio", "LPR@30", "fig_fd004_block_missing_lpr30.png", "FD004 block missing late-risk test"),
        ("sensor_drift", "critical_30_late_prediction_ratio", "LPR@30", "fig_fd004_drift_lpr30.png", "FD004 drift late-risk test"),
        ("sensor_drift", "nasa_score", "NASA score", "fig_fd004_drift_nasa.png", "FD004 drift NASA-score test"),
    ]:
        sub = df[(df["subset"] == "FD004") & (df["scenario"] == scenario)].copy()
        if sub.empty:
            continue
        plt.figure(figsize=(13.0, 8.8))
        for model in display_order:
            d = sub[sub["model"] == model].sort_values("level")
            if d.empty:
                continue
            is_proposed = model == "rast_gru_v2"
            plt.plot(
                d["level"],
                d[f"{metric}_mean"],
                marker="o",
                markersize=8.8 if is_proposed else 7.2,
                linewidth=3.4 if is_proposed else 2.7,
                alpha=0.98 if is_proposed else 0.86,
                label=MODEL_DISPLAY[model],
            )
            plt.fill_between(
                d["level"].to_numpy(float),
                d[f"{metric}_ci95_low"].to_numpy(float),
                d[f"{metric}_ci95_high"].to_numpy(float),
                alpha=0.16 if is_proposed else 0.08,
            )
        plt.xlabel("Perturbation level", fontsize=17)
        plt.ylabel(ylabel, fontsize=17)
        plt.title(title, fontsize=18)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.grid(linestyle=":", alpha=0.38)
        plt.legend(frameon=False, fontsize=14, ncol=2)
        plt.figtext(
            0.5,
            0.01,
            "Mean and nested-seed 95% interval over 5 composite training seeds x 5 perturbation seeds.",
            ha="center",
            fontsize=12,
        )
        plt.tight_layout(rect=(0, 0.035, 1, 1))
        plt.savefig(OUT_FIG / fname, dpi=600)
        plt.savefig(OUT_FIG / Path(fname).with_suffix(".pdf"), bbox_inches="tight")
        plt.close()


def plot_threshold_sensitivity() -> None:
    path = PROJECT_ROOT / "paper_outputs" / "tables" / "table_threshold_sensitivity.csv"
    if not path.exists():
        return
    df = pd.read_csv(path)
    sub = df[(df["subset"] == "FD004") & (df["model"] == "rast_gru_v2")].copy()
    if sub.empty:
        return
    plt.figure(figsize=(8.8, 5.8))
    for late_threshold, label in [(0.0, "LPR"), (5.0, "SLPR>5"), (10.0, "SLPR>10"), (15.0, "SLPR>15")]:
        d = sub[sub["late_error_threshold"] == late_threshold].sort_values("rul_threshold")
        if d.empty:
            continue
        plt.plot(d["rul_threshold"], d["ratio_mean"], marker="o", linewidth=2.2, markersize=6.5, label=label)
    plt.xlabel("Critical-zone true-RUL threshold", fontsize=14)
    plt.ylabel("Late-prediction ratio", fontsize=14)
    plt.title("FD004 OCM-MST-GRU threshold sensitivity", fontsize=15)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.grid(linestyle=":", alpha=0.35)
    plt.legend(frameon=False, fontsize=12)
    plt.tight_layout()
    plt.savefig(OUT_FIG / "fig_fd004_threshold_sensitivity.png", dpi=600)
    plt.savefig(OUT_FIG / "fig_fd004_threshold_sensitivity.pdf", bbox_inches="tight")
    plt.close()


def _default_manuscript_dir() -> Path | None:
    for candidate in PROJECT_ROOT.parent.iterdir():
        if candidate.is_dir() and (candidate / "main_revised.tex").exists():
            return candidate
    return None


def sync_figures_to_manuscript(manuscript_dir: Path | None) -> None:
    if manuscript_dir is None:
        print("[INFO] No manuscript directory found; revised figures were kept in project outputs only.")
        return
    figure_dir = manuscript_dir / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    for figure in [*OUT_FIG.glob("*.png"), *OUT_FIG.glob("*.pdf")]:
        shutil.copy2(figure, figure_dir / figure.name)
    print(f"Wrote manuscript figure copies to {figure_dir}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Build revised result tables and diagnostic figures.")
    default_manuscript = _default_manuscript_dir()
    parser.add_argument(
        "--manuscript-dir",
        default=str(default_manuscript) if default_manuscript is not None else "",
        help="Optional paper folder whose figures/ directory should receive figure copies. Defaults to the sibling folder containing main_revised.tex.",
    )
    parser.add_argument(
        "--no-manuscript-copy",
        action="store_true",
        help="Only write project outputs; do not copy figures into the manuscript folder.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    OUT_TABLE.mkdir(parents=True, exist_ok=True)
    OUT_FIG.mkdir(parents=True, exist_ok=True)
    build_tables()
    plot_multiseed_rmse()
    plot_fd004_pareto()
    plot_predictions()
    plot_robustness()
    plot_threshold_sensitivity()
    print(f"Wrote tables to {OUT_TABLE}")
    print(f"Wrote figures to {OUT_FIG}")
    manuscript_dir = None if args.no_manuscript_copy or not args.manuscript_dir else Path(args.manuscript_dir)
    sync_figures_to_manuscript(manuscript_dir)

if __name__ == "__main__":
    main()
