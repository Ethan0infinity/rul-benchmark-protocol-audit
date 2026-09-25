from __future__ import annotations

import argparse
import itertools
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_normalized_perturbation_evidence import (
    centered_bootstrap_p,
    engine_count_matrix,
    holm_adjust,
    sampled_seed_metrics,
)
from analyze_fixed_benchmark_evidence import exact_seed_signflip_p
from run_formal_benchmark import FORMAL_BENCHMARK_SEEDS


VARIANTS = {
    "global": "Global standardization",
    "official_state": "Rounded official-state grouping",
    "kmeans": "K-means condition grouping",
    "continuous": "Continuous condition correction",
}
METRICS = ("rmse", "nasa_per_engine", "lpr30")


def run_dir(results_dir: Path, variant: str, seed: int) -> Path:
    if variant == "kmeans":
        return results_dir / f"paper_main_v3_seed{seed}" / "FD004" / "rast_gru_v2"
    return (
        results_dir
        / f"condition_control_{variant}_fd004_seed{seed}"
        / "FD004"
        / "rast_gru_v2"
    )


def load_arrays(results_dir: Path) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    truth_reference: np.ndarray | None = None
    units_reference: np.ndarray | None = None
    predictions: dict[str, list[np.ndarray]] = {variant: [] for variant in VARIANTS}
    for variant in VARIANTS:
        for seed in FORMAL_BENCHMARK_SEEDS:
            path = run_dir(results_dir, variant, seed) / "test_predictions.csv"
            if not path.exists():
                raise FileNotFoundError(
                    f"Missing formal condition-control prediction: {path}. "
                    "Run scripts/run_condition_normalization_controls.py first."
                )
            frame = pd.read_csv(path).sort_values("unit_id")
            truth = frame["true_rul"].to_numpy(float)
            units = frame["unit_id"].to_numpy(int)
            if truth_reference is None:
                truth_reference = truth
                units_reference = units
            elif not np.array_equal(units_reference, units) or not np.allclose(truth_reference, truth):
                raise ValueError(f"Official FD004 test engines differ for variant={variant} seed={seed}")
            predictions[variant].append(frame["pred_rul"].to_numpy(float))
    assert truth_reference is not None and units_reference is not None
    return truth_reference, {key: np.stack(value) for key, value in predictions.items()}, units_reference


def analyze(
    results_dir: Path,
    *,
    reps: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    truth, predictions, _ = load_arrays(results_dir)
    rng = np.random.default_rng(random_seed)
    seed_count = len(FORMAL_BENCHMARK_SEEDS)
    seed_draw = rng.integers(0, seed_count, size=(reps, seed_count))
    engine_draw = rng.integers(0, len(truth), size=(reps, len(truth)))
    counts = engine_count_matrix(engine_draw, len(truth))
    rep_index = np.arange(reps)[:, None]
    unit_counts = np.ones((len(truth), 1), dtype=np.int16)

    summary_rows: list[dict[str, object]] = []
    for variant, display in VARIANTS.items():
        for metric in METRICS:
            seed_values = np.asarray(
                [
                    sampled_seed_metrics(
                        truth,
                        predictions[variant][index : index + 1],
                        unit_counts,
                        metric,
                    )[0, 0]
                    for index in range(seed_count)
                ]
            )
            summary_rows.append(
                {
                    "variant": variant,
                    "display_name": display,
                    "metric": metric,
                    "mean": float(seed_values.mean()),
                    "seed_sd": float(seed_values.std(ddof=1)),
                    "seed_min": float(seed_values.min()),
                    "seed_max": float(seed_values.max()),
                    "seed_count": seed_count,
                }
            )

    contrast_rows: list[dict[str, object]] = []
    kmeans_sample = {
        metric: sampled_seed_metrics(truth, predictions["kmeans"], counts, metric)
        for metric in METRICS
    }
    for comparator in ("global", "official_state", "continuous"):
        for metric in METRICS:
            comparator_sample = sampled_seed_metrics(
                truth,
                predictions[comparator],
                counts,
                metric,
            )
            samples = (
                kmeans_sample[metric][seed_draw, rep_index].mean(axis=1)
                - comparator_sample[seed_draw, rep_index].mean(axis=1)
            )
            seed_effects = np.asarray(
                [
                    sampled_seed_metrics(
                        truth,
                        predictions["kmeans"][index : index + 1],
                        unit_counts,
                        metric,
                    )[0, 0]
                    - sampled_seed_metrics(
                        truth,
                        predictions[comparator][index : index + 1],
                        unit_counts,
                        metric,
                    )[0, 0]
                    for index in range(seed_count)
                ]
            )
            observed = float(seed_effects.mean())
            p_value = centered_bootstrap_p(samples, observed)
            low, high = np.quantile(samples, [0.025, 0.975])
            contrast_rows.append(
                {
                    "comparison": f"kmeans minus {comparator}",
                    "comparator": comparator,
                    "metric": metric,
                    "mean_difference": observed,
                    "nominal_crossed_ci95_low": float(low),
                    "nominal_crossed_ci95_high": float(high),
                    "centered_bootstrap_p": p_value,
                    "bootstrap_p_mcse": math.sqrt(p_value * (1.0 - p_value) / (reps + 1)),
                    "exact_seed_signflip_p": exact_seed_signflip_p(seed_effects),
                    "bootstrap_repetitions": reps,
                    "evidence_level": "RETROSPECTIVE NORMALIZATION SENSITIVITY",
                }
            )
    contrasts = pd.DataFrame(contrast_rows)
    contrasts["holm_adjusted_p_all9"] = holm_adjust(contrasts["centered_bootstrap_p"].to_numpy(float))
    contrasts["decision"] = np.where(
        contrasts["holm_adjusted_p_all9"] < 0.05,
        np.where(
            contrasts["mean_difference"] < 0.0,
            "crossed-Holm direction: lower K-means; exact five-seed unresolved",
            "crossed-Holm direction: lower comparator; exact five-seed unresolved",
        ),
        "no adjusted evidence",
    )
    return pd.DataFrame(summary_rows), contrasts


def plot_summary(summary: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(9.3, 3.4))
    colors = ["#5b7fa3", "#4e9a8d", "#b8863b", "#8b6f9c"]
    for ax, metric in zip(axes, METRICS):
        part = summary[summary["metric"] == metric].copy()
        part["variant"] = pd.Categorical(part["variant"], categories=list(VARIANTS), ordered=True)
        part = part.sort_values("variant")
        x = np.arange(len(part))
        ax.bar(
            x,
            part["mean"],
            yerr=part["seed_sd"],
            color=colors,
            capsize=3,
            linewidth=0.7,
            edgecolor="#333333",
        )
        ax.set_xticks(x, ["Global", "Official", "K-means", "Continuous"], rotation=28, ha="right")
        ax.set_title(metric.replace("_", " ").upper(), loc="left", fontsize=10.5)
        ax.set_ylabel("Mean over five seeds")
        ax.grid(axis="y", alpha=0.25, linestyle=":")
    fig.suptitle("FD004 normalization controls under an identical OCM-Asym protocol", fontsize=11.5)
    fig.text(0.5, 0.01, "Error bars are seed-level SDs, not confidence intervals.", ha="center", fontsize=8.5)
    fig.tight_layout(rect=(0, 0.05, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def write_latex(summary: pd.DataFrame, contrasts: pd.DataFrame, output: Path) -> None:
    metric_display = {
        "rmse": "RMSE",
        "nasa_per_engine": "NASA/engine",
        "lpr30": "LPR@30",
    }
    rows = []
    for row in summary.to_dict("records"):
        rows.append(
            f"{row['display_name']} & {metric_display[row['metric']]} & "
            f"{row['mean']:.3f} & {row['seed_sd']:.3f} & "
            f"[{row['seed_min']:.3f}, {row['seed_max']:.3f}] \\\\"
        )
    contrast_rows = []
    for row in contrasts.to_dict("records"):
        contrast_rows.append(
            f"{VARIANTS[row['comparator']]} & {metric_display[row['metric']]} & "
            f"{row['mean_difference']:.3f} & "
            f"[{row['nominal_crossed_ci95_low']:.3f}, {row['nominal_crossed_ci95_high']:.3f}] & "
            f"{row['holm_adjusted_p_all9']:.3f} & {row['exact_seed_signflip_p']:.3f} \\\\"
        )
    text = "\n".join(
        [
            r"\begin{table*}[!t]",
            r"\centering",
            r"\scriptsize",
            r"\caption{[RETROSPECTIVE NORMALIZATION SENSITIVITY] FD004 normalization variants under the same engine splits, seeds, selected sensors, OCM-Asym configuration, objective, optimizer, checkpoint rule, and 80-epoch budget. Panel A reports five-seed descriptive summaries. Panel B reports K-means minus comparator effects; negative values favor K-means. Family S-NORM-9 contains all nine contrasts. Its Holm and exact sign-flip values are diagnostics, not confirmatory decisions.}",
            r"\label{tab:condition-normalization-controls}",
            r"\textbf{Panel A: absolute results}\\[2pt]",
            r"\begin{tabular}{llrrr}",
            r"\toprule",
            r"Variant & Metric & Mean & Seed SD & Seed range \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\vspace{4pt}",
            r"\textbf{Panel B: K-means minus control}\\[2pt]",
            r"\begin{tabular}{llrrrr}",
            r"\toprule",
            r"Control & Metric & Mean $\Delta$ & Nominal 95\% CI & Holm$_9$ $p$ & Sign-flip $p$ \\",
            r"\midrule",
            *contrast_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    output.write_text(text, encoding="utf-8")

    absolute_text = "\n".join(
        [
            r"\begin{table}[!t]",
            r"\centering",
            r"\scriptsize",
        r"\caption{[RETROSPECTIVE SENSITIVITY; S-NORM-9] Absolute FD004 normalization-control results. Values are five-seed descriptive summaries under the same OCM-Asym configuration and budget.}",
            r"\label{tab:condition-normalization-absolute}",
            r"\begin{tabular}{llrrr}",
            r"\toprule",
            r"Variant & Metric & Mean & Seed SD & Seed range \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    (output.parent / "table_condition_normalization_absolute.tex").write_text(
        absolute_text,
        encoding="utf-8",
    )
    contrast_text = "\n".join(
        [
            r"\begin{table*}[!t]",
            r"\centering",
            r"\scriptsize",
        r"\caption{[RETROSPECTIVE SENSITIVITY; S-NORM-9] K-means minus matched normalization controls. Negative effects favor K-means. Nominal intervals, Holm$_9$, and exact five-seed sign-flip values are diagnostic sensitivity summaries; none is assigned a confirmatory role.}",
            r"\label{tab:condition-normalization-contrasts}",
            r"\begin{tabular}{llrrrr}",
            r"\toprule",
            r"Control & Metric & Mean $\Delta$ & Nominal interval & Holm$_9$ & Sign-flip \\",
            r"\midrule",
            *contrast_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    (output.parent / "table_condition_normalization_contrasts.tex").write_text(
        contrast_text,
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the FD004 normalization-control experiment.")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "paper_outputs" / "analysis_working"))
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    summary, contrasts = analyze(
        Path(args.results_dir),
        reps=args.bootstrap_reps,
        random_seed=2026072606,
    )
    summary.to_csv(output / "condition_normalization_summary.csv", index=False)
    contrasts.to_csv(output / "condition_normalization_contrasts.csv", index=False)
    plot_summary(summary, output / "figures" / "fig_condition_normalization_controls")
    write_latex(summary, contrasts, output / "table_condition_normalization_controls.tex")
    print(
        f"CONDITION_NORMALIZATION_ANALYSIS_READY summary={len(summary)} contrasts={len(contrasts)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
