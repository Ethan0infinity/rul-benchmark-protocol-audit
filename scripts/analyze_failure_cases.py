from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mpl_cache"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from build_advanced_evidence import FORMAL_BENCHMARK_SEEDS, FORMAL_BENCHMARK_SUBSETS, load_prediction_grid

plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9.5})


def classify_disagreement(proposed_late: np.ndarray, baseline_late: np.ndarray) -> np.ndarray:
    return np.select(
        [baseline_late & ~proposed_late, proposed_late & ~baseline_late, proposed_late & baseline_late],
        ["proposed_safer", "proposed_later", "both_late"],
        default="neither_late",
    )


def build_failure_cases(
    grid: dict[tuple[int, str, str], pd.DataFrame], baseline: str = "rast_gru", proposed: str = "rast_gru_v2"
) -> pd.DataFrame:
    rows = []
    for seed in FORMAL_BENCHMARK_SEEDS:
        for subset in FORMAL_BENCHMARK_SUBSETS:
            base = grid[(seed, subset, baseline)].rename(columns={"pred_rul": "baseline_pred_rul"})
            prop = grid[(seed, subset, proposed)].rename(columns={"pred_rul": "proposed_pred_rul"})
            merged = base.merge(
                prop[["unit_id", "true_rul", "proposed_pred_rul"]],
                on=["unit_id", "true_rul"],
                how="inner",
                validate="one_to_one",
            )
            base_error = merged["baseline_pred_rul"].to_numpy(float) - merged["true_rul"].to_numpy(float)
            prop_error = merged["proposed_pred_rul"].to_numpy(float) - merged["true_rul"].to_numpy(float)
            critical = merged["true_rul"].to_numpy(float) <= 30.0
            base_late = critical & (base_error > 0.0)
            prop_late = critical & (prop_error > 0.0)
            merged = merged.assign(
                seed=seed,
                subset=subset,
                baseline_model=baseline,
                proposed_model=proposed,
                baseline_error=base_error,
                proposed_error=prop_error,
                absolute_error_delta=np.abs(prop_error) - np.abs(base_error),
                squared_error_delta=prop_error**2 - base_error**2,
                critical_zone=critical,
                baseline_late=base_late,
                proposed_late=prop_late,
                baseline_severe_late_10=critical & (base_error > 10.0),
                proposed_severe_late_10=critical & (prop_error > 10.0),
                late_disagreement=classify_disagreement(prop_late, base_late),
                rul_band=pd.cut(
                    merged["true_rul"],
                    bins=[-np.inf, 30, 60, 90, np.inf],
                    labels=["0-30", "31-60", "61-90", "91+"],
                ).astype(str),
            )
            rows.append(merged)
    return pd.concat(rows, ignore_index=True)


def summarize_failure_cases(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (subset, band), group in frame.groupby(["subset", "rul_band"], observed=True):
        rows.append(
            {
                "subset": subset,
                "rul_band": band,
                "paired_engines_across_seeds": len(group),
                "absolute_error_delta_mean": group["absolute_error_delta"].mean(),
                "absolute_error_delta_median": group["absolute_error_delta"].median(),
                "proposed_lower_absolute_error_share": np.mean(group["absolute_error_delta"] < 0.0),
                "proposed_safer_share": np.mean(group["late_disagreement"] == "proposed_safer"),
                "proposed_later_share": np.mean(group["late_disagreement"] == "proposed_later"),
                "both_late_share": np.mean(group["late_disagreement"] == "both_late"),
            }
        )
    return pd.DataFrame(rows)


def plot_fd004_cases(frame: pd.DataFrame, output: Path) -> None:
    data = frame[frame["subset"] == "FD004"].copy()
    palette = {
        "proposed_safer": "#2a9d8f",
        "proposed_later": "#d1495b",
        "both_late": "#f4a261",
        "neither_late": "#6c757d",
    }
    fig, axes = plt.subplots(1, 2, figsize=(11.6, 4.8))
    sns.scatterplot(
        data=data,
        x="true_rul",
        y="absolute_error_delta",
        hue="late_disagreement",
        palette=palette,
        s=27,
        alpha=0.65,
        ax=axes[0],
    )
    axes[0].axhline(0.0, color="#333333", linewidth=1.0, linestyle="--")
    axes[0].set_xlabel("True RUL (cycles)")
    axes[0].set_ylabel("Absolute-error change: OCM minus RAST-GRU")
    axes[0].set_title("FD004 paired error changes")
    axes[0].legend(title="Late-risk outcome", frameon=False, fontsize=8)

    counts = (
        data[data["critical_zone"]]
        .groupby(["seed", "late_disagreement"], observed=True)
        .size()
        .reset_index(name="engines")
    )
    sns.barplot(data=counts, x="late_disagreement", y="engines", hue="late_disagreement", palette=palette, ax=axes[1], errorbar="sd", legend=False)
    axes[1].set_xlabel("")
    axes[1].set_ylabel("Critical engines per seed")
    axes[1].set_title("FD004 paired late-indicator disagreements")
    axes[1].tick_params(axis="x", rotation=20)
    for ax in axes:
        ax.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze paired OCM-MST-GRU versus RAST-GRU failure cases.")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--baseline", default="rast_gru")
    args = parser.parse_args()

    grid = load_prediction_grid(Path(args.results_dir))
    cases = build_failure_cases(grid, baseline=args.baseline)
    summary = summarize_failure_cases(cases)
    extremes = (
        cases.sort_values("absolute_error_delta", ascending=False)
        .groupby(["subset", "seed"], as_index=False)
        .head(10)
        .reset_index(drop=True)
    )
    output = PROJECT_ROOT / "paper_outputs" / "advanced_evidence"
    figure_dir = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    cases.to_csv(output / "failure_case_engine_level.csv", index=False)
    summary.to_csv(output / "failure_case_summary.csv", index=False)
    extremes.to_csv(output / "failure_case_largest_regressions.csv", index=False)
    plot_fd004_cases(cases, figure_dir / "fig_fd004_failure_cases")
    print(f"FAILURE_CASE_ANALYSIS_READY {output}")


if __name__ == "__main__":
    main()
