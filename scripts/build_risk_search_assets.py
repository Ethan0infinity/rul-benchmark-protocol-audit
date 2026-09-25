from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "paper_outputs" / "advanced_evidence"


def main() -> None:
    source = PROJECT_ROOT / "reports" / "risk_hyperparameter_search.csv"
    frame = pd.read_csv(source).sort_values("rank")
    OUTPUT.mkdir(parents=True, exist_ok=True)
    frame.to_csv(OUTPUT / "risk_hyperparameter_search.csv", index=False)

    pivot = frame.pivot(
        index="late_life_weight",
        columns="late_over_weight",
        values="best_val_risk_score",
    )
    fig, ax = plt.subplots(figsize=(7.2, 5.5))
    sns.heatmap(pivot, annot=True, fmt=".4f", cmap="viridis_r", linewidths=0.6, ax=ax)
    ax.set_xlabel("Late-overestimation weight")
    ax.set_ylabel("Late-life weight")
    ax.set_title("FD004 validation-only risk-weight search")
    fig.tight_layout()
    figure_dir = OUTPUT / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(figure_dir / "fig_risk_weight_search.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "fig_risk_weight_search.png", dpi=400, bbox_inches="tight")
    plt.close(fig)

    best = frame.iloc[0]
    second = frame.iloc[1]
    summary = {
        "selected_late_life_weight": float(best["late_life_weight"]),
        "selected_late_over_weight": float(best["late_over_weight"]),
        "selected_validation_risk_score": float(best["best_val_risk_score"]),
        "runner_up_validation_risk_score": float(second["best_val_risk_score"]),
        "absolute_gap_to_runner_up": float(second["best_val_risk_score"] - best["best_val_risk_score"]),
        "relative_gap_to_runner_up": float(
            (second["best_val_risk_score"] - best["best_val_risk_score"]) / best["best_val_risk_score"]
        ),
        "selection_uses_test_metrics": False,
    }
    (OUTPUT / "risk_hyperparameter_search_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    print(
        "RISK_SEARCH_ASSETS_READY "
        f"selected={summary['selected_late_life_weight']}/{summary['selected_late_over_weight']} "
        f"gap={summary['absolute_gap_to_runner_up']:.6f}"
    )


if __name__ == "__main__":
    main()
