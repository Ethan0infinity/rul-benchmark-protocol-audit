from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wasserstein_distance


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rul.config import load_config
from rul.training import prepare_data


SEEDS = (42, 123, 2024, 2025, 2026)
SUBSETS = ("FD002", "FD004")
STRATEGIES = ("uniform_single", "fixed_multi", "critical_stratified")


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit validation endpoint target distributions against test endpoints.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "paper_main.yaml"))
    args = parser.parse_args()

    base = load_config(args.config)
    target_rows: list[dict] = []
    test_targets: dict[str, np.ndarray] = {}
    for subset in SUBSETS:
        for seed in SEEDS:
            cfg = copy.deepcopy(base)
            cfg["project"]["seed"] = seed
            cfg["data"]["validation_endpoint_confirmation"] = True
            prepared = prepare_data(cfg, subset=subset)
            if seed == SEEDS[0]:
                test_targets[subset] = prepared.test.y.astype(float)
            for strategy in STRATEGIES:
                endpoint_data = prepared.validation_endpoint_sets[strategy]
                for unit_id, target in zip(endpoint_data.unit_ids, endpoint_data.y):
                    target_rows.append(
                        {
                            "subset": subset,
                            "seed": seed,
                            "distribution": strategy,
                            "unit_id": int(unit_id),
                            "rul": float(target),
                        }
                    )
        for unit_id, target in zip(prepared.test.unit_ids, test_targets[subset]):
            target_rows.append(
                {
                    "subset": subset,
                    "seed": -1,
                    "distribution": "test_endpoint",
                    "unit_id": int(unit_id),
                    "rul": float(target),
                }
            )

    output = PROJECT_ROOT / "paper_outputs" / "advanced_evidence"
    figure_dir = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    targets = pd.DataFrame(target_rows)
    targets.to_csv(output / "validation_endpoint_distribution_targets.csv", index=False)

    cluster_rows: list[dict] = []
    for subset in SUBSETS:
        for strategy in STRATEGIES:
            selected = targets[
                (targets["subset"] == subset)
                & (targets["distribution"] == strategy)
            ]
            counts = selected.groupby(["seed", "unit_id"], as_index=False).size()["size"].to_numpy(float)
            cluster_rows.append(
                {
                    "subset": subset,
                    "endpoint_strategy": strategy,
                    "validation_engine_seed_clusters": int(len(counts)),
                    "validation_endpoints_per_engine_min": int(np.min(counts)),
                    "validation_endpoints_per_engine_median": float(np.median(counts)),
                    "validation_endpoints_per_engine_mean": float(np.mean(counts)),
                    "validation_endpoints_per_engine_max": int(np.max(counts)),
                    "validation_cluster_id": "seed + unit_id",
                    "validation_endpoint_role": "checkpoint selection only",
                    "test_endpoint_count": int(len(test_targets[subset])),
                    "test_endpoints_per_engine": 1,
                    "inferential_bootstrap_primary_unit": "matched test engine",
                }
            )
    pd.DataFrame(cluster_rows).to_csv(output / "validation_endpoint_cluster_audit.csv", index=False)

    summary_rows: list[dict] = []
    for subset in SUBSETS:
        test = test_targets[subset]
        for strategy in STRATEGIES:
            values = targets[(targets["subset"] == subset) & (targets["distribution"] == strategy)]["rul"].to_numpy(float)
            summary_rows.append(
                {
                    "subset": subset,
                    "endpoint_strategy": strategy,
                    "endpoint_count": len(values),
                    "critical_30_fraction": float(np.mean(values <= 30.0)),
                    "mean_rul": float(np.mean(values)),
                    "median_rul": float(np.median(values)),
                    "wasserstein_distance_to_test": float(wasserstein_distance(values, test)),
                    "test_endpoint_count": len(test),
                    "test_critical_30_fraction": float(np.mean(test <= 30.0)),
                    "test_mean_rul": float(np.mean(test)),
                }
            )
    pd.DataFrame(summary_rows).to_csv(output / "validation_endpoint_distribution_summary.csv", index=False)

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "font.size": 7,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "legend.frameon": False,
        }
    )
    colors = {
        "uniform_single": "#667085",
        "fixed_multi": "#2F6690",
        "critical_stratified": "#C65D2E",
        "test_endpoint": "#2F7D5C",
    }
    labels = {
        "uniform_single": "Uniform single",
        "fixed_multi": "Fixed multi",
        "critical_stratified": "Critical-stratified",
        "test_endpoint": "Test endpoint",
    }
    bins = np.arange(0, 131, 10)
    centers = (bins[:-1] + bins[1:]) / 2
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 2.75), sharey=True)
    for ax, subset in zip(axes, SUBSETS):
        for distribution in (*STRATEGIES, "test_endpoint"):
            values = targets[(targets["subset"] == subset) & (targets["distribution"] == distribution)]["rul"].to_numpy(float)
            counts, _ = np.histogram(values, bins=bins)
            proportions = counts / max(counts.sum(), 1)
            ax.plot(centers, proportions, marker="o", markersize=2.8, linewidth=1.2,
                    color=colors[distribution], label=labels[distribution])
        ax.axvspan(0, 30, color="#C65D2E", alpha=0.08, linewidth=0)
        ax.set_title(subset, fontsize=8, fontweight="bold")
        ax.set_xlabel("Endpoint RUL (cycles)")
        ax.set_xlim(0, 130)
        ax.grid(axis="y", color="0.88", linewidth=0.6)
    axes[0].set_ylabel("Proportion per 10-cycle bin")
    axes[1].legend(loc="upper right", fontsize=6.5)
    fig.tight_layout()
    stem = figure_dir / "fig_validation_endpoint_distribution"
    fig.savefig(stem.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight")
    fig.savefig(stem.with_suffix(".png"), dpi=600, bbox_inches="tight")
    fig.savefig(stem.with_suffix(".tiff"), dpi=600, bbox_inches="tight")
    plt.close(fig)
    print(f"VALIDATION_ENDPOINT_DISTRIBUTION_READY rows={len(targets)} output={output}")


if __name__ == "__main__":
    main()
