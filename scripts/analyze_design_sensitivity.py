from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, silhouette_score
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_advanced_evidence import prediction_metrics
from rul.cmapss import SETTINGS, load_subset
from rul.preprocessing import split_units
from run_formal_benchmark import FORMAL_BENCHMARK_SEEDS


K_VALUES = (4, 5, 6, 7, 8)
AUX_SCALES = (0.5, 1.0, 2.0)


def experiment_dir(root: Path, seed: int, family: str, value: float) -> Path:
    if (family == "cluster_k" and value == 6) or (family == "aux_scale" and value == 1):
        return root / f"paper_main_v3_seed{seed}" / "FD004" / "rast_gru_v2"
    label = f"k{int(value)}" if family == "cluster_k" else f"aux{str(value).replace('.', 'p')}"
    return root / f"design_sensitivity_{label}_fd004_seed{seed}" / "FD004" / "rast_gru_v2"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze model and training-only clustering sensitivity.")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "paper_outputs" / "advanced_evidence"))
    args = parser.parse_args()
    root = Path(args.results_dir)
    rows = []
    for seed in FORMAL_BENCHMARK_SEEDS:
        for family, values in (("cluster_k", K_VALUES), ("aux_scale", AUX_SCALES)):
            for value in values:
                directory = experiment_dir(root, seed, family, value)
                pred_path = directory / "test_predictions.csv"
                if not pred_path.exists():
                    raise FileNotFoundError(pred_path)
                metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8-sig"))
                rows.append({"seed": seed, "family": family, "value": value, "best_epoch": metrics["best_epoch"],
                             **prediction_metrics(pd.read_csv(pred_path))})
    long = pd.DataFrame(rows)
    summary = long.groupby(["family", "value"], as_index=False).agg(
        seed_count=("seed", "nunique"),
        **{f"{m}_mean": (m, "mean") for m in ("rmse", "nasa_per_engine", "lpr30", "mle30", "late_cvar95_30")},
        **{f"{m}_std": (m, lambda x: x.std(ddof=1)) for m in ("rmse", "nasa_per_engine", "lpr30", "mle30", "late_cvar95_30")},
    )

    train, _ = load_subset(PROJECT_ROOT / "data" / "raw", "FD004", 125)
    cluster_rows = []
    common_predictions: dict[tuple[int, int], np.ndarray] = {}
    for seed in FORMAL_BENCHMARK_SEEDS:
        train_units, _ = split_units(train.unit_id.to_numpy(), .2, seed)
        core = train[train.unit_id.isin(train_units)]
        setting_scaler = StandardScaler().fit(core[SETTINGS])
        settings = setting_scaler.transform(core[SETTINGS])
        common_settings = setting_scaler.transform(train[SETTINGS])
        _, declared_setting_labels = np.unique(
            core[SETTINGS].round(0).to_numpy(), axis=0, return_inverse=True
        )
        reference = KMeans(n_clusters=6, random_state=42, n_init=10).fit_predict(settings)
        sample_idx = np.linspace(0, len(settings) - 1, min(10000, len(settings)), dtype=int)
        for k in K_VALUES:
            estimator = KMeans(n_clusters=k, random_state=42, n_init=10).fit(settings)
            labels = estimator.labels_
            common_predictions[(seed, k)] = estimator.predict(common_settings)
            cluster_rows.append({"seed": seed, "k": k,
                                 "silhouette": float(silhouette_score(settings[sample_idx], labels[sample_idx])),
                                 "ari_vs_k6": float(adjusted_rand_score(reference, labels)),
                                 "ari_vs_declared_setting_combinations": float(
                                     adjusted_rand_score(declared_setting_labels, labels)
                                 ),
                                 "minimum_cluster_fraction": float(np.bincount(labels).min() / len(labels))})
    cluster = pd.DataFrame(cluster_rows)
    cluster_summary = cluster.groupby("k", as_index=False).agg(
        seed_count=("seed", "nunique"), silhouette_mean=("silhouette", "mean"),
        silhouette_std=("silhouette", lambda x: x.std(ddof=1)), ari_vs_k6_mean=("ari_vs_k6", "mean"),
        ari_vs_declared_setting_combinations_mean=("ari_vs_declared_setting_combinations", "mean"),
        minimum_cluster_fraction_mean=("minimum_cluster_fraction", "mean"))
    cross_rows = []
    for k in K_VALUES:
        for left_index, left_seed in enumerate(FORMAL_BENCHMARK_SEEDS):
            for right_seed in FORMAL_BENCHMARK_SEEDS[left_index + 1:]:
                cross_rows.append({"k": k, "left_seed": left_seed, "right_seed": right_seed,
                                   "cross_seed_ari": float(adjusted_rand_score(
                                       common_predictions[(left_seed, k)], common_predictions[(right_seed, k)]))})
    cross = pd.DataFrame(cross_rows)
    cross_summary = cross.groupby("k", as_index=False).agg(
        cross_seed_ari_mean=("cross_seed_ari", "mean"),
        cross_seed_ari_std=("cross_seed_ari", lambda x: x.std(ddof=1)))
    cluster_summary = cluster_summary.merge(cross_summary, on="k", validate="one_to_one")
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    long.to_csv(output / "hyperparameter_sensitivity_seed_metrics.csv", index=False)
    summary.to_csv(output / "hyperparameter_sensitivity_summary.csv", index=False)
    cluster.to_csv(output / "condition_cluster_k_diagnostics_seed.csv", index=False)
    cross.to_csv(output / "condition_cluster_k_cross_seed_stability.csv", index=False)
    cluster_summary.to_csv(output / "condition_cluster_k_diagnostics_summary.csv", index=False)
    print(f"DESIGN_SENSITIVITY_ANALYSIS_READY runs={len(long)} cluster_rows={len(cluster)}")


if __name__ == "__main__":
    main()
