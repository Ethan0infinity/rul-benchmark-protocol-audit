from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mpl_cache"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from rul.cmapss import SENSORS, load_subset
from rul.config import load_config
from rul.features import select_features
from rul.preprocessing import fit_transform_train_val_test, split_units

plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9.5})


def condition_separation(x: np.ndarray, labels: np.ndarray) -> tuple[float, float]:
    global_mean = np.mean(x, axis=0)
    between = []
    within = []
    for label in np.unique(labels):
        group = x[labels == label]
        center = np.mean(group, axis=0)
        between.append(float(np.mean((center - global_mean) ** 2)))
        within.append(float(np.mean((group - center) ** 2)))
    return float(np.mean(between)), float(np.mean(within))


def grouped_condition_prediction(
    x: np.ndarray, labels: np.ndarray, groups: np.ndarray, *, folds: int, seed: int
) -> pd.DataFrame:
    unique_groups = np.unique(groups)
    splitter = GroupKFold(n_splits=min(int(folds), len(unique_groups)))
    rows = []
    for fold, (train_index, test_index) in enumerate(splitter.split(x, labels, groups), start=1):
        classifier = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=500, class_weight="balanced", random_state=seed),
        )
        classifier.fit(x[train_index], labels[train_index])
        prediction = classifier.predict(x[test_index])
        rows.append(
            {
                "fold": fold,
                "accuracy": accuracy_score(labels[test_index], prediction),
                "balanced_accuracy": balanced_accuracy_score(labels[test_index], prediction),
                "macro_f1": f1_score(labels[test_index], prediction, average="macro", zero_division=0),
                "test_rows": len(test_index),
                "test_engines": len(np.unique(groups[test_index])),
            }
        )
    return pd.DataFrame(rows)


def grouped_rul_regression(
    x: np.ndarray, target: np.ndarray, groups: np.ndarray, *, folds: int
) -> pd.DataFrame:
    splitter = GroupKFold(n_splits=min(int(folds), len(np.unique(groups))))
    rows = []
    for fold, (train_index, test_index) in enumerate(splitter.split(x, target, groups), start=1):
        regressor = make_pipeline(StandardScaler(), Ridge(alpha=1.0))
        regressor.fit(x[train_index], target[train_index])
        prediction = regressor.predict(x[test_index])
        rows.append(
            {
                "fold": fold,
                "r2": r2_score(target[test_index], prediction),
                "mae": mean_absolute_error(target[test_index], prediction),
                "test_rows": len(test_index),
                "test_engines": len(np.unique(groups[test_index])),
            }
        )
    return pd.DataFrame(rows)


def plot_pca(before: np.ndarray, after: np.ndarray, labels: np.ndarray, output: Path) -> None:
    shared_pca = PCA(n_components=2, random_state=42).fit(np.vstack([before, after]))
    embeddings = (shared_pca.transform(before), shared_pca.transform(after))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.7), sharex=False, sharey=False)
    for ax, embedding, title in zip(
        axes,
        embeddings,
        ("(a) Global standardization", "(b) Condition-aware standardization"),
    ):
        scatter = ax.scatter(embedding[:, 0], embedding[:, 1], c=labels, cmap="tab10", s=5, alpha=0.28, rasterized=True)
        ax.set_title(title, loc="left", fontsize=10)
        ax.set_xlabel("PC1")
        ax.set_ylabel("PC2")
        ax.grid(alpha=0.18)
    handles, legend_labels = scatter.legend_elements()
    fig.legend(
        handles,
        legend_labels,
        title="Operating regime",
        loc="lower center",
        bbox_to_anchor=(0.5, -0.01),
        ncol=6,
        frameon=False,
        fontsize=8.5,
        title_fontsize=9,
    )
    fig.tight_layout(rect=(0.0, 0.14, 1.0, 1.0))
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Diagnose condition-aware normalization without using validation/test fitting.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "paper_main.yaml"))
    parser.add_argument("--subset", default="FD004")
    parser.add_argument("--sample-rows", type=int, default=50000)
    parser.add_argument("--folds", type=int, default=5)
    args = parser.parse_args()

    cfg = load_config(args.config)
    seed = int(cfg["project"].get("seed", 42))
    data_root = Path(cfg["data"].get("root", "data/raw"))
    if not data_root.is_absolute():
        data_root = PROJECT_ROOT / data_root
    train_df, test_df = load_subset(data_root, args.subset, cfg["data"].get("rul_cap", 125))
    train_units, val_units = split_units(
        train_df["unit_id"].to_numpy(), float(cfg["data"].get("validation_split", 0.2)), seed
    )
    train_core = train_df[train_df["unit_id"].isin(train_units)].copy()
    validation = train_df[train_df["unit_id"].isin(val_units)].copy()
    selected = select_features(
        train_core,
        feature_mode=cfg["data"].get("feature_mode", "selected"),
        max_selected_sensors=int(cfg["data"].get("max_selected_sensors", 14)),
        include_settings=bool(cfg["data"].get("include_settings", True)),
    )
    train_scaled, _, _, scaler = fit_transform_train_val_test(
        train_core,
        validation,
        test_df,
        selected.features,
        scaler_name=cfg["data"].get("scaler", "condition_standard"),
        subset=args.subset,
        condition_clusters=cfg["data"].get("condition_clusters"),
        random_state=seed,
    )
    sensor_features = [feature for feature in selected.features if feature in SENSORS]
    labels = scaler.predict_condition(train_core)
    raw = train_core[sensor_features].to_numpy(float)
    normalized = train_scaled[sensor_features].to_numpy(float)
    groups = train_core["unit_id"].to_numpy(int)

    rng = np.random.default_rng(seed)
    count = min(int(args.sample_rows), len(train_core))
    index = np.sort(rng.choice(len(train_core), size=count, replace=False))
    raw_sample = StandardScaler().fit_transform(raw[index])
    normalized_sample = normalized[index]
    labels_sample = labels[index]
    groups_sample = groups[index]
    degradation_stage_sample = (train_core["rul"].to_numpy(float)[index] <= 30.0).astype(int)
    rul_sample = train_core["rul"].to_numpy(float)[index]

    before_cv = grouped_condition_prediction(raw_sample, labels_sample, groups_sample, folds=args.folds, seed=seed)
    before_cv["representation"] = "global_standardized_raw_sensors"
    after_cv = grouped_condition_prediction(normalized_sample, labels_sample, groups_sample, folds=args.folds, seed=seed)
    after_cv["representation"] = "condition_aware_normalized_sensors"
    cv = pd.concat([before_cv, after_cv], ignore_index=True)

    degradation_before = grouped_condition_prediction(
        raw_sample, degradation_stage_sample, groups_sample, folds=args.folds, seed=seed
    )
    degradation_before["representation"] = "global_standardized_raw_sensors"
    degradation_after = grouped_condition_prediction(
        normalized_sample, degradation_stage_sample, groups_sample, folds=args.folds, seed=seed
    )
    degradation_after["representation"] = "condition_aware_normalized_sensors"
    degradation_cv = pd.concat([degradation_before, degradation_after], ignore_index=True)
    degradation_cv["target"] = "critical_stage_true_rul_le_30"
    rul_before = grouped_rul_regression(raw_sample, rul_sample, groups_sample, folds=args.folds)
    rul_before["representation"] = "global_standardized_raw_sensors"
    rul_after = grouped_rul_regression(normalized_sample, rul_sample, groups_sample, folds=args.folds)
    rul_after["representation"] = "condition_aware_normalized_sensors"
    rul_cv = pd.concat([rul_before, rul_after], ignore_index=True)

    diagnostics = []
    for name, values in (
        ("global_standardized_raw_sensors", raw_sample),
        ("condition_aware_normalized_sensors", normalized_sample),
    ):
        between, within = condition_separation(values, labels_sample)
        subset_cv = cv[cv["representation"] == name]
        diagnostics.append(
            {
                "representation": name,
                "sample_rows": count,
                "training_engines": len(np.unique(groups_sample)),
                "conditions": len(np.unique(labels_sample)),
                "between_condition_centroid_variance": between,
                "within_condition_variance": within,
                "between_to_within_ratio": between / max(within, 1e-12),
                "grouped_cv_accuracy_mean": subset_cv["accuracy"].mean(),
                "grouped_cv_balanced_accuracy_mean": subset_cv["balanced_accuracy"].mean(),
                "grouped_cv_macro_f1_mean": subset_cv["macro_f1"].mean(),
            }
        )

    output = PROJECT_ROOT / "paper_outputs" / "advanced_evidence"
    figure_dir = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(diagnostics).to_csv(output / "condition_normalization_diagnostics.csv", index=False)
    cv.to_csv(output / "condition_normalization_grouped_cv.csv", index=False)
    degradation_cv.to_csv(output / "degradation_stage_grouped_cv.csv", index=False)
    degradation_summary = degradation_cv.groupby("representation", as_index=False).agg(
        folds=("fold", "count"),
        balanced_accuracy_mean=("balanced_accuracy", "mean"),
        balanced_accuracy_std=("balanced_accuracy", lambda x: x.std(ddof=1)),
        macro_f1_mean=("macro_f1", "mean"),
        macro_f1_std=("macro_f1", lambda x: x.std(ddof=1)),
    )
    degradation_summary.to_csv(output / "degradation_stage_separability_summary.csv", index=False)
    rul_cv.to_csv(output / "degradation_rul_grouped_cv.csv", index=False)
    rul_summary = rul_cv.groupby("representation", as_index=False).agg(
        folds=("fold", "count"),
        r2_mean=("r2", "mean"),
        r2_std=("r2", lambda x: x.std(ddof=1)),
        mae_mean=("mae", "mean"),
        mae_std=("mae", lambda x: x.std(ddof=1)),
    )
    rul_summary.to_csv(output / "degradation_rul_separability_summary.csv", index=False)
    plot_pca(raw_sample, normalized_sample, labels_sample, figure_dir / "fig_condition_normalization_pca")
    metadata = {
        "protocol_version": "3.0",
        "subset": args.subset,
        "fit_scope": "training-core engines only",
        "seed": seed,
        "selected_sensors": sensor_features,
        "sample_rows": count,
        "interpretation_boundary": "Descriptive mechanism diagnostic; not causal evidence of improved prediction.",
        "degradation_stage_target": "true training-core RUL <= 30 cycles; grouped by engine",
        "continuous_degradation_target": "capped training-core RUL; grouped by engine Ridge probe",
        "pca_basis": "single PCA basis fitted to the vertically concatenated before/after diagnostic samples",
    }
    (output / "condition_normalization_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"CONDITION_NORMALIZATION_ANALYSIS_READY {output}")


if __name__ == "__main__":
    main()
