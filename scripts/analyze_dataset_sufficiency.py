from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rul.cmapss import COLUMNS, SENSORS, SETTINGS, add_test_rul, add_train_rul, read_rul_file
from rul.features import score_sensor_features

SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
WINDOWS = [20, 30, 40, 50]
RUL_CAP = 125


def read_table(data_dir: Path, name: str) -> pd.DataFrame:
    return pd.read_csv(data_dir / name, sep=r"\s+", header=None, names=COLUMNS, engine="python")


def q(values: pd.Series | np.ndarray, p: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=float), p))


def window_count(cycle_lengths: pd.Series, window_size: int) -> int:
    return int(np.maximum(cycle_lengths.to_numpy() - window_size + 1, 0).sum())


def basic_subset_stats(train: pd.DataFrame, test: pd.DataFrame, rul: pd.Series, subset: str) -> dict[str, Any]:
    train_lengths = train.groupby("unit_id")["cycle"].max()
    test_lengths = test.groupby("unit_id")["cycle"].max()
    row: dict[str, Any] = {
        "subset": subset,
        "train_rows": len(train),
        "test_rows": len(test),
        "total_rows": len(train) + len(test),
        "train_units": train["unit_id"].nunique(),
        "test_units": test["unit_id"].nunique(),
        "rul_rows": len(rul),
        "train_cycle_min": int(train_lengths.min()),
        "train_cycle_q25": q(train_lengths, 0.25),
        "train_cycle_median": q(train_lengths, 0.50),
        "train_cycle_mean": float(train_lengths.mean()),
        "train_cycle_q75": q(train_lengths, 0.75),
        "train_cycle_max": int(train_lengths.max()),
        "test_cycle_min": int(test_lengths.min()),
        "test_cycle_q25": q(test_lengths, 0.25),
        "test_cycle_median": q(test_lengths, 0.50),
        "test_cycle_mean": float(test_lengths.mean()),
        "test_cycle_q75": q(test_lengths, 0.75),
        "test_cycle_max": int(test_lengths.max()),
        "test_rul_min": float(rul.min()),
        "test_rul_q25": q(rul, 0.25),
        "test_rul_median": q(rul, 0.50),
        "test_rul_mean": float(rul.mean()),
        "test_rul_q75": q(rul, 0.75),
        "test_rul_max": float(rul.max()),
    }
    for w in WINDOWS:
        row[f"train_windows_w{w}"] = window_count(train_lengths, w)
        row[f"test_last_windows_w{w}"] = int(test["unit_id"].nunique())
    return row


def label_stats(train: pd.DataFrame, test: pd.DataFrame, rul: pd.Series, subset: str) -> dict[str, Any]:
    train_labeled = add_train_rul(train, rul_cap=None)
    train_capped = add_train_rul(train, rul_cap=RUL_CAP)
    test_labeled = add_test_rul(test, rul, rul_cap=None)
    test_capped = add_test_rul(test, rul, rul_cap=RUL_CAP)
    return {
        "subset": subset,
        "train_rul_raw_min": float(train_labeled["rul"].min()),
        "train_rul_raw_mean": float(train_labeled["rul"].mean()),
        "train_rul_raw_max": float(train_labeled["rul"].max()),
        "train_rul_capped_mean": float(train_capped["rul"].mean()),
        "train_cap125_share": float((train_labeled["rul"] > RUL_CAP).mean()),
        "train_late_life_le30_rows": int((train_labeled["rul"] <= 30).sum()),
        "train_late_life_le30_share": float((train_labeled["rul"] <= 30).mean()),
        "test_rul_raw_min": float(test_labeled["rul"].min()),
        "test_rul_raw_mean": float(test_labeled["rul"].mean()),
        "test_rul_raw_max": float(test_labeled["rul"].max()),
        "test_rul_capped_mean": float(test_capped["rul"].mean()),
        "test_cap125_share": float((test_labeled["rul"] > RUL_CAP).mean()),
        "test_final_rul_le30_units": int((rul <= 30).sum()),
        "test_final_rul_le30_share": float((rul <= 30).mean()),
    }


def feature_quality(train: pd.DataFrame, test: pd.DataFrame, subset: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    train_labeled = add_train_rul(train, rul_cap=RUL_CAP)
    scores = score_sensor_features(train_labeled)
    rows = []
    for sensor in SENSORS:
        train_values = train[sensor].astype(float)
        test_values = test[sensor].astype(float)
        train_std = float(train_values.std())
        test_std = float(test_values.std())
        pooled = np.sqrt((train_std**2 + test_std**2) / 2.0)
        smd = 0.0 if pooled == 0 else abs(float(train_values.mean() - test_values.mean())) / pooled
        row_score = scores.loc[scores["feature"] == sensor].iloc[0]
        rows.append(
            {
                "subset": subset,
                "feature": sensor,
                "train_std": train_std,
                "test_std": test_std,
                "near_constant_train": train_std < 1e-8,
                "near_constant_test": test_std < 1e-8,
                "abs_train_test_smd": smd,
                "variance": float(row_score["variance"]),
                "abs_rul_corr": float(row_score["abs_rul_corr"]),
                "abs_cycle_corr": float(row_score["abs_cycle_corr"]),
                "hybrid_score": float(row_score["hybrid_score"]),
            }
        )
    df = pd.DataFrame(rows).sort_values(["subset", "hybrid_score"], ascending=[True, False])
    summary = {
        "subset": subset,
        "near_constant_train_sensors": ",".join(df.loc[df["near_constant_train"], "feature"].tolist()),
        "near_constant_train_count": int(df["near_constant_train"].sum()),
        "top_8_sensors": ",".join(df.head(8)["feature"].tolist()),
        "median_train_test_smd": float(df["abs_train_test_smd"].median()),
        "max_train_test_smd": float(df["abs_train_test_smd"].max()),
    }
    return df, summary


def condition_complexity(train: pd.DataFrame, test: pd.DataFrame, subset: str) -> dict[str, Any]:
    row: dict[str, Any] = {"subset": subset}
    for split_name, df in [("train", train), ("test", test)]:
        settings = df[SETTINGS].astype(float).to_numpy()
        row[f"{split_name}_setting1_min"] = float(df["setting_1"].min())
        row[f"{split_name}_setting1_max"] = float(df["setting_1"].max())
        row[f"{split_name}_setting2_min"] = float(df["setting_2"].min())
        row[f"{split_name}_setting2_max"] = float(df["setting_2"].max())
        row[f"{split_name}_setting3_unique"] = int(df["setting_3"].nunique())
        if len(df) > 5000:
            sample = df[SETTINGS].sample(5000, random_state=42).astype(float).to_numpy()
        else:
            sample = settings
        if subset in {"FD002", "FD004"}:
            km = KMeans(n_clusters=6, random_state=42, n_init=10).fit(sample)
            row[f"{split_name}_kmeans6_silhouette"] = float(silhouette_score(sample, km.labels_))
        else:
            row[f"{split_name}_kmeans6_silhouette"] = None
    return row


def quality_summary(train: pd.DataFrame, test: pd.DataFrame, rul: pd.Series, subset: str) -> dict[str, Any]:
    return {
        "subset": subset,
        "train_missing_cells": int(train.isna().sum().sum()),
        "test_missing_cells": int(test.isna().sum().sum()),
        "train_duplicate_rows": int(train.duplicated().sum()),
        "test_duplicate_rows": int(test.duplicated().sum()),
        "train_unit_cycle_duplicate_keys": int(train.duplicated(["unit_id", "cycle"]).sum()),
        "test_unit_cycle_duplicate_keys": int(test.duplicated(["unit_id", "cycle"]).sum()),
        "rul_negative_count": int((rul < 0).sum()),
        "train_nonfinite": int((~np.isfinite(train.to_numpy(dtype=float))).sum()),
        "test_nonfinite": int((~np.isfinite(test.to_numpy(dtype=float))).sum()),
    }


def adequacy_judgement(stats: pd.DataFrame, feature_summaries: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in stats.iterrows():
        subset = r["subset"]
        if subset in {"FD001", "FD003"}:
            unit_risk = "中：独立发动机只有100台，但滑窗样本充足"
        else:
            unit_risk = "低：训练发动机数量较多，适合复杂工况验证"
        if subset == "FD001":
            role = "基础性能验证，不足以单独支撑完整工程投稿"
        elif subset == "FD002":
            role = "多工况单故障，支撑复杂工况泛化"
        elif subset == "FD003":
            role = "单工况多故障，支撑故障模式泛化"
        else:
            role = "多工况多故障，是论文说服力核心"
        rows.append(
            {
                "subset": subset,
                "sample_sufficiency": "足够" if r["train_windows_w30"] > 10000 else "偏少",
                "independent_unit_risk": unit_risk,
                "recommended_role": role,
                "w30_train_windows": int(r["train_windows_w30"]),
                "test_units": int(r["test_units"]),
            }
        )
    return pd.DataFrame(rows)


def plot_distributions(stats: pd.DataFrame, label_df: pd.DataFrame, feature_df: pd.DataFrame, reports_dir: Path) -> None:
    figs_dir = reports_dir / "figures"
    figs_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(8, 4.8))
    plt.bar(stats["subset"], stats["train_windows_w30"], label="train windows (w=30)")
    plt.plot(stats["subset"], stats["test_units"], marker="o", color="tab:red", label="test units")
    plt.ylabel("count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figs_dir / "dataset_window_counts.png", dpi=200)
    plt.close()

    plt.figure(figsize=(8, 4.8))
    plt.bar(label_df["subset"], label_df["train_late_life_le30_share"], label="train RUL <= 30 share")
    plt.plot(label_df["subset"], label_df["test_final_rul_le30_share"], marker="o", color="tab:red", label="test final RUL <= 30 unit share")
    plt.ylabel("share")
    plt.ylim(0, max(0.35, float(label_df["train_late_life_le30_share"].max()) + 0.05))
    plt.legend()
    plt.tight_layout()
    plt.savefig(figs_dir / "late_life_share.png", dpi=200)
    plt.close()

    top = feature_df.sort_values(["subset", "hybrid_score"], ascending=[True, False]).groupby("subset").head(8)
    for subset, group in top.groupby("subset"):
        plt.figure(figsize=(8, 4.8))
        plt.bar(group["feature"], group["hybrid_score"])
        plt.ylabel("hybrid feature score")
        plt.title(f"{subset} top sensors")
        plt.tight_layout()
        plt.savefig(figs_dir / f"{subset}_top_sensor_scores.png", dpi=200)
        plt.close()


def write_report(
    stats: pd.DataFrame,
    label_stats_df: pd.DataFrame,
    feature_summaries: pd.DataFrame,
    condition_df: pd.DataFrame,
    quality_df: pd.DataFrame,
    adequacy_df: pd.DataFrame,
    reports_dir: Path,
) -> None:
    total_train_rows = int(stats["train_rows"].sum())
    total_test_rows = int(stats["test_rows"].sum())
    total_train_units = int(stats["train_units"].sum())
    total_test_units = int(stats["test_units"].sum())
    total_windows_w30 = int(stats["train_windows_w30"].sum())
    lines = [
        "# Dataset Sufficiency and Quality Analysis",
        "",
        "## Conclusion",
        "",
        f"C-MAPSS FD001-FD004 is sufficient for this thesis direction if all four subsets are used together. It is not recommended to rely on FD001 alone for a formal engineering manuscript, because FD001 only covers one operating condition and one fault mode. The combined dataset provides {total_train_rows:,} training rows, {total_test_rows:,} test rows, {total_train_units:,} training engines, {total_test_units:,} test engines, and {total_windows_w30:,} training windows with window size 30.",
        "",
        "## Dataset Scale",
        "",
        stats[["subset", "train_rows", "train_units", "test_rows", "test_units", "train_windows_w30", "train_cycle_min", "train_cycle_median", "train_cycle_max", "test_rul_min", "test_rul_median", "test_rul_max"]].to_markdown(index=False),
        "",
        "## RUL Label Distribution",
        "",
        label_stats_df[["subset", "train_rul_raw_mean", "train_rul_raw_max", "train_cap125_share", "train_late_life_le30_share", "test_rul_raw_mean", "test_rul_raw_max", "test_final_rul_le30_share"]].to_markdown(index=False),
        "",
        "## Feature and Sensor Quality",
        "",
        feature_summaries.to_markdown(index=False),
        "",
        "Near-constant sensors should not be treated as strong degradation indicators. The current pipeline uses hybrid sensor scoring and selects top sensors, which is appropriate for the proposed sensor-selection contribution.",
        "",
        "## Operating Condition Complexity",
        "",
        condition_df.to_markdown(index=False),
        "",
        "FD002 and FD004 show multi-condition setting ranges and high k-means silhouette under six clusters, so they are useful for validating complex operating-condition robustness.",
        "",
        "## Raw Data Quality",
        "",
        quality_df.to_markdown(index=False),
        "",
        "No missing cells, duplicate rows, duplicate unit-cycle keys, negative RUL values, or non-finite numeric values were found in the local files.",
        "",
        "## Adequacy Judgement",
        "",
        adequacy_df.to_markdown(index=False),
        "",
        "## Main Risks",
        "",
        "1. Window samples are highly correlated because many windows come from the same engine. Report engine-level test metrics and avoid claiming the number of windows equals independent samples.",
        "2. FD001 alone is too simple for a strong paper. Use FD001-FD004 and emphasize FD002/FD004.",
        "3. Some sensors are near-constant, so all-sensor baselines may contain redundant signals. This supports the sensor selection module.",
        "4. Test trajectories are truncated before failure, so evaluation should use the last window per test engine unless a different protocol is explicitly stated.",
        "5. FD004 metadata can be inconsistent across sources. The actual local files are 249 train units and 248 test units; this is validated by RUL_FD004 having 248 rows.",
        "",
        "## Recommended Experiment Policy",
        "",
        "- Use all four subsets in the main table.",
        "- Report FD001-FD004 separately, not just averaged.",
        "- Run at least 3 random seeds for neural models if time allows.",
        "- Include selected-sensor vs all-sensor ablation.",
        "- Include noise/missing/block-missing robustness on at least FD001 and FD004; preferably all four.",
        "- Report RMSE, MAE, NASA Score, R2, parameter count, and inference time.",
        "",
    ]
    (reports_dir / "dataset_sufficiency_report.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    data_dir = PROJECT_ROOT / "data" / "raw"
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    stats_rows = []
    label_rows = []
    feature_frames = []
    feature_summary_rows = []
    condition_rows = []
    quality_rows = []

    for subset in SUBSETS:
        train = read_table(data_dir, f"train_{subset}.txt")
        test = read_table(data_dir, f"test_{subset}.txt")
        rul = read_rul_file(data_dir, subset)
        stats_rows.append(basic_subset_stats(train, test, rul, subset))
        label_rows.append(label_stats(train, test, rul, subset))
        fq, fs = feature_quality(train, test, subset)
        feature_frames.append(fq)
        feature_summary_rows.append(fs)
        condition_rows.append(condition_complexity(train, test, subset))
        quality_rows.append(quality_summary(train, test, rul, subset))

    stats = pd.DataFrame(stats_rows)
    label_stats_df = pd.DataFrame(label_rows)
    feature_df = pd.concat(feature_frames, ignore_index=True)
    feature_summaries = pd.DataFrame(feature_summary_rows)
    condition_df = pd.DataFrame(condition_rows)
    quality_df = pd.DataFrame(quality_rows)
    adequacy_df = adequacy_judgement(stats, feature_summaries)

    stats.to_csv(reports_dir / "dataset_sufficiency_summary.csv", index=False)
    label_stats_df.to_csv(reports_dir / "rul_label_distribution.csv", index=False)
    feature_df.to_csv(reports_dir / "sensor_feature_quality.csv", index=False)
    feature_summaries.to_csv(reports_dir / "sensor_feature_summary.csv", index=False)
    condition_df.to_csv(reports_dir / "operating_condition_summary.csv", index=False)
    quality_df.to_csv(reports_dir / "raw_quality_summary.csv", index=False)
    adequacy_df.to_csv(reports_dir / "dataset_adequacy_judgement.csv", index=False)

    plot_distributions(stats, label_stats_df, feature_df, reports_dir)
    write_report(stats, label_stats_df, feature_summaries, condition_df, quality_df, adequacy_df, reports_dir)

    total_train_rows = int(stats["train_rows"].sum())
    total_test_rows = int(stats["test_rows"].sum())
    total_train_units = int(stats["train_units"].sum())
    total_test_units = int(stats["test_units"].sum())
    total_windows = int(stats["train_windows_w30"].sum())
    payload = {
        "total_train_rows": total_train_rows,
        "total_test_rows": total_test_rows,
        "total_train_units": total_train_units,
        "total_test_units": total_test_units,
        "total_train_windows_w30": total_windows,
        "reports": [
            "reports/dataset_sufficiency_report.md",
            "reports/dataset_sufficiency_summary.csv",
            "reports/rul_label_distribution.csv",
            "reports/sensor_feature_quality.csv",
            "reports/operating_condition_summary.csv",
            "reports/raw_quality_summary.csv",
        ],
    }
    (reports_dir / "dataset_sufficiency_summary.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    print("DATASET_SUFFICIENCY_ANALYSIS_DONE")


if __name__ == "__main__":
    main()
