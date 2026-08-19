from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


MODEL_DISPLAY_NAMES = {
    "gru": "GRU",
    "lstm": "LSTM",
    "cnn": "1D-CNN",
    "cnn1d": "1D-CNN",
    "cnn_lstm": "CNN-LSTM",
    "tcn": "TCN",
    "tcn_gru": "TCN-GRU",
    "attention_gru": "Attention-GRU",
    "bigru_attention": "BiGRU-Attention",
    "transformer_lite": "Transformer-lite",
    "dual_attention_tcn": "Dual-attention TCN",
    "sensor_graph_gru": "SensorGraph-GRU",
    "quantile_gru": "Quantile-GRU",
    "rs_tcn_gru": "RS-TCN-GRU",
    "rast_gru": "RAST-GRU",
    "rast_gru_v2": "OCM-MST-GRU",
}


MAIN_COLUMNS = [
    "experiment_name",
    "subset",
    "model",
    "model_display",
    "evidence_level",
    "created_time",
    "test_rmse",
    "test_mae",
    "test_r2",
    "test_nasa_score",
    "parameters",
    "batch_inference_ms",
    "single_sample_inference_ms",
    "cpu_single_sample_inference_ms",
    "run_dir",
]

CRITICAL_ZONE_COLUMNS = [
    "experiment_name",
    "subset",
    "model",
    "model_display",
    "evidence_level",
    "test_critical_30_count",
    "test_critical_30_rmse",
    "test_critical_30_mae",
    "test_critical_30_late_prediction_ratio",
    "test_critical_30_over_error_mean",
    "test_critical_30_under_error_mean",
    "test_critical_50_count",
    "test_critical_50_rmse",
    "test_critical_50_mae",
    "test_critical_50_late_prediction_ratio",
    "test_critical_50_over_error_mean",
    "test_critical_50_under_error_mean",
    "run_dir",
]


def model_display_name(model_name: str) -> str:
    return MODEL_DISPLAY_NAMES.get(str(model_name), str(model_name))


def evidence_level_for_row(row: dict[str, Any]) -> str:
    experiment_name = str(row.get("experiment_name", ""))
    result_status = str(row.get("result_status", ""))
    if row.get("allowed_for_paper") is not True:
        return "invalid"
    if result_status == "formal" and experiment_name.startswith("paper_main_v3_seed"):
        seed = row.get("seed")
        return "formal_single_seed" if seed in {42, "42"} else "formal_multi_seed_candidate"
    if result_status == "supplementary" or experiment_name.startswith("ablation_"):
        return "supplementary_ablation"
    if "robustness" in experiment_name.lower():
        return "robustness"
    return "invalid"


def _round_metric(value: Any, digits: int = 3) -> Any:
    if isinstance(value, (int, str)) or value is None:
        return value
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value


def build_main_table(rows: list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    df = _normalise_identity_columns(pd.DataFrame(rows).copy())
    df = _add_paper_display_columns(df)
    available = [col for col in MAIN_COLUMNS if col in df.columns]
    sort_cols = [col for col in ["experiment_name", "subset", "model"] if col in available]
    df = df[available].sort_values(sort_cols).reset_index(drop=True)
    for col in [
        "test_rmse",
        "test_mae",
        "test_r2",
        "test_nasa_score",
        "batch_inference_ms",
        "single_sample_inference_ms",
        "cpu_single_sample_inference_ms",
    ]:
        if col in df.columns:
            df[col] = df[col].map(_round_metric)
    return df


def build_complexity_table(rows: list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    df = _normalise_identity_columns(pd.DataFrame(rows).copy())
    df = _add_paper_display_columns(df)
    cols = [
        c
        for c in [
            "experiment_name",
            "subset",
            "model",
            "model_display",
            "evidence_level",
            "parameters",
            "batch_inference_ms",
            "single_sample_inference_ms",
            "cpu_single_sample_inference_ms",
            "test_rmse",
            "test_nasa_score",
        ]
        if c in df.columns
    ]
    sort_cols = [col for col in ["experiment_name", "subset", "parameters", "model"] if col in cols]
    df = df[cols].sort_values(sort_cols).reset_index(drop=True)
    for col in ["batch_inference_ms", "single_sample_inference_ms", "cpu_single_sample_inference_ms", "test_rmse", "test_nasa_score"]:
        if col in df.columns:
            df[col] = df[col].map(_round_metric)
    return df


def build_critical_zone_table(rows: list[dict[str, Any]] | pd.DataFrame) -> pd.DataFrame:
    df = _normalise_identity_columns(pd.DataFrame(rows).copy())
    df = _add_paper_display_columns(df)
    available = [col for col in CRITICAL_ZONE_COLUMNS if col in df.columns]
    sort_cols = [col for col in ["experiment_name", "subset", "model"] if col in available]
    df = df[available].sort_values(sort_cols).reset_index(drop=True)
    for col in [
        "test_critical_30_rmse",
        "test_critical_30_mae",
        "test_critical_30_late_prediction_ratio",
        "test_critical_30_over_error_mean",
        "test_critical_30_under_error_mean",
        "test_critical_50_rmse",
        "test_critical_50_mae",
        "test_critical_50_late_prediction_ratio",
        "test_critical_50_over_error_mean",
        "test_critical_50_under_error_mean",
    ]:
        if col in df.columns:
            df[col] = df[col].map(_round_metric)
    return df


def _normalise_identity_columns(df: pd.DataFrame) -> pd.DataFrame:
    if not df.empty and "experiment_name" not in df.columns:
        df["experiment_name"] = "default"
    elif "experiment_name" in df.columns:
        df["experiment_name"] = df["experiment_name"].fillna("default")
    return df


def _add_paper_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    if "model" in df.columns:
        df["model_display"] = df["model"].map(model_display_name)
    if "evidence_level" not in df.columns:
        df["evidence_level"] = [evidence_level_for_row(row) for row in df.to_dict(orient="records")]
    return df


def filter_metric_rows(
    rows: list[dict[str, Any]],
    *,
    experiment_prefix: str | None = None,
    experiment_names: list[str] | None = None,
    subsets: list[str] | None = None,
    models: list[str] | None = None,
) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    names = set(experiment_names or [])
    subset_set = set(subsets or [])
    model_set = set(models or [])

    for row in rows:
        experiment_name = str(row.get("experiment_name") or "default")
        if experiment_prefix and not experiment_name.startswith(experiment_prefix):
            continue
        if names and experiment_name not in names:
            continue
        if subset_set and row.get("subset") not in subset_set:
            continue
        if model_set and row.get("model") not in model_set:
            continue
        normalized = dict(row)
        normalized["experiment_name"] = experiment_name
        filtered.append(normalized)
    return filtered


def _is_archived_path(path: Path, root: Path) -> bool:
    try:
        rel_parts = path.relative_to(root).parts
    except ValueError:
        rel_parts = path.parts
    return any(part.startswith("archive_") for part in rel_parts)


def collect_metrics(results_dir: str | Path, include_archives: bool = False) -> list[dict[str, Any]]:
    import json

    root = Path(results_dir).resolve()
    project_root = root.parent
    rows: list[dict[str, Any]] = []
    for path in root.rglob("metrics.json"):
        if not include_archives and _is_archived_path(path, root):
            continue
        with path.open("r", encoding="utf-8-sig") as f:
            row = json.load(f)
        try:
            row["run_dir"] = path.parent.relative_to(project_root).as_posix()
        except ValueError:
            row["run_dir"] = path.parent.as_posix()
        row.setdefault("created_time", datetime.fromtimestamp(path.stat().st_mtime).astimezone().isoformat(timespec="seconds"))
        rows.append(row)
    return rows


def write_table_bundle(rows: list[dict[str, Any]], output_dir: str | Path) -> dict[str, str]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    main = build_main_table(rows)
    complexity = build_complexity_table(rows)
    critical = build_critical_zone_table(rows)
    main_csv = output_dir / "table_main_accuracy.csv"
    main_md = output_dir / "table_main_accuracy.md"
    complexity_csv = output_dir / "table_complexity.csv"
    complexity_md = output_dir / "table_complexity.md"
    critical_csv = output_dir / "table_critical_zone.csv"
    critical_md = output_dir / "table_critical_zone.md"
    main.to_csv(main_csv, index=False)
    main_md.write_text(main.to_markdown(index=False) + "\n", encoding="utf-8")
    complexity.to_csv(complexity_csv, index=False)
    complexity_md.write_text(complexity.to_markdown(index=False) + "\n", encoding="utf-8")
    critical.to_csv(critical_csv, index=False)
    critical_md.write_text(critical.to_markdown(index=False) + "\n", encoding="utf-8")
    return {
        "main_csv": str(main_csv),
        "main_md": str(main_md),
        "complexity_csv": str(complexity_csv),
        "complexity_md": str(complexity_md),
        "critical_zone_csv": str(critical_csv),
        "critical_zone_md": str(critical_md),
    }
