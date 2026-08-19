from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rul.reporting import _is_archived_path, collect_metrics, filter_metric_rows, model_display_name, write_table_bundle


MULTISEED_CORE_MODELS = [
    "gru",
    "lstm",
    "tcn",
    "tcn_gru",
    "rast_gru",
    "cnn_lstm",
    "attention_gru",
    "bigru_attention",
    "transformer_lite",
    "dual_attention_tcn",
    "sensor_graph_gru",
    "quantile_gru",
    "rast_gru_v2",
]
MULTISEED_METRICS = [
    "test_rmse",
    "test_mae",
    "test_nasa_score",
    "test_critical_30_late_prediction_ratio",
    "single_sample_inference_ms",
    "cpu_single_sample_inference_ms",
]

ABLATION_COLUMNS = [
    "experiment_name",
    "subset",
    "model",
    "model_display",
    "ablation_variant",
    "evidence_level",
    "result_status",
    "allowed_for_paper",
    "completed_epochs",
    "scaler",
    "append_missing_mask",
    "model_use_reliability_gate",
    "model_use_channel_gate",
    "model_use_temporal_attention",
    "model_use_local_trend",
    "test_rmse",
    "test_mae",
    "test_r2",
    "test_nasa_score",
    "test_critical_30_rmse",
    "test_critical_30_mae",
    "test_critical_30_late_prediction_ratio",
    "test_critical_50_rmse",
    "test_critical_50_mae",
    "test_critical_50_late_prediction_ratio",
    "parameters",
    "single_sample_inference_ms",
    "cpu_single_sample_inference_ms",
    "run_dir",
]

ABLATION_ROUND_COLUMNS = [
    "test_rmse",
    "test_mae",
    "test_r2",
    "test_nasa_score",
    "test_critical_30_rmse",
    "test_critical_30_mae",
    "test_critical_30_late_prediction_ratio",
    "test_critical_50_rmse",
    "test_critical_50_mae",
    "test_critical_50_late_prediction_ratio",
    "single_sample_inference_ms",
    "cpu_single_sample_inference_ms",
]


def _keep_row(
    row: dict,
    *,
    experiment_prefix: str | None,
    experiment_names: list[str] | None,
    subsets: list[str] | None,
    models: list[str] | None,
) -> bool:
    filtered = filter_metric_rows(
        [row],
        experiment_prefix=experiment_prefix,
        experiment_names=experiment_names,
        subsets=subsets,
        models=models,
    )
    return bool(filtered)


def collect_robustness(
    results_dir: Path,
    *,
    experiment_prefix: str | None = None,
    experiment_names: list[str] | None = None,
    subsets: list[str] | None = None,
    models: list[str] | None = None,
) -> pd.DataFrame:
    rows = []
    project_root = results_dir.resolve().parent
    for path in results_dir.rglob("robustness.json"):
        if _is_archived_path(path, results_dir):
            continue
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        metrics_path = path.parent / "metrics.json"
        metrics = {}
        if metrics_path.exists():
            metrics = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
        experiment_name = metrics.get("experiment_name", "default")
        identity = {
            "experiment_name": experiment_name,
            "subset": payload.get("subset"),
            "model": payload.get("model"),
        }
        if not _keep_row(
            identity,
            experiment_prefix=experiment_prefix,
            experiment_names=experiment_names,
            subsets=subsets,
            models=models,
        ):
            continue
        for row in payload.get("rows", []):
            row = dict(row)
            row["experiment_name"] = experiment_name
            row["subset"] = payload.get("subset")
            row["model"] = payload.get("model")
            row["created_time"] = metrics.get("created_time")
            try:
                row["run_dir"] = path.parent.resolve().relative_to(project_root).as_posix()
            except ValueError:
                row["run_dir"] = path.parent.as_posix()
            rows.append(row)
    return pd.DataFrame(rows)


def _round_cell(value: object, digits: int = 3) -> object:
    if isinstance(value, (int, str)) or value is None:
        return value
    try:
        return round(float(value), digits)
    except (TypeError, ValueError):
        return value


def write_ablation_table(results_dir: Path, output_dir: Path) -> dict[str, str]:
    ablation_summary = results_dir / "ablation_runs.csv"
    if not ablation_summary.exists():
        return {}

    ablation = pd.read_csv(ablation_summary)
    if ablation.empty:
        return {}

    if "model" in ablation.columns:
        ablation["model_display"] = ablation["model"].map(model_display_name)
    if "evidence_level" not in ablation.columns:
        ablation["evidence_level"] = "supplementary_ablation"
    if "ablation_variant" not in ablation.columns and "experiment_name" in ablation.columns:
        ablation["ablation_variant"] = ablation["experiment_name"].astype(str).str.replace("ablation_", "", regex=False)

    available = [column for column in ABLATION_COLUMNS if column in ablation.columns]
    sort_columns = [column for column in ["subset", "ablation_variant", "model"] if column in available]
    ablation = ablation[available].sort_values(sort_columns).reset_index(drop=True)
    for column in ABLATION_ROUND_COLUMNS:
        if column in ablation.columns:
            ablation[column] = ablation[column].map(_round_cell)

    output_dir.mkdir(parents=True, exist_ok=True)
    ablation_csv = output_dir / "table_ablation.csv"
    ablation_md = output_dir / "table_ablation.md"
    ablation.to_csv(ablation_csv, index=False)
    ablation_md.write_text(ablation.to_markdown(index=False) + "\n", encoding="utf-8")
    return {"ablation_csv": str(ablation_csv), "ablation_md": str(ablation_md)}


def write_fd004_multiseed_ablation_table(results_dir: Path, output_dir: Path) -> dict[str, str]:
    ablation_summary = results_dir / "ablation_fd004_multiseed_runs.csv"
    if not ablation_summary.exists():
        return {}

    ablation = pd.read_csv(ablation_summary)
    if ablation.empty or "ablation_variant" not in ablation.columns:
        return {}
    if "model" in ablation.columns:
        ablation["model_display"] = ablation["model"].map(model_display_name)
    metrics = [
        "test_rmse",
        "test_mae",
        "test_nasa_score",
        "test_critical_30_late_prediction_ratio",
        "test_critical_30_severe_late_5_ratio",
        "test_critical_30_severe_late_10_ratio",
        "parameters",
        "single_sample_inference_ms",
        "cpu_single_sample_inference_ms",
    ]
    for metric in metrics:
        if metric in ablation.columns:
            ablation[metric] = pd.to_numeric(ablation[metric], errors="coerce")
    grouped = ablation.groupby(["subset", "model", "model_display", "ablation_variant"], as_index=False)
    summary = grouped.agg(
        seed_count=("seed", "nunique"),
        seeds=("seed", lambda x: ",".join(str(int(seed)) for seed in sorted(pd.Series(x).dropna().unique()))),
        **{
            f"{metric}_mean": (metric, "mean")
            for metric in metrics
            if metric in ablation.columns
        },
        **{
            f"{metric}_std": (metric, lambda x: x.std(ddof=1) if x.count() > 1 else 0.0)
            for metric in metrics
            if metric in ablation.columns
        },
    )
    for column in summary.columns:
        if column.endswith("_mean") or column.endswith("_std"):
            summary[column] = summary[column].map(_round_cell)
    variant_order = {
        "full": 0,
        "no_condition_norm": 1,
        "no_missing_mask": 2,
        "no_reliability_gate": 3,
        "no_temporal_attention": 4,
        "no_degradation_aug": 5,
        "weighted_huber_no_asymmetry": 6,
    }
    summary["variant_order"] = summary["ablation_variant"].map(variant_order).fillna(999)
    summary = summary.sort_values(["subset", "variant_order", "model"]).drop(columns=["variant_order"]).reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    ablation_csv = output_dir / "table_ablation_fd004_multiseed.csv"
    ablation_md = output_dir / "table_ablation_fd004_multiseed.md"
    summary.to_csv(ablation_csv, index=False)
    ablation_md.write_text(summary.to_markdown(index=False) + "\n", encoding="utf-8")
    return {"ablation_fd004_multiseed_csv": str(ablation_csv), "ablation_fd004_multiseed_md": str(ablation_md)}


def _seed_from_row(row: dict) -> int | None:
    seed = row.get("seed")
    if pd.notna(seed):
        try:
            return int(seed)
        except (TypeError, ValueError):
            pass
    match = re.search(r"paper_main_v3_seed(\d+)", str(row.get("experiment_name", "")))
    return int(match.group(1)) if match else None


def _normalise_multiseed_rows(rows: list[dict]) -> pd.DataFrame:
    normalized = []
    for row in rows:
        experiment_name = str(row.get("experiment_name", ""))
        if not experiment_name.startswith("paper_main_v3_seed"):
            continue
        if row.get("model") not in MULTISEED_CORE_MODELS:
            continue
        if row.get("result_status") not in {None, "", "formal"}:
            continue
        if row.get("allowed_for_paper") is False:
            continue
        seed = _seed_from_row(row)
        if seed is None:
            continue
        item = dict(row)
        item["seed"] = seed
        normalized.append(item)

    if not normalized:
        return pd.DataFrame()

    df = pd.DataFrame(normalized)
    for metric in MULTISEED_METRICS:
        if metric in df.columns:
            df[metric] = pd.to_numeric(df[metric], errors="coerce")
    return df


def build_multiseed_summary(rows: list[dict]) -> pd.DataFrame:
    df = _normalise_multiseed_rows(rows)
    if df.empty:
        return pd.DataFrame()

    summary_rows = []
    for (subset, model), group in df.groupby(["subset", "model"], sort=True):
        seeds = sorted({int(seed) for seed in group["seed"].dropna().tolist()})
        if len(seeds) < 3:
            continue
        out = {
            "subset": subset,
            "model": model,
            "model_display": model_display_name(model),
            "seed_count": len(seeds),
            "seeds": ",".join(str(seed) for seed in seeds),
        }
        for metric in MULTISEED_METRICS:
            if metric not in group.columns:
                continue
            values = group[metric].dropna()
            if values.empty:
                continue
            out[f"{metric}_mean"] = _round_cell(values.mean())
            out[f"{metric}_std"] = _round_cell(values.std(ddof=1) if len(values) > 1 else 0.0)
        summary_rows.append(out)

    return pd.DataFrame(summary_rows).sort_values(["subset", "model"]).reset_index(drop=True)


def build_multiseed_overall(rows: list[dict]) -> pd.DataFrame:
    df = _normalise_multiseed_rows(rows)
    if df.empty:
        return pd.DataFrame()

    rows = []
    for model, group in df.groupby("model", sort=True):
        seeds = sorted({int(seed) for seed in group["seed"].dropna().tolist()})
        out = {
            "model": model,
            "model_display": model_display_name(model),
            "subset_count": int(group["subset"].nunique()) if "subset" in group.columns else 0,
            "seed_count": len(seeds),
            "seeds": ",".join(str(seed) for seed in seeds),
        }
        for metric in MULTISEED_METRICS:
            if metric not in group.columns:
                continue
            seed_means = group.groupby("seed")[metric].mean().dropna()
            if seed_means.empty:
                continue
            out[f"{metric}_mean"] = _round_cell(seed_means.mean())
            out[f"{metric}_std"] = _round_cell(seed_means.std(ddof=1) if len(seed_means) > 1 else 0.0)
        rows.append(out)

    return pd.DataFrame(rows).sort_values("model").reset_index(drop=True)


def write_multiseed_summary(rows: list[dict], output_dir: Path) -> dict[str, str]:
    summary = build_multiseed_summary(rows)
    if summary.empty:
        return {}
    output_dir.mkdir(parents=True, exist_ok=True)
    multiseed_csv = output_dir / "table_multiseed_summary.csv"
    multiseed_md = output_dir / "table_multiseed_summary.md"
    summary.to_csv(multiseed_csv, index=False)
    multiseed_md.write_text(summary.to_markdown(index=False) + "\n", encoding="utf-8")

    outputs = {"multiseed_csv": str(multiseed_csv), "multiseed_md": str(multiseed_md)}
    overall = build_multiseed_overall(rows)
    if not overall.empty:
        overall_csv = output_dir / "table_multiseed_overall.csv"
        overall_md = output_dir / "table_multiseed_overall.md"
        overall.to_csv(overall_csv, index=False)
        overall_md.write_text(overall.to_markdown(index=False) + "\n", encoding="utf-8")
        outputs.update({"multiseed_overall_csv": str(overall_csv), "multiseed_overall_md": str(overall_md)})
    return outputs


def write_severe_missing_probe_table(results_dir: Path, output_dir: Path) -> dict[str, str]:
    probe_summary = results_dir / "severe_missing_probe_runs.csv"
    if not probe_summary.exists():
        return {}
    probe = pd.read_csv(probe_summary)
    if probe.empty:
        return {}
    if "model" in probe.columns:
        probe["model_display"] = probe["model"].map(model_display_name)
    columns = [
        column
        for column in [
            "experiment_name",
            "subset",
            "model",
            "model_display",
            "result_status",
            "allowed_for_paper",
            "probe_train_missing_rate",
            "probe_train_block_missing_rate",
            "test_rmse",
            "test_mae",
            "test_nasa_score",
            "test_critical_30_late_prediction_ratio",
            "run_dir",
        ]
        if column in probe.columns
    ]
    probe = probe[columns].sort_values([column for column in ["subset", "model", "experiment_name"] if column in columns]).reset_index(drop=True)
    for column in ["test_rmse", "test_mae", "test_nasa_score", "test_critical_30_late_prediction_ratio"]:
        if column in probe.columns:
            probe[column] = probe[column].map(_round_cell)

    output_dir.mkdir(parents=True, exist_ok=True)
    probe_csv = output_dir / "table_severe_missing_probe.csv"
    probe_md = output_dir / "table_severe_missing_probe.md"
    probe.to_csv(probe_csv, index=False)
    probe_md.write_text(probe.to_markdown(index=False) + "\n", encoding="utf-8")
    return {"severe_missing_probe_csv": str(probe_csv), "severe_missing_probe_md": str(probe_md)}


def _manifest_paths_relative_to_project(outputs: dict[str, str]) -> dict[str, str]:
    relative_outputs: dict[str, str] = {}
    for key, value in outputs.items():
        path = Path(value)
        if path.is_absolute():
            try:
                relative_outputs[key] = path.relative_to(PROJECT_ROOT).as_posix()
                continue
            except ValueError:
                pass
        relative_outputs[key] = path.as_posix()
    return relative_outputs


def write_filtered_run_summary(rows: list[dict], results_dir: Path) -> Path:
    df = pd.DataFrame(rows)
    out = results_dir / "paper_main_runs_all.csv"
    if df.empty:
        df.to_csv(out, index=False)
        return out
    if "model_display" not in df.columns and "model" in df.columns:
        df["model_display"] = df["model"].map(model_display_name)
    sort_cols = [col for col in ["experiment_name", "subset", "model"] if col in df.columns]
    if sort_cols:
        df = df.sort_values(sort_cols).reset_index(drop=True)
    df.to_csv(out, index=False)
    return out


def remove_stale_optional_outputs(outputs: dict[str, str], output_dir: Path) -> None:
    for key, filenames in {
        "severe_missing_probe_csv": ["table_severe_missing_probe.csv", "table_severe_missing_probe.md"],
    }.items():
        if key in outputs:
            continue
        for filename in filenames:
            path = output_dir / filename
            if path.exists():
                path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paper-ready tables from experiment results.")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "paper_outputs" / "tables"))
    parser.add_argument("--experiment-prefix", default=None, help="Only include runs whose experiment_name starts with this prefix.")
    parser.add_argument("--experiment-name", action="append", dest="experiment_names", default=None, help="Only include this exact experiment_name. Can be used multiple times.")
    parser.add_argument("--subsets", nargs="+", default=None, help="Only include selected C-MAPSS subsets.")
    parser.add_argument("--models", nargs="+", default=None, help="Only include selected model names.")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    all_metric_rows = collect_metrics(results_dir)
    rows = filter_metric_rows(
        all_metric_rows,
        experiment_prefix=args.experiment_prefix,
        experiment_names=args.experiment_names,
        subsets=args.subsets,
        models=args.models,
    )
    if not rows:
        raise FileNotFoundError(f"No metrics.json files found under {results_dir}")
    if args.experiment_prefix or args.experiment_names:
        summary_path = write_filtered_run_summary(rows, results_dir)
        outputs_for_summary = {"paper_main_runs_all_csv": str(summary_path)}
    else:
        outputs_for_summary = {}
    outputs = write_table_bundle(rows, output_dir)
    outputs.update(outputs_for_summary)

    robustness = collect_robustness(
        results_dir,
        experiment_prefix=args.experiment_prefix,
        experiment_names=args.experiment_names,
        subsets=args.subsets,
        models=args.models,
    )
    if not robustness.empty:
        robust_csv = output_dir / "table_robustness.csv"
        robust_md = output_dir / "table_robustness.md"
        robustness.to_csv(robust_csv, index=False)
        robust_md.write_text(robustness.to_markdown(index=False) + "\n", encoding="utf-8")
        outputs["robustness_csv"] = str(robust_csv)
        outputs["robustness_md"] = str(robust_md)

    outputs.update(write_ablation_table(results_dir, output_dir))
    outputs.update(write_fd004_multiseed_ablation_table(results_dir, output_dir))
    outputs.update(write_multiseed_summary(all_metric_rows, output_dir))
    outputs.update(write_severe_missing_probe_table(results_dir, output_dir))
    remove_stale_optional_outputs(outputs, output_dir)

    manifest = output_dir / "manifest.json"
    manifest_outputs = _manifest_paths_relative_to_project(outputs)
    manifest.write_text(json.dumps(manifest_outputs, indent=2), encoding="utf-8")
    print(json.dumps(manifest_outputs, indent=2))
    print(f"Saved paper tables to {output_dir}")


if __name__ == "__main__":
    main()
