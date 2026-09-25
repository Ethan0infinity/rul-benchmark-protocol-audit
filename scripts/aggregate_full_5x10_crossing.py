"""Aggregate the locked full 5 x 10 C-MAPSS finite design."""
from __future__ import annotations

import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "paper_outputs/full_5x10_crossing/full_5x10_run_manifest.csv"
LOCK_NOTE = ROOT / "docs/FULL_5X10_MANIFEST_QA_20260925.md"
PLAN = ROOT / "docs/FULL_5X10_ANALYSIS_PLAN_FROZEN.md"
AMENDMENT = ROOT / "docs/FULL_5X10_ANALYSIS_PLAN_AMENDMENT_01.md"
OUT = ROOT / "paper_outputs/full_5x10_crossing/analysis"
EXPECTED_MANIFEST_SHA256 = "BAE85A8CE7B463350C5751A7833EE5A68FFD7129CEAFDB03CF65D7C931789DAB"
TASKS = ("FD001", "FD002", "FD003", "FD004")
SPLITS = (42, 123, 2024, 2025, 2026)
STREAMS = (31415, 27182, 16180, 14142, 17320, 42, 123, 2024, 2025, 2026)
POINTS = {"rast": "rast_gru", "asym": "rast_gru_v2"}
PRIMARY = ("rmse", "nasa_per_engine", "lpr30")
ENDPOINTS = PRIMARY + ("mae", "slpr30_5", "slpr30_10")
TOL = 2e-6


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def close(actual: float, expected: float) -> bool:
    return math.isfinite(actual) and math.isfinite(expected) and math.isclose(
        actual, expected, rel_tol=TOL, abs_tol=TOL
    )


def load_launcher():
    path = ROOT / "scripts/run_full_5x10_crossing.py"
    spec = importlib.util.spec_from_file_location("full_5x10_launcher", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load launcher: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def verify_manifest_and_protocol() -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    actual_hash = sha256(MANIFEST)
    if actual_hash != EXPECTED_MANIFEST_SHA256:
        raise ValueError(f"Manifest hash changed: {actual_hash}")
    for path, expected in (
        (PLAN, "02696177678D946625B4EDB62005877F80D0F433DB149F3BADF8F0E8629A02E9"),
        (AMENDMENT, "745F4F40A014272CE062B37383009A257AB81B26BF642D9B06DDBD1AB1AF4411"),
    ):
        if sha256(path) != expected:
            raise ValueError(f"Frozen protocol source changed: {path}")

    manifest = pd.read_csv(MANIFEST)
    if len(manifest) != 400 or manifest.status.ne("completed").any():
        raise ValueError("Expected exactly 400 completed canonical rows")
    key_cols = ["subset", "split_level", "training_stream", "point"]
    if manifest.duplicated(key_cols).any():
        raise ValueError("Duplicate canonical cell")
    expected_keys = {
        (task, split, stream, point)
        for task in TASKS
        for split in SPLITS
        for stream in STREAMS
        for point in POINTS
    }
    actual_keys = set(
        zip(
            manifest.subset.astype(str),
            manifest.split_level.astype(int),
            manifest.training_stream.astype(int),
            manifest.point.astype(str),
        )
    )
    if actual_keys != expected_keys:
        raise ValueError("Manifest identities do not equal the frozen 5 x 10 design")

    launcher = load_launcher()
    base = launcher.load_config(ROOT / "configs/paper_main.yaml")
    jobs = launcher.jobs(base, 80, launcher.RESULTS_DIR)
    if len(jobs) != len(manifest):
        raise ValueError("Frozen launcher schedule length differs from manifest")
    for row, job in zip(manifest.itertuples(index=False), jobs):
        if (
            int(row.index) != job["run_index"] if "run_index" in job else False
        ):
            raise ValueError("Schedule order mismatch")
        if (
            row.subset != job["subset"]
            or int(row.split_level) != job["split_level"]
            or int(row.training_stream) != job["training_stream"]
            or row.point != job["point"]
            or row.model != job["model"]
            or row.run_dir != job["run_dir"].relative_to(ROOT).as_posix()
            or row.config_sha256 != launcher.config_fingerprint(job["config"])
            or int(row.planned_epochs) != 80
        ):
            raise ValueError(f"Manifest/schedule mismatch at row {row.index}")
        if not launcher.complete(job, 80):
            raise ValueError(f"Run failed launcher completion check: {row.run_dir}")
    return manifest, jobs


def read_run(row: Any, job: dict[str, Any]) -> dict[str, Any]:
    run_dir = ROOT / row.run_dir
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8-sig"))
    history = pd.read_csv(run_dir / "epoch_history.csv")
    predictions = pd.read_csv(run_dir / "test_predictions.csv")

    completed = int(metrics.get("completed_epochs", -1))
    patience = int(job["config"]["training"].get("patience", -1))
    reason = metrics.get("training_complete_reason")
    if metrics.get("selection_metric") != "val_last_risk_score" or metrics.get(
        "checkpoint_selection_metric"
    ) != "val_last_risk_score":
        raise ValueError(f"Wrong checkpoint selection metric: {row.run_dir}")
    if completed != len(history) or completed < 1:
        raise ValueError(f"History length mismatch: {row.run_dir}")
    if history.epoch.astype(int).tolist() != list(range(1, completed + 1)):
        raise ValueError(f"Non-contiguous epoch history: {row.run_dir}")
    if reason == "early_stopped":
        if completed >= 80 or patience < 1 or int(history.iloc[-1].patience_wait) < patience:
            raise ValueError(f"Early-stop rule not met: {row.run_dir}")
    elif reason == "max_epochs":
        if completed != 80:
            raise ValueError(f"Max-epoch run ended early: {row.run_dir}")
    else:
        raise ValueError(f"Unknown stop reason: {row.run_dir}")

    required_columns = {"unit_id", "end_cycle", "true_rul", "pred_rul"}
    if not required_columns.issubset(predictions.columns) or predictions.empty:
        raise ValueError(f"Prediction schema invalid: {row.run_dir}")
    if predictions.unit_id.duplicated().any():
        raise ValueError(f"Duplicate test engine: {row.run_dir}")
    pred_values = predictions[["end_cycle", "true_rul", "pred_rul"]].to_numpy(dtype=float)
    if not np.isfinite(pred_values).all():
        raise ValueError(f"Non-finite prediction data: {row.run_dir}")

    error = predictions.pred_rul.to_numpy(float) - predictions.true_rul.to_numpy(float)
    true = predictions.true_rul.to_numpy(float)
    late_zone = error[true <= 30.0]
    score = np.where(error < 0, np.exp(-error / 13.0) - 1.0, np.exp(error / 10.0) - 1.0)
    n_engines = int(predictions.unit_id.nunique())
    values = {
        "rmse": float(np.sqrt(np.mean(error**2))),
        "mae": float(np.mean(np.abs(error))),
        "nasa_per_engine": float(np.sum(score) / n_engines),
        "lpr30": float(np.mean(late_zone > 0)) if len(late_zone) else float("nan"),
        "slpr30_5": float(np.mean(late_zone > 5.0)) if len(late_zone) else float("nan"),
        "slpr30_10": float(np.mean(late_zone > 10.0)) if len(late_zone) else float("nan"),
    }
    if int(metrics.get("n_test_units", -1)) != n_engines:
        raise ValueError(f"Test engine count mismatch: {row.run_dir}")
    checks = {
        "rmse": float(metrics["test_rmse"]),
        "mae": float(metrics["test_mae"]),
        "nasa_per_engine": float(metrics["test_nasa_score"]) / n_engines,
        "lpr30": float(metrics["test_critical_30_late_prediction_ratio"]),
        "slpr30_5": float(metrics["test_critical_30_severe_late_5_ratio"]),
        "slpr30_10": float(metrics["test_critical_30_severe_late_10_ratio"]),
    }
    for endpoint in ENDPOINTS:
        if not close(values[endpoint], checks[endpoint]):
            raise ValueError(f"Prediction/metrics mismatch for {endpoint}: {row.run_dir}")
    for key, value in metrics.items():
        if isinstance(value, (int, float)) and not isinstance(value, bool) and not math.isfinite(float(value)):
            raise ValueError(f"Non-finite metric {key}: {row.run_dir}")

    return {
        "task": row.subset,
        "split_level": int(row.split_level),
        "training_stream": int(row.training_stream),
        "point": row.point,
        "model": row.model,
        "run_dir": row.run_dir,
        "completed_epochs": completed,
        "stop_reason": reason,
        "n_engines": n_engines,
        **values,
    }


def make_outputs(manifest: pd.DataFrame, jobs: list[dict[str, Any]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    raw_rows = []
    engine_identity: dict[str, pd.DataFrame] = {}
    for row, job in zip(manifest.itertuples(index=False), jobs):
        run_dir = ROOT / row.run_dir
        pred = pd.read_csv(run_dir / "test_predictions.csv").sort_values("unit_id").reset_index(drop=True)
        identity = pred[["unit_id", "end_cycle", "true_rul"]]
        if row.subset in engine_identity and not identity.equals(engine_identity[row.subset]):
            raise ValueError(f"Test-engine identity differs within {row.subset}")
        engine_identity[row.subset] = identity
        raw_rows.append(read_run(row, job))

    raw = pd.DataFrame(raw_rows)
    key = ["task", "split_level", "training_stream"]
    rast = raw[raw.point == "rast"].set_index(key)
    asym = raw[raw.point == "asym"].set_index(key)
    if not rast.index.equals(asym.index):
        raise ValueError("RAST/Asym matched-cell identities differ")
    pairs = []
    for k in rast.index:
        row = {"task": k[0], "split_level": k[1], "training_stream": k[2]}
        for endpoint in ENDPOINTS:
            row[endpoint + "_rast"] = float(rast.loc[k, endpoint])
            row[endpoint + "_asym"] = float(asym.loc[k, endpoint])
            row[endpoint + "_asym_minus_rast"] = row[endpoint + "_asym"] - row[endpoint + "_rast"]
        pairs.append(row)
    paired = pd.DataFrame(pairs).sort_values(key).reset_index(drop=True)
    if len(paired) != 200:
        raise ValueError("Expected 200 matched task/split/stream contrasts")

    summary_rows = []
    for task in TASKS:
        group = paired[paired.task == task]
        for endpoint in ENDPOINTS:
            values = group[f"{endpoint}_asym_minus_rast"].to_numpy(float)
            if len(values) != 50 or not np.isfinite(values).all():
                raise ValueError(f"Incomplete/non-finite finite-design contrast: {task} {endpoint}")
            summary_rows.append({
                "task": task,
                "endpoint": endpoint,
                "cells": len(values),
                "negative": int(np.sum(values < 0)),
                "zero": int(np.sum(values == 0)),
                "positive": int(np.sum(values > 0)),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "minimum": float(np.min(values)),
                "maximum": float(np.max(values)),
            })
    summary = pd.DataFrame(summary_rows)

    rank_rows = []
    for task in TASKS:
        group = paired[paired.task == task]
        for endpoint in ENDPOINTS:
            values = group[f"{endpoint}_asym_minus_rast"].to_numpy(float)
            rank_rows.append({
                "task": task,
                "endpoint": endpoint,
                "matched_cells": len(values),
                "asym_lower_metric_cells": int(np.sum(values < 0)),
                "tied_cells": int(np.sum(values == 0)),
                "rast_lower_metric_cells": int(np.sum(values > 0)),
                "interpretation": "pairwise finite-design rank support; not a population or universal model ranking",
            })
    rank_support = pd.DataFrame(rank_rows)

    band_rows = []
    for task in TASKS:
        for endpoint, margins in (("rmse", (0.5, 1.0, 2.0)), ("lpr30", (0.01, 0.025, 0.05))):
            values = paired.loc[paired.task == task, f"{endpoint}_asym_minus_rast"].to_numpy(float)
            for margin in margins:
                band_rows.append({
                    "task": task,
                    "endpoint": endpoint,
                    "candidate_margin": margin,
                    "below_negative_margin": int(np.sum(values < -margin)),
                    "within_inclusive_band": int(np.sum(np.abs(values) <= margin)),
                    "above_positive_margin": int(np.sum(values > margin)),
                    "n": len(values),
                    "interpretation": "post-result candidate magnitude band; not a SESOI or equivalence test",
                })
    bands = pd.DataFrame(band_rows)

    marginal_rows = []
    for task in TASKS:
        subset = paired[paired.task == task]
        for endpoint in ENDPOINTS:
            col = f"{endpoint}_asym_minus_rast"
            for level, group in subset.groupby("split_level", sort=True):
                marginal_rows.append({"task": task, "endpoint": endpoint, "factor": "split_level", "level": int(level), "n": len(group), "mean_contrast": float(group[col].mean()), "median_contrast": float(group[col].median())})
            for level, group in subset.groupby("training_stream", sort=True):
                marginal_rows.append({"task": task, "endpoint": endpoint, "factor": "training_stream", "level": int(level), "n": len(group), "mean_contrast": float(group[col].mean()), "median_contrast": float(group[col].median())})
    marginals = pd.DataFrame(marginal_rows)

    raw.to_csv(OUT / "full_5x10_run_metrics.csv", index=False)
    paired.to_csv(OUT / "full_5x10_paired_contrasts.csv", index=False)
    summary.to_csv(OUT / "full_5x10_finite_design_summary.csv", index=False)
    rank_support.to_csv(OUT / "full_5x10_pairwise_rank_support.csv", index=False)
    bands.to_csv(OUT / "full_5x10_candidate_band_occupancy.csv", index=False)
    marginals.to_csv(OUT / "full_5x10_factor_marginals.csv", index=False)

    figures = OUT / "figures"
    figures.mkdir(exist_ok=True)
    for task in TASKS:
        for endpoint in PRIMARY:
            grid = paired[paired.task == task].pivot(index="split_level", columns="training_stream", values=f"{endpoint}_asym_minus_rast").reindex(index=SPLITS, columns=STREAMS)
            fig, ax = plt.subplots(figsize=(11, 4.8), constrained_layout=True)
            bound = float(np.nanmax(np.abs(grid.to_numpy())))
            image = ax.imshow(grid.to_numpy(), aspect="auto", cmap="RdBu_r", vmin=-bound, vmax=bound)
            ax.set_xticks(range(len(STREAMS)), [str(v) for v in STREAMS], rotation=35, ha="right")
            ax.set_yticks(range(len(SPLITS)), [str(v) for v in SPLITS])
            ax.set_xlabel("Training-stream level")
            ax.set_ylabel("Split level")
            ax.set_title(f"{task}: Asym minus RAST, {endpoint}")
            fig.colorbar(image, ax=ax, label=f"Paired contrast in {endpoint}")
            fig.savefig(figures / f"heatmap_{task}_{endpoint}.png", dpi=220)
            plt.close(fig)

    record = {
        "analysis_role": "result-informed retrospective finite-design description",
        "manifest_path": MANIFEST.relative_to(ROOT).as_posix(),
        "manifest_sha256": sha256(MANIFEST),
        "analysis_plan_sha256": sha256(PLAN),
        "amendment_01_sha256": sha256(AMENDMENT),
        "aggregation_script_sha256": sha256(Path(__file__)),
        "canonical_cells": 400,
        "matched_contrasts": 200,
        "contrast_direction": "rast_gru_v2 (Asym) minus rast_gru (RAST)",
        "primary_endpoints": list(PRIMARY),
        "secondary_endpoints": ["mae", "slpr30_5", "slpr30_10"],
        "candidate_bands": {"rmse": [0.5, 1.0, 2.0], "lpr30": [0.01, 0.025, 0.05]},
        "candidate_bands_note": "No NASA/engine margin was introduced because none was specified in the locked material.",
        "inference_boundary": "finite-design summaries only; no population p-values or variance components",
    }
    (OUT / "aggregation_provenance.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    print(json.dumps({"status": "FULL_5X10_AGGREGATION_PASS", "outputs": 6, "heatmaps": 12, "manifest_sha256": record["manifest_sha256"], "script_sha256": record["aggregation_script_sha256"]}, indent=2))


def main() -> None:
    manifest, jobs = verify_manifest_and_protocol()
    make_outputs(manifest, jobs)


if __name__ == "__main__":
    main()
