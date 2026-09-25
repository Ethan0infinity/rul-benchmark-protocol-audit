from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rul.cmapss import COLUMNS, add_train_rul
from rul.features import select_features
from rul.metrics import critical_zone_metrics, mae, nasa_score, rmse
from rul.preprocessing import split_units


@dataclass
class BenchmarkResult:
    name: str
    status: str
    detail: str


def _close(actual: float, expected: float, tol: float = 1e-9) -> bool:
    return abs(float(actual) - float(expected)) <= tol


def _pass(name: str, detail: str) -> BenchmarkResult:
    return BenchmarkResult(name=name, status="PASS", detail=detail)


def _fail(name: str, detail: str) -> BenchmarkResult:
    return BenchmarkResult(name=name, status="FAIL", detail=detail)


def run_toy_rul_metric_benchmark() -> BenchmarkResult:
    y_true = np.array([10.0, 20.0, 30.0, 40.0])
    y_pred = np.array([12.0, 18.0, 33.0, 35.0])
    expected_rmse = math.sqrt((2.0**2 + (-2.0) ** 2 + 3.0**2 + (-5.0) ** 2) / 4.0)
    expected_mae = 3.0
    expected_nasa = (
        math.exp(2.0 / 10.0)
        - 1.0
        + math.exp(2.0 / 13.0)
        - 1.0
        + math.exp(3.0 / 10.0)
        - 1.0
        + math.exp(5.0 / 13.0)
        - 1.0
    )
    critical = critical_zone_metrics(y_true, y_pred)
    checks = [
        _close(rmse(y_true, y_pred), expected_rmse),
        _close(mae(y_true, y_pred), expected_mae),
        _close(nasa_score(y_true, y_pred), expected_nasa),
        critical["critical_30_count"] == 3,
        _close(critical["critical_30_late_prediction_ratio"], 2.0 / 3.0),
    ]
    if not all(checks):
        return _fail(
            "toy_rul_metric",
            f"metric mismatch rmse={rmse(y_true, y_pred)} mae={mae(y_true, y_pred)} nasa={nasa_score(y_true, y_pred)}",
        )
    return _pass("toy_rul_metric", "RMSE/MAE/NASA/critical-zone metrics match recomputed toy gold values")


def run_nasa_score_direction_benchmark() -> BenchmarkResult:
    true = np.array([20.0])
    pred_late = np.array([30.0])
    pred_early = np.array([10.0])
    late_score = nasa_score(true, pred_late)
    early_score = nasa_score(true, pred_early)
    if late_score <= early_score:
        return _fail(
            "nasa_score_direction",
            f"late over-estimation should be penalized more than early under-estimation, late={late_score}, early={early_score}",
        )
    return _pass("nasa_score_direction", "NASA score penalizes RUL over-estimation more heavily than under-estimation")


def _synthetic_cmapss(units: int = 8, cycles: int = 25) -> pd.DataFrame:
    rows = []
    for unit in range(1, units + 1):
        for cycle in range(1, cycles + 1):
            settings = [0.1 * unit, 0.01 * cycle, 1.0]
            sensors = [cycle * (i / 50.0) + unit * 0.03 for i in range(1, 22)]
            rows.append([unit, cycle, *settings, *sensors])
    return pd.DataFrame(rows, columns=COLUMNS)


def run_leakage_guard_benchmark() -> BenchmarkResult:
    train = add_train_rul(_synthetic_cmapss(), rul_cap=125)
    train_units, val_units = split_units(train["unit_id"].to_numpy(), validation_split=0.25, seed=42)
    train_core = train[train["unit_id"].isin(train_units)].copy()
    val = train[train["unit_id"].isin(val_units)].copy()
    selected = select_features(train_core, feature_mode="selected", max_selected_sensors=6, include_settings=True)

    checks = [
        set(train_units).isdisjoint(set(val_units)),
        set(val["unit_id"].unique()).isdisjoint(set(train_core["unit_id"].unique())),
        len(selected.selected_sensors) == 6,
        all(sensor.startswith("s") for sensor in selected.selected_sensors),
    ]
    if not all(checks):
        return _fail("leakage_guard", "train/validation split or train-core-only feature selection invariant failed")
    return _pass("leakage_guard", "unit split and train-core-only feature selection invariants hold")


def run_fd001_smoke_baseline() -> BenchmarkResult:
    cmd = [
        sys.executable,
        str(PROJECT_ROOT / "scripts" / "train.py"),
        "--config",
        str(PROJECT_ROOT / "configs" / "quick_fd001.yaml"),
        "--epochs",
        "2",
        "--no-batch-bar",
    ]
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=600)
    if proc.returncode != 0:
        return _fail("fd001_smoke_baseline", proc.stdout[-1000:])
    return _pass("fd001_smoke_baseline", "2-epoch FD001 smoke training command completed")


def run_all_benchmarks(include_smoke_training: bool = False) -> list[dict[str, Any]]:
    results = [
        run_toy_rul_metric_benchmark(),
        run_nasa_score_direction_benchmark(),
        run_leakage_guard_benchmark(),
    ]
    if include_smoke_training:
        results.append(run_fd001_smoke_baseline())
    return [asdict(result) for result in results]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RUL numeric benchmarks.")
    parser.add_argument("--include-smoke-training", action="store_true", help="Also run a 2-epoch FD001 smoke training benchmark.")
    parser.add_argument("--no-write", action="store_true", help="Do not write benchmark reports.")
    args = parser.parse_args()

    rows = run_all_benchmarks(include_smoke_training=args.include_smoke_training)
    failures = [row for row in rows if row["status"] != "PASS"]
    for row in rows:
        print(f"[BENCHMARK {row['status']}] {row['name']}: {row['detail']}")
    if not args.no_write:
        out = PROJECT_ROOT / "benchmark" / "reference_results" / "latest_results.json"
        out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"Saved benchmark results to {out}")
    if failures:
        raise SystemExit(1)
    print("RUL_BENCHMARK_PASS")


if __name__ == "__main__":
    main()
