from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rul.reporting import collect_metrics, filter_metric_rows
from rul.protocol import load_protocol_lock, validate_metric_row_against_protocol


COMMON_REQUIRED_METRIC_KEYS = [
    "experiment_name",
    "subset",
    "model",
    "run_dir",
    "best_epoch",
    "test_rmse",
    "test_mae",
    "test_r2",
    "test_nasa_score",
    "test_critical_30_rmse",
    "test_critical_50_rmse",
    "completed_epochs",
    "planned_epochs",
    "training_complete_reason",
    "single_sample_inference_ms",
]

C_MAPSS_REQUIRED_METRIC_KEYS = [
    "selection_metric",
    "best_selection_score",
    "best_checkpoint_score",
    "checkpoint_selection_metric",
    "best_epoch_val_all_loss",
    "completed_epochs",
    "planned_epochs",
    "training_complete_reason",
    "augmentation_seed",
    "deterministic_training",
    "val_last_rul_mean",
    "val_last_rul_median",
    "cpu_single_sample_inference_ms",
]

NCMAPSS_REQUIRED_METRIC_KEYS = [
    "dataset",
    "seed",
    "checkpoint_selection_metric",
    "best_val_rmse",
    "best_val_risk_score",
    "parameters",
    "unit_macro_rmse",
    "unit_macro_mae",
    "unit_macro_nasa_per_window",
    "unit_macro_lpr30",
]

PROTOCOL_METRIC_KEYS = [
    "protocol_name",
    "protocol_version",
    "result_status",
    "allowed_for_paper",
    "is_smoke",
    "is_archive",
]


def resolve_run_dir(run_dir: str | Path) -> Path:
    path = Path(str(run_dir))
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def is_ncmapss_row(row: dict) -> bool:
    dataset = str(row.get("dataset", "")).upper()
    return dataset.startswith("N-CMAPSS") or str(row.get("subset", "")).upper() == "DS02"


def check_rows(
    rows: list[dict],
    expected_count: int | None = None,
    min_epochs: int | None = None,
    require_protocol_v2: bool = False,
) -> list[str]:
    issues: list[str] = []
    protocol_lock = load_protocol_lock() if require_protocol_v2 else None
    if expected_count is not None and len(rows) != expected_count:
        issues.append(f"expected {expected_count} metrics rows, found {len(rows)}")

    seen: set[tuple[str, str, str]] = set()
    for row in rows:
        identity = (
            str(row.get("experiment_name", "default")),
            str(row.get("subset", "")),
            str(row.get("model", "")),
        )
        if identity in seen:
            issues.append(f"duplicate run identity: experiment={identity[0]} subset={identity[1]} model={identity[2]}")
        seen.add(identity)

        dataset_required = NCMAPSS_REQUIRED_METRIC_KEYS if is_ncmapss_row(row) else C_MAPSS_REQUIRED_METRIC_KEYS
        missing = [key for key in [*COMMON_REQUIRED_METRIC_KEYS, *dataset_required] if key not in row]
        if missing:
            issues.append(f"{identity}: missing metric keys {missing}")
        if require_protocol_v2:
            missing_protocol = [key for key in PROTOCOL_METRIC_KEYS if key not in row]
            if missing_protocol:
                issues.append(f"{identity}: missing protocol metric keys {missing_protocol}")

        completed_epochs = int(row.get("completed_epochs", 0) or 0)
        planned_epochs = int(row.get("planned_epochs", 0) or 0)
        complete_reason = str(row.get("training_complete_reason", ""))
        if min_epochs is not None and completed_epochs < min_epochs and complete_reason != "early_stopped":
            issues.append(
                f"{identity}: completed_epochs={completed_epochs} is below min_epochs={min_epochs} "
                f"and run did not early-stop"
            )
        if planned_epochs and completed_epochs > planned_epochs:
            issues.append(f"{identity}: completed_epochs={completed_epochs} exceeds planned_epochs={planned_epochs}")

        if require_protocol_v2:
            assert protocol_lock is not None
            selection_metric = str(row.get("selection_metric", ""))
            if selection_metric not in {"val_last_risk_score"}:
                issues.append(f"{identity}: selection_metric={selection_metric!r} is not engine-level validation")
            if str(row.get("validation_last_strategy", "")) != "simulated":
                issues.append(f"{identity}: validation_last_strategy must be simulated for protocol v3")
            if str(row.get("scaler", "")) != "condition_standard":
                issues.append(f"{identity}: scaler must be condition_standard for protocol v3")
            if row.get("append_missing_mask") is not True:
                issues.append(f"{identity}: append_missing_mask must be true for protocol v3")
            if row.get("deterministic_training") is not True:
                issues.append(f"{identity}: deterministic_training must be true for protocol v3")
            for protocol_issue in validate_metric_row_against_protocol(row, protocol_lock):
                issues.append(f"{identity}: {protocol_issue}")

        run_dir = resolve_run_dir(str(row.get("run_dir", "")))
        if not run_dir.exists():
            issues.append(f"{identity}: run_dir does not exist: {run_dir}")
            continue
        required_files = ["metrics.json", "run_config.yaml", "best_model.pt", "test_predictions.csv"]
        if not is_ncmapss_row(row):
            required_files.append("val_last_cutoff_summary.json")
        for filename in required_files:
            if not (run_dir / filename).exists():
                issues.append(f"{identity}: missing {filename} in {run_dir}")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Check that paper tables will use complete, traceable metrics rows.")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--experiment-prefix", default=None)
    parser.add_argument("--experiment-name", action="append", dest="experiment_names", default=None)
    parser.add_argument("--subsets", nargs="+", default=None)
    parser.add_argument("--models", nargs="+", default=None)
    parser.add_argument("--expected-count", type=int, default=None)
    parser.add_argument("--min-epochs", type=int, default=None, help="Reject runs shorter than this unless they early-stopped.")
    parser.add_argument(
        "--require-protocol-v3",
        "--require-protocol-v2",
        dest="require_protocol_v2",
        action="store_true",
        help="Require the locked paper_main_v3 protocol fields.",
    )
    args = parser.parse_args()

    rows = filter_metric_rows(
        collect_metrics(args.results_dir),
        experiment_prefix=args.experiment_prefix,
        experiment_names=args.experiment_names,
        subsets=args.subsets,
        models=args.models,
    )
    issues = check_rows(
        rows,
        expected_count=args.expected_count,
        min_epochs=args.min_epochs,
        require_protocol_v2=args.require_protocol_v2,
    )
    print(f"[CONSISTENCY] metrics_rows={len(rows)} results_dir={args.results_dir}")
    if rows:
        for row in sorted(rows, key=lambda item: (item.get("experiment_name", ""), item.get("subset", ""), item.get("model", ""))):
            print(
                "[ROW] "
                f"experiment={row.get('experiment_name')} subset={row.get('subset')} model={row.get('model')} "
                f"rmse={float(row.get('test_rmse', 0.0)):.4f} nasa={float(row.get('test_nasa_score', 0.0)):.2f} "
                f"run_dir={row.get('run_dir')}"
            )
    if issues:
        print("[CONSISTENCY FAIL]")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)
    print("[CONSISTENCY PASS]")


if __name__ == "__main__":
    main()
