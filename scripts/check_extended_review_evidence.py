from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).absolute().parents[1]


def require_manifest(
    path: Path,
    *,
    expected_rows: int,
    key_columns: tuple[str, ...],
    expected_levels: dict[str, set[object]],
) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing run manifest: {path}")
    frame = pd.read_csv(path)
    if len(frame) != expected_rows:
        raise ValueError(f"{path.name}: rows={len(frame)}, expected={expected_rows}")
    if frame.duplicated(list(key_columns)).any():
        raise ValueError(f"{path.name}: duplicate cells in {key_columns}")
    for column, expected in expected_levels.items():
        actual = set(frame[column].tolist())
        if actual != expected:
            raise ValueError(f"{path.name}: {column}={sorted(actual)}, expected={sorted(expected)}")
    if "status" in frame and not set(frame["status"]).issubset({"completed", "reused"}):
        raise ValueError(f"{path.name}: incomplete statuses={sorted(set(frame['status']))}")
    if frame.astype(str).apply(lambda column: column.str.contains("smoke", case=False).any()).any():
        raise ValueError(f"{path.name}: smoke artifact present")
    for raw in frame["run_dir"]:
        run_dir = Path(raw)
        if not run_dir.is_absolute():
            run_dir = PROJECT_ROOT / run_dir
        for filename in ("metrics.json", "test_predictions.csv", "best_model.pt"):
            if not (run_dir / filename).exists():
                raise FileNotFoundError(f"Incomplete run cell: {run_dir / filename}")
    return frame


def require_csv(path: Path, expected_rows: int) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing analysis artifact: {path}")
    frame = pd.read_csv(path)
    if len(frame) != expected_rows:
        raise ValueError(f"{path.name}: rows={len(frame)}, expected={expected_rows}")
    if frame.isna().any().any():
        bad = frame.columns[frame.isna().any()].tolist()
        raise ValueError(f"{path.name}: missing values in {bad}")
    return frame


def check_extended_review_evidence(project_root: Path = PROJECT_ROOT) -> list[str]:
    results = project_root / "results"
    evidence = project_root / "paper_outputs" / "advanced_evidence"
    errors: list[str] = []

    def require_rotation_evidence() -> pd.DataFrame:
        frame = require_manifest(
            results / "ncmapss_dev_rotation" / "run_manifest.csv",
            expected_rows=54,
            key_columns=("point", "test_unit", "training_seed"),
            expected_levels={
                "point": {"RAST", "Core", "Asym"},
                "test_unit": {2, 5, 10, 16, 18, 20},
                "training_seed": {42, 123, 2024},
            },
        )
        for row in frame.itertuples(index=False):
            run_dir = Path(row.run_dir)
            if not run_dir.is_absolute():
                run_dir = project_root / run_dir
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8-sig"))
            expected = {
                "result_status": "exploratory_dev_unit_rotation",
                "evidence_level": "exploratory_dev_unit_rotation",
                "test_source": "development_unit",
                "operating_point": row.point,
                "rotation_test_unit": int(row.test_unit),
                "rotation_validation_unit": int(row.validation_unit),
            }
            mismatched = {
                key: (metrics.get(key), value)
                for key, value in expected.items()
                if metrics.get(key) != value
            }
            if mismatched:
                raise ValueError(f"{run_dir}: rotation metadata mismatch={mismatched}")
        return frame

    checks = [
        lambda: require_manifest(
            results / "augmentation_distribution_sensitivity" / "run_manifest.csv",
            expected_rows=75,
            key_columns=("point", "condition", "training_seed"),
            expected_levels={
                "point": {"RAST", "Core", "Asym"},
                "condition": {"p0", "p025", "p05", "p1", "mix50"},
                "training_seed": {42, 123, 2024, 2025, 2026},
            },
        ),
        lambda: require_manifest(
            results / "seed_factorial_fd004" / "run_manifest.csv",
            expected_rows=27,
            key_columns=("point", "split_seed", "training_seed"),
            expected_levels={
                "point": {"RAST", "Core", "Asym"},
                "split_seed": {42, 123, 2024},
                "training_seed": {42, 123, 2024},
            },
        ),
        require_rotation_evidence,
        lambda: require_csv(evidence / "augmentation_distribution_seed_level.csv", 75),
        lambda: require_csv(evidence / "augmentation_distribution_summary.csv", 15),
        lambda: require_csv(evidence / "augmentation_distribution_paired_intervals.csv", 150),
        lambda: require_csv(evidence / "seed_factorial_fd004_cells.csv", 27),
        lambda: require_csv(evidence / "seed_factorial_fd004_decomposition.csv", 12),
        lambda: require_csv(evidence / "seed_factorial_fd004_paired_intervals.csv", 12),
        lambda: require_csv(evidence / "ncmapss_dev_rotation_cells.csv", 54),
        lambda: require_csv(evidence / "ncmapss_dev_rotation_summary.csv", 3),
        lambda: require_csv(evidence / "ncmapss_dev_rotation_paired_intervals.csv", 9),
        lambda: require_csv(evidence / "fd002_protocol_transfer_seed_metrics.csv", 25),
        lambda: require_csv(evidence / "fd002_protocol_transfer_summary.csv", 5),
        lambda: require_csv(evidence / "fd002_protocol_transfer_paired_bootstrap.csv", 20),
    ]
    for check in checks:
        try:
            check()
        except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
            errors.append(str(exc))
    return errors


def main() -> None:
    errors = check_extended_review_evidence()
    if errors:
        for error in errors:
            print(f"[EXTENDED EVIDENCE FAIL] {error}")
        raise SystemExit(1)
    print("EXTENDED_REVIEW_EVIDENCE_PASS")


if __name__ == "__main__":
    main()
