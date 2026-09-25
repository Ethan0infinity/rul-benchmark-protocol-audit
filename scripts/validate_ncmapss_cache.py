from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rul.ncmapss import load_prepared_cache


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the leakage-safe N-CMAPSS DS02 window cache.")
    parser.add_argument(
        "--cache",
        default=str(PROJECT_ROOT / "data" / "external" / "processed" / "ncmapss_ds02_smp100_win50.npz"),
    )
    args = parser.parse_args()
    path = Path(args.cache)
    prepared = load_prepared_cache(path)
    splits = {"train": prepared.train, "validation": prepared.val, "test": prepared.test}
    issues: list[str] = []
    expected_units = {"train": {2, 5, 10, 16, 18}, "validation": {20}, "test": {11, 14, 15}}
    expected_features = len(prepared.feature_names)
    rows: dict[str, object] = {}
    for name, split in splits.items():
        units = set(map(int, np.unique(split.unit_ids)))
        if units != expected_units[name]:
            issues.append(f"{name} units={sorted(units)} expected={sorted(expected_units[name])}")
        if split.x.ndim != 3 or split.x.shape[1:] != (50, expected_features):
            issues.append(f"{name} x shape={split.x.shape}")
        if len(split.y) != len(split.x) or len(split.unit_ids) != len(split.x):
            issues.append(f"{name} row counts differ")
        if not np.isfinite(split.x).all() or not np.isfinite(split.y).all():
            issues.append(f"{name} contains NaN/Inf")
        if float(split.y.min()) < 0:
            issues.append(f"{name} contains negative RUL")
        rows[name] = {
            "windows": int(len(split.y)),
            "shape": list(split.x.shape),
            "units": sorted(units),
            "rul_min": float(split.y.min()),
            "rul_max": float(split.y.max()),
        }
    if expected_units["train"] & expected_units["validation"] or expected_units["train"] & expected_units["test"]:
        issues.append("unit split overlap")

    report = {
        "cache": str(path.relative_to(PROJECT_ROOT)),
        "sha256": sha256_file(path),
        "allow_pickle_required": False,
        "feature_count": expected_features,
        "splits": rows,
        "issues": issues,
    }
    output = PROJECT_ROOT / "reports" / "ncmapss_cache_validation.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if issues:
        print("NCMAPSS_CACHE_VALIDATION_FAIL")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)
    print(
        f"NCMAPSS_CACHE_VALIDATION_PASS sha256={report['sha256']} "
        f"train={rows['train']['windows']} validation={rows['validation']['windows']} test={rows['test']['windows']}"
    )


if __name__ == "__main__":
    main()
