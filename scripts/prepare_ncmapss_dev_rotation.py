from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rul.ncmapss import load_prepared_cache, prepare_ncmapss_ds02, save_prepared_cache


DEV_UNITS = (2, 5, 10, 16, 18, 20)


def rotation_split(test_unit: int) -> tuple[list[int], list[int], list[int]]:
    if test_unit not in DEV_UNITS:
        raise ValueError(f"Unknown DS02 development unit: {test_unit}")
    position = DEV_UNITS.index(test_unit)
    validation_unit = DEV_UNITS[(position + 1) % len(DEV_UNITS)]
    train_units = [unit for unit in DEV_UNITS if unit not in {test_unit, validation_unit}]
    return train_units, [validation_unit], [test_unit]


def audit_cache(prepared, train_units: list[int], validation_units: list[int], test_units: list[int]) -> dict:
    actual = {
        "train": set(map(int, np.unique(prepared.train.unit_ids))),
        "validation": set(map(int, np.unique(prepared.val.unit_ids))),
        "test": set(map(int, np.unique(prepared.test.unit_ids))),
    }
    expected = {
        "train": set(train_units),
        "validation": set(validation_units),
        "test": set(test_units),
    }
    if actual != expected:
        raise ValueError(f"N-CMAPSS rotation cache unit mismatch: actual={actual}, expected={expected}")
    if actual["train"] & actual["validation"] or actual["train"] & actual["test"] or actual["validation"] & actual["test"]:
        raise ValueError("N-CMAPSS rotation cache contains overlapping unit partitions.")
    for split_name, split in (("train", prepared.train), ("validation", prepared.val), ("test", prepared.test)):
        if split.x.ndim != 3 or split.x.shape[-1] != 20:
            raise ValueError(f"{split_name} cache has unexpected feature shape: {split.x.shape}")
        if not np.isfinite(split.x).all() or not np.isfinite(split.y).all():
            raise ValueError(f"{split_name} cache contains non-finite values.")
    if prepared.metadata.get("test_source") != "dev":
        raise ValueError("Development-unit rotation cache must declare test_source='dev'.")
    return {
        "train_units": sorted(actual["train"]),
        "validation_unit": sorted(actual["validation"])[0],
        "test_unit": sorted(actual["test"])[0],
        "feature_count": int(prepared.train.x.shape[-1]),
        "train_windows": int(len(prepared.train.y)),
        "validation_windows": int(len(prepared.val.y)),
        "test_windows": int(len(prepared.test.y)),
        "finite_values": True,
        "partitions_disjoint": True,
    }


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build six leakage-safe N-CMAPSS development-unit rotation caches."
    )
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "data" / "external" / "raw" / "N-CMAPSS_DS02-006.h5"),
    )
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "data" / "external" / "processed" / "ncmapss_dev_rotation"),
    )
    parser.add_argument("--sampling", type=int, default=100)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--condition-clusters", type=int, default=6)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    report_rows = []
    for index, test_unit in enumerate(DEV_UNITS, 1):
        train_units, validation_units, test_units = rotation_split(test_unit)
        output = output_dir / f"ncmapss_ds02_dev_test{test_unit}_smp{args.sampling}_win{args.window_size}.npz"
        print(
            f"[N-CMAPSS DEV ROTATION {index}/{len(DEV_UNITS)}] train={train_units} "
            f"validation={validation_units} test={test_units}",
            flush=True,
        )
        if args.dry_run:
            continue
        if output.exists() and not args.overwrite:
            prepared = load_prepared_cache(output)
            audit = audit_cache(prepared, train_units, validation_units, test_units)
            report_rows.append(
                {
                    **audit,
                    "cache": output.relative_to(PROJECT_ROOT).as_posix(),
                    "cache_sha256": sha256_file(output),
                    "status": "reused_and_validated",
                }
            )
            continue
        prepared = prepare_ncmapss_ds02(
            args.input,
            sampling=args.sampling,
            window_size=args.window_size,
            stride=args.stride,
            condition_clusters=args.condition_clusters,
            train_units=train_units,
            validation_units=validation_units,
            test_units=test_units,
            test_source="dev",
        )
        save_prepared_cache(prepared, output)
        audit = audit_cache(prepared, train_units, validation_units, test_units)
        report_rows.append(
            {
                **audit,
                "cache": output.relative_to(PROJECT_ROOT).as_posix(),
                "cache_sha256": sha256_file(output),
                "status": "created_and_validated",
            }
        )
    if args.dry_run:
        print("NCMAPSS_DEV_ROTATION_DRY_RUN_PASS")
        return
    report = PROJECT_ROOT / "reports" / "ncmapss_dev_rotation_preparation.json"
    report.write_text(json.dumps(report_rows, indent=2) + "\n", encoding="utf-8")
    print(f"NCMAPSS_DEV_ROTATION_CACHE_READY folds={len(report_rows)} report={report}")


if __name__ == "__main__":
    main()
