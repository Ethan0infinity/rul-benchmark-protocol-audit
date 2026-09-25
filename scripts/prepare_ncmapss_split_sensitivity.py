from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rul.ncmapss import TEST_UNITS, prepare_ncmapss_ds02, save_prepared_cache


DEV_UNITS = [2, 5, 10, 16, 18, 20]
VALIDATION_UNITS = [2, 10, 20]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare three predeclared N-CMAPSS DS02 development splits.")
    parser.add_argument("--raw", default=str(PROJECT_ROOT / "data" / "external" / "raw" / "N-CMAPSS_DS02-006.h5"))
    args = parser.parse_args()
    output = PROJECT_ROOT / "data" / "external" / "processed"
    output.mkdir(parents=True, exist_ok=True)
    manifest = []
    for val_unit in VALIDATION_UNITS:
        train_units = [unit for unit in DEV_UNITS if unit != val_unit]
        target = output / f"ncmapss_ds02_smp100_win50_val{val_unit}.npz"
        if val_unit == 20 and (output / "ncmapss_ds02_smp100_win50.npz").exists():
            source = output / "ncmapss_ds02_smp100_win50.npz"
            target.write_bytes(source.read_bytes())
        else:
            prepared = prepare_ncmapss_ds02(args.raw, sampling=100, window_size=50, stride=1,
                                            condition_clusters=6, train_units=train_units,
                                            validation_units=[val_unit], test_units=TEST_UNITS)
            save_prepared_cache(prepared, target)
        manifest.append(
            {
                "validation_unit": val_unit,
                "train_units": train_units,
                "test_units": TEST_UNITS,
                "cache": target.relative_to(PROJECT_ROOT).as_posix(),
                "bytes": target.stat().st_size,
                "sha256": sha256_file(target),
            }
        )
        print(f"[N-CMAPSS SPLIT CACHE] val={val_unit} train={train_units} path={target}")
    report = PROJECT_ROOT / "reports" / "ncmapss_split_sensitivity_manifest.json"
    report.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"NCMAPSS_SPLIT_CACHES_READY count={len(manifest)} report={report}")


if __name__ == "__main__":
    main()
