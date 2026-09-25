from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
import zipfile
from datetime import datetime
from pathlib import Path

import h5py

PROJECT_ROOT = Path(__file__).resolve().parents[1]
OFFICIAL_URL = "https://phm-datasets.s3.amazonaws.com/NASA/17.+Turbofan+Engine+Degradation+Simulation+Data+Set+2.zip"
EXPECTED_ARCHIVE_BYTES = 15_760_443_389
EXPECTED_DS02_BYTES = 2_450_472_504
DS02_FILENAME = "N-CMAPSS_DS02-006.h5"
TRANSFER_MIRROR_URL = (
    "https://huggingface.co/datasets/NovaBenya/N-CMAPSS_DS02-006.h5/resolve/main/"
    "N-CMAPSS_DS02-006.h5?download=true"
)
REQUIRED_KEYS = {
    "W_dev",
    "X_s_dev",
    "X_v_dev",
    "Y_dev",
    "A_dev",
    "W_test",
    "X_s_test",
    "X_v_test",
    "Y_test",
    "A_test",
    "W_var",
    "X_s_var",
    "X_v_var",
    "A_var",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def locate_ds02_entry(archive: zipfile.ZipFile) -> zipfile.ZipInfo:
    matches = [entry for entry in archive.infolist() if Path(entry.filename).name == DS02_FILENAME]
    if len(matches) != 1:
        raise ValueError(f"Expected exactly one {DS02_FILENAME} entry, found {len(matches)}")
    return matches[0]


def extract_ds02(archive_path: Path, output_path: Path, *, overwrite: bool = False) -> None:
    if output_path.exists() and not overwrite:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(archive_path) as archive:
        entry = locate_ds02_entry(archive)
        temp_path = output_path.with_suffix(output_path.suffix + ".extracting")
        with archive.open(entry) as source, temp_path.open("wb") as target:
            shutil.copyfileobj(source, target, length=8 * 1024 * 1024)
        temp_path.replace(output_path)


def audit_h5(path: Path) -> dict[str, object]:
    with h5py.File(path, "r") as hdf:
        keys = set(hdf.keys())
        missing = sorted(REQUIRED_KEYS - keys)
        if missing:
            raise ValueError(f"N-CMAPSS DS02 is missing required HDF5 keys: {missing}")
        report = {
            "dataset": "N-CMAPSS DS02-006",
            "path": path.relative_to(PROJECT_ROOT).as_posix(),
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
            "keys": sorted(keys),
            "W_dev_shape": list(hdf["W_dev"].shape),
            "X_s_dev_shape": list(hdf["X_s_dev"].shape),
            "Y_dev_shape": list(hdf["Y_dev"].shape),
            "A_dev_shape": list(hdf["A_dev"].shape),
            "W_test_shape": list(hdf["W_test"].shape),
            "X_s_test_shape": list(hdf["X_s_test"].shape),
            "Y_test_shape": list(hdf["Y_test"].shape),
            "A_test_shape": list(hdf["A_test"].shape),
            "audited_at": datetime.now().astimezone().isoformat(timespec="seconds"),
            "source_url": OFFICIAL_URL,
            "transfer_mirror_url": TRANSFER_MIRROR_URL,
        }
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract and audit N-CMAPSS DS02 from the official NASA archive.")
    parser.add_argument(
        "--archive",
        default=str(PROJECT_ROOT / "data" / "external" / "raw" / "N-CMAPSS_full.zip"),
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "external" / "raw" / DS02_FILENAME),
    )
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    archive_path = Path(args.archive)
    output_path = Path(args.output)
    if not output_path.exists() or args.overwrite:
        if not archive_path.exists():
            raise FileNotFoundError(
                f"Missing DS02 file and NASA archive. Expected either {output_path} or {archive_path}.\n"
                f"Official archive: {OFFICIAL_URL}\nTransfer mirror for DS02 only: {TRANSFER_MIRROR_URL}"
            )
        if archive_path.stat().st_size != EXPECTED_ARCHIVE_BYTES:
            raise ValueError(
                f"Archive size mismatch: expected {EXPECTED_ARCHIVE_BYTES}, found {archive_path.stat().st_size}. "
                "Resume the official download before extraction."
            )
        extract_ds02(archive_path, output_path, overwrite=args.overwrite)
    if output_path.stat().st_size != EXPECTED_DS02_BYTES:
        raise ValueError(
            f"DS02 size mismatch: expected {EXPECTED_DS02_BYTES}, found {output_path.stat().st_size}. "
            "Resume the download before auditing."
        )
    report = audit_h5(output_path)
    report_path = PROJECT_ROOT / "reports" / "ncmapss_ds02_audit.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(
        "NCMAPSS_DS02_READY "
        f"dev_rows={report['Y_dev_shape'][0]} test_rows={report['Y_test_shape'][0]} sha256={report['sha256']}"
    )


if __name__ == "__main__":
    main()
