from __future__ import annotations

import argparse
import csv
import hashlib
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def verify_manifest(package_root: str | Path, expected_rows: int | None = None) -> dict[str, int]:
    root = Path(package_root).resolve()
    manifest_path = root / "results_manifest.csv"
    if not manifest_path.exists():
        raise FileNotFoundError(f"Missing manifest: {manifest_path}")

    missing = 0
    mismatches = 0
    rows = 0
    with manifest_path.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows += 1
            for path_col, hash_col in [
                ("metrics_path", "sha256_metrics"),
                ("predictions_path", "sha256_predictions"),
                ("checkpoint_path", "sha256_checkpoint"),
            ]:
                rel = row.get(path_col, "")
                expected = row.get(hash_col, "")
                path = root / rel
                if not rel or not path.exists():
                    missing += 1
                    continue
                actual = sha256_file(path)
                if actual != expected:
                    mismatches += 1

    if expected_rows is not None and rows != expected_rows:
        raise AssertionError(f"Expected {expected_rows} manifest rows, found {rows}.")

    result = {"rows": rows, "missing": missing, "hash_mismatch": mismatches}
    if missing or mismatches:
        print(
            "MANIFEST_VERIFY_FAIL "
            f"rows={rows} missing={missing} hash_mismatch={mismatches}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print(f"MANIFEST_VERIFY_PASS rows={rows} missing={missing} hash_mismatch={mismatches}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify results_manifest.csv hashes against packaged files.")
    parser.add_argument("--package-root", default=".")
    parser.add_argument("--expected-rows", type=int, default=260)
    args = parser.parse_args()
    verify_manifest(args.package_root, expected_rows=args.expected_rows)


if __name__ == "__main__":
    main()
