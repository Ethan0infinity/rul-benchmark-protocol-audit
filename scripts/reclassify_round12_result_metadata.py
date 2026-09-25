from __future__ import annotations

import csv
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rul.protocol import classify_result_status, load_protocol_lock


RESULTS_ROOT = PROJECT_ROOT / "results" / "round12_new_evidence"
OUTPUT = (
    PROJECT_ROOT
    / "paper_outputs"
    / "round12_new_evidence"
    / "metadata_reclassification_audit.csv"
)
FIELDS = (
    "protocol_name",
    "protocol_version",
    "result_status",
    "is_smoke",
    "is_archive",
    "allowed_for_paper",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def main() -> None:
    lock = load_protocol_lock()
    rows = []
    manifest_root = PROJECT_ROOT / "paper_outputs" / "round12_new_evidence"
    referenced_paths = set()
    for manifest in manifest_root.glob("*_run_manifest.csv"):
        frame = pd.read_csv(manifest)
        if "run_dir" in frame:
            referenced_paths.update(
                PROJECT_ROOT / str(run_dir) / "metrics.json"
                for run_dir in frame["run_dir"]
            )
    for path in sorted(referenced_paths):
        if not path.exists():
            raise FileNotFoundError(path)
        metrics = json.loads(path.read_text(encoding="utf-8-sig"))
        if metrics.get("dataset") == "N-CMAPSS_DS02":
            continue
        if RESULTS_ROOT not in path.parents:
            if (
                metrics.get("result_status") not in {"formal", "supplementary"}
                or metrics.get("allowed_for_paper") is not True
            ):
                raise RuntimeError(
                    f"Shared external result is not a registered artifact: {path}"
                )
            continue
        experiment = str(metrics.get("experiment_name", ""))
        expected = classify_result_status(experiment, path.parent, lock)
        if expected["result_status"] != "supplementary":
            raise RuntimeError(
                f"Round-12 experiment is not registered as supplementary: {experiment}"
            )
        before_hash = sha256(path)
        before = {field: metrics.get(field) for field in FIELDS}
        numeric_before = {
            key: value
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        for field in FIELDS:
            metrics[field] = expected[field]
        numeric_after = {
            key: value
            for key, value in metrics.items()
            if isinstance(value, (int, float)) and not isinstance(value, bool)
        }
        if numeric_before != numeric_after:
            raise RuntimeError(f"Numeric field changed during reclassification: {path}")
        path.write_text(
            json.dumps(metrics, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        after_hash = sha256(path)
        rows.append(
            {
                "metrics_path": path.relative_to(PROJECT_ROOT).as_posix(),
                "experiment_name": experiment,
                "before_sha256": before_hash,
                "after_sha256": after_hash,
                "changed_fields": ";".join(
                    field
                    for field in FIELDS
                    if before[field] != metrics[field]
                ),
                "numeric_fields_changed": numeric_before != numeric_after,
                **{f"before_{field}": before[field] for field in FIELDS},
                **{f"after_{field}": metrics[field] for field in FIELDS},
            }
        )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    print(
        "ROUND12_METADATA_RECLASSIFICATION_PASS "
        f"files={len(rows)} numeric_fields_changed=0 output={OUTPUT}"
    )


if __name__ == "__main__":
    main()
