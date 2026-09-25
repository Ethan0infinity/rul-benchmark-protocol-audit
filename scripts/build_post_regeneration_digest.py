"""Write a post-regeneration digest separate from deposit-byte checksums."""

from __future__ import annotations

import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOTS = {
    "analysis-working": PROJECT_ROOT / "paper_outputs" / "analysis_working",
    "advanced-evidence": PROJECT_ROOT / "paper_outputs" / "advanced_evidence",
    "manuscript-evidence": PROJECT_ROOT / "paper_outputs" / "manuscript_v3",
    "release-canonical": PROJECT_ROOT / "paper_outputs" / "release_v1",
    "revised-working": PROJECT_ROOT / "paper_outputs" / "revised_tables",
    "revised-figures": PROJECT_ROOT / "paper_outputs" / "revised_figures",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def row_count(path: Path) -> int | None:
    if path.suffix.lower() != ".csv":
        return None
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def deposit_checksums() -> dict[str, str]:
    checksum_path = PROJECT_ROOT.parents[1] / "checksums.sha256"
    if not checksum_path.exists():
        return {}
    rows: dict[str, str] = {}
    for line in checksum_path.read_text(encoding="utf-8-sig").splitlines():
        if "  " not in line:
            continue
        digest, relative = line.split("  ", 1)
        rows[relative.strip().replace("\\", "/")] = digest.strip().lower()
    return rows


def main() -> None:
    deposited = deposit_checksums()
    records: list[dict[str, object]] = []
    for role, root in OUTPUT_ROOTS.items():
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            relative = path.relative_to(PROJECT_ROOT).as_posix()
            delivery_relative = f"supplementary/replication_package/{relative}"
            digest = sha256(path)
            original = deposited.get(delivery_relative, "")
            records.append(
                {
                    "source_role": role,
                    "project_relative_path": relative,
                    "sha256": digest,
                    "bytes": path.stat().st_size,
                    "csv_data_rows": row_count(path),
                    "deposit_sha256": original,
                    "matches_fresh_deposit": bool(original and original == digest),
                    "comparison_scope": (
                        "byte equality" if original else "post-regeneration digest only"
                    ),
                }
            )

    report_dir = PROJECT_ROOT / "reports" / "stored_results_pipeline"
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "post_regeneration_digest.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)
    summary = {
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "file_count": len(records),
        "fresh_deposit_checksum_available": bool(deposited),
        "matching_deposit_files": sum(
            bool(row["matches_fresh_deposit"]) for row in records
        ),
        "contract": (
            "The delivery-root checksum authenticates freshly extracted bytes. "
            "This digest records regenerated outputs and does not redefine the "
            "deposited checksum authority."
        ),
        "digest_csv": csv_path.relative_to(PROJECT_ROOT).as_posix(),
    }
    (report_dir / "post_regeneration_digest.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        "POST_REGENERATION_DIGEST_PASS "
        f"files={len(records)} deposit_available={bool(deposited)}"
    )


if __name__ == "__main__":
    main()
