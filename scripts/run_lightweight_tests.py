from __future__ import annotations

import csv
import json
from pathlib import Path

from release_evidence_gate import validate_release


ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    passed = 0
    skipped_full_only = 0
    profile = json.loads((ROOT / "delivery_profile.json").read_text(encoding="utf-8-sig"))
    assert profile["profile"] == "lightweight-release-only"
    passed += 1
    assert not (ROOT / "results_manifest.csv").exists() and not (ROOT / "results").exists()
    passed += 1
    assert validate_release(ROOT / "paper_outputs" / "release_v1") == []
    passed += 1
    with (ROOT / "lightweight_manifest.csv").open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows and all(row["capability"] == "release-evidence-audit" for row in rows)
    passed += 1
    for relative in ("benchmark", "eval_harness", "tests"):
        if not (ROOT / relative).exists():
            skipped_full_only += 1
    print(
        f"LIGHTWEIGHT_TESTS_PASS tests={passed} skipped_full_only={skipped_full_only} "
        "profile=release-only"
    )


if __name__ == "__main__":
    main()
