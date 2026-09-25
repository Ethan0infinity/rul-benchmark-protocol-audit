from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from release_evidence_gate import validate_release


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    profile_path = PROJECT_ROOT / "delivery_profile.json"
    manifest_path = PROJECT_ROOT / "lightweight_manifest.csv"
    release = PROJECT_ROOT / "paper_outputs" / "release_v1"
    issues: list[str] = []
    if not profile_path.exists():
        issues.append("missing delivery_profile.json")
    else:
        profile = json.loads(profile_path.read_text(encoding="utf-8-sig"))
        if profile.get("schema_version") != "1.0":
            issues.append("delivery profile schema_version must be 1.0")
        if profile.get("profile") != "lightweight-release-only":
            issues.append("delivery profile is not lightweight-release-only")
        expected_capabilities = {
            "supports_release_audit": True,
            "supports_stored_result_regeneration": False,
            "supports_retraining": False,
        }
        for field, expected in expected_capabilities.items():
            if profile.get(field) is not expected:
                issues.append(f"delivery profile capability mismatch: {field}")
    if (PROJECT_ROOT / "results_manifest.csv").exists() or (PROJECT_ROOT / "results").exists():
        issues.append("lightweight package must not declare or contain the full result tree")
    if not manifest_path.exists():
        issues.append("missing lightweight_manifest.csv")
    else:
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        if not rows:
            issues.append("lightweight manifest is empty")
        for row in rows:
            target = PROJECT_ROOT / Path(row["path"])
            if not target.is_file():
                issues.append(f"manifest target missing: {row['path']}")
            elif sha256_file(target) != row["sha256"]:
                issues.append(f"manifest hash mismatch: {row['path']}")
    issues.extend(validate_release(release))
    if issues:
        print("LIGHTWEIGHT_VERIFY_FAIL")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)
    print("LIGHTWEIGHT_VERIFY_PASS profile=lightweight-release-only")


if __name__ == "__main__":
    main()
