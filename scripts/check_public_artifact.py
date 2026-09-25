from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate immutable public artifact metadata.")
    parser.add_argument("--allow-pending-metadata", action="store_true")
    args = parser.parse_args()
    path = PROJECT_ROOT / "artifact_metadata.json"
    metadata = json.loads(path.read_text(encoding="utf-8"))
    issues = []
    repository = str(metadata.get("repository_url", ""))
    archive = str(metadata.get("archive_url", ""))
    doi = str(metadata.get("doi", ""))
    commit = str(metadata.get("commit_hash", ""))
    release = str(metadata.get("release_tag", ""))
    cache_distribution = str(metadata.get("ncmapss_cache_distribution", ""))
    if not repository.startswith("https://"):
        issues.append("repository_url must be a real HTTPS URL")
    if not release:
        issues.append("release_tag is empty")
    if not re.fullmatch(r"[0-9a-fA-F]{7,64}", commit):
        issues.append("commit_hash is missing or invalid")
    if not archive.startswith("https://") and not doi.startswith("10."):
        issues.append("archive_url or DOI is required")
    if cache_distribution not in {"included_permitted", "excluded"}:
        issues.append("ncmapss_cache_distribution must be 'included_permitted' or 'excluded' after license review")
    if issues:
        if args.allow_pending_metadata:
            print("WARNING_PENDING_PUBLIC_ARTIFACT " + " | ".join(issues))
            print("PUBLIC_ARTIFACT_PENDING local audit only")
            return
        print("PUBLIC_ARTIFACT_FAIL")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)
    print(f"PUBLIC_ARTIFACT_PASS repository={repository} release={release} doi={doi or archive}")


if __name__ == "__main__":
    main()
