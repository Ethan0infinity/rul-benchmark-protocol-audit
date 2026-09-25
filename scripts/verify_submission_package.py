from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import sys
from pathlib import Path


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def parse_checksum_file(path: Path) -> dict[str, str]:
    checksums: dict[str, str] = {}
    for line_number, raw_line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not raw_line.strip():
            continue
        try:
            digest, relative_path = raw_line.split("  ", 1)
        except ValueError as exc:
            raise ValueError(f"Malformed checksum line {line_number}: {raw_line!r}") from exc
        checksums[relative_path] = digest
    return checksums


def verify_package_checksums(package_root: Path) -> list[str]:
    errors: list[str] = []
    checksum_path = package_root / "checksums.sha256"
    if not checksum_path.exists():
        return ["Missing package-root checksums.sha256."]
    expected = parse_checksum_file(checksum_path)
    actual_paths = {
        path.relative_to(package_root).as_posix()
        for path in package_root.rglob("*")
        if path.is_file() and path != checksum_path
    }
    missing = sorted(set(expected) - actual_paths)
    extra = sorted(actual_paths - set(expected))
    if missing:
        errors.append(f"Checksum inventory references missing files: {missing[:5]}")
    if extra:
        errors.append(f"Checksum inventory omits delivered files: {extra[:5]}")
    for relative_path in sorted(set(expected) & actual_paths):
        observed = sha256_file(package_root / Path(relative_path))
        if observed != expected[relative_path]:
            errors.append(f"Package checksum mismatch: {relative_path}")
    return errors


def verify_release_manifest(package_root: Path) -> list[str]:
    errors: list[str] = []
    release_root = (
        package_root
        / "supplementary"
        / "replication_package"
        / "paper_outputs"
        / "release_v1"
    )
    manifest_path = release_root / "release_manifest.csv"
    if not manifest_path.exists():
        return ["Missing release_v1/release_manifest.csv."]
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return ["Release manifest contains no artifact rows."]
    artifact_ids: set[str] = set()
    release_paths: set[str] = set()
    for row in rows:
        artifact_id = row.get("artifact_id", "")
        relative_path = row.get("release_path", "")
        expected_hash = row.get("release_sha256", "")
        if not artifact_id or artifact_id in artifact_ids:
            errors.append(f"Missing or duplicate artifact_id: {artifact_id!r}")
        artifact_ids.add(artifact_id)
        if not relative_path or relative_path in release_paths:
            errors.append(f"Missing or duplicate release_path: {relative_path!r}")
        release_paths.add(relative_path)
        target = release_root / Path(relative_path)
        if not target.exists():
            errors.append(f"Release manifest target is missing: {relative_path}")
        elif sha256_file(target) != expected_hash:
            errors.append(f"Release manifest hash mismatch: {relative_path}")
    return errors


def verify_portability(package_root: Path) -> list[str]:
    errors: list[str] = []
    for path in package_root.rglob("*"):
        relative_path = path.relative_to(package_root).as_posix()
        if not relative_path.isascii():
            errors.append(f"Non-ASCII package path: {relative_path}")
        if "\\" in relative_path:
            errors.append(f"Backslash package path: {relative_path}")
        if path.is_file() and path.suffix.lower() == ".sh" and b"\r\n" in path.read_bytes():
            errors.append(f"CRLF shell script: {relative_path}")
    duplicate_checksum = (
        package_root
        / "supplementary"
        / "replication_package"
        / "ARTIFACT_SHA256SUMS.txt"
    )
    if duplicate_checksum.exists():
        errors.append("Duplicate checksum authority ARTIFACT_SHA256SUMS.txt is present.")
    return errors


def load_delivery_profile(package_root: Path) -> tuple[str, list[str]]:
    profile_path = package_root / "delivery_profile.json"
    if not profile_path.exists():
        return "full", ["missing delivery_profile.json"]
    payload = json.loads(profile_path.read_text(encoding="utf-8-sig"))
    profile = str(payload.get("profile", "")).strip()
    issues: list[str] = []
    if payload.get("schema_version") != "1.0":
        issues.append("delivery_profile schema_version must be 1.0")
    expected = {
        "full": (True, True, True),
        "lightweight-release-only": (True, False, False),
    }.get(profile)
    if expected is None:
        issues.append(f"unknown delivery profile: {profile!r}")
    else:
        fields = (
            "supports_release_audit",
            "supports_stored_result_regeneration",
            "supports_retraining",
        )
        for field, value in zip(fields, expected):
            if payload.get(field) is not value:
                issues.append(f"delivery profile capability mismatch: {field}")
    return profile, issues


def verify_required_files(package_root: Path, profile: str) -> list[str]:
    required = [
        "manuscript/main_revised.tex",
        "manuscript/main_revised.pdf",
        "manuscript/references.bib",
        "manuscript/generated/release_sync.tex",
        "cross_document_release.json",
        "supplementary/manuscript/supplement_en.tex",
        "supplementary/manuscript/supplement_en.pdf",
        "supplementary/manuscript/generated/release_sync.tex",
        "supplementary/replication_package/README_REPLICATION.md",
        "supplementary/replication_package/paper_outputs/release_v1/release_manifest.csv",
    ]
    replication = package_root / "supplementary" / "replication_package"
    if profile == "full":
        required.extend(
            [
                "supplementary/replication_package/results_manifest.csv",
                "supplementary/replication_package/reproduce_minimal.sh",
            ]
        )
    elif profile == "lightweight-release-only":
        required.extend(
            [
                "supplementary/replication_package/lightweight_manifest.csv",
                "supplementary/replication_package/reproduce_release_only.sh",
                "supplementary/replication_package/reproduce_release_only.ps1",
            ]
        )
        forbidden = [replication / "results_manifest.csv", replication / "results"]
        if any(path.exists() for path in forbidden):
            return ["Lightweight delivery contains a full-results contract or result tree."]
    else:
        return [f"Unknown delivery profile: {profile!r}"]
    return [
        f"Missing required file: {relative_path}"
        for relative_path in required
        if not (package_root / Path(relative_path)).exists()
    ]


def verify_cross_document_release(package_root: Path) -> list[str]:
    main_sync = package_root / "manuscript" / "generated" / "release_sync.tex"
    supplement_sync = (
        package_root
        / "supplementary"
        / "manuscript"
        / "generated"
        / "release_sync.tex"
    )
    if not main_sync.exists() or not supplement_sync.exists():
        return ["Cross-document release identity is missing."]
    if main_sync.read_bytes() != supplement_sync.read_bytes():
        return ["Main and supplementary release identities differ."]
    text = main_sync.read_text(encoding="utf-8-sig")
    if "\\EvidenceReleaseID" not in text or "\\EvidenceProtocolVersion" not in text:
        return ["Cross-document release identity is incomplete."]
    report_path = package_root / "cross_document_release.json"
    if not report_path.exists():
        return ["Cross-document release report is missing."]
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    release_match = re.search(
        r"\\EvidenceReleaseID\}\{([^}]+)\}",
        text,
    )
    expected = {
        "release_id": release_match.group(1) if release_match else "",
        "main_source_sha256": sha256_file(
            package_root / "manuscript" / "main_revised.tex"
        ),
        "supplement_source_sha256": sha256_file(
            package_root
            / "supplementary"
            / "manuscript"
            / "supplement_en.tex"
        ),
        "sync_file_sha256": sha256_file(main_sync),
    }
    return [
        f"Cross-document release report is stale: {field}"
        for field, value in expected.items()
        if report.get(field) != value
    ]


def verify_public_metadata(package_root: Path, allow_pending: bool) -> list[str]:
    replication = package_root / "supplementary" / "replication_package"
    artifact_path = replication / "artifact_metadata.json"
    author_path = replication / "author_metadata.json"
    issues: list[str] = []
    if not artifact_path.exists():
        issues.append("artifact_metadata.json is missing")
        artifact = {}
    else:
        artifact = json.loads(artifact_path.read_text(encoding="utf-8-sig"))
    if not author_path.exists():
        issues.append("author_metadata.json is missing")
        author = {}
    else:
        author = json.loads(author_path.read_text(encoding="utf-8-sig"))

    repository = str(artifact.get("repository_url", "")).strip()
    archive = str(artifact.get("archive_url", "")).strip()
    release_tag = str(artifact.get("release_tag", "")).strip()
    commit_hash = str(artifact.get("commit_hash", "")).strip()
    doi = str(artifact.get("doi", "")).strip()
    if not repository.startswith("https://"):
        issues.append("repository_url must be a real HTTPS URL")
    if not (release_tag or re.fullmatch(r"[0-9a-fA-F]{7,64}", commit_hash)):
        issues.append("release_tag or an immutable commit_hash is required")
    if not (archive.startswith("https://") or doi):
        issues.append("archive_url or DOI/record identifier is required for public mode")

    for field in ("name_en", "affiliation", "institution", "city", "country", "email"):
        if not str(author.get(field, "")).strip():
            issues.append(f"author metadata field is empty: {field}")
    if not isinstance(author.get("corresponding_author"), bool):
        issues.append("corresponding_author must be explicitly true or false")

    if issues and allow_pending:
        print("WARNING_PENDING_PUBLIC_METADATA " + " | ".join(issues))
        return []
    if issues:
        return [
            "Public-submission metadata is incomplete: " + "; ".join(issues)
            + ". Use --allow-pending-metadata only for internal validation."
        ]
    return []


def verify_release_authority(package_root: Path) -> list[str]:
    report_path = (
        package_root
        / "supplementary"
        / "replication_package"
        / "reports"
        / "release_authority_gate.json"
    )
    if not report_path.exists():
        return ["Release authority report is missing."]
    report = json.loads(report_path.read_text(encoding="utf-8-sig"))
    issues = list(report.get("issues", []))
    subgates = list(report.get("subgates", []))
    if report.get("status") != "PASS" or issues:
        return [
            "Release authority report contains unresolved issues: "
            + ", ".join(str(issue) for issue in issues)
        ]
    failed = [
        str(row.get("name", "unknown"))
        for row in subgates
        if row.get("status") != "PASS" or row.get("exit_code") != 0
    ]
    if failed:
        return [
            "Release authority sub-gates are not all PASS: "
            + ", ".join(failed)
        ]
    return []


def verify(package_root: Path, allow_pending_metadata: bool = False) -> None:
    root = package_root.resolve()
    profile, profile_issues = load_delivery_profile(root)
    checks = [
        profile_issues,
        verify_required_files(root, profile),
        verify_cross_document_release(root),
        verify_portability(root),
        verify_package_checksums(root),
        verify_release_manifest(root),
        verify_release_authority(root),
        verify_public_metadata(root, allow_pending_metadata),
    ]
    errors = [error for group in checks for error in group]
    if errors:
        for error in errors:
            print(f"ERROR {error}")
        raise SystemExit(1)
    print(
        "SUBMISSION_PACKAGE_VERIFY_PASS "
        f"root={root} profile={profile} pending_metadata_allowed={allow_pending_metadata}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify a package from its own root without relying on the source workspace."
    )
    parser.add_argument("package_root", nargs="?", default=".")
    parser.add_argument("--allow-pending-metadata", action="store_true")
    args = parser.parse_args()
    verify(Path(args.package_root), allow_pending_metadata=args.allow_pending_metadata)


if __name__ == "__main__":
    main()
