"""Reject a submission assembled from mismatched main/supplement releases."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import re


SCRIPT_ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def resolve_layout(
    start: Path,
) -> tuple[Path, Path, Path, bool]:
    start = start.resolve()
    for candidate in (start, *start.parents):
        package_main = candidate / "manuscript" / "main_revised.tex"
        package_supplement = (
            candidate
            / "supplementary"
            / "manuscript"
            / "supplement_en.tex"
        )
        if package_main.exists() and package_supplement.exists():
            return (
                candidate / "manuscript",
                candidate / "supplementary" / "manuscript",
                candidate / "cross_document_release.json",
                True,
            )

    project_root = start
    if not (project_root / "scripts").is_dir():
        for candidate in (start, *start.parents):
            if (candidate / "scripts" / "check_cross_document_sync.py").exists():
                project_root = candidate
                break
    main_root = project_root.parent / "正文"
    supplement_root = project_root / "supplementary"
    return (
        main_root,
        supplement_root,
        project_root / "reports" / "cross_document_release.json",
        False,
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate synchronized main/supplement release identity."
    )
    parser.add_argument("--package-root", default=str(SCRIPT_ROOT))
    args = parser.parse_args()

    main_root, supplement_root, report_path, packaged = resolve_layout(
        Path(args.package_root)
    )
    issues: list[str] = []
    main_sync = main_root / "generated" / "release_sync.tex"
    supplement_sync = supplement_root / "generated" / "release_sync.tex"
    required_sources = [
        main_root / "main_revised.tex",
        supplement_root / "supplement_en.tex",
    ]
    if not packaged:
        required_sources.extend(
            [
                main_root / "main_zh.tex",
                supplement_root / "supplement_zh.tex",
            ]
        )
    for path in [main_sync, supplement_sync, *required_sources]:
        if not path.exists():
            issues.append(f"missing {path}")
    if issues:
        print("CROSS_DOCUMENT_SYNC_FAIL")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)

    if main_sync.read_bytes() != supplement_sync.read_bytes():
        issues.append("main and supplementary release_sync.tex differ")
    sync_text = main_sync.read_text(encoding="utf-8-sig")
    match = re.search(r"\\EvidenceReleaseID\}\{([^}]+)\}", sync_text)
    release_id = match.group(1) if match else ""
    if not release_id:
        issues.append("release_sync.tex does not define EvidenceReleaseID")
    for source in required_sources:
        text = source.read_text(encoding="utf-8-sig")
        if r"\input{generated/release_sync.tex}" not in text:
            issues.append(
                f"{source.name} does not include generated/release_sync.tex"
            )

    for source in required_sources:
        pdf = source.with_suffix(".pdf")
        timestamp_tolerance = 2.0 if packaged else 0.0
        if not pdf.exists():
            issues.append(f"missing compiled PDF {pdf.name}")
        elif (
            pdf.stat().st_mtime + timestamp_tolerance
            < max(
                source.stat().st_mtime,
                main_sync.stat().st_mtime,
            )
        ):
            issues.append(f"{pdf.name} predates its source or release identity")

    if not report_path.exists():
        issues.append(f"missing {report_path}")
    else:
        report = json.loads(report_path.read_text(encoding="utf-8-sig"))
        expected = {
            "main_source_sha256": sha256(main_root / "main_revised.tex"),
            "supplement_source_sha256": sha256(
                supplement_root / "supplement_en.tex"
            ),
            "sync_file_sha256": sha256(main_sync),
        }
        if not packaged:
            expected.update(
                {
                    "main_zh_source_sha256": sha256(main_root / "main_zh.tex"),
                    "supplement_zh_source_sha256": sha256(
                        supplement_root / "supplement_zh.tex"
                    ),
                }
            )
        if report.get("release_id") != release_id:
            issues.append("release report ID differs from release_sync.tex")
        for field, value in expected.items():
            if report.get(field) != value:
                issues.append(f"release report hash is stale: {field}")

    if issues:
        print("CROSS_DOCUMENT_SYNC_FAIL")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)
    print(
        f"CROSS_DOCUMENT_SYNC_PASS release_id={release_id} "
        f"layout={'package' if packaged else 'workspace'}"
    )


if __name__ == "__main__":
    main()
