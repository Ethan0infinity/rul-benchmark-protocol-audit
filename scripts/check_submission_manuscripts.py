from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


SCRIPT_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = (
    "Internal protocol-v3 draft",
    "obsolete protocol-v2",
    "Withheld until",
    "withheld until",
    "single-seed stress-test",
    "200 formal runs",
    "ten evaluated models",
    "nine reproduced lightweight baselines",
    "TBD",
)
INPUT_PATTERN = re.compile(r"\\(?:input|include)\{([^}]+)\}")
GRAPHIC_PATTERN = re.compile(r"\\bestgraphic(?:\[[^\]]*\])?\{([^}]+)\}")
GRAPHIC_SUFFIXES = (".pdf", ".png", ".svg", ".tiff")


def resolve_layout(root: Path) -> tuple[Path, Path, Path, bool]:
    root = root.resolve()
    package_root = next(
        (
            candidate
            for candidate in (root, *root.parents)
            if (candidate / "manuscript" / "main_revised.tex").exists()
            and (candidate / "supplementary" / "manuscript" / "supplement_en.tex").exists()
        ),
        None,
    )
    if package_root is not None:
        return (
            package_root / "manuscript",
            package_root / "supplementary" / "manuscript",
            package_root / "supplementary" / "replication_package" / "author_metadata.json",
            True,
        )
    manuscript = root.parent / "正文"
    if not (manuscript / "main_revised.tex").exists():
        raise FileNotFoundError(f"Cannot locate manuscript layout from {root}")
    return manuscript, root / "supplementary", root / "author_metadata.json", False


def check_compiled(
    source: Path,
    issues: list[str],
    *,
    timestamp_tolerance_seconds: float = 0.0,
) -> None:
    pdf = source.with_suffix(".pdf")
    log = source.with_suffix(".log")
    if not source.exists():
        issues.append(f"missing source: {source}")
        return
    text = source.read_text(encoding="utf-8-sig")
    if r"\appendix" in text:
        issues.append(f"{source.name} embeds audit appendices that belong in Supplementary Information")
    for token in FORBIDDEN:
        if token in text:
            issues.append(f"{source.name} contains forbidden draft token: {token}")
    if not pdf.exists() or pdf.stat().st_size < 10000:
        issues.append(f"missing/empty compiled PDF for {source.name}")
    elif (
        pdf.stat().st_mtime + timestamp_tolerance_seconds
        < source.stat().st_mtime
    ):
        issues.append(f"compiled PDF predates {source.name}")
    if log.exists():
        log_text = log.read_text(encoding="utf-8", errors="replace").lower()
        if "undefined references" in log_text or "undefined citations" in log_text:
            issues.append(f"{source.name} compile log contains undefined references/citations")
        if "overfull \\hbox" in log_text:
            issues.append(f"{source.name} compile log contains an overfull box")


def check_tex_dependencies(
    root_source: Path,
    issues: list[str],
) -> None:
    document_root = root_source.resolve().parent
    pending = [root_source]
    visited: set[Path] = set()
    while pending:
        source = pending.pop().resolve()
        if source in visited:
            continue
        visited.add(source)
        if not source.exists():
            issues.append(f"missing TeX dependency: {source}")
            continue
        text = source.read_text(encoding="utf-8-sig", errors="replace")
        for match in INPUT_PATTERN.finditer(text):
            raw = match.group(1).strip()
            candidate = document_root / raw
            if not candidate.suffix:
                candidate = candidate.with_suffix(".tex")
            if not candidate.exists():
                issues.append(
                    f"{root_source.name} dependency is missing: "
                    f"{candidate.relative_to(document_root).as_posix()}"
                )
            elif candidate.suffix.lower() == ".tex":
                pending.append(candidate)
        for match in GRAPHIC_PATTERN.finditer(text):
            raw = match.group(1).strip()
            base = document_root / raw
            candidates = (
                [base]
                if base.suffix
                else [base.with_suffix(suffix) for suffix in GRAPHIC_SUFFIXES]
            )
            if not any(candidate.exists() for candidate in candidates):
                issues.append(
                    f"{root_source.name} graphic dependency is missing: {raw}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate source-workspace or packaged manuscript layouts.")
    parser.add_argument("--package-root", default=str(SCRIPT_ROOT))
    parser.add_argument("--allow-pending-metadata", action="store_true")
    args = parser.parse_args()
    root = Path(args.package_root)
    manuscript, supplementary, metadata_path, packaged = resolve_layout(root)
    issues: list[str] = []
    timestamp_tolerance = 2.0 if packaged else 0.0
    check_compiled(
        manuscript / "main_revised.tex",
        issues,
        timestamp_tolerance_seconds=timestamp_tolerance,
    )
    check_tex_dependencies(manuscript / "main_revised.tex", issues)
    if not packaged:
        check_compiled(manuscript / "main_zh.tex", issues)
        check_tex_dependencies(manuscript / "main_zh.tex", issues)
    check_compiled(
        supplementary / "supplement_en.tex",
        issues,
        timestamp_tolerance_seconds=timestamp_tolerance,
    )
    check_tex_dependencies(supplementary / "supplement_en.tex", issues)
    if not packaged:
        check_compiled(supplementary / "supplement_zh.tex", issues)
        check_tex_dependencies(supplementary / "supplement_zh.tex", issues)

    metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig")) if metadata_path.exists() else {}
    pending: list[str] = []
    for field in ("name_en", "affiliation", "institution", "city", "country", "email"):
        if not str(metadata.get(field, "")).strip():
            pending.append(f"author metadata field is empty: {field}")
    if not isinstance(metadata.get("corresponding_author"), bool):
        pending.append("corresponding_author must be explicitly true or false")
    orcid = str(metadata.get("orcid", "")).strip()
    if orcid and (len(orcid) != 19 or orcid.count("-") != 3):
        issues.append("ORCID is malformed; leave it empty or provide the real ORCID")
    if pending and args.allow_pending_metadata:
        print("WARNING_PENDING_AUTHOR_METADATA " + " | ".join(pending))
    else:
        issues.extend(pending)

    if not packaged:
        report_path = root / "reports" / "submission_manuscript_gate.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps({"manuscript_dir": str(manuscript), "issues": issues}, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
    if issues:
        print("SUBMISSION_MANUSCRIPT_FAIL")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)
    print(f"SUBMISSION_MANUSCRIPT_PASS layout={'package' if packaged else 'workspace'}")


if __name__ == "__main__":
    main()
