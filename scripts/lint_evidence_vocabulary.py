from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess


PROJECT_ROOT = Path(__file__).resolve().parents[1]
FORBIDDEN = (
    "A Controlled Accuracy--Risk Evaluation Framework",
    "Risk-Aware RUL Prediction with Controlled Sensor-Degradation Evaluation",
    "PRIMARY FROZEN",
    "PRIMARY LOCKED",
    "POST-FREEZE",
    "POST-LOCK",
    "confirmation experiment",
    "checkpoint-selection confirmation",
    "retrospective symmetric confirmation",
    "validation-endpoint confirmation",
    "confirmation interval",
    "confirmation test",
    "locked reference",
    "locked formal checkpoint",
    "locked composite distribution",
    "deterministic locked benchmark",
    "locally frozen",
    "frozen sensor-budget",
    "suggestive",
    "提示性",
    "95\\% CI",
    "95% CI",
    "Pr(left lower)",
    "Pr(Asym better)",
    "Sensor-fault evidence",
    "fault generator",
    "robustness superiority",
    "architecture attribution",
    "checkpoint incompatibility",
    "primary fixed-benchmark estimation",
    "nominal crossed intervals",
    "Holm decision",
    "CI: left lower",
    "CI: right lower",
    "OCM-RAST-GRU",
    "primary operating configuration",
    "pre-registered",
    "preregistered",
    "预注册",
)
TEXT_SUFFIXES = {
    ".cff",
    ".csv",
    ".json",
    ".log",
    ".md",
    ".ps1",
    ".py",
    ".sh",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}
PDF_SUFFIX = ".pdf"
CODE_OR_HISTORICAL_PARTS = {
    "benchmark",
    "configs",
    "eval_harness",
    "results",
    "scripts",
    "src",
    "tests",
}
PORTABILITY_PATTERNS = (
    ("replacement character", "\ufffd"),
    ("author Windows profile", "C:" + r"\Users\Ethan"),
    ("author Python installation", "C:" + r"\DevTools"),
    ("author workspace", "D:" + r"\论文"),
)
READER_ALLOWLIST = {
    "docs/evidence_vocabulary.md": (
        "controlled terminology dictionary listing rejected historical terms"
    ),
}
GENERATED_GATE_ARTIFACTS = {
    "evidence_vocabulary_gate.json",
}


def default_roots() -> list[Path]:
    manuscript_root = next(
        (
            candidate
            for candidate in PROJECT_ROOT.parent.iterdir()
            if candidate.is_dir() and (candidate / "main_revised.tex").exists()
        ),
        PROJECT_ROOT.parent / "正文",
    )
    return [
        PROJECT_ROOT / "paper_outputs" / "release_v1",
        PROJECT_ROOT / "paper_outputs" / "manuscript_v3",
        PROJECT_ROOT / "supplementary",
        PROJECT_ROOT / "CITATION.cff",
        PROJECT_ROOT / "README.md",
        PROJECT_ROOT / "README_REVISED.md",
        manuscript_root / "main_revised.tex",
        manuscript_root / "main_revised.pdf",
        manuscript_root / "main_zh.tex",
        manuscript_root / "main_zh.pdf",
    ]


def iter_files(root: Path):
    if root.is_file():
        yield root
    elif root.is_dir():
        yield from (path for path in root.rglob("*") if path.is_file())


def is_generated_gate_artifact(path: Path) -> bool:
    """Exclude gate output that necessarily quotes the vocabulary it audits."""
    return path.name in GENERATED_GATE_ARTIFACTS


def extract_pdf_text(path: Path, pdftotext: str) -> str:
    completed = subprocess.run(
        [pdftotext, str(path), "-"],
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(completed.stderr.strip() or "pdftotext failed")
    return completed.stdout


def allowlisted_occurrence(path: Path, root: Path, token: str) -> str | None:
    try:
        relative = path.relative_to(root)
    except ValueError:
        relative = path
    parts = set(relative.parts)
    relative_posix = relative.as_posix()
    for suffix, reason in READER_ALLOWLIST.items():
        if relative_posix.endswith(suffix):
            return reason
    if parts & CODE_OR_HISTORICAL_PARTS:
        return "implementation, test fixture, or stored-run provenance"
    if "paper_outputs" in parts and "release_v1" not in parts:
        return "historical or intermediate generated evidence outside the semantic release"
    return None


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Reject obsolete or overstated evidence terminology in visible release artifacts."
    )
    parser.add_argument("paths", nargs="*", type=Path)
    parser.add_argument(
        "--package-wide",
        action="store_true",
        help="Scan the complete delivery and record explicit code/historical allowlist hits.",
    )
    parser.add_argument(
        "--report",
        default=str(PROJECT_ROOT / "reports" / "evidence_vocabulary_gate.json"),
    )
    args = parser.parse_args()
    issues: list[str] = []
    checked = 0
    checked_pdfs = 0
    skipped_pdfs: list[str] = []
    allowlisted: list[dict[str, str]] = []
    seen: set[Path] = set()
    pdftotext = shutil.which("pdftotext")
    report_path = Path(args.report).resolve()
    roots = [root.resolve() for root in (args.paths or default_roots())]
    for root in roots:
        for path in iter_files(root):
            path = path.resolve()
            if path == report_path or is_generated_gate_artifact(path):
                continue
            if path in seen:
                continue
            seen.add(path)
            suffix = path.suffix.lower()
            if suffix not in TEXT_SUFFIXES and suffix != PDF_SUFFIX:
                continue
            if suffix == PDF_SUFFIX:
                # Referenced figure text is already embedded in the compiled
                # manuscript or Supplement PDF. Ignore unreferenced working
                # figure copies so this gate represents the reader-visible
                # dependency closure rather than a workspace-directory scan.
                if path.parent.name == "figures":
                    continue
                if not pdftotext:
                    skipped_pdfs.append(str(path))
                    continue
                try:
                    text = extract_pdf_text(path, pdftotext)
                except RuntimeError as exc:
                    issues.append(f"{path}: PDF text extraction failed: {exc}")
                    continue
                checked_pdfs += 1
            else:
                text = path.read_text(encoding="utf-8-sig", errors="replace")
            checked += 1
            folded = text.casefold()
            for token in FORBIDDEN:
                if token.casefold() in folded:
                    reason = (
                        allowlisted_occurrence(path, root, token)
                        if args.package_wide
                        else None
                    )
                    if reason:
                        allowlisted.append(
                            {
                                "path": path.relative_to(root).as_posix(),
                                "token": token,
                                "reason": reason,
                            }
                        )
                    else:
                        issues.append(f"{path}: obsolete token {token!r}")
            if args.package_wide:
                for label, token in PORTABILITY_PATTERNS:
                    if token in text:
                        issues.append(f"{path}: nonportable {label}")
    report = {
        "status": "PASS" if not issues else "FAIL",
        "checked_files": checked,
        "checked_pdfs": checked_pdfs,
        "pdf_extraction_available": bool(pdftotext),
        "skipped_pdfs": skipped_pdfs,
        "package_wide": args.package_wide,
        "allowlisted_occurrence_count": len(allowlisted),
        "allowlisted_occurrences": allowlisted[:500],
        "issues": issues,
    }
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if issues:
        print(f"EVIDENCE_VOCABULARY_FAIL files={checked} issues={len(issues)}")
        for issue in issues[:100]:
            print(f"- {issue}")
        raise SystemExit(1)
    suffix = "" if pdftotext else f" pdf_skipped={len(skipped_pdfs)}"
    print(
        f"EVIDENCE_VOCABULARY_PASS files={checked} "
        f"pdfs={checked_pdfs}{suffix}"
    )


if __name__ == "__main__":
    main()
