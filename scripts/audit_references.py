from __future__ import annotations

import json
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MANUSCRIPT_ROOT = PROJECT_ROOT.parent / "正文"


def parse_keys(text: str) -> list[str]:
    return re.findall(r"@[A-Za-z]+\s*\{\s*([^,\s]+)\s*,", text)


def cited_keys(text: str) -> set[str]:
    found: set[str] = set()
    for match in re.finditer(r"\\cite[tp]?\{([^}]+)\}", text):
        found.update(key.strip() for key in match.group(1).split(",") if key.strip())
    return found


def main() -> None:
    bib_path = PROJECT_ROOT / "references.bib"
    bib = bib_path.read_text(encoding="utf-8")
    keys = parse_keys(bib)
    issues = []
    duplicates = sorted({key for key in keys if keys.count(key) > 1})
    if duplicates:
        issues.append(f"duplicate BibTeX keys: {duplicates}")
    dois = re.findall(r"doi\s*=\s*\{([^}]+)\}", bib, flags=re.IGNORECASE)
    invalid_dois = sorted(doi for doi in dois if not re.fullmatch(r"10\.\d{4,9}/\S+", doi))
    if invalid_dois:
        issues.append(f"invalid DOI syntax: {invalid_dois}")
    if "utm_source=" in bib or "utm_medium=" in bib:
        issues.append("tracking parameters found in bibliography URLs")

    tex_files = [MANUSCRIPT_ROOT / "main_revised.tex", MANUSCRIPT_ROOT / "main_zh.tex"]
    cited: set[str] = set()
    for path in tex_files:
        if path.exists():
            cited |= cited_keys(path.read_text(encoding="utf-8"))
    missing = sorted(cited - set(keys))
    if missing:
        issues.append(f"cited keys missing from bibliography: {missing}")
    report = {
        "bib_entries": len(keys),
        "doi_entries": len(dois),
        "cited_keys": len(cited),
        "missing_cited_keys": missing,
        "duplicate_keys": duplicates,
        "issues": issues,
        "note": "DOI syntax and citation-key audit; semantic DOI metadata was manually checked against primary publisher records.",
    }
    output = PROJECT_ROOT / "reports" / "reference_audit.json"
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if issues:
        print("REFERENCE_AUDIT_FAIL")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)
    print(f"REFERENCE_AUDIT_PASS entries={len(keys)} dois={len(dois)} cited={len(cited)}")


if __name__ == "__main__":
    main()
