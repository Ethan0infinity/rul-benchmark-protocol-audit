from __future__ import annotations

import argparse
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANUSCRIPT = PROJECT_ROOT.parent / "正文"


def collapse_results_fallback(path: Path, generated_name: str, discussion_heading: str) -> str:
    text = path.read_text(encoding="utf-8-sig")
    start_token = rf"\IfFileExists{{generated/{generated_name}}}{{%"
    discussion_token = rf"\section{{{discussion_heading}}}"
    start = text.find(start_token)
    if start < 0:
        if rf"\input{{generated/{generated_name}}}" in text:
            return text
        raise ValueError(f"Missing generated-results switch in {path}")
    discussion = text.find(discussion_token, start)
    if discussion < 0:
        raise ValueError(f"Missing discussion section after result fallback in {path}")
    close = text.rfind("}\n", start, discussion)
    if close < 0:
        raise ValueError(f"Could not locate fallback closing brace in {path}")
    replacement = rf"\input{{generated/{generated_name}}}" + "\n\n"
    return text[:start] + replacement + text[discussion:]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Remove obsolete embedded result fallbacks after protocol-v3 generated evidence exists."
    )
    parser.add_argument("--manuscript-dir", default=str(DEFAULT_MANUSCRIPT))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    manuscript = Path(args.manuscript_dir)
    targets = [
        (manuscript / "main_revised.tex", "results_section_en.tex", "Discussion"),
        (manuscript / "main_zh.tex", "results_section_zh.tex", "讨论"),
    ]
    for source, generated, discussion in targets:
        generated_path = manuscript / "generated" / generated
        if not generated_path.exists():
            raise FileNotFoundError(f"Refusing to finalize without generated evidence: {generated_path}")
        updated = collapse_results_fallback(source, generated, discussion)
        print(f"[FINALIZE] {source} old_chars={len(source.read_text(encoding='utf-8-sig'))} new_chars={len(updated)}")
        if not args.dry_run:
            source.write_text(updated, encoding="utf-8", newline="\n")
    print("MANUSCRIPT_SOURCE_FINALIZE_DRY_RUN_PASS" if args.dry_run else "MANUSCRIPT_SOURCE_FINALIZE_PASS")


if __name__ == "__main__":
    main()
