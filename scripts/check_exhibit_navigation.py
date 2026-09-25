"""Verify stable LaTeX labels named in the replication exhibit map."""

from __future__ import annotations

import argparse
from pathlib import Path
import re


PROJECT_ROOT = Path(__file__).resolve().parents[1]
LABEL_PATTERN = re.compile(r"`((?:tab|fig):[A-Za-z0-9_.:-]+)`")


def workspace_manuscript_root() -> Path:
    for candidate in PROJECT_ROOT.parent.iterdir():
        if candidate.is_dir() and (candidate / "main_revised.tex").exists():
            return candidate
    raise FileNotFoundError("Workspace manuscript directory was not found.")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Check that exhibit-map LaTeX labels resolve in the delivery."
    )
    parser.add_argument("--package-root", default="")
    args = parser.parse_args()

    package_root = Path(args.package_root).resolve() if args.package_root else None
    readme_candidates = [
        PROJECT_ROOT / "README_REPLICATION.md",
        PROJECT_ROOT / "replication_package" / "README_REPLICATION.md",
    ]
    readme = next((path for path in readme_candidates if path.exists()), None)
    if readme is None:
        raise FileNotFoundError("Replication README was not found.")

    if package_root is None:
        roots = [workspace_manuscript_root(), PROJECT_ROOT / "supplementary"]
    else:
        roots = [
            package_root / "manuscript",
            package_root / "supplementary" / "manuscript",
        ]
    tex_text = "\n".join(
        path.read_text(encoding="utf-8-sig", errors="replace")
        for root in roots
        if root.exists()
        for path in sorted(root.rglob("*.tex"))
    )
    labels = sorted(set(LABEL_PATTERN.findall(readme.read_text(encoding="utf-8-sig"))))
    if not labels:
        raise ValueError("Replication README contains no stable exhibit labels.")
    missing = [label for label in labels if f"\\label{{{label}}}" not in tex_text]
    if missing:
        raise ValueError(f"Unresolved exhibit-map labels: {missing}")
    print(f"EXHIBIT_NAVIGATION_PASS labels={len(labels)}")


if __name__ == "__main__":
    main()
