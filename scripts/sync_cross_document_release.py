"""Write one release identity into main and supplementary generated sources."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent
MAIN_ROOT = next(
    (
        candidate
        for candidate in WORKSPACE_ROOT.iterdir()
        if candidate.is_dir() and (candidate / "main_revised.tex").exists()
    ),
    WORKSPACE_ROOT / "正文",
)
SUPPLEMENT_ROOT = PROJECT_ROOT / "supplementary"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Synchronize main/supplement release identity.")
    parser.add_argument("--release-id", default="2026-08-09-r20")
    args = parser.parse_args()

    content = "\n".join(
        [
            "% Generated cross-document release identity.",
            rf"\newcommand{{\EvidenceReleaseID}}{{{args.release_id}}}",
            r"\newcommand{\EvidenceProtocolVersion}{5.1}",
            "",
        ]
    )
    targets = [
        MAIN_ROOT / "generated" / "release_sync.tex",
        SUPPLEMENT_ROOT / "generated" / "release_sync.tex",
    ]
    for target in targets:
        target.parent.mkdir(parents=True, exist_ok=True)
        current = (
            target.read_text(encoding="utf-8-sig")
            if target.exists()
            else None
        )
        if current != content:
            target.write_text(content, encoding="utf-8", newline="\n")

    report = {
        "release_id": args.release_id,
        "protocol_version": "5.1",
        "main_source_sha256": sha256(MAIN_ROOT / "main_revised.tex"),
        "main_zh_source_sha256": sha256(MAIN_ROOT / "main_zh.tex"),
        "supplement_source_sha256": sha256(SUPPLEMENT_ROOT / "supplement_en.tex"),
        "supplement_zh_source_sha256": sha256(SUPPLEMENT_ROOT / "supplement_zh.tex"),
        "sync_file_sha256": sha256(targets[0]),
    }
    report_path = PROJECT_ROOT / "reports" / "cross_document_release.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        f"CROSS_DOCUMENT_RELEASE_READY release_id={args.release_id} "
        f"sha256={report['sync_file_sha256']}"
    )


if __name__ == "__main__":
    main()
