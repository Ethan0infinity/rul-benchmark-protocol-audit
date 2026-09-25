"""Run the single release-facing authority gate and record every sub-gate."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import re
import subprocess
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def workspace_manuscript_root() -> Path:
    for candidate in PROJECT_ROOT.parent.iterdir():
        if candidate.is_dir() and (candidate / "main_revised.tex").exists():
            return candidate
    raise FileNotFoundError(
        "Could not locate the workspace manuscript directory containing "
        "main_revised.tex."
    )


def console_safe(text: str) -> str:
    encoding = sys.stdout.encoding or "utf-8"
    return text.encode(encoding, errors="replace").decode(
        encoding,
        errors="replace",
    )


def portable_argument(value: str, cwd: Path) -> str:
    if value == sys.executable:
        return "python"
    candidate = Path(value)
    if candidate.is_absolute():
        return Path(os.path.relpath(candidate, cwd)).as_posix()
    return value.replace("\\", "/")


def sanitize_output(text: str, cwd: Path) -> str:
    sanitized = text
    roots = sorted(
        {
            str(cwd.resolve()),
            cwd.resolve().as_posix(),
            str(PROJECT_ROOT.resolve()),
            PROJECT_ROOT.resolve().as_posix(),
        },
        key=len,
        reverse=True,
    )
    for root in roots:
        sanitized = sanitized.replace(root, ".")
        sanitized = sanitized.replace(
            json.dumps(root, ensure_ascii=False)[1:-1],
            ".",
        )
    return sanitized


def run_command(
    name: str,
    command: list[str],
    *,
    cwd: Path,
) -> dict[str, object]:
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    completed = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    output = sanitize_output(
        (completed.stdout + completed.stderr).strip(),
        cwd,
    )
    print(console_safe(output))
    return {
        "name": name,
        "command": [portable_argument(value, cwd) for value in command],
        "cwd": ".",
        "exit_code": completed.returncode,
        "status": "PASS" if completed.returncode == 0 else "FAIL",
        "output_tail": output.splitlines()[-20:],
    }


def release_id_from(package_root: Path | None) -> str:
    candidates = []
    if package_root is not None:
        candidates.append(
            package_root / "manuscript" / "generated" / "release_sync.tex"
        )
    try:
        candidates.append(
            workspace_manuscript_root() / "generated" / "release_sync.tex"
        )
    except FileNotFoundError:
        pass
    for path in candidates:
        if not path.exists():
            continue
        match = re.search(
            r"\\EvidenceReleaseID\}\{([^}]+)\}",
            path.read_text(encoding="utf-8-sig"),
        )
        if match:
            return match.group(1)
    return ""


def git_commit() -> str:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=PROJECT_ROOT,
            text=True,
            capture_output=True,
            check=False,
        )
    except OSError:
        return ""
    return completed.stdout.strip() if completed.returncode == 0 else ""


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run all release-facing sub-gates through one authority."
    )
    parser.add_argument("--package-root", default="")
    parser.add_argument("--allow-pending-metadata", action="store_true")
    parser.add_argument(
        "--report",
        default=str(
            PROJECT_ROOT
            / "reports"
            / "release_authority_gate.json"
        ),
    )
    args = parser.parse_args()

    package_root = (
        Path(args.package_root).resolve()
        if args.package_root
        else None
    )
    delivery_profile = "workspace"
    if package_root is not None:
        profile_path = package_root / "delivery_profile.json"
        if profile_path.exists():
            delivery_profile = str(
                json.loads(
                    profile_path.read_text(encoding="utf-8-sig")
                ).get("profile", "unknown")
            )
    python = sys.executable
    if delivery_profile == "lightweight-release-only":
        commands: list[tuple[str, list[str], Path]] = [
            (
                "lightweight_package",
                [python, "scripts/verify_lightweight_package.py"],
                PROJECT_ROOT,
            ),
            (
                "lightweight_tests",
                [python, "scripts/run_lightweight_tests.py"],
                PROJECT_ROOT,
            ),
            (
                "release_evidence",
                [python, "scripts/release_evidence_gate.py"],
                PROJECT_ROOT,
            ),
        ]
    else:
        commands = [
            (
                "figure_data_consistency",
                [python, "scripts/check_figure_data_consistency.py"],
                PROJECT_ROOT,
            ),
            (
                "advanced_evidence",
                [python, "scripts/check_advanced_evidence.py"],
                PROJECT_ROOT,
            ),
            (
                "release_evidence",
                [python, "scripts/release_evidence_gate.py"],
                PROJECT_ROOT,
            ),
        ]
    tex_preflight_command = [
        python,
        "scripts/check_tex_source_build.py",
        "--preflight-only",
    ]
    if package_root is not None:
        tex_preflight_command.extend(["--package-root", str(package_root)])
    commands.insert(
        0,
        (
            "tex_toolchain_preflight",
            tex_preflight_command,
            PROJECT_ROOT,
        ),
    )
    if package_root is None:
        commands.insert(
            2,
            (
                "manuscript_evidence",
                [python, "scripts/check_manuscript_evidence.py"],
                PROJECT_ROOT,
            ),
        )
    if (PROJECT_ROOT / "results_manifest.csv").exists():
        commands.append(
            (
                "results_manifest",
                [
                    python,
                    "scripts/verify_results_manifest.py",
                    "--package-root",
                    ".",
                    "--expected-rows",
                    "260",
                ],
                PROJECT_ROOT,
            )
        )
    public_artifact_command = [
        python,
        "scripts/check_public_artifact.py",
    ]
    if args.allow_pending_metadata:
        public_artifact_command.append("--allow-pending-metadata")
    commands.append(
        (
            "public_artifact_metadata",
            public_artifact_command,
            PROJECT_ROOT,
        )
    )
    tex_source_command = [
        python,
        "scripts/check_tex_source_build.py",
    ]
    if package_root is not None:
        tex_source_command.extend(
            ["--package-root", str(package_root)]
        )
    commands.append(
        (
            "tex_source_build",
            tex_source_command,
            PROJECT_ROOT,
        )
    )

    if package_root is None:
        manuscript_root = workspace_manuscript_root()
        vocabulary_paths = [
            str(PROJECT_ROOT / "paper_outputs" / "release_v1"),
            str(PROJECT_ROOT / "paper_outputs" / "manuscript_v3"),
            str(PROJECT_ROOT / "supplementary"),
            str(manuscript_root / "main_revised.tex"),
            str(manuscript_root / "main_revised.pdf"),
            str(manuscript_root / "main_zh.tex"),
            str(manuscript_root / "main_zh.pdf"),
            str(PROJECT_ROOT / "README_REVISED.md"),
        ]
        manuscript_command = [
            python,
            "scripts/check_submission_manuscripts.py",
        ]
        cross_command = [
            python,
            "scripts/check_cross_document_sync.py",
        ]
        exhibit_command = [
            python,
            "scripts/check_exhibit_navigation.py",
        ]
    else:
        vocabulary_paths = [str(package_root)]
        manuscript_command = [
            python,
            "scripts/check_submission_manuscripts.py",
            "--package-root",
            str(package_root),
        ]
        cross_command = [
            python,
            "scripts/check_cross_document_sync.py",
            "--package-root",
            str(package_root),
        ]
        exhibit_command = [
            python,
            "scripts/check_exhibit_navigation.py",
            "--package-root",
            str(package_root),
        ]
    if args.allow_pending_metadata:
        manuscript_command.append("--allow-pending-metadata")
    commands.extend(
        [
            (
                "package_closure",
                [
                    python,
                    "scripts/check_package_closure.py",
                    "--package-root",
                    str(package_root or PROJECT_ROOT),
                ],
                PROJECT_ROOT,
            ),
            (
                "evidence_vocabulary",
                [
                    python,
                    "scripts/lint_evidence_vocabulary.py",
                    *(["--package-wide"] if package_root is not None else []),
                    *vocabulary_paths,
                ],
                PROJECT_ROOT,
            ),
            ("submission_manuscript", manuscript_command, PROJECT_ROOT),
            ("cross_document_sync", cross_command, PROJECT_ROOT),
            ("exhibit_navigation", exhibit_command, PROJECT_ROOT),
        ]
    )

    records = [
        run_command(name, command, cwd=cwd)
        for name, command, cwd in commands
    ]
    issues = [
        record["name"]
        for record in records
        if record["status"] != "PASS"
    ]
    report = {
        "schema_version": "1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "release_id": release_id_from(package_root),
        "code_commit": git_commit(),
        "package_root": (
            Path(os.path.relpath(package_root, PROJECT_ROOT)).as_posix()
            if package_root
            else "."
        ),
        "delivery_profile": delivery_profile,
        "archive_sha256": None,
        "metadata_mode": (
            "pending-author-local-audit"
            if args.allow_pending_metadata
            else "strict-public-release"
        ),
        "public_release_ready": (
            not args.allow_pending_metadata and not issues
        ),
        "subgates": records,
        "issues": issues,
        "status": "PASS" if not issues else "FAIL",
    }
    report_path = Path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if issues:
        print("RELEASE_AUTHORITY_FAIL " + ", ".join(issues))
        raise SystemExit(1)
    print(
        f"RELEASE_AUTHORITY_PASS subgates={len(records)} "
        f"release_id={report['release_id']}"
    )


if __name__ == "__main__":
    main()
