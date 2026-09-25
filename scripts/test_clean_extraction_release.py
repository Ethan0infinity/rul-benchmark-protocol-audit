"""Exercise a submission ZIP from a clean directory without source-workspace imports."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import zipfile


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent


def run(command: list[str], cwd: Path) -> dict[str, object]:
    env = os.environ.copy()
    env["PYTHONPATH"] = ""
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
    output = (completed.stdout + completed.stderr).strip()
    return {
        "command": [
            "python" if value == sys.executable else value.replace("\\", "/")
            for value in command
        ],
        "cwd": ".",
        "exit_code": completed.returncode,
        "output_tail": output.splitlines()[-30:],
    }


def pending_metadata(package_root: Path) -> list[str]:
    replication = (
        package_root / "supplementary" / "replication_package"
    )
    artifact = json.loads(
        (replication / "artifact_metadata.json").read_text(
            encoding="utf-8-sig"
        )
    )
    author = json.loads(
        (replication / "author_metadata.json").read_text(
            encoding="utf-8-sig"
        )
    )
    pending = []
    artifact_fields = {
        "repository_url": artifact.get("repository_url"),
        "immutable_tag_or_commit": artifact.get(
            "immutable_tag_or_commit"
        ),
        "archive_url_or_doi": artifact.get("archive_url_or_doi"),
    }
    for name, value in artifact_fields.items():
        if not str(value or "").strip():
            pending.append(name)
    for name in ("affiliation", "institution"):
        if not str(author.get(name) or "").strip():
            pending.append(name)
    if author.get("corresponding_author") not in (True, False):
        pending.append("corresponding_author")
    return pending


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Extract a release ZIP into a temporary directory and verify "
            "both pending-metadata and strict-public modes."
        )
    )
    parser.add_argument(
        "--archive",
        default=str(
            WORKSPACE_ROOT
            / "cmapss_benchmark_governance_full_submission.zip"
        ),
    )
    parser.add_argument(
        "--report",
        default=str(
            PROJECT_ROOT
            / "reports"
            / "clean_extraction_release_test.json"
        ),
    )
    parser.add_argument(
        "--expect-strict-pass",
        action="store_true",
        help="Require complete public and author metadata.",
    )
    parser.add_argument(
        "--pipeline-mode",
        choices=("none", "quick", "full"),
        default="none",
        help="Optionally run the stored-results route inside the temporary extraction.",
    )
    args = parser.parse_args()

    archive = Path(args.archive).resolve()
    if not archive.exists():
        raise FileNotFoundError(f"Submission archive not found: {archive}")

    with tempfile.TemporaryDirectory(
        prefix="cmapss_release_audit_"
    ) as temporary:
        extracted = Path(temporary) / "delivery"
        extracted.mkdir()
        with zipfile.ZipFile(archive) as handle:
            handle.extractall(extracted)

        local = run(
            [
                sys.executable,
                "verify_submission_package.py",
                ".",
                "--allow-pending-metadata",
            ],
            extracted,
        )
        pending = pending_metadata(extracted)
        strict = run(
            [
                sys.executable,
                "verify_submission_package.py",
                ".",
            ],
            extracted,
        )
        authority = run(
            [
                sys.executable,
                "run_release_authority.py",
                "--allow-pending-metadata",
            ],
            extracted,
        )
        pipeline: dict[str, object] | None = None
        if args.pipeline_mode != "none":
            replication = (
                extracted / "supplementary" / "replication_package"
            )
            pipeline_command = [
                sys.executable,
                "scripts/run_stored_results_pipeline.py",
                "--heartbeat-seconds",
                "30",
                "--log-dir",
                f"reports/clean_extraction_{args.pipeline_mode}",
            ]
            if args.pipeline_mode == "quick":
                pipeline_command.append("--quick")
            pipeline = run(pipeline_command, replication)

    issues = []
    if local["exit_code"] != 0:
        issues.append("pending-metadata package verification failed")
    if authority["exit_code"] != 0:
        issues.append("clean-extraction release authority failed")
    if pipeline is not None and pipeline["exit_code"] != 0:
        issues.append(
            f"clean-extraction {args.pipeline_mode} stored-results route failed"
        )
    if args.expect_strict_pass:
        if pending:
            issues.append(
                "strict pass requested with pending fields: "
                + ", ".join(pending)
            )
        if strict["exit_code"] != 0:
            issues.append("strict public verification failed")
    elif pending and strict["exit_code"] == 0:
        issues.append(
            "strict public verification produced a false positive"
        )
    elif not pending and strict["exit_code"] != 0:
        issues.append(
            "strict public verification failed despite complete metadata"
        )

    report = {
        "schema_version": "1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "archive": archive.name,
        "temporary_extraction": True,
        "source_workspace_pythonpath_disabled": True,
        "pending_fields": pending,
        "pending_mode": local,
        "strict_mode": strict,
        "release_authority": authority,
        "stored_results_pipeline": pipeline,
        "status": "PASS" if not issues else "FAIL",
        "issues": issues,
    }
    output = Path(args.report)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if issues:
        for issue in issues:
            print(f"ERROR {issue}")
        raise SystemExit(1)
    strict_status = (
        "PASS"
        if strict["exit_code"] == 0
        else "EXPECTED_FAIL_PENDING_METADATA"
    )
    print(
        "CLEAN_EXTRACTION_RELEASE_PASS "
        f"pending_mode=PASS strict_mode={strict_status}"
    )


if __name__ == "__main__":
    main()
