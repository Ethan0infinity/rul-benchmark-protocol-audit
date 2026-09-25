"""Run release checks without mutating an immutable submission deposit."""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile


def package_root_from_script() -> Path:
    direct = Path(__file__).resolve().parent
    if (direct / "supplementary" / "replication_package").is_dir():
        return direct
    raise FileNotFoundError(
        "This launcher must run from a submission-package delivery root."
    )


def checksum_issues(package_root: Path) -> list[str]:
    authority = package_root / "checksums.sha256"
    if not authority.exists():
        return ["missing checksums.sha256"]
    issues: list[str] = []
    for line_number, line in enumerate(
        authority.read_text(encoding="utf-8-sig").splitlines(), start=1
    ):
        if not line.strip():
            continue
        try:
            expected, relative = line.split("  ", 1)
        except ValueError:
            issues.append(f"invalid checksum line {line_number}")
            continue
        path = package_root / Path(relative)
        if not path.is_file():
            issues.append(f"missing: {relative}")
            continue
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            issues.append(f"mismatch: {relative}")
    return issues


def run_gate(
    package_root: Path,
    *,
    allow_pending_metadata: bool,
    report: Path,
    runtime_root: Path,
) -> int:
    replication = package_root / "supplementary" / "replication_package"
    command = [
        sys.executable,
        "scripts/release_authority_gate.py",
        "--package-root",
        str(package_root),
        "--report",
        str(report),
    ]
    if allow_pending_metadata:
        command.append("--allow-pending-metadata")
    runtime_root.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env.update(
        {
            "PYTHONUTF8": "1",
            "PYTHONIOENCODING": "utf-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "MPLCONFIGDIR": str(runtime_root / "matplotlib"),
            "XDG_CACHE_HOME": str(runtime_root / "xdg"),
            "PYTEST_ADDOPTS": (
                f"-p no:cacheprovider --basetemp={runtime_root / 'pytest'}"
            ),
        }
    )
    completed = subprocess.run(command, cwd=replication, env=env, check=False)
    return int(completed.returncode)


def copy_reports(package_root: Path, destination: Path, authority_report: Path) -> None:
    destination.mkdir(parents=True, exist_ok=True)
    if authority_report.exists():
        shutil.copy2(authority_report, destination / "release_authority_gate.json")
    source = package_root / "supplementary" / "replication_package" / "reports"
    if source.exists():
        shutil.copytree(source, destination / "replication_reports", dirs_exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Verify a submission package in a disposable copy by default. "
            "Use --in-place only while constructing a package before checksums exist."
        )
    )
    parser.add_argument("--allow-pending-metadata", action="store_true")
    parser.add_argument("--in-place", action="store_true")
    parser.add_argument(
        "--output-dir",
        default="",
        help="External directory for verification reports; defaults beside the deposit.",
    )
    parser.add_argument(
        "--workspace",
        default="",
        help="Optional new/empty external workspace retained after verification.",
    )
    args = parser.parse_args()

    deposit = package_root_from_script()
    if args.in_place:
        report = (
            deposit
            / "supplementary"
            / "replication_package"
            / "reports"
            / "release_authority_gate.json"
        )
        with tempfile.TemporaryDirectory(prefix="rul_release_build_") as runtime:
            return_code = run_gate(
                deposit,
                allow_pending_metadata=args.allow_pending_metadata,
                report=report,
                runtime_root=Path(runtime),
            )
        raise SystemExit(return_code)

    before = checksum_issues(deposit)
    if before:
        raise SystemExit("DEPOSIT_CHECKSUM_FAIL before verification: " + "; ".join(before[:10]))
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    output = (
        Path(args.output_dir).resolve()
        if args.output_dir
        else deposit.parent / f"{deposit.name}_verification_{timestamp}"
    )
    if output.exists():
        raise SystemExit(f"--output-dir must not already exist: {output}")

    temporary: tempfile.TemporaryDirectory[str] | None = None
    if args.workspace:
        workspace = Path(args.workspace).resolve()
        if workspace.exists() and any(workspace.iterdir()):
            raise SystemExit(f"--workspace must be new or empty: {workspace}")
        workspace.mkdir(parents=True, exist_ok=True)
    else:
        temporary = tempfile.TemporaryDirectory(prefix="rul_release_verify_")
        workspace = Path(temporary.name)
    working_package = workspace / "package"
    shutil.copytree(deposit, working_package)
    report = workspace / "reports" / "release_authority_gate.json"
    return_code = run_gate(
        working_package,
        allow_pending_metadata=args.allow_pending_metadata,
        report=report,
        runtime_root=workspace / "runtime",
    )
    copy_reports(working_package, output, report)
    after = checksum_issues(deposit)
    audit = {
        "schema_version": "1.0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "mode": "disposable-copy",
        "deposit": str(deposit),
        "working_copy": str(working_package) if args.workspace else "temporary",
        "gate_exit_code": return_code,
        "deposit_checksum_before": "PASS",
        "deposit_checksum_after": "PASS" if not after else "FAIL",
        "deposit_checksum_issues_after": after,
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "immutable_deposit_verification.json").write_text(
        json.dumps(audit, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    if temporary is not None:
        temporary.cleanup()
    if after:
        raise SystemExit("DEPOSIT_MUTATION_DETECTED: " + "; ".join(after[:10]))
    if return_code:
        raise SystemExit(return_code)
    print(f"RELEASE_AUTHORITY_IMMUTABLE_PASS reports={output}")


if __name__ == "__main__":
    main()
