"""Stage-aware stored-results regeneration with logs and resume support."""

from __future__ import annotations

import argparse
import csv
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import threading
import time
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PIPELINE_VERSION = "release-5.0"
CONTRACT_SCHEMA_VERSION = "2.0"

try:
    import psutil
except ImportError:  # pragma: no cover - optional monitoring dependency
    psutil = None


@dataclass(frozen=True)
class Stage:
    name: str
    command: tuple[str, ...]
    quick: bool = False
    predicate: Callable[[], bool] | None = None
    skip_reason: str = ""


def python_command(*arguments: str) -> tuple[str, ...]:
    return (sys.executable, *arguments)


def stages() -> list[Stage]:
    outer_checksum = PROJECT_ROOT.parents[1] / "checksums.sha256"
    ncmapss_cache = PROJECT_ROOT / "data" / "external" / "processed" / "ncmapss_ds02_smp100_win50.npz"
    return [
        Stage(
            "verify_deposit_checksum",
            python_command("scripts/verify_artifact_checksums.py"),
            quick=True,
            predicate=outer_checksum.exists,
            skip_reason="not nested inside a checksum-audited submission delivery",
        ),
        Stage(
            "verify_results_manifest",
            python_command(
                "scripts/verify_results_manifest.py",
                "--package-root",
                ".",
                "--expected-rows",
                "260",
            ),
            quick=True,
            predicate=(PROJECT_ROOT / "results_manifest.csv").exists,
            skip_reason="source workspace has no packaged results_manifest.csv",
        ),
        Stage(
            "benchmark_readiness",
            python_command("scripts/check_benchmark_readiness.py"),
            quick=True,
        ),
        Stage(
            "paper_tables",
            python_command(
                "scripts/build_paper_tables.py",
                "--experiment-prefix",
                "paper_main_v3_seed",
            ),
            quick=True,
        ),
        Stage(
            "paired_statistics",
            python_command(
                "scripts/run_statistical_tests.py",
                "--experiment-prefix",
                "paper_main_v3_seed",
                "--baseline-models",
                "gru",
                "lstm",
                "tcn",
                "tcn_gru",
                "rast_gru",
                "cnn_lstm",
                "attention_gru",
                "bigru_attention",
                "transformer_lite",
                "dual_attention_tcn",
                "sensor_graph_gru",
                "quantile_gru",
            ),
            quick=True,
        ),
        Stage(
            "ncmapss_cache",
            python_command("scripts/validate_ncmapss_cache.py"),
            predicate=ncmapss_cache.exists,
            skip_reason="redistribution-reviewed N-CMAPSS cache is absent",
        ),
        Stage(
            "checkpoint_integrity",
            python_command("scripts/check_checkpoint_integrity.py"),
        ),
        Stage(
            "ablation_consistency",
            python_command("scripts/check_ablation_consistency.py"),
        ),
        Stage(
            "ncmapss_readiness",
            python_command("scripts/check_ncmapss_readiness.py"),
        ),
        Stage(
            "fixed_benchmark_statistics",
            python_command(
                "scripts/analyze_release_statistics.py",
                "--max-t-calibration-simulations",
                "1000",
                "--max-t-calibration-bootstrap-reps",
                "999",
            ),
        ),
        Stage(
            "preference_selection_stability",
            python_command("scripts/analyze_preference_selection_stability.py"),
        ),
        Stage(
            "no_overlap_augmentation",
            python_command(
                "scripts/analyze_augmentation_distribution_sensitivity.py",
                "--reuse-stress",
            ),
        ),
        Stage(
            "retrospective_retraining_analysis",
            python_command("scripts/analyze_round12_new_experiments.py"),
            quick=True,
        ),
        Stage(
            "retrospective_retraining_check",
            python_command("scripts/check_round12_new_evidence.py"),
            quick=True,
        ),
        Stage(
            "retrospective_retraining_assets",
            python_command("scripts/build_round12_new_evidence_assets.py"),
            quick=True,
        ),
        Stage(
            "revised_assets",
            python_command("scripts/build_revised_assets.py"),
        ),
        Stage(
            "advanced_evidence",
            python_command("scripts/build_advanced_evidence.py", "--bootstrap-reps", "5000"),
        ),
        Stage(
            "advanced_evidence_check",
            python_command("scripts/check_advanced_evidence.py"),
        ),
        Stage(
            "manuscript_evidence",
            python_command("scripts/build_manuscript_evidence.py"),
        ),
        Stage(
            "manuscript_evidence_check",
            python_command("scripts/check_manuscript_evidence.py"),
        ),
        Stage(
            "release_evidence",
            python_command("scripts/build_release_evidence.py"),
        ),
        Stage(
            "release_evidence_gate",
            python_command("scripts/release_evidence_gate.py"),
        ),
        Stage(
            "evidence_vocabulary",
            python_command("scripts/lint_evidence_vocabulary.py"),
        ),
        Stage(
            "post_regeneration_digest",
            python_command("scripts/build_post_regeneration_digest.py"),
        ),
        Stage(
            "package_closure",
            python_command("scripts/check_package_closure.py", "--package-root", "."),
        ),
        Stage(
            "final_quality_gate",
            python_command(
                "scripts/paper_quality_gate.py",
                "--stage",
                "final",
                "--experiment-prefix",
                "paper_main_v3_seed",
                "--expected-count",
                "260",
                "--skip-data-validation",
                "--skip-public-artifact",
                "--skip-submission-manuscript",
            ),
        ),
    ]


def portable_command(command: list[str]) -> list[str]:
    """Remove host-specific interpreter paths from persisted provenance."""
    if not command:
        return []
    normalized = list(command)
    if Path(normalized[0]).name.lower() in {"python", "python.exe"}:
        normalized[0] = "python"
    return normalized


def command_fingerprint(stage: Stage) -> str:
    payload = json.dumps(
        {
            "pipeline_version": PIPELINE_VERSION,
            "name": stage.name,
            "command": portable_command(stage.command),
        },
        sort_keys=True,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def write_state(path: Path, state: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(state, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def execution_context() -> dict[str, object]:
    delivery_root = PROJECT_ROOT.parents[1]
    checksum_authority = delivery_root / "checksums.sha256"
    metadata_path = PROJECT_ROOT / "artifact_metadata.json"
    metadata = (
        json.loads(metadata_path.read_text(encoding="utf-8-sig"))
        if metadata_path.exists()
        else {}
    )
    distributed = checksum_authority.exists() and (
        PROJECT_ROOT / "results_manifest.csv"
    ).exists()
    return {
        "kind": (
            "distributed-package-workspace" if distributed else "author-workspace"
        ),
        "project_root": "." if distributed else "author-workspace",
        "checksum_authority": (
            checksum_authority.relative_to(delivery_root).as_posix()
            if distributed
            else "absent"
        ),
        "checksum_authority_sha256": (
            hashlib.sha256(checksum_authority.read_bytes()).hexdigest()
            if checksum_authority.exists()
            else ""
        ),
        "release_id": str(metadata.get("release_id", "")),
        "semantic_version": str(metadata.get("semantic_version", "")),
        "pipeline_commit": str(metadata.get("commit_hash", "")),
        "ncmapss_cache_distribution": str(
            metadata.get("ncmapss_cache_distribution", "unknown")
        ),
    }


def archive_stale_runtime_authority(log_dir: Path) -> None:
    authority = log_dir / "runtime_contract_full.json"
    if not authority.exists():
        return
    try:
        old_version = str(
            json.loads(authority.read_text(encoding="utf-8-sig")).get(
                "pipeline_version", "unknown"
            )
        )
    except (OSError, json.JSONDecodeError):
        old_version = "unknown"
    if old_version == PIPELINE_VERSION:
        return
    historical = log_dir / "historical" / old_version
    historical.mkdir(parents=True, exist_ok=True)
    for name in (
        "runtime_contract.json",
        "runtime_contract.csv",
        "runtime_contract_full.json",
        "runtime_contract_full.csv",
        "runtime_contract_quick.json",
        "runtime_contract_quick.csv",
        "pipeline_state.json",
    ):
        source = log_dir / name
        if source.exists():
            destination = historical / name
            if destination.exists():
                destination = historical / f"previous_{name}"
            shutil.move(str(source), str(destination))


def archive_runtime_authority_after_checksum(
    stage_name: str,
    status: str,
    log_dir: Path,
) -> None:
    if stage_name == "verify_deposit_checksum" and status in {"PASS", "SKIP"}:
        archive_stale_runtime_authority(log_dir)


def monitor_peak_rss(process: subprocess.Popen[str], stop: threading.Event, peak: list[int]) -> None:
    if psutil is None:
        return
    root = psutil.Process(process.pid)
    while not stop.wait(0.2):
        try:
            processes = [root, *root.children(recursive=True)]
            total = sum(item.memory_info().rss for item in processes if item.is_running())
            peak[0] = max(peak[0], total)
        except (psutil.Error, ProcessLookupError):
            break


def heartbeat_stage(
    stage: Stage,
    process: subprocess.Popen[str],
    stop: threading.Event,
    start: float,
    peak: list[int],
    log_path: Path,
    interval_seconds: float,
) -> None:
    while not stop.wait(interval_seconds):
        if process.poll() is not None:
            return
        elapsed = time.perf_counter() - start
        peak_mb = peak[0] / (1024**2) if peak[0] else None
        log_bytes = log_path.stat().st_size if log_path.exists() else 0
        print(
            f"[STAGE HEARTBEAT] {stage.name} elapsed={elapsed:.1f}s "
            f"peak_rss_mb={peak_mb} log_bytes={log_bytes}",
            flush=True,
        )


def run_stage(
    stage: Stage,
    log_path: Path,
    *,
    heartbeat_seconds: float,
) -> dict[str, object]:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    started = datetime.now(timezone.utc)
    start = time.perf_counter()
    peak = [0]
    stop = threading.Event()
    env = os.environ.copy()
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    with log_path.open("w", encoding="utf-8", newline="\n") as log:
        log.write(f"stage={stage.name}\n")
        log.write(f"command={' '.join(portable_command(stage.command))}\n")
        log.flush()
        process = subprocess.Popen(
            stage.command,
            cwd=PROJECT_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        )
        monitor = threading.Thread(
            target=monitor_peak_rss,
            args=(process, stop, peak),
            daemon=True,
        )
        monitor.start()
        heartbeat = threading.Thread(
            target=heartbeat_stage,
            args=(
                stage,
                process,
                stop,
                start,
                peak,
                log_path,
                heartbeat_seconds,
            ),
            daemon=True,
        )
        heartbeat.start()
        assert process.stdout is not None
        for line in process.stdout:
            console_encoding = sys.stdout.encoding or "utf-8"
            display = line.encode(console_encoding, errors="replace").decode(
                console_encoding,
                errors="replace",
            )
            print(display, end="", flush=True)
            log.write(line)
        return_code = process.wait()
        stop.set()
        monitor.join(timeout=1.0)
        heartbeat.join(timeout=1.0)
    elapsed = time.perf_counter() - start
    return {
        "name": stage.name,
        "command": portable_command(stage.command),
        "command_fingerprint": command_fingerprint(stage),
        "status": "PASS" if return_code == 0 else "FAIL",
        "exit_code": return_code,
        "started_utc": started.isoformat(),
        "elapsed_seconds": elapsed,
        "peak_rss_mb": peak[0] / (1024**2) if peak[0] else None,
        "log": log_path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix(),
    }


def write_runtime_contract(
    log_dir: Path,
    records: list[dict[str, object]],
    *,
    mode: str,
    expected_stage_count: int,
) -> None:
    generated = datetime.now(timezone.utc)
    run_id = generated.strftime("%Y%m%dT%H%M%SZ")
    contract = {
        "schema_version": CONTRACT_SCHEMA_VERSION,
        "pipeline_version": PIPELINE_VERSION,
        "generated_utc": generated.isoformat(),
        "run_id": run_id,
        "mode": mode,
        "overall_status": (
            "PASS"
            if len(records) == expected_stage_count
            and all(row.get("status") in {"PASS", "SKIP"} for row in records)
            else "FAIL"
        ),
        "stage_count": len(records),
        "expected_stage_count": expected_stage_count,
        "execution_context": execution_context(),
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "python": sys.version.split()[0],
            "logical_cpus": os.cpu_count(),
            "psutil_monitoring": psutil is not None,
        },
        "total_elapsed_seconds": sum(
            float(row.get("elapsed_seconds") or 0.0) for row in records
        ),
        "peak_stage_rss_mb": max(
            (float(row.get("peak_rss_mb") or 0.0) for row in records),
            default=0.0,
        ),
        "stages": records,
        "resume_contract": (
            "--resume skips only stages recorded PASS under the same pipeline version "
            "and command fingerprint; --from-stage requires all earlier stages to be "
            "PASS or predicate-satisfied SKIP in the state file"
        ),
        "cache_contract": (
            "pipeline_state.json is the stage cache. A cached PASS is reusable only "
            "when pipeline version and command fingerprint match."
        ),
        "provenance_scope": (
            "This contract records the environment that produced this run. A deposited "
            "contract is not evidence that a third party reproduced the route."
        ),
    }
    contract_json = (
        json.dumps(contract, indent=2, ensure_ascii=False) + "\n"
    )
    mode_json = log_dir / f"runtime_contract_{mode}.json"
    mode_json.write_text(contract_json, encoding="utf-8")
    fields = [
        "name",
        "status",
        "exit_code",
        "elapsed_seconds",
        "peak_rss_mb",
        "started_utc",
        "log",
    ]
    mode_csv = log_dir / f"runtime_contract_{mode}.csv"
    with mode_csv.open(
        "w",
        encoding="utf-8",
        newline="",
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)
    archive_dir = log_dir / "contracts"
    archive_dir.mkdir(parents=True, exist_ok=True)
    (archive_dir / f"{run_id}_{mode}.json").write_text(
        contract_json,
        encoding="utf-8",
    )
    shutil.copy2(
        mode_csv,
        archive_dir / f"{run_id}_{mode}.csv",
    )
    context_kind = str(contract["execution_context"]["kind"])
    context_json = log_dir / f"runtime_contract_{context_kind}_{mode}.json"
    context_csv = log_dir / f"runtime_contract_{context_kind}_{mode}.csv"
    context_json.write_text(contract_json, encoding="utf-8")
    shutil.copy2(mode_csv, context_csv)
    compatibility_json = log_dir / "runtime_contract.json"
    compatibility_csv = log_dir / "runtime_contract.csv"
    if mode == "full" or not compatibility_json.exists():
        shutil.copy2(mode_json, compatibility_json)
        shutil.copy2(mode_csv, compatibility_csv)


def normalize_log_dir(value: str | Path) -> Path:
    path = Path(value)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    try:
        path.relative_to(PROJECT_ROOT.resolve())
    except ValueError as exc:
        raise ValueError(
            f"--log-dir must resolve inside the project root: {path}"
        ) from exc
    return path


def prerequisite_satisfied(
    stage: Stage,
    prior: dict[str, object] | None,
) -> bool:
    prior = prior or {}
    if prior.get("status") == "PASS":
        return True
    return bool(
        prior.get("status") == "SKIP"
        and stage.predicate is not None
        and not stage.predicate()
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the stored-results pipeline with stage logs and resume support."
    )
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--from-stage", default="")
    parser.add_argument("--list-stages", action="store_true")
    parser.add_argument(
        "--log-dir",
        default=str(PROJECT_ROOT / "reports" / "stored_results_pipeline"),
    )
    parser.add_argument(
        "--heartbeat-seconds",
        type=float,
        default=15.0,
        help="Print progress for a silent running stage at this interval.",
    )
    args = parser.parse_args()

    declared = stages()
    if args.list_stages:
        for index, stage in enumerate(declared, start=1):
            print(f"{index:02d} {stage.name}{' [quick]' if stage.quick else ''}")
        return

    active = [stage for stage in declared if stage.quick] if args.quick else declared
    names = [stage.name for stage in active]
    if args.from_stage and args.from_stage not in names:
        raise SystemExit(
            f"Unknown --from-stage {args.from_stage!r}; choose one of: {', '.join(names)}"
        )

    try:
        log_dir = normalize_log_dir(args.log_dir)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc
    log_dir.mkdir(parents=True, exist_ok=True)
    state_path = log_dir / "pipeline_state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8-sig"))
    else:
        state = {
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "pipeline_version": PIPELINE_VERSION,
            "execution_context": execution_context(),
            "stages": {},
        }
    if state.get("pipeline_version") != PIPELINE_VERSION:
        state = {
            "schema_version": CONTRACT_SCHEMA_VERSION,
            "pipeline_version": PIPELINE_VERSION,
            "execution_context": execution_context(),
            "stages": {},
        }
    stage_state: dict[str, dict[str, object]] = state.setdefault("stages", {})

    start_index = names.index(args.from_stage) if args.from_stage else 0
    if args.from_stage:
        declared_by_name = {stage.name: stage for stage in active}
        incomplete = []
        for name in names[:start_index]:
            prior_stage = declared_by_name[name]
            if prerequisite_satisfied(prior_stage, stage_state.get(name)):
                continue
            incomplete.append(name)
        if incomplete:
            raise SystemExit(
                "--from-stage requires earlier PASS records in pipeline_state.json; "
                f"missing: {', '.join(incomplete)}"
            )

    records: list[dict[str, object]] = []
    for index, stage in enumerate(active):
        if index < start_index:
            continue
        fingerprint = command_fingerprint(stage)
        prior = stage_state.get(stage.name, {})
        if (
            args.resume
            and prior.get("status") == "PASS"
            and prior.get("command_fingerprint") == fingerprint
        ):
            print(f"[RESUME SKIP] {stage.name}", flush=True)
            records.append(prior)
            continue
        if stage.predicate is not None and not stage.predicate():
            record = {
                "name": stage.name,
                "command": portable_command(stage.command),
                "command_fingerprint": fingerprint,
                "status": "SKIP",
                "exit_code": 0,
                "started_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_seconds": 0.0,
                "peak_rss_mb": None,
                "log": "",
                "reason": stage.skip_reason,
            }
            print(f"[STAGE SKIP] {stage.name}: {stage.skip_reason}", flush=True)
        else:
            print(f"[STAGE START] {stage.name}", flush=True)
            record = run_stage(
                stage,
                log_dir / "logs" / f"{index + 1:02d}_{stage.name}.log",
                heartbeat_seconds=max(1.0, args.heartbeat_seconds),
            )
            print(
                f"[STAGE {record['status']}] {stage.name} "
                f"elapsed={record['elapsed_seconds']:.1f}s "
                f"peak_rss_mb={record['peak_rss_mb']}",
                flush=True,
            )
        # A distributed package must be checked before its deposited runtime
        # authority is archived or replaced by this regeneration run.
        archive_runtime_authority_after_checksum(
            stage.name,
            str(record["status"]),
            log_dir,
        )
        stage_state[stage.name] = record
        write_state(state_path, state)
        records.append(record)
        if record["status"] == "FAIL":
            write_runtime_contract(
                log_dir,
                records,
                mode="quick" if args.quick else "full",
                expected_stage_count=len(active),
            )
            raise SystemExit(int(record["exit_code"]))

    write_runtime_contract(
        log_dir,
        records,
        mode="quick" if args.quick else "full",
        expected_stage_count=len(active),
    )
    print(
        f"STORED_RESULTS_PIPELINE_PASS stages={len(records)} "
        f"quick={args.quick} log_dir={log_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
