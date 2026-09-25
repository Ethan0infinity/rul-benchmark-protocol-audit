from __future__ import annotations

import argparse
import json
import locale
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = PROJECT_ROOT / "reports" / "round12_new_evidence"
STATE_PATH = REPORT_ROOT / "pipeline_state.json"
STAGES = (
    "independent",
    "crossed",
    "cmapss_preprocessing",
    "preference",
    "factorial",
    "lofo",
    "ncmapss",
    "ncmapss_horizon",
    "reclassify",
    "analysis",
    "completeness",
    "assets",
)


def command_for(stage: str, epochs: int) -> list[str]:
    python = sys.executable
    if stage in {
        "independent",
        "crossed",
        "cmapss_preprocessing",
        "preference",
        "factorial",
        "lofo",
        "ncmapss",
        "ncmapss_horizon",
    }:
        return [
            python,
            "scripts/run_round12_new_experiments.py",
            "--families",
            stage,
            "--epochs",
            str(epochs),
            "--replace-manifest",
        ]
    if stage == "analysis":
        return [python, "scripts/analyze_round12_new_experiments.py"]
    if stage == "reclassify":
        return [python, "scripts/reclassify_round12_result_metadata.py"]
    if stage == "completeness":
        return [
            python,
            "scripts/check_round12_new_evidence.py",
            "--epochs",
            str(epochs),
        ]
    if stage == "assets":
        command = [
            python,
            "scripts/build_round12_new_evidence_assets.py",
            "--copy-tables-to",
            str(PROJECT_ROOT / "supplementary" / "generated"),
        ]
        manuscript = PROJECT_ROOT.parent / "正文"
        if manuscript.exists():
            command.extend(
                [
                    str(manuscript / "generated"),
                    "--copy-figures-to",
                    str(manuscript / "figures"),
                ]
            )
        return command
    raise ValueError(stage)


def write_state(state: dict[str, object]) -> None:
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    STATE_PATH.write_text(
        json.dumps(state, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def safe_print(message: str, *, end: str = "\n") -> None:
    try:
        print(message, end=end, flush=True)
    except (BrokenPipeError, OSError, UnicodeEncodeError):
        pass


def run_stage(
    stage: str,
    command: list[str],
    state: dict[str, object],
    *,
    live_output: bool,
) -> None:
    started = datetime.now().astimezone()
    log_path = REPORT_ROOT / f"{stage}.log"
    safe_print(f"[ROUND12 PIPELINE START] {stage}")
    state[stage] = {
        "status": "RUNNING",
        "started_at": started.isoformat(),
        "command": command,
        "log": log_path.relative_to(PROJECT_ROOT).as_posix(),
    }
    write_state(state)
    with log_path.open("a", encoding="utf-8", newline="") as log:
        log.write(f"\n[{started.isoformat()}] {' '.join(command)}\n")
        process = subprocess.Popen(
            command,
            cwd=PROJECT_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding=locale.getpreferredencoding(False),
            errors="replace",
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            if live_output:
                safe_print(line, end="")
        return_code = process.wait()
    finished = datetime.now().astimezone()
    state[stage] = {
        **state[stage],
        "status": "PASS" if return_code == 0 else "FAIL",
        "finished_at": finished.isoformat(),
        "elapsed_seconds": (finished - started).total_seconds(),
        "return_code": return_code,
    }
    write_state(state)
    if return_code != 0:
        raise SystemExit(return_code)
    safe_print(f"[ROUND12 PIPELINE PASS] {stage}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Resume-safe coordinator for Round-12 retraining, analysis, "
            "strict completeness checking, and manuscript asset generation."
        )
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--from-stage", choices=STAGES, default=STAGES[0])
    parser.add_argument("--through-stage", choices=STAGES, default=STAGES[-1])
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--live-output",
        action="store_true",
        help="Echo child output as well as writing stage logs.",
    )
    args = parser.parse_args()
    start = STAGES.index(args.from_stage)
    end = STAGES.index(args.through_stage)
    if end < start:
        raise SystemExit("--through-stage must not precede --from-stage")
    selected = STAGES[start : end + 1]
    commands = [(stage, command_for(stage, args.epochs)) for stage in selected]
    if args.dry_run:
        for stage, command in commands:
            print(f"{stage}: {' '.join(command)}")
        print("ROUND12_NEW_EVIDENCE_PIPELINE_DRY_RUN_PASS")
        return
    state: dict[str, object] = {}
    if STATE_PATH.exists():
        state = json.loads(STATE_PATH.read_text(encoding="utf-8"))
    for stage, command in commands:
        run_stage(stage, command, state, live_output=args.live_output)
    safe_print("ROUND12_NEW_EVIDENCE_PIPELINE_PASS")


if __name__ == "__main__":
    main()
