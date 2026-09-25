from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).absolute().parents[1]
SCRIPTS_DIR = PROJECT_ROOT / "scripts"
sys.path.insert(0, str(SCRIPTS_DIR))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rul.reporting import collect_metrics, filter_metric_rows

from check_result_consistency import check_rows
from audit_paper_tables import regenerate_tables_dry_run
from run_formal_benchmark import FORMAL_BENCHMARK_MODELS, FORMAL_BENCHMARK_SEEDS, FORMAL_BENCHMARK_SUBSETS


DEFAULT_MAIN_EXPERIMENT = "paper_main_v3_seed42"
MULTISEED_REQUIRED_SEEDS = set(FORMAL_BENCHMARK_SEEDS)
MULTISEED_REQUIRED_SUBSETS = set(FORMAL_BENCHMARK_SUBSETS)
MULTISEED_REQUIRED_MODELS = set(FORMAL_BENCHMARK_MODELS)


@dataclass
class GateResult:
    name: str
    status: str
    detail: str


def default_gate_options(stage: str) -> dict[str, int | None | bool]:
    if stage == "final":
        return {"expected_count": 260, "min_epochs": 30, "require_protocol_v2": True}
    if stage == "preflight":
        return {"expected_count": 0, "min_epochs": None, "require_protocol_v2": True}
    raise ValueError(f"Unknown quality gate stage: {stage}")


def claim_level_for_state(completed_runs: int, expected_runs: int, passed: bool, stage: str) -> str:
    if not passed or completed_runs <= 0:
        return "no_performance_claim_allowed"
    if completed_runs >= expected_runs and stage == "final":
        return "single_seed_claim_allowed"
    return "protocol_or_partial_results_only"


def claim_level_for_evidence(
    completed_runs: int,
    expected_runs: int,
    passed: bool,
    stage: str,
    evidence: dict[str, object],
) -> str:
    if not passed or completed_runs <= 0:
        return "no_performance_claim_allowed"
    if completed_runs >= expected_runs and bool(evidence.get("paper_tables_ready")) and bool(evidence.get("multi_seed_ready")):
        return "multi_seed_claim_allowed"
    if completed_runs >= expected_runs and bool(evidence.get("paper_tables_ready")):
        return "single_seed_claim_allowed"
    return claim_level_for_state(completed_runs, expected_runs, passed, stage)


def resolve_gate_experiment_names(stage: str, experiment_names: list[str] | None, experiment_prefix: str | None) -> list[str] | None:
    if experiment_names:
        return experiment_names
    if experiment_prefix:
        return None
    return None


def select_rows_for_gate(
    rows: list[dict],
    *,
    stage: str,
    experiment_names: list[str] | None,
    experiment_prefix: str | None,
) -> list[dict]:
    if stage == "preflight" and not experiment_names and not experiment_prefix:
        return []
    if stage == "final" and not experiment_names and not experiment_prefix:
        experiment_prefix = "paper_main_v3_seed"
    return filter_metric_rows(
        rows,
        experiment_names=resolve_gate_experiment_names(stage, experiment_names, experiment_prefix),
        experiment_prefix=experiment_prefix,
    )


def _csv_row_count(path: Path) -> int:
    if not path.exists():
        return 0
    with path.open("r", encoding="utf-8") as f:
        lines = [line for line in f.read().splitlines() if line.strip()]
    return max(0, len(lines) - 1)


def _status_from_ready(ready: bool, has_partial_evidence: bool) -> str:
    if ready:
        return "complete"
    return "partial" if has_partial_evidence else "not_started"


def multi_seed_progress(project_root: Path) -> dict[str, int | str | bool]:
    rows = collect_metrics(project_root / "results")
    completed: set[tuple[int, str, str]] = set()
    for row in rows:
        experiment_name = str(row.get("experiment_name", ""))
        if not experiment_name.startswith("paper_main_v3_seed"):
            continue
        if row.get("result_status") not in {None, "", "formal"}:
            continue
        if row.get("allowed_for_paper") is False:
            continue
        try:
            seed = int(row.get("seed"))
        except (TypeError, ValueError):
            continue
        subset = str(row.get("subset", ""))
        model = str(row.get("model", ""))
        if seed in MULTISEED_REQUIRED_SEEDS and subset in MULTISEED_REQUIRED_SUBSETS and model in MULTISEED_REQUIRED_MODELS:
            completed.add((seed, subset, model))

    expected = len(MULTISEED_REQUIRED_SEEDS) * len(MULTISEED_REQUIRED_SUBSETS) * len(MULTISEED_REQUIRED_MODELS)
    completed_count = len(completed)
    ready = completed_count >= expected
    return {
        "multi_seed_expected": expected,
        "multi_seed_completed": completed_count,
        "multi_seed_ready": ready,
        "multi_seed_status": _status_from_ready(ready, completed_count > 0),
    }


def evidence_statuses(project_root: Path) -> dict[str, str | bool]:
    results_dir = project_root / "results"
    tables_dir = project_root / "paper_outputs" / "tables"

    ablation_summary_rows = _csv_row_count(results_dir / "ablation_runs.csv")
    ablation_table_rows = _csv_row_count(tables_dir / "table_ablation.csv")
    robustness_table_rows = _csv_row_count(tables_dir / "table_robustness.csv")
    statistical_rows = _csv_row_count(tables_dir / "table_statistical_tests.csv")

    required_core_tables = [
        tables_dir / "table_main_accuracy.csv",
        tables_dir / "table_complexity.csv",
        tables_dir / "table_critical_zone.csv",
    ]
    core_tables_ready = all(_csv_row_count(path) > 0 for path in required_core_tables)
    ablation_ready = ablation_summary_rows > 0 and ablation_table_rows > 0
    robustness_ready = robustness_table_rows > 0
    statistical_ready = statistical_rows > 0
    paper_tables_ready = core_tables_ready and ablation_ready and robustness_ready and statistical_ready

    has_any_table = any(path.exists() for path in tables_dir.glob("table_*.csv")) if tables_dir.exists() else False
    statuses = {
        "ablation_status": _status_from_ready(ablation_ready, ablation_summary_rows > 0 or ablation_table_rows > 0),
        "robustness_status": _status_from_ready(robustness_ready, robustness_table_rows > 0),
        "statistical_test_status": _status_from_ready(statistical_ready, statistical_rows > 0),
        "tables_status": _status_from_ready(paper_tables_ready, has_any_table),
        "ablation_ready": ablation_ready,
        "robustness_ready": robustness_ready,
        "paper_tables_ready": paper_tables_ready,
    }
    statuses.update(multi_seed_progress(project_root))
    return statuses


def _run_command(name: str, cmd: list[str], timeout: int = 600) -> GateResult:
    proc = subprocess.run(cmd, cwd=PROJECT_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, timeout=timeout)
    if proc.returncode == 0:
        return GateResult(name, "PASS", proc.stdout.strip().splitlines()[-1] if proc.stdout.strip() else "ok")
    return GateResult(name, "FAIL", proc.stdout[-2000:])


def _update_workflow_state(
    results: list[GateResult],
    experiment_names: list[str] | None,
    completed_runs: int,
    *,
    expected_runs: int,
    stage: str,
    total_metric_rows: int,
) -> None:
    path = PROJECT_ROOT / "workflow_state.json"
    state = {}
    if path.exists():
        state = json.loads(path.read_text(encoding="utf-8-sig"))
    passed = all(result.status == "PASS" for result in results)
    evidence = evidence_statuses(PROJECT_ROOT)
    claim_level = claim_level_for_evidence(completed_runs, expected_runs, passed, stage, evidence)
    formal_results_ready = bool(state.get("formal_results_ready", False)) or (
        passed and completed_runs >= expected_runs and bool(evidence["paper_tables_ready"])
    )
    if completed_runs >= expected_runs and bool(evidence["paper_tables_ready"]):
        stage_value = "final_passed" if stage == "final" and passed else "single_seed_main_completed"
    else:
        stage_value = f"{stage}_passed" if passed else f"{stage}_failed"
    state.update(
        {
            "stage": stage_value,
            "current_protocol": "paper_main_v3",
            "main_experiment_expected": expected_runs,
            "main_experiment_completed": completed_runs,
            "total_metric_rows": total_metric_rows,
            "ablation_status": evidence["ablation_status"],
            "robustness_status": evidence["robustness_status"],
            "statistical_test_status": evidence["statistical_test_status"],
            "tables_status": evidence["tables_status"],
            "multi_seed_status": evidence["multi_seed_status"],
            "multi_seed_expected": evidence["multi_seed_expected"],
            "multi_seed_completed": evidence["multi_seed_completed"],
            "paper_claim_level": claim_level,
            "completed_main_runs": completed_runs,
            "formal_results_ready": formal_results_ready,
            "main_experiment_status": "complete" if completed_runs >= expected_runs else "not_started" if completed_runs == 0 else "partial",
            "ablation_ready": evidence["ablation_ready"],
            "robustness_ready": evidence["robustness_ready"],
            "paper_tables_ready": evidence["paper_tables_ready"],
            "multi_seed_ready": evidence["multi_seed_ready"],
            "last_quality_gate": {
                "time": datetime.now().astimezone().isoformat(timespec="seconds"),
                "status": "PASS" if passed else "FAIL",
                "stage": stage,
                "experiment_names": experiment_names or [],
            },
        }
    )
    path.write_text(json.dumps(state, indent=2, ensure_ascii=False), encoding="utf-8")


def run_quality_gate(args: argparse.Namespace) -> list[GateResult]:
    results: list[GateResult] = []
    python = sys.executable
    defaults = default_gate_options(args.stage)
    expected_count = args.expected_count if args.expected_count is not None else defaults["expected_count"]
    min_epochs = args.min_epochs if args.min_epochs is not None else defaults["min_epochs"]
    require_protocol_v2 = bool(args.require_protocol_v2 or defaults["require_protocol_v2"])

    if not args.skip_compile:
        results.append(_run_command("python_syntax", [python, "scripts/check_python_syntax.py"], timeout=300))
    if not args.skip_tests:
        results.append(_run_command("core_tests", [python, "scripts/run_core_tests.py"], timeout=300))
    if not args.skip_data_validation:
        results.append(_run_command("data_validation", [python, "scripts/validate_data.py"], timeout=300))
    if not args.skip_benchmark:
        results.append(_run_command("numeric_benchmark", [python, "benchmark/run_benchmark.py", "--no-write"], timeout=300))
    if not args.skip_evals:
        results.append(_run_command("eval_harness", [python, "eval_harness/run_evals.py", "--no-write", "--min-scenarios", "13"], timeout=300))
    if args.stage == "final":
        results.append(_run_command("risk_protocol", [python, "scripts/check_risk_protocol.py"], timeout=300))
        results.append(_run_command("reference_audit", [python, "scripts/audit_references.py"], timeout=300))
        artifact_metadata_path = PROJECT_ROOT / "artifact_metadata.json"
        artifact_metadata = (
            json.loads(artifact_metadata_path.read_text(encoding="utf-8-sig"))
            if artifact_metadata_path.exists()
            else {}
        )
        if artifact_metadata.get("ncmapss_cache_distribution") == "excluded" and not (
            PROJECT_ROOT / "data" / "external" / "processed" / "ncmapss_ds02_smp100_win50.npz"
        ).exists():
            results.append(GateResult("ncmapss_cache", "PASS", "cache excluded after redistribution review; source/preparation route required"))
        else:
            results.append(_run_command("ncmapss_cache", [python, "scripts/validate_ncmapss_cache.py"], timeout=300))
        results.append(_run_command("checkpoint_integrity", [python, "scripts/check_checkpoint_integrity.py"], timeout=600))
        results.append(_run_command("ablation_consistency", [python, "scripts/check_ablation_consistency.py"], timeout=300))
        results.append(_run_command("ncmapss_readiness", [python, "scripts/check_ncmapss_readiness.py"], timeout=600))
        results.append(_run_command("advanced_evidence", [python, "scripts/check_advanced_evidence.py"], timeout=300))
        results.append(
            _run_command(
                "extended_review_evidence",
                [python, "scripts/check_extended_review_evidence.py"],
                timeout=300,
            )
        )
        results.append(_run_command("figure_data_consistency", [python, "scripts/check_figure_data_consistency.py"], timeout=300))
        results.append(_run_command("manuscript_evidence", [python, "scripts/check_manuscript_evidence.py"], timeout=300))
        if not args.skip_submission_manuscript:
            submission_command = [python, "scripts/check_submission_manuscripts.py"]
            if args.skip_public_artifact:
                submission_command.append("--allow-pending-metadata")
            results.append(_run_command("submission_manuscript", submission_command, timeout=300))
        if not args.skip_public_artifact:
            results.append(_run_command("public_artifact", [python, "scripts/check_public_artifact.py"], timeout=60))

    all_rows = collect_metrics(args.results_dir)
    rows = select_rows_for_gate(
        all_rows,
        stage=args.stage,
        experiment_names=args.experiment_names,
        experiment_prefix=args.experiment_prefix,
    )
    main_rows = filter_metric_rows(all_rows, experiment_prefix="paper_main_v3_seed")
    issues = check_rows(
        rows,
        expected_count=expected_count,
        min_epochs=min_epochs,
        require_protocol_v2=require_protocol_v2,
    )
    if args.stage == "final" and not rows:
        results.append(
            GateResult(
                "final_results_presence",
                "FAIL",
                "final gate requires formal results; current rows=0",
            )
        )
    results.append(
        GateResult(
            "protocol_consistency",
            "FAIL" if issues else "PASS",
            "; ".join(issues[:8]) if issues else f"{len(rows)} metrics rows satisfy requested consistency checks",
        )
    )

    if rows:
        _, table_issues = regenerate_tables_dry_run(rows)
        results.append(
            GateResult(
                "table_regeneration_dry_run",
                "FAIL" if table_issues else "PASS",
                "; ".join(table_issues[:8]) if table_issues else "tables regenerate from metrics without forbidden tokens",
            )
        )
    else:
        results.append(GateResult("table_regeneration_dry_run", "PASS", "no rows yet; table dry-run skipped safely"))

    expected_runs = int(expected_count or 0) if args.stage == "final" else 24
    gate_experiment_names = resolve_gate_experiment_names(args.stage, args.experiment_names, args.experiment_prefix)
    _update_workflow_state(
        results,
        gate_experiment_names,
        len(main_rows),
        expected_runs=expected_runs,
        stage=args.stage,
        total_metric_rows=len(all_rows),
    )
    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the paper-level quality gate before table generation.")
    parser.add_argument("--stage", choices=["preflight", "final"], default="preflight")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--experiment-name", action="append", dest="experiment_names", default=None)
    parser.add_argument("--experiment-prefix", default=None)
    parser.add_argument("--expected-count", type=int, default=None)
    parser.add_argument("--min-epochs", type=int, default=None)
    parser.add_argument(
        "--require-protocol-v3",
        "--require-protocol-v2",
        dest="require_protocol_v2",
        action="store_true",
    )
    parser.add_argument("--skip-compile", action="store_true")
    parser.add_argument("--skip-tests", action="store_true")
    parser.add_argument("--skip-data-validation", action="store_true")
    parser.add_argument("--skip-benchmark", action="store_true")
    parser.add_argument("--skip-evals", action="store_true")
    parser.add_argument(
        "--skip-submission-manuscript",
        action="store_true",
        help="Skip outer manuscript/PDF freshness checks in a stored-results-only replay.",
    )
    parser.add_argument(
        "--skip-public-artifact",
        action="store_true",
        help="Internal deposit-ready audit only; final submission must not use this override.",
    )
    args = parser.parse_args()

    results = run_quality_gate(args)
    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    (reports_dir / "paper_quality_gate_last.json").write_text(
        json.dumps([asdict(result) for result in results], indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for result in results:
        print(f"[GATE {result.status}] {result.name}: {result.detail}")
    if any(result.status != "PASS" for result in results):
        print("PAPER_FINAL_GATE_FAIL" if args.stage == "final" else "PAPER_PREFLIGHT_FAIL")
        raise SystemExit(1)
    print("PAPER_FINAL_GATE_PASS" if args.stage == "final" else "PAPER_PREFLIGHT_PASS")


if __name__ == "__main__":
    main()
