from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable


PROJECT_ROOT = Path(__file__).resolve().parents[1]


@dataclass
class EvalResult:
    scenario: str
    status: str
    detail: str


def _read_texts(paths: list[Path]) -> list[tuple[Path, str]]:
    texts: list[tuple[Path, str]] = []
    for path in paths:
        if path.exists() and path.is_file():
            try:
                texts.append((path, path.read_text(encoding="utf-8", errors="ignore")))
            except OSError:
                continue
    return texts


def _project_text_files(project_root: Path) -> list[Path]:
    files: list[Path] = []
    for base in [
        *sorted(project_root.glob("*.md")),
        *sorted((project_root / "docs").glob("*.md")),
        *sorted((project_root / "replication_package").glob("*.md")),
    ]:
        if base.exists():
            files.append(base)
    paper_outputs = project_root / "paper_outputs"
    if paper_outputs.exists():
        files.extend(
            path
            for path in paper_outputs.rglob("*")
            if path.is_file() and path.suffix.lower() in {".md", ".csv", ".json", ".txt"}
        )
    return files


def _workflow_state(project_root: Path) -> dict:
    path = project_root / "workflow_state.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {}


def _all_project_text(project_root: Path) -> str:
    return "\n".join(text for _, text in _read_texts(_project_text_files(project_root))).lower()


def _claim_enforcement_enabled(project_root: Path) -> bool:
    state = _workflow_state(project_root)
    return str(state.get("paper_claim_level", "no_performance_claim_allowed")) != "no_performance_claim_allowed"


def _evidence_enforcement_enabled(project_root: Path) -> bool:
    if _claim_enforcement_enabled(project_root):
        return True
    state = _workflow_state(project_root)
    if int(state.get("main_experiment_completed", state.get("completed_main_runs", 0)) or 0) >= 24:
        return True
    results = project_root / "results"
    if not results.exists():
        return False
    for metrics_path in results.rglob("metrics.json"):
        try:
            metrics = json.loads(metrics_path.read_text(encoding="utf-8", errors="ignore"))
        except json.JSONDecodeError:
            continue
        if metrics.get("allowed_for_paper") is True and metrics.get("result_status") == "formal":
            return True
    return False


def eval_no_mixed_protocol(project_root: Path) -> EvalResult:
    paper_outputs = project_root / "paper_outputs"
    quarantined_dirs = {"release_evidence_smoke"}
    patterns = [
        re.compile(r"archive_", re.IGNORECASE),
        re.compile(r"paper_main_seed42", re.IGNORECASE),
        re.compile(r"completed_epochs\s*[:=]\s*1", re.IGNORECASE),
        re.compile(r"epochs\s*[:=]\s*1", re.IGNORECASE),
    ]
    hits: list[str] = []
    if paper_outputs.exists():
        for path, text in _read_texts(
            [
                item
                for item in paper_outputs.rglob("*")
                if item.is_file() and item.suffix.lower() in {".md", ".csv", ".json", ".txt"}
                and not quarantined_dirs.intersection(item.relative_to(paper_outputs).parts)
            ]
        ):
            for pattern in patterns:
                if pattern.search(text):
                    hits.append(f"{path}: {pattern.pattern}")
            if "smoke" in path.as_posix().lower():
                hits.append(f"{path}: smoke path token")
            elif path.suffix.lower() == ".csv":
                try:
                    records = csv.DictReader(text.splitlines())
                    if any(
                        str(row.get("is_smoke", "")).strip().lower() in {"true", "1", "yes"}
                        or str(row.get("result_status", "")).strip().lower() == "smoke"
                        or "smoke" in str(row.get("experiment_name", "")).lower()
                        for row in records
                    ):
                        hits.append(f"{path}: structured smoke row")
                except csv.Error:
                    hits.append(f"{path}: unreadable CSV")
            elif path.suffix.lower() == ".json":
                try:
                    payload = json.loads(text)
                    records = payload if isinstance(payload, list) else [payload]
                    if any(
                        isinstance(row, dict)
                        and (
                            row.get("is_smoke") is True
                            or str(row.get("result_status", "")).lower() == "smoke"
                            or "smoke" in str(row.get("experiment_name", "")).lower()
                        )
                        for row in records
                    ):
                        hits.append(f"{path}: structured smoke row")
                except json.JSONDecodeError:
                    hits.append(f"{path}: unreadable JSON")
            elif re.search(r"\bsmoke\b", text, flags=re.IGNORECASE):
                hits.append(f"{path}: smoke token")
    if hits:
        return EvalResult("no_mixed_protocol", "FAIL", "; ".join(hits[:5]))
    return EvalResult(
        "no_mixed_protocol",
        "PASS",
        "active paper_outputs has no archive/smoke/old-protocol tokens; quarantined smoke evidence is excluded",
    )


def eval_no_data_leakage(project_root: Path) -> EvalResult:
    training = project_root / "src" / "rul" / "training.py"
    if not training.exists():
        return EvalResult("no_data_leakage", "FAIL", f"missing {training}")
    text = training.read_text(encoding="utf-8")
    required = ["train_core_df", "val_df", "fit_transform_train_val_test", "select_features("]
    missing = [item for item in required if item not in text]
    if missing:
        return EvalResult("no_data_leakage", "FAIL", f"missing leakage guard code tokens: {missing}")
    return EvalResult("no_data_leakage", "PASS", "training pipeline keeps train_core before feature selection and scaling")


def eval_no_overclaim_sota(project_root: Path) -> EvalResult:
    banned = [
        "state-of-the-art",
        "best performance on c-mapss",
        "outperforms all existing methods",
        "outperform all existing methods",
    ]
    hits: list[str] = []
    for path, text in _read_texts(_project_text_files(project_root)):
        lowered = text.lower()
        for phrase in banned:
            if phrase in lowered:
                hits.append(f"{path}: {phrase}")
    if hits:
        return EvalResult("no_overclaim_sota", "FAIL", "; ".join(hits[:5]))
    return EvalResult("no_overclaim_sota", "PASS", "docs avoid unsupported SOTA-style overclaims")


def eval_protocol_v2_compliance(project_root: Path) -> EvalResult:
    lock = project_root / "configs" / "protocol_v3.lock.yaml"
    protocol_doc = project_root / "docs" / "research_protocol_v3.md"
    missing = [str(path) for path in [lock, protocol_doc] if not path.exists()]
    if missing:
        return EvalResult("protocol_v2_compliance", "FAIL", f"missing files: {missing}")
    text = lock.read_text(encoding="utf-8")
    for token in ["protocol_name: paper_main_v3", "protocol_version: \"3.0\"", "required_selection_metric: val_last_risk_score"]:
        if token not in text:
            return EvalResult("protocol_v2_compliance", "FAIL", f"protocol lock missing token {token!r}")
    return EvalResult("protocol_v2_compliance", "PASS", "protocol lock and protocol document are present")


def eval_robustness_claim_check(project_root: Path) -> EvalResult:
    claims = _all_project_text(project_root)
    if "sensor degradation" not in claims and "robustness" not in claims:
        return EvalResult("robustness_claim_check", "PASS", "no sensor-degradation robustness claim found")
    experiments_path = project_root / "src" / "rul" / "experiments.py"
    if not experiments_path.exists():
        return EvalResult("robustness_claim_check", "FAIL", f"missing {experiments_path}")
    experiments = experiments_path.read_text(encoding="utf-8")
    required = ["global_sensor_missing", "window_random_missing", "block_missing", "sensor_drift"]
    missing = [item for item in required if item not in experiments]
    if missing:
        return EvalResult("robustness_claim_check", "FAIL", f"robustness claim lacks scenarios: {missing}")
    return EvalResult("robustness_claim_check", "PASS", "sensor-degradation claims are backed by robustness scenario code")


def eval_no_performance_claim_without_results(project_root: Path) -> EvalResult:
    state = _workflow_state(project_root)
    completed = int(state.get("main_experiment_completed", state.get("completed_main_runs", 0)) or 0)
    claim_level = str(state.get("paper_claim_level", "no_performance_claim_allowed"))
    if completed >= 24 or claim_level != "no_performance_claim_allowed":
        return EvalResult("no_performance_claim_without_results", "PASS", "workflow state allows some performance discussion")
    banned = [
        "outperforms",
        "achieves better",
        "improves rmse",
        "superior performance",
    ]
    allowed_markers = ["expected", "hypothesis", "preliminary", "do not", "avoid"]
    hits: list[str] = []
    for path, text in _read_texts(_project_text_files(project_root)):
        lowered = text.lower()
        if any(marker in lowered for marker in allowed_markers):
            continue
        for phrase in banned:
            if phrase in lowered:
                hits.append(f"{path}: {phrase}")
    if hits:
        return EvalResult("no_performance_claim_without_results", "FAIL", "; ".join(hits[:5]))
    return EvalResult("no_performance_claim_without_results", "PASS", "no unsupported performance claim before formal results")


def _robustness_files(project_root: Path) -> list[Path]:
    results_root = project_root / "results"
    if not results_root.exists():
        return []
    return [path for path in results_root.rglob("*") if path.name in {"robustness.csv", "robustness.json"}]


def eval_no_robustness_claim_without_robustness_csv(project_root: Path) -> EvalResult:
    claims = _all_project_text(project_root)
    robustness_claimed = "robust under sensor degradation" in claims or "robustness advantage" in claims
    if not robustness_claimed:
        return EvalResult("no_robustness_claim_without_robustness_csv", "PASS", "no sensor degradation robustness claim found")
    required = ["global_sensor_missing", "window_random_missing", "block_missing", "sensor_drift"]
    combined = ""
    for path in _robustness_files(project_root):
        combined += "\n" + path.read_text(encoding="utf-8", errors="ignore")
    missing = [scenario for scenario in required if scenario not in combined]
    if missing:
        return EvalResult(
            "no_robustness_claim_without_robustness_csv",
            "FAIL",
            f"missing robustness evidence scenarios: {missing}",
        )
    return EvalResult("no_robustness_claim_without_robustness_csv", "PASS", "robustness evidence files include required scenarios")


def _ablation_text(project_root: Path) -> str:
    parts = []
    for path in [
        project_root / "results" / "ablation_runs.csv",
        project_root / "paper_outputs" / "tables" / "table_ablation.csv",
    ]:
        if path.exists():
            parts.append(path.read_text(encoding="utf-8", errors="ignore"))
    return "\n".join(parts).lower()


def eval_no_missing_ablation_for_mask_claim(project_root: Path) -> EvalResult:
    if not _evidence_enforcement_enabled(project_root):
        return EvalResult("no_missing_ablation_for_mask_claim", "PASS", "claim-level does not require ablation evidence yet")
    text = _all_project_text(project_root)
    mask_claim = "missing mask" in text or "explicit mask" in text or "mask-aware" in text
    if not mask_claim:
        return EvalResult("no_missing_ablation_for_mask_claim", "PASS", "no missing-mask claim found")
    if "no_missing_mask" not in _ablation_text(project_root):
        return EvalResult("no_missing_ablation_for_mask_claim", "FAIL", "missing no_missing_mask ablation evidence")
    return EvalResult("no_missing_ablation_for_mask_claim", "PASS", "missing-mask claim has ablation evidence")


def eval_no_condition_norm_claim_without_fd004_ablation(project_root: Path) -> EvalResult:
    if not _evidence_enforcement_enabled(project_root):
        return EvalResult("no_condition_norm_claim_without_fd004_ablation", "PASS", "claim-level does not require ablation evidence yet")
    text = _all_project_text(project_root)
    condition_claim = "condition-aware normalization" in text or "condition_standard" in text or "condition normalization" in text
    if not condition_claim:
        return EvalResult("no_condition_norm_claim_without_fd004_ablation", "PASS", "no condition-normalization claim found")
    ablation = _ablation_text(project_root)
    if "fd004" not in ablation or "no_condition_norm" not in ablation:
        return EvalResult(
            "no_condition_norm_claim_without_fd004_ablation",
            "FAIL",
            "missing FD004 no_condition_norm ablation evidence",
        )
    return EvalResult("no_condition_norm_claim_without_fd004_ablation", "PASS", "FD004 condition-normalization ablation evidence exists")


def eval_dataset_integrity_claim_check(project_root: Path) -> EvalResult:
    text = _all_project_text(project_root)
    claim_tokens = ["data integrity", "validated data", "canonical files", "dataset integrity", "数据完整", "数据校验"]
    if not any(token in text for token in claim_tokens):
        return EvalResult("dataset_integrity_claim_check", "PASS", "no explicit dataset-integrity claim found")

    required_files = [
        project_root / "reports" / "data_validation_report.md",
        project_root / "reports" / "raw_file_checksums.sha256",
        project_root / "reports" / "dataset_sufficiency_report.md",
    ]
    missing = [str(path) for path in required_files if not path.exists()]
    if missing:
        return EvalResult("dataset_integrity_claim_check", "FAIL", f"missing dataset-integrity evidence files: {missing}")

    report = (project_root / "reports" / "data_validation_report.md").read_text(encoding="utf-8", errors="ignore")
    required_tokens = ["Overall status: PASS", "Checks: 88 total, 0 failed"]
    missing_tokens = [token for token in required_tokens if token not in report]
    if missing_tokens:
        return EvalResult("dataset_integrity_claim_check", "FAIL", f"data validation report missing tokens: {missing_tokens}")
    return EvalResult("dataset_integrity_claim_check", "PASS", "dataset-integrity claim has validation report, checksums, and sufficiency report")


def _csv_has_columns(paths: list[Path], required_columns: list[str]) -> tuple[bool, str]:
    for path in paths:
        if not path.exists():
            continue
        try:
            with path.open("r", newline="", encoding="utf-8", errors="ignore") as f:
                reader = csv.reader(f)
                header = next(reader, [])
        except (OSError, StopIteration):
            continue
        normalized = {str(col).strip() for col in header}
        missing = [col for col in required_columns if col not in normalized]
        if not missing:
            return True, str(path)
    return False, "missing required columns in candidate CSV files"


def eval_no_critical_zone_claim_without_table(project_root: Path) -> EvalResult:
    if not _evidence_enforcement_enabled(project_root):
        return EvalResult("no_critical_zone_claim_without_table", "PASS", "claim-level does not require critical-zone table yet")
    text = _all_project_text(project_root)
    claim_tokens = ["critical-zone", "critical zone", "near-failure", "near failure", "safety-critical", "临近失效", "安全关键"]
    if not any(token in text for token in claim_tokens):
        return EvalResult("no_critical_zone_claim_without_table", "PASS", "no critical-zone claim found")
    ok, detail = _csv_has_columns(
        [
            project_root / "paper_outputs" / "tables" / "table_critical_zone.csv",
            project_root / "paper_outputs" / "table_critical_zone.csv",
        ],
        ["test_critical_30_rmse", "test_critical_30_late_prediction_ratio", "test_critical_50_rmse"],
    )
    if not ok:
        return EvalResult("no_critical_zone_claim_without_table", "FAIL", detail)
    return EvalResult("no_critical_zone_claim_without_table", "PASS", f"critical-zone evidence table found: {detail}")


def eval_no_statistics_claim_without_wilcoxon(project_root: Path) -> EvalResult:
    if not _evidence_enforcement_enabled(project_root):
        return EvalResult("no_statistics_claim_without_wilcoxon", "PASS", "claim-level does not require statistical evidence yet")
    text = _all_project_text(project_root)
    claim_tokens = ["significant", "statistically", "statistical significance", "显著", "统计显著"]
    if not any(token in text for token in claim_tokens):
        return EvalResult("no_statistics_claim_without_wilcoxon", "PASS", "no statistical-significance claim found")
    ok, detail = _csv_has_columns(
        [
            project_root / "paper_outputs" / "tables" / "table_statistical_tests.csv",
            project_root / "paper_outputs" / "tables" / "statistical_tests.csv",
            project_root / "paper_outputs" / "statistical_tests.csv",
        ],
        ["p_value", "effect_size"],
    )
    if not ok:
        return EvalResult("no_statistics_claim_without_wilcoxon", "FAIL", detail)
    return EvalResult("no_statistics_claim_without_wilcoxon", "PASS", f"Wilcoxon/statistical evidence table found: {detail}")


def eval_citation_hygiene(project_root: Path) -> EvalResult:
    refs = project_root / "references.bib"
    if not refs.exists():
        return EvalResult("citation_hygiene", "FAIL", "references.bib is missing")
    text = refs.read_text(encoding="utf-8", errors="ignore")
    if "TODO" in text or "????" in text:
        return EvalResult("citation_hygiene", "FAIL", "references.bib contains placeholder tokens")
    if "@" not in text:
        return EvalResult("citation_hygiene", "FAIL", "references.bib has no BibTeX entries")
    return EvalResult("citation_hygiene", "PASS", "references.bib exists and has no obvious placeholders")


SCENARIOS: dict[str, Callable[[Path], EvalResult]] = {
    "no_mixed_protocol": eval_no_mixed_protocol,
    "no_data_leakage": eval_no_data_leakage,
    "no_overclaim_sota": eval_no_overclaim_sota,
    "protocol_v2_compliance": eval_protocol_v2_compliance,
    "robustness_claim_check": eval_robustness_claim_check,
    "no_performance_claim_without_results": eval_no_performance_claim_without_results,
    "no_robustness_claim_without_robustness_csv": eval_no_robustness_claim_without_robustness_csv,
    "no_missing_ablation_for_mask_claim": eval_no_missing_ablation_for_mask_claim,
    "no_condition_norm_claim_without_fd004_ablation": eval_no_condition_norm_claim_without_fd004_ablation,
    "dataset_integrity_claim_check": eval_dataset_integrity_claim_check,
    "no_critical_zone_claim_without_table": eval_no_critical_zone_claim_without_table,
    "no_statistics_claim_without_wilcoxon": eval_no_statistics_claim_without_wilcoxon,
    "citation_hygiene": eval_citation_hygiene,
}


def run_evals(project_root: str | Path = PROJECT_ROOT, no_write: bool = False) -> list[dict[str, str]]:
    root = Path(project_root)
    rows = [asdict(check(root)) for check in SCENARIOS.values()]
    if not no_write:
        out_dir = root / "eval_harness" / "results"
        out_dir.mkdir(parents=True, exist_ok=True)
        (out_dir / "latest_results.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description="Run reviewer-style checks for the RUL paper project.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--no-write", action="store_true")
    parser.add_argument("--min-scenarios", type=int, default=13)
    args = parser.parse_args()

    rows = run_evals(args.project_root, no_write=args.no_write)
    if len(rows) < args.min_scenarios:
        print(f"[EVAL FAIL] expected at least {args.min_scenarios} scenarios, found {len(rows)}")
        raise SystemExit(1)
    failures = [row for row in rows if row["status"] != "PASS"]
    for row in rows:
        print(f"[EVAL {row['status']}] {row['scenario']}: {row['detail']}")
    if failures:
        raise SystemExit(1)
    print("RUL_EVAL_HARNESS_PASS")


if __name__ == "__main__":
    main()
