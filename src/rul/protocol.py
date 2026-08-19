from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_PROTOCOL_LOCK = PROJECT_ROOT / "configs" / "protocol_v3.lock.yaml"


def load_protocol_lock(path: str | Path | None = None) -> dict[str, Any]:
    """Load the frozen paper protocol contract."""
    lock_path = Path(path) if path is not None else DEFAULT_PROTOCOL_LOCK
    with lock_path.open("r", encoding="utf-8") as f:
        lock = yaml.safe_load(f) or {}
    lock.setdefault("protocol_name", "paper_main_v3")
    lock["protocol_version"] = str(lock.get("protocol_version", "3.0"))
    lock.setdefault("forbidden_result_prefixes", ["archive_", "smoke", "quick"])
    lock.setdefault("allowed_experiment_prefix", "paper_main_v3_seed")
    lock.setdefault("allowed_supplementary_prefixes", ["ablation_"])
    lock.setdefault("required_metrics", [])
    return lock


def _starts_with_any(value: str, prefixes: list[str]) -> bool:
    lowered = value.lower()
    return any(lowered.startswith(prefix.lower()) for prefix in prefixes)


def _path_has_part(path: str | Path | None, predicate: Any) -> bool:
    if path is None:
        return False
    return any(predicate(part.lower()) for part in Path(path).parts)


def classify_result_status(
    experiment_name: str | None,
    run_dir: str | Path | None = None,
    lock: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return protocol metadata used to decide whether a run can enter paper tables."""
    lock = lock or load_protocol_lock()
    experiment = str(experiment_name or "default")
    lower_experiment = experiment.lower()
    forbidden = [str(item).lower() for item in lock.get("forbidden_result_prefixes", [])]
    allowed_prefix = str(lock.get("allowed_experiment_prefix", "")).lower()
    supplementary = [str(item) for item in lock.get("allowed_supplementary_prefixes", [])]

    is_archive = _starts_with_any(lower_experiment, ["archive_"]) or _path_has_part(run_dir, lambda part: part.startswith("archive_"))
    is_smoke = "smoke" in lower_experiment or _path_has_part(run_dir, lambda part: "smoke" in part)
    is_quick = "quick" in lower_experiment or _path_has_part(run_dir, lambda part: "quick" in part)
    has_forbidden_prefix = _starts_with_any(lower_experiment, forbidden)
    is_formal = bool(allowed_prefix and lower_experiment.startswith(allowed_prefix)) and not (is_archive or is_smoke or is_quick)
    is_supplementary = _starts_with_any(experiment, supplementary) and not (is_archive or is_smoke or is_quick or has_forbidden_prefix)

    if is_archive:
        status = "archive"
    elif is_smoke:
        status = "smoke"
    elif is_quick:
        status = "quick"
    elif is_formal:
        status = "formal"
    elif is_supplementary:
        status = "supplementary"
    else:
        status = "draft"

    return {
        "protocol_name": str(lock.get("protocol_name", "paper_main_v3")),
        "protocol_version": str(lock.get("protocol_version", "3.0")),
        "result_status": status,
        "is_smoke": bool(is_smoke),
        "is_archive": bool(is_archive),
        "allowed_for_paper": status in {"formal", "supplementary"},
    }


def validate_metric_row_against_protocol(row: dict[str, Any], lock: dict[str, Any] | None = None) -> list[str]:
    """Return protocol issues for one metrics.json row."""
    lock = lock or load_protocol_lock()
    issues: list[str] = []
    experiment = str(row.get("experiment_name", "default"))
    run_dir = row.get("run_dir")
    expected_status = classify_result_status(experiment, run_dir, lock)

    for key in lock.get("required_metrics", []):
        if key not in row:
            issues.append(f"missing required metric: {key}")

    if str(row.get("protocol_name", "")) != str(lock.get("protocol_name")):
        issues.append(f"protocol_name={row.get('protocol_name')!r} != {lock.get('protocol_name')!r}")
    if str(row.get("protocol_version", "")) != str(lock.get("protocol_version")):
        issues.append(f"protocol_version={row.get('protocol_version')!r} != {lock.get('protocol_version')!r}")

    if expected_status["result_status"] in {"archive", "smoke", "quick", "draft"}:
        issues.append(f"experiment {experiment!r} is forbidden or not allowed for protocol v3 paper tables")
    if row.get("allowed_for_paper") is not True:
        issues.append("row is not allowed_for_paper")
    if bool(row.get("is_archive")):
        issues.append("row is marked as archive")
    if bool(row.get("is_smoke")):
        issues.append("row is marked as smoke")

    if str(row.get("selection_metric", "")) != str(lock.get("required_selection_metric")):
        issues.append(f"selection_metric={row.get('selection_metric')!r} != {lock.get('required_selection_metric')!r}")
    if str(row.get("validation_last_strategy", "")) != str(lock.get("required_validation_last_strategy")):
        issues.append(
            f"validation_last_strategy={row.get('validation_last_strategy')!r} "
            f"!= {lock.get('required_validation_last_strategy')!r}"
        )
    if str(row.get("scaler", "")) != str(lock.get("required_scaler")):
        issues.append(f"scaler={row.get('scaler')!r} != {lock.get('required_scaler')!r}")
    if row.get("append_missing_mask") is not bool(lock.get("required_append_missing_mask", True)):
        issues.append("append_missing_mask does not match protocol lock")

    return issues
