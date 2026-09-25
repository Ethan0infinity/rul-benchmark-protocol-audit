from __future__ import annotations

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rul.config import load_config


SEARCH_KEYS = ("late_life_weight", "late_over_weight")
LOCKED_KEYS = (
    "selection_metric",
    "smooth_late_risk_weight",
    "smooth_late_risk_temperature",
    "reliability_supervision_weight",
    "consistency_weight",
    "consistency_noise_std",
    "quantile_calibration_weight",
    "selection_lpr_weight",
    "selection_severe_late_weight",
)


def main() -> None:
    report_path = PROJECT_ROOT / "reports" / "risk_hyperparameter_search.json"
    if not report_path.exists():
        raise SystemExit("RISK_PROTOCOL_FAIL missing reports/risk_hyperparameter_search.json")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("test_metrics_used_for_selection") is not False:
        raise SystemExit("RISK_PROTOCOL_FAIL test metrics must be excluded from selection")
    selected = report.get("selected", {})
    main = load_config(PROJECT_ROOT / "configs" / "paper_main.yaml")
    ablation = load_config(PROJECT_ROOT / "configs" / "paper_ablation.yaml")
    issues = []
    for key in SEARCH_KEYS:
        if float(main["training"][key]) != float(selected[key]):
            issues.append(f"paper_main {key}={main['training'][key]} selected={selected[key]}")
    for key in (*SEARCH_KEYS, *LOCKED_KEYS):
        if main["training"].get(key) != ablation["training"].get(key):
            issues.append(f"main/ablation mismatch {key}")
    if main["training"].get("selection_metric") != "val_last_risk_score":
        issues.append("selection_metric must be val_last_risk_score")
    if issues:
        print("RISK_PROTOCOL_FAIL")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)
    print(
        "RISK_PROTOCOL_PASS "
        f"late_life_weight={selected['late_life_weight']} late_over_weight={selected['late_over_weight']}"
    )


if __name__ == "__main__":
    main()
