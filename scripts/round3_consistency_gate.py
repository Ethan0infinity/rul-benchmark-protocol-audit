from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER_ROOT = PROJECT_ROOT.parent / "正文"


def require(condition: bool, message: str, failures: list[str]) -> None:
    if condition:
        print(f"PASS {message}")
    else:
        failures.append(message)
        print(f"FAIL {message}")


def main() -> None:
    failures: list[str] = []
    primary_path = PROJECT_ROOT / "paper_outputs" / "round3_evidence" / "round3_primary_joint_inference.csv"
    endpoint_path = PROJECT_ROOT / "paper_outputs" / "round3_evidence" / "round3_endpoint_sensitivity.csv"
    event_path = PROJECT_ROOT / "paper_outputs" / "round3_evidence" / "round3_late_event_counts.csv"
    condition_path = PROJECT_ROOT / "paper_outputs" / "round3_evidence" / "condition_normalization_summary.csv"
    primary = pd.read_csv(primary_path)
    endpoint = pd.read_csv(endpoint_path)
    events = pd.read_csv(event_path)
    condition = pd.read_csv(condition_path)

    require(len(primary) == 12, "round3 primary record has 12 contrasts", failures)
    require(
        int((primary["holm_adjusted_p_all12"] < 0.05).sum()) == 0,
        "no primary contrast is labeled as Holm12-supported",
        failures,
    )
    fd004_lpr = primary[(primary["subset"] == "FD004") & (primary["metric"] == "lpr30")].iloc[0]
    require(
        abs(float(fd004_lpr["holm_adjusted_p_all12"]) - 0.19198080191980801) < 1e-12,
        "FD004 LPR Holm12 value matches the machine-readable record",
        failures,
    )
    require(len(endpoint) == 80, "endpoint sensitivity has all 80 tau-epsilon-task cells", failures)
    require(len(events) == 800, "event-support record has 800 model-seed endpoint cells", failures)
    require(
        events.loc[~events["conditional_severity_defined"], "conditional_mean_exceedance"].isna().all(),
        "zero-event conditional severity is undefined",
        failures,
    )
    require(
        len(condition) == 12 and set(condition["variant"]) == {"global", "official_state", "kmeans", "continuous"},
        "normalization summary covers four variants and three metrics",
        failures,
    )

    complete_controls = 0
    for seed in (42, 123, 2024, 2025, 2026):
        for variant in ("global", "official_state", "continuous"):
            metrics_path = (
                PROJECT_ROOT
                / "results"
                / f"condition_control_{variant}_fd004_seed{seed}"
                / "FD004"
                / "rast_gru_v2"
                / "metrics.json"
            )
            if metrics_path.exists():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
                if int(metrics.get("planned_epochs", 0)) == 80:
                    complete_controls += 1
    require(complete_controls == 15, "all 15 new formal normalization-control runs are complete", failures)

    english = (PAPER_ROOT / "main_revised.tex").read_text(encoding="utf-8")
    chinese = (PAPER_ROOT / "main_zh.tex").read_text(encoding="utf-8")
    supplement_en = (PROJECT_ROOT / "supplementary" / "supplement_en.tex").read_text(encoding="utf-8")
    supplement_zh = (PROJECT_ROOT / "supplementary" / "supplement_zh.tex").read_text(encoding="utf-8")
    combined_main = english + "\n" + chinese
    require("0.058" not in combined_main, "superseded p=0.058 is absent from both manuscripts", failures)
    require("Holm$_{12}$" in combined_main, "both manuscripts expose the 12-test family", failures)
    require(
        not re.search(r"negative (evidence|result|finding)", english, flags=re.IGNORECASE),
        "English manuscript avoids negative-evidence/result shorthand",
        failures,
    )
    for obsolete in (
        "table_endpoint_selection_confirmation",
        "table_core_asym_paired",
        "table_core_asym_decision_cost",
        "table_stress_auc_direct_predecessor",
        "table_core_rast_fixed_full",
        "table_stress_decision_cost_rho",
    ):
        require(
            obsolete not in supplement_en and obsolete not in supplement_zh,
            f"rendered supplements exclude obsolete {obsolete}",
            failures,
        )
    require(
        "\\tableofcontents\n\\clearpage" in supplement_en
        and "\\tableofcontents\n\\clearpage" in supplement_zh,
        "supplement titles precede all floats",
        failures,
    )

    required = [
        PROJECT_ROOT / "paper_outputs" / "round3_evidence" / "round3_analysis_metadata.json",
        PROJECT_ROOT / "paper_outputs" / "round3_evidence" / "round3_multiplicity_registry.csv",
        PROJECT_ROOT / "paper_outputs" / "round3_evidence" / "round3_small_cluster_calibration.csv",
        PROJECT_ROOT / "paper_outputs" / "round3_evidence" / "mask_observation_sensitivity_summary.csv",
        PROJECT_ROOT / "paper_outputs" / "round3_evidence" / "inference_latency_protocol_summary.csv",
        PROJECT_ROOT / "docs" / "analysis_chronology.md",
        PROJECT_ROOT / "docs" / "round3_revision_decisions.md",
    ]
    for path in required:
        require(path.exists() and path.stat().st_size > 0, f"required artifact exists: {path.name}", failures)

    if failures:
        print(f"ROUND3_CONSISTENCY_FAIL count={len(failures)}")
        for failure in failures:
            print(f"- {failure}")
        raise SystemExit(1)
    print("ROUND3_CONSISTENCY_PASS")


if __name__ == "__main__":
    main()
