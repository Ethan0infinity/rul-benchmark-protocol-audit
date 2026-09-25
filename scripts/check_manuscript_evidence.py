from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).absolute().parents[1]
OUTPUT = PROJECT_ROOT / "paper_outputs" / "manuscript_v3"
ADVANCED_MANIFEST = PROJECT_ROOT / "paper_outputs" / "advanced_evidence" / "advanced_evidence_manifest.json"

REQUIRED = [
    "numbers.tex",
    "evidence_ledger.csv",
    "table_cmapss_macro.tex",
    "table_efficiency.tex",
    "table_benchmark_fairness.tex",
    "table_late_tail_risk.tex",
    "table_threshold_paired_ci.tex",
    "table_ablation_fd004.tex",
    "table_stress_auc_direct_predecessor.tex",
    "table_ncmapss.tex",
    "table_interval_calibration.tex",
    "table_conformal_calibration.tex",
    "table_design_sensitivity.tex",
    "table_hierarchical_direct_predecessor.tex",
    "table_close_prior_fd002.tex",
    "algorithm_training_inference.tex",
    "results_section_en.tex",
    "results_section_zh.tex",
    "table_main_analysis_family_registry.tex",
    "table_main_analysis_family_registry_zh.tex",
    "table_round16_crossed_level_selection.tex",
    "table_round16_crossed_level_selection_zh.tex",
    "table_round16_preprocessing_resources.tex",
    "table_round16_preprocessing_resources_zh.tex",
    "table_round16_lofo_estimand_comparison.tex",
    "table_round16_lofo_estimand_comparison_zh.tex",
    "table_round16_lofo_event_support.tex",
    "table_round16_lofo_event_support_zh.tex",
    "table_round19_lofo_stream_support.tex",
    "table_round19_lofo_stream_support_zh.tex",
    "table_round19_governance_workflow.tex",
    "table_round19_governance_workflow_zh.tex",
    "table_round19_retrospective_dashboard.tex",
    "table_round19_retrospective_dashboard_zh.tex",
    "table_preference_selection_stability.tex",
    "table_preference_selection_stability_zh.tex",
    "table_cmapss_macro_zh.tex",
    "table_late_tail_risk_zh.tex",
    "table_ablation_fd004_zh.tex",
    "table_stress_auc_direct_predecessor_zh.tex",
    "table_ncmapss_zh.tex",
    "table_conformal_calibration_zh.tex",
    "table_efficiency_zh.tex",
    "table_close_prior_fd002_zh.tex",
    "table_hierarchical_direct_predecessor_zh.tex",
    "table_checkpoint_sensitivity.tex",
    "table_checkpoint_sensitivity_zh.tex",
    "table_protocol_track_sensitivity.tex",
    "table_protocol_track_sensitivity_zh.tex",
    "table_fixed_subset_direct.tex",
    "table_fixed_subset_direct_zh.tex",
    "table_ocm_attribution.tex",
    "table_ocm_attribution_zh.tex",
    "table_ncmapss_units_direct.tex",
    "table_ncmapss_units_direct_zh.tex",
    "table_ncmapss_split_sensitivity.tex",
    "table_ncmapss_split_sensitivity_zh.tex",
    "table_hyperparameter_sensitivity.tex",
    "table_hyperparameter_sensitivity_zh.tex",
    "table_condition_k_diagnostics.tex",
    "table_condition_k_diagnostics_zh.tex",
    "table_track_b_absolute.tex",
    "table_track_b_absolute_zh.tex",
    "table_architecture_attribution_absolute.tex",
    "table_architecture_attribution_absolute_zh.tex",
    "table_architecture_selected_comparators.tex",
    "table_architecture_selected_comparators_zh.tex",
    "table_core_asym_cmapss.tex",
    "table_core_asym_cmapss_zh.tex",
    "table_core_asym_paired.tex",
    "table_core_asym_paired_zh.tex",
    "table_validation_endpoint_audit.tex",
    "table_validation_endpoint_audit_zh.tex",
    "table_stress_family_uncertainty.tex",
    "table_stress_family_uncertainty_zh.tex",
    "table_metric_best_comparators.tex",
    "table_metric_best_comparators_zh.tex",
    "table_ncmapss_k_sensitivity.tex",
    "table_ncmapss_k_sensitivity_zh.tex",
    "table_endpoint_selection_confirmation.tex",
    "table_endpoint_selection_confirmation_zh.tex",
    "table_validation_endpoint_distribution.tex",
    "table_validation_endpoint_distribution_zh.tex",
    "table_endpoint_confirmation_interval_directions.tex",
    "table_endpoint_confirmation_interval_directions_zh.tex",
    "table_validation_endpoint_paired_directions.tex",
    "table_validation_endpoint_paired_directions_zh.tex",
    "table_ncmapss_validation_k_selection.tex",
    "table_official_dual_mixer.tex",
    "table_official_dual_mixer_zh.tex",
    "table_official_dual_mixer_summary.tex",
    "table_official_dual_mixer_summary_zh.tex",
    "table_ncmapss_validation_k_selection_zh.tex",
    "table_training_augmentation_protocol.tex",
    "table_training_augmentation_protocol_zh.tex",
    "table_stress_generator_protocol.tex",
    "table_stress_generator_protocol_zh.tex",
    "table_stress_equivalence_sensitivity.tex",
    "table_stress_equivalence_sensitivity_zh.tex",
    "table_cross_backbone_protocol_buildup.tex",
    "table_cross_backbone_protocol_buildup_zh.tex",
    "table_core_asym_decision_cost.tex",
    "table_core_asym_decision_cost_zh.tex",
    "table_endpoint_epoch_agreement.tex",
    "table_endpoint_epoch_agreement_zh.tex",
    "table_stress_operating_points.tex",
    "table_stress_operating_points_zh.tex",
]


def main() -> None:
    prerequisite_markers = [OUTPUT / "numbers.tex", OUTPUT / "evidence_ledger.csv"]
    if not OUTPUT.exists() or not all(path.exists() for path in prerequisite_markers):
        print("MANUSCRIPT_EVIDENCE_PRECONDITION_FAIL")
        print(
            "- Required predecessor has not run: "
            "python scripts/build_manuscript_evidence.py"
        )
        print(
            "- Recommended ordered wrapper: "
            "python scripts/run_stored_results_pipeline.py --resume"
        )
        raise SystemExit(2)
    issues = []
    if not ADVANCED_MANIFEST.exists():
        issues.append("missing advanced evidence manifest")
    else:
        manifest = json.loads(ADVANCED_MANIFEST.read_text(encoding="utf-8-sig"))
        if manifest.get("protocol_version") != "3.0" or int(manifest.get("formal_run_count", 0)) != 260:
            issues.append("advanced evidence manifest is not a complete protocol-v3 artifact")
    for filename in REQUIRED:
        path = OUTPUT / filename
        if not path.exists() or path.stat().st_size == 0:
            issues.append(f"missing/empty {filename}")
    numbers = OUTPUT / "numbers.tex"
    if numbers.exists():
        text = numbers.read_text(encoding="utf-8-sig")
        for token in (r"{\FormalRunCount}{260}", r"{\EvaluatedModelCount}{13}", r"{\BaselineModelCount}{12}"):
            if token not in text:
                issues.append(f"numbers.tex missing locked token {token}")
        if "TBD" in text or "paper_main_v2" in text:
            issues.append("numbers.tex contains a placeholder or obsolete protocol token")
        if ADVANCED_MANIFEST.exists() and numbers.stat().st_mtime < ADVANCED_MANIFEST.stat().st_mtime:
            issues.append("numbers.tex predates the advanced evidence manifest")
    ledger = OUTPUT / "evidence_ledger.csv"
    if ledger.exists():
        frame = pd.read_csv(ledger)
        if len(frame) < 10 or frame[["macro", "source_file", "selector", "value"]].isna().any().any():
            issues.append("evidence ledger is incomplete")
    forbidden_visible_terms = {
        "accuracy-safety": "obsolete accuracy-safety terminology",
        "absolute corrupted-curve auc": "obsolete unnormalized stress-AUC wording",
        "absolute corrupted lpr-auc": "obsolete unnormalized LPR-AUC wording",
    }
    for path in OUTPUT.glob("*.tex"):
        content = path.read_text(encoding="utf-8-sig").lower()
        for token, description in forbidden_visible_terms.items():
            if token in content:
                issues.append(f"{path.name} contains {description}")
    if issues:
        print("MANUSCRIPT_EVIDENCE_FAIL")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)
    print(f"MANUSCRIPT_EVIDENCE_PASS files={len(REQUIRED)}")


if __name__ == "__main__":
    main()
