from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).absolute().parents[1]
OUTPUT = PROJECT_ROOT / "paper_outputs" / "advanced_evidence"


REQUIRED_CSV = {
    "aggregation_summary.csv": 13,
    "benchmark_rank_summary.csv": 13,
    "benchmark_fairness.csv": 13,
    "late_tail_risk_summary.csv": 13,
    "pareto_analysis.csv": 13,
    "pareto_weight_sensitivity.csv": 13,
    "pareto_seed_stability_long.csv": 65,
    "pareto_seed_stability_summary.csv": 13,
    "interval_calibration_seed_subset.csv": 40,
    "interval_calibration_summary.csv": 2,
    "threshold_rank_stability_fd004.csv": 13,
    "threshold_paired_bootstrap.csv": 768,
    "ablation_fd004_multiseed_summary.csv": 13,
    "hierarchical_bootstrap_bca.csv": 48,
    "stress_multiseed_summary.csv": 1,
    "stress_degradation_auc_summary.csv": 1,
    "reliability_fault_localization_seed_level.csv": 100,
    "reliability_fault_localization_summary.csv": 4,
    "reliability_calibration_bins.csv": 20,
    "attention_mechanism_summary.csv": 2,
    "ncmapss_ds02_runs.csv": 65,
    "ncmapss_ds02_summary.csv": 13,
    "decision_cost_sensitivity.csv": 52,
    "design_sensitivity_seed_level.csv": 25,
    "design_sensitivity_summary.csv": 5,
    "condition_normalization_diagnostics.csv": 2,
    "condition_normalization_grouped_cv.csv": 10,
    "degradation_stage_grouped_cv.csv": 10,
    "degradation_stage_separability_summary.csv": 2,
    "degradation_rul_grouped_cv.csv": 10,
    "degradation_rul_separability_summary.csv": 2,
    "failure_case_engine_level.csv": 1000,
    "failure_case_summary.csv": 16,
    "failure_case_largest_regressions.csv": 200,
    "close_prior_fd002_runs.csv": 10,
    "close_prior_fd002_summary.csv": 2,
    "close_prior_fd002_engine_pairs.csv": 1000,
    "close_prior_fd002_paired_bootstrap.csv": 2,
    "conformal_interval_seed_subset.csv": 40,
    "conformal_interval_summary.csv": 2,
    "conformal_interval_predictions.csv": 1000,
    "asymmetry_fd004_seed_metrics.csv": 10,
    "asymmetry_fd004_summary.csv": 2,
    "asymmetry_fd004_paired_bootstrap.csv": 8,
    "asymmetry_fd004_conformal_seed.csv": 10,
    "asymmetry_fd004_conformal_summary.csv": 2,
    "subset_engine_paired_bootstrap.csv": 192,
    "ncmapss_ds02_per_unit_seed.csv": 195,
    "ncmapss_ds02_per_unit_summary.csv": 39,
    "interval_reliability_seed_subset.csv": 240,
    "interval_reliability_summary.csv": 12,
    "simplification_ncmapss_seed_metrics.csv": 10,
    "simplification_ncmapss_summary.csv": 2,
    "simplification_ncmapss_paired_bootstrap.csv": 3,
    "checkpoint_sensitivity_seed_metrics.csv": 10,
    "checkpoint_sensitivity_summary.csv": 2,
    "checkpoint_sensitivity_paired_bootstrap.csv": 4,
    "standard_protocol_track_seed_metrics.csv": 60,
    "standard_protocol_track_summary.csv": 12,
    "standard_protocol_track_comparison.csv": 6,
    "standard_protocol_track_paired_bootstrap.csv": 18,
    "ocm_attribution_seed_metrics.csv": 20,
    "ocm_attribution_summary.csv": 4,
    "ocm_attribution_paired_bootstrap.csv": 12,
    "hyperparameter_sensitivity_seed_metrics.csv": 40,
    "hyperparameter_sensitivity_summary.csv": 8,
    "condition_cluster_k_diagnostics_seed.csv": 25,
    "condition_cluster_k_cross_seed_stability.csv": 50,
    "condition_cluster_k_diagnostics_summary.csv": 5,
    "stress_pair_extended_seed_level.csv": 1550,
    "stress_pair_extended_auc_seed_level.csv": 450,
    "stress_pair_extended_worst_seed_level.csv": 450,
    "stress_pair_extended_auc_comparison.csv": 99,
    "stress_pair_extended_worst_comparison.csv": 99,
    "stress_family_uncertainty.csv": 99,
    "stress_equivalence_margin_sensitivity.csv": 54,
    "architecture_attribution_absolute.csv": 5,
    "architecture_selected_comparator_bootstrap.csv": 8,
    "track_b_absolute_ranking.csv": 6,
    "ocm_core_asym_subset_seed.csv": 40,
    "ocm_core_asym_summary.csv": 2,
    "ocm_core_asym_paired_bootstrap.csv": 4,
    "validation_endpoint_audit.csv": 20,
    "benchmark_efficiency_enhanced.csv": 13,
    "selected_comparator_evidence.csv": 12,
    "ncmapss_k_sensitivity_seed_metrics.csv": 15,
    "ncmapss_k_sensitivity_summary.csv": 3,
    "stress_global_missing_failure_curve.csv": 12,
    "ncmapss_split_sensitivity_seed_metrics.csv": 30,
    "ncmapss_split_sensitivity_summary.csv": 6,
    "endpoint_selection_confirmation_seed_metrics.csv": 90,
    "endpoint_selection_confirmation_summary.csv": 18,
    "endpoint_selection_confirmation_bootstrap.csv": 54,
    "validation_endpoint_distribution_targets.csv": 6500,
    "validation_endpoint_distribution_summary.csv": 6,
    "ncmapss_validation_only_k_selection.csv": 3,
    "ncmapss_validation_selected_k_seed_metrics.csv": 5,
    "ncmapss_validation_only_k_verdict.csv": 1,
    "official_dual_mixer_seed_metrics.csv": 20,
    "official_dual_mixer_summary.csv": 4,
    "official_dual_mixer_paired_bootstrap.csv": 12,
    "official_dual_mixer_operating_points.csv": 24,
    "core_asym_major_comparator_bootstrap.csv": 18,
    "core_asym_rast_subset_bootstrap.csv": 24,
    "core_asym_decision_cost_bootstrap.csv": 24,
    "endpoint_selection_task_summary.csv": 18,
    "endpoint_selection_epoch_agreement.csv": 6,
    "cross_backbone_protocol_buildup_seed_metrics.csv": 60,
    "cross_backbone_protocol_buildup_summary.csv": 12,
    "cross_backbone_protocol_buildup_transitions.csv": 40,
    "stress_operating_points_curve_means.csv": 675,
    "stress_operating_points_uncertainty.csv": 198,
    "stress_engine_level_predictions.csv": 500000,
    "stress_operating_points_engine_bootstrap.csv": 162,
    "fast_cudnn_repeat_seed_metrics.csv": 24,
    "fast_cudnn_repeat_summary.csv": 16,
    "tcn_gru_seed_audit.csv": 20,
    "tcn_gru_subset_robust_summary.csv": 4,
    "run_quality_cell_audit.csv": 260,
    "run_quality_model_robust_summary.csv": 13,
    "run_quality_leave_one_seed_out.csv": 65,
    "run_quality_component_trajectories.csv": 100,
    "fd002_protocol_transfer_seed_metrics.csv": 25,
    "fd002_protocol_transfer_summary.csv": 5,
    "fd002_protocol_transfer_paired_bootstrap.csv": 20,
}

STRESS_OPERATING_POINTS = {"Core", "Asym"}
STRESS_SCENARIOS = {
    "gaussian_noise",
    "global_sensor_missing",
    "window_random_missing",
    "block_missing",
    "correlated_group_missing",
    "sensor_drift",
    "sensor_bias_shift",
    "stuck_at_fault",
    "burst_noise",
}
STRESS_ENGINE_METRICS = {
    "rmse",
    "critical_30_late_prediction_ratio",
    "critical_30_signed_error_mean",
    "critical_30_early_error_magnitude",
    "critical_30_mean_late_excess",
    "critical_30_late_cvar95",
    "critical_30_decision_cost_2",
    "critical_30_decision_cost_5",
    "critical_30_decision_cost_10",
}

REQUIRED_COLUMNS = {
    "endpoint_selection_confirmation_bootstrap.csv": {
        "subset",
        "endpoint_strategy",
        "left_variant",
        "right_variant",
        "metric",
        "mean_difference_left_minus_right",
        "percentile_ci95_low",
        "percentile_ci95_high",
    },
    "validation_endpoint_distribution_summary.csv": {
        "subset",
        "endpoint_strategy",
        "endpoint_count",
        "critical_30_fraction",
        "wasserstein_distance_to_test",
        "test_critical_30_fraction",
    },
    "official_dual_mixer_paired_bootstrap.csv": {
        "subset",
        "metric",
        "dual_mean",
        "dual_std",
        "ocm_mean",
        "ocm_std",
        "mean_paired_difference_dual_minus_ocm",
        "percentile_ci95_low",
        "percentile_ci95_high",
    },
    "official_dual_mixer_operating_points.csv": {
        "ocm_variant",
        "subset",
        "metric",
        "mean_paired_difference_dual_minus_ocm",
        "percentile_ci95_low",
        "percentile_ci95_high",
        "holm_adjusted_p",
    },
    "cross_backbone_protocol_buildup_transitions.csv": {
        "model",
        "left_stage",
        "right_stage",
        "metric",
        "mean_difference_right_minus_left",
        "percentile_ci95_low",
        "percentile_ci95_high",
    },
    "stress_operating_points_engine_bootstrap.csv": {
        "operating_point",
        "scenario",
        "metric",
        "mean_difference_point_minus_rast",
        "hierarchical_bootstrap_ci95_low",
        "hierarchical_bootstrap_ci95_high",
        "test_engine_count",
        "inference_unit",
    },
    "core_asym_rast_subset_bootstrap.csv": {
        "ocm_variant",
        "comparator",
        "subset",
        "metric",
        "mean_difference_ocm_minus_rast",
        "percentile_ci95_low",
        "percentile_ci95_high",
        "inference_unit",
    },
    "fast_cudnn_repeat_summary.csv": {
        "model",
        "stage",
        "metric",
        "fixed_seed_repeat_sd_mean",
        "cross_seed_sd_original_grid",
        "repeat_to_cross_seed_sd_ratio",
    },
    "tcn_gru_seed_audit.csv": {
        "subset",
        "seed",
        "test_rmse",
        "best_checkpoint_epoch",
        "minimum_validation_loss_epoch",
        "checkpoint_selection_reversal",
        "prediction_has_nan_or_inf",
        "catastrophic_cell",
    },
    "fd002_protocol_transfer_paired_bootstrap.csv": {
        "comparison",
        "left",
        "right",
        "metric",
        "mean_left_minus_right",
        "ci95_low",
        "ci95_high",
        "classification",
        "bootstrap_primary_unit",
    },
}


def main() -> None:
    issues = []
    counts = {}
    for filename, minimum in REQUIRED_CSV.items():
        path = OUTPUT / filename
        if not path.exists():
            issues.append(f"missing {filename}")
            continue
        frame = pd.read_csv(path)
        counts[filename] = len(frame)
        if len(frame) < minimum:
            issues.append(f"{filename} rows={len(frame)} minimum={minimum}")
        missing_columns = REQUIRED_COLUMNS.get(filename, set()) - set(frame.columns)
        if missing_columns:
            issues.append(f"{filename} missing columns={sorted(missing_columns)}")

    reliability_seed_path = OUTPUT / "reliability_fault_localization_seed_level.csv"
    reliability_summary_path = OUTPUT / "reliability_fault_localization_summary.csv"
    if reliability_seed_path.exists() and reliability_summary_path.exists():
        reliability_seed = pd.read_csv(reliability_seed_path)
        reliability_summary = pd.read_csv(reliability_summary_path)
        expected_seeds = {42, 123, 2024, 2025, 2026}
        expected_rates = {0.1, 0.2, 0.3, 0.4}
        training_seeds = set(reliability_seed["training_seed"].astype(int))
        perturbation_seeds = set(reliability_seed["perturbation_seed"].astype(int))
        seed_rates = set(reliability_seed["missing_rate"].round(6))
        summary_rates = set(reliability_summary["missing_rate"].round(6))
        if training_seeds != expected_seeds:
            issues.append(f"reliability training seeds={sorted(training_seeds)} expected={sorted(expected_seeds)}")
        if perturbation_seeds != expected_seeds:
            issues.append(
                f"reliability perturbation seeds={sorted(perturbation_seeds)} expected={sorted(expected_seeds)}"
            )
        if seed_rates != expected_rates or summary_rates != expected_rates:
            issues.append(
                "reliability missing-rate grid must equal [0.1, 0.2, 0.3, 0.4] "
                f"(seed-level={sorted(seed_rates)}, summary={sorted(summary_rates)})"
            )
        expected_combinations = len(expected_seeds) * len(expected_seeds) * len(expected_rates)
        unique_combinations = reliability_seed[
            ["training_seed", "perturbation_seed", "missing_rate"]
        ].drop_duplicates()
        if len(unique_combinations) != expected_combinations or len(reliability_seed) != expected_combinations:
            issues.append(
                f"reliability grid rows={len(reliability_seed)} unique={len(unique_combinations)} "
                f"expected={expected_combinations}"
            )

    legacy_stress_path = OUTPUT / "stress_family_uncertainty.csv"
    operating_stress_path = OUTPUT / "stress_operating_points_uncertainty.csv"
    if legacy_stress_path.exists() and operating_stress_path.exists():
        legacy = pd.read_csv(legacy_stress_path)[
            ["scenario", "metric", "nested_bootstrap_ci95_low", "nested_bootstrap_ci95_high"]
        ]
        operating = pd.read_csv(operating_stress_path)
        asym = operating[operating["operating_point"] == "Asym"][
            ["scenario", "metric", "nested_bootstrap_ci95_low", "nested_bootstrap_ci95_high"]
        ]
        paired = legacy.merge(asym, on=["scenario", "metric"], suffixes=("_legacy", "_operating"), validate="one_to_one")
        if len(paired) != len(legacy):
            issues.append(f"Asym stress CI rows={len(paired)} expected={len(legacy)}")
        for bound in ("low", "high"):
            gap = (
                paired[f"nested_bootstrap_ci95_{bound}_legacy"]
                - paired[f"nested_bootstrap_ci95_{bound}_operating"]
            ).abs().max()
            if float(gap) > 1e-12:
                issues.append(f"Asym stress {bound} CI mismatch max_abs={gap}")
    engine_stress_path = OUTPUT / "stress_operating_points_engine_bootstrap.csv"
    if engine_stress_path.exists():
        engine_stress = pd.read_csv(engine_stress_path)
        if set(engine_stress["operating_point"]) != STRESS_OPERATING_POINTS:
            issues.append("engine-level stress evidence must contain Core and Asym")
        if set(engine_stress["scenario"]) != STRESS_SCENARIOS:
            issues.append("engine-level stress evidence does not match the declared nine-scenario grid")
        if set(engine_stress["metric"]) != STRESS_ENGINE_METRICS:
            issues.append("engine-level stress evidence does not match the declared nine-metric grid")
        if set(engine_stress["test_engine_count"].astype(int)) != {248}:
            issues.append("engine-level stress evidence must resample 248 matched FD004 engines")
        expected_cells = len(STRESS_OPERATING_POINTS) * len(STRESS_SCENARIOS) * len(STRESS_ENGINE_METRICS)
        unique_cells = engine_stress[["operating_point", "scenario", "metric"]].drop_duplicates()
        if len(unique_cells) != expected_cells or len(engine_stress) != expected_cells:
            issues.append(
                f"engine-level stress cells={len(engine_stress)} unique={len(unique_cells)} expected={expected_cells}"
            )
    required_figures = [
        OUTPUT / "figures" / "fig_method_architecture.pdf",
        OUTPUT / "figures" / "fig_critical_difference.pdf",
        OUTPUT / "figures" / "fig_fd004_threshold_rank_heatmap.pdf",
        OUTPUT / "figures" / "fig_mechanism_diagnostics.pdf",
        OUTPUT / "figures" / "fig_decision_cost_sensitivity.pdf",
        OUTPUT / "figures" / "fig_core_asym_decision_cost.pdf",
        OUTPUT / "figures" / "fig_cross_backbone_protocol_buildup.pdf",
        OUTPUT / "figures" / "fig_stress_operating_points.pdf",
        OUTPUT / "figures" / "fig_interval_calibration.pdf",
        OUTPUT / "figures" / "fig_condition_normalization_pca.pdf",
        OUTPUT / "figures" / "fig_fd004_failure_cases.pdf",
        OUTPUT / "figures" / "fig_conformal_interval_calibration.pdf",
        OUTPUT / "figures" / "fig_interval_reliability_curve.pdf",
        OUTPUT / "figures" / "fig_stress_nine_family_heatmap.pdf",
        OUTPUT / "figures" / "fig_stress_global_missing_failure.pdf",
        OUTPUT / "figures" / "fig_validation_endpoint_distribution.pdf",
        PROJECT_ROOT / "paper_outputs" / "revised_figures" / "fig_fd004_random_missing_rmse.pdf",
        PROJECT_ROOT / "paper_outputs" / "revised_figures" / "fig_fd004_random_missing_lpr30.pdf",
        PROJECT_ROOT / "paper_outputs" / "revised_figures" / "fig_fd004_global_missing_rmse.pdf",
        PROJECT_ROOT / "paper_outputs" / "revised_figures" / "fig_fd004_block_missing_lpr30.pdf",
        PROJECT_ROOT / "paper_outputs" / "revised_figures" / "fig_fd004_drift_lpr30.pdf",
        PROJECT_ROOT / "paper_outputs" / "revised_figures" / "fig_fd004_drift_nasa.pdf",
    ]
    for path in required_figures:
        if not path.exists() or path.stat().st_size < 1000:
            issues.append(f"missing/empty figure {path.name}")
    manifest_path = OUTPUT / "advanced_evidence_manifest.json"
    if not manifest_path.exists():
        issues.append("missing advanced_evidence_manifest.json")
    else:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
        if manifest.get("protocol_version") != "3.0":
            issues.append(f"unexpected advanced evidence protocol {manifest.get('protocol_version')}")
        if int(manifest.get("formal_run_count", 0)) != 260:
            issues.append(f"advanced evidence formal_run_count={manifest.get('formal_run_count')} expected=260")
        if int(manifest.get("ablation_fd004_multiseed_rows", 0)) != 65:
            issues.append(
                f"advanced evidence ablation rows={manifest.get('ablation_fd004_multiseed_rows')} expected=65"
            )
    if not (OUTPUT / "figure_data_manifest.json").exists():
        issues.append("missing figure_data_manifest.json")
    generated = PROJECT_ROOT / "paper_outputs" / "manuscript_v3"
    nan_tables = []
    for path in generated.glob("*.tex"):
        text = path.read_text(encoding="utf-8-sig")
        if re.search(r"(?:^|&)\s*nan\s*(?:&|\\\\)", text, flags=re.IGNORECASE | re.MULTILINE):
            nan_tables.append(path.name)
    if nan_tables:
        issues.append(f"generated LaTeX contains bare nan cells: {nan_tables}")
    report = {"counts": counts, "issues": issues}
    (PROJECT_ROOT / "reports" / "advanced_evidence_gate.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    if issues:
        print("ADVANCED_EVIDENCE_FAIL")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)
    print(f"ADVANCED_EVIDENCE_PASS files={len(REQUIRED_CSV)} figures={len(required_figures)}")


if __name__ == "__main__":
    main()
