from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
OUTPUT = PROJECT_ROOT / "paper_outputs" / "round12_new_evidence"
EXPECTED = {
    "independent": 80,
    "crossed": 72,
    "cmapss_preprocessing": 30,
    "preference": 90,
    "factorial": 40,
    "lofo": 40,
    "ncmapss_sampling_window": 45,
    "ncmapss_horizon_matched": 15,
}


def verify_training_manifest(
    family: str,
    expected_rows: int,
    *,
    epochs: int,
) -> list[str]:
    path = OUTPUT / f"{family}_run_manifest.csv"
    if not path.exists():
        return [f"missing manifest: {path}"]
    frame = pd.read_csv(path)
    issues = []
    if len(frame) != expected_rows:
        issues.append(f"{family}: expected {expected_rows} rows, found {len(frame)}")
    if "planned_epochs" in frame and not (
        pd.to_numeric(frame["planned_epochs"]) == epochs
    ).all():
        issues.append(f"{family}: non-{epochs}-epoch manifest rows")
    for row in frame.to_dict("records"):
        run_dir = PROJECT_ROOT / str(row["run_dir"])
        metrics_path = run_dir / "metrics.json"
        checkpoint_path = run_dir / "best_model.pt"
        predictions_path = run_dir / "test_predictions.csv"
        history_path = run_dir / "epoch_history.csv"
        if not all(
            path.exists()
            for path in (
                metrics_path,
                checkpoint_path,
                predictions_path,
                history_path,
            )
        ):
            issues.append(f"{family}: incomplete run {run_dir}")
            continue
        metrics = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
        if int(metrics.get("planned_epochs", -1)) != epochs:
            issues.append(f"{family}: wrong planned epochs {run_dir}")
        if (
            family.startswith("ncmapss_")
            and metrics.get("result_status") != "computational_proxy_sensitivity"
        ):
            issues.append(f"{family}: incorrect evidence status in {run_dir}")
        formal_reuse = str(row.get("artifact_reuse_source", "")).startswith(
            "formal_composite_seed"
        )
        if not family.startswith("ncmapss_"):
            expected_status = "formal" if formal_reuse else "supplementary"
            if (
                metrics.get("result_status") != expected_status
                or metrics.get("allowed_for_paper") is not True
            ):
                issues.append(
                    f"{family}: result status is not {expected_status} in {run_dir}"
                )
        try:
            checkpoint = torch.load(
                checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
        except Exception as exc:
            issues.append(f"{family}: unreadable checkpoint {run_dir}: {exc}")
            continue
        if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
            issues.append(f"{family}: invalid checkpoint {run_dir}")
            continue
        required_metrics = (
            "test_rmse",
            "test_nasa_score",
            "test_critical_30_late_prediction_ratio",
            "best_epoch",
        )
        for metric in required_metrics:
            try:
                value = float(metrics[metric])
            except (KeyError, TypeError, ValueError):
                issues.append(f"{family}: missing numeric {metric} in {run_dir}")
                continue
            if not np.isfinite(value):
                issues.append(f"{family}: non-finite {metric} in {run_dir}")
        predictions = pd.read_csv(predictions_path)
        numeric_predictions = predictions.select_dtypes(include=[np.number])
        if numeric_predictions.empty or not np.isfinite(
            numeric_predictions.to_numpy(dtype=float)
        ).all():
            issues.append(f"{family}: non-finite predictions in {run_dir}")
        history = pd.read_csv(history_path)
        score_column = next(
            (
                column
                for column in ("selection_score", "val_risk_score")
                if column in history
            ),
            None,
        )
        if score_column is None or "epoch" not in history:
            issues.append(f"{family}: checkpoint score history missing in {run_dir}")
        else:
            score = pd.to_numeric(history[score_column], errors="coerce")
            if score.isna().all():
                issues.append(f"{family}: checkpoint scores non-numeric in {run_dir}")
            else:
                objective_epoch = int(
                    history.loc[score.idxmin(), "epoch"]
                )
                stored_epoch = int(metrics["best_epoch"])
                if stored_epoch != objective_epoch:
                    issues.append(
                        f"{family}: checkpoint objective mismatch in {run_dir}: "
                        f"stored={stored_epoch}, minimum={objective_epoch}"
                    )
    return issues


def verify_designs() -> list[str]:
    issues = []
    independent = pd.read_csv(OUTPUT / "independent_run_manifest.csv")
    if independent["training_stream_seed"].nunique() != 10:
        issues.append("independent: expected ten separately seeded training streams")
    if set(independent["split_seed"]) != {42}:
        issues.append("independent: split must remain fixed at 42")
    if set(independent["point"]) != {"rast", "asym"}:
        issues.append("independent: direct-comparator points are incomplete")
    if not (
        independent.groupby(["subset", "point"])["training_stream_seed"].nunique()
        == 10
    ).all():
        issues.append("independent: every task/configuration must have ten streams")

    crossed = pd.read_csv(OUTPUT / "crossed_run_manifest.csv")
    if set(crossed["split_seed"]) != {42, 123, 2024}:
        issues.append("crossed: expected split seeds 42, 123, and 2024")
    if set(crossed["training_stream_seed"]) != {31415, 27182, 16180}:
        issues.append("crossed: expected three separately controlled training streams")
    crossed_cell_counts = crossed.groupby(["subset", "point"]).size()
    if not (crossed_cell_counts == 9).all():
        issues.append("crossed: every task/configuration must contain a full 3x3 grid")
    if len(crossed[["split_seed", "training_stream_seed"]].drop_duplicates()) != 9:
        issues.append("crossed: split and training stream are not fully crossed")

    preprocessing = pd.read_csv(OUTPUT / "cmapss_preprocessing_run_manifest.csv")
    if set(preprocessing["subset"]) != {"FD004"}:
        issues.append("cmapss preprocessing: expected the direct FD004 comparator")
    if set(preprocessing["training_stream_seed"]) != {42, 123, 2024}:
        issues.append("cmapss preprocessing: expected three composite-seed repetitions")
    expected_points = {
        ("window", 20, 20, 14),
        ("reference", 30, 30, 14),
        ("window", 50, 50, 14),
        ("sensor_budget", 8, 30, 8),
        ("sensor_budget", 21, 30, 21),
    }
    actual_points = set(
        zip(
            preprocessing["audit_axis"],
            preprocessing["audit_level"],
            preprocessing["window_size"],
            preprocessing["sensor_budget"],
            strict=True,
        )
    )
    if actual_points != expected_points:
        issues.append("cmapss preprocessing: one-factor design points are incomplete")
    if not (
        preprocessing.groupby(["subset", "point", "training_stream_seed"]).size()
        == 5
    ).all():
        issues.append("cmapss preprocessing: each task/configuration/stream needs five points")

    preference = pd.read_csv(OUTPUT / "preference_run_manifest.csv")
    if preference[["split_seed", "training_stream_seed"]].drop_duplicates().shape[0] != 9:
        issues.append("preference: expected 3x3 split-by-stream cells")
    candidates = preference[preference["point"] == "asym"]
    references = preference[preference["point"] == "rast_reference"]
    if candidates["candidate"].nunique() != 9:
        issues.append("preference: expected nine Asym preference candidates")
    if not (
        candidates.groupby(["split_seed", "training_stream_seed"])[
            "candidate"
        ].nunique()
        == 9
    ).all():
        issues.append("preference: one or more split-stream grids are incomplete")
    if len(references) != 9 or not (
        references.groupby(["split_seed", "training_stream_seed"]).size() == 1
    ).all():
        issues.append("preference: expected one matched RAST reference per grid")

    factorial = pd.read_csv(OUTPUT / "factorial_run_manifest.csv")
    cells = set(
        zip(
            factorial["backbone"],
            factorial["risk_factor"],
            factorial["consistency_factor"],
            strict=True,
        )
    )
    expected_cells = {
        (backbone, risk, consistency)
        for backbone in ("rast", "ocm")
        for risk in (0, 1)
        for consistency in (0, 1)
    }
    if cells != expected_cells:
        issues.append(
            "factorial: incomplete 2x2x2 backbone-risk-consistency cells"
        )
    if factorial["training_stream_seed"].nunique() != 5:
        issues.append("factorial: expected five training streams")
    if not (
        factorial.groupby("training_stream_seed").size() == 8
    ).all():
        issues.append("factorial: every training stream must contain eight cells")

    lofo = pd.read_csv(OUTPUT / "lofo_run_manifest.csv")
    if set(lofo["held_out_family"]) != {
        "noise",
        "random_missingness",
        "block_missingness",
        "drift",
    }:
        issues.append(
            "lofo: expected noise, random-missingness, block-missingness, "
            "and drift families"
        )
    if lofo["training_stream_seed"].nunique() != 5:
        issues.append("lofo: expected five training streams")
    if not (
        lofo.groupby(["held_out_family", "point"])[
            "training_stream_seed"
        ].nunique()
        == 5
    ).all():
        issues.append("lofo: every family/configuration must have five streams")
    exposure_path = OUTPUT / "lofo_active_exposure_matrix.csv"
    if not exposure_path.exists():
        issues.append("lofo: active-exposure matrix is missing")
    else:
        exposure = pd.read_csv(exposure_path)
        noise = exposure[exposure["held_out_family"] == "noise"]
        if len(noise) != 10:
            issues.append("lofo noise: expected ten point-stream exposure rows")
        for column in (
            "primary_gaussian_noise_std",
            "consistency_weight",
            "consistency_gaussian_noise_std",
        ):
            if column not in noise or not np.isclose(noise[column], 0.0).all():
                issues.append(f"lofo noise: retained training exposure in {column}")

    ncmapss = pd.read_csv(OUTPUT / "ncmapss_sampling_window_run_manifest.csv")
    if set(ncmapss["sampling"]) != {50, 100, 200}:
        issues.append("ncmapss: sampling grid is incomplete")
    if set(ncmapss["window_size"]) != {30, 50, 70}:
        issues.append("ncmapss: window grid is incomplete")
    if ncmapss["seed"].nunique() != 5:
        issues.append("ncmapss: expected five seeds")
    if not (
        ncmapss.groupby(["sampling", "window_size"])["seed"].nunique() == 5
    ).all():
        issues.append("ncmapss: every sampling-window cell must have five seeds")
    horizon = pd.read_csv(OUTPUT / "ncmapss_horizon_matched_run_manifest.csv")
    expected_horizon = {(50, 121), (100, 61), (200, 31)}
    actual_horizon = set(
        zip(horizon["sampling"], horizon["window_size"], strict=True)
    )
    if actual_horizon != expected_horizon:
        issues.append("ncmapss horizon: exact-span density controls are incomplete")
    if horizon["seed"].nunique() != 5 or not (
        horizon.groupby(["sampling", "window_size"])["seed"].nunique() == 5
    ).all():
        issues.append("ncmapss horizon: every density cell must have five seeds")
    if not (pd.to_numeric(horizon["source_record_span"]) == 6001).all():
        issues.append("ncmapss horizon: source-record span must equal 6001")
    manifests = [
        independent,
        preference,
        factorial,
        lofo,
        ncmapss,
        horizon,
        crossed,
        preprocessing,
    ]
    family_cells = sum(len(frame) for frame in manifests)
    unique_runs = len(
        {
            str(run_dir)
            for frame in manifests
            for run_dir in frame["run_dir"].tolist()
        }
    )
    if family_cells != 412 or unique_runs != 373:
        issues.append(
            "retrospective extension: expected 412 family cells and 373 unique artifacts, "
            f"found {family_cells} and {unique_runs}"
        )
    identity = pd.concat(
        [frame[["run_dir", "artifact_id"]] for frame in manifests],
        ignore_index=True,
    ).drop_duplicates()
    if identity.groupby("run_dir")["artifact_id"].nunique().max() != 1:
        issues.append("artifact IDs do not map uniquely from run directories")
    if identity.groupby("artifact_id")["run_dir"].nunique().max() != 1:
        issues.append("artifact IDs collide across run directories")
    return issues


def verify_analysis_outputs() -> list[str]:
    required = {
        "independent_stream_paired_summary.csv": 12,
        "crossed_split_stream_effects.csv": 108,
        "crossed_split_stream_summary.csv": 12,
        "crossed_level_subset_effects.csv": 1560,
        "crossed_level_subset_sensitivity.csv": 24,
        "crossed_leave_one_level_sensitivity.csv": 72,
        "cmapss_preprocessing_paired_effects.csv": 45,
        "cmapss_preprocessing_paired_summary.csv": 15,
        "cmapss_preprocessing_reference_changes.csv": 72,
        "cmapss_preprocessing_reference_change_summary.csv": 24,
        "cmapss_preprocessing_resource_summary.csv": 10,
        "cmapss_preprocessing_leave_one_seed.csv": 45,
        "preference_retraining_candidate_summary.csv": 9,
        "preference_retraining_selected_cells.csv": 9,
        "preference_retraining_downstream_effects.csv": 9,
        "factorial_stream_effects.csv": 105,
        "factorial_effect_summary.csv": 21,
        "lofo_perturbation_summary.csv": 36,
        "lofo_paired_effect_summary.csv": 54,
        "lofo_training_stream_effects.csv": 270,
        "lofo_clean_context_deduplicated.csv": 40,
        "lofo_active_exposure_matrix.csv": 40,
        "lofo_family_stream_effects.csv": 60,
        "lofo_family_effect_summary.csv": 12,
        "lofo_scenario_auc_stream_effects.csv": 75,
        "lofo_point_scenario_auc.csv": 150,
        "lofo_level_paired_degradation_effects.csv": 270,
        "lofo_level_paired_degradation_summary.csv": 54,
        "lofo_metric_schema.csv": 3,
        "lofo_late_event_rows.csv": 2700,
        "lofo_late_event_support.csv": 108,
        "lofo_late_event_support_by_training_stream.csv": 540,
        "lofo_lpr_tolerance_stream_effects.csv": 270,
        "lofo_lpr_tolerance_summary.csv": 54,
        "factorial_design_matrix.csv": 8,
        "ncmapss_sampling_window_summary.csv": 9,
        "ncmapss_horizon_matched_summary.csv": 3,
        "retrospective_family_dashboard.csv": 25,
        "governance_workflow.csv": 6,
    }
    issues = []
    for name, minimum_rows in required.items():
        path = OUTPUT / name
        if not path.exists():
            issues.append(f"missing analysis output: {name}")
            continue
        rows = len(pd.read_csv(path))
        if rows < minimum_rows:
            issues.append(f"{name}: expected at least {minimum_rows} rows, found {rows}")
    perturbation = OUTPUT / "lofo_held_out_family_perturbation_rows.csv"
    if not perturbation.exists() or len(pd.read_csv(perturbation)) != 1100:
        issues.append("lofo held-out-family evaluation must contain exactly 1100 rows")
    else:
        frame = pd.read_csv(perturbation)
        if frame["perturbation_seed"].nunique() != 5:
            issues.append("lofo evaluation must contain five perturbation seeds")
        expected_scenarios = {
            "noise": {"clean", "gaussian_noise"},
            "random_missingness": {
                "clean",
                "global_sensor_missing",
                "window_random_missing",
            },
            "block_missingness": {"clean", "block_missing"},
            "drift": {"clean", "sensor_drift"},
        }
        for family, scenarios in expected_scenarios.items():
            actual = set(
                frame.loc[frame["held_out_family"] == family, "scenario"]
            )
            if actual != scenarios:
                issues.append(
                    f"lofo {family}: expected scenarios {sorted(scenarios)}, "
                    f"found {sorted(actual)}"
                )
    family_summary = OUTPUT / "lofo_family_effect_summary.csv"
    if family_summary.exists():
        frame = pd.read_csv(family_summary)
        zero_anchored = {
            "absolute_auc_asym_minus_rast_mean",
            "degradation_auc_asym_minus_rast_mean",
            "nonzero_absolute_mean_asym_minus_rast_mean",
            "nonzero_degradation_mean_asym_minus_rast_mean",
        }
        missing = sorted(zero_anchored - set(frame.columns))
        if missing:
            issues.append(f"lofo curve estimands missing columns: {missing}")
    semantic_index = PROJECT_ROOT / "paper_outputs" / "retrospective_governance_v1" / "artifact_index.json"
    if not semantic_index.exists():
        issues.append("stable retrospective governance semantic index is missing")
    return issues


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Verify completeness of all Round-12 newly trained evidence."
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--skip-analysis", action="store_true")
    args = parser.parse_args()
    issues = []
    for family in (
        "independent",
        "crossed",
        "cmapss_preprocessing",
        "preference",
        "factorial",
        "lofo",
    ):
        issues.extend(
            verify_training_manifest(
                family,
                EXPECTED[family],
                epochs=args.epochs,
            )
        )
    issues.extend(
        verify_training_manifest(
            "ncmapss_sampling_window",
            EXPECTED["ncmapss_sampling_window"],
            epochs=args.epochs,
        )
    )
    issues.extend(
        verify_training_manifest(
            "ncmapss_horizon_matched",
            EXPECTED["ncmapss_horizon_matched"],
            epochs=args.epochs,
        )
    )
    if not issues:
        issues.extend(verify_designs())
    if not args.skip_analysis and not issues:
        issues.extend(verify_analysis_outputs())
    if issues:
        for issue in issues:
            print(f"ROUND12_EVIDENCE_ERROR {issue}")
        raise SystemExit(1)
    print(
        "ROUND12_NEW_EVIDENCE_PASS "
        "family_cells=412 unique_training_artifacts=373 "
        "lofo_evaluations=200 lofo_rows=1100"
    )


if __name__ == "__main__":
    main()
