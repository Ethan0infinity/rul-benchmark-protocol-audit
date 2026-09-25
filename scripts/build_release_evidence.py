from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = PROJECT_ROOT / "paper_outputs" / "analysis_working"
DEFAULT_OUTPUT = PROJECT_ROOT / "paper_outputs" / "release_v1"
SCHEMA_VERSION = "release-evidence-v1"
PRIMARY_KEYS = {
    "primary-fixed-task-estimates": "subset|metric",
    "fixed-transfer-aggregation": "scope|aggregation|metric",
    "endpoint-tolerance-sensitivity": "subset|rul_threshold_tau|late_tolerance_epsilon",
    "late-event-support": "subset|model|seed|rul_threshold_tau|late_tolerance_epsilon",
    "small-cluster-calibration": "engine_count|seed_sd|method|estimand",
    "max-t-family-calibration": "dgp|method|estimand",
    "feature-selection-stability": "sensor",
    "multiplicity-registry": "family_id",
    "condition-normalization-summary": "variant|metric",
    "condition-normalization-contrasts": "comparison|comparator|metric",
    "normalized-perturbation-contrasts": "operating_point|comparator|scenario|metric",
    "normalized-perturbation-scale": "family|normalized_amplitude",
    "mask-metadata-sensitivity": "model|scenario|level",
    "inference-latency": "display_name|hardware|batch_size",
    "fd004-preference-development-search": "late_life_weight|late_over_weight",
    "metric-schema": "metric_id",
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_tree(path: Path) -> str:
    digest = hashlib.sha256()
    for item in sorted(candidate for candidate in path.rglob("*") if candidate.is_file()):
        digest.update(item.relative_to(path).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(sha256_file(item).encode("ascii"))
        digest.update(b"\n")
    return digest.hexdigest()


def row_count(path: Path) -> int | str:
    if path.suffix.lower() != ".csv":
        return ""
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return max(0, sum(1 for _ in csv.reader(handle)) - 1)


def generator_sha256(generator: str) -> str:
    candidate = PROJECT_ROOT / generator
    if not candidate.exists():
        candidate = PROJECT_ROOT / "scripts" / "build_release_evidence.py"
    return sha256_file(candidate)


def write_preference_search_artifacts(source_csv: Path, output_csv: Path, output_tex: Path) -> None:
    fields = [
        "rank",
        "subset",
        "development_seed",
        "late_life_weight",
        "late_over_weight",
        "smooth_late_risk_weight",
        "selection_lpr_weight",
        "selection_severe_late_weight",
        "best_epoch",
        "completed_epochs",
        "best_val_risk_score",
        "best_val_rmse",
    ]
    with source_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({field: row[field] for field in fields} for row in rows)
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{Complete nine-cell FD004 validation-only preference-development search.}",
        r"\label{tab:fd004-preference-search}",
        r"\small",
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"Rank & $\lambda_{\mathrm{life}}$ & $\lambda_{\mathrm{late}}$ & Best epoch & Validation score & Validation RMSE \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['rank']} & {float(row['late_life_weight']):.2f} & "
            f"{float(row['late_over_weight']):.2f} & {row['best_epoch']} & "
            f"{float(row['best_val_risk_score']):.4f} & "
            f"{float(row['best_val_rmse']):.3f} " + r"\\"
        )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{minipage}{0.98\linewidth}\footnotesize Notes: This initial search uses one FD004 development split and seed 31415. Rank uses the validation risk score only; no test metric enters selection. A later retrospective 3$\times$3 split--stream refit repeats the fixed nine-point grid and is reported separately; its selection counts remain conditional on those nine examined cells and do not provide selection-adjusted test performance.\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    output_tex.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def reset_output(path: Path) -> None:
    resolved = path.resolve()
    root = PROJECT_ROOT.resolve()
    if resolved == root or root not in resolved.parents:
        raise ValueError(f"Refusing to reset output outside the project: {resolved}")
    if resolved.exists():
        shutil.rmtree(resolved)
    resolved.mkdir(parents=True, exist_ok=True)


def write_perturbation_summary(source_csv: Path, output_tex: Path) -> None:
    labels = {
        "rmse": "RMSE",
        "critical_30_late_prediction_ratio": "LPR@30",
        "critical_30_mean_late_excess": "ZIMLE@30",
    }
    counts: dict[tuple[str, str], dict[str, int]] = {}
    with source_csv.open("r", encoding="utf-8-sig", newline="") as handle:
        for row in csv.DictReader(handle):
            key = (row["operating_point"], row["metric"])
            bucket = counts.setdefault(key, {"lower_ocm": 0, "lower_rast": 0, "unresolved": 0})
            status = row["adjusted_evidence_status"].lower()
            if "lower rast" in status:
                bucket["lower_rast"] += 1
            elif "lower" in status and ("asym" in status or "core" in status or "ocm" in status):
                bucket["lower_ocm"] += 1
            else:
                bucket["unresolved"] += 1
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        r"\caption{S-PERTURB-54 normalized-coordinate perturbation diagnostics after one joint Holm adjustment.}",
        r"\label{tab:normalized-perturbation-summary}",
        r"\small",
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"OCM configuration & Metric & Lower OCM & Lower RAST & Unresolved \\",
        r"\midrule",
    ]
    for point in ("Asym", "Core"):
        for metric in ("rmse", "critical_30_late_prediction_ratio", "critical_30_mean_late_excess"):
            count = counts[(point, metric)]
            lines.append(
                f"{point} & {labels[metric]} & {count['lower_ocm']} & "
                f"{count['lower_rast']} & {count['unresolved']} " + r"\\"
            )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\begin{minipage}{0.98\linewidth}\footnotesize Notes: Each row summarizes nine perturbation families. "
            r"No adjusted contrast favors an OCM configuration. ZIMLE averages thresholded late excess over all "
            r"eligible engines. This family measures algorithmic sensitivity in normalized coordinates, not physical "
            r"sensor-fault robustness.\end{minipage}",
            r"\end{table}",
            "",
        ]
    )
    output_tex.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def copy_artifact(
    *,
    source: Path,
    output: Path,
    target_name: str,
    artifact_id: str,
    family_id: str,
    generator: str,
    evidence_role: str,
    records: list[dict[str, object]],
) -> None:
    if not source.exists():
        raise FileNotFoundError(source)
    target = output / target_name
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, target)
    records.append(
        {
            "artifact_id": artifact_id,
            "release_path": target.relative_to(output).as_posix(),
            "schema_version": SCHEMA_VERSION,
            "family_id": family_id,
            "evidence_role": evidence_role,
            "generator": generator,
            "generator_sha256": generator_sha256(generator),
            "primary_key": PRIMARY_KEYS.get(artifact_id, "not applicable"),
            "source_path": source.relative_to(PROJECT_ROOT).as_posix(),
            "source_sha256": sha256_file(source),
            "release_sha256": sha256_file(target),
            "row_count": row_count(target),
        }
    )


def build_release(source: Path, output: Path) -> Path:
    reset_output(output)
    advanced = PROJECT_ROOT / "paper_outputs" / "advanced_evidence"
    records: list[dict[str, object]] = []
    mappings = [
        (
            source / "primary_fixed_task_estimates.csv",
            "primary_fixed_task_estimates.csv",
            "primary-fixed-task-estimates",
            "E-FIXED-12",
            "scripts/analyze_release_statistics.py",
            "main result-informed fixed-benchmark estimation; not confirmatory",
        ),
        (
            source / "primary_fixed_task_bootstrap_draws.npz",
            "primary_fixed_task_bootstrap_draws.npz",
            "primary-bootstrap-draws",
            "E-FIXED-12",
            "scripts/analyze_release_statistics.py",
            "shared-draw diagnostic input",
        ),
        (
            source / "primary_seed_level_effects.csv",
            "primary_seed_level_effects.csv",
            "primary-seed-level-effects",
            "E-FIXED-12",
            "scripts/analyze_fixed_benchmark_evidence.py",
            "seed-level paired diagnostic underlying Figure 2",
        ),
        (
            source / "fixed_transfer_aggregation_summary.csv",
            "fixed_transfer_aggregation_summary.csv",
            "fixed-transfer-aggregation",
            "D-TRANSFER-AGG",
            "scripts/analyze_fixed_benchmark_evidence.py",
            "descriptive fixed-task and engine-pooled aggregation sensitivity",
        ),
        (
            source / "development_excluded_seed_effects.csv",
            "development_excluded_seed_effects.csv",
            "development-excluded-seed-effects",
            "D-TRANSFER-AGG",
            "scripts/analyze_fixed_benchmark_evidence.py",
            "FD001-FD003 task-wise and aggregation-wise seed effects",
        ),
        (
            source / "preference_selection_stability.csv",
            "preference_selection_stability.csv",
            "preference-selection-stability",
            "D-PREFERENCE",
            "scripts/analyze_preference_selection_stability.py",
            "validation-endpoint resampling diagnostic; same fitted candidates",
        ),
        (
            source / "preference_selection_leave_one_out.csv",
            "preference_selection_leave_one_out.csv",
            "preference-selection-leave-one-out",
            "D-PREFERENCE",
            "scripts/analyze_preference_selection_stability.py",
            "leave-one-validation-engine-out selection diagnostic",
        ),
        (
            source / "preference_selection_oob_regret.csv",
            "preference_selection_oob_regret.csv",
            "preference-selection-oob-regret",
            "D-PREFERENCE",
            "scripts/analyze_preference_selection_stability.py",
            "out-of-bag endpoint-resampling regret diagnostic",
        ),
        (
            source / "endpoint_tolerance_sensitivity.csv",
            "endpoint_tolerance_sensitivity.csv",
            "endpoint-tolerance-sensitivity",
            "S-ENDPOINT-80",
            "scripts/analyze_release_statistics.py",
            "sensitivity analysis",
        ),
        (
            source / "late_event_support.csv",
            "late_event_support.csv",
            "late-event-support",
            "D-LATE-SUPPORT",
            "scripts/analyze_release_statistics.py",
            "descriptive denominator and event audit",
        ),
        (
            source / "small_cluster_calibration.csv",
            "small_cluster_calibration.csv",
            "small-cluster-calibration",
            "E-FIXED-12",
            "scripts/analyze_release_statistics.py",
            "inferential limitation diagnostic",
        ),
        (
            source / "max_t_family_calibration.csv",
            "max_t_family_calibration.csv",
            "max-t-family-calibration",
            "E-FIXED-12",
            "scripts/analyze_release_statistics.py",
            "12-dimensional family-wise calibration diagnostic",
        ),
        (
            source / "feature_selection_stability.csv",
            "feature_selection_stability.csv",
            "feature-selection-stability",
            "D-ABLATION",
            "scripts/analyze_release_statistics.py",
            "descriptive feature-ranking audit",
        ),
        (
            source / "multiplicity_registry.csv",
            "multiplicity_registry.csv",
            "multiplicity-registry",
            "ALL",
            "scripts/analyze_release_statistics.py",
            "release-level family and evidence registry",
        ),
        (
            source / "condition_normalization_summary.csv",
            "condition_normalization_summary.csv",
            "condition-normalization-summary",
            "S-NORM-9",
            "scripts/analyze_condition_normalization_controls.py",
            "matched sensitivity control",
        ),
        (
            source / "condition_normalization_contrasts.csv",
            "condition_normalization_contrasts.csv",
            "condition-normalization-contrasts",
            "S-NORM-9",
            "scripts/analyze_condition_normalization_controls.py",
            "matched sensitivity control",
        ),
        (
            advanced / "normalized_perturbation_contrasts.csv",
            "normalized_perturbation_contrasts.csv",
            "normalized-perturbation-contrasts",
            "S-PERTURB-54",
            "scripts/analyze_normalized_perturbation_evidence.py",
            "normalized-coordinate sensitivity",
        ),
        (
            advanced / "no_overlap_augmentation_audit.csv",
            "no_overlap_augmentation_audit.csv",
            "no-overlap-augmentation-audit",
            "D-AUGDIST",
            "scripts/analyze_augmentation_distribution_sensitivity.py",
            "p=0 no-training-perturbation-overlap diagnostic",
        ),
        (
            source / "stress_amplitude_calibration.csv",
            "normalized_perturbation_scale_calibration.csv",
            "normalized-perturbation-scale",
            "S-PERTURB-54",
            "scripts/calibrate_stress_amplitudes.py",
            "empirical normalized-scale mapping; no raw-unit claim",
        ),
        (
            source / "mask_observation_sensitivity_summary.csv",
            "mask_metadata_sensitivity_summary.csv",
            "mask-metadata-sensitivity",
            "D-AUGDIST",
            "scripts/analyze_mask_observation_sensitivity.py",
            "metadata error diagnostic",
        ),
        (
            source / "inference_latency_protocol_summary.csv",
            "inference_latency_protocol_summary.csv",
            "inference-latency",
            "D-RUNTIME",
            "scripts/benchmark_inference_protocol.py",
            "descriptive runtime audit",
        ),
        (
            source / "fixed_benchmark_analysis_metadata.json",
            "fixed_benchmark_analysis_metadata.json",
            "analysis-metadata",
            "E-FIXED-12",
            "scripts/analyze_release_statistics.py",
            "algorithm and RNG disclosure",
        ),
    ]
    for item in mappings:
        copy_artifact(
            source=item[0],
            output=output,
            target_name=item[1],
            artifact_id=item[2],
            family_id=item[3],
            generator=item[4],
            evidence_role=item[5],
            records=records,
        )

    table_mappings = [
        ("table_primary_fixed_task_estimates.tex", "table_primary_fixed_task_estimates.tex"),
        ("table_primary_seed_sign_patterns.tex", "table_primary_seed_sign_patterns.tex"),
        ("table_development_excluded_summary.tex", "table_development_excluded_summary.tex"),
        ("table_preference_selection_stability.tex", "table_preference_selection_stability.tex"),
        ("table_preference_selection_stability_zh.tex", "table_preference_selection_stability_zh.tex"),
        ("table_late_event_support.tex", "table_late_event_support.tex"),
        ("table_small_cluster_calibration.tex", "table_small_cluster_calibration.tex"),
        ("table_max_t_family_calibration.tex", "table_max_t_family_calibration.tex"),
        ("table_feature_selection_stability.tex", "table_feature_selection_stability.tex"),
        ("table_multiplicity_registry.tex", "table_multiplicity_registry.tex"),
        ("table_inference_latency_protocol.tex", "table_inference_latency_protocol.tex"),
        ("table_condition_normalization_absolute.tex", "table_condition_normalization_absolute.tex"),
        ("table_condition_normalization_contrasts.tex", "table_condition_normalization_contrasts.tex"),
    ]
    for source_name, target_name in table_mappings:
        copy_artifact(
            source=source / source_name,
            output=output,
            target_name=f"tables/{target_name}",
            artifact_id=Path(target_name).stem,
            family_id="DISPLAY",
            generator="release table generators",
            evidence_role="display table generated from release CSV",
            records=records,
        )

    perturbation_table = output / "tables" / "table_normalized_perturbation_summary.tex"
    write_perturbation_summary(
        output / "normalized_perturbation_contrasts.csv",
        perturbation_table,
    )
    records.append(
        {
            "artifact_id": "table-normalized-perturbation-summary",
            "release_path": perturbation_table.relative_to(output).as_posix(),
            "schema_version": SCHEMA_VERSION,
            "family_id": "S-PERTURB-54",
            "evidence_role": "display table generated from release CSV",
            "generator": "scripts/build_release_evidence.py",
            "generator_sha256": generator_sha256("scripts/build_release_evidence.py"),
            "primary_key": "not applicable",
            "source_path": "paper_outputs/release_v1/normalized_perturbation_contrasts.csv",
            "source_sha256": sha256_file(output / "normalized_perturbation_contrasts.csv"),
            "release_sha256": sha256_file(perturbation_table),
            "row_count": "",
        }
    )

    figure_mappings = [
        ("fig_primary_fixed_task_diagnostic_envelope", "fig_primary_fixed_task_diagnostic_envelope"),
        ("fig_endpoint_tolerance_sensitivity", "fig_endpoint_tolerance_sensitivity"),
        ("fig_condition_normalization_controls", "fig_condition_normalization_controls"),
        ("fig_mask_observation_sensitivity", "fig_mask_metadata_sensitivity"),
    ]
    for source_stem, target_stem in figure_mappings:
        for suffix in (".pdf", ".png"):
            copy_artifact(
                source=source / "figures" / f"{source_stem}{suffix}",
                output=output,
                target_name=f"figures/{target_stem}{suffix}",
                artifact_id=f"{target_stem}-{suffix[1:]}",
                family_id="DISPLAY",
                generator="release figure generators",
                evidence_role="display figure generated from release CSV",
                records=records,
            )

    chronology_rows = []
    chronology_objects = [
        ("formal training grid", PROJECT_ROOT / "results", "not externally timestamped"),
        (
            "stored primary predictions",
            PROJECT_ROOT / "results" / "paper_main_v3_seed42",
            "generated before current analysis; pre-result status not externally verifiable",
        ),
        (
            "12-contrast estimand and family",
            source / "primary_fixed_task_estimates.csv",
            "defined after earlier result review; treated as fixed-benchmark estimation",
        ),
        (
            "max-t and null calibration code",
            PROJECT_ROOT / "scripts" / "analyze_fixed_benchmark_evidence.py",
            "post-result diagnostic",
        ),
        (
            "release multiplicity registry",
            source / "multiplicity_registry.csv",
            "post-result governance artifact",
        ),
    ]
    for object_name, path, status in chronology_objects:
        timestamp = datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat()
        chronology_rows.append(
            {
                "analysis_object": object_name,
                "local_file_timestamp_utc": timestamp,
                "git_commit": "unavailable: source directory is not a Git repository",
                "sha256": sha256_file(path) if path.is_file() else sha256_tree(path),
                "pre_test_summary_status": status,
                "external_timestamp": "none",
                "evidence_role": "local provenance only",
            }
        )
    chronology_path = output / "analysis_object_chronology.csv"
    with chronology_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(chronology_rows[0]))
        writer.writeheader()
        writer.writerows(chronology_rows)
    records.append(
        {
            "artifact_id": "analysis-object-chronology",
            "release_path": chronology_path.relative_to(output).as_posix(),
            "schema_version": SCHEMA_VERSION,
            "family_id": "GOVERNANCE",
            "evidence_role": "local provenance only; not an externally timestamped prospective analysis plan",
            "generator": "scripts/build_release_evidence.py",
            "generator_sha256": generator_sha256("scripts/build_release_evidence.py"),
            "primary_key": "analysis_object",
            "source_path": "",
            "source_sha256": "",
            "release_sha256": sha256_file(chronology_path),
            "row_count": row_count(chronology_path),
        }
    )

    preference_source = advanced / "risk_hyperparameter_search.csv"
    preference_csv = output / "fd004_preference_development_search.csv"
    preference_tex = output / "tables" / "table_fd004_preference_development_search.tex"
    preference_tex.parent.mkdir(parents=True, exist_ok=True)
    write_preference_search_artifacts(preference_source, preference_csv, preference_tex)
    for artifact_id, path, family_id, role, primary_key in [
        (
            "fd004-preference-development-search",
            preference_csv,
            "D-PREFERENCE-9",
            "single-split validation-only development record",
            PRIMARY_KEYS["fd004-preference-development-search"],
        ),
        (
            "table-fd004-preference-development-search",
            preference_tex,
            "DISPLAY",
            "display table generated from the nine-cell development record",
            "not applicable",
        ),
    ]:
        records.append(
            {
                "artifact_id": artifact_id,
                "release_path": path.relative_to(output).as_posix(),
                "schema_version": SCHEMA_VERSION,
                "family_id": family_id,
                "evidence_role": role,
                "generator": "scripts/build_release_evidence.py",
                "generator_sha256": generator_sha256("scripts/build_release_evidence.py"),
                "primary_key": primary_key,
                "source_path": preference_source.relative_to(PROJECT_ROOT).as_posix(),
                "source_sha256": sha256_file(preference_source),
                "release_sha256": sha256_file(path),
                "row_count": row_count(path),
            }
        )

    metric_schema_path = output / "metric_schema.json"
    metric_schema = {
        "schema_version": SCHEMA_VERSION,
        "metrics": [
            {
                "metric_id": "LPR@tau,epsilon",
                "numerator": "eligible engines with prediction minus true RUL greater than epsilon",
                "denominator": "all engines with true RUL at or below tau",
                "undefined_when": "no eligible engines",
            },
            {
                "metric_id": "ZIMLE@tau,epsilon",
                "numerator": "sum of max(prediction minus true RUL minus epsilon, 0)",
                "denominator": "all engines with true RUL at or below tau",
                "undefined_when": "no eligible engines",
            },
            {
                "metric_id": "CMLE@tau,epsilon",
                "numerator": "sum of positive late excess beyond epsilon",
                "denominator": "late-event engines only",
                "undefined_when": "no late event",
            },
            {
                "metric_id": "late_CVaR95",
                "numerator": "mean of the upper 5 percent of positive late excess values",
                "denominator": "late-event tail observations",
                "undefined_when": "no late event",
            },
        ],
    }
    metric_schema_path.write_text(
        json.dumps(metric_schema, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    records.append(
        {
            "artifact_id": "metric-schema",
            "release_path": metric_schema_path.relative_to(output).as_posix(),
            "schema_version": SCHEMA_VERSION,
            "family_id": "SCHEMA",
            "evidence_role": "machine-readable metric denominator and undefined-value contract",
            "generator": "scripts/build_release_evidence.py",
            "generator_sha256": generator_sha256("scripts/build_release_evidence.py"),
            "primary_key": PRIMARY_KEYS["metric-schema"],
            "source_path": "",
            "source_sha256": "",
            "release_sha256": sha256_file(metric_schema_path),
            "row_count": "",
        }
    )

    manifest_path = output / "release_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    metadata = {
        "release_id": "fixed-benchmark-release-v1",
        "schema_version": SCHEMA_VERSION,
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "source_of_truth": "paper_outputs/release_v1",
        "historical_round_files": "retained outside the release directory for local provenance only",
        "external_archive": {
            "public_url": None,
            "immutable_release": None,
            "doi": None,
            "status": "pending author upload; no identifier fabricated",
        },
        "confirmatory_status": "none; all inferential quantities are fixed-benchmark estimates or diagnostics",
        "manifest_sha256": sha256_file(manifest_path),
    }
    (output / "release_metadata.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    (output / "README.md").write_text(
        "# Fixed-Benchmark Evidence Release v1\n\n"
        "This directory is the only current machine-readable source of truth for the manuscript. "
        "Historical revision-round files are retained outside this directory only for local provenance "
        "and are excluded from the submission archive.\n\n"
        "No analysis family is confirmatory. Five training-seed levels, anti-conservative null "
        "calibration, and the absence of an externally timestamped prospective analysis plan limit the evidence "
        "to fixed-benchmark estimation and sensitivity diagnostics.\n\n"
        "`release_manifest.csv` records every artifact's family, generator and generator hash, primary "
        "key, source hash, release hash, schema version, and row count. `metric_schema.json` fixes the "
        "LPR/ZIMLE/CMLE denominator and undefined-value contracts. `analysis_object_chronology.csv` records local provenance and "
        "explicitly states where external timestamps or Git commits are unavailable.\n",
        encoding="utf-8",
        newline="\n",
    )
    print(f"RELEASE_EVIDENCE_READY artifacts={len(records)} output={output}")
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the semantic release-level evidence directory.")
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    args = parser.parse_args()
    build_release(Path(args.source_dir), Path(args.output_dir))


if __name__ == "__main__":
    main()
