from __future__ import annotations

import argparse
import subprocess
import sys
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def commands(manuscript_dir: Path | None) -> list[tuple[str, list[str]]]:
    python = sys.executable
    items: list[tuple[str, list[str]]] = [
        ("data validation", [python, "scripts/validate_data.py"]),
        ("core tests", [python, "scripts/run_core_tests.py"]),
        ("risk protocol", [python, "scripts/check_risk_protocol.py"]),
        ("reference audit", [python, "scripts/audit_references.py"]),
        ("benchmark readiness", [python, "scripts/check_benchmark_readiness.py"]),
        ("checkpoint integrity", [python, "scripts/check_checkpoint_integrity.py"]),
        ("model metadata repair", [python, "scripts/repair_model_metadata.py"]),
        ("post-selection empirical interval adjustment", [python, "scripts/calibrate_prediction_intervals.py"]),
        ("closest-prior baseline", [python, "scripts/run_close_prior_baseline.py", "--no-batch-bar"]),
        ("five-seed ablation", [python, "scripts/run_multiseed_ablation.py", "--no-batch-bar"]),
        ("ablation consistency", [python, "scripts/check_ablation_consistency.py"]),
        ("N-CMAPSS benchmark", [python, "scripts/run_ncmapss_benchmark.py"]),
        ("N-CMAPSS readiness", [python, "scripts/check_ncmapss_readiness.py"]),
        ("five-by-five stress tests", [python, "scripts/run_multiseed_stress_tests.py"]),
        ("model mechanisms", [python, "scripts/analyze_model_mechanisms.py"]),
        ("design sensitivity", [python, "scripts/run_design_sensitivity.py", "--no-batch-bar"]),
        ("condition mechanism", [python, "scripts/analyze_condition_normalization.py"]),
        ("failure cases", [python, "scripts/analyze_failure_cases.py"]),
        ("paper tables", [python, "scripts/build_paper_tables.py", "--experiment-prefix", "paper_main_v3_seed"]),
        ("threshold sensitivity", [python, "scripts/analyze_threshold_sensitivity.py"]),
        (
            "paired statistical tests",
            [
                python,
                "scripts/run_statistical_tests.py",
                "--experiment-prefix",
                "paper_main_v3_seed",
                "--baseline-models",
                "gru",
                "lstm",
                "tcn",
                "tcn_gru",
                "rast_gru",
                "cnn_lstm",
                "attention_gru",
                "bigru_attention",
                "transformer_lite",
                "dual_attention_tcn",
                "sensor_graph_gru",
                "quantile_gru",
            ],
        ),
        ("trajectory diagnostics", [python, "scripts/analyze_prediction_trajectory.py"]),
        ("recomputed numbers", [python, "scripts/recompute_paper_numbers.py", "--experiment-prefix", "paper_main_v3_seed"]),
        ("result brief", [python, "scripts/write_result_brief.py"]),
        ("revised vector assets", [python, "scripts/build_revised_assets.py"]),
        ("advanced evidence", [python, "scripts/build_advanced_evidence.py", "--bootstrap-reps", "5000"]),
        ("advanced evidence gate", [python, "scripts/check_advanced_evidence.py"]),
        ("figure-data gate", [python, "scripts/check_figure_data_consistency.py"]),
        (
            "augmentation-distribution sensitivity",
            [python, "scripts/run_augmentation_distribution_sensitivity.py"],
        ),
        (
            "augmentation-distribution analysis",
            [python, "scripts/analyze_augmentation_distribution_sensitivity.py"],
        ),
        (
            "split-by-training-seed factorial",
            [python, "scripts/run_split_training_seed_factorial.py"],
        ),
        (
            "split-by-training-seed analysis",
            [python, "scripts/analyze_split_training_seed_factorial.py"],
        ),
        (
            "N-CMAPSS development-unit rotation cache",
            [python, "scripts/prepare_ncmapss_dev_rotation.py"],
        ),
        (
            "N-CMAPSS development-unit rotation",
            [python, "scripts/run_ncmapss_dev_rotation.py"],
        ),
        (
            "N-CMAPSS development-unit rotation analysis",
            [python, "scripts/analyze_ncmapss_dev_rotation.py"],
        ),
        (
            "extended reviewer-evidence gate",
            [python, "scripts/check_extended_review_evidence.py"],
        ),
    ]
    manuscript_command = [python, "scripts/build_manuscript_evidence.py"]
    extended_command = [
        python,
        "scripts/build_extended_review_tables.py",
        "--copy-to",
        str(PROJECT_ROOT / "supplementary" / "generated"),
    ]
    if manuscript_dir is not None:
        manuscript_command.extend(["--copy-to", str(manuscript_dir / "generated")])
        extended_command.extend(["--copy-to", str(manuscript_dir / "generated")])
    items.extend(
        [
            ("manuscript evidence", manuscript_command),
            ("extended manuscript evidence", extended_command),
            ("manuscript evidence gate", [python, "scripts/check_manuscript_evidence.py"]),
        ]
    )
    return items


def main() -> None:
    parser = argparse.ArgumentParser(description="Run all protocol-v3 evidence steps after the 260-run formal grid.")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--manuscript-dir", default="")
    args = parser.parse_args()

    if args.manuscript_dir:
        manuscript_dir = Path(args.manuscript_dir)
    else:
        candidate = PROJECT_ROOT.parent / "正文"
        manuscript_dir = candidate if candidate.exists() else None

    items = commands(manuscript_dir)
    print(f"[POST-FORMAL PIPELINE] steps={len(items)} python={sys.executable}", flush=True)
    for index, (name, command) in enumerate(items, start=1):
        print(f"[POST-FORMAL {index:02d}/{len(items):02d}] {name}: {' '.join(command)}", flush=True)
        if args.dry_run:
            continue
        started = datetime.now().astimezone()
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        elapsed = (datetime.now().astimezone() - started).total_seconds()
        print(f"[POST-FORMAL DONE] {name} elapsed_sec={elapsed:.1f}", flush=True)
    print("POST_FORMAL_PIPELINE_PASS" if not args.dry_run else "POST_FORMAL_PIPELINE_DRY_RUN_PASS", flush=True)


if __name__ == "__main__":
    main()
