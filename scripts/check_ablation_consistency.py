from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rul.config import load_config
from run_formal_benchmark import FORMAL_BENCHMARK_SEEDS
from run_multiseed_ablation import MAJOR_FD004_VARIANTS, PROTOCOL_SECTIONS


METRICS = (
    "test_rmse",
    "test_mae",
    "test_nasa_score",
    "test_critical_30_late_prediction_ratio",
)


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8-sig"))


def _protocol_sections(path: Path) -> dict:
    cfg = load_config(path)
    return {section: cfg.get(section, {}) for section in PROTOCOL_SECTIONS}


def check_ablation_consistency(
    project_root: Path,
    *,
    seeds: list[int],
    subset: str = "FD004",
) -> list[str]:
    results = project_root / "results"
    summary_path = results / "ablation_fd004_multiseed_runs.csv"
    if not summary_path.exists():
        return [f"missing ablation summary: {summary_path}"]
    summary = pd.read_csv(summary_path)
    issues: list[str] = []
    expected_variants = set(MAJOR_FD004_VARIANTS)

    for seed in seeds:
        seed_rows = summary[pd.to_numeric(summary.get("seed"), errors="coerce") == seed]
        found = set(seed_rows.get("ablation_variant", pd.Series(dtype=str)).astype(str))
        missing = sorted(expected_variants - found)
        if missing:
            issues.append(f"seed={seed} missing variants: {','.join(missing)}")

        main_dir = results / f"paper_main_v3_seed{seed}" / subset / "rast_gru_v2"
        main_metrics_path = main_dir / "metrics.json"
        main_config_path = main_dir / "run_config.yaml"
        if not main_metrics_path.exists() or not main_config_path.exists():
            issues.append(f"seed={seed} missing formal main-model artifact")
            continue
        main = _read_json(main_metrics_path)
        full_rows = seed_rows[seed_rows.get("ablation_variant", "") == "full"]
        if len(full_rows) != 1:
            issues.append(f"seed={seed} expected one full reference row, found {len(full_rows)}")
        else:
            full = full_rows.iloc[0]
            for metric in METRICS:
                if metric not in full or metric not in main:
                    issues.append(f"seed={seed} missing metric {metric} for full/main comparison")
                    continue
                if abs(float(full[metric]) - float(main[metric])) > 1e-12:
                    issues.append(
                        f"seed={seed} full/main mismatch {metric}: "
                        f"{float(full[metric]):.12g} != {float(main[metric]):.12g}"
                    )

        main_protocol = _protocol_sections(main_config_path)
        for variant in sorted(expected_variants - {"full"}):
            variant_dir = results / f"ablation_fd004_seed{seed}_{variant}" / subset / "rast_gru_v2"
            config_path = variant_dir / "run_config.yaml"
            metrics_path = variant_dir / "metrics.json"
            if not config_path.exists() or not metrics_path.exists():
                issues.append(f"seed={seed} variant={variant} missing run artifacts")
                continue
            cfg = load_config(config_path)
            metrics = _read_json(metrics_path)
            if int(cfg.get("training", {}).get("epochs", 0)) != 80:
                issues.append(f"seed={seed} variant={variant} epochs must equal 80")
            if int(cfg.get("training", {}).get("patience", 0)) != 12:
                issues.append(f"seed={seed} variant={variant} patience must equal 12")
            if int(metrics.get("planned_epochs", 0)) != 80:
                issues.append(f"seed={seed} variant={variant} metrics planned_epochs must equal 80")
            if cfg.get("training", {}).get("selection_metric") != main_protocol["training"].get("selection_metric"):
                issues.append(f"seed={seed} variant={variant} selection metric differs from main protocol")

    return issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify that FD004 ablation evidence closes against the formal main benchmark.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--subset", default="FD004")
    parser.add_argument("--seeds", nargs="+", type=int, default=FORMAL_BENCHMARK_SEEDS)
    args = parser.parse_args()

    issues = check_ablation_consistency(Path(args.project_root), seeds=args.seeds, subset=args.subset)
    if issues:
        print("ABLATION_CONSISTENCY_FAIL")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)
    print(f"ABLATION_CONSISTENCY_PASS seeds={len(args.seeds)} variants={len(MAJOR_FD004_VARIANTS)}")


if __name__ == "__main__":
    main()
