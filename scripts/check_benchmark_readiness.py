from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rul.reporting import collect_metrics
from run_formal_benchmark import (
    FORMAL_BENCHMARK_MODELS,
    FORMAL_BENCHMARK_SEEDS,
    FORMAL_BENCHMARK_SUBSETS,
    MIN_FORMAL_PLANNED_EPOCHS,
    formal_benchmark_metrics_row_is_valid,
)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Check whether the formal benchmark grid is complete.")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--models", nargs="+", default=FORMAL_BENCHMARK_MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=FORMAL_BENCHMARK_SEEDS)
    parser.add_argument("--subsets", nargs="+", default=FORMAL_BENCHMARK_SUBSETS)
    parser.add_argument("--min-planned-epochs", type=int, default=MIN_FORMAL_PLANNED_EPOCHS)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    rows = collect_metrics(args.results_dir)
    present = {
        (
            str(row.get("experiment_name", "")),
            str(row.get("subset", "")),
            str(row.get("model", "")),
        )
        for row in rows
        if formal_benchmark_metrics_row_is_valid(row, min_planned_epochs=args.min_planned_epochs)
    }

    missing: list[tuple[int, str, str]] = []
    for seed in args.seeds:
        experiment_name = f"paper_main_v3_seed{seed}"
        for subset in args.subsets:
            for model in args.models:
                if (experiment_name, subset, model) not in present:
                    missing.append((seed, subset, model))

    expected = len(args.seeds) * len(args.subsets) * len(args.models)
    complete = expected - len(missing)
    print(f"BENCHMARK_READINESS expected={expected} complete={complete} missing={len(missing)}")
    if missing:
        print("BENCHMARK_READINESS_MISSING")
        for seed, subset, model in missing[:120]:
            print(f"seed={seed} subset={subset} model={model}")
        if len(missing) > 120:
            print(f"... {len(missing) - 120} more missing runs")
        raise SystemExit(1)
    print("BENCHMARK_READINESS_PASS")


if __name__ == "__main__":
    main()
