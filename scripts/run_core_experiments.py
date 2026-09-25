from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rul.config import load_config
from rul.training import train_model
from run_formal_benchmark import (
    FORMAL_BENCHMARK_SEEDS,
    FORMAL_BENCHMARK_SUBSETS,
    existing_formal_benchmark_run_metrics,
)


def core_config(base: dict, subset: str, seed: int) -> dict:
    """Return the full OCM model with a symmetric late-error multiplier."""
    cfg = copy.deepcopy(base)
    cfg["project"]["seed"] = int(seed)
    cfg["project"]["experiment_name"] = f"core_ocm_seed{seed}"
    cfg["augmentation"]["seed"] = int(seed)
    cfg["model"]["name"] = "rast_gru_v2"
    cfg["training"]["late_over_weight"] = 1.0
    cfg.setdefault("progress", {})["batch_bar"] = False
    cfg["data"]["subset"] = subset
    return cfg


def fd004_existing_dir(seed: int) -> Path:
    return (
        PROJECT_ROOT
        / "results"
        / f"ablation_fd004_seed{seed}_weighted_huber_no_asymmetry"
        / "FD004"
        / "rast_gru_v2"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the five-seed OCM-Core grid (late-overprediction multiplier = 1)."
    )
    parser.add_argument("--subsets", nargs="+", default=FORMAL_BENCHMARK_SUBSETS)
    parser.add_argument("--seeds", nargs="+", type=int, default=FORMAL_BENCHMARK_SEEDS)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--no-batch-bar", action="store_true", help="Accepted for CLI consistency; bars are disabled by default.")
    parser.add_argument("--rerun-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base = load_config(PROJECT_ROOT / "configs" / "paper_main.yaml")
    jobs: list[tuple[str, int, dict, Path, Path | None]] = []
    for seed in args.seeds:
        for subset in args.subsets:
            cfg = core_config(base, subset, seed)
            if args.epochs is not None:
                cfg["training"]["epochs"] = int(args.epochs)
            run_dir = PROJECT_ROOT / "results" / f"core_ocm_seed{seed}" / subset / "rast_gru_v2"
            reusable = fd004_existing_dir(seed) if subset == "FD004" and args.epochs is None else None
            jobs.append((subset, seed, cfg, run_dir, reusable))

    print(f"[OCM CORE GRID] jobs={len(jobs)} subsets={args.subsets} seeds={args.seeds}", flush=True)
    if args.dry_run:
        for subset, seed, _, run_dir, reusable in jobs:
            source = f" reuse={reusable}" if reusable and reusable.exists() else ""
            print(f"- subset={subset} seed={seed} output={run_dir}{source}")
        print("OCM_CORE_DRY_RUN_PASS")
        return

    rows: list[dict] = []
    for index, (subset, seed, cfg, run_dir, reusable) in enumerate(jobs, start=1):
        if reusable is not None and (reusable / "metrics.json").exists():
            row = json.loads((reusable / "metrics.json").read_text(encoding="utf-8-sig"))
            row["core_source_run_dir"] = str(reusable)
            print(f"[OCM CORE REUSE {index}/{len(jobs)}] subset={subset} seed={seed}", flush=True)
        else:
            existing = existing_formal_benchmark_run_metrics(run_dir, cfg)
            if not args.rerun_complete and args.epochs is None and existing is not None:
                row = existing
                print(f"[OCM CORE SKIP {index}/{len(jobs)}] subset={subset} seed={seed}", flush=True)
            else:
                print(f"[OCM CORE RUN {index}/{len(jobs)}] subset={subset} seed={seed}", flush=True)
                row = train_model(
                    cfg,
                    subset=subset,
                    model_name="rast_gru_v2",
                    run_index=index,
                    total_runs=len(jobs),
                )
        row["core_variant"] = "OCM-MST-GRU-Core"
        rows.append(row)

    output = PROJECT_ROOT / "results" / "core_ocm_runs.json"
    output.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"OCM_CORE_EXPERIMENTS_READY runs={len(rows)} output={output}")


if __name__ == "__main__":
    main()
