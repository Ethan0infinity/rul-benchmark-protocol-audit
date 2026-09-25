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
from run_formal_benchmark import FORMAL_BENCHMARK_SEEDS, existing_formal_benchmark_run_metrics


BUILDUP_VARIANTS = ("base_only", "plus_risk", "plus_risk_cons")


def make_buildup_config(base: dict, variant: str, seed: int) -> dict:
    cfg = copy.deepcopy(base)
    cfg["project"]["seed"] = seed
    cfg["project"]["experiment_name"] = f"attribution_ocm_{variant}_fd004_seed{seed}"
    cfg["augmentation"]["seed"] = seed
    cfg["model"]["name"] = "rast_gru_v2"
    cfg["model"]["use_uncertainty_head"] = False
    cfg["training"]["smooth_late_risk_weight"] = 0.0
    cfg["training"]["reliability_supervision_weight"] = 0.0
    cfg["training"]["consistency_weight"] = 0.0
    cfg["training"]["quantile_calibration_weight"] = 0.0
    if variant == "plus_risk":
        cfg["training"]["smooth_late_risk_weight"] = 0.10
    elif variant == "plus_risk_cons":
        cfg["training"]["smooth_late_risk_weight"] = 0.10
        cfg["training"]["consistency_weight"] = 0.05
    elif variant != "base_only":
        raise ValueError(variant)
    return cfg


def make_track_b_config(base: dict, seed: int) -> dict:
    cfg = copy.deepcopy(base)
    cfg["project"]["seed"] = seed
    cfg["project"]["results_dir"] = "results/standard_protocol_track"
    cfg["project"]["experiment_name"] = f"standard_protocol_fd004_seed{seed}"
    cfg["augmentation"]["seed"] = seed
    cfg["model"]["name"] = "rast_gru_v2"
    return cfg


def completed(run_dir: Path, cfg: dict) -> bool:
    row = existing_formal_benchmark_run_metrics(run_dir, cfg)
    return bool(row and int(row.get("planned_epochs", 0)) >= int(cfg["training"]["epochs"]))


def main() -> None:
    parser = argparse.ArgumentParser(description="Run OCM protocol and objective-attribution controls on FD004.")
    parser.add_argument("--seeds", nargs="+", type=int, default=FORMAL_BENCHMARK_SEEDS)
    parser.add_argument("--variants", nargs="+", default=["track_b", *BUILDUP_VARIANTS])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--rerun-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    main_base = load_config(PROJECT_ROOT / "configs" / "paper_main.yaml")
    track_b_base = load_config(PROJECT_ROOT / "configs" / "standard_protocol_fd004.yaml")
    jobs: list[tuple[str, int, dict, Path]] = []
    for seed in args.seeds:
        for variant in args.variants:
            cfg = make_track_b_config(track_b_base, seed) if variant == "track_b" else make_buildup_config(main_base, variant, seed)
            if args.epochs is not None:
                cfg["training"]["epochs"] = args.epochs
            cfg.setdefault("progress", {})["batch_bar"] = False
            run_dir = PROJECT_ROOT / cfg["project"]["results_dir"] / cfg["project"]["experiment_name"] / "FD004" / "rast_gru_v2"
            jobs.append((variant, seed, cfg, run_dir))

    print(f"[ATTRIBUTION GRID] jobs={len(jobs)} variants={args.variants} seeds={args.seeds}", flush=True)
    if args.dry_run:
        for variant, seed, _, run_dir in jobs:
            print(f"- {variant} seed={seed}: {run_dir}")
        print("ATTRIBUTION_DRY_RUN_PASS")
        return

    rows = []
    for index, (variant, seed, cfg, run_dir) in enumerate(jobs, start=1):
        if not args.rerun_complete and args.epochs is None and completed(run_dir, cfg):
            row = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8-sig"))
            print(f"[ATTRIBUTION SKIP {index}/{len(jobs)}] variant={variant} seed={seed}", flush=True)
        else:
            print(f"[ATTRIBUTION RUN {index}/{len(jobs)}] variant={variant} seed={seed}", flush=True)
            row = train_model(cfg, subset="FD004", model_name="rast_gru_v2", run_index=index, total_runs=len(jobs))
        row["attribution_variant"] = variant
        rows.append(row)
    output = PROJECT_ROOT / "results" / "attribution_experiments.json"
    output.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"ATTRIBUTION_EXPERIMENTS_READY runs={len(rows)} output={output}")


if __name__ == "__main__":
    main()
