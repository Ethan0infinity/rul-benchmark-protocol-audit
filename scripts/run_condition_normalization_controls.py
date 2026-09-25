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


CONTROL_SCALERS = {
    "global": "standard",
    "official_state": "official_state_standard",
    "continuous": "continuous_condition_standard",
}


def make_config(base: dict, variant: str, seed: int) -> dict:
    if variant not in CONTROL_SCALERS:
        raise ValueError(f"Unknown normalization control: {variant}")
    cfg = copy.deepcopy(base)
    cfg["project"]["seed"] = int(seed)
    cfg["project"]["split_seed"] = int(seed)
    cfg["project"]["initialization_seed"] = int(seed)
    cfg["project"]["experiment_name"] = f"condition_control_{variant}_fd004_seed{seed}"
    cfg["augmentation"]["seed"] = int(seed)
    cfg["data"]["subset"] = "FD004"
    cfg["data"]["scaler"] = CONTROL_SCALERS[variant]
    cfg["model"]["name"] = "rast_gru_v2"
    cfg.setdefault("progress", {})["batch_bar"] = False
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run leakage-safe FD004 normalization controls under the locked "
            "OCM-Asym model, split, optimization budget, and five formal seeds."
        )
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=FORMAL_BENCHMARK_SEEDS)
    parser.add_argument("--variants", nargs="+", choices=sorted(CONTROL_SCALERS), default=sorted(CONTROL_SCALERS))
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--rerun-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base = load_config(PROJECT_ROOT / "configs" / "paper_main.yaml")
    jobs = [
        (variant, int(seed), make_config(base, variant, int(seed)))
        for seed in args.seeds
        for variant in args.variants
    ]
    print(
        "[CONDITION CONTROL] "
        f"new_jobs={len(jobs)} variants={args.variants} seeds={args.seeds}; "
        "K-means rows are reused from the locked paper_main_v3 runs.",
        flush=True,
    )
    if args.dry_run:
        for variant, seed, cfg in jobs:
            print(
                f"- variant={variant} scaler={cfg['data']['scaler']} seed={seed} "
                f"experiment={cfg['project']['experiment_name']}",
                flush=True,
            )
        print("CONDITION_CONTROL_DRY_RUN_PASS")
        return

    rows = []
    for index, (variant, seed, cfg) in enumerate(jobs, start=1):
        if args.epochs is not None:
            cfg["training"]["epochs"] = int(args.epochs)
        run_dir = (
            PROJECT_ROOT
            / "results"
            / cfg["project"]["experiment_name"]
            / "FD004"
            / "rast_gru_v2"
        )
        existing = existing_formal_benchmark_run_metrics(run_dir, cfg)
        if not args.rerun_complete and args.epochs is None and existing is not None:
            row = existing
            print(f"[CONDITION SKIP {index}/{len(jobs)}] {variant} seed={seed}", flush=True)
        else:
            print(f"[CONDITION RUN {index}/{len(jobs)}] {variant} seed={seed}", flush=True)
            row = train_model(
                cfg,
                subset="FD004",
                model_name="rast_gru_v2",
                run_index=index,
                total_runs=len(jobs),
            )
        row.update(
            {
                "normalization_variant": variant,
                "normalization_scaler": CONTROL_SCALERS[variant],
                "normalization_control_status": (
                    "short_smoke_run" if args.epochs is not None else "formal_five_seed_control"
                ),
            }
        )
        rows.append(row)

    suffix = "_smoke" if args.epochs is not None else ""
    output = PROJECT_ROOT / "results" / f"condition_normalization_controls{suffix}.json"
    output.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"CONDITION_CONTROL_READY runs={len(rows)} output={output}", flush=True)


if __name__ == "__main__":
    main()
