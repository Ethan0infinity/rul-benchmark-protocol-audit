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


K_VALUES = (4, 5, 7, 8)
AUX_SCALES = (0.5, 2.0)


def tag(value: float) -> str:
    return str(value).replace(".", "p")


def make_config(base: dict, family: str, value: float, seed: int) -> dict:
    cfg = copy.deepcopy(base)
    cfg["project"]["seed"] = seed
    cfg["augmentation"]["seed"] = seed
    cfg["model"]["name"] = "rast_gru_v2"
    if family == "cluster_k":
        cfg["data"]["condition_clusters"]["FD004"] = int(value)
        cfg["project"]["experiment_name"] = f"design_sensitivity_k{int(value)}_fd004_seed{seed}"
    elif family == "aux_scale":
        cfg["project"]["experiment_name"] = f"design_sensitivity_aux{tag(value)}_fd004_seed{seed}"
        for key in ("smooth_late_risk_weight", "reliability_supervision_weight", "consistency_weight", "quantile_calibration_weight"):
            cfg["training"][key] = float(base["training"][key]) * value
    else:
        raise ValueError(family)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run FD004 cluster-count and auxiliary-weight sensitivity checks.")
    parser.add_argument("--seeds", nargs="+", type=int, default=FORMAL_BENCHMARK_SEEDS)
    parser.add_argument("--families", nargs="+", default=["cluster_k", "aux_scale"])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--rerun-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    base = load_config(PROJECT_ROOT / "configs" / "paper_main.yaml")
    jobs = []
    for seed in args.seeds:
        if "cluster_k" in args.families:
            jobs.extend(("cluster_k", float(k), seed, make_config(base, "cluster_k", k, seed)) for k in K_VALUES)
        if "aux_scale" in args.families:
            jobs.extend(("aux_scale", scale, seed, make_config(base, "aux_scale", scale, seed)) for scale in AUX_SCALES)
    print(f"[DESIGN SENSITIVITY] jobs={len(jobs)} seeds={args.seeds} families={args.families}", flush=True)
    if args.dry_run:
        for family, value, seed, cfg in jobs:
            print(f"- {family}={value:g} seed={seed} experiment={cfg['project']['experiment_name']}")
        print("DESIGN_SENSITIVITY_DRY_RUN_PASS")
        return
    rows = []
    for index, (family, value, seed, cfg) in enumerate(jobs, start=1):
        if args.epochs is not None:
            cfg["training"]["epochs"] = args.epochs
        cfg.setdefault("progress", {})["batch_bar"] = False
        run_dir = PROJECT_ROOT / "results" / cfg["project"]["experiment_name"] / "FD004" / "rast_gru_v2"
        existing = existing_formal_benchmark_run_metrics(run_dir, cfg)
        if not args.rerun_complete and args.epochs is None and existing is not None:
            row = existing
            print(f"[DESIGN SKIP {index}/{len(jobs)}] {family}={value:g} seed={seed}", flush=True)
        else:
            print(f"[DESIGN RUN {index}/{len(jobs)}] {family}={value:g} seed={seed}", flush=True)
            row = train_model(cfg, subset="FD004", model_name="rast_gru_v2", run_index=index, total_runs=len(jobs))
        row.update({"sensitivity_family": family, "sensitivity_value": value})
        rows.append(row)
    output = PROJECT_ROOT / "results" / "design_sensitivity_experiments.json"
    output.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"DESIGN_SENSITIVITY_READY runs={len(rows)} output={output}")


if __name__ == "__main__":
    main()
