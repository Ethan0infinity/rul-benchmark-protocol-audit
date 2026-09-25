from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rul.config import load_config
from rul.ncmapss import load_prepared_cache, prepare_ncmapss_ds02, save_prepared_cache
from rul.utils import get_device
from run_formal_benchmark import FORMAL_BENCHMARK_SEEDS
from run_ncmapss_benchmark import checkpoint_is_valid, train_one, write_summary


RAW = PROJECT_ROOT / "data" / "external" / "raw" / "N-CMAPSS_DS02-006.h5"
PROCESSED = PROJECT_ROOT / "data" / "external" / "processed"


def cache_path(k: int) -> Path:
    if k == 6:
        return PROCESSED / "ncmapss_ds02_smp100_win50.npz"
    return PROCESSED / f"ncmapss_ds02_smp100_win50_k{k}.npz"


def ensure_cache(k: int) -> Path:
    path = cache_path(k)
    if path.exists():
        return path
    print(f"[N-CMAPSS K={k}] preparing leakage-safe cache", flush=True)
    prepared = prepare_ncmapss_ds02(
        RAW,
        sampling=100,
        window_size=50,
        stride=1,
        condition_clusters=k,
    )
    save_prepared_cache(prepared, path)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Run five-seed N-CMAPSS condition-cluster sensitivity.")
    parser.add_argument("--k-values", nargs="+", type=int, default=[4, 6, 8])
    parser.add_argument("--seeds", nargs="+", type=int, default=FORMAL_BENCHMARK_SEEDS)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rerun-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    total = len(args.k_values) * len(args.seeds)
    print(f"[N-CMAPSS K GRID] jobs={total} k={args.k_values} seeds={args.seeds}")
    if args.dry_run:
        for k in args.k_values:
            for seed in args.seeds:
                reuse = k == 6
                print(f"- K={k} seed={seed} reuse_locked_k6={reuse}")
        print("NCMAPSS_K_SENSITIVITY_DRY_RUN_PASS")
        return

    base = copy.deepcopy(load_config(PROJECT_ROOT / "configs" / "paper_main.yaml"))
    device = get_device(args.device)
    all_rows = []
    for k in args.k_values:
        prepared = load_prepared_cache(ensure_cache(k))
        rows = []
        for seed in args.seeds:
            prefix = "ncmapss_ds02" if k == 6 else f"ncmapss_ds02_k{k}"
            experiment_name = f"{prefix}_seed{seed}"
            run_dir = PROJECT_ROOT / "results" / experiment_name / "DS02" / "rast_gru_v2"
            if not args.rerun_complete and args.epochs == 80 and checkpoint_is_valid(run_dir):
                row = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8-sig"))
                print(f"[N-CMAPSS K SKIP] K={k} seed={seed}", flush=True)
            else:
                print(f"[N-CMAPSS K RUN] K={k} seed={seed}", flush=True)
                row = train_one(
                    prepared,
                    model_name="rast_gru_v2",
                    seed=seed,
                    run_dir=run_dir,
                    epochs=args.epochs,
                    patience=args.patience,
                    batch_size=args.batch_size,
                    device=device,
                    protocol_config=base,
                    experiment_name=experiment_name,
                )
            row["condition_clusters"] = k
            rows.append(row)
            all_rows.append(row)
        write_summary(rows, PROJECT_ROOT / "paper_outputs" / "advanced_evidence", prefix=f"ncmapss_ds02_k{k}")

    frame = pd.DataFrame(all_rows)
    metrics = ["unit_macro_rmse", "unit_macro_mae", "unit_macro_nasa_per_window", "unit_macro_lpr30"]
    summary = frame.groupby("condition_clusters", as_index=False).agg(
        seed_count=("seed", "nunique"),
        **{f"{metric}_mean": (metric, "mean") for metric in metrics},
        **{f"{metric}_std": (metric, lambda x: x.std(ddof=1)) for metric in metrics},
    )
    output = PROJECT_ROOT / "paper_outputs" / "advanced_evidence"
    frame.to_csv(output / "ncmapss_k_sensitivity_seed_metrics.csv", index=False)
    summary.to_csv(output / "ncmapss_k_sensitivity_summary.csv", index=False)
    print(f"NCMAPSS_K_SENSITIVITY_READY rows={len(frame)}")


if __name__ == "__main__":
    main()
