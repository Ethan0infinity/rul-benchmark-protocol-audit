from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rul.config import load_config
from rul.experiments import metrics_rows_to_csv
from rul.training import train_model


def make_probe_config(
    base: dict,
    *,
    seed: int,
    missing_rate: float,
    block_missing_rate: float,
    no_batch_bar: bool,
) -> dict:
    cfg = copy.deepcopy(base)
    cfg["project"]["seed"] = seed
    cfg["project"]["experiment_name"] = f"severe_missing_probe_seed{seed}"
    cfg.setdefault("model", {})["name"] = "rast_gru_v2"
    cfg.setdefault("augmentation", {})["profile"] = "missing_noise_drift"
    cfg["augmentation"]["train_missing_rate"] = float(missing_rate)
    cfg["augmentation"]["train_block_missing_rate"] = float(block_missing_rate)
    if no_batch_bar:
        cfg.setdefault("progress", {})["batch_bar"] = False
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a small FD004 severe-missing mitigation probe for OCM-MST-GRU.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "paper_main.yaml"))
    parser.add_argument("--subset", default="FD004")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--missing-rate", type=float, default=0.20)
    parser.add_argument("--block-missing-rate", type=float, default=0.10)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--no-batch-bar", action="store_true", help="Hide tqdm batch bars and keep concise PyCharm output.")
    args = parser.parse_args()

    base = load_config(args.config)
    cfg = make_probe_config(
        base,
        seed=args.seed,
        missing_rate=args.missing_rate,
        block_missing_rate=args.block_missing_rate,
        no_batch_bar=args.no_batch_bar,
    )
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs

    print(
        "[SEVERE MISSING PROBE START] "
        f"subset={args.subset} model=rast_gru_v2 seed={args.seed} "
        f"train_missing_rate={args.missing_rate} train_block_missing_rate={args.block_missing_rate}",
        flush=True,
    )
    row = train_model(cfg, subset=args.subset, model_name="rast_gru_v2", run_index=1, total_runs=1)
    row["probe_name"] = "severe_missing_mitigation"
    row["probe_train_missing_rate"] = args.missing_rate
    row["probe_train_block_missing_rate"] = args.block_missing_rate

    out = PROJECT_ROOT / base["project"].get("results_dir", "results") / "severe_missing_probe_runs.csv"
    metrics_rows_to_csv([row], out)
    print(
        "[SEVERE MISSING PROBE DONE] "
        f"test_rmse={row.get('test_rmse', 0.0):.4f} "
        f"test_nasa_score={row.get('test_nasa_score', 0.0):.2f} "
        f"run_dir={row.get('run_dir')}",
        flush=True,
    )
    print(f"[SEVERE MISSING PROBE FINISH] saved summary to {out}", flush=True)


if __name__ == "__main__":
    main()
