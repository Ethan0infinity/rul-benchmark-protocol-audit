from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rul.config import load_config
from rul.experiments import metrics_rows_to_csv, run_experiment_grid


def main() -> None:
    parser = argparse.ArgumentParser(description="Run C-MAPSS subset/model grid.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--subsets", nargs="+", default=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--models", nargs="+", default=["gru", "tcn", "tcn_gru", "rs_tcn_gru"])
    args = parser.parse_args()

    cfg = load_config(args.config)
    rows = run_experiment_grid(cfg, args.subsets, args.models)
    out = PROJECT_ROOT / cfg["project"].get("results_dir", "results") / "experiment_grid_summary.csv"
    metrics_rows_to_csv(rows, out)
    print(f"Saved grid summary to {out}")


if __name__ == "__main__":
    main()
