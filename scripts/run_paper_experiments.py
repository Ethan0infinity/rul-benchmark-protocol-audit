from __future__ import annotations

import argparse
import csv
import copy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rul.config import load_config
from rul.experiments import metrics_rows_to_csv
from rul.training import train_model


def _summary_identity(row: dict) -> tuple[str, str, str, str]:
    return (
        str(row.get("experiment_name", "")),
        str(row.get("seed", "")),
        str(row.get("subset", "")),
        str(row.get("model", "")),
    )


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def write_paper_main_summaries(rows: list[dict], results_dir: str | Path) -> dict[str, Path | list[Path]]:
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    legacy = results_dir / "paper_main_runs.csv"
    latest = results_dir / "paper_main_runs_latest.csv"
    all_runs = results_dir / "paper_main_runs_all.csv"

    metrics_rows_to_csv(rows, legacy)
    metrics_rows_to_csv(rows, latest)

    seed_paths: list[Path] = []
    for seed in sorted({str(row.get("seed", "unknown")) for row in rows}):
        seed_rows = [row for row in rows if str(row.get("seed", "unknown")) == seed]
        seed_path = results_dir / f"paper_main_runs_seed{seed}.csv"
        metrics_rows_to_csv(seed_rows, seed_path)
        seed_paths.append(seed_path)

    merged: dict[tuple[str, str, str, str], dict] = {}
    for row in _read_csv_rows(all_runs):
        merged[_summary_identity(row)] = row
    for row in rows:
        merged[_summary_identity(row)] = row
    merged_rows = sorted(merged.values(), key=_summary_identity)
    metrics_rows_to_csv(merged_rows, all_runs)

    return {"legacy": legacy, "latest": latest, "all": all_runs, "seed": seed_paths}


def main() -> None:
    parser = argparse.ArgumentParser(description="Run paper main experiments for B mainline.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "paper_main.yaml"))
    parser.add_argument("--subsets", nargs="+", default=["FD001", "FD002", "FD003", "FD004"])
    parser.add_argument("--models", nargs="+", default=["gru", "lstm", "tcn", "tcn_gru", "rast_gru", "rast_gru_v2"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42])
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--no-batch-bar", action="store_true", help="Hide tqdm batch bars and keep one clear console line per epoch.")
    args = parser.parse_args()

    base = load_config(args.config)
    rows = []
    total_runs = len(args.seeds) * len(args.subsets) * len(args.models)
    run_number = 0
    print(
        f"[GRID START] paper main experiments total_runs={total_runs} "
        f"subsets={','.join(args.subsets)} models={','.join(args.models)} seeds={','.join(map(str, args.seeds))}",
        flush=True,
    )
    for seed in args.seeds:
        for subset in args.subsets:
            for model_name in args.models:
                run_number += 1
                print(f"[GRID {run_number:03d}/{total_runs:03d}] seed={seed} subset={subset} model={model_name}", flush=True)
                cfg = copy.deepcopy(base)
                cfg["project"]["seed"] = seed
                cfg["project"]["experiment_name"] = f"paper_main_v3_seed{seed}"
                cfg["model"]["name"] = model_name
                if args.epochs is not None:
                    cfg["training"]["epochs"] = args.epochs
                if args.no_batch_bar:
                    cfg.setdefault("progress", {})["batch_bar"] = False
                row = train_model(cfg, subset=subset, model_name=model_name, run_index=run_number, total_runs=total_runs)
                rows.append(row)
                print(
                    f"[GRID {run_number:03d}/{total_runs:03d} DONE] "
                    f"test_rmse={row.get('test_rmse', 0.0):.4f} "
                    f"test_mae={row.get('test_mae', 0.0):.4f} "
                    f"best_epoch={row.get('best_epoch')} "
                    f"run_dir={row.get('run_dir')}",
                    flush=True,
                )

    results_dir = PROJECT_ROOT / base["project"].get("results_dir", "results")
    outputs = write_paper_main_summaries(rows, results_dir)
    print(
        "[GRID FINISH] saved paper main run summaries to "
        f"{outputs['latest']} and {outputs['all']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
