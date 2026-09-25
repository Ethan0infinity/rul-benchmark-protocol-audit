from __future__ import annotations

import argparse
import copy
import csv
import itertools
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rul.config import load_config
from rul.training import train_model


LATE_LIFE_WEIGHTS = [1.25, 1.5, 2.0]
LATE_OVER_WEIGHTS = [1.0, 1.25, 1.5]


def main() -> None:
    parser = argparse.ArgumentParser(description="Validation-only risk-weight search on the FD004 development split.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "paper_main.yaml"))
    parser.add_argument("--subset", default="FD004")
    parser.add_argument("--seed", type=int, default=31415)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--no-batch-bar", action="store_true")
    args = parser.parse_args()

    grid = list(itertools.product(LATE_LIFE_WEIGHTS, LATE_OVER_WEIGHTS))
    print(
        f"[RISK SEARCH] subset={args.subset} seed={args.seed} combinations={len(grid)} "
        f"epochs={args.epochs} selection=val_last_risk_score",
        flush=True,
    )
    if args.dry_run:
        for late_life, late_over in grid:
            print(f"late_life_weight={late_life:.2f} late_over_weight={late_over:.2f}")
        return

    base = load_config(args.config)
    rows: list[dict[str, object]] = []
    for index, (late_life, late_over) in enumerate(grid, start=1):
        cfg = copy.deepcopy(base)
        tag = f"ll{late_life:.2f}_lo{late_over:.2f}".replace(".", "p")
        cfg["project"]["experiment_name"] = f"tuning_risk_v3_{tag}"
        cfg["project"]["seed"] = int(args.seed)
        cfg.setdefault("augmentation", {})["seed"] = int(args.seed)
        cfg["training"]["epochs"] = int(args.epochs)
        cfg["training"]["patience"] = int(args.patience)
        cfg["training"]["late_life_weight"] = float(late_life)
        cfg["training"]["late_over_weight"] = float(late_over)
        if args.no_batch_bar:
            cfg.setdefault("progress", {})["batch_bar"] = False
        print(f"[RISK SEARCH {index}/{len(grid)}] {tag}", flush=True)
        run_dir = PROJECT_ROOT / cfg["project"].get("results_dir", "results") / cfg["project"]["experiment_name"] / args.subset / "rast_gru_v2"
        metrics_path = run_dir / "metrics.json"
        checkpoint_path = run_dir / "best_model.pt"
        metrics = None
        if metrics_path.exists() and checkpoint_path.exists():
            try:
                candidate = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
                if (
                    int(candidate.get("planned_epochs", 0)) == int(args.epochs)
                    and candidate.get("checkpoint_selection_metric") == "val_last_risk_score"
                    and isinstance(checkpoint, dict)
                    and "model_state" in checkpoint
                ):
                    metrics = candidate
                    print(f"[RISK SEARCH REUSE] {run_dir}", flush=True)
            except Exception:  # noqa: BLE001
                metrics = None
        if metrics is None:
            metrics = train_model(cfg, subset=args.subset, model_name="rast_gru_v2", run_index=index, total_runs=len(grid))
        rows.append(
            {
                "rank": 0,
                "subset": args.subset,
                "development_seed": args.seed,
                "late_life_weight": late_life,
                "late_over_weight": late_over,
                "smooth_late_risk_weight": cfg["training"]["smooth_late_risk_weight"],
                "selection_lpr_weight": cfg["training"]["selection_lpr_weight"],
                "selection_severe_late_weight": cfg["training"]["selection_severe_late_weight"],
                "best_epoch": metrics["best_epoch"],
                "completed_epochs": metrics["completed_epochs"],
                "best_val_risk_score": metrics["best_selection_score"],
                "best_val_rmse": metrics["best_epoch_val_last_rmse"],
                "run_dir": metrics["run_dir"],
            }
        )

    rows.sort(key=lambda row: float(row["best_val_risk_score"]))
    for rank, row in enumerate(rows, start=1):
        row["rank"] = rank
    report_dir = PROJECT_ROOT / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    csv_path = report_dir / "risk_hyperparameter_search.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    payload = {
        "selection_rule": "minimum validation-only val_last_risk_score",
        "development_subset": args.subset,
        "development_seed": args.seed,
        "grid": {
            "late_life_weight": LATE_LIFE_WEIGHTS,
            "late_over_weight": LATE_OVER_WEIGHTS,
        },
        "selected": rows[0],
        "test_metrics_used_for_selection": False,
    }
    json_path = report_dir / "risk_hyperparameter_search.json"
    json_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"[RISK SEARCH PASS] selected={rows[0]} csv={csv_path} json={json_path}", flush=True)


if __name__ == "__main__":
    main()
