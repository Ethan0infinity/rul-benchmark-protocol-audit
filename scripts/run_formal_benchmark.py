from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rul.config import load_config
from rul.training import train_model
from run_paper_experiments import write_paper_main_summaries


FORMAL_BENCHMARK_MODELS = [
    "gru",
    "lstm",
    "tcn",
    "tcn_gru",
    "rast_gru",
    "cnn_lstm",
    "attention_gru",
    "bigru_attention",
    "transformer_lite",
    "dual_attention_tcn",
    "sensor_graph_gru",
    "quantile_gru",
    "rast_gru_v2",
]
FORMAL_BENCHMARK_SEEDS = [42, 123, 2024, 2025, 2026]
FORMAL_BENCHMARK_SUBSETS = ["FD001", "FD002", "FD003", "FD004"]
MIN_FORMAL_PLANNED_EPOCHS = 80
MIN_FORMAL_COMPLETED_EPOCHS = 30


def formal_benchmark_metrics_row_is_valid(
    row: dict,
    *,
    min_planned_epochs: int = MIN_FORMAL_PLANNED_EPOCHS,
    min_completed_epochs: int = MIN_FORMAL_COMPLETED_EPOCHS,
) -> bool:
    if row.get("result_status") != "formal" or row.get("allowed_for_paper") is not True:
        return False
    try:
        planned_epochs = int(row.get("planned_epochs", 0) or 0)
        completed_epochs = int(row.get("completed_epochs", 0) or 0)
    except (TypeError, ValueError):
        return False
    if planned_epochs < min_planned_epochs:
        return False
    if completed_epochs >= min_completed_epochs:
        return True
    return str(row.get("training_complete_reason", "")) == "early_stopped"


def _training_protocol_matches(run_dir: Path, expected_config: dict) -> bool:
    config_path = run_dir / "run_config.yaml"
    if not config_path.exists():
        return False
    actual = load_config(config_path)
    return all(actual.get(section, {}) == expected_config.get(section, {}) for section in ("data", "model", "training", "augmentation"))


def existing_formal_benchmark_run_metrics(run_dir: Path, expected_config: dict | None = None) -> dict | None:
    metrics_path = run_dir / "metrics.json"
    checkpoint_path = run_dir / "best_model.pt"
    if not metrics_path.exists() or not checkpoint_path.exists():
        return None
    try:
        with metrics_path.open("r", encoding="utf-8-sig") as f:
            row = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    try:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001 - invalid checkpoints must force a deterministic rerun.
        return None
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        return None
    if expected_config is not None and not _training_protocol_matches(run_dir, expected_config):
        return None
    row["run_dir"] = str(run_dir)
    return row


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run the stronger benchmark experiment grid.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "paper_main.yaml"))
    parser.add_argument("--subsets", nargs="+", default=FORMAL_BENCHMARK_SUBSETS)
    parser.add_argument("--models", nargs="+", default=FORMAL_BENCHMARK_MODELS)
    parser.add_argument("--seeds", nargs="+", type=int, default=FORMAL_BENCHMARK_SEEDS)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--no-batch-bar", action="store_true", help="Hide tqdm batch bars for cleaner PyCharm output.")
    parser.add_argument(
        "--rerun-complete",
        action="store_true",
        help="Rerun runs that already have valid 80-epoch formal metrics. By default they are reused.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the planned grid without training. Useful before starting a long run.",
    )
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    total_runs = len(args.seeds) * len(args.subsets) * len(args.models)
    print(
        "[BENCHMARK GRID] "
        f"total_runs={total_runs} subsets={','.join(args.subsets)} "
        f"models={','.join(args.models)} seeds={','.join(map(str, args.seeds))}",
        flush=True,
    )
    if args.dry_run:
        print("[BENCHMARK GRID] dry run only; no training started.", flush=True)
        return

    base = load_config(args.config)
    min_planned_epochs = int(base.get("training", {}).get("epochs", MIN_FORMAL_PLANNED_EPOCHS))
    results_root = PROJECT_ROOT / base["project"].get("results_dir", "results")
    rows = []
    run_number = 0
    for seed in args.seeds:
        for subset in args.subsets:
            for model in args.models:
                run_number += 1
                cfg = copy.deepcopy(base)
                cfg["project"]["seed"] = int(seed)
                cfg["project"]["experiment_name"] = f"{base['project']['experiment_name']}_seed{seed}"
                cfg.setdefault("augmentation", {})["seed"] = int(seed)
                cfg["model"]["name"] = model
                if args.epochs is not None:
                    cfg["training"]["epochs"] = int(args.epochs)
                if args.no_batch_bar:
                    cfg.setdefault("progress", {})["batch_bar"] = False
                run_dir = results_root / cfg["project"]["experiment_name"] / subset / model
                existing = existing_formal_benchmark_run_metrics(run_dir, cfg)
                if (
                    not args.rerun_complete
                    and args.epochs is None
                    and existing is not None
                    and formal_benchmark_metrics_row_is_valid(existing, min_planned_epochs=min_planned_epochs)
                ):
                    print(
                        f"[BENCHMARK SKIP {run_number}/{total_runs}] valid existing run "
                        f"seed={seed} subset={subset} model={model} "
                        f"epochs={existing.get('completed_epochs')}/{existing.get('planned_epochs')}",
                        flush=True,
                    )
                    rows.append(existing)
                    continue
                print(
                    f"[BENCHMARK RUN {run_number}/{total_runs}] seed={seed} subset={subset} model={model}",
                    flush=True,
                )
                rows.append(train_model(cfg, subset=subset, model_name=model, run_index=run_number, total_runs=total_runs))

    outputs = write_paper_main_summaries(rows, PROJECT_ROOT / base["project"].get("results_dir", "results"))
    print(f"[BENCHMARK GRID DONE] wrote summaries: {outputs}", flush=True)


if __name__ == "__main__":
    main()
