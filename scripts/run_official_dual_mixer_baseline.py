from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rul.config import load_config
from rul.training import train_model


SEEDS = (42, 123, 2024, 2025, 2026)
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
MODEL_NAME = "official_dual_mixer"
UPSTREAM_REPOSITORY = "https://github.com/fuen1590/PhmDeepLearningProjects"
UPSTREAM_COMMIT = "727c020cabba2c6ae96e8f0e28f7f0121b292e81"


def experiment_name(seed: int) -> str:
    return f"official_dual_mixer_seed{seed}"


def configured(base: dict, seed: int, epochs: int, patience: int) -> dict:
    cfg = copy.deepcopy(base)
    cfg["project"]["seed"] = seed
    cfg["project"]["experiment_name"] = experiment_name(seed)
    cfg["data"]["window_size"] = 30
    cfg["data"]["rul_cap"] = 125
    cfg["data"]["feature_mode"] = "manual_fd001"
    cfg["data"]["include_settings"] = False
    cfg["data"]["append_missing_mask"] = False
    cfg["data"]["scaler"] = "minmax"
    cfg["model"]["name"] = MODEL_NAME
    cfg["model"]["hidden_size"] = 32
    cfg["model"]["dropout"] = 0.0
    cfg["model"]["use_uncertainty_head"] = False
    cfg["training"]["epochs"] = epochs
    cfg["training"]["patience"] = patience
    cfg["training"]["batch_size"] = 1024
    cfg["training"]["loss"] = "mse"
    cfg["training"]["selection_metric"] = "val_loss"
    cfg["training"]["smooth_late_risk_weight"] = 0.0
    cfg["training"]["reliability_supervision_weight"] = 0.0
    cfg["training"]["consistency_weight"] = 0.0
    cfg["training"]["quantile_calibration_weight"] = 0.0
    cfg["augmentation"]["profile"] = "none"
    cfg["augmentation"]["train_noise_std"] = 0.0
    cfg["augmentation"]["train_missing_rate"] = 0.0
    cfg["augmentation"]["train_block_missing_rate"] = 0.0
    cfg["augmentation"]["train_drift_rate"] = 0.0
    cfg["augmentation"]["seed"] = seed
    cfg.setdefault("progress", {})["batch_bar"] = False
    return cfg


def read_metrics(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the official-code-derived Dual-Mixer architecture on the locked engine splits."
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "paper_main.yaml"))
    parser.add_argument("--subsets", nargs="+", choices=SUBSETS, default=list(SUBSETS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    jobs = [(subset, seed) for subset in args.subsets for seed in args.seeds]
    if args.dry_run:
        print(
            f"OFFICIAL_DUAL_MIXER_DRY_RUN jobs={len(jobs)} epochs={args.epochs} "
            f"upstream_commit={UPSTREAM_COMMIT}"
        )
        return

    base = load_config(args.config)
    results_root = PROJECT_ROOT / base["project"].get("results_dir", "results")
    rows: list[dict] = []
    for index, (subset, seed) in enumerate(jobs, 1):
        run_dir = results_root / experiment_name(seed) / subset / MODEL_NAME
        metrics_path = run_dir / "metrics.json"
        if args.reuse and metrics_path.exists():
            print(f"[DUAL-MIXER {index}/{len(jobs)}] reuse subset={subset} seed={seed}", flush=True)
            metrics = read_metrics(metrics_path)
        else:
            print(f"[DUAL-MIXER {index}/{len(jobs)}] train subset={subset} seed={seed}", flush=True)
            metrics = train_model(
                configured(base, seed, args.epochs, args.patience),
                subset=subset,
                model_name=MODEL_NAME,
                run_index=index,
                total_runs=len(jobs),
            )
        rows.append(
            {
                "subset": subset,
                "seed": seed,
                "model": MODEL_NAME,
                "model_display": "Dual-Mixer (official-code-derived)",
                "upstream_repository": UPSTREAM_REPOSITORY,
                "upstream_commit": UPSTREAM_COMMIT,
                "protocol_scope": "official architecture and recommended hyperparameters; locked local engine split",
                "test_rmse": metrics["test_rmse"],
                "test_mae": metrics["test_mae"],
                "test_nasa_per_engine": metrics["test_nasa_score"] / metrics["n_test_units"],
                "test_lpr30": metrics["test_critical_30_late_prediction_ratio"],
                "test_slpr30_10": metrics["test_critical_30_severe_late_10_ratio"],
                "parameters": metrics["parameters"],
                "best_epoch": metrics["best_epoch"],
            }
        )

    output = PROJECT_ROOT / "paper_outputs" / "advanced_evidence"
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    frame.to_csv(output / "official_dual_mixer_seed_metrics.csv", index=False)
    summary = frame.groupby(["subset", "model", "model_display"], as_index=False).agg(
        seed_count=("seed", "nunique"),
        rmse_mean=("test_rmse", "mean"),
        rmse_std=("test_rmse", lambda x: x.std(ddof=1)),
        nasa_per_engine_mean=("test_nasa_per_engine", "mean"),
        nasa_per_engine_std=("test_nasa_per_engine", lambda x: x.std(ddof=1)),
        lpr30_mean=("test_lpr30", "mean"),
        lpr30_std=("test_lpr30", lambda x: x.std(ddof=1)),
        slpr30_10_mean=("test_slpr30_10", "mean"),
        parameters=("parameters", "first"),
    )
    summary.to_csv(output / "official_dual_mixer_summary.csv", index=False)
    print(f"OFFICIAL_DUAL_MIXER_READY jobs={len(jobs)} rows={len(frame)} output={output}")


if __name__ == "__main__":
    main()
