from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rul.config import load_config
from rul.training import train_model
from run_formal_benchmark import FORMAL_BENCHMARK_SEEDS


BASELINE = "regime_dual_attention_cnn_gru"
PROPOSED = "rast_gru_v2"
SUBSET = "FD002"


def reusable(run_dir: Path, expected: dict) -> dict | None:
    required = [run_dir / "metrics.json", run_dir / "run_config.yaml", run_dir / "best_model.pt", run_dir / "test_predictions.csv"]
    if not all(path.exists() for path in required):
        return None
    actual = load_config(run_dir / "run_config.yaml")
    if any(actual.get(section, {}) != expected.get(section, {}) for section in ("data", "model", "training", "augmentation")):
        return None
    try:
        checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    except Exception:  # noqa: BLE001
        return None
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        return None
    return json.loads((run_dir / "metrics.json").read_text(encoding="utf-8-sig"))


def formal_proposed(results_root: Path, seed: int) -> tuple[dict, Path]:
    run_dir = results_root / f"paper_main_v3_seed{seed}" / SUBSET / PROPOSED
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists() or not (run_dir / "test_predictions.csv").exists():
        raise FileNotFoundError(f"Formal proposed-model FD002 artifact is missing: {run_dir}")
    row = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
    if row.get("protocol_version") != "3.0" or row.get("result_status") != "formal":
        raise ValueError(f"Unexpected formal reference protocol/status: {metrics_path}")
    return row, run_dir


def prediction_pairs(baseline_dir: Path, proposed_dir: Path, seed: int) -> pd.DataFrame:
    baseline = pd.read_csv(baseline_dir / "test_predictions.csv").rename(columns={"pred_rul": "baseline_pred_rul"})
    proposed = pd.read_csv(proposed_dir / "test_predictions.csv").rename(columns={"pred_rul": "proposed_pred_rul"})
    merged = baseline.merge(
        proposed[["unit_id", "true_rul", "proposed_pred_rul"]],
        on=["unit_id", "true_rul"],
        validate="one_to_one",
    )
    baseline_error = merged["baseline_pred_rul"] - merged["true_rul"]
    proposed_error = merged["proposed_pred_rul"] - merged["true_rul"]
    critical = merged["true_rul"] <= 30.0
    return merged.assign(
        seed=seed,
        subset=SUBSET,
        baseline_error=baseline_error,
        proposed_error=proposed_error,
        absolute_error_difference_baseline_minus_proposed=baseline_error.abs() - proposed_error.abs(),
        critical_zone=critical,
        baseline_late=critical & (baseline_error > 0.0),
        proposed_late=critical & (proposed_error > 0.0),
    )


def summarize_models(rows: list[dict]) -> pd.DataFrame:
    frame = pd.DataFrame(rows)
    metrics = [
        "test_rmse",
        "test_mae",
        "test_nasa_score",
        "test_critical_30_late_prediction_ratio",
        "parameters",
        "profiled_flops_batch1",
        "single_sample_inference_ms",
        "cpu_single_sample_inference_ms",
    ]
    return frame.groupby(["model", "comparison_role"], as_index=False).agg(
        seed_count=("seed", "nunique"),
        **{f"{metric}_mean": (metric, "mean") for metric in metrics},
        **{f"{metric}_std": (metric, lambda x: x.std(ddof=1)) for metric in metrics},
    )


def paired_bootstrap(pairs: pd.DataFrame, reps: int = 5000, seed: int = 20260711) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    seed_values = sorted(pairs["seed"].unique())
    statistics = {"mae_difference": [], "lpr30_difference": []}
    for _ in range(int(reps)):
        sampled_seeds = rng.choice(seed_values, size=len(seed_values), replace=True)
        chunks = []
        for sampled_seed in sampled_seeds:
            group = pairs[pairs["seed"] == sampled_seed]
            chunks.append(group.iloc[rng.integers(0, len(group), size=len(group))])
        sample = pd.concat(chunks, ignore_index=True)
        statistics["mae_difference"].append(sample["absolute_error_difference_baseline_minus_proposed"].mean())
        critical = sample[sample["critical_zone"]]
        statistics["lpr30_difference"].append(critical["baseline_late"].mean() - critical["proposed_late"].mean())
    critical_observed = pairs[pairs["critical_zone"]]
    observed = {
        "mae_difference": float(pairs["absolute_error_difference_baseline_minus_proposed"].mean()),
        "lpr30_difference": float(critical_observed["baseline_late"].mean() - critical_observed["proposed_late"].mean()),
    }
    rows = []
    for metric, values in statistics.items():
        array = np.asarray(values, dtype=float)
        rows.append(
            {
                "metric": metric,
                "observed_baseline_minus_proposed": observed[metric],
                "bootstrap_mean_baseline_minus_proposed": float(np.mean(array)),
                "ci95_low": float(np.quantile(array, 0.025)),
                "ci95_high": float(np.quantile(array, 0.975)),
                "probability_proposed_better": float(np.mean(array > 0.0)),
                "bootstrap_repetitions": int(reps),
                "resampling_levels": "training seed, then FD002 test engines",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a five-seed FD002 close-prior architecture comparison.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "paper_main.yaml"))
    parser.add_argument("--seeds", nargs="+", type=int, default=FORMAL_BENCHMARK_SEEDS)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--no-batch-bar", action="store_true")
    parser.add_argument("--rerun-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base = load_config(args.config)
    results_root = PROJECT_ROOT / base["project"].get("results_dir", "results")
    print(f"[CLOSE PRIOR] subset={SUBSET} baseline={BASELINE} seeds={len(args.seeds)} new_training_runs={len(args.seeds)}")
    if args.dry_run:
        print("CLOSE_PRIOR_DRY_RUN_PASS")
        return

    rows: list[dict] = []
    pairs: list[pd.DataFrame] = []
    for index, seed in enumerate(args.seeds, start=1):
        proposed_row, proposed_dir = formal_proposed(results_root, seed)
        proposed_row = dict(proposed_row)
        proposed_row["comparison_role"] = "proposed formal reference"
        rows.append(proposed_row)

        cfg = copy.deepcopy(base)
        cfg["project"]["seed"] = int(seed)
        cfg["project"]["experiment_name"] = f"close_prior_fd002_v3_seed{seed}"
        cfg["augmentation"]["seed"] = int(seed)
        cfg["model"]["name"] = BASELINE
        cfg["training"]["reliability_supervision_weight"] = 0.0
        cfg["training"]["consistency_weight"] = 0.0
        cfg["training"]["quantile_calibration_weight"] = 0.0
        if args.no_batch_bar:
            cfg.setdefault("progress", {})["batch_bar"] = False
        baseline_dir = results_root / cfg["project"]["experiment_name"] / SUBSET / BASELINE
        baseline_row = None if args.rerun_complete else reusable(baseline_dir, cfg)
        if baseline_row is None:
            print(f"[CLOSE PRIOR RUN {index}/{len(args.seeds)}] seed={seed}", flush=True)
            baseline_row = train_model(cfg, subset=SUBSET, model_name=BASELINE, run_index=index, total_runs=len(args.seeds))
        else:
            print(f"[CLOSE PRIOR SKIP {index}/{len(args.seeds)}] seed={seed}", flush=True)
        baseline_row = dict(baseline_row)
        baseline_row["comparison_role"] = "protocol-adapted close-prior architecture"
        rows.append(baseline_row)
        pairs.append(prediction_pairs(baseline_dir, proposed_dir, seed))

    output = PROJECT_ROOT / "paper_outputs" / "advanced_evidence"
    output.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output / "close_prior_fd002_runs.csv", index=False)
    summarize_models(rows).to_csv(output / "close_prior_fd002_summary.csv", index=False)
    pair_frame = pd.concat(pairs, ignore_index=True)
    pair_frame.to_csv(output / "close_prior_fd002_engine_pairs.csv", index=False)
    paired_bootstrap(pair_frame, reps=args.bootstrap_reps).to_csv(output / "close_prior_fd002_paired_bootstrap.csv", index=False)
    print(f"CLOSE_PRIOR_BASELINE_READY {output}")


if __name__ == "__main__":
    main()
