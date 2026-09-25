from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rul.config import load_config
from rul.training import train_model


SEEDS = (42, 123, 2024, 2025, 2026)
SUBSETS = ("FD002", "FD004")
VARIANTS = {
    "rast": ("rast_gru", 1.5),
    "ocm_core": ("rast_gru_v2", 1.0),
    "ocm_asym": ("rast_gru_v2", 1.5),
}
STRATEGIES = ("uniform_single", "fixed_multi", "critical_stratified")
METRICS = (
    "test_rmse",
    "test_nasa_per_engine",
    "test_critical_30_late_prediction_ratio",
    "test_critical_30_severe_late_10_ratio",
)


def experiment_name(variant: str, seed: int) -> str:
    return f"endpoint_selection_{variant}_seed{seed}"


def run_directory(results_dir: Path, variant: str, seed: int, subset: str, model: str) -> Path:
    return results_dir / experiment_name(variant, seed) / subset / model


def configured(base: dict, variant: str, seed: int, epochs: int, patience: int) -> tuple[dict, str]:
    model, late_weight = VARIANTS[variant]
    cfg = copy.deepcopy(base)
    cfg["project"]["seed"] = seed
    cfg["project"]["experiment_name"] = experiment_name(variant, seed)
    cfg["model"]["name"] = model
    cfg["training"]["late_over_weight"] = late_weight
    cfg["training"]["epochs"] = epochs
    cfg["training"]["patience"] = patience
    cfg["data"]["validation_endpoint_confirmation"] = True
    cfg["augmentation"]["seed"] = seed
    cfg.setdefault("progress", {})["batch_bar"] = False
    return cfg, model


def bootstrap_strategy_comparisons(long: pd.DataFrame, results_dir: Path, reps: int) -> pd.DataFrame:
    rows: list[dict] = []
    rng = np.random.default_rng(20260717)
    for subset in SUBSETS:
        for strategy in STRATEGIES:
            for left_variant, right_variant in (("ocm_core", "rast"), ("ocm_asym", "rast"), ("ocm_core", "ocm_asym")):
                left_model = VARIANTS[left_variant][0]
                right_model = VARIANTS[right_variant][0]
                pairs: dict[int, dict[str, np.ndarray]] = {}
                for seed in SEEDS:
                    left_path = run_directory(results_dir, left_variant, seed, subset, left_model) / f"test_predictions_endpoint_{strategy}.csv"
                    right_path = run_directory(results_dir, right_variant, seed, subset, right_model) / f"test_predictions_endpoint_{strategy}.csv"
                    left = pd.read_csv(left_path)
                    right = pd.read_csv(right_path)
                    merged = left.merge(right, on=["unit_id", "true_rul"], suffixes=("_left", "_right"), validate="one_to_one")
                    if merged["unit_id"].duplicated().any():
                        raise ValueError(f"Expected one official test endpoint per engine: {subset=} {strategy=} {seed=}")
                    merged = merged.sort_values("unit_id").reset_index(drop=True)
                    pairs[seed] = {
                        "unit_id": merged["unit_id"].to_numpy(int),
                        "true": merged["true_rul"].to_numpy(float),
                        "left": merged["pred_rul_left"].to_numpy(float),
                        "right": merged["pred_rul_right"].to_numpy(float),
                    }
                for metric in ("rmse", "lpr30", "slpr30_10"):
                    def score(arrays: dict[str, np.ndarray], side: str, indices: np.ndarray | None = None) -> float:
                        true = arrays["true"] if indices is None else arrays["true"][indices]
                        pred = arrays[side] if indices is None else arrays[side][indices]
                        error = pred - true
                        if metric == "rmse":
                            return float(np.sqrt(np.mean(error**2)))
                        critical = true <= 30.0
                        threshold = 0.0 if metric == "lpr30" else 10.0
                        return float(np.mean(error[critical] > threshold))

                    observed = float(np.mean([score(frame, "left") - score(frame, "right") for frame in pairs.values()]))
                    samples = np.empty(reps, dtype=float)
                    for rep in range(reps):
                        values = []
                        for sampled_seed in rng.choice(SEEDS, size=len(SEEDS), replace=True):
                            arrays = pairs[int(sampled_seed)]
                            unit_ids = arrays["unit_id"]
                            sampled_units = rng.choice(unit_ids, size=len(unit_ids), replace=True)
                            sampled_indices = np.searchsorted(unit_ids, sampled_units)
                            values.append(
                                score(arrays, "left", sampled_indices) - score(arrays, "right", sampled_indices)
                            )
                        samples[rep] = float(np.mean(values))
                    rows.append(
                        {
                            "subset": subset,
                            "endpoint_strategy": strategy,
                            "left_variant": left_variant,
                            "right_variant": right_variant,
                            "metric": metric,
                            "mean_difference_left_minus_right": observed,
                            "percentile_ci95_low": float(np.quantile(samples, 0.025)),
                            "percentile_ci95_high": float(np.quantile(samples, 0.975)),
                            "bootstrap_repetitions": reps,
                            "bootstrap_primary_unit": "matched_test_engine",
                            "test_endpoints_per_engine": 1,
                            "validation_endpoints_used_in_bootstrap": False,
                            "validation_endpoint_role": "checkpoint_selection_only",
                        }
                    )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Confirm checkpoint selection across three validation endpoint designs.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "paper_main.yaml"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--subsets", nargs="+", choices=SUBSETS, default=list(SUBSETS))
    parser.add_argument("--variants", nargs="+", choices=tuple(VARIANTS), default=list(VARIANTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--reuse", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base = load_config(args.config)
    results_dir = PROJECT_ROOT / base["project"].get("results_dir", "results")
    jobs = [(variant, subset, seed) for variant in args.variants for subset in args.subsets for seed in args.seeds]
    if args.dry_run:
        print(f"ENDPOINT_CONFIRMATION_DRY_RUN jobs={len(jobs)} epochs={args.epochs} strategies={len(STRATEGIES)}")
        for index, (variant, subset, seed) in enumerate(jobs, 1):
            print(f"{index:02d}/{len(jobs)} variant={variant} subset={subset} seed={seed}")
        return

    collected = []
    for index, (variant, subset, seed) in enumerate(jobs, 1):
        cfg, model = configured(base, variant, seed, args.epochs, args.patience)
        run_dir = run_directory(results_dir, variant, seed, subset, model)
        confirmation_path = run_dir / "checkpoint_endpoint_confirmation.csv"
        if args.reuse and confirmation_path.exists():
            print(f"[ENDPOINT {index}/{len(jobs)}] reuse variant={variant} subset={subset} seed={seed}", flush=True)
        else:
            print(f"[ENDPOINT {index}/{len(jobs)}] train variant={variant} subset={subset} seed={seed}", flush=True)
            train_model(cfg, subset=subset, model_name=model, run_index=index, total_runs=len(jobs))
        frame = pd.read_csv(confirmation_path)
        frame["variant"] = variant
        collected.append(frame)

    output = PROJECT_ROOT / "paper_outputs" / "advanced_evidence"
    output.mkdir(parents=True, exist_ok=True)
    long = pd.concat(collected, ignore_index=True)
    long.to_csv(output / "endpoint_selection_confirmation_seed_metrics.csv", index=False)
    summary = long.groupby(["variant", "subset", "endpoint_strategy"], as_index=False).agg(
        seed_count=("seed", "nunique"),
        validation_endpoint_count_mean=("validation_endpoint_count", "mean"),
        validation_critical_30_count_mean=("validation_critical_30_count", "mean"),
        best_epoch_mean=("best_epoch", "mean"),
        **{f"{metric}_mean": (metric, "mean") for metric in METRICS},
        **{f"{metric}_std": (metric, lambda x: x.std(ddof=1)) for metric in METRICS},
    )
    summary.to_csv(output / "endpoint_selection_confirmation_summary.csv", index=False)
    if set(args.subsets) == set(SUBSETS) and set(args.variants) == set(VARIANTS) and set(args.seeds) == set(SEEDS):
        bootstrap = bootstrap_strategy_comparisons(long, results_dir, args.bootstrap_reps)
        bootstrap.to_csv(output / "endpoint_selection_confirmation_bootstrap.csv", index=False)
    print(f"ENDPOINT_CONFIRMATION_READY jobs={len(jobs)} rows={len(long)} output={output}")


if __name__ == "__main__":
    main()
