from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

def trapezoidal_integral(
    y: np.ndarray,
    x: np.ndarray | None = None,
    dx: float = 1.0,
    axis: int = -1,
) -> np.ndarray | np.floating:
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x=x, dx=dx, axis=axis)
    return np.trapz(y, x=x, dx=dx, axis=axis)


TRAPEZOID = trapezoidal_integral
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from check_checkpoint_integrity import check_checkpoint
from rul.experiments import evaluate_robustness, metrics_rows_to_csv
from run_formal_benchmark import FORMAL_BENCHMARK_MODELS, FORMAL_BENCHMARK_SEEDS


PERTURBATION_SEEDS = [42, 123, 2024, 2025, 2026]
METRICS = ["rmse", "nasa_score", "critical_30_late_prediction_ratio"]


def nested_seed_ci(values: pd.DataFrame, column: str, *, reps: int, seed: int) -> tuple[float, float]:
    matrix = values.pivot(index="training_seed", columns="perturbation_seed", values=column).to_numpy(float)
    if not np.isfinite(matrix).all():
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    samples = np.empty(reps, dtype=float)
    for rep in range(reps):
        train_idx = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
        perturb_idx = rng.integers(0, matrix.shape[1], size=matrix.shape[1])
        samples[rep] = float(matrix[np.ix_(train_idx, perturb_idx)].mean())
    return float(np.quantile(samples, 0.025)), float(np.quantile(samples, 0.975))


def summarize_stress(rows: pd.DataFrame, *, bootstrap_reps: int) -> pd.DataFrame:
    output = []
    for group_index, ((subset, model, scenario, level), group) in enumerate(
        rows.groupby(["subset", "model", "scenario", "level"], sort=True)
    ):
        item: dict[str, object] = {
            "subset": subset,
            "model": model,
            "scenario": scenario,
            "level": level,
            "training_seed_count": int(group["training_seed"].nunique()),
            "perturbation_seed_count": int(group["perturbation_seed"].nunique()),
            "observation_count": int(len(group)),
        }
        for metric_index, metric in enumerate(METRICS):
            values = pd.to_numeric(group[metric], errors="coerce")
            item[f"{metric}_mean"] = float(values.mean())
            item[f"{metric}_std"] = float(values.std(ddof=1))
            low, high = nested_seed_ci(
                group,
                metric,
                reps=bootstrap_reps,
                seed=20260710 + group_index * 10 + metric_index,
            )
            item[f"{metric}_ci95_low"] = low
            item[f"{metric}_ci95_high"] = high
            item[f"{metric}_worst"] = float(values.max())
        output.append(item)
    return pd.DataFrame(output)


def degradation_auc(rows: pd.DataFrame) -> pd.DataFrame:
    clean = rows[rows["scenario"] == "clean"].copy()
    output = []
    scenarios = sorted(set(rows["scenario"]) - {"clean"})
    for (subset, model, training_seed, perturbation_seed), base in clean.groupby(
        ["subset", "model", "training_seed", "perturbation_seed"]
    ):
        clean_row = base.iloc[0]
        for scenario in scenarios:
            curve = rows[
                (rows["subset"] == subset)
                & (rows["model"] == model)
                & (rows["training_seed"] == training_seed)
                & (rows["perturbation_seed"] == perturbation_seed)
                & (rows["scenario"] == scenario)
            ].sort_values("level")
            if curve.empty:
                continue
            levels = np.concatenate([[0.0], curve["level"].to_numpy(float)])
            max_level = float(levels.max())
            x = levels / max_level if max_level > 0 else levels
            item: dict[str, object] = {
                "subset": subset,
                "model": model,
                "training_seed": int(training_seed),
                "perturbation_seed": int(perturbation_seed),
                "scenario": scenario,
                "max_level": max_level,
            }
            for metric in METRICS:
                clean_value = float(clean_row[metric])
                values = np.concatenate([[clean_value], curve[metric].to_numpy(float)])
                item[f"{metric}_auc"] = float(TRAPEZOID(values, x))
                relative = (values - clean_value) / clean_value if clean_value else np.zeros_like(values)
                item[f"{metric}_relative_degradation_auc"] = float(TRAPEZOID(relative, x))
                item[f"{metric}_worst_relative_degradation"] = float(np.max(relative))
            output.append(item)
    return pd.DataFrame(output)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run two-level training-seed by perturbation-seed stress tests.")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "paper_outputs" / "advanced_evidence"))
    parser.add_argument("--subsets", nargs="+", default=["FD004"])
    parser.add_argument("--models", nargs="+", default=FORMAL_BENCHMARK_MODELS)
    parser.add_argument("--training-seeds", nargs="+", type=int, default=FORMAL_BENCHMARK_SEEDS)
    parser.add_argument("--perturbation-seeds", nargs="+", type=int, default=PERTURBATION_SEEDS)
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    run_dirs = [
        results_dir / f"paper_main_v3_seed{training_seed}" / subset / model
        for training_seed in args.training_seeds
        for subset in args.subsets
        for model in args.models
    ]
    failures = []
    for run_dir in run_dirs:
        issue = check_checkpoint(run_dir / "best_model.pt")
        if issue:
            failures.append((run_dir, issue))
    total_evaluations = len(run_dirs) * len(args.perturbation_seeds)
    print(
        f"[STRESS GRID] trained_runs={len(run_dirs)} perturbation_seeds={len(args.perturbation_seeds)} "
        f"evaluations={total_evaluations} checkpoint_failures={len(failures)}",
        flush=True,
    )
    if failures:
        for run_dir, issue in failures[:20]:
            print(f"- {run_dir}: {issue}")
        raise SystemExit("STRESS_GRID_BLOCKED_INVALID_CHECKPOINTS")
    if args.dry_run:
        print("STRESS_GRID_DRY_RUN_PASS")
        return

    rows = []
    index = 0
    for run_dir in run_dirs:
        for perturbation_seed in args.perturbation_seeds:
            index += 1
            print(
                f"[STRESS {index}/{total_evaluations}] run={run_dir} perturbation_seed={perturbation_seed}",
                flush=True,
            )
            evaluated = evaluate_robustness(run_dir, perturbation_seed=perturbation_seed, save=False)
            for row in evaluated:
                row["run_dir"] = str(run_dir)
            rows.extend(evaluated)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    raw = pd.DataFrame(rows)
    raw_path = output_dir / "stress_multiseed_seed_level.csv"
    metrics_rows_to_csv(rows, raw_path)
    summary = summarize_stress(raw, bootstrap_reps=args.bootstrap_reps)
    summary.to_csv(output_dir / "stress_multiseed_summary.csv", index=False)
    auc = degradation_auc(raw)
    auc.to_csv(output_dir / "stress_degradation_auc_seed_level.csv", index=False)
    auc_summary = auc.groupby(["subset", "model", "scenario"], as_index=False).mean(numeric_only=True)
    auc_summary.to_csv(output_dir / "stress_degradation_auc_summary.csv", index=False)
    print(f"STRESS_GRID_READY {output_dir}")


if __name__ == "__main__":
    main()
