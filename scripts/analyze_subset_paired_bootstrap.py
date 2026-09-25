from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_advanced_evidence import MODEL_DISPLAY, PRACTICAL_THRESHOLDS, nasa_contribution
from run_formal_benchmark import FORMAL_BENCHMARK_MODELS, FORMAL_BENCHMARK_SEEDS, FORMAL_BENCHMARK_SUBSETS


PROPOSED = "rast_gru_v2"
METRICS = ("rmse", "mae", "nasa_per_engine", "lpr30")


def metric(true_rul: np.ndarray, baseline_error: np.ndarray, proposed_error: np.ndarray, name: str) -> float:
    if name == "rmse":
        return float(np.sqrt(np.mean(baseline_error**2)) - np.sqrt(np.mean(proposed_error**2)))
    if name == "mae":
        return float(np.mean(np.abs(baseline_error)) - np.mean(np.abs(proposed_error)))
    if name == "nasa_per_engine":
        return float(np.mean(nasa_contribution(baseline_error) - nasa_contribution(proposed_error)))
    if name == "lpr30":
        critical = true_rul <= 30.0
        return float(np.mean(baseline_error[critical] > 0.0) - np.mean(proposed_error[critical] > 0.0))
    raise ValueError(name)


def paired_arrays(results_dir: Path, subset: str, baseline: str) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    rows = {}
    for seed in FORMAL_BENCHMARK_SEEDS:
        root = results_dir / f"paper_main_v3_seed{seed}" / subset
        base = pd.read_csv(root / baseline / "test_predictions.csv")
        proposed = pd.read_csv(root / PROPOSED / "test_predictions.csv")
        merged = base.merge(
            proposed,
            on=["unit_id", "true_rul"],
            suffixes=("_baseline", "_proposed"),
            validate="one_to_one",
        )
        true = merged["true_rul"].to_numpy(float)
        rows[seed] = (
            true,
            merged["pred_rul_baseline"].to_numpy(float) - true,
            merged["pred_rul_proposed"].to_numpy(float) - true,
        )
    return rows


def bootstrap_cell(
    arrays: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]],
    name: str,
    *,
    reps: int,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    seeds = list(FORMAL_BENCHMARK_SEEDS)
    observed = float(np.mean([metric(*arrays[seed], name) for seed in seeds]))
    samples = np.empty(reps, dtype=float)
    for rep in range(reps):
        values = []
        for sampled_seed in rng.choice(seeds, size=len(seeds), replace=True):
            true, baseline_error, proposed_error = arrays[int(sampled_seed)]
            indices = rng.integers(0, len(true), size=len(true))
            values.append(metric(true[indices], baseline_error[indices], proposed_error[indices], name))
        samples[rep] = float(np.mean(values))
    threshold = PRACTICAL_THRESHOLDS[name]
    return {
        "mean_paired_difference_baseline_minus_proposed": observed,
        "percentile_ci95_low": float(np.quantile(samples, 0.025)),
        "percentile_ci95_high": float(np.quantile(samples, 0.975)),
        "probability_proposed_better": float(np.mean(samples > 0.0)),
        "probability_tie": float(np.mean(np.isclose(samples, 0.0, atol=1e-12))),
        "probability_proposed_worse": float(np.mean(samples < 0.0)),
        "practical_threshold": threshold,
        "probability_exceeds_practical_threshold": float(np.mean(samples > threshold)),
        "bootstrap_repetitions": reps,
        "composite_seed_count": len(seeds),
        "engine_count_per_seed": len(next(iter(arrays.values()))[0]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fixed-subset paired engine bootstrap evidence.")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "paper_outputs" / "advanced_evidence" / "subset_engine_paired_bootstrap.csv"))
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    args = parser.parse_args()

    rows = []
    baselines = [model for model in FORMAL_BENCHMARK_MODELS if model != PROPOSED]
    for baseline_index, baseline in enumerate(baselines):
        for subset_index, subset in enumerate(FORMAL_BENCHMARK_SUBSETS):
            arrays = paired_arrays(Path(args.results_dir), subset, baseline)
            for metric_index, name in enumerate(METRICS):
                rng = np.random.default_rng(20260715 + baseline_index * 1000 + subset_index * 100 + metric_index)
                rows.append(
                    {
                        "subset": subset,
                        "baseline_model": baseline,
                        "baseline_display": MODEL_DISPLAY[baseline],
                        "proposed_model": PROPOSED,
                        "metric": name,
                        **bootstrap_cell(arrays, name, reps=args.bootstrap_reps, rng=rng),
                    }
                )
    frame = pd.DataFrame(rows)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output, index=False)
    print(f"SUBSET_PAIRED_BOOTSTRAP_READY rows={len(frame)} reps={args.bootstrap_reps} output={output}")


if __name__ == "__main__":
    main()
