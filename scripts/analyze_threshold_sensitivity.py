from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_formal_benchmark import FORMAL_BENCHMARK_MODELS, FORMAL_BENCHMARK_SEEDS, FORMAL_BENCHMARK_SUBSETS


def _prediction_path(results_dir: Path, seed: int, subset: str, model: str) -> Path:
    return results_dir / f"paper_main_v3_seed{seed}" / subset / model / "test_predictions.csv"


def compute_threshold_rows(
    results_dir: Path,
    *,
    seeds: list[int],
    subsets: list[str],
    models: list[str],
    rul_thresholds: list[float],
    late_error_thresholds: list[float],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for seed in seeds:
        for subset in subsets:
            for model in models:
                path = _prediction_path(results_dir, seed, subset, model)
                if not path.exists():
                    continue
                pred = pd.read_csv(path)
                err = pred["pred_rul"] - pred["true_rul"]
                for rul_threshold in rul_thresholds:
                    zone = pred["true_rul"] <= float(rul_threshold)
                    zone_count = int(zone.sum())
                    for late_threshold in late_error_thresholds:
                        if zone_count == 0:
                            ratio = 0.0
                        else:
                            ratio = float(((err > float(late_threshold)) & zone).sum() / zone_count)
                        metric_name = "LPR" if late_threshold == 0 else f"SLPR>{late_threshold:g}"
                        rows.append(
                            {
                                "seed": seed,
                                "subset": subset,
                                "model": model,
                                "rul_threshold": float(rul_threshold),
                                "late_error_threshold": float(late_threshold),
                                "metric_name": metric_name,
                                "zone_engine_count": zone_count,
                                "ratio": ratio,
                            }
                        )
    return rows


def summarize(rows: list[dict[str, object]]) -> pd.DataFrame:
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    summary = (
        df.groupby(["subset", "model", "rul_threshold", "late_error_threshold", "metric_name"], as_index=False)
        .agg(
            seed_count=("seed", "nunique"),
            zone_engine_count_mean=("zone_engine_count", "mean"),
            ratio_mean=("ratio", "mean"),
            ratio_std=("ratio", "std"),
        )
        .sort_values(["subset", "rul_threshold", "late_error_threshold", "model"])
        .reset_index(drop=True)
    )
    summary["ratio_std"] = summary["ratio_std"].fillna(0.0)
    return summary


def paired_threshold_bootstrap(
    results_dir: Path,
    *,
    seeds: list[int],
    subsets: list[str],
    models: list[str],
    rul_thresholds: list[float],
    late_error_thresholds: list[float],
    proposed_model: str,
    reps: int,
    random_seed: int = 20260711,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, object]] = []
    baselines = [model for model in models if model != proposed_model]
    for subset in subsets:
        proposed_by_seed = {
            seed: pd.read_csv(_prediction_path(results_dir, seed, subset, proposed_model)) for seed in seeds
        }
        for baseline in baselines:
            paired_by_seed = {}
            for seed in seeds:
                baseline_frame = pd.read_csv(_prediction_path(results_dir, seed, subset, baseline))
                proposed_frame = proposed_by_seed[seed]
                paired = baseline_frame.merge(
                    proposed_frame,
                    on=["unit_id", "true_rul"],
                    suffixes=("_baseline", "_proposed"),
                    validate="one_to_one",
                )
                paired_by_seed[seed] = paired
            for rul_threshold in rul_thresholds:
                for late_threshold in late_error_thresholds:
                    paired_differences = []
                    seed_vectors = {}
                    for seed, frame in paired_by_seed.items():
                        zone = frame["true_rul"].to_numpy(float) <= rul_threshold
                        baseline_late = (
                            frame.loc[zone, "pred_rul_baseline"].to_numpy(float)
                            - frame.loc[zone, "true_rul"].to_numpy(float)
                            > late_threshold
                        ).astype(float)
                        proposed_late = (
                            frame.loc[zone, "pred_rul_proposed"].to_numpy(float)
                            - frame.loc[zone, "true_rul"].to_numpy(float)
                            > late_threshold
                        ).astype(float)
                        seed_vectors[seed] = baseline_late - proposed_late
                        paired_differences.extend(seed_vectors[seed].tolist())
                    observed = float(np.mean(paired_differences))
                    vector_lengths = {len(vector) for vector in seed_vectors.values()}
                    if len(vector_lengths) != 1:
                        raise ValueError("Paired critical-zone engine counts must match across composite seeds.")
                    matrix = np.stack([seed_vectors[seed] for seed in seeds], axis=0)
                    engine_count = matrix.shape[1]
                    sampled_seed_indices = rng.integers(0, len(seeds), size=(reps, len(seeds)))
                    # Test-engine identities are crossed with composite seeds:
                    # every seed predicts the same official engines.  A single
                    # engine draw is therefore shared across all selected seeds
                    # in each bootstrap replicate.
                    sampled_engine_indices = rng.integers(0, engine_count, size=(reps, engine_count))
                    sampled = matrix[
                        sampled_seed_indices[:, :, None],
                        sampled_engine_indices[:, None, :],
                    ]
                    bootstrap = sampled.mean(axis=(1, 2))
                    rows.append(
                        {
                            "subset": subset,
                            "baseline_model": baseline,
                            "proposed_model": proposed_model,
                            "rul_threshold": rul_threshold,
                            "late_error_threshold": late_threshold,
                            "metric_name": "LPR" if late_threshold == 0 else f"SLPR>{late_threshold:g}",
                            "paired_difference_baseline_minus_proposed": observed,
                            "percentile_ci95_low": float(np.quantile(bootstrap, 0.025)),
                            "percentile_ci95_high": float(np.quantile(bootstrap, 0.975)),
                            "probability_proposed_lower": float(np.mean(bootstrap > 0.0)),
                            "probability_tie": float(np.mean(np.isclose(bootstrap, 0.0, atol=1e-12))),
                            "probability_proposed_higher": float(np.mean(bootstrap < 0.0)),
                            "composite_seed_count": len(seeds),
                            "test_engine_count": engine_count,
                            "bootstrap_repetitions": reps,
                            "resampling_design": "crossed composite-seed x matched-test-engine bootstrap",
                        }
                    )
    return pd.DataFrame(rows)


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Analyze LPR/SLPR threshold sensitivity from formal prediction files.")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "paper_outputs" / "tables"))
    parser.add_argument("--seeds", nargs="+", type=int, default=FORMAL_BENCHMARK_SEEDS)
    parser.add_argument("--subsets", nargs="+", default=FORMAL_BENCHMARK_SUBSETS)
    parser.add_argument("--models", nargs="+", default=FORMAL_BENCHMARK_MODELS)
    parser.add_argument("--rul-thresholds", nargs="+", type=float, default=[20.0, 30.0, 40.0, 50.0])
    parser.add_argument("--late-error-thresholds", nargs="+", type=float, default=[0.0, 5.0, 10.0, 15.0])
    parser.add_argument("--proposed-model", default="rast_gru_v2")
    parser.add_argument("--bootstrap-reps", type=int, default=2000)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows = compute_threshold_rows(
        Path(args.results_dir),
        seeds=args.seeds,
        subsets=args.subsets,
        models=args.models,
        rul_thresholds=args.rul_thresholds,
        late_error_thresholds=args.late_error_thresholds,
    )
    if not rows:
        raise FileNotFoundError("No prediction files found for threshold sensitivity analysis.")
    seed_level = pd.DataFrame(rows)
    summary = summarize(rows)
    seed_csv = output_dir / "table_threshold_sensitivity_seed_level.csv"
    summary_csv = output_dir / "table_threshold_sensitivity.csv"
    summary_md = output_dir / "table_threshold_sensitivity.md"
    seed_level.to_csv(seed_csv, index=False)
    summary.to_csv(summary_csv, index=False)
    summary_md.write_text(summary.to_markdown(index=False) + "\n", encoding="utf-8")
    paired = paired_threshold_bootstrap(
        Path(args.results_dir),
        seeds=args.seeds,
        subsets=args.subsets,
        models=args.models,
        rul_thresholds=args.rul_thresholds,
        late_error_thresholds=args.late_error_thresholds,
        proposed_model=args.proposed_model,
        reps=args.bootstrap_reps,
    )
    paired.to_csv(output_dir / "table_threshold_paired_bootstrap.csv", index=False)
    print(f"THRESHOLD_SENSITIVITY_PASS rows={len(rows)} summary_rows={len(summary)} output={summary_csv}")


if __name__ == "__main__":
    main()
