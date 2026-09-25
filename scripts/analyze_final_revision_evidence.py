from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_advanced_evidence import prediction_metrics
from rul.cmapss import load_subset
from rul.config import load_config
from rul.preprocessing import make_simulated_last_windows, split_units
from run_formal_benchmark import FORMAL_BENCHMARK_SEEDS, FORMAL_BENCHMARK_SUBSETS


PROPOSED = "rast_gru_v2"
METRICS = ("rmse", "mae", "nasa_per_engine", "lpr30", "mle30", "late_cvar95_30")


def metric_from_arrays(true: np.ndarray, pred: np.ndarray, metric: str) -> float:
    error = pred - true
    if metric == "rmse":
        return float(np.sqrt(np.mean(error**2)))
    if metric == "nasa_per_engine":
        contribution = np.where(error < 0.0, np.exp(-error / 13.0) - 1.0, np.exp(error / 10.0) - 1.0)
        return float(np.mean(contribution))
    critical = true <= 30.0
    late = np.maximum(error[critical], 0.0)
    if metric == "lpr30":
        return float(np.mean(error[critical] > 0.0))
    if metric == "late_cvar95_30":
        positive = late[late > 0.0]
        if len(positive) == 0:
            return 0.0
        threshold = float(np.quantile(positive, 0.95))
        return float(np.mean(positive[positive >= threshold]))
    raise ValueError(metric)


def paired_prediction_bootstrap(
    left_paths: dict[int, Path],
    right_paths: dict[int, Path],
    *,
    left_label: str,
    right_label: str,
    comparison: str,
    metrics: tuple[str, ...] = ("rmse", "nasa_per_engine", "lpr30", "late_cvar95_30"),
    reps: int = 5000,
) -> pd.DataFrame:
    paired: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for seed in FORMAL_BENCHMARK_SEEDS:
        left = pd.read_csv(left_paths[seed] / "test_predictions.csv")
        right = pd.read_csv(right_paths[seed] / "test_predictions.csv")
        merged = left.merge(
            right,
            on=["unit_id", "true_rul"],
            suffixes=("_left", "_right"),
            validate="one_to_one",
        )
        paired[seed] = (
            merged["true_rul"].to_numpy(float),
            merged["pred_rul_left"].to_numpy(float),
            merged["pred_rul_right"].to_numpy(float),
        )
    rows = []
    for metric_index, metric in enumerate(metrics):
        rng = np.random.default_rng(20260716 + metric_index * 17 + len(comparison))
        observed = np.mean(
            [
                metric_from_arrays(true, left, metric) - metric_from_arrays(true, right, metric)
                for true, left, right in paired.values()
            ]
        )
        samples = np.empty(reps, dtype=float)
        for rep in range(reps):
            values = []
            for sampled_seed in rng.choice(FORMAL_BENCHMARK_SEEDS, size=5, replace=True):
                true, left, right = paired[int(sampled_seed)]
                indices = rng.integers(0, len(true), size=len(true))
                values.append(
                    metric_from_arrays(true[indices], left[indices], metric)
                    - metric_from_arrays(true[indices], right[indices], metric)
                )
            samples[rep] = float(np.mean(values))
        rows.append(
            {
                "comparison": comparison,
                "left_model": left_label,
                "right_model": right_label,
                "metric": metric,
                "mean_difference_left_minus_right": float(observed),
                "percentile_ci95_low": float(np.quantile(samples, 0.025)),
                "percentile_ci95_high": float(np.quantile(samples, 0.975)),
                "probability_left_lower": float(np.mean(samples < 0.0)),
                "probability_right_lower": float(np.mean(samples > 0.0)),
                "bootstrap_repetitions": reps,
            }
        )
    return pd.DataFrame(rows)


def prediction_run(results: Path, seed: int, subset: str, variant: str) -> Path:
    if variant == "asym":
        return results / f"paper_main_v3_seed{seed}" / subset / PROPOSED
    if subset == "FD004":
        return results / f"ablation_fd004_seed{seed}_weighted_huber_no_asymmetry" / subset / PROPOSED
    return results / f"core_ocm_seed{seed}" / subset / PROPOSED


def summarize_core(results: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict] = []
    for variant in ("core", "asym"):
        for seed in FORMAL_BENCHMARK_SEEDS:
            for subset in FORMAL_BENCHMARK_SUBSETS:
                run_dir = prediction_run(results, seed, subset, variant)
                predictions = pd.read_csv(run_dir / "test_predictions.csv")
                rows.append({"variant": variant, "seed": seed, "subset": subset, **prediction_metrics(predictions)})
    subset = pd.DataFrame(rows)
    macro = subset.groupby(["variant", "seed"], as_index=False)[list(METRICS)].mean()
    summary = macro.groupby("variant", as_index=False).agg(
        seed_count=("seed", "nunique"),
        **{f"{metric}_mean": (metric, "mean") for metric in METRICS},
        **{f"{metric}_std": (metric, lambda x: x.std(ddof=1)) for metric in METRICS},
    )
    return subset, summary


def core_asym_bootstrap(results: Path, reps: int = 5000) -> pd.DataFrame:
    paired: dict[tuple[int, str], tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for seed in FORMAL_BENCHMARK_SEEDS:
        for subset in FORMAL_BENCHMARK_SUBSETS:
            core = pd.read_csv(prediction_run(results, seed, subset, "core") / "test_predictions.csv")
            asym = pd.read_csv(prediction_run(results, seed, subset, "asym") / "test_predictions.csv")
            merged = core.merge(
                asym,
                on=["unit_id", "true_rul"],
                suffixes=("_core", "_asym"),
                validate="one_to_one",
            )
            paired[(seed, subset)] = (
                merged["true_rul"].to_numpy(float),
                merged["pred_rul_core"].to_numpy(float),
                merged["pred_rul_asym"].to_numpy(float),
            )
    rows = []
    for metric_index, metric in enumerate(("rmse", "nasa_per_engine", "lpr30", "late_cvar95_30")):
        rng = np.random.default_rng(20260716 + 313 * metric_index)
        observed_values = []
        for seed in FORMAL_BENCHMARK_SEEDS:
            subset_values = []
            for subset in FORMAL_BENCHMARK_SUBSETS:
                true, core, asym = paired[(seed, subset)]
                subset_values.append(metric_from_arrays(true, core, metric) - metric_from_arrays(true, asym, metric))
            observed_values.append(float(np.mean(subset_values)))
        samples = np.empty(reps, dtype=float)
        for rep in range(reps):
            seed_values = []
            for sampled_seed in rng.choice(FORMAL_BENCHMARK_SEEDS, size=5, replace=True):
                subset_values = []
                for subset in FORMAL_BENCHMARK_SUBSETS:
                    true, core, asym = paired[(int(sampled_seed), subset)]
                    indices = rng.integers(0, len(true), size=len(true))
                    subset_values.append(
                        metric_from_arrays(true[indices], core[indices], metric)
                        - metric_from_arrays(true[indices], asym[indices], metric)
                    )
                seed_values.append(float(np.mean(subset_values)))
            samples[rep] = float(np.mean(seed_values))
        rows.append(
            {
                "metric": metric,
                "mean_difference_core_minus_asym": float(np.mean(observed_values)),
                "percentile_ci95_low": float(np.quantile(samples, 0.025)),
                "percentile_ci95_high": float(np.quantile(samples, 0.975)),
                "probability_core_lower": float(np.mean(samples < 0.0)),
                "probability_asym_lower": float(np.mean(samples > 0.0)),
                "bootstrap_repetitions": reps,
                "fixed_subset_count": 4,
                "composite_seed_count": 5,
            }
        )
    return pd.DataFrame(rows)


def summarize_predictions(paths: list[Path]) -> dict[str, float]:
    rows = [prediction_metrics(pd.read_csv(path / "test_predictions.csv")) for path in paths]
    result: dict[str, float] = {"seed_count": float(len(rows))}
    for metric in METRICS:
        values = np.asarray([row[metric] for row in rows], dtype=float)
        result[f"{metric}_mean"] = float(values.mean())
        result[f"{metric}_std"] = float(values.std(ddof=1))
    return result


def architecture_attribution(results: Path) -> pd.DataFrame:
    specs = [
        (
            "RAST-GRU",
            "Common Base",
            [results / f"paper_main_v3_seed{s}" / "FD004" / "rast_gru" for s in FORMAL_BENCHMARK_SEEDS],
        ),
        (
            "OCM backbone",
            "Common Base",
            [results / f"attribution_ocm_base_only_fd004_seed{s}" / "FD004" / PROPOSED for s in FORMAL_BENCHMARK_SEEDS],
        ),
        (
            "RAST-GRU",
            "Track B",
            [results / "standard_protocol_track" / f"standard_protocol_fd004_seed{s}" / "FD004" / "rast_gru" for s in FORMAL_BENCHMARK_SEEDS],
        ),
        (
            "OCM backbone",
            "Track B",
            [results / "standard_protocol_track" / f"standard_protocol_fd004_seed{s}" / "FD004" / PROPOSED for s in FORMAL_BENCHMARK_SEEDS],
        ),
        (
            "Full OCM",
            "Common Risk",
            [results / f"paper_main_v3_seed{s}" / "FD004" / PROPOSED for s in FORMAL_BENCHMARK_SEEDS],
        ),
    ]
    rows = []
    for model, protocol, paths in specs:
        missing = [path for path in paths if not (path / "test_predictions.csv").exists()]
        if missing:
            raise FileNotFoundError(missing[0])
        rows.append({"model_display": model, "protocol": protocol, **summarize_predictions(paths)})
    frame = pd.DataFrame(rows)
    for protocol, group in frame.groupby("protocol"):
        for metric in ("rmse_mean", "nasa_per_engine_mean", "lpr30_mean", "late_cvar95_30_mean"):
            frame.loc[group.index, f"{metric.removesuffix('_mean')}_rank_within_protocol"] = (
                group[metric].rank(method="average", ascending=True).to_numpy()
            )
    return frame


def track_b_absolute(source: Path) -> pd.DataFrame:
    frame = pd.read_csv(source / "standard_protocol_track_summary.csv")
    frame = frame[frame["track"] == "standardized_conventional_protocol"].copy()
    for metric in ("rmse_mean", "nasa_per_engine_mean", "lpr30_mean", "late_cvar95_30_mean"):
        frame[f"{metric.removesuffix('_mean')}_rank"] = frame[metric].rank(method="average")
    return frame.sort_values(["rmse_rank", "lpr30_rank"])


def selected_comparator_evidence(source: Path) -> pd.DataFrame:
    aggregation = pd.read_csv(source / "aggregation_summary.csv")
    fixed = pd.read_csv(source / "subset_engine_paired_bootstrap.csv")
    selectors = {
        "rmse": "macro_rmse_mean",
        "nasa_per_engine": "macro_nasa_per_engine_mean",
        "lpr30": "macro_lpr30_mean",
    }
    rows = []
    for metric, column in selectors.items():
        competitors = aggregation[aggregation["model"] != PROPOSED]
        selected = competitors.loc[competitors[column].idxmin()]
        part = fixed[(fixed["baseline_model"] == selected["model"]) & (fixed["metric"] == metric)].copy()
        part["selection_role"] = "metric-best controlled baseline"
        part["selection_metric"] = metric
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def endpoint_audit(results: Path) -> pd.DataFrame:
    rows = []
    for seed in FORMAL_BENCHMARK_SEEDS:
        for subset in FORMAL_BENCHMARK_SUBSETS:
            run_dir = results / f"paper_main_v3_seed{seed}" / subset / PROPOSED
            config = load_config(run_dir / "run_config.yaml")
            data_root = Path(config["data"]["root"])
            if not data_root.is_absolute():
                data_root = PROJECT_ROOT / data_root
            train, _ = load_subset(data_root, subset, config["data"].get("rul_cap", 125))
            _, val_units = split_units(
                train["unit_id"].to_numpy(),
                float(config["data"].get("validation_split", 0.2)),
                int(seed),
            )
            val = train[train["unit_id"].isin(val_units)].copy()
            endpoints = make_simulated_last_windows(
                val,
                ["setting_1"],
                int(config["data"]["window_size"]),
                seed=int(seed),
                min_rul=float(config["data"].get("validation_last_min_rul", 1.0)),
                max_rul=float(config["data"].get("validation_last_max_rul", 125.0)),
            )
            payload = np.column_stack([endpoints.unit_ids, endpoints.end_cycles, endpoints.y]).tobytes()
            rows.append(
                {
                    "subset": subset,
                    "seed": seed,
                    "validation_engines": int(len(endpoints.y)),
                    "n30": int(np.sum(endpoints.y <= 30.0)),
                    "n10": int(np.sum(endpoints.y <= 10.0)),
                    "truncation_rul_mean": float(np.mean(endpoints.y)),
                    "truncation_rul_min": float(np.min(endpoints.y)),
                    "truncation_rul_max": float(np.max(endpoints.y)),
                    "endpoint_draw_fixed_across_epochs": True,
                    "endpoint_seed": seed,
                    "endpoint_fingerprint_sha256": hashlib.sha256(payload).hexdigest(),
                }
            )
    return pd.DataFrame(rows)


def enhanced_efficiency(results: Path, fairness_path: Path) -> pd.DataFrame:
    fairness = pd.read_csv(fairness_path)
    rows = []
    for record in fairness.to_dict("records"):
        model = record["model"]
        metrics = []
        for seed in FORMAL_BENCHMARK_SEEDS:
            path = results / f"paper_main_v3_seed{seed}" / "FD004" / model / "metrics.json"
            metrics.append(json.loads(path.read_text(encoding="utf-8-sig")))
        executed = np.asarray([row["completed_epochs"] for row in metrics], dtype=float)
        best = np.asarray([row["best_epoch"] for row in metrics], dtype=float)
        elapsed = np.asarray([row["training_elapsed_sec"] for row in metrics], dtype=float)
        record.update(
            {
                "checkpoint_epoch_mean": float(best.mean()),
                "checkpoint_epoch_std": float(best.std(ddof=1)),
                "executed_epochs_mean": float(executed.mean()),
                "executed_epochs_std": float(executed.std(ddof=1)),
                "seconds_per_epoch_mean": float(np.mean(elapsed / executed)),
                "seconds_per_epoch_std": float(np.std(elapsed / executed, ddof=1)),
            }
        )
        rows.append(record)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build evidence requested in the final attribution-focused review.")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--source-dir", default=str(PROJECT_ROOT / "paper_outputs" / "advanced_evidence"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "paper_outputs" / "advanced_evidence"))
    args = parser.parse_args()
    results = Path(args.results_dir)
    source = Path(args.source_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    core_subset, core_summary = summarize_core(results)
    core_subset.to_csv(output / "ocm_core_asym_subset_seed.csv", index=False)
    core_summary.to_csv(output / "ocm_core_asym_summary.csv", index=False)
    core_asym_bootstrap(results).to_csv(output / "ocm_core_asym_paired_bootstrap.csv", index=False)
    architecture_attribution(results).to_csv(output / "architecture_attribution_absolute.csv", index=False)
    track_b = track_b_absolute(source)
    track_b.to_csv(output / "track_b_absolute_ranking.csv", index=False)
    selected_comparator_evidence(source).to_csv(output / "selected_comparator_evidence.csv", index=False)
    common_base_bootstrap = paired_prediction_bootstrap(
        {s: results / f"attribution_ocm_base_only_fd004_seed{s}" / "FD004" / PROPOSED for s in FORMAL_BENCHMARK_SEEDS},
        {s: results / f"paper_main_v3_seed{s}" / "FD004" / "rast_gru" for s in FORMAL_BENCHMARK_SEEDS},
        left_label="OCM backbone",
        right_label="RAST-GRU",
        comparison="common_base_architecture",
    )
    strongest_track_b = track_b[track_b["model"] != PROPOSED].assign(
        mean_rank=track_b[["rmse_rank", "nasa_per_engine_rank", "lpr30_rank"]].mean(axis=1)
    ).sort_values("mean_rank").iloc[0]["model"]
    track_b_bootstrap = paired_prediction_bootstrap(
        {
            s: results / "standard_protocol_track" / f"standard_protocol_fd004_seed{s}" / "FD004" / PROPOSED
            for s in FORMAL_BENCHMARK_SEEDS
        },
        {
            s: results / "standard_protocol_track" / f"standard_protocol_fd004_seed{s}" / "FD004" / strongest_track_b
            for s in FORMAL_BENCHMARK_SEEDS
        },
        left_label="OCM backbone",
        right_label=str(strongest_track_b),
        comparison="track_b_strongest_competitor",
    )
    pd.concat([common_base_bootstrap, track_b_bootstrap], ignore_index=True).to_csv(
        output / "architecture_selected_comparator_bootstrap.csv", index=False
    )
    endpoint_audit(results).to_csv(output / "validation_endpoint_audit.csv", index=False)
    enhanced_efficiency(results, source / "benchmark_fairness.csv").to_csv(
        output / "benchmark_efficiency_enhanced.csv", index=False
    )
    print(
        "FINAL_REVISION_EVIDENCE_READY "
        f"core_rows={len(core_subset)} endpoint_rows={len(FORMAL_BENCHMARK_SEEDS) * len(FORMAL_BENCHMARK_SUBSETS)}"
    )


if __name__ == "__main__":
    main()
