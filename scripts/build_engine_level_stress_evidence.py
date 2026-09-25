from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
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
PROJECT_ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_stress_pair_extension import SCENARIOS
from rul.config import load_config
from rul.experiments import evaluate_robustness
from run_formal_benchmark import FORMAL_BENCHMARK_SEEDS
from run_multiseed_stress_tests import PERTURBATION_SEEDS


POINTS = ("RAST", "Core", "Asym")
PRIMARY_METRICS = (
    "rmse",
    "critical_30_late_prediction_ratio",
    "critical_30_signed_error_mean",
    "critical_30_early_error_magnitude",
    "critical_30_mean_late_excess",
    "critical_30_late_cvar95",
    "critical_30_decision_cost_2",
    "critical_30_decision_cost_5",
    "critical_30_decision_cost_10",
)


def run_dir(point: str, seed: int) -> Path:
    if point == "RAST":
        return PROJECT_ROOT / "results" / f"paper_main_v3_seed{seed}" / "FD004" / "rast_gru"
    if point == "Asym":
        return PROJECT_ROOT / "results" / f"paper_main_v3_seed{seed}" / "FD004" / "rast_gru_v2"
    if point == "Core":
        return (
            PROJECT_ROOT
            / "results"
            / f"ablation_fd004_seed{seed}_weighted_huber_no_asymmetry"
            / "FD004"
            / "rast_gru_v2"
        )
    raise ValueError(point)


def collect_engine_predictions(output: Path) -> pd.DataFrame:
    rows: list[dict] = []
    common_robustness = load_config(PROJECT_ROOT / "configs" / "paper_main.yaml")["robustness"]
    total = len(POINTS) * len(FORMAL_BENCHMARK_SEEDS) * len(PERTURBATION_SEEDS)
    index = 0
    for point in POINTS:
        for training_seed in FORMAL_BENCHMARK_SEEDS:
            for perturbation_seed in PERTURBATION_SEEDS:
                index += 1
                print(
                    f"[ENGINE STRESS {index}/{total}] point={point} "
                    f"train={training_seed} perturb={perturbation_seed}",
                    flush=True,
                )
                engine_rows: list[dict] = []
                evaluate_robustness(
                    run_dir(point, training_seed),
                    perturbation_seed=perturbation_seed,
                    save=False,
                    engine_rows=engine_rows,
                    robustness_override=common_robustness,
                )
                for row in engine_rows:
                    row["operating_point"] = point
                rows.extend(engine_rows)
    frame = pd.DataFrame(rows)
    frame.to_csv(output, index=False)
    return frame


def _curve_bootstrap_values(
    pair: pd.DataFrame,
    point: str,
    scenario: str,
    engine_draws: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, float]]:
    point_frame = pair[pair["operating_point"] == point]
    clean = point_frame[point_frame["scenario"] == "clean"]
    fault = point_frame[point_frame["scenario"] == scenario]
    levels = np.array([0.0, *sorted(fault["level"].astype(float).unique())], dtype=float)
    units = np.array(sorted(clean["unit_id"].astype(int).unique()), dtype=int)
    squared = []
    late = []
    critical = []
    errors = []
    for level in levels:
        level_frame = clean if level == 0.0 else fault[np.isclose(fault["level"].astype(float), level)]
        level_frame = level_frame.set_index("unit_id").loc[units]
        squared.append(level_frame["squared_error"].to_numpy(float))
        late.append(level_frame["late_indicator_30"].to_numpy(float))
        critical.append(level_frame["critical_30"].to_numpy(float))
        errors.append(level_frame["error"].to_numpy(float))
    squared_array = np.stack(squared)
    late_array = np.stack(late)
    critical_array = np.stack(critical)
    error_array = np.stack(errors)
    sampled_rmse = np.sqrt(squared_array[:, engine_draws].mean(axis=2)).T
    sampled_lpr = (
        late_array[:, engine_draws].sum(axis=2)
        / np.maximum(critical_array[:, engine_draws].sum(axis=2), 1.0)
    ).T
    sampled_error = np.transpose(error_array[:, engine_draws], (1, 0, 2))
    sampled_critical = np.transpose(critical_array[:, engine_draws], (1, 0, 2))
    sampled_denominator = np.maximum(sampled_critical.sum(axis=2), 1.0)
    sampled_positive = np.maximum(sampled_error, 0.0) * sampled_critical
    sampled_early = np.maximum(-sampled_error, 0.0) * sampled_critical
    sampled_signed = (sampled_positive - sampled_early).sum(axis=2) / sampled_denominator
    sampled_early_magnitude = sampled_early.sum(axis=2) / sampled_denominator
    sampled_mle = sampled_positive.sum(axis=2) / sampled_denominator
    sampled_cost2 = (sampled_early + 2.0 * sampled_positive).sum(axis=2) / sampled_denominator
    sampled_cost5 = (sampled_early + 5.0 * sampled_positive).sum(axis=2) / sampled_denominator
    sampled_cost10 = (sampled_early + 10.0 * sampled_positive).sum(axis=2) / sampled_denominator
    sampled_cvar = np.zeros(sampled_mle.shape, dtype=float)
    for level_index in range(len(levels)):
        values = sampled_error[:, level_index, :].copy()
        valid = (values > 0.0) & (sampled_critical[:, level_index, :] > 0.0)
        values[~valid] = np.nan
        has_valid = np.any(valid, axis=1)
        quantiles = np.full(values.shape[0], np.inf, dtype=float)
        quantiles[has_valid] = np.nanquantile(values[has_valid], 0.95, axis=1)
        tail = valid & (values >= quantiles[:, None])
        numerator = np.nansum(np.where(tail, values, np.nan), axis=1)
        denominator = np.maximum(tail.sum(axis=1), 1)
        sampled_cvar[:, level_index] = np.where(has_valid, numerator / denominator, 0.0)
    width = float(levels[-1] - levels[0])
    bootstrap = {
        "rmse": TRAPEZOID(sampled_rmse, levels, axis=1) / width,
        "critical_30_late_prediction_ratio": TRAPEZOID(sampled_lpr, levels, axis=1) / width,
        "critical_30_signed_error_mean": TRAPEZOID(sampled_signed, levels, axis=1) / width,
        "critical_30_early_error_magnitude": TRAPEZOID(sampled_early_magnitude, levels, axis=1) / width,
        "critical_30_mean_late_excess": TRAPEZOID(sampled_mle, levels, axis=1) / width,
        "critical_30_late_cvar95": TRAPEZOID(sampled_cvar, levels, axis=1) / width,
        "critical_30_decision_cost_2": TRAPEZOID(sampled_cost2, levels, axis=1) / width,
        "critical_30_decision_cost_5": TRAPEZOID(sampled_cost5, levels, axis=1) / width,
        "critical_30_decision_cost_10": TRAPEZOID(sampled_cost10, levels, axis=1) / width,
    }
    observed_rmse = np.sqrt(squared_array.mean(axis=1))
    observed_lpr = late_array.sum(axis=1) / np.maximum(critical_array.sum(axis=1), 1.0)
    observed_positive = np.maximum(error_array, 0.0) * critical_array
    observed_early = np.maximum(-error_array, 0.0) * critical_array
    observed_denominator = np.maximum(critical_array.sum(axis=1), 1.0)
    observed_signed = (observed_positive - observed_early).sum(axis=1) / observed_denominator
    observed_early_magnitude = observed_early.sum(axis=1) / observed_denominator
    observed_mle = observed_positive.sum(axis=1) / observed_denominator
    observed_cost2 = (observed_early + 2.0 * observed_positive).sum(axis=1) / observed_denominator
    observed_cost5 = (observed_early + 5.0 * observed_positive).sum(axis=1) / observed_denominator
    observed_cost10 = (observed_early + 10.0 * observed_positive).sum(axis=1) / observed_denominator
    observed_cvar = []
    for level_index in range(len(levels)):
        values = error_array[level_index][
            (error_array[level_index] > 0.0) & (critical_array[level_index] > 0.0)
        ]
        if len(values):
            quantile = np.quantile(values, 0.95)
            observed_cvar.append(float(np.mean(values[values >= quantile])))
        else:
            observed_cvar.append(0.0)
    observed_cvar = np.asarray(observed_cvar, dtype=float)
    observed = {
        "rmse": float(TRAPEZOID(observed_rmse, levels) / width),
        "critical_30_late_prediction_ratio": float(TRAPEZOID(observed_lpr, levels) / width),
        "critical_30_signed_error_mean": float(TRAPEZOID(observed_signed, levels) / width),
        "critical_30_early_error_magnitude": float(TRAPEZOID(observed_early_magnitude, levels) / width),
        "critical_30_mean_late_excess": float(TRAPEZOID(observed_mle, levels) / width),
        "critical_30_late_cvar95": float(TRAPEZOID(observed_cvar, levels) / width),
        "critical_30_decision_cost_2": float(TRAPEZOID(observed_cost2, levels) / width),
        "critical_30_decision_cost_5": float(TRAPEZOID(observed_cost5, levels) / width),
        "critical_30_decision_cost_10": float(TRAPEZOID(observed_cost10, levels) / width),
    }
    return bootstrap, observed


def build_hierarchical_intervals(
    predictions: pd.DataFrame,
    *,
    bootstrap_reps: int,
    engine_pool_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    training_seeds = list(FORMAL_BENCHMARK_SEEDS)
    perturbation_seeds = list(PERTURBATION_SEEDS)
    seed_index = {seed: index for index, seed in enumerate(training_seeds)}
    perturb_index = {seed: index for index, seed in enumerate(perturbation_seeds)}
    pool = {
        metric: {
            point: {scenario: np.empty((5, 5, engine_pool_size), dtype=np.float64) for scenario in SCENARIOS}
            for point in POINTS
        }
        for metric in PRIMARY_METRICS
    }
    observed = {
        metric: {point: {scenario: np.empty((5, 5), dtype=np.float64) for scenario in SCENARIOS} for point in POINTS}
        for metric in PRIMARY_METRICS
    }
    failure_pool = {
        point: {level: np.empty((5, 5, engine_pool_size), dtype=np.float64) for level in (0.1, 0.2, 0.3, 0.4)}
        for point in POINTS
    }
    failure_observed = {
        point: {level: np.empty((5, 5), dtype=np.float64) for level in (0.1, 0.2, 0.3, 0.4)}
        for point in POINTS
    }

    # The three experimental factors are crossed.  Reusing one engine
    # bootstrap pool across every training-seed x perturbation-seed cell
    # preserves the correlation induced by repeatedly evaluating the same
    # official FD004 engines.
    shared_engine_rng = np.random.default_rng(20260720)
    shared_engine_draws = shared_engine_rng.integers(
        0, 248, size=(engine_pool_size, 248)
    )

    for training_seed in training_seeds:
        for perturbation_seed in perturbation_seeds:
            pair = predictions[
                (predictions["training_seed"] == training_seed)
                & (predictions["perturbation_seed"] == perturbation_seed)
            ]
            units = np.array(sorted(pair[pair["scenario"] == "clean"]["unit_id"].astype(int).unique()))
            if len(units) != 248:
                raise ValueError(f"Expected 248 matched FD004 engines, found {len(units)}")
            engine_draws = shared_engine_draws
            i = seed_index[training_seed]
            j = perturb_index[perturbation_seed]
            for point in POINTS:
                for scenario in SCENARIOS:
                    sampled, point_values = _curve_bootstrap_values(pair, point, scenario, engine_draws)
                    for metric in PRIMARY_METRICS:
                        pool[metric][point][scenario][i, j] = sampled[metric]
                        observed[metric][point][scenario][i, j] = point_values[metric]
                global_frame = pair[
                    (pair["operating_point"] == point)
                    & (pair["scenario"] == "global_sensor_missing")
                ]
                for level in failure_pool[point]:
                    level_frame = global_frame[np.isclose(global_frame["level"].astype(float), level)]
                    level_frame = level_frame.set_index("unit_id").loc[units]
                    late = level_frame["late_indicator_30"].to_numpy(float)
                    critical = level_frame["critical_30"].to_numpy(float)
                    failure_pool[point][level][i, j] = late[engine_draws].sum(axis=1) / np.maximum(
                        critical[engine_draws].sum(axis=1), 1.0
                    )
                    failure_observed[point][level][i, j] = float(late.sum() / critical.sum())

    rng = np.random.default_rng(20260721)
    training_draw = rng.integers(0, 5, size=(bootstrap_reps, 5))
    perturbation_draw = rng.integers(0, 5, size=(bootstrap_reps, 5))
    engine_pool_draw = rng.integers(0, engine_pool_size, size=bootstrap_reps)
    bootstrap_index = engine_pool_draw[:, None, None]

    def resample(values: np.ndarray) -> np.ndarray:
        selected = values[
            training_draw[:, :, None],
            perturbation_draw[:, None, :],
            bootstrap_index,
        ]
        return selected.mean(axis=(1, 2))

    rows = []
    for point in ("Core", "Asym"):
        for scenario in SCENARIOS:
            for metric in PRIMARY_METRICS:
                point_samples = resample(pool[metric][point][scenario])
                rast_samples = resample(pool[metric]["RAST"][scenario])
                difference = point_samples - rast_samples
                low, high = np.quantile(difference, [0.025, 0.975])
                classification = "unresolved"
                if high < 0.0:
                    classification = f"CI supports lower {point}"
                elif low > 0.0:
                    classification = "CI supports lower RAST"
                rows.append(
                    {
                        "operating_point": point,
                        "comparator": "RAST",
                        "scenario": scenario,
                        "metric": metric,
                        "mean_difference_point_minus_rast": float(
                            observed[metric][point][scenario].mean()
                            - observed[metric]["RAST"][scenario].mean()
                        ),
                        "hierarchical_bootstrap_ci95_low": float(low),
                        "hierarchical_bootstrap_ci95_high": float(high),
                        "probability_point_lower": float(np.mean(difference < 0.0)),
                        "bootstrap_sign_proportion_point_lower": float(np.mean(difference < 0.0)),
                        "ci_classification": classification,
                        "training_seed_count": 5,
                        "perturbation_seed_count": 5,
                        "test_engine_count": 248,
                        "bootstrap_repetitions": bootstrap_reps,
                        "engine_bootstrap_pool_size": engine_pool_size,
                        "inference_unit": "crossed composite-training-seed x perturbation-seed x matched-FD004-test-engine",
                    }
                )

    failure_rows = []
    for point in POINTS:
        for level in failure_pool[point]:
            samples = resample(failure_pool[point][level])
            low, high = np.quantile(samples, [0.025, 0.975])
            failure_rows.append(
                {
                    "operating_point": point,
                    "level": level,
                    "lpr_mean": float(failure_observed[point][level].mean()),
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                    "training_seed_count": 5,
                    "perturbation_seed_count": 5,
                    "test_engine_count": 248,
                    "bootstrap_repetitions": bootstrap_reps,
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(failure_rows)


def plot_failure_case(frame: pd.DataFrame, output: Path) -> None:
    styles = {
        "RAST": ("RAST-GRU", "#2f6f9f", "o"),
        "Core": ("OCM-Core", "#2a9d8f", "^"),
        "Asym": ("OCM-Asym", "#c44e52", "s"),
    }
    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for point in POINTS:
        group = frame[frame["operating_point"] == point].sort_values("level")
        label, color, marker = styles[point]
        ax.plot(group["level"], group["lpr_mean"], label=label, color=color, marker=marker, linewidth=2.2)
        ax.fill_between(group["level"], group["ci95_low"], group["ci95_high"], color=color, alpha=0.12)
    ax.set_xlabel("Global sensor-missing rate")
    ax.set_ylabel("LPR@30")
    ax.grid(linestyle=":", alpha=0.4)
    ax.legend(frameon=False, ncol=3)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build crossed training-seed/perturbation-seed/engine stress-test intervals."
    )
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "paper_outputs" / "advanced_evidence"))
    parser.add_argument("--reuse-predictions", action="store_true")
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--engine-bootstrap-pool", type=int, default=1000)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    figure_dir = output / "figures"
    figure_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = output / "stress_engine_level_predictions.csv"
    if args.reuse_predictions:
        predictions = pd.read_csv(predictions_path)
    else:
        predictions = collect_engine_predictions(predictions_path)
    intervals, failure = build_hierarchical_intervals(
        predictions,
        bootstrap_reps=args.bootstrap_reps,
        engine_pool_size=args.engine_bootstrap_pool,
    )
    intervals.to_csv(output / "stress_operating_points_engine_bootstrap.csv", index=False)
    failure.to_csv(output / "stress_global_missing_failure_curve.csv", index=False)
    plot_failure_case(failure, figure_dir / "fig_stress_global_missing_failure")
    print(
        f"ENGINE_LEVEL_STRESS_READY predictions={len(predictions)} "
        f"intervals={len(intervals)} failure_rows={len(failure)}"
    )


if __name__ == "__main__":
    main()
