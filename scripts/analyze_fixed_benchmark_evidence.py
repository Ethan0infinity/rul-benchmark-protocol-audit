"""Build fixed-benchmark estimation and calibration evidence."""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import beta as beta_distribution

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from analyze_normalized_perturbation_evidence import (
    DIRECT_PREDECESSOR,
    PRIMARY_METRICS,
    PROPOSED,
    RUL_THRESHOLDS,
    STRESS_METRICS,
    TOLERANCES,
    centered_bootstrap_p,
    engine_count_matrix,
    fixed_task_arrays,
    holm_adjust,
    sampled_seed_metrics,
)
from build_advanced_evidence import MODEL_DISPLAY
from run_formal_benchmark import FORMAL_BENCHMARK_SEEDS, FORMAL_BENCHMARK_SUBSETS


def exact_seed_signflip_p(seed_effects: np.ndarray) -> float:
    effects = np.asarray(seed_effects, dtype=float)
    observed = abs(float(effects.mean()))
    signs = np.asarray(list(itertools.product((-1.0, 1.0), repeat=len(effects))))
    null_statistics = np.abs((signs * effects[None, :]).mean(axis=1))
    return float(np.mean(null_statistics >= observed - 1e-15))


def diagnostic_max_t_envelope(
    bootstrap_samples: np.ndarray,
    observed: np.ndarray,
    *,
    confidence: float = 0.95,
    near_zero_se: float = 1e-12,
) -> dict[str, np.ndarray | float]:
    """Construct a studentized max-t diagnostic envelope.

    Columns are contrasts and rows are shared bootstrap draws. Standard errors
    are bootstrap standard deviations. Contrasts with a standard error no
    larger than ``near_zero_se`` receive a zero-width envelope and do not
    contribute to the maximum statistic. The construction is diagnostic
    because five training-seed levels are insufficient to establish nominal
    family-wise coverage.
    """

    samples = np.asarray(bootstrap_samples, dtype=float)
    point = np.asarray(observed, dtype=float)
    if samples.ndim != 2 or point.ndim != 1 or samples.shape[1] != len(point):
        raise ValueError("bootstrap_samples must be B x K and observed must have length K")
    standard_error = samples.std(axis=0, ddof=1)
    active = np.isfinite(standard_error) & (standard_error > float(near_zero_se))
    centered_t = np.zeros_like(samples)
    centered_t[:, active] = np.abs(
        (samples[:, active] - point[None, active]) / standard_error[None, active]
    )
    max_t = centered_t.max(axis=1)
    critical_value = float(np.quantile(max_t, confidence))
    half_width = critical_value * standard_error
    half_width[~active] = 0.0
    adjusted_p = np.ones(len(point), dtype=float)
    for index in np.flatnonzero(active):
        observed_t = abs(point[index] / standard_error[index])
        adjusted_p[index] = float(
            (np.sum(max_t >= observed_t) + 1) / (len(max_t) + 1)
        )
    return {
        "standard_error": standard_error,
        "active_contrast": active,
        "max_t": max_t,
        "critical_value": critical_value,
        "interval_low": point - half_width,
        "interval_high": point + half_width,
        "adjusted_p": adjusted_p,
    }


def joint_primary_inference(
    results_dir: Path,
    *,
    reps: int,
    random_seed: int,
) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(random_seed)
    seed_count = len(FORMAL_BENCHMARK_SEEDS)
    seed_draw = rng.integers(0, seed_count, size=(reps, seed_count))
    rep_index = np.arange(reps)[:, None]
    records: list[dict[str, object]] = []
    bootstrap_columns: list[np.ndarray] = []

    for subset in FORMAL_BENCHMARK_SUBSETS:
        truth, proposed, predecessor = fixed_task_arrays(results_dir, subset)
        engine_draw = rng.integers(0, len(truth), size=(reps, len(truth)))
        counts = engine_count_matrix(engine_draw, len(truth))
        unit_counts = np.ones((len(truth), 1), dtype=np.int16)
        for metric in PRIMARY_METRICS:
            proposed_sample = sampled_seed_metrics(truth, proposed, counts, metric)
            predecessor_sample = sampled_seed_metrics(truth, predecessor, counts, metric)
            samples = (
                proposed_sample[seed_draw, rep_index].mean(axis=1)
                - predecessor_sample[seed_draw, rep_index].mean(axis=1)
            )
            seed_effects = np.asarray(
                [
                    sampled_seed_metrics(
                        truth,
                        proposed[index : index + 1],
                        unit_counts,
                        metric,
                    )[0, 0]
                    - sampled_seed_metrics(
                        truth,
                        predecessor[index : index + 1],
                        unit_counts,
                        metric,
                    )[0, 0]
                    for index in range(seed_count)
                ],
                dtype=float,
            )
            observed = float(seed_effects.mean())
            p_value = centered_bootstrap_p(samples, observed)
            low, high = np.quantile(samples, [0.025, 0.975])
            records.append(
                {
                    "evidence_level": "MAIN FIXED-BENCHMARK ESTIMATION",
                    "subset": subset,
                    "comparison": "OCM-Asym minus RAST-GRU",
                    "metric": metric,
                    "mean_difference": observed,
                    "nominal_crossed_ci95_low": float(low),
                    "nominal_crossed_ci95_high": float(high),
                    "centered_bootstrap_p": p_value,
                    "bootstrap_p_mcse": math.sqrt(p_value * (1.0 - p_value) / (reps + 1)),
                    "minimum_attainable_bootstrap_p": 1.0 / (reps + 1),
                    "exact_seed_signflip_p": exact_seed_signflip_p(seed_effects),
                    "seed_effect_min": float(seed_effects.min()),
                    "seed_effect_max": float(seed_effects.max()),
                    "composite_seed_count": seed_count,
                    "matched_test_engine_count": len(truth),
                    "bootstrap_repetitions": reps,
                }
            )
            bootstrap_columns.append(samples)

    frame = pd.DataFrame(records)
    samples_matrix = np.column_stack(bootstrap_columns)
    frame["holm_adjusted_p_metric_family"] = np.nan
    for _, indices in frame.groupby("metric").groups.items():
        frame.loc[indices, "holm_adjusted_p_metric_family"] = holm_adjust(
            frame.loc[indices, "centered_bootstrap_p"].to_numpy(float)
        )
    frame["holm_adjusted_p_all12"] = holm_adjust(frame["centered_bootstrap_p"].to_numpy(float))

    observed = frame["mean_difference"].to_numpy(float)
    envelope = diagnostic_max_t_envelope(samples_matrix, observed)
    frame["bootstrap_standard_error"] = envelope["standard_error"]
    frame["max_t_critical_value_diagnostic"] = envelope["critical_value"]
    frame["diagnostic_max_t_envelope_low"] = envelope["interval_low"]
    frame["diagnostic_max_t_envelope_high"] = envelope["interval_high"]
    frame["max_t_diagnostic_p"] = envelope["adjusted_p"]
    frame["max_t_active_contrast"] = envelope["active_contrast"]
    frame["primary_estimation_status"] = "fixed-benchmark estimate; no confirmatory decision"
    return frame, samples_matrix


def primary_seed_level_effects(results_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subset in FORMAL_BENCHMARK_SUBSETS:
        truth, proposed, predecessor = fixed_task_arrays(results_dir, subset)
        unit_counts = np.ones((len(truth), 1), dtype=np.int16)
        for metric in PRIMARY_METRICS:
            proposed_metric = sampled_seed_metrics(truth, proposed, unit_counts, metric)[:, 0]
            predecessor_metric = sampled_seed_metrics(truth, predecessor, unit_counts, metric)[:, 0]
            for seed, effect in zip(
                FORMAL_BENCHMARK_SEEDS,
                proposed_metric - predecessor_metric,
            ):
                rows.append(
                    {
                        "subset": subset,
                        "metric": metric,
                        "composite_seed": int(seed),
                        "ocm_asym_minus_rast": float(effect),
                        "matched_test_engine_count": len(truth),
                        "role": "seed-level paired diagnostic; fixed task",
                    }
                )
    return pd.DataFrame(rows)


def development_excluded_seed_effects(results_dir: Path) -> pd.DataFrame:
    """Report FD001--FD003 effects without treating tasks as a sampled population."""

    subsets = FORMAL_BENCHMARK_SUBSETS[:3]
    cached = {subset: fixed_task_arrays(results_dir, subset) for subset in subsets}
    rows: list[dict[str, object]] = []
    for metric in PRIMARY_METRICS:
        task_seed_effects: dict[str, np.ndarray] = {}
        truths: list[np.ndarray] = []
        proposed: list[np.ndarray] = []
        predecessor: list[np.ndarray] = []
        for subset in subsets:
            truth, point, direct = cached[subset]
            unit_counts = np.ones((len(truth), 1), dtype=np.int16)
            effects = (
                sampled_seed_metrics(truth, point, unit_counts, metric)[:, 0]
                - sampled_seed_metrics(truth, direct, unit_counts, metric)[:, 0]
            )
            task_seed_effects[subset] = effects
            truths.append(truth)
            proposed.append(point)
            predecessor.append(direct)
            for seed, effect in zip(FORMAL_BENCHMARK_SEEDS, effects):
                rows.append(
                    {
                        "scope": "FD001-FD003 development-task-excluded",
                        "aggregation": "task-specific",
                        "subset": subset,
                        "metric": metric,
                        "composite_seed": int(seed),
                        "ocm_minus_rast": float(effect),
                        "task_count": 1,
                        "engine_count": len(truth),
                        "interpretation": "descriptive paired seed effect; no task-population inference",
                    }
                )

        pooled_truth = np.concatenate(truths)
        pooled_proposed = np.concatenate(proposed, axis=1)
        pooled_predecessor = np.concatenate(predecessor, axis=1)
        pooled_counts = np.ones((len(pooled_truth), 1), dtype=np.int16)
        pooled_effects = (
            sampled_seed_metrics(
                pooled_truth,
                pooled_proposed,
                pooled_counts,
                metric,
            )[:, 0]
            - sampled_seed_metrics(
                pooled_truth,
                pooled_predecessor,
                pooled_counts,
                metric,
            )[:, 0]
        )
        macro_effects = np.vstack(
            [task_seed_effects[subset] for subset in subsets]
        ).mean(axis=0)
        for seed_index, seed in enumerate(FORMAL_BENCHMARK_SEEDS):
            for aggregation, effect in (
                ("equal-task macro", macro_effects[seed_index]),
                ("engine-pooled recomputation", pooled_effects[seed_index]),
            ):
                rows.append(
                    {
                        "scope": "FD001-FD003 development-task-excluded",
                        "aggregation": aggregation,
                        "subset": "FD001-FD003",
                        "metric": metric,
                        "composite_seed": int(seed),
                        "ocm_minus_rast": float(effect),
                        "task_count": len(subsets),
                        "engine_count": len(pooled_truth),
                        "interpretation": (
                            "descriptive fixed-task aggregation; tasks are not resampled"
                        ),
                    }
                )
    return pd.DataFrame(rows)


def fixed_transfer_summaries(results_dir: Path) -> pd.DataFrame:
    """Separate transfer-only, all-task, equal-task, and engine-pooled estimands."""

    cached: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        subset: fixed_task_arrays(results_dir, subset)
        for subset in FORMAL_BENCHMARK_SUBSETS
    }
    rows: list[dict[str, object]] = []
    scopes = {
        "FD001-FD003 fixed-transfer": FORMAL_BENCHMARK_SUBSETS[:3],
        "FD001-FD004 all fixed tasks": FORMAL_BENCHMARK_SUBSETS,
    }
    for scope, subsets in scopes.items():
        for metric in PRIMARY_METRICS:
            task_effects: list[float] = []
            truths: list[np.ndarray] = []
            proposed: list[np.ndarray] = []
            predecessor: list[np.ndarray] = []
            for subset in subsets:
                truth, point, direct = cached[subset]
                unit_counts = np.ones((len(truth), 1), dtype=np.int16)
                point_metric = sampled_seed_metrics(truth, point, unit_counts, metric)[:, 0]
                direct_metric = sampled_seed_metrics(truth, direct, unit_counts, metric)[:, 0]
                task_effects.append(float((point_metric - direct_metric).mean()))
                truths.append(truth)
                proposed.append(point)
                predecessor.append(direct)
            pooled_truth = np.concatenate(truths)
            pooled_proposed = np.concatenate(proposed, axis=1)
            pooled_predecessor = np.concatenate(predecessor, axis=1)
            pooled_counts = np.ones((len(pooled_truth), 1), dtype=np.int16)
            pooled_effect = (
                sampled_seed_metrics(pooled_truth, pooled_proposed, pooled_counts, metric)[:, 0]
                - sampled_seed_metrics(pooled_truth, pooled_predecessor, pooled_counts, metric)[:, 0]
            )
            rows.extend(
                [
                    {
                        "scope": scope,
                        "aggregation": "equal-task macro",
                        "metric": metric,
                        "ocm_minus_rast": float(np.mean(task_effects)),
                        "task_count": len(subsets),
                        "engine_count": int(sum(len(item) for item in truths)),
                        "composite_seed_count": len(FORMAL_BENCHMARK_SEEDS),
                        "interpretation": "descriptive fixed-task estimand; tasks are not resampled",
                    },
                    {
                        "scope": scope,
                        "aggregation": "engine-pooled recomputation",
                        "metric": metric,
                        "ocm_minus_rast": float(pooled_effect.mean()),
                        "task_count": len(subsets),
                        "engine_count": int(len(pooled_truth)),
                        "composite_seed_count": len(FORMAL_BENCHMARK_SEEDS),
                        "interpretation": "descriptive engine-pooled estimand; not task-population evidence",
                    },
                ]
            )
    return pd.DataFrame(rows)


def endpoint_sensitivity(
    results_dir: Path,
    *,
    reps: int,
    random_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    seed_count = len(FORMAL_BENCHMARK_SEEDS)
    seed_draw = rng.integers(0, seed_count, size=(reps, seed_count))
    rep_index = np.arange(reps)[:, None]
    rows: list[dict[str, object]] = []
    for subset in FORMAL_BENCHMARK_SUBSETS:
        truth, proposed, predecessor = fixed_task_arrays(results_dir, subset)
        engine_draw = rng.integers(0, len(truth), size=(reps, len(truth)))
        counts = engine_count_matrix(engine_draw, len(truth))
        for threshold in RUL_THRESHOLDS:
            eligible = truth <= threshold
            for tolerance in TOLERANCES:
                proposed_sample = sampled_seed_metrics(
                    truth,
                    proposed,
                    counts,
                    "lpr30",
                    rul_threshold=threshold,
                    tolerance=tolerance,
                )
                predecessor_sample = sampled_seed_metrics(
                    truth,
                    predecessor,
                    counts,
                    "lpr30",
                    rul_threshold=threshold,
                    tolerance=tolerance,
                )
                samples = (
                    proposed_sample[seed_draw, rep_index].mean(axis=1)
                    - predecessor_sample[seed_draw, rep_index].mean(axis=1)
                )
                seed_effects = (
                    (proposed[:, eligible] - truth[None, eligible] > tolerance).mean(axis=1)
                    - (predecessor[:, eligible] - truth[None, eligible] > tolerance).mean(axis=1)
                )
                observed = float(seed_effects.mean())
                low, high = np.quantile(samples, [0.025, 0.975])
                p_value = centered_bootstrap_p(samples, observed)
                rows.append(
                    {
                        "evidence_level": "RETROSPECTIVE ENDPOINT SENSITIVITY",
                        "subset": subset,
                        "rul_threshold_tau": threshold,
                        "late_tolerance_epsilon": tolerance,
                        "mean_difference_ocm_minus_rast": observed,
                        "nominal_crossed_ci95_low": float(low),
                        "nominal_crossed_ci95_high": float(high),
                        "centered_bootstrap_p": p_value,
                        "bootstrap_p_mcse": math.sqrt(p_value * (1.0 - p_value) / (reps + 1)),
                        "exact_seed_signflip_p": exact_seed_signflip_p(seed_effects),
                        "eligible_engine_count": int(eligible.sum()),
                        "bootstrap_repetitions": reps,
                    }
                )
    frame = pd.DataFrame(rows)
    frame["holm_adjusted_p_all80"] = holm_adjust(frame["centered_bootstrap_p"].to_numpy(float))
    frame["sensitivity_decision"] = np.where(
        frame["holm_adjusted_p_all80"] < 0.05,
        np.where(
            frame["mean_difference_ocm_minus_rast"] < 0.0,
            "adjusted evidence: lower OCM-Asym",
            "adjusted evidence: lower RAST-GRU",
        ),
        "no adjusted evidence",
    )
    return frame


def late_event_counts(results_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    points = ((PROPOSED, "OCM-Asym"), (DIRECT_PREDECESSOR, "RAST-GRU"))
    for subset in FORMAL_BENCHMARK_SUBSETS:
        truth, proposed, predecessor = fixed_task_arrays(results_dir, subset)
        arrays = {PROPOSED: proposed, DIRECT_PREDECESSOR: predecessor}
        for model, display in points:
            for seed_index, seed in enumerate(FORMAL_BENCHMARK_SEEDS):
                error = arrays[model][seed_index] - truth
                for threshold in RUL_THRESHOLDS:
                    eligible = truth <= threshold
                    for tolerance in TOLERANCES:
                        late_error = error[eligible]
                        events = late_error > tolerance
                        exceedance = late_error[events] - tolerance
                        rows.append(
                            {
                                "subset": subset,
                                "model": model,
                                "display_name": display,
                                "seed": seed,
                                "rul_threshold_tau": threshold,
                                "late_tolerance_epsilon": tolerance,
                                "eligible_engine_count": int(eligible.sum()),
                                "late_event_count": int(events.sum()),
                                "late_event_rate": float(events.mean()),
                                "conditional_mean_exceedance": (
                                    float(exceedance.mean()) if len(exceedance) else np.nan
                                ),
                                "conditional_q95_exceedance": (
                                    float(np.quantile(exceedance, 0.95)) if len(exceedance) else np.nan
                                ),
                                "zero_inflated_mean_exceedance": float(
                                    np.maximum(late_error - tolerance, 0.0).mean()
                                ),
                                "conditional_severity_defined": bool(len(exceedance)),
                            }
                        )
    return pd.DataFrame(rows)


def design_matched_small_cluster_simulation(
    *,
    simulations: int,
    bootstrap_reps: int,
    random_seed: int,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_seed)
    rows: list[dict[str, object]] = []
    settings = ((100, 0.5), (100, 1.0), (248, 0.5), (248, 1.0))
    for engine_count, seed_sd in settings:
        rejection = {"crossed_centered": 0, "exact_seed_signflip": 0}
        coverage = {"crossed_percentile": 0, "product_weighted_percentile": 0}
        for _ in range(simulations):
            seed_effect = rng.normal(0.0, seed_sd, size=(5, 1))
            engine_effect = rng.normal(0.0, 1.0, size=(1, engine_count))
            noise = rng.normal(0.0, 1.0, size=(5, engine_count))
            values = seed_effect + engine_effect + noise
            observed_seed = values.mean(axis=1)
            observed = float(observed_seed.mean())

            seed_draw = rng.integers(0, 5, size=(bootstrap_reps, 5))
            engine_draw = rng.integers(
                0,
                engine_count,
                size=(bootstrap_reps, engine_count),
            )
            counts = engine_count_matrix(engine_draw, engine_count)
            engine_sampled = values @ counts / float(engine_count)
            sampled = engine_sampled[seed_draw, np.arange(bootstrap_reps)[:, None]].mean(axis=1)
            low, high = np.quantile(sampled, [0.025, 0.975])
            coverage["crossed_percentile"] += int(low <= 0.0 <= high)
            rejection["crossed_centered"] += int(centered_bootstrap_p(sampled, observed) < 0.05)
            rejection["exact_seed_signflip"] += int(exact_seed_signflip_p(observed_seed) < 0.05)

            seed_weights = rng.exponential(size=(bootstrap_reps, 5))
            seed_weights /= seed_weights.sum(axis=1, keepdims=True)
            engine_weights = rng.exponential(size=(engine_count, bootstrap_reps))
            engine_weights /= engine_weights.sum(axis=0, keepdims=True)
            weighted_by_seed = values @ engine_weights
            weighted = np.sum(weighted_by_seed.T * seed_weights, axis=1)
            weighted_low, weighted_high = np.quantile(weighted, [0.025, 0.975])
            coverage["product_weighted_percentile"] += int(weighted_low <= 0.0 <= weighted_high)

        for method, count in coverage.items():
            rate = count / simulations
            rows.append(
                {
                    "engine_count": engine_count,
                    "seed_sd": seed_sd,
                    "method": method,
                    "estimand": "null-coverage",
                    "estimate": rate,
                    "monte_carlo_se": math.sqrt(rate * (1.0 - rate) / simulations),
                    "simulations": simulations,
                    "bootstrap_repetitions_per_simulation": bootstrap_reps,
                    "interpretation": "design-matched illustrative calibration, not a proof of validity",
                }
            )
        for method, count in rejection.items():
            rate = count / simulations
            rows.append(
                {
                    "engine_count": engine_count,
                    "seed_sd": seed_sd,
                    "method": method,
                    "estimand": "null-rejection-rate",
                    "estimate": rate,
                    "monte_carlo_se": math.sqrt(rate * (1.0 - rate) / simulations),
                    "simulations": simulations,
                    "bootstrap_repetitions_per_simulation": bootstrap_reps,
                    "interpretation": "design-matched illustrative calibration, not a proof of validity",
                }
            )
    return pd.DataFrame(rows)


def max_t_family_calibration(
    results_dir: Path,
    *,
    simulations: int,
    bootstrap_reps: int,
    random_seed: int,
) -> pd.DataFrame:
    """Calibrate the 12-contrast envelope under an empirical swap-null DGP.

    Each simulation preserves the observed midpoint and prediction-difference
    magnitudes. One Rademacher sign is drawn per training seed and one per
    matched engine within each fixed task; their product swaps the two
    configurations around their midpoint. The same signs are used for all
    three metrics within a task, and the seed signs are shared across tasks.
    Bootstrap rows then share one seed draw across all 12 contrasts and one
    task-specific matched-engine draw across the three metrics. This preserves
    the declared crossed dependence without treating the four tasks as a
    sampled task population.
    """

    task_arrays = {
        subset: fixed_task_arrays(results_dir, subset)
        for subset in FORMAL_BENCHMARK_SUBSETS
    }
    rng = np.random.default_rng(random_seed)
    family_cover = 0
    pointwise_cover = np.zeros(len(FORMAL_BENCHMARK_SUBSETS) * len(PRIMARY_METRICS), dtype=int)
    holm_any = 0
    critical_values: list[float] = []
    finite_floor = 1.0 / (bootstrap_reps + 1)

    for _ in range(simulations):
        seed_sign = rng.choice((-1.0, 1.0), size=(len(FORMAL_BENCHMARK_SEEDS), 1))
        null_tasks: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        for subset, (truth, proposed, predecessor) in task_arrays.items():
            engine_sign = rng.choice((-1.0, 1.0), size=(1, len(truth)))
            swap = seed_sign * engine_sign
            midpoint = 0.5 * (proposed + predecessor)
            half_difference = 0.5 * (proposed - predecessor)
            null_tasks[subset] = (
                truth,
                midpoint + swap * half_difference,
                midpoint - swap * half_difference,
            )

        seed_draw = rng.integers(
            0,
            len(FORMAL_BENCHMARK_SEEDS),
            size=(bootstrap_reps, len(FORMAL_BENCHMARK_SEEDS)),
        )
        rep_index = np.arange(bootstrap_reps)[:, None]
        observed_columns: list[float] = []
        sample_columns: list[np.ndarray] = []
        for subset in FORMAL_BENCHMARK_SUBSETS:
            truth, proposed, predecessor = null_tasks[subset]
            engine_draw = rng.integers(
                0,
                len(truth),
                size=(bootstrap_reps, len(truth)),
            )
            counts = engine_count_matrix(engine_draw, len(truth))
            unit_counts = np.ones((len(truth), 1), dtype=np.int16)
            for metric in PRIMARY_METRICS:
                left_sample = sampled_seed_metrics(truth, proposed, counts, metric)
                right_sample = sampled_seed_metrics(truth, predecessor, counts, metric)
                samples = (
                    left_sample[seed_draw, rep_index].mean(axis=1)
                    - right_sample[seed_draw, rep_index].mean(axis=1)
                )
                sample_columns.append(samples)
                seed_effects = np.asarray(
                    [
                        sampled_seed_metrics(
                            truth,
                            proposed[index : index + 1],
                            unit_counts,
                            metric,
                        )[0, 0]
                        - sampled_seed_metrics(
                            truth,
                            predecessor[index : index + 1],
                            unit_counts,
                            metric,
                        )[0, 0]
                        for index in range(len(FORMAL_BENCHMARK_SEEDS))
                    ],
                    dtype=float,
                )
                observed_columns.append(float(seed_effects.mean()))

        observed = np.asarray(observed_columns, dtype=float)
        samples_matrix = np.column_stack(sample_columns)
        envelope = diagnostic_max_t_envelope(samples_matrix, observed)
        low = np.asarray(envelope["interval_low"], dtype=float)
        high = np.asarray(envelope["interval_high"], dtype=float)
        covered = (low <= 0.0) & (0.0 <= high)
        family_cover += int(covered.all())
        pointwise_cover += covered.astype(int)
        critical_values.append(float(envelope["critical_value"]))
        p_values = np.asarray(
            [
                centered_bootstrap_p(samples_matrix[:, index], observed[index])
                for index in range(len(observed))
            ],
            dtype=float,
        )
        holm_any += int(np.any(holm_adjust(p_values) < 0.05))

    rows: list[dict[str, object]] = []

    def add_rate(method: str, estimand: str, count: int, denominator: int) -> None:
        rate = count / denominator
        exact_low = (
            0.0
            if count == 0
            else float(beta_distribution.ppf(0.025, count, denominator - count + 1))
        )
        exact_high = (
            1.0
            if count == denominator
            else float(beta_distribution.ppf(0.975, count + 1, denominator - count))
        )
        rows.append(
            {
                "dgp": "empirical midpoint swap-null with crossed seed and engine Rademacher signs",
                "method": method,
                "estimand": estimand,
                "estimate": rate,
                "monte_carlo_se": math.sqrt(rate * (1.0 - rate) / denominator),
                "event_count": count,
                "exact_binomial_95_low": exact_low,
                "exact_binomial_95_high": exact_high,
                "simulations": simulations,
                "bootstrap_repetitions_per_simulation": bootstrap_reps,
                "bootstrap_finite_p_floor": finite_floor,
                "rng_seed": random_seed,
                "task_count": len(FORMAL_BENCHMARK_SUBSETS),
                "contrast_count": len(pointwise_cover),
                "training_seed_levels": len(FORMAL_BENCHMARK_SEEDS),
                "task_resampling": "none; FD001-FD004 are fixed",
                "interpretation": "calibration diagnostic, not confirmatory validation",
            }
        )

    add_rate(
        "studentized max-t diagnostic envelope",
        "family-wise null coverage over all 12 contrasts",
        family_cover,
        simulations,
    )
    add_rate(
        "centered bootstrap plus Holm12",
        "family-wise null rejection probability",
        holm_any,
        simulations,
    )
    for index, (subset, metric) in enumerate(
        itertools.product(FORMAL_BENCHMARK_SUBSETS, PRIMARY_METRICS)
    ):
        add_rate(
            "studentized max-t diagnostic envelope",
            f"pointwise null coverage: {subset} {metric}",
            int(pointwise_cover[index]),
            simulations,
        )
    rows.append(
        {
            "dgp": "empirical midpoint swap-null with crossed seed and engine Rademacher signs",
            "method": "studentized max-t diagnostic envelope",
            "estimand": "mean bootstrap critical value",
            "estimate": float(np.mean(critical_values)),
            "monte_carlo_se": float(np.std(critical_values, ddof=1) / math.sqrt(simulations)),
            "event_count": np.nan,
            "exact_binomial_95_low": np.nan,
            "exact_binomial_95_high": np.nan,
            "simulations": simulations,
            "bootstrap_repetitions_per_simulation": bootstrap_reps,
            "bootstrap_finite_p_floor": finite_floor,
            "rng_seed": random_seed,
            "task_count": len(FORMAL_BENCHMARK_SUBSETS),
            "contrast_count": len(pointwise_cover),
            "training_seed_levels": len(FORMAL_BENCHMARK_SEEDS),
            "task_resampling": "none; FD001-FD004 are fixed",
            "interpretation": "calibration diagnostic, not confirmatory validation",
        }
    )
    return pd.DataFrame(rows)


def feature_selection_stability(results_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for subset in FORMAL_BENCHMARK_SUBSETS:
        for seed in FORMAL_BENCHMARK_SEEDS:
            path = (
                results_dir
                / f"paper_main_v3_seed{seed}"
                / subset
                / PROPOSED
                / "feature_scores.csv"
            )
            scores = pd.read_csv(path)
            selected = set(scores.head(14)["feature"].astype(str))
            for rank, row in scores.iterrows():
                rows.append(
                    {
                        "subset": subset,
                        "seed": seed,
                        "sensor": str(row["feature"]),
                        "rank": int(rank + 1),
                        "selected_top14": str(row["feature"]) in selected,
                        "hybrid_score": float(row["hybrid_score"]),
                    }
                )
    detail = pd.DataFrame(rows)
    summary = (
        detail.groupby("sensor", as_index=False)
        .agg(
            selected_cells=("selected_top14", "sum"),
            total_cells=("selected_top14", "size"),
            mean_rank=("rank", "mean"),
            min_rank=("rank", "min"),
            max_rank=("rank", "max"),
        )
        .sort_values(["selected_cells", "mean_rank", "sensor"], ascending=[False, True, True])
    )
    summary["selection_frequency"] = summary["selected_cells"] / summary["total_cells"]
    return summary


def multiplicity_registry() -> pd.DataFrame:
    frame = pd.DataFrame(
        [
            {
                "family_id": "E-FIXED-12",
                "evidence_level": "main result-informed fixed-task estimation",
                "scope": "4 tasks x 3 metrics = 12 OCM-Asym vs RAST-GRU contrasts",
                "adjustment": "Holm12 retained only as a diagnostic multiplicity summary",
                "interval": "crossed diagnostic envelope plus design-specific max-t diagnostic envelope",
                "predefined_before_test_summary": "no externally verifiable timestamp",
                "primary_unit": "matched official test engine within each fixed task; five training seeds",
                "decision_role": "estimation only; cannot support confirmatory claims",
            },
            {
                "family_id": "S-ENDPOINT-80",
                "evidence_level": "sensitivity analysis",
                "scope": "4 tau x 5 epsilon x 4 tasks = 80 endpoint contrasts",
                "adjustment": "one Holm family across all 80 tests",
                "interval": "uncalibrated crossed-percentile diagnostic envelope",
                "predefined_before_test_summary": "no",
                "primary_unit": "matched official test engine within each fixed task; five training seeds",
                "decision_role": "sensitivity only",
            },
            {
                "family_id": "S-PERTURB-54",
                "evidence_level": "sensitivity analysis",
                "scope": "2 operating points x 9 perturbation families x 3 metrics = 54 contrasts",
                "adjustment": "one Holm family across all 54 tests",
                "interval": "crossed-factor percentile diagnostic envelope",
                "predefined_before_test_summary": "no",
                "primary_unit": "matched FD004 engine crossed with training and perturbation seed",
                "decision_role": "normalized-space algorithmic sensitivity only",
            },
            {
                "family_id": "S-NORM-9",
                "evidence_level": "matched sensitivity control",
                "scope": "K-means minus 3 normalization controls x 3 metrics = 9 contrasts",
                "adjustment": "one Holm family across all 9 tests",
                "interval": "uncalibrated crossed-percentile diagnostic envelope",
                "predefined_before_test_summary": "no",
                "primary_unit": "matched FD004 engine within five training seeds",
                "decision_role": "construction sensitivity only",
            },
            {
                "family_id": "S-FD002-20",
                "evidence_level": "exploratory protocol audit",
                "scope": "five matched operating-point comparisons x four metrics = 20 contrasts",
                "adjustment": "one Holm family across all 20 tests",
                "interval": "uncalibrated crossed-percentile diagnostic envelope",
                "predefined_before_test_summary": "no",
                "primary_unit": "matched FD002 engine within five training seeds",
                "decision_role": "exploratory transfer sensitivity only",
            },
            {
                "family_id": "D-DUALMIXER-A12",
                "evidence_level": "descriptive comparator",
                "scope": "official-code-derived Dual-Mixer comparator family A, 12 contrasts",
                "adjustment": "Holm12 in source artifact",
                "interval": "nominal paired bootstrap interval",
                "predefined_before_test_summary": "no",
                "primary_unit": "matched engine within fixed task",
                "decision_role": "descriptive external-code context",
            },
            {
                "family_id": "D-DUALMIXER-B12",
                "evidence_level": "descriptive comparator",
                "scope": "official-code-derived Dual-Mixer comparator family B, 12 contrasts",
                "adjustment": "Holm12 in source artifact",
                "interval": "nominal paired bootstrap interval",
                "predefined_before_test_summary": "no",
                "primary_unit": "matched engine within fixed task",
                "decision_role": "descriptive external-code context",
            },
            {
                "family_id": "D-ENDPOINT-DESIGN",
                "evidence_level": "design diagnostic",
                "scope": "single, multi-endpoint, and critical-stratified checkpoint designs",
                "adjustment": "none; full effects and selected epochs reported",
                "interval": "nominal paired intervals",
                "predefined_before_test_summary": "no",
                "primary_unit": "matched engine and training seed within FD002/FD004",
                "decision_role": "configuration sensitivity; no decision claim",
            },
            {
                "family_id": "D-CHECKPOINT",
                "evidence_level": "design diagnostic",
                "scope": "checkpoint-rule sensitivity contrasts",
                "adjustment": "none",
                "interval": "nominal paired intervals",
                "predefined_before_test_summary": "no",
                "primary_unit": "training seed and matched engine",
                "decision_role": "configuration sensitivity; no decision claim",
            },
            {
                "family_id": "D-ABLATION",
                "evidence_level": "descriptive ablation",
                "scope": "13 FD004 removal/objective variants across five seeds",
                "adjustment": "none",
                "interval": "seed SD and descriptive paired summaries",
                "predefined_before_test_summary": "partly",
                "primary_unit": "training seed",
                "decision_role": "module-specific diagnostics; no causal attribution",
            },
            {
                "family_id": "D-AUGDIST",
                "evidence_level": "descriptive sensitivity",
                "scope": "augmentation-distribution operating points",
                "adjustment": "none",
                "interval": "descriptive paired intervals/probabilities",
                "predefined_before_test_summary": "no",
                "primary_unit": "training seed",
                "decision_role": "training-distribution sensitivity only",
            },
            {
                "family_id": "D-SEEDGRID-3X3",
                "evidence_level": "descriptive sensitivity",
                "scope": "3 split seeds x 3 training seeds on FD004",
                "adjustment": "none",
                "interval": "descriptive variance decomposition",
                "predefined_before_test_summary": "no",
                "primary_unit": "split-seed x training-seed cell",
                "decision_role": "randomness decomposition only",
            },
            {
                "family_id": "D-NCMAPSS-ROT",
                "evidence_level": "exploratory computational-proxy audit",
                "scope": "N-CMAPSS DS02 development-unit rotations and K sensitivity",
                "adjustment": "none",
                "interval": "descriptive unit-level summaries",
                "predefined_before_test_summary": "no",
                "primary_unit": "official or rotated development unit",
                "decision_role": "failure analysis; not external validation",
            },
            {
                "family_id": "D-STREAM10-FIXED",
                "evidence_level": "retrospective fixed-design sensitivity",
                "scope": "4 fixed tasks x 2 configurations x 10 separately seeded training streams = 80 cells",
                "adjustment": "none",
                "interval": "finite-design stream-level effects and sign counts",
                "predefined_before_test_summary": "no",
                "primary_unit": "training stream within one fixed engine split and task",
                "decision_role": "finite-design stability description; no population inference",
            },
            {
                "family_id": "D-CROSSED-3X3",
                "evidence_level": "retrospective crossed finite-design audit",
                "scope": "4 fixed tasks x 2 configurations x 3 engine splits x 3 training streams = 72 cells",
                "adjustment": "none",
                "interval": "cell effects, sign counts, finite-cell additive-residual RMS, and level-subset sensitivity",
                "predefined_before_test_summary": "no",
                "primary_unit": "engine-split x training-stream cell within fixed task",
                "decision_role": "selected-grid arithmetic only; no population variance components",
            },
            {
                "family_id": "D-CMAPSS-PREP5",
                "evidence_level": "retrospective preprocessing sensitivity",
                "scope": "FD004 x 2 configurations x 3 composite seeds x 5 one-factor settings = 30 cells",
                "adjustment": "none",
                "interval": "finite-design stream effects and reference-relative changes",
                "predefined_before_test_summary": "no",
                "primary_unit": "composite-seed repetition within fixed FD004",
                "decision_role": "window and sensor-budget dependence; not hyperparameter optimization",
            },
            {
                "family_id": "D-PREF-REFIT-3X3",
                "evidence_level": "retrospective selection-operation audit",
                "scope": "9 preference candidates x 3 engine splits x 3 training streams plus matched references",
                "adjustment": "none; selection counts are conditional on the fixed grid",
                "interval": "none",
                "predefined_before_test_summary": "no",
                "primary_unit": "split-stream cell; test engines are reused across selected cells",
                "decision_role": "selection-instability diagnostic; not selection-adjusted performance evidence",
            },
            {
                "family_id": "D-SHARED-FACT-8",
                "evidence_level": "retrospective limited factorial",
                "scope": "2 backbone x 2 shared-risk x 2 shared-consistency cells on FD004 across five streams",
                "adjustment": "none",
                "interval": "finite-design stream-level factorial effects",
                "predefined_before_test_summary": "no",
                "primary_unit": "training stream within the fixed FD004 split",
                "decision_role": "shared-objective factor description; OCM-specific components remain bundled",
            },
            {
                "family_id": "D-LOFO-4",
                "evidence_level": "retrospective augmentation-overlap audit",
                "scope": "4 held-out perturbation families x 2 configurations x 5 training streams",
                "adjustment": "none",
                "interval": "absolute corrupted AUC, clean-corrected degradation AUC, and level effects",
                "predefined_before_test_summary": "no",
                "primary_unit": "training stream; five perturbation seeds are averaged within scenario",
                "decision_role": "algorithmic sensitivity to declared held-out families; not physical-fault robustness",
            },
            {
                "family_id": "D-NCMAPSS-SW9",
                "evidence_level": "retrospective computational-proxy sensitivity",
                "scope": "3 sampling intervals x 3 sampled-record windows x 5 streams on N-CMAPSS DS02",
                "adjustment": "none",
                "interval": "finite-design cell summaries",
                "predefined_before_test_summary": "no",
                "primary_unit": "training stream; the same three official test units are reused",
                "decision_role": "joint representation dependence only; density and horizon are confounded",
            },
            {
                "family_id": "D-NCMAPSS-H3",
                "evidence_level": "retrospective computational-proxy control",
                "scope": "3 sampling intervals with source-record span fixed at 6001 across five streams",
                "adjustment": "none",
                "interval": "finite-design cell summaries",
                "predefined_before_test_summary": "no",
                "primary_unit": "training stream; the same three official test units are reused",
                "decision_role": "sampling-density sensitivity at one fixed source-record span; not external validation",
            },
            {
                "family_id": "D-LATE-SUPPORT",
                "evidence_level": "descriptive diagnostic",
                "scope": "conditional positive-tail severity cells",
                "adjustment": "none; undefined when no positive event occurs",
                "interval": "none",
                "predefined_before_test_summary": "no",
                "primary_unit": "eligible engine within task and seed",
                "decision_role": "event-count context, not hypothesis testing",
            },
        ]
    )
    navigation = {
        "E-FIXED-12": (
            "paper_outputs/release_v1/primary_fixed_task_estimates.csv",
            "python scripts/analyze_fixed_benchmark_evidence.py",
        ),
        "S-ENDPOINT-80": (
            "paper_outputs/release_v1/endpoint_tolerance_sensitivity.csv",
            "python scripts/analyze_fixed_benchmark_evidence.py",
        ),
        "S-PERTURB-54": (
            "paper_outputs/release_v1/normalized_perturbation_contrasts.csv",
            "python scripts/analyze_normalized_perturbation_evidence.py",
        ),
        "D-NCMAPSS-ROT": (
            "paper_outputs/advanced_evidence/ncmapss_dev_rotation_summary.csv",
            "python scripts/analyze_ncmapss_dev_rotation.py",
        ),
        "D-STREAM10-FIXED": (
            "paper_outputs/round12_new_evidence/independent_stream_effects.csv",
            "python scripts/analyze_round12_new_experiments.py --families independent",
        ),
        "D-CROSSED-3X3": (
            "paper_outputs/round12_new_evidence/crossed_split_stream_summary.csv",
            "python scripts/analyze_round12_new_experiments.py --families crossed",
        ),
        "D-CMAPSS-PREP5": (
            "paper_outputs/round12_new_evidence/cmapss_preprocessing_paired_summary.csv",
            "python scripts/analyze_round12_new_experiments.py --families cmapss_preprocessing",
        ),
        "D-PREF-REFIT-3X3": (
            "paper_outputs/round12_new_evidence/preference_cell_summary.csv",
            "python scripts/analyze_round12_new_experiments.py --families preference",
        ),
        "D-SHARED-FACT-8": (
            "paper_outputs/round12_new_evidence/factorial_effect_summary.csv",
            "python scripts/analyze_round12_new_experiments.py --families factorial",
        ),
        "D-LOFO-4": (
            "paper_outputs/round12_new_evidence/lofo_family_effect_summary.csv",
            "python scripts/analyze_round12_new_experiments.py --families lofo",
        ),
        "D-NCMAPSS-SW9": (
            "paper_outputs/round12_new_evidence/ncmapss_sampling_window_summary.csv",
            "python scripts/analyze_round12_new_experiments.py --families ncmapss",
        ),
        "D-NCMAPSS-H3": (
            "paper_outputs/round12_new_evidence/ncmapss_horizon_matched_summary.csv",
            "python scripts/analyze_round12_new_experiments.py --families ncmapss_horizon",
        ),
    }
    frame["formation_time"] = "after result review; local chronology only"
    frame["result_informed_status"] = frame[
        "predefined_before_test_summary"
    ].map(
        lambda value: (
            "partly result-informed"
            if value == "partly"
            else "result-informed"
        )
    )
    frame["source_artifact"] = frame["family_id"].map(
        lambda family: navigation.get(
            family,
            (
                "paper_outputs/analysis_working/multiplicity_registry.csv",
                "python scripts/analyze_fixed_benchmark_evidence.py",
            ),
        )[0]
    )
    frame["generator_command"] = frame["family_id"].map(
        lambda family: navigation.get(
            family,
            (
                "paper_outputs/analysis_working/multiplicity_registry.csv",
                "python scripts/analyze_fixed_benchmark_evidence.py",
            ),
        )[1]
    )
    return frame


def plot_endpoint_sensitivity(frame: pd.DataFrame, output: Path) -> None:
    part = frame[np.isclose(frame["rul_threshold_tau"], 30.0)].copy()
    fig, axes = plt.subplots(2, 2, figsize=(8.4, 6.2), sharex=True)
    for ax, subset in zip(axes.ravel(), FORMAL_BENCHMARK_SUBSETS):
        cell = part[part["subset"] == subset].sort_values("late_tolerance_epsilon")
        x = cell["late_tolerance_epsilon"].to_numpy(float)
        effect = cell["mean_difference_ocm_minus_rast"].to_numpy(float)
        low = cell["nominal_crossed_ci95_low"].to_numpy(float)
        high = cell["nominal_crossed_ci95_high"].to_numpy(float)
        adjusted = cell["holm_adjusted_p_all80"].to_numpy(float) < 0.05
        ax.axhline(0.0, color="#333333", linestyle="--", linewidth=1.0)
        ax.fill_between(x, low, high, color="#9aa0a6", alpha=0.22)
        ax.plot(x, effect, color="#365f8d", marker="o", linewidth=1.7)
        if adjusted.any():
            ax.scatter(x[adjusted], effect[adjusted], color="#b33a3a", zorder=3)
        ax.set_title(subset, loc="left", fontsize=10.5)
        ax.set_ylabel("OCM-Asym minus RAST-GRU")
        ax.grid(alpha=0.25, linestyle=":")
    for ax in axes[-1]:
        ax.set_xlabel(r"Late tolerance $\epsilon$ (cycles), $\tau=30$")
    fig.suptitle("Late-overprediction endpoint sensitivity", fontsize=11.5)
    fig.text(
        0.5,
        0.01,
        "Bands are diagnostic crossed envelopes; highlighted markers follow the declared joint 80-cell diagnostic marker rule.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.035, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def plot_primary_simultaneous_forest(
    frame: pd.DataFrame,
    seed_effects: pd.DataFrame,
    output: Path,
) -> None:
    labels = {
        "rmse": "RMSE difference (cycles)",
        "nasa_per_engine": "NASA/engine difference",
        "lpr30": "LPR@30 difference",
    }
    seed_colors = ["#0072B2", "#E69F00", "#009E73", "#CC79A7", "#56B4E9"]
    seed_markers = ["o", "s", "^", "D", "P"]
    fig, axes = plt.subplots(1, 3, figsize=(9.5, 4.0), sharey=True)
    for ax, metric in zip(axes, PRIMARY_METRICS):
        part = frame[frame["metric"] == metric].copy()
        part["subset"] = pd.Categorical(
            part["subset"],
            categories=list(reversed(FORMAL_BENCHMARK_SUBSETS)),
            ordered=True,
        )
        part = part.sort_values("subset")
        y = np.arange(len(part))
        effect = part["mean_difference"].to_numpy(float)
        ax.axvline(0.0, color="#222222", linewidth=1.0, linestyle="--")
        for index in range(len(part)):
            subset = str(part.iloc[index]["subset"])
            seed_part = seed_effects[
                (seed_effects["subset"] == subset)
                & (seed_effects["metric"] == metric)
            ].sort_values("composite_seed")
            offsets = np.linspace(-0.16, 0.16, len(seed_part))
            values = seed_part["ocm_asym_minus_rast"].to_numpy(float)
            for seed_index, ((_, seed_row), offset) in enumerate(
                zip(seed_part.iterrows(), offsets)
            ):
                ax.scatter(
                    float(seed_row["ocm_asym_minus_rast"]),
                    y[index] + offset,
                    color=seed_colors[seed_index],
                    marker=seed_markers[seed_index],
                    s=25,
                    edgecolor="white",
                    linewidth=0.35,
                    zorder=3,
                    label=(
                        f"seed {int(seed_row['composite_seed'])}"
                        if metric == PRIMARY_METRICS[0] and index == 0
                        else None
                    ),
                )
            ax.scatter(
                effect[index],
                y[index],
                color="#111111",
                marker="|",
                s=85,
                linewidth=1.8,
                zorder=4,
            )
        ax.set_xlabel(labels[metric])
        ax.grid(axis="x", linestyle=":", alpha=0.35)
        ax.set_title(metric.replace("_", " ").upper(), loc="left", fontsize=10.5)
        if ax is axes[0]:
            task_labels = [
                "FD004\n(development-informed)"
                if str(subset) == "FD004"
                else str(subset)
                for subset in part["subset"]
            ]
            ax.set_yticks(y, task_labels)
        else:
            ax.tick_params(axis="y", labelleft=False)
    fig.suptitle("Seed-level paired effects on four fixed benchmark tasks", fontsize=11.5)
    fig.text(
        0.5,
        0.005,
        "Marker shape and color identify five coupled composite-seed levels; black tick: fixed-task mean. No uncertainty envelope is drawn; sign counts and resampling diagnostics are reported separately.",
        ha="center",
        fontsize=8.5,
    )
    handles, legend_labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(
            handles,
            legend_labels,
            loc="lower center",
            ncol=5,
            frameon=False,
            bbox_to_anchor=(0.5, 0.035),
            fontsize=7.8,
        )
    fig.tight_layout(rect=(0, 0.09, 1, 0.97))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def write_latex_tables(
    primary: pd.DataFrame,
    seed_effects: pd.DataFrame,
    transfer_summary: pd.DataFrame,
    counts: pd.DataFrame,
    calibration: pd.DataFrame,
    max_t_calibration: pd.DataFrame,
    stability: pd.DataFrame,
    registry: pd.DataFrame,
    output_dir: Path,
) -> None:
    support_lookup: dict[str, str] = {}
    tau30 = counts[
        np.isclose(counts["rul_threshold_tau"], 30.0)
        & np.isclose(counts["late_tolerance_epsilon"], 0.0)
    ]
    for subset, group in tau30.groupby("subset", sort=False):
        pieces: list[str] = []
        eligible = int(group["eligible_engine_count"].iloc[0])
        for display_name, short_name in (("OCM-Asym", "O"), ("RAST-GRU", "R")):
            events = group[group["display_name"] == display_name]["late_event_count"]
            pieces.append(f"{short_name} {int(events.min())}--{int(events.max())}")
        support_lookup[str(subset)] = f"$N_{{30}}={eligible}$; " + "/".join(pieces)

    rows = []
    for row in primary.to_dict("records"):
        support = support_lookup[row["subset"]] if row["metric"] == "lpr30" else "--"
        rows.append(
            f"{row['subset']} & {row['metric'].replace('_', ' ')} & "
            f"{row['mean_difference']:.3f} & "
            f"[{row['nominal_crossed_ci95_low']:.3f}, {row['nominal_crossed_ci95_high']:.3f}] & "
            f"[{row['diagnostic_max_t_envelope_low']:.3f}, {row['diagnostic_max_t_envelope_high']:.3f}] & "
            f"{row['holm_adjusted_p_all12']:.3f} & {row['exact_seed_signflip_p']:.3f} & "
            f"{support} \\\\"
        )
    text = "\n".join(
        [
            r"\begin{table*}[!t]",
            r"\centering",
            r"\scriptsize",
            r"\caption{Main result-informed fixed-benchmark OCM-Asym minus RAST-GRU estimates (family E-FIXED-12). The crossed bootstrap resamples five training-seed levels and matched engines independently within each fixed task. The pointwise interval, Holm$_{12}$ value, max-$t$ envelope, and exact sign-flip value are diagnostics rather than calibrated confirmatory procedures. The exact sign-flip calculation enumerates all $2^5$ signs and cannot attain a two-sided value below 0.0625.}",
            r"\label{tab:primary-fixed-task-estimates}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{llrrrrrl}",
            r"\toprule",
            r"Task & Metric & Mean $\Delta$ & Diagnostic interval & Diagnostic max-$t$ envelope & Holm$_{12}$ diagnostic & Exact sign-flip diagnostic & LPR support \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\par\footnotesize LPR support reports the number of eligible engines and the five-seed event-count ranges for OCM-Asym (O) and RAST-GRU (R).",
            r"\end{table*}",
            "",
        ]
    )
    (output_dir / "table_primary_fixed_task_estimates.tex").write_text(text, encoding="utf-8")

    count_rows = []
    count_part = counts[
        np.isclose(counts["rul_threshold_tau"], 30.0)
        & counts["late_tolerance_epsilon"].isin([0.0, 5.0, 10.0])
    ]
    grouped_counts = count_part.groupby(
        ["subset", "display_name", "late_tolerance_epsilon"],
        as_index=False,
    ).agg(
        eligible=("eligible_engine_count", "first"),
        event_min=("late_event_count", "min"),
        event_max=("late_event_count", "max"),
        undefined_seeds=("conditional_severity_defined", lambda values: int((~values).sum())),
    )
    for row in grouped_counts.to_dict("records"):
        count_rows.append(
            f"{row['subset']} & {row['display_name']} & {row['late_tolerance_epsilon']:.0f} & "
            f"{int(row['eligible'])} & {int(row['event_min'])}--{int(row['event_max'])} & "
            f"{int(row['undefined_seeds'])}/5 \\\\"
        )
    count_text = "\n".join(
        [
            r"\begin{table}[!t]",
            r"\centering",
            r"\scriptsize",
            r"\caption{Near-failure event support at $\tau=30$. Event ranges are across the five composite seeds. Conditional positive-tail severity is undefined, rather than set to zero, when a seed has no event beyond tolerance $\epsilon$.}",
            r"\label{tab:late-event-support}",
            r"\begin{tabular}{llrrrr}",
            r"\toprule",
            r"Task & Point & $\epsilon$ & Eligible & Event range & Undefined seeds \\",
            r"\midrule",
            *count_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    (output_dir / "table_late_event_support.tex").write_text(count_text, encoding="utf-8")

    calibration_rows = []
    for row in calibration.to_dict("records"):
        target = "Coverage" if row["estimand"] == "null-coverage" else "Type-I"
        calibration_rows.append(
            f"{int(row['engine_count'])} & {row['seed_sd']:.1f} & "
            f"{row['method'].replace('_', ' ')} & {target} & "
            f"{row['estimate']:.3f} & {row['monte_carlo_se']:.3f} \\\\"
        )
    calibration_text = "\n".join(
        [
            r"\begin{table*}[!t]",
            r"\centering",
            r"\scriptsize",
            r"\caption{Illustrative null calibration with five seed clusters. The crossed percentile and centered-bootstrap procedures are anti-conservative in these design-matched simulations, whereas the exact five-seed sign-flip test cannot attain a two-sided $p<0.05$ (minimum $0.0625$). These simulations diagnose small-cluster uncertainty; they do not validate any one procedure universally.}",
            r"\label{tab:small-cluster-calibration}",
            r"\begin{tabular}{rrllrr}",
            r"\toprule",
            r"Engines & Seed SD & Method & Quantity & Estimate & MCSE \\",
            r"\midrule",
            *calibration_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    (output_dir / "table_small_cluster_calibration.tex").write_text(
        calibration_text,
        encoding="utf-8",
    )

    family_rows = []
    for row in max_t_calibration.to_dict("records"):
        method = str(row["method"]).replace("_", r"\_")
        estimand = str(row["estimand"]).replace("_", r"\_")
        if pd.isna(row["event_count"]):
            event_count = "--"
            exact_interval = "--"
        else:
            event_count = str(int(row["event_count"]))
            exact_interval = (
                f"[{row['exact_binomial_95_low']:.3f}, "
                f"{row['exact_binomial_95_high']:.3f}]"
            )
        family_rows.append(
            f"{method} & {estimand} & "
            f"{row['estimate']:.3f} & {row['monte_carlo_se']:.3f} & "
            f"{event_count} & {exact_interval} & "
            f"{int(row['simulations'])} & {int(row['bootstrap_repetitions_per_simulation'])} \\\\"
        )
    family_text = "\n".join(
        [
            r"\begin{table*}[!t]",
            r"\centering",
            r"\scriptsize",
            r"\caption{Empirical 12-dimensional swap-null calibration. The simulation preserves observed prediction midpoints and difference magnitudes, uses one Rademacher sign per training seed and matched engine, shares seed signs across the four fixed tasks, and shares each task's engine signs across RMSE, NASA/engine, and LPR@30. The four tasks are never resampled. Exact binomial intervals quantify Monte Carlo uncertainty for event rates; zero observed failures are not proof of general conservatism.}",
            r"\label{tab:max-t-family-calibration}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{p{0.17\linewidth}p{0.31\linewidth}rrrlrr}",
            r"\toprule",
            r"Method & Calibration quantity & Estimate & MCSE & Events & Exact 95\% binomial interval & Simulations & $B$ \\",
            r"\midrule",
            *family_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table*}",
            "",
        ]
    )
    (output_dir / "table_max_t_family_calibration.tex").write_text(
        family_text,
        encoding="utf-8",
    )

    stable_rows = []
    for row in stability.head(21).to_dict("records"):
        stable_rows.append(
            f"{row['sensor']} & {int(row['selected_cells'])}/{int(row['total_cells'])} & "
            f"{row['selection_frequency']:.2f} & {row['mean_rank']:.1f} & "
            f"{int(row['min_rank'])}--{int(row['max_rank'])} \\\\"
        )
    stable_text = "\n".join(
        [
            r"\begin{table}[!t]",
            r"\centering",
            r"\scriptsize",
            r"\caption{Training-only sensor-ranking stability across 20 task--seed cells. Scores combine normalized variance (0.30), absolute Spearman correlation with capped RUL (0.40), and absolute Spearman correlation with cycle (0.30). Ties are resolved by ascending sensor number; the top 14 implement the fixed sensor-budget constraint.}",
            r"\label{tab:feature-selection-stability}",
            r"\begin{tabular}{lrrrr}",
            r"\toprule",
            r"Sensor & Selected cells & Frequency & Mean rank & Rank range \\",
            r"\midrule",
            *stable_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    (output_dir / "table_feature_selection_stability.tex").write_text(stable_text, encoding="utf-8")

    transfer = transfer_summary[
        transfer_summary["scope"] == "FD001-FD003 fixed-transfer"
    ].copy()
    transfer_rows = [
        f"{row['aggregation']} & {row['metric'].replace('_', ' ')} & "
        f"{row['ocm_minus_rast']:.3f} & {int(row['task_count'])} & "
        f"{int(row['engine_count'])} \\\\"
        for row in transfer.to_dict("records")
    ]
    transfer_text = "\n".join(
        [
            r"\begin{table}[!t]",
            r"\centering",
            r"\small",
            r"\caption{Development-task-excluded FD001--FD003 descriptive estimates. Equal-task and engine-pooled rows define different fixed-sample estimands; neither resamples tasks or supports task-population inference. Seed-wise and task-wise values are supplied in \texttt{development\_excluded\_seed\_effects.csv}.}",
            r"\label{tab:development-excluded-summary}",
            r"\begin{tabular}{llrrr}",
            r"\toprule",
            r"Aggregation & Metric & OCM$-$RAST & Tasks & Engines \\",
            r"\midrule",
            *transfer_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    (output_dir / "table_development_excluded_summary.tex").write_text(
        transfer_text,
        encoding="utf-8",
    )

    sign_rows = []
    for (subset, metric), group in seed_effects.groupby(["subset", "metric"], sort=False):
        values = group["ocm_asym_minus_rast"].to_numpy(float)
        sign_rows.append(
            f"{subset} & {metric.replace('_', ' ')} & {values.mean():.3f} & "
            f"{int(np.sum(values < 0))} / {int(np.sum(values > 0))} / "
            f"{int(np.sum(values == 0))} & {len(values)} \\\\"
        )
    seed_text = "\n".join(
        [
            r"\begin{table}[!t]",
            r"\centering",
            r"\small",
            r"\caption{Five-seed sign patterns for the fixed-task OCM-Asym minus RAST-GRU effects shown in Figure~2. Counts are negative/positive/tied seed-level paired effects and are descriptive diagnostics.}",
            r"\label{tab:primary-seed-sign-patterns}",
            r"\begin{tabular}{llrrr}",
            r"\toprule",
            r"Task & Metric & Mean effect & $- / + / 0$ & Seeds \\",
            r"\midrule",
            *sign_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    (output_dir / "table_primary_seed_sign_patterns.tex").write_text(
        seed_text,
        encoding="utf-8",
    )

    registry_rows = []
    for row in registry.to_dict("records"):
        registry_rows.append(
            f"{row['family_id']} & {row['evidence_level']} & "
            f"{row['scope']} & "
            f"{row['adjustment']} & "
            f"{row['decision_role']} \\\\"
        )
    registry_text = "\n".join(
        [
            r"{\scriptsize",
            r"\begin{longtable}{p{0.13\linewidth}p{0.18\linewidth}p{0.26\linewidth}p{0.31\linewidth}}",
            r"\caption{Release-level multiplicity and evidence registry. No family is assigned a confirmatory role because the five training-seed levels and local-only chronology do not provide calibrated or externally timestamped confirmation. Full scopes, intervals, primary units, and predefinition status are machine-readable in the release registry.}\label{tab:multiplicity-registry}\\",
            r"\toprule",
            r"Family ID & Evidence level & Multiplicity rule & Role \\",
            r"\midrule",
            r"\endfirsthead",
            r"\multicolumn{4}{l}{\scriptsize\itshape Table~\thetable\ continued from the previous page}\\",
            r"\toprule",
            r"Family ID & Evidence level & Multiplicity rule & Role \\",
            r"\midrule",
            r"\endhead",
            r"\midrule",
            r"\multicolumn{4}{r}{\scriptsize Continued on the next page}\\",
            r"\endfoot",
            r"\bottomrule",
            r"\endlastfoot",
            *[
                f"{row['family_id']} & {row['evidence_level']} & "
                f"{row['adjustment']} & {row['decision_role']} \\\\"
                for row in registry.to_dict("records")
            ],
            r"\end{longtable}",
            r"}",
            "",
        ]
    )
    (output_dir / "table_multiplicity_registry.tex").write_text(
        registry_text,
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build fixed-benchmark estimation and calibration evidence.")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "paper_outputs" / "analysis_working"))
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    parser.add_argument("--calibration-simulations", type=int, default=300)
    parser.add_argument("--calibration-bootstrap-reps", type=int, default=499)
    parser.add_argument("--max-t-calibration-simulations", type=int, default=1000)
    parser.add_argument("--max-t-calibration-bootstrap-reps", type=int, default=999)
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)

    primary, samples = joint_primary_inference(
        results_dir,
        reps=args.bootstrap_reps,
        random_seed=2026072603,
    )
    primary.to_csv(output / "primary_fixed_task_estimates.csv", index=False)
    primary_seeds = primary_seed_level_effects(results_dir)
    primary_seeds.to_csv(output / "primary_seed_level_effects.csv", index=False)
    transfer_summary = fixed_transfer_summaries(results_dir)
    transfer_summary.to_csv(
        output / "fixed_transfer_aggregation_summary.csv", index=False
    )
    development_excluded_seed_effects(results_dir).to_csv(
        output / "development_excluded_seed_effects.csv", index=False
    )
    np.savez_compressed(
        output / "primary_fixed_task_bootstrap_draws.npz",
        samples=samples,
        columns=primary[["subset", "metric"]].astype(str).agg("|".join, axis=1).to_numpy(),
    )

    endpoint = endpoint_sensitivity(
        results_dir,
        reps=args.bootstrap_reps,
        random_seed=2026072604,
    )
    endpoint.to_csv(output / "endpoint_tolerance_sensitivity.csv", index=False)

    counts = late_event_counts(results_dir)
    counts.to_csv(output / "late_event_support.csv", index=False)

    calibration = design_matched_small_cluster_simulation(
        simulations=args.calibration_simulations,
        bootstrap_reps=args.calibration_bootstrap_reps,
        random_seed=2026072605,
    )
    calibration.to_csv(output / "small_cluster_calibration.csv", index=False)

    family_calibration = max_t_family_calibration(
        results_dir,
        simulations=args.max_t_calibration_simulations,
        bootstrap_reps=args.max_t_calibration_bootstrap_reps,
        random_seed=2026072701,
    )
    family_calibration.to_csv(output / "max_t_family_calibration.csv", index=False)

    stability = feature_selection_stability(results_dir)
    stability.to_csv(output / "feature_selection_stability.csv", index=False)
    registry = multiplicity_registry()
    registry.to_csv(output / "multiplicity_registry.csv", index=False)

    plot_endpoint_sensitivity(
        endpoint,
        output / "figures" / "fig_endpoint_tolerance_sensitivity",
    )
    plot_primary_simultaneous_forest(
        primary,
        primary_seeds,
        output / "figures" / "fig_primary_fixed_task_diagnostic_envelope",
    )
    write_latex_tables(
        primary,
        primary_seeds,
        transfer_summary,
        counts,
        calibration,
        family_calibration,
        stability,
        registry,
        output,
    )

    run_metadata = {
        "analysis": "fixed-benchmark evidence release",
        "bootstrap_algorithm": {
            "factors": ["composite seed", "matched official test engine"],
            "sampling": "independent empirical resampling with replacement at each factor",
            "nonlinear_metric_order": (
                "for each bootstrap row and sampled seed, aggregate sampled matched-engine losses "
                "inside the metric first (including square-mean-root for RMSE); then average the "
                "seed-specific metric values; finally subtract OCM and RAST aggregates"
            ),
            "repetitions": args.bootstrap_reps,
            "centered_p": "(1 + count(|Delta_b - Delta_hat| >= |Delta_hat|)) / (B + 1)",
            "finite_p_floor": 1.0 / (args.bootstrap_reps + 1),
            "p_mcse": "sqrt(p_hat * (1-p_hat) / (B+1))",
            "max_t_envelope": (
                "studentized diagnostic envelope across 12 fixed-task contrasts; "
                "calibrated only under the documented empirical swap-null"
            ),
            "studentization": "bootstrap SD per contrast; SE <= 1e-12 is inactive with zero-width envelope",
            "dependency": "shared bootstrap row and seed draw across all contrasts; task-specific engine draws shared across each task's three metrics",
            "rng_seed": 2026072603,
        },
        "small_cluster_sensitivity": {
            "seed_levels": len(FORMAL_BENCHMARK_SEEDS),
            "exact_signflip_patterns": 2 ** len(FORMAL_BENCHMARK_SEEDS),
            "calibration_simulations": args.calibration_simulations,
            "bootstrap_repetitions_per_simulation": args.calibration_bootstrap_reps,
            "scope": "illustrative design-matched calibration; not a proof of inferential validity",
        },
        "max_t_family_calibration": {
            "dgp": "empirical midpoint swap-null",
            "training_seed_signs": "one Rademacher sign per seed, shared across fixed tasks",
            "engine_signs": "one task-specific Rademacher sign per matched engine, shared across three metrics",
            "task_resampling": "none",
            "simulations": args.max_t_calibration_simulations,
            "bootstrap_repetitions_per_simulation": args.max_t_calibration_bootstrap_reps,
            "rng_seed": 2026072701,
            "scope": "design-specific diagnostic; not confirmatory validation",
        },
    }
    (output / "fixed_benchmark_analysis_metadata.json").write_text(
        json.dumps(run_metadata, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    print(
        "FIXED_BENCHMARK_EVIDENCE_READY "
        f"primary={len(primary)} endpoint={len(endpoint)} event_cells={len(counts)} "
        f"calibration={len(calibration)} max_t={len(family_calibration)} features={len(stability)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
