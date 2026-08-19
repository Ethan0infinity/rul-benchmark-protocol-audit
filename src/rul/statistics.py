from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def _average_ranks(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values)
    ranks = np.empty(len(values), dtype=np.float64)
    sorted_values = values[order]
    start = 0
    while start < len(values):
        end = start + 1
        while end < len(values) and sorted_values[end] == sorted_values[start]:
            end += 1
        average_rank = (start + 1 + end) / 2.0
        ranks[order[start:end]] = average_rank
        start = end
    return ranks


def wilcoxon_signed_rank(
    baseline_errors: Iterable[float],
    proposed_errors: Iterable[float],
) -> dict[str, float]:
    """Wilcoxon signed-rank test for paired absolute errors.

    Positive differences mean the proposed method has lower error than the
    baseline. The p-value uses a normal approximation, which is sufficient for
    the experiment-summary role here and avoids adding a SciPy dependency.
    """
    baseline = np.asarray(baseline_errors, dtype=np.float64)
    proposed = np.asarray(proposed_errors, dtype=np.float64)
    if baseline.shape != proposed.shape:
        raise ValueError("baseline_errors and proposed_errors must have the same shape")

    diff = baseline - proposed
    diff = diff[diff != 0]
    n = int(len(diff))
    if n == 0:
        return {"n": 0, "w_plus": 0.0, "w_minus": 0.0, "statistic": 0.0, "p_value": 1.0, "effect_size": 0.0}

    ranks = _average_ranks(np.abs(diff))
    w_plus = float(np.sum(ranks[diff > 0]))
    w_minus = float(np.sum(ranks[diff < 0]))
    statistic = float(min(w_plus, w_minus))

    mean = n * (n + 1) / 4.0
    variance = n * (n + 1) * (2 * n + 1) / 24.0
    if variance <= 0:
        p_value = 1.0
    else:
        z = (statistic - mean + 0.5) / math.sqrt(variance)
        p_value = float(math.erfc(abs(z) / math.sqrt(2.0)))

    rank_total = n * (n + 1) / 2.0
    effect_size = float((w_plus - w_minus) / rank_total) if rank_total else 0.0
    return {
        "n": n,
        "w_plus": w_plus,
        "w_minus": w_minus,
        "statistic": statistic,
        "p_value": max(0.0, min(1.0, p_value)),
        "effect_size": effect_size,
    }


def paired_bootstrap_mean_diff(
    baseline_values: Iterable[float],
    proposed_values: Iterable[float],
    *,
    n_bootstrap: int = 2000,
    confidence: float = 0.95,
    seed: int = 20260708,
) -> dict[str, float]:
    """Bootstrap CI for mean paired improvement.

    The reported difference is ``baseline - proposed``. Positive values mean
    the proposed model has a lower metric value than the baseline.
    """
    baseline = np.asarray(baseline_values, dtype=np.float64)
    proposed = np.asarray(proposed_values, dtype=np.float64)
    if baseline.shape != proposed.shape:
        raise ValueError("baseline_values and proposed_values must have the same shape")
    mask = np.isfinite(baseline) & np.isfinite(proposed)
    diff = baseline[mask] - proposed[mask]
    n = int(len(diff))
    if n == 0:
        return {"mean_diff": 0.0, "ci_low": 0.0, "ci_high": 0.0, "n_bootstrap": 0}

    rng = np.random.default_rng(seed)
    indices = rng.integers(0, n, size=(int(n_bootstrap), n))
    samples = diff[indices].mean(axis=1)
    alpha = (1.0 - confidence) / 2.0
    return {
        "mean_diff": float(diff.mean()),
        "ci_low": float(np.quantile(samples, alpha)),
        "ci_high": float(np.quantile(samples, 1.0 - alpha)),
        "n_bootstrap": int(n_bootstrap),
    }


def exact_mcnemar_test(
    baseline_indicator: Iterable[float],
    proposed_indicator: Iterable[float],
) -> dict[str, float]:
    """Exact McNemar/binomial test for paired binary indicators.

    Indicators should be 1 for an undesirable event, such as a late
    prediction. ``baseline_only`` counts engines where only the baseline
    produced the event; ``proposed_only`` counts engines where only the
    proposed model produced the event. Positive effect size means fewer events
    for the proposed model.
    """
    baseline = np.asarray(baseline_indicator, dtype=np.float64)
    proposed = np.asarray(proposed_indicator, dtype=np.float64)
    if baseline.shape != proposed.shape:
        raise ValueError("baseline_indicator and proposed_indicator must have the same shape")
    mask = np.isfinite(baseline) & np.isfinite(proposed)
    b = baseline[mask] > 0.5
    p = proposed[mask] > 0.5
    baseline_only = int(np.sum(b & ~p))
    proposed_only = int(np.sum(~b & p))
    discordant = baseline_only + proposed_only
    if discordant == 0:
        p_value = 1.0
    else:
        tail = min(baseline_only, proposed_only)
        p_value = 2.0 * sum(math.comb(discordant, i) for i in range(tail + 1)) * (0.5 ** discordant)
        p_value = min(1.0, p_value)
    n = int(np.sum(mask))
    return {
        "n": n,
        "baseline_only": baseline_only,
        "proposed_only": proposed_only,
        "discordant_pairs": discordant,
        "statistic": float(min(baseline_only, proposed_only)),
        "p_value": float(p_value),
        "effect_size": float((baseline_only - proposed_only) / n) if n else 0.0,
    }
