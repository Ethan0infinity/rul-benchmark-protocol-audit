from __future__ import annotations

import math
from typing import Iterable

import numpy as np


def rmse(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(math.sqrt(np.mean((y_pred - y_true) ** 2)))


def mae(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    return float(np.mean(np.abs(y_pred - y_true)))


def r2_score_np(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    denom = np.sum((y_true - np.mean(y_true)) ** 2)
    if denom == 0:
        return 0.0
    return float(1.0 - np.sum((y_true - y_pred) ** 2) / denom)


def nasa_score(y_true: Iterable[float], y_pred: Iterable[float]) -> float:
    """NASA PHM score.

    Positive error means predicted RUL is larger than the true RUL, which is
    late and penalized more heavily.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    score = np.where(err < 0, np.exp(-err / 13.0) - 1.0, np.exp(err / 10.0) - 1.0)
    return float(np.sum(score))


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    metrics = {
        "rmse": rmse(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "r2": r2_score_np(y_true, y_pred),
        "nasa_score": nasa_score(y_true, y_pred),
    }
    metrics.update(critical_zone_metrics(y_true, y_pred))
    return metrics


def _threshold_label(threshold: float) -> str:
    return str(int(threshold)) if float(threshold).is_integer() else str(threshold).replace(".", "_")


def critical_zone_metrics(
    y_true: Iterable[float],
    y_pred: Iterable[float],
    thresholds: Iterable[float] = (30.0, 50.0),
) -> dict[str, float | int | bool | None]:
    """Return near-failure metrics for the declared RUL thresholds.

    The zero-filled ``over_error_mean``, ``late_q95``, and ``late_cvar95``
    fields are retained for backward compatibility. Manuscript CMLE and
    conditional-tail claims use only the corresponding ``conditional_*``
    fields, which are ``None`` when no positive-error event is observed.
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    rows: dict[str, float | int | bool | None] = {}
    for threshold in thresholds:
        mask = y_true <= float(threshold)
        label = _threshold_label(float(threshold))
        prefix = f"critical_{label}"
        count = int(np.sum(mask))
        rows[f"{prefix}_count"] = count
        if count == 0:
            rows[f"{prefix}_rmse"] = 0.0
            rows[f"{prefix}_mae"] = 0.0
            rows[f"{prefix}_late_prediction_ratio"] = 0.0
            rows[f"{prefix}_severe_late_5_ratio"] = 0.0
            rows[f"{prefix}_severe_late_10_ratio"] = 0.0
            rows[f"{prefix}_early_prediction_ratio"] = 0.0
            rows[f"{prefix}_over_error_mean"] = 0.0
            rows[f"{prefix}_under_error_mean"] = 0.0
            rows[f"{prefix}_mean_late_excess"] = 0.0
            rows[f"{prefix}_late_q95"] = 0.0
            rows[f"{prefix}_late_cvar95"] = 0.0
            rows[f"{prefix}_late_event_count"] = 0
            rows[f"{prefix}_conditional_tail_defined"] = False
            rows[f"{prefix}_conditional_over_error_mean"] = None
            rows[f"{prefix}_conditional_late_q95"] = None
            rows[f"{prefix}_conditional_late_cvar95"] = None
            for ratio in (1, 2, 5, 10):
                rows[f"{prefix}_decision_cost_{ratio}"] = 0.0
            continue

        zone_true = y_true[mask]
        zone_pred = y_pred[mask]
        zone_err = err[mask]
        over = zone_err[zone_err > 0]
        under = zone_err[zone_err < 0]
        rows[f"{prefix}_rmse"] = rmse(zone_true, zone_pred)
        rows[f"{prefix}_mae"] = mae(zone_true, zone_pred)
        rows[f"{prefix}_late_prediction_ratio"] = float(np.mean(zone_err > 0))
        rows[f"{prefix}_severe_late_5_ratio"] = float(np.mean(zone_err > 5.0))
        rows[f"{prefix}_severe_late_10_ratio"] = float(np.mean(zone_err > 10.0))
        rows[f"{prefix}_early_prediction_ratio"] = float(np.mean(zone_err < 0))
        rows[f"{prefix}_late_event_count"] = int(len(over))
        rows[f"{prefix}_conditional_tail_defined"] = bool(len(over))
        rows[f"{prefix}_over_error_mean"] = float(np.mean(over)) if len(over) else 0.0
        rows[f"{prefix}_under_error_mean"] = float(np.mean(under)) if len(under) else 0.0
        positive = np.maximum(zone_err, 0.0)
        rows[f"{prefix}_mean_late_excess"] = float(np.mean(positive))
        if len(over):
            late_q95 = float(np.quantile(over, 0.95))
            late_cvar95 = float(np.mean(over[over >= late_q95]))
        else:
            late_q95 = late_cvar95 = 0.0
        rows[f"{prefix}_late_q95"] = late_q95
        rows[f"{prefix}_late_cvar95"] = late_cvar95
        rows[f"{prefix}_conditional_over_error_mean"] = float(np.mean(over)) if len(over) else None
        rows[f"{prefix}_conditional_late_q95"] = late_q95 if len(over) else None
        rows[f"{prefix}_conditional_late_cvar95"] = late_cvar95 if len(over) else None
        for ratio in (1, 2, 5, 10):
            rows[f"{prefix}_decision_cost_{ratio}"] = float(
                np.mean(np.maximum(-zone_err, 0.0) + ratio * np.maximum(zone_err, 0.0))
            )
    return rows


def monotonicity_metrics(data: object, y_pred: Iterable[float], tolerance: float = 0.0) -> dict[str, float]:
    """Measure prediction increases along each engine trajectory.

    RUL should generally decrease as cycle increases. A violation is counted
    when the next prediction for the same engine is larger than the previous
    prediction by more than ``tolerance``.
    """
    unit_ids = np.asarray(getattr(data, "unit_ids"), dtype=np.int64)
    end_cycles = np.asarray(getattr(data, "end_cycles"), dtype=np.int64)
    preds = np.asarray(y_pred, dtype=np.float64)
    pair_count = 0
    violation_count = 0
    violation_sum = 0.0

    for unit_id in np.unique(unit_ids):
        indices = np.where(unit_ids == unit_id)[0]
        if len(indices) < 2:
            continue
        order = indices[np.argsort(end_cycles[indices])]
        diffs = np.diff(preds[order])
        violations = diffs > tolerance
        pair_count += len(diffs)
        violation_count += int(np.sum(violations))
        violation_sum += float(np.sum(diffs[violations] - tolerance)) if np.any(violations) else 0.0

    return {
        "monotonic_pairs": int(pair_count),
        "monotonic_violation_count": int(violation_count),
        "monotonic_violation_rate": float(violation_count / pair_count) if pair_count else 0.0,
        "monotonic_violation_mean": float(violation_sum / violation_count) if violation_count else 0.0,
    }
