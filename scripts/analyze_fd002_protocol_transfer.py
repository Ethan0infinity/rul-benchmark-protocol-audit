from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rul.metrics import critical_zone_metrics, nasa_score


SEEDS = (42, 123, 2024, 2025, 2026)
POINTS = ("rast_core", "transformer_core", "ocm_core", "rast_asym", "ocm_asym")
COMPARISONS = (
    ("rast_core", "rast_asym", "RAST protocol effect"),
    ("ocm_core", "ocm_asym", "OCM protocol effect"),
    ("ocm_core", "rast_core", "Core architecture contrast"),
    ("ocm_core", "transformer_core", "Core strongest-reference contrast"),
    ("ocm_asym", "rast_asym", "Asym architecture-protocol contrast"),
)
METRICS = ("rmse", "nasa_per_engine", "lpr30", "cost_5")


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    """Return Holm-adjusted p-values while preserving the original row order."""
    values = np.asarray(p_values, dtype=float)
    if values.ndim != 1 or not np.all(np.isfinite(values)):
        raise ValueError("Holm adjustment requires a finite one-dimensional p-value array.")
    order = np.argsort(values, kind="stable")
    adjusted_sorted = np.maximum.accumulate(
        (len(values) - np.arange(len(values))) * values[order]
    )
    adjusted = np.empty_like(values)
    adjusted[order] = np.minimum(adjusted_sorted, 1.0)
    return adjusted


def metric(true: np.ndarray, pred: np.ndarray, name: str) -> float:
    error = pred - true
    if name == "rmse":
        return float(np.sqrt(np.mean(error**2)))
    if name == "nasa_per_engine":
        return float(nasa_score(true, pred) / len(true))
    critical = critical_zone_metrics(true, pred, thresholds=(30.0,))
    if name == "lpr30":
        return float(critical["critical_30_late_prediction_ratio"])
    if name == "cost_5":
        return float(critical["critical_30_decision_cost_5"])
    raise ValueError(name)


def load_manifest(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    expected = {(point, seed) for point in POINTS for seed in SEEDS}
    observed = {(str(row.point), int(row.seed)) for row in frame.itertuples(index=False)}
    if expected != observed:
        raise ValueError(f"Incomplete FD002 transfer manifest: missing={sorted(expected-observed)} extra={sorted(observed-expected)}")
    if frame["status"].isin(["planned"]).any():
        pending = frame.loc[frame["status"].eq("planned"), ["point", "seed"]].to_dict("records")
        raise RuntimeError(f"FD002 transfer training is incomplete: {pending}")
    return frame


def load_predictions(manifest: pd.DataFrame) -> dict[tuple[str, int], pd.DataFrame]:
    predictions: dict[tuple[str, int], pd.DataFrame] = {}
    reference_endpoints: pd.DataFrame | None = None
    for row in manifest.itertuples(index=False):
        path = PROJECT_ROOT / str(row.run_dir) / "test_predictions.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path).sort_values("unit_id").reset_index(drop=True)
        if frame["unit_id"].duplicated().any():
            raise ValueError(f"Expected one official endpoint per FD002 test engine: {path}")
        if len(frame) != 259:
            raise ValueError(f"Expected 259 official FD002 test-engine endpoints, found {len(frame)}: {path}")
        endpoints = frame[["unit_id", "true_rul"]].copy()
        if reference_endpoints is None:
            reference_endpoints = endpoints
        elif not endpoints.equals(reference_endpoints):
            raise ValueError(f"FD002 endpoint identity or true RUL differs across operating points: {path}")
        predictions[(str(row.point), int(row.seed))] = frame
    return predictions


def seed_metrics(predictions: dict[tuple[str, int], pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for point in POINTS:
        for seed in SEEDS:
            frame = predictions[(point, seed)]
            true = frame["true_rul"].to_numpy(float)
            pred = frame["pred_rul"].to_numpy(float)
            row = {"point": point, "seed": seed, "engines": len(frame)}
            row.update({name: metric(true, pred, name) for name in METRICS})
            rows.append(row)
    return pd.DataFrame(rows)


def paired_intervals(
    predictions: dict[tuple[str, int], pd.DataFrame], reps: int
) -> pd.DataFrame:
    rows = []
    for comparison_index, (left, right, label) in enumerate(COMPARISONS):
        merged_by_seed = {}
        for seed in SEEDS:
            merged = predictions[(left, seed)].merge(
                predictions[(right, seed)],
                on=["unit_id", "true_rul"],
                suffixes=("_left", "_right"),
                validate="one_to_one",
            ).sort_values("unit_id").reset_index(drop=True)
            merged_by_seed[seed] = merged
            if len(merged) != 259:
                raise ValueError(
                    f"Matched FD002 comparison {left} vs {right}, seed {seed}, has {len(merged)} engines; expected 259"
                )
        for metric_index, metric_name in enumerate(METRICS):
            observed = []
            for frame in merged_by_seed.values():
                true = frame["true_rul"].to_numpy(float)
                observed.append(
                    metric(true, frame["pred_rul_left"].to_numpy(float), metric_name)
                    - metric(true, frame["pred_rul_right"].to_numpy(float), metric_name)
                )
            rng = np.random.default_rng(20260722 + comparison_index * 100 + metric_index)
            samples = np.empty(reps, dtype=float)
            for rep in range(reps):
                effects = []
                for sampled_seed in rng.choice(SEEDS, len(SEEDS), replace=True):
                    frame = merged_by_seed[int(sampled_seed)]
                    indices = rng.integers(0, len(frame), len(frame))
                    true = frame["true_rul"].to_numpy(float)[indices]
                    effects.append(
                        metric(true, frame["pred_rul_left"].to_numpy(float)[indices], metric_name)
                        - metric(true, frame["pred_rul_right"].to_numpy(float)[indices], metric_name)
                    )
                samples[rep] = float(np.mean(effects))
            low, high = np.quantile(samples, [0.025, 0.975])
            observed_mean = float(np.mean(observed))
            centered = samples - observed_mean
            nominal_p = float(
                (1 + np.count_nonzero(np.abs(centered) >= abs(observed_mean)))
                / (reps + 1)
            )
            classification = "unresolved"
            if high < 0:
                classification = f"CI supports lower {left}"
            elif low > 0:
                classification = f"CI supports lower {right}"
            rows.append(
                {
                    "comparison": label,
                    "left": left,
                    "right": right,
                    "metric": metric_name,
                    "mean_left_minus_right": observed_mean,
                    "ci95_low": float(low),
                    "ci95_high": float(high),
                    "nominal_p_centered_bootstrap": nominal_p,
                    "classification_nominal_ci": classification,
                    "bootstrap_primary_unit": "matched FD002 test engine nested in composite seed",
                    "bootstrap_reps": reps,
                }
            )
    frame = pd.DataFrame(rows)
    frame["holm_p_20"] = holm_adjust(
        frame["nominal_p_centered_bootstrap"].to_numpy(float)
    )
    frame["holm_significant_0_05"] = frame["holm_p_20"] < 0.05
    # Backward-compatible alias for artifact gates; manuscript interpretation
    # uses the explicit nominal and Holm-aware columns below.
    frame["classification"] = frame["classification_nominal_ci"]
    frame["classification_holm_20"] = np.where(
        frame["holm_significant_0_05"],
        frame["classification_nominal_ci"],
        "unresolved after Holm correction",
    )
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the matched five-seed FD002 protocol-transfer grid.")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=PROJECT_ROOT / "results" / "fd002_protocol_transfer" / "run_manifest.csv",
    )
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    args = parser.parse_args()

    manifest = load_manifest(args.manifest)
    predictions = load_predictions(manifest)
    seeds = seed_metrics(predictions)
    summary = seeds.groupby("point", as_index=False)[list(METRICS)].agg(["mean", "std"])
    summary.columns = ["point" if a == "point" else f"{a}_{b}" for a, b in summary.columns]
    intervals = paired_intervals(predictions, args.bootstrap_reps)

    output = PROJECT_ROOT / "paper_outputs" / "advanced_evidence"
    output.mkdir(parents=True, exist_ok=True)
    seeds.to_csv(output / "fd002_protocol_transfer_seed_metrics.csv", index=False)
    summary.to_csv(output / "fd002_protocol_transfer_summary.csv", index=False)
    intervals.to_csv(output / "fd002_protocol_transfer_paired_bootstrap.csv", index=False)
    print(
        "FD002_PROTOCOL_TRANSFER_EVIDENCE_READY "
        f"seed_rows={len(seeds)} summary_rows={len(summary)} intervals={len(intervals)}"
    )


if __name__ == "__main__":
    main()
