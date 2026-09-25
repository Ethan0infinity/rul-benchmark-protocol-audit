from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rul.metrics import nasa_score


SEEDS = (42, 123, 2024, 2025, 2026)
SUBSETS = ("FD001", "FD002", "FD003", "FD004")


def prediction_path(seed: int, subset: str, model: str) -> Path:
    if model == "dual_mixer":
        return PROJECT_ROOT / "results" / f"official_dual_mixer_seed{seed}" / subset / "official_dual_mixer" / "test_predictions.csv"
    if model == "ocm_core":
        if subset == "FD004":
            return PROJECT_ROOT / "results" / f"ablation_fd004_seed{seed}_weighted_huber_no_asymmetry" / subset / "rast_gru_v2" / "test_predictions.csv"
        return PROJECT_ROOT / "results" / f"core_ocm_seed{seed}" / subset / "rast_gru_v2" / "test_predictions.csv"
    return PROJECT_ROOT / "results" / f"paper_main_v3_seed{seed}" / subset / "rast_gru_v2" / "test_predictions.csv"


def metric_value(frame: pd.DataFrame, side: str, metric: str, indices: np.ndarray | None = None) -> float:
    part = frame if indices is None else frame.iloc[indices]
    true = part["true_rul"].to_numpy(float)
    pred = part[f"pred_rul_{side}"].to_numpy(float)
    if metric == "rmse":
        return float(np.sqrt(np.mean((pred - true) ** 2)))
    if metric == "nasa_per_engine":
        return float(nasa_score(true, pred) / len(part))
    critical = true <= 30.0
    return float(np.mean((pred - true)[critical] > 0.0))


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the official-code-derived Dual-Mixer reference baseline.")
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    args = parser.parse_args()

    seed_metrics_path = PROJECT_ROOT / "paper_outputs" / "advanced_evidence" / "official_dual_mixer_seed_metrics.csv"
    seed_metrics = pd.read_csv(seed_metrics_path)
    if len(seed_metrics) != 20 or set(seed_metrics["seed"].astype(int)) != set(SEEDS):
        raise ValueError("Official Dual-Mixer evidence must contain 4 subsets x 5 seeds before analysis.")

    rng = np.random.default_rng(20260717)
    rows: list[dict] = []
    for subset in SUBSETS:
        paired: dict[int, pd.DataFrame] = {}
        for seed in SEEDS:
            dual = pd.read_csv(prediction_path(seed, subset, "dual_mixer"))
            ocm = pd.read_csv(prediction_path(seed, subset, "ocm"))
            paired[seed] = dual.merge(
                ocm,
                on=["unit_id", "true_rul"],
                suffixes=("_dual", "_ocm"),
                validate="one_to_one",
            )
        for metric in ("rmse", "nasa_per_engine", "lpr30"):
            dual_values = [metric_value(frame, "dual", metric) for frame in paired.values()]
            ocm_values = [metric_value(frame, "ocm", metric) for frame in paired.values()]
            observed = float(np.mean(np.asarray(dual_values) - np.asarray(ocm_values)))
            samples = np.empty(args.bootstrap_reps, dtype=float)
            for rep in range(args.bootstrap_reps):
                effects = []
                for sampled_seed in rng.choice(SEEDS, size=len(SEEDS), replace=True):
                    frame = paired[int(sampled_seed)]
                    indices = rng.integers(0, len(frame), size=len(frame))
                    effects.append(
                        metric_value(frame, "dual", metric, indices) - metric_value(frame, "ocm", metric, indices)
                    )
                samples[rep] = float(np.mean(effects))
            low, high = np.quantile(samples, [0.025, 0.975])
            rows.append(
                {
                    "subset": subset,
                    "metric": metric,
                    "dual_mean": float(np.mean(dual_values)),
                    "dual_std": float(np.std(dual_values, ddof=1)),
                    "ocm_mean": float(np.mean(ocm_values)),
                    "ocm_std": float(np.std(ocm_values, ddof=1)),
                    "mean_paired_difference_dual_minus_ocm": observed,
                    "percentile_ci95_low": float(low),
                    "percentile_ci95_high": float(high),
                    "probability_ocm_lower": float(np.mean(samples > 0.0)),
                    "training_seed_count": len(SEEDS),
                    "bootstrap_repetitions": args.bootstrap_reps,
                    "comparison_scope": "official-code-derived architecture/recommended settings versus locked common-risk OCM; not pure architecture attribution",
                }
            )
    output = PROJECT_ROOT / "paper_outputs" / "advanced_evidence" / "official_dual_mixer_paired_bootstrap.csv"
    pd.DataFrame(rows).to_csv(output, index=False)

    operating_rows: list[dict] = []
    for variant_index, variant in enumerate(("ocm_core", "ocm_asym")):
        for subset_index, subset in enumerate(SUBSETS):
            paired = {}
            for seed in SEEDS:
                dual = pd.read_csv(prediction_path(seed, subset, "dual_mixer"))
                ocm = pd.read_csv(prediction_path(seed, subset, variant))
                paired[seed] = dual.merge(
                    ocm,
                    on=["unit_id", "true_rul"],
                    suffixes=("_dual", "_ocm"),
                    validate="one_to_one",
                )
            for metric_index, metric in enumerate(("rmse", "nasa_per_engine", "lpr30")):
                local_rng = np.random.default_rng(20260717 + variant_index * 1000 + subset_index * 100 + metric_index)
                dual_values = [metric_value(frame, "dual", metric) for frame in paired.values()]
                ocm_values = [metric_value(frame, "ocm", metric) for frame in paired.values()]
                samples = np.empty(args.bootstrap_reps, dtype=float)
                for rep in range(args.bootstrap_reps):
                    effects = []
                    for sampled_seed in local_rng.choice(SEEDS, size=len(SEEDS), replace=True):
                        frame = paired[int(sampled_seed)]
                        indices = local_rng.integers(0, len(frame), size=len(frame))
                        effects.append(
                            metric_value(frame, "dual", metric, indices)
                            - metric_value(frame, "ocm", metric, indices)
                        )
                    samples[rep] = float(np.mean(effects))
                raw_p = min(1.0, 2.0 * min(float(np.mean(samples <= 0.0)), float(np.mean(samples >= 0.0))))
                operating_rows.append(
                    {
                        "ocm_variant": variant,
                        "subset": subset,
                        "metric": metric,
                        "dual_mean": float(np.mean(dual_values)),
                        "ocm_mean": float(np.mean(ocm_values)),
                        "mean_paired_difference_dual_minus_ocm": float(np.mean(np.asarray(dual_values) - np.asarray(ocm_values))),
                        "percentile_ci95_low": float(np.quantile(samples, 0.025)),
                        "percentile_ci95_high": float(np.quantile(samples, 0.975)),
                        "probability_ocm_lower": float(np.mean(samples > 0.0)),
                        "bootstrap_two_sided_p": raw_p,
                        "bootstrap_repetitions": args.bootstrap_reps,
                    }
                )
    operating = pd.DataFrame(operating_rows)
    operating["holm_adjusted_p"] = np.nan
    for variant, indices in operating.groupby("ocm_variant").groups.items():
        part = operating.loc[indices]
        order = np.argsort(part["bootstrap_two_sided_p"].to_numpy(float))
        raw = part["bootstrap_two_sided_p"].to_numpy(float)[order]
        adjusted_sorted = np.maximum.accumulate(np.minimum(1.0, raw * (len(raw) - np.arange(len(raw)))))
        adjusted = np.empty_like(adjusted_sorted)
        adjusted[order] = adjusted_sorted
        operating.loc[indices, "holm_adjusted_p"] = adjusted
    operating["holm_family"] = "12 subset-metric contrasts within the named OCM operating point"
    operating_output = PROJECT_ROOT / "paper_outputs" / "advanced_evidence" / "official_dual_mixer_operating_points.csv"
    operating.to_csv(operating_output, index=False)
    print(
        f"OFFICIAL_DUAL_MIXER_ANALYSIS_READY rows={len(rows)} operating_rows={len(operating)} "
        f"output={output} operating_output={operating_output}"
    )


if __name__ == "__main__":
    main()
