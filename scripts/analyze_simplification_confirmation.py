from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 123, 2024, 2025, 2026)
METRICS = ("unit_macro_rmse", "unit_macro_nasa_per_window", "unit_macro_lpr30")


def main() -> None:
    parser = argparse.ArgumentParser(description="Confirm the no-asymmetry simplification on the fixed N-CMAPSS task.")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "paper_outputs" / "advanced_evidence"))
    parser.add_argument("--bootstrap-reps", type=int, default=10000)
    args = parser.parse_args()

    output = Path(args.output_dir)
    full = pd.read_csv(output / "ncmapss_ds02_runs.csv")
    full = full.loc[full["model"] == "rast_gru_v2", ["seed", *METRICS]].copy()
    full["variant"] = "asymmetric_full"
    symmetric = pd.read_csv(output / "ncmapss_ds02_symmetric_candidate_runs.csv")
    symmetric = symmetric[["seed", *METRICS]].copy()
    symmetric["variant"] = "symmetric_candidate"
    long = pd.concat([full, symmetric], ignore_index=True)
    if set(long["seed"].astype(int)) != set(SEEDS) or len(long) != 10:
        raise ValueError("The simplification confirmation requires both variants under all five seeds.")

    summary = long.groupby("variant", as_index=False).agg(
        seed_count=("seed", "nunique"),
        **{f"{metric}_mean": (metric, "mean") for metric in METRICS},
        **{f"{metric}_std": (metric, lambda x: x.std(ddof=1)) for metric in METRICS},
    )
    wide = long.pivot(index="seed", columns="variant", values=list(METRICS))
    rng = np.random.default_rng(20260715)
    bootstrap_rows = []
    for metric in METRICS:
        difference = (
            wide[(metric, "asymmetric_full")].to_numpy(float)
            - wide[(metric, "symmetric_candidate")].to_numpy(float)
        )
        indices = rng.integers(0, len(difference), size=(args.bootstrap_reps, len(difference)))
        samples = difference[indices].mean(axis=1)
        bootstrap_rows.append(
            {
                "metric": metric,
                "mean_difference_full_minus_symmetric": float(np.mean(difference)),
                "percentile_ci95_low": float(np.quantile(samples, 0.025)),
                "percentile_ci95_high": float(np.quantile(samples, 0.975)),
                "probability_symmetric_lower": float(np.mean(samples > 0.0)),
                "probability_tie": float(np.mean(np.isclose(samples, 0.0, atol=1e-12))),
                "probability_symmetric_higher": float(np.mean(samples < 0.0)),
                "seed_count": len(difference),
                "bootstrap_repetitions": args.bootstrap_reps,
            }
        )
    bootstrap = pd.DataFrame(bootstrap_rows)
    full_indexed = summary.set_index("variant")
    tradeoff = (
        full_indexed.loc["symmetric_candidate", "unit_macro_rmse_mean"]
        < full_indexed.loc["asymmetric_full", "unit_macro_rmse_mean"]
        and full_indexed.loc["symmetric_candidate", "unit_macro_lpr30_mean"]
        > full_indexed.loc["asymmetric_full", "unit_macro_lpr30_mean"]
    )
    verdict = {
        "task": "fixed N-CMAPSS DS02 protocol",
        "design_status": "external confirmation of a post-hoc FD004 simplification hypothesis",
        "accuracy_safety_tradeoff_observed": bool(tradeoff),
        "uniform_simplification_confirmed": False,
        "recommended_interpretation": "retain the locked full model, make no stable-benefit claim for the asymmetric multiplier, and report the dataset-dependent trade-off",
    }
    long.to_csv(output / "simplification_ncmapss_seed_metrics.csv", index=False)
    summary.to_csv(output / "simplification_ncmapss_summary.csv", index=False)
    bootstrap.to_csv(output / "simplification_ncmapss_paired_bootstrap.csv", index=False)
    (output / "simplification_ncmapss_verdict.json").write_text(
        json.dumps(verdict, indent=2), encoding="utf-8", newline="\n"
    )
    print(f"SIMPLIFICATION_CONFIRMATION_READY tradeoff={tradeoff} output={output}")


if __name__ == "__main__":
    main()
