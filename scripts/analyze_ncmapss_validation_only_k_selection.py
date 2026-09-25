from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit N-CMAPSS K selection using validation evidence only.")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "paper_outputs" / "advanced_evidence"))
    args = parser.parse_args()
    output = Path(args.output_dir)

    frames = []
    for k in (4, 6, 8):
        path = output / f"ncmapss_ds02_k{k}_runs.csv"
        frame = pd.read_csv(path)
        frame = frame[frame["model"] == "rast_gru_v2"].copy()
        if frame["seed"].nunique() != 5:
            raise ValueError(f"Expected five OCM seeds for K={k}, found {frame['seed'].nunique()}")
        frame["condition_clusters"] = k
        frames.append(frame)
    long = pd.concat(frames, ignore_index=True)

    validation_summary = long.groupby("condition_clusters", as_index=False).agg(
        seed_count=("seed", "nunique"),
        validation_risk_score_mean=("best_val_risk_score", "mean"),
        validation_risk_score_std=("best_val_risk_score", lambda x: x.std(ddof=1)),
        validation_rmse_mean=("val_rmse", "mean"),
        validation_lpr30_mean=("val_critical_30_late_prediction_ratio", "mean"),
        test_unit_macro_rmse_mean=("unit_macro_rmse", "mean"),
        test_unit_macro_nasa_mean=("unit_macro_nasa_per_window", "mean"),
        test_unit_macro_lpr30_mean=("unit_macro_lpr30", "mean"),
    ).sort_values("condition_clusters")
    selected_k = int(validation_summary.sort_values("validation_risk_score_mean").iloc[0]["condition_clusters"])
    validation_summary["selected_by_validation_only"] = validation_summary["condition_clusters"] == selected_k
    validation_summary["selection_rule"] = "lowest five-seed mean validation risk score; test metrics excluded"
    validation_summary.to_csv(output / "ncmapss_validation_only_k_selection.csv", index=False)

    selected_runs = long[long["condition_clusters"] == selected_k].copy()
    selected_runs.to_csv(output / "ncmapss_validation_selected_k_seed_metrics.csv", index=False)
    verdict = pd.DataFrame(
        [
            {
                "selected_k": selected_k,
                "selected_validation_risk_mean": float(
                    validation_summary.loc[
                        validation_summary["condition_clusters"] == selected_k,
                        "validation_risk_score_mean",
                    ].iloc[0]
                ),
                "selected_test_unit_macro_rmse_mean": float(selected_runs["unit_macro_rmse"].mean()),
                "selected_test_unit_macro_lpr30_mean": float(selected_runs["unit_macro_lpr30"].mean()),
                "post_lock_test_best_k": int(
                    validation_summary.sort_values("test_unit_macro_rmse_mean").iloc[0]["condition_clusters"]
                ),
                "interpretation": (
                    "validation-only selection does not reproduce the post-lock test-best K; "
                    "DS02 remains an exploratory external task rather than stable external confirmation"
                ),
            }
        ]
    )
    verdict.to_csv(output / "ncmapss_validation_only_k_verdict.csv", index=False)
    print(f"NCMAPSS_VALIDATION_ONLY_K_READY selected_k={selected_k} output={output}")


if __name__ == "__main__":
    main()
