from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).absolute().parents[1]
OUTPUT = PROJECT_ROOT / "paper_outputs" / "advanced_evidence"
METRICS = {
    "test_rmse": "RMSE",
    "test_nasa_per_engine": "NASA/engine",
    "test_critical_30_late_prediction_ratio": "LPR@30",
    "test_critical_30_late_cvar95": "Late CVaR95",
}


def main() -> None:
    rows = json.loads((PROJECT_ROOT / "results" / "fast_cudnn_repeat_audit.json").read_text(encoding="utf-8-sig"))
    frame = pd.DataFrame(rows)
    required = {
        "audit_model",
        "audit_stage",
        "audit_seed",
        "audit_replicate",
        "test_nasa_score",
        "n_test_units",
        *(metric for metric in METRICS if metric != "test_nasa_per_engine"),
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Fast-cuDNN audit is missing columns: {sorted(missing)}")
    if (frame["n_test_units"] <= 0).any():
        raise ValueError("Fast-cuDNN audit contains a non-positive test-engine count.")
    frame["test_nasa_per_engine"] = frame["test_nasa_score"] / frame["n_test_units"]
    if len(frame) != 24:
        raise ValueError(f"Expected 24 trajectories, found {len(frame)}")
    frame.to_csv(OUTPUT / "fast_cudnn_repeat_seed_metrics.csv", index=False)

    cross = pd.read_csv(OUTPUT / "cross_backbone_protocol_buildup_seed_metrics.csv")
    metric_map = {
        "test_rmse": "rmse",
        "test_nasa_per_engine": "nasa_per_engine",
        "test_critical_30_late_prediction_ratio": "lpr30",
        "test_critical_30_late_cvar95": "late_cvar95_30",
    }
    summary_rows = []
    for (model, stage), group in frame.groupby(["audit_model", "audit_stage"], sort=True):
        cross_group = cross[(cross["model"] == model) & (cross["stage"] == stage)]
        for metric, display in METRICS.items():
            within_sd = group.groupby("audit_seed")[metric].std(ddof=1)
            within_range = group.groupby("audit_seed")[metric].agg(lambda values: float(values.max() - values.min()))
            cross_seed_sd = float(cross_group[metric_map[metric]].std(ddof=1))
            mean_within_sd = float(within_sd.mean())
            summary_rows.append(
                {
                    "model": model,
                    "stage": stage,
                    "metric": display,
                    "fixed_seed_repeat_sd_mean": mean_within_sd,
                    "fixed_seed_repeat_range_max": float(within_range.max()),
                    "cross_seed_sd_original_grid": cross_seed_sd,
                    "repeat_to_cross_seed_sd_ratio": mean_within_sd / cross_seed_sd if cross_seed_sd > 0 else np.nan,
                    "seed_count": int(group["audit_seed"].nunique()),
                    "replicates_per_seed": int(group["audit_replicate"].nunique()),
                    "interpretation": "fast-cuDNN fixed-seed rerun variability versus original five-seed variability",
                }
            )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(OUTPUT / "fast_cudnn_repeat_summary.csv", index=False)
    print(
        "FAST_CUDNN_REPEAT_ANALYSIS_READY "
        f"trajectories={len(frame)} summaries={len(summary)} "
        f"max_ratio={summary['repeat_to_cross_seed_sd_ratio'].max():.3f}"
    )


if __name__ == "__main__":
    main()
