from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_advanced_evidence import prediction_metrics
from run_formal_benchmark import FORMAL_BENCHMARK_SEEDS


VARIANTS = ("base_only", "plus_risk", "plus_risk_cons", "full")


def run_dir(root: Path, seed: int, variant: str) -> Path:
    if variant == "full":
        return root / f"paper_main_v3_seed{seed}" / "FD004" / "rast_gru_v2"
    return root / f"attribution_ocm_{variant}_fd004_seed{seed}" / "FD004" / "rast_gru_v2"


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze OCM objective build-up attribution controls.")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "paper_outputs" / "advanced_evidence"))
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    args = parser.parse_args()
    root = Path(args.results_dir)
    frames: dict[tuple[int, str], pd.DataFrame] = {}
    rows = []
    for seed in FORMAL_BENCHMARK_SEEDS:
        for variant in VARIANTS:
            directory = run_dir(root, seed, variant)
            pred_path = directory / "test_predictions.csv"
            if not pred_path.exists():
                raise FileNotFoundError(pred_path)
            pred = pd.read_csv(pred_path)
            frames[(seed, variant)] = pred
            metrics = json.loads((directory / "metrics.json").read_text(encoding="utf-8-sig"))
            rows.append({"seed": seed, "variant": variant, "best_epoch": metrics["best_epoch"],
                         "parameters": metrics["parameters"], **prediction_metrics(pred)})
    long = pd.DataFrame(rows)
    metric_names = ("rmse", "mae", "nasa_per_engine", "lpr30", "mle30", "late_cvar95_30")
    summary = long.groupby("variant", as_index=False).agg(
        seed_count=("seed", "nunique"),
        **{f"{m}_mean": (m, "mean") for m in metric_names},
        **{f"{m}_std": (m, lambda x: x.std(ddof=1)) for m in metric_names},
    )

    rng = np.random.default_rng(20260716)
    comparisons = (("base_only", "plus_risk"), ("plus_risk", "plus_risk_cons"), ("plus_risk_cons", "full"), ("base_only", "full"))
    bootstrap_rows = []
    for left, right in comparisons:
        for metric in ("rmse", "nasa_per_engine", "lpr30"):
            observed = float(long[long.variant == left].set_index("seed")[metric].sub(
                long[long.variant == right].set_index("seed")[metric]).mean())
            samples = []
            for _ in range(args.bootstrap_reps):
                seed_values = []
                for seed in rng.choice(FORMAL_BENCHMARK_SEEDS, size=5, replace=True):
                    a = frames[(int(seed), left)]
                    b = frames[(int(seed), right)]
                    merged = a.merge(b, on=["unit_id", "true_rul"], suffixes=("_a", "_b"), validate="one_to_one")
                    idx = rng.integers(0, len(merged), len(merged))
                    sample = merged.iloc[idx]
                    ma = prediction_metrics(sample.rename(columns={"pred_rul_a": "pred_rul"}))
                    mb = prediction_metrics(sample.rename(columns={"pred_rul_b": "pred_rul"}))
                    seed_values.append(ma[metric] - mb[metric])
                samples.append(float(np.mean(seed_values)))
            bootstrap_rows.append({"left": left, "right": right, "metric": metric,
                                   "mean_difference_left_minus_right": observed,
                                   "ci95_low": float(np.quantile(samples, .025)),
                                   "ci95_high": float(np.quantile(samples, .975)),
                                   "bootstrap_repetitions": args.bootstrap_reps})
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    long.to_csv(output / "ocm_attribution_seed_metrics.csv", index=False)
    summary.to_csv(output / "ocm_attribution_summary.csv", index=False)
    pd.DataFrame(bootstrap_rows).to_csv(output / "ocm_attribution_paired_bootstrap.csv", index=False)
    print(f"OCM_ATTRIBUTION_READY rows={len(long)} comparisons={len(bootstrap_rows)}")


if __name__ == "__main__":
    main()
