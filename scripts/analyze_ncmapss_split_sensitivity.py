from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEEDS = [42, 123, 2024, 2025, 2026]
MODELS = ["rast_gru", "rast_gru_v2"]
VAL_UNITS = [2, 10, 20]


def main() -> None:
    parser = argparse.ArgumentParser(description="Consolidate N-CMAPSS development-split sensitivity evidence.")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    args = parser.parse_args()
    rows = []
    root = Path(args.results_dir)
    for val_unit in VAL_UNITS:
        prefix = "ncmapss_ds02" if val_unit == 20 else f"ncmapss_ds02_val{val_unit}"
        for seed in SEEDS:
            for model in MODELS:
                path = root / f"{prefix}_seed{seed}" / "DS02" / model / "metrics.json"
                if not path.exists():
                    raise FileNotFoundError(path)
                item = json.loads(path.read_text(encoding="utf-8-sig"))
                rows.append({"validation_unit": val_unit, "seed": seed, "model": model,
                             "unit_macro_rmse": item["unit_macro_rmse"],
                             "unit_macro_mae": item["unit_macro_mae"],
                             "unit_macro_nasa_per_window": item["unit_macro_nasa_per_window"],
                             "unit_macro_lpr30": item["unit_macro_lpr30"]})
    long = pd.DataFrame(rows)
    summary = long.groupby(["validation_unit", "model"], as_index=False).agg(
        seed_count=("seed", "nunique"),
        **{f"{metric}_mean": (metric, "mean") for metric in
           ("unit_macro_rmse", "unit_macro_mae", "unit_macro_nasa_per_window", "unit_macro_lpr30")},
        **{f"{metric}_std": (metric, lambda x: x.std(ddof=1)) for metric in
           ("unit_macro_rmse", "unit_macro_mae", "unit_macro_nasa_per_window", "unit_macro_lpr30")},
    )
    output = PROJECT_ROOT / "paper_outputs" / "advanced_evidence"
    long.to_csv(output / "ncmapss_split_sensitivity_seed_metrics.csv", index=False)
    summary.to_csv(output / "ncmapss_split_sensitivity_summary.csv", index=False)
    print(f"NCMAPSS_SPLIT_SENSITIVITY_READY rows={len(long)} summary={len(summary)}")


if __name__ == "__main__":
    main()
