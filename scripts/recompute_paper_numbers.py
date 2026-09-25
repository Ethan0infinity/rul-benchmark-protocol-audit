from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rul.reporting import collect_metrics, filter_metric_rows


def recompute_numbers(
    results_dir: str | Path,
    *,
    experiment_names: list[str] | None = None,
    experiment_prefix: str | None = None,
) -> pd.DataFrame:
    rows = filter_metric_rows(
        collect_metrics(results_dir),
        experiment_names=experiment_names,
        experiment_prefix=experiment_prefix,
    )
    keep = [
        "experiment_name",
        "subset",
        "model",
        "seed",
        "protocol_version",
        "result_status",
        "allowed_for_paper",
        "best_epoch",
        "completed_epochs",
        "test_rmse",
        "test_mae",
        "test_r2",
        "test_nasa_score",
        "test_critical_30_rmse",
        "test_critical_50_rmse",
        "parameters",
        "single_sample_inference_ms",
        "cpu_single_sample_inference_ms",
        "run_dir",
    ]
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=keep)
    return df[[col for col in keep if col in df.columns]].sort_values(["experiment_name", "subset", "model"]).reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Recompute paper numbers from metrics.json files.")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--experiment-name", action="append", dest="experiment_names", default=None)
    parser.add_argument("--experiment-prefix", default=None)
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "paper_outputs" / "recomputed_numbers"))
    args = parser.parse_args()

    df = recompute_numbers(args.results_dir, experiment_names=args.experiment_names, experiment_prefix=args.experiment_prefix)
    out = Path(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    csv_path = out / "recomputed_metrics.csv"
    json_path = out / "recomputed_metrics.json"
    df.to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(df.to_dict(orient="records"), indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Saved recomputed metrics to {csv_path}")
    print(f"RECOMPUTE_ROWS={len(df)}")


if __name__ == "__main__":
    main()
