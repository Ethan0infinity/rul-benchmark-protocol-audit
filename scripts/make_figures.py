from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rul.plotting import plot_metric_bars, plot_predictions


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate paper-ready starter figures.")
    parser.add_argument("--run-dir", default=str(PROJECT_ROOT / "results" / "FD001" / "rs_tcn_gru"))
    parser.add_argument("--summary-csv", default=str(PROJECT_ROOT / "results" / "summary.csv"))
    parser.add_argument("--metric", default="test_rmse")
    args = parser.parse_args()

    run_dir = Path(args.run_dir)
    prediction_csv = run_dir / "test_predictions.csv"
    if prediction_csv.exists():
        plot_predictions(prediction_csv, run_dir / "fig_test_predictions.png")
        print(f"Saved {run_dir / 'fig_test_predictions.png'}")
    summary_csv = Path(args.summary_csv)
    if summary_csv.exists():
        plot_metric_bars(summary_csv, summary_csv.parent / f"fig_{args.metric}.png", metric=args.metric)
        print(f"Saved {summary_csv.parent / f'fig_{args.metric}.png'}")


if __name__ == "__main__":
    main()
