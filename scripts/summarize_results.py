from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def collect_metrics(results_dir: Path) -> pd.DataFrame:
    rows = []
    for path in results_dir.rglob("metrics.json"):
        with path.open("r", encoding="utf-8-sig") as f:
            row = json.load(f)
        row["run_dir"] = str(path.parent)
        rows.append(row)
    if not rows:
        raise FileNotFoundError(f"No metrics.json files found under {results_dir}")
    return pd.DataFrame(rows)


def to_markdown_table(df: pd.DataFrame, output_path: Path) -> None:
    cols = [
        "subset",
        "model",
        "test_rmse",
        "test_mae",
        "test_r2",
        "test_nasa_score",
        "parameters",
        "batch_inference_ms",
        "n_features",
    ]
    available = [c for c in cols if c in df.columns]
    table = df[available].sort_values(["subset", "test_rmse"]).to_markdown(index=False)
    output_path.write_text("# Experiment Summary\n\n" + table + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Collect experiment metrics.")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    df = collect_metrics(results_dir)
    csv_path = results_dir / "summary.csv"
    md_path = results_dir / "summary.md"
    df.to_csv(csv_path, index=False)
    to_markdown_table(df, md_path)
    print(f"Saved {csv_path}")
    print(f"Saved {md_path}")


if __name__ == "__main__":
    main()
