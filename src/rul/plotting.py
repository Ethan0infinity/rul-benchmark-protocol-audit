from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_predictions(prediction_csv: str | Path, output_path: str | Path) -> None:
    df = pd.read_csv(prediction_csv)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4.8))
    plt.plot(df["unit_id"], df["true_rul"], "o-", label="True RUL", linewidth=1.2)
    plt.plot(df["unit_id"], df["pred_rul"], "s-", label="Predicted RUL", linewidth=1.2)
    plt.xlabel("Test unit")
    plt.ylabel("RUL")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()


def plot_metric_bars(summary_csv: str | Path, output_path: str | Path, metric: str = "test_rmse") -> None:
    df = pd.read_csv(summary_csv)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model_label = df["model_display"].astype(str) if "model_display" in df.columns else df["model"].astype(str)
    label = df["subset"].astype(str) + "-" + model_label if "subset" in df.columns else model_label
    plt.figure(figsize=(max(8, len(df) * 0.55), 4.8))
    plt.bar(label, df[metric])
    plt.xticks(rotation=45, ha="right")
    plt.ylabel(metric)
    plt.tight_layout()
    plt.savefig(output_path, dpi=200)
    plt.close()
