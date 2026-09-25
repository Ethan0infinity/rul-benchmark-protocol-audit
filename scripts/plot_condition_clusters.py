from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
from sklearn.cluster import KMeans

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MPL_CACHE_DIR = PROJECT_ROOT / ".mpl_cache"
MPL_CACHE_DIR.mkdir(exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")

sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rul.cmapss import COLUMNS
from rul.config import load_config


def read_cmapss_train(data_dir: str | Path, subset: str) -> pd.DataFrame:
    path = Path(data_dir) / f"train_{subset}.txt"
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS, engine="python")


def plot_condition_clusters(df: pd.DataFrame, *, subset: str, n_clusters: int, output_path: str | Path) -> Path:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    settings = df[["setting_1", "setting_2", "setting_3"]].astype(float)
    labels = KMeans(n_clusters=int(n_clusters), n_init=10, random_state=42).fit_predict(settings)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(5.5, 4.2))
    scatter = ax.scatter(df["setting_1"], df["setting_2"], c=labels, s=8, cmap="tab10", alpha=0.65)
    ax.set_xlabel("setting_1")
    ax.set_ylabel("setting_2")
    ax.set_title(f"{subset} operating condition clusters")
    legend = ax.legend(*scatter.legend_elements(), title="cluster", loc="best", frameon=True)
    ax.add_artist(legend)
    fig.tight_layout()
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Plot C-MAPSS operating condition clusters for protocol v2.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "paper_main.yaml"))
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "raw"))
    parser.add_argument("--subsets", nargs="+", default=["FD002", "FD004"])
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "reports" / "figures"))
    args = parser.parse_args()

    cfg = load_config(args.config)
    clusters = cfg.get("data", {}).get("condition_clusters", {})
    for subset in args.subsets:
        n_clusters = int(clusters.get(subset, 6 if subset in {"FD002", "FD004"} else 1))
        df = read_cmapss_train(args.data_dir, subset)
        out = Path(args.output_dir) / f"condition_clusters_{subset}.png"
        plot_condition_clusters(df, subset=subset, n_clusters=n_clusters, output_path=out)
        print(f"Saved {out}")


if __name__ == "__main__":
    main()
