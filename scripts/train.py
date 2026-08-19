from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rul.config import load_config
from rul.training import train_model


def main() -> None:
    parser = argparse.ArgumentParser(description="Train one C-MAPSS RUL model.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "default.yaml"))
    parser.add_argument("--subset", default=None, help="FD001, FD002, FD003, or FD004.")
    parser.add_argument(
        "--model",
        default=None,
        help=(
            "mlp, cnn1d, cnn_lstm, lstm, gru, attention_gru, bigru_attention, "
            "tcn, tcn_gru, transformer_lite, dual_attention_tcn, sensor_graph_gru, quantile_gru, "
            "rs_tcn_gru, rast_gru, rast_gru_v2."
        ),
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--no-batch-bar", action="store_true", help="Hide tqdm batch bars and keep one clear console line per epoch.")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.subset:
        cfg["data"]["subset"] = args.subset
    if args.model:
        cfg["model"]["name"] = args.model
    if args.epochs is not None:
        cfg["training"]["epochs"] = args.epochs
    if args.batch_size is not None:
        cfg["training"]["batch_size"] = args.batch_size
    if args.no_batch_bar:
        cfg.setdefault("progress", {})["batch_bar"] = False

    metrics = train_model(cfg, subset=args.subset, model_name=args.model)
    print("Training finished.", flush=True)
    print(
        f"subset={metrics['subset']} model={metrics['model']} "
        f"test_rmse={metrics['test_rmse']:.4f} test_mae={metrics['test_mae']:.4f}",
        flush=True,
    )
    print(f"run_dir={metrics.get('run_dir')}", flush=True)
    print(f"live_progress={Path(metrics.get('run_dir', '.')) / 'progress.json'}", flush=True)
    print(f"epoch_history={Path(metrics.get('run_dir', '.')) / 'epoch_history.csv'}", flush=True)


if __name__ == "__main__":
    main()
