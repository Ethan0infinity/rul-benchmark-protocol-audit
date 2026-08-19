from __future__ import annotations

import csv
import hashlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rul.config import load_config
from rul.models import build_model


MODEL_NAMES = [
    "mlp",
    "cnn1d",
    "lstm",
    "gru",
    "bigru",
    "tcn",
    "tcn_gru",
    "cnn_lstm",
    "attention_gru",
    "bigru_attention",
    "transformer_lite",
    "dual_attention_tcn",
    "regime_dual_attention_cnn_gru",
    "sensor_graph_gru",
    "quantile_gru",
    "official_dual_mixer",
    "rs_tcn_gru",
    "rast_gru",
    "rast_gru_v2",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def verify_configs() -> None:
    config_dir = PROJECT_ROOT / "configs"
    for name in ["default.yaml", "paper_main.yaml"]:
        config = load_config(config_dir / name)
        if not isinstance(config, dict) or not config:
            raise RuntimeError(f"Configuration is empty: {name}")

    protocol_path = config_dir / "protocol_v3.lock.yaml"
    with protocol_path.open("r", encoding="utf-8") as handle:
        protocol = yaml.safe_load(handle)
    if not isinstance(protocol, dict) or not protocol:
        raise RuntimeError("Protocol lock is empty")


def verify_models() -> None:
    torch.manual_seed(0)
    sample = torch.randn(3, 30, 12)
    for name in MODEL_NAMES:
        model = build_model(
            name,
            input_features=12,
            window_size=30,
            hidden_size=16,
            tcn_channels=[8, 16],
            kernel_size=3,
            dropout=0.0,
        )
        model.eval()
        with torch.inference_mode():
            prediction = model(sample)
        if prediction.shape != (3,) or not torch.isfinite(prediction).all():
            raise RuntimeError(f"Invalid forward result for model: {name}")


def verify_release_evidence() -> int:
    release_dir = PROJECT_ROOT / "paper_outputs" / "release_v1"
    manifest_path = release_dir / "release_manifest.csv"
    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError("Release manifest contains no artifacts")

    for row in rows:
        path = release_dir / row["release_path"]
        if not path.is_file():
            raise FileNotFoundError(f"Missing release artifact: {path.name}")
        expected_hash = row.get("release_sha256", "").strip().lower()
        if expected_hash and sha256(path) != expected_hash:
            raise RuntimeError(f"SHA-256 mismatch: {path.name}")

        expected_rows = row.get("row_count", "").strip()
        if expected_rows and path.suffix.lower() == ".csv":
            observed_rows = len(pd.read_csv(path))
            if observed_rows != int(expected_rows):
                raise RuntimeError(
                    f"Row-count mismatch for {path.name}: "
                    f"expected {expected_rows}, observed {observed_rows}"
                )
    return len(rows)


def main() -> None:
    verify_configs()
    verify_models()
    artifact_count = verify_release_evidence()
    print(f"Python: {sys.version.split()[0]}")
    print(f"NumPy: {np.__version__}")
    print(f"pandas: {pd.__version__}")
    print(f"PyTorch: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Supported model forward checks: {len(MODEL_NAMES)}")
    print(f"Verified compact evidence artifacts: {artifact_count}")
    print("MINIMAL_REPRODUCIBILITY_CHECK_PASS")


if __name__ == "__main__":
    main()
