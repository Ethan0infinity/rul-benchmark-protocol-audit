from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rul.ncmapss import prepare_ncmapss_ds02, save_prepared_cache


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a leakage-safe sampled N-CMAPSS DS02 window cache.")
    parser.add_argument(
        "--input",
        default=str(PROJECT_ROOT / "data" / "external" / "raw" / "N-CMAPSS_DS02-006.h5"),
    )
    parser.add_argument(
        "--output",
        default=str(PROJECT_ROOT / "data" / "external" / "processed" / "ncmapss_ds02_smp100_win50.npz"),
    )
    parser.add_argument("--sampling", type=int, default=100)
    parser.add_argument("--window-size", type=int, default=50)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--condition-clusters", type=int, default=6)
    args = parser.parse_args()

    print("[N-CMAPSS PREP] reading sampled development/test arrays", flush=True)
    prepared = prepare_ncmapss_ds02(
        args.input,
        sampling=args.sampling,
        window_size=args.window_size,
        stride=args.stride,
        condition_clusters=args.condition_clusters,
    )
    save_prepared_cache(prepared, args.output)
    report = {
        **prepared.metadata,
        "train_windows": int(len(prepared.train.y)),
        "validation_windows": int(len(prepared.val.y)),
        "test_windows": int(len(prepared.test.y)),
        "input_features_without_mask": int(prepared.train.x.shape[-1]),
        "cache": Path(args.output).relative_to(PROJECT_ROOT).as_posix(),
    }
    report_path = PROJECT_ROOT / "reports" / "ncmapss_ds02_preparation.json"
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(
        "NCMAPSS_CACHE_READY "
        f"train={report['train_windows']} val={report['validation_windows']} test={report['test_windows']} "
        f"features={report['input_features_without_mask']}"
    )


if __name__ == "__main__":
    main()
