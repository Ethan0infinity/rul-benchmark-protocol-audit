from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_formal_benchmark import FORMAL_BENCHMARK_MODELS, FORMAL_BENCHMARK_SEEDS, FORMAL_BENCHMARK_SUBSETS


def check_checkpoint(path: Path) -> str | None:
    if not path.exists():
        return "missing"
    with path.open("rb") as fh:
        header = fh.read(3)
    if header.startswith(b"\xef\xbb\xbf"):
        return "UTF-8 BOM prefix indicates binary checkpoint corruption"
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:  # noqa: BLE001 - the gate must report every serialization failure.
        return f"torch.load failed: {type(exc).__name__}: {exc}"
    if not isinstance(payload, dict) or "model_state" not in payload:
        return "checkpoint payload lacks model_state"
    return None


def expected_checkpoint_paths(results_dir: Path) -> list[Path]:
    return [
        results_dir / f"paper_main_v3_seed{seed}" / subset / model / "best_model.pt"
        for seed in FORMAL_BENCHMARK_SEEDS
        for subset in FORMAL_BENCHMARK_SUBSETS
        for model in FORMAL_BENCHMARK_MODELS
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Load every formal benchmark checkpoint and reject binary corruption.")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    args = parser.parse_args()

    failures = []
    paths = expected_checkpoint_paths(Path(args.results_dir))
    for path in paths:
        issue = check_checkpoint(path)
        if issue:
            failures.append((path, issue))
    if failures:
        print(f"CHECKPOINT_INTEGRITY_FAIL total={len(paths)} failed={len(failures)}")
        for path, issue in failures[:25]:
            print(f"- {path}: {issue}")
        if len(failures) > 25:
            print(f"- ... {len(failures) - 25} additional failures")
        raise SystemExit(1)
    print(f"CHECKPOINT_INTEGRITY_PASS total={len(paths)}")


if __name__ == "__main__":
    main()
