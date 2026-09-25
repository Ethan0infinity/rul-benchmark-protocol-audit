from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from run_formal_benchmark import FORMAL_BENCHMARK_MODELS, FORMAL_BENCHMARK_SEEDS


def main() -> None:
    issues = []
    for seed in FORMAL_BENCHMARK_SEEDS:
        for model in FORMAL_BENCHMARK_MODELS:
            run_dir = PROJECT_ROOT / "results" / f"ncmapss_ds02_seed{seed}" / "DS02" / model
            metrics_path = run_dir / "metrics.json"
            checkpoint_path = run_dir / "best_model.pt"
            if not metrics_path.exists() or not checkpoint_path.exists():
                issues.append(f"missing {seed}/{model}")
                continue
            try:
                metrics = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
                checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            except Exception as exc:  # noqa: BLE001
                issues.append(f"invalid {seed}/{model}: {exc}")
                continue
            if metrics.get("result_status") != "external_validation" or int(metrics.get("planned_epochs", 0)) < 80:
                issues.append(f"incomplete {seed}/{model}")
            if metrics.get("checkpoint_selection_metric") != "val_risk_score":
                issues.append(f"old selection protocol {seed}/{model}")
            if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
                issues.append(f"invalid checkpoint payload {seed}/{model}")
    expected = len(FORMAL_BENCHMARK_MODELS) * len(FORMAL_BENCHMARK_SEEDS)
    if issues:
        print(f"NCMAPSS_READINESS_FAIL expected={expected} issues={len(issues)}")
        for issue in issues[:30]:
            print(f"- {issue}")
        raise SystemExit(1)
    print(f"NCMAPSS_READINESS_PASS runs={expected}")


if __name__ == "__main__":
    main()
