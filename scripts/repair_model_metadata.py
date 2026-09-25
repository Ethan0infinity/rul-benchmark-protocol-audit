from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rul.config import load_config
from rul.training import model_feature_metadata


def corrected_payload(metrics_path: Path) -> tuple[dict, dict[str, tuple[object, object]]]:
    payload = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
    config_path = metrics_path.parent / "run_config.yaml"
    if not config_path.exists():
        return payload, {}
    config = load_config(config_path)
    model_name = str(payload.get("model", config.get("model", {}).get("name", "")))
    append_mask = bool(config.get("data", {}).get("append_missing_mask", False))
    expected = model_feature_metadata(model_name, config.get("model", {}), append_mask)
    changes = {
        key: (payload.get(key), value)
        for key, value in expected.items()
        if payload.get(key) != value
    }
    payload.update(expected)
    return payload, changes


def main() -> None:
    parser = argparse.ArgumentParser(description="Correct architecture metadata without changing predictions or metrics.")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    changed_files = 0
    changed_fields = 0
    for metrics_path in sorted(Path(args.results_dir).rglob("metrics.json")):
        payload, changes = corrected_payload(metrics_path)
        if not changes:
            continue
        changed_files += 1
        changed_fields += len(changes)
        print(f"[MODEL METADATA] {metrics_path.relative_to(PROJECT_ROOT)} fields={sorted(changes)}")
        if not args.dry_run:
            metrics_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    mode = "DRY_RUN" if args.dry_run else "PASS"
    print(f"MODEL_METADATA_REPAIR_{mode} files={changed_files} fields={changed_fields}")


if __name__ == "__main__":
    main()
