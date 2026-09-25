from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rul.config import load_config
from rul.ncmapss import load_prepared_cache
from rul.utils import get_device, save_json
from prepare_ncmapss_dev_rotation import DEV_UNITS, rotation_split
from run_ncmapss_benchmark import checkpoint_is_valid, train_one


POINTS = {
    "RAST": ("rast_gru", 1.5),
    "Core": ("rast_gru_v2", 1.0),
    "Asym": ("rast_gru_v2", 1.5),
}
SEEDS = (42, 123, 2024)


def cache_path(cache_dir: Path, test_unit: int) -> Path:
    return cache_dir / f"ncmapss_ds02_dev_test{test_unit}_smp100_win50.npz"


def experiment_name(point: str, test_unit: int, seed: int, epochs: int) -> str:
    suffix = f"_smoke{epochs}" if epochs < 80 else ""
    return f"ncmapss_dev_rotation_{point.lower()}_test{test_unit}_seed{seed}{suffix}"


def write_manifest(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "point", "test_unit", "validation_unit", "training_seed", "model", "run_dir", "cache", "status"
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run RAST/Core/Asym on six N-CMAPSS development-unit rotations."
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "paper_main.yaml"))
    parser.add_argument(
        "--cache-dir",
        default=str(PROJECT_ROOT / "data" / "external" / "processed" / "ncmapss_dev_rotation"),
    )
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--points", nargs="+", choices=tuple(POINTS), default=list(POINTS))
    parser.add_argument("--test-units", nargs="+", type=int, choices=DEV_UNITS, default=list(DEV_UNITS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(SEEDS))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=12)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--rerun-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base = load_config(args.config)
    cache_dir = Path(args.cache_dir)
    results_dir = Path(args.results_dir)
    jobs = [(point, test_unit, seed) for point in args.points for test_unit in args.test_units for seed in args.seeds]
    manifest = PROJECT_ROOT / "results" / "ncmapss_dev_rotation" / (
        "smoke_manifest.csv" if args.epochs < 80 else "run_manifest.csv"
    )
    print(
        f"[N-CMAPSS DEV ROTATION] jobs={len(jobs)} points={args.points} "
        f"test_units={args.test_units} seeds={args.seeds}",
        flush=True,
    )
    rows = []
    device = get_device(args.device)
    prepared_cache: dict[int, object] = {}
    for index, (point, test_unit, seed) in enumerate(jobs, 1):
        model, late_weight = POINTS[point]
        _, validation_units, _ = rotation_split(test_unit)
        cache = cache_path(cache_dir, test_unit)
        name = experiment_name(point, test_unit, seed, args.epochs)
        run_dir = results_dir / name / "DS02" / model
        if args.dry_run:
            print(
                f"- {index:02d}/{len(jobs)} {point} test={test_unit} "
                f"validation={validation_units[0]} seed={seed}: {run_dir}",
                flush=True,
            )
            continue
        if not cache.exists():
            raise FileNotFoundError(f"Missing rotation cache: {cache}")
        if not args.rerun_complete and checkpoint_is_valid(run_dir):
            metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8-sig"))
            status = "reused"
            print(f"[N-CMAPSS ROTATION SKIP {index}/{len(jobs)}] {point} test={test_unit} seed={seed}", flush=True)
        else:
            if test_unit not in prepared_cache:
                prepared_cache[test_unit] = load_prepared_cache(cache)
            cfg = copy.deepcopy(base)
            cfg["project"]["seed"] = int(seed)
            cfg["project"]["experiment_name"] = name
            cfg["model"]["name"] = model
            cfg["training"]["late_over_weight"] = float(late_weight)
            cfg["augmentation"]["seed"] = int(seed)
            print(f"[N-CMAPSS ROTATION RUN {index}/{len(jobs)}] {point} test={test_unit} seed={seed}", flush=True)
            metrics = train_one(
                prepared_cache[test_unit],
                model_name=model,
                seed=seed,
                run_dir=run_dir,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                device=device,
                protocol_config=cfg,
                experiment_name=name,
            )
            status = "completed"
        metrics.update(
            {
                "result_status": "exploratory_dev_unit_rotation",
                "evidence_level": "exploratory_dev_unit_rotation",
                "operating_point": point,
                "rotation_test_unit": int(test_unit),
                "rotation_validation_unit": int(validation_units[0]),
                "test_source": "development_unit",
            }
        )
        save_json(metrics, run_dir / "metrics.json")
        rows.append(
            {
                "point": point,
                "test_unit": test_unit,
                "validation_unit": validation_units[0],
                "training_seed": seed,
                "model": model,
                "run_dir": run_dir.relative_to(PROJECT_ROOT).as_posix(),
                "cache": cache.relative_to(PROJECT_ROOT).as_posix(),
                "status": status,
            }
        )
        write_manifest(rows, manifest)

    if args.dry_run:
        print("NCMAPSS_DEV_ROTATION_GRID_DRY_RUN_PASS")
    else:
        print(f"NCMAPSS_DEV_ROTATION_GRID_READY rows={len(rows)} manifest={manifest}")


if __name__ == "__main__":
    main()
