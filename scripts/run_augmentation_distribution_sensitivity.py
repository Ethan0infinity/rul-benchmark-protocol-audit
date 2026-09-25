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
from rul.training import train_model
from run_formal_benchmark import FORMAL_BENCHMARK_SEEDS


POINTS = ("RAST", "Core", "Asym")
CONDITIONS = {
    "p0": (0.0, 1.0, "no primary degradation augmentation; OCM consistency view unchanged"),
    "p025": (0.25, 1.0, "independent mechanism probability 0.25"),
    "p05": (0.5, 1.0, "independent mechanism probability 0.50"),
    "p1": (1.0, 1.0, "locked composite corruption on every window"),
    "mix50": (1.0, 0.5, "50% clean and 50% composite-corrupted windows"),
}


def locked_run_dir(point: str, seed: int) -> Path:
    if point == "RAST":
        return PROJECT_ROOT / "results" / f"paper_main_v3_seed{seed}" / "FD004" / "rast_gru"
    if point == "Asym":
        return PROJECT_ROOT / "results" / f"paper_main_v3_seed{seed}" / "FD004" / "rast_gru_v2"
    if point == "Core":
        return (
            PROJECT_ROOT
            / "results"
            / f"ablation_fd004_seed{seed}_weighted_huber_no_asymmetry"
            / "FD004"
            / "rast_gru_v2"
        )
    raise ValueError(point)


def make_config(base: dict, point: str, condition: str, seed: int, epochs: int | None) -> dict:
    mechanism_probability, window_probability, _ = CONDITIONS[condition]
    cfg = copy.deepcopy(base)
    model = "rast_gru" if point == "RAST" else "rast_gru_v2"
    suffix = f"_smoke{epochs}" if epochs is not None else ""
    cfg["project"]["seed"] = int(seed)
    cfg["project"]["results_dir"] = "results/augmentation_distribution_sensitivity"
    cfg["project"]["experiment_name"] = (
        f"augmentation_distribution_fd004_{point.lower()}_{condition}_seed{seed}{suffix}"
    )
    cfg["model"]["name"] = model
    cfg["training"]["late_over_weight"] = 1.0 if point == "Core" else 1.5
    if epochs is not None:
        cfg["training"]["epochs"] = int(epochs)
    cfg["augmentation"]["seed"] = int(seed)
    cfg["augmentation"]["mechanism_apply_probability"] = float(mechanism_probability)
    cfg["augmentation"]["corrupted_window_probability"] = float(window_probability)
    cfg.setdefault("progress", {})["batch_bar"] = False
    return cfg


def configured_run_dir(cfg: dict) -> Path:
    return (
        PROJECT_ROOT
        / cfg["project"]["results_dir"]
        / cfg["project"]["experiment_name"]
        / "FD004"
        / cfg["model"]["name"]
    )


def completed(run_dir: Path, cfg: dict) -> bool:
    metrics_path = run_dir / "metrics.json"
    config_path = run_dir / "run_config.yaml"
    checkpoint_path = run_dir / "best_model.pt"
    if not (metrics_path.exists() and config_path.exists() and checkpoint_path.exists()):
        return False
    actual = load_config(config_path)
    return all(actual.get(section, {}) == cfg.get(section, {}) for section in ("data", "model", "training", "augmentation"))


def write_manifest(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "point", "condition", "training_seed", "model", "run_dir", "source", "status",
        "mechanism_apply_probability", "corrupted_window_probability", "description",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the FD004 augmentation-distribution sensitivity grid.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "paper_main.yaml"))
    parser.add_argument("--points", nargs="+", choices=POINTS, default=list(POINTS))
    parser.add_argument("--conditions", nargs="+", choices=tuple(CONDITIONS), default=list(CONDITIONS))
    parser.add_argument("--seeds", nargs="+", type=int, default=FORMAL_BENCHMARK_SEEDS)
    parser.add_argument("--epochs", type=int, default=None, help="Use only for an isolated smoke run.")
    parser.add_argument("--rerun-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base = load_config(args.config)
    jobs = [(point, condition, seed) for point in args.points for condition in args.conditions for seed in args.seeds]
    print(
        f"[AUGMENTATION DISTRIBUTION GRID] jobs={len(jobs)} points={args.points} "
        f"conditions={args.conditions} seeds={args.seeds}",
        flush=True,
    )
    rows: list[dict] = []
    manifest = PROJECT_ROOT / "results" / "augmentation_distribution_sensitivity" / (
        "smoke_manifest.csv" if args.epochs is not None else "run_manifest.csv"
    )
    for index, (point, condition, seed) in enumerate(jobs, start=1):
        mechanism_probability, window_probability, description = CONDITIONS[condition]
        cfg = make_config(base, point, condition, seed, args.epochs)
        model = cfg["model"]["name"]
        reused_locked = args.epochs is None and condition == "p1"
        run_dir = locked_run_dir(point, seed) if reused_locked else configured_run_dir(cfg)
        source = "locked formal result" if reused_locked else "augmentation sensitivity run"
        status = "planned"
        if args.dry_run:
            print(f"- {index:02d}/{len(jobs)} {point} {condition} seed={seed}: {run_dir}")
        elif reused_locked:
            if not (run_dir / "metrics.json").exists() or not (run_dir / "best_model.pt").exists():
                raise FileNotFoundError(f"Locked p=1 result is incomplete: {run_dir}")
            status = "reused"
            print(f"[AUGMENTATION REUSE {index}/{len(jobs)}] {point} {condition} seed={seed}", flush=True)
        elif not args.rerun_complete and completed(run_dir, cfg):
            status = "reused"
            print(f"[AUGMENTATION SKIP {index}/{len(jobs)}] {point} {condition} seed={seed}", flush=True)
        else:
            print(f"[AUGMENTATION RUN {index}/{len(jobs)}] {point} {condition} seed={seed}", flush=True)
            train_model(cfg, subset="FD004", model_name=model, run_index=index, total_runs=len(jobs))
            status = "completed"
        rows.append(
            {
                "point": point,
                "condition": condition,
                "training_seed": seed,
                "model": model,
                "run_dir": run_dir.relative_to(PROJECT_ROOT).as_posix(),
                "source": source,
                "status": status,
                "mechanism_apply_probability": mechanism_probability,
                "corrupted_window_probability": window_probability,
                "description": description,
            }
        )
        if not args.dry_run:
            write_manifest(rows, manifest)

    if args.dry_run:
        print("AUGMENTATION_DISTRIBUTION_DRY_RUN_PASS")
        return
    print(f"AUGMENTATION_DISTRIBUTION_GRID_READY rows={len(rows)} manifest={manifest}")


if __name__ == "__main__":
    main()
