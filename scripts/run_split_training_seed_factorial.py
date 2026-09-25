from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rul.config import load_config
from rul.training import train_model


POINTS = {
    "RAST": ("rast_gru", 1.5),
    "Core": ("rast_gru_v2", 1.0),
    "Asym": ("rast_gru_v2", 1.5),
}
DEFAULT_SPLIT_SEEDS = (42, 123, 2024)
DEFAULT_TRAINING_SEEDS = (42, 123, 2024)


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


def experiment_name(point: str, split_seed: int, training_seed: int, epochs: int | None) -> str:
    suffix = f"_smoke{epochs}" if epochs is not None else ""
    return f"seed_factorial_fd004_{point.lower()}_split{split_seed}_train{training_seed}{suffix}"


def make_config(
    base: dict,
    point: str,
    split_seed: int,
    training_seed: int,
    epochs: int | None,
) -> tuple[dict, str]:
    model, late_weight = POINTS[point]
    cfg = copy.deepcopy(base)
    cfg["project"]["seed"] = int(training_seed)
    cfg["project"]["split_seed"] = int(split_seed)
    cfg["project"]["initialization_seed"] = int(training_seed)
    cfg["project"].pop("shuffle_seed", None)
    cfg["project"]["experiment_name"] = experiment_name(
        point, split_seed, training_seed, epochs
    )
    cfg["model"]["name"] = model
    cfg["training"]["late_over_weight"] = float(late_weight)
    if epochs is not None:
        cfg["training"]["epochs"] = int(epochs)
    cfg["augmentation"]["seed"] = int(training_seed)
    cfg.setdefault("progress", {})["batch_bar"] = False
    return cfg, model


def configured_run_dir(cfg: dict, model: str) -> Path:
    return (
        PROJECT_ROOT
        / cfg["project"].get("results_dir", "results")
        / cfg["project"]["experiment_name"]
        / "FD004"
        / model
    )


def complete(run_dir: Path, cfg: dict) -> bool:
    required = ("metrics.json", "run_config.yaml", "best_model.pt", "test_predictions.csv")
    if not all((run_dir / name).exists() for name in required):
        return False
    actual = load_config(run_dir / "run_config.yaml")
    keys = (
        ("project", "split_seed"),
        ("project", "initialization_seed"),
        ("model", "name"),
        ("training", "late_over_weight"),
        ("augmentation", "seed"),
    )
    return all(actual.get(section, {}).get(key) == cfg.get(section, {}).get(key) for section, key in keys)


def write_manifest(rows: list[dict], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "point",
        "split_seed",
        "training_seed",
        "model",
        "run_dir",
        "source",
        "status",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def validate_locked_result(run_dir: Path, seed: int) -> None:
    for name in ("metrics.json", "run_config.yaml", "best_model.pt", "test_predictions.csv"):
        if not (run_dir / name).exists():
            raise FileNotFoundError(f"Locked diagonal result is incomplete: {run_dir / name}")
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8-sig"))
    if int(metrics["seed"]) != int(seed):
        raise ValueError(f"Locked result seed mismatch in {run_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a small FD004 split-seed by training-seed factorial audit."
    )
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "paper_main.yaml"))
    parser.add_argument("--points", nargs="+", choices=tuple(POINTS), default=list(POINTS))
    parser.add_argument("--split-seeds", nargs="+", type=int, default=list(DEFAULT_SPLIT_SEEDS))
    parser.add_argument(
        "--training-seeds", nargs="+", type=int, default=list(DEFAULT_TRAINING_SEEDS)
    )
    parser.add_argument("--epochs", type=int, default=None, help="Use only for an isolated smoke run.")
    parser.add_argument("--rerun-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base = load_config(args.config)
    jobs = [
        (point, split_seed, training_seed)
        for point in args.points
        for split_seed in args.split_seeds
        for training_seed in args.training_seeds
    ]
    manifest = PROJECT_ROOT / "results" / "seed_factorial_fd004" / (
        "smoke_manifest.csv" if args.epochs is not None else "run_manifest.csv"
    )
    print(
        f"[SEED FACTORIAL] jobs={len(jobs)} points={args.points} "
        f"split_seeds={args.split_seeds} training_seeds={args.training_seeds}",
        flush=True,
    )
    rows: list[dict] = []
    for index, (point, split_seed, training_seed) in enumerate(jobs, 1):
        cfg, model = make_config(base, point, split_seed, training_seed, args.epochs)
        diagonal = args.epochs is None and split_seed == training_seed
        run_dir = locked_run_dir(point, split_seed) if diagonal else configured_run_dir(cfg, model)
        source = "locked composite-seed diagonal" if diagonal else "factorial audit run"
        status = "planned"
        if args.dry_run:
            print(
                f"- {index:02d}/{len(jobs)} {point} split={split_seed} "
                f"training={training_seed}: {run_dir}",
                flush=True,
            )
        elif diagonal:
            validate_locked_result(run_dir, split_seed)
            status = "reused"
            print(
                f"[FACTORIAL REUSE {index}/{len(jobs)}] {point} "
                f"split={split_seed} training={training_seed}",
                flush=True,
            )
        elif not args.rerun_complete and complete(run_dir, cfg):
            status = "reused"
            print(
                f"[FACTORIAL SKIP {index}/{len(jobs)}] {point} "
                f"split={split_seed} training={training_seed}",
                flush=True,
            )
        else:
            print(
                f"[FACTORIAL RUN {index}/{len(jobs)}] {point} "
                f"split={split_seed} training={training_seed}",
                flush=True,
            )
            train_model(cfg, subset="FD004", model_name=model, run_index=index, total_runs=len(jobs))
            status = "completed"
        rows.append(
            {
                "point": point,
                "split_seed": split_seed,
                "training_seed": training_seed,
                "model": model,
                "run_dir": run_dir.relative_to(PROJECT_ROOT).as_posix(),
                "source": source,
                "status": status,
            }
        )
        if not args.dry_run:
            write_manifest(rows, manifest)

    if args.dry_run:
        print("SEED_FACTORIAL_DRY_RUN_PASS")
    else:
        print(f"SEED_FACTORIAL_READY rows={len(rows)} manifest={manifest}")


if __name__ == "__main__":
    main()
