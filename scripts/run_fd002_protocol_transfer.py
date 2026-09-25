from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rul.config import load_config
from rul.training import train_model
from run_formal_benchmark import FORMAL_BENCHMARK_SEEDS, existing_formal_benchmark_run_metrics


POINTS = {
    "rast_core": ("rast_gru", 1.0),
    "transformer_core": ("transformer_lite", 1.0),
    "ocm_core": ("rast_gru_v2", 1.0),
    "rast_asym": ("rast_gru", 1.5),
    "ocm_asym": ("rast_gru_v2", 1.5),
}
NEW_POINTS = {"rast_core", "transformer_core"}


def transfer_config(base: dict, point: str, seed: int, epochs: int | None = None) -> dict:
    model, late_weight = POINTS[point]
    cfg = copy.deepcopy(base)
    cfg["project"]["seed"] = int(seed)
    cfg["project"]["results_dir"] = "results/fd002_protocol_transfer"
    cfg["project"]["experiment_name"] = f"fd002_protocol_transfer_{point}_seed{seed}"
    cfg["data"]["subset"] = "FD002"
    cfg["model"]["name"] = model
    cfg["training"]["late_over_weight"] = float(late_weight)
    cfg["augmentation"]["seed"] = int(seed)
    cfg.setdefault("progress", {})["batch_bar"] = False
    if epochs is not None:
        cfg["training"]["epochs"] = int(epochs)
    return cfg


def source_dir(point: str, seed: int) -> Path | None:
    model, _ = POINTS[point]
    if point == "ocm_core":
        return PROJECT_ROOT / "results" / f"core_ocm_seed{seed}" / "FD002" / model
    if point in {"rast_asym", "ocm_asym"}:
        return PROJECT_ROOT / "results" / f"paper_main_v3_seed{seed}" / "FD002" / model
    return None


def output_dir(point: str, seed: int) -> Path:
    model, _ = POINTS[point]
    return (
        PROJECT_ROOT
        / "results"
        / "fd002_protocol_transfer"
        / f"fd002_protocol_transfer_{point}_seed{seed}"
        / "FD002"
        / model
    )


def verified_reusable_source(source: Path, expected: dict) -> bool:
    if existing_formal_benchmark_run_metrics(source) is None:
        return False
    actual = load_config(source / "run_config.yaml")
    expected_data = copy.deepcopy(expected["data"])
    actual_data = copy.deepcopy(actual["data"])
    # Historical formal runs leave this convenience field at the base-config
    # value; train_model receives the actual subset explicitly.
    expected_data.pop("subset", None)
    actual_data.pop("subset", None)
    expected_augmentation = copy.deepcopy(expected["augmentation"])
    actual_augmentation = copy.deepcopy(actual["augmentation"])
    # These gates were added to the serialized configuration after the Core
    # runs; their runtime defaults were already 1.0.
    for key in ("mechanism_apply_probability", "corrupted_window_probability"):
        expected_augmentation.setdefault(key, 1.0)
        actual_augmentation.setdefault(key, 1.0)
    return (
        actual_data == expected_data
        and actual.get("model", {}) == expected.get("model", {})
        and actual.get("training", {}) == expected.get("training", {})
        and actual_augmentation == expected_augmentation
    )


def write_manifest(rows: list[dict]) -> Path:
    output = PROJECT_ROOT / "results" / "fd002_protocol_transfer" / "run_manifest.csv"
    output.parent.mkdir(parents=True, exist_ok=True)
    fields = ["point", "seed", "model", "late_over_weight", "status", "run_dir", "source_run_dir"]
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows([{key: row.get(key, "") for key in fields} for row in rows])
    return output


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the matched five-seed FD002 protocol-transfer grid. Three points reuse verified "
            "post-lock/formal runs; only RAST-Core and Transformer-Core require new training."
        )
    )
    parser.add_argument("--points", nargs="+", choices=tuple(POINTS), default=list(POINTS))
    parser.add_argument("--seeds", nargs="+", type=int, default=FORMAL_BENCHMARK_SEEDS)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--rerun-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base = load_config(PROJECT_ROOT / "configs" / "paper_main.yaml")
    jobs = [(point, seed, transfer_config(base, point, seed, args.epochs)) for point in args.points for seed in args.seeds]
    new_count = sum(point in NEW_POINTS for point, _, _ in jobs)
    print(f"[FD002 PROTOCOL TRANSFER] jobs={len(jobs)} new_training={new_count} reusable={len(jobs)-new_count}", flush=True)

    rows: list[dict] = []
    for index, (point, seed, cfg) in enumerate(jobs, 1):
        model, late_weight = POINTS[point]
        source = source_dir(point, seed) if args.epochs is None else None
        target = output_dir(point, seed)
        selected = source if source is not None else target
        source_text = source.relative_to(PROJECT_ROOT).as_posix() if source is not None else ""
        status = "planned"
        if source is not None:
            if not verified_reusable_source(source, cfg):
                raise RuntimeError(f"Reusable source does not match the declared protocol: {source}")
            status = "verified_reuse"
        elif existing_formal_benchmark_run_metrics(target, cfg) is not None and args.epochs is None and not args.rerun_complete:
            status = "existing_complete"
        rows.append(
            {
                "point": point,
                "seed": seed,
                "model": model,
                "late_over_weight": late_weight,
                "status": status,
                "run_dir": selected.relative_to(PROJECT_ROOT).as_posix(),
                "source_run_dir": source_text,
            }
        )
        print(f"- [{index:02d}/{len(jobs):02d}] point={point} seed={seed} status={status} run={selected}", flush=True)

    if args.dry_run:
        print("FD002_PROTOCOL_TRANSFER_DRY_RUN_PASS")
        return

    for index, (point, seed, cfg) in enumerate(jobs, 1):
        row = rows[index - 1]
        if row["status"] in {"verified_reuse", "existing_complete"}:
            print(f"[FD002 TRANSFER SKIP {index}/{len(jobs)}] point={point} seed={seed} status={row['status']}", flush=True)
            continue
        model, _ = POINTS[point]
        print(f"[FD002 TRANSFER RUN {index}/{len(jobs)}] point={point} seed={seed}", flush=True)
        metrics = train_model(cfg, subset="FD002", model_name=model, run_index=index, total_runs=len(jobs))
        row["status"] = "trained"
        row["run_dir"] = Path(metrics["run_dir"]).resolve().relative_to(PROJECT_ROOT).as_posix()
        write_manifest(rows)

    manifest = write_manifest(rows)
    summary = PROJECT_ROOT / "results" / "fd002_protocol_transfer" / "run_summary.json"
    summary.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"FD002_PROTOCOL_TRANSFER_READY rows={len(rows)} manifest={manifest}")


if __name__ == "__main__":
    main()
