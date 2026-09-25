"""Run the canonical 5 x 10 C-MAPSS split-by-training-stream crossing.

This is a separate, resume-safe family.  It deliberately does not reuse the
retrospective 3 x 3, preference, or fixed-split artifacts because those
families may differ in configuration, checkpoint selection, or provenance.

Design: 4 tasks x 5 split levels x 10 training streams x 2 configurations
(RAST-GRU and OCM-Asym) = 400 training runs.
"""

from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
from pathlib import Path
from typing import Any

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rul.config import load_config
from rul.training import train_model


SPLIT_LEVELS = (42, 123, 2024, 2025, 2026)
TRAINING_STREAMS = (31415, 27182, 16180, 14142, 17320, 42, 123, 2024, 2025, 2026)
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
CONFIGURATIONS = (("rast", "rast_gru"), ("asym", "rast_gru_v2"))
RESULTS_DIR = "results/full_5x10_crossing"
OUTPUT_DIR = PROJECT_ROOT / "paper_outputs" / "full_5x10_crossing"


def configure(
    cfg: dict[str, Any],
    *,
    split_level: int,
    stream: int,
    model: str,
    epochs: int,
    results_dir: str,
) -> dict[str, Any]:
    cfg["project"]["results_dir"] = results_dir
    cfg["project"]["experiment_name"] = (
        f"full5x10_{'asym' if model == 'rast_gru_v2' else 'rast'}_"
        f"{split_level}_{stream}"
    )
    cfg["project"]["seed"] = int(stream)
    cfg["project"]["split_seed"] = int(split_level)
    cfg["project"]["initialization_seed"] = int(stream)
    cfg["project"]["shuffle_seed"] = int(stream)
    cfg["augmentation"]["seed"] = int(stream)
    cfg["model"]["name"] = model
    cfg["training"]["epochs"] = int(epochs)
    cfg.setdefault("progress", {})["batch_bar"] = False
    return cfg


def jobs(base: dict[str, Any], epochs: int, results_dir: str) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for subset in SUBSETS:
        for split_level in SPLIT_LEVELS:
            for stream in TRAINING_STREAMS:
                for point, model in CONFIGURATIONS:
                    cfg = configure(
                        copy.deepcopy(base),
                        split_level=split_level,
                        stream=stream,
                        model=model,
                        epochs=epochs,
                        results_dir=results_dir,
                    )
                    run_dir = (
                        PROJECT_ROOT / RESULTS_DIR / cfg["project"]["experiment_name"]
                        / subset / model
                    )
                    output.append({
                        "subset": subset,
                        "split_level": split_level,
                        "training_stream": stream,
                        "point": point,
                        "model": model,
                        "config": cfg,
                        "run_dir": run_dir,
                    })
    return output


def config_fingerprint(cfg: dict[str, Any]) -> str:
    selected = {
        section: cfg.get(section, {})
        for section in ("project", "data", "model", "training", "augmentation")
    }
    payload = json.dumps(selected, sort_keys=True, ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def complete(job: dict[str, Any], epochs: int) -> bool:
    run_dir: Path = job["run_dir"]
    required = [
        run_dir / "metrics.json",
        run_dir / "run_config.yaml",
        run_dir / "best_model.pt",
        run_dir / "test_predictions.csv",
        run_dir / "epoch_history.csv",
    ]
    if not all(path.exists() for path in required):
        return False
    try:
        metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8-sig"))
        actual = load_config(run_dir / "run_config.yaml")
        checkpoint = torch.load(run_dir / "best_model.pt", map_location="cpu", weights_only=False)
    except Exception:
        return False
    if int(metrics.get("planned_epochs", -1)) != int(epochs):
        return False
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        return False
    expected = job["config"]
    return all(actual.get(section, {}) == expected.get(section, {}) for section in (
        "project", "data", "model", "training", "augmentation"
    ))


def write_manifest(rows: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = [
        "index", "subset", "split_level", "training_stream", "point", "model",
        "run_dir", "config_sha256", "status", "planned_epochs",
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "paper_main.yaml"))
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--smoke", action="store_true", help="Run one FD001 RAST job for two epochs.")
    parser.add_argument("--rerun-complete", action="store_true")
    args = parser.parse_args()

    base = load_config(args.config)
    results_dir = "results/full_5x10_smoke" if args.smoke else RESULTS_DIR
    scheduled = jobs(base, args.epochs, results_dir)
    if args.smoke:
        scheduled = [job for job in scheduled if job["subset"] == "FD001" and job["point"] == "rast" and job["split_level"] == 42 and job["training_stream"] == 31415][:1]
    manifest_name = "full_5x10_smoke_manifest.csv" if args.smoke else "full_5x10_run_manifest.csv"
    protocol_name = "full_5x10_smoke_protocol.json" if args.smoke else "full_5x10_protocol.json"
    manifest = OUTPUT_DIR / manifest_name
    protocol = OUTPUT_DIR / protocol_name
    protocol.parent.mkdir(parents=True, exist_ok=True)
    protocol.write_text(json.dumps({
        "design": "4 tasks x 5 split levels x 10 training streams x 2 configurations",
        "planned_runs": len(scheduled) if args.smoke else 400,
        "split_levels": list(SPLIT_LEVELS),
        "training_streams": list(TRAINING_STREAMS),
        "subsets": list(SUBSETS),
        "configurations": [{"point": p, "model": m} for p, m in CONFIGURATIONS],
        "checkpoint_selection": "val_last_risk_score",
        "epochs": args.epochs,
        "source_config": str(Path(args.config).resolve()),
        "evidence_role": "retrospective finite factorial descriptive audit",
        "artifact_reuse": "none; every cell has a dedicated run directory",
        "smoke": bool(args.smoke),
        "results_dir": results_dir,
    }, indent=2), encoding="utf-8")

    print(f"FULL_5X10 planned_runs={len(scheduled)} canonical_total=400 epochs={args.epochs}", flush=True)
    rows: list[dict[str, Any]] = []
    for index, job in enumerate(scheduled, start=1):
        run_dir: Path = job["run_dir"]
        status = "planned"
        if args.dry_run:
            print(f"- {index:03d}/{len(scheduled)} {job['point']} {job['subset']} split={job['split_level']} stream={job['training_stream']} -> {run_dir.relative_to(PROJECT_ROOT).as_posix()}", flush=True)
        elif not args.rerun_complete and complete(job, args.epochs):
            status = "reused"
            print(f"[FULL_5X10 SKIP {index}/{len(scheduled)}] {job['point']} {job['subset']} split={job['split_level']} stream={job['training_stream']}", flush=True)
        else:
            status = "completed"
            print(f"[FULL_5X10 RUN {index}/{len(scheduled)}] {job['point']} {job['subset']} split={job['split_level']} stream={job['training_stream']}", flush=True)
            train_model(job["config"], subset=job["subset"], model_name=job["model"], run_index=index, total_runs=len(scheduled))
        rows.append({
            "index": index,
            "subset": job["subset"],
            "split_level": job["split_level"],
            "training_stream": job["training_stream"],
            "point": job["point"],
            "model": job["model"],
            "run_dir": run_dir.relative_to(PROJECT_ROOT).as_posix(),
            "config_sha256": config_fingerprint(job["config"]),
            "status": status,
            "planned_epochs": args.epochs,
        })
        if not args.dry_run:
            write_manifest(rows, manifest)
    if args.dry_run:
        print(f"FULL_5X10_DRY_RUN_PASS jobs={len(scheduled)} manifest={manifest}")
    else:
        print(f"FULL_5X10_READY jobs={len(rows)} manifest={manifest}")


if __name__ == "__main__":
    main()
