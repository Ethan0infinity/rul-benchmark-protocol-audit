from __future__ import annotations

import argparse
import copy
import json
import shutil
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rul.config import load_config
from rul.training import train_model
from run_formal_benchmark import FORMAL_BENCHMARK_SEEDS, existing_formal_benchmark_run_metrics


MODELS = ("rast_gru", "transformer_lite")
STAGES = (
    "global_scaling",
    "condition_normalization",
    "plus_mask_channels",
    "plus_degradation_augmentation",
    "plus_asymmetric_base_loss",
    "plus_risk_checkpoint",
)


def staged_config(main: dict, conventional: dict, model: str, stage: str, seed: int) -> dict:
    if stage not in STAGES:
        raise ValueError(stage)
    cfg = copy.deepcopy(conventional)
    cfg["project"]["seed"] = int(seed)
    cfg["project"]["results_dir"] = "results/cross_backbone_protocol_buildup"
    cfg["project"]["experiment_name"] = f"protocol_buildup_{stage}_seed{seed}"
    cfg["model"] = copy.deepcopy(main["model"])
    cfg["model"]["name"] = model
    cfg["augmentation"]["seed"] = int(seed)
    cfg.setdefault("progress", {})["batch_bar"] = False

    stage_index = STAGES.index(stage)
    if stage_index >= 1:
        cfg["data"]["scaler"] = "condition_standard"
        cfg["data"]["condition_clusters"] = {"FD004": 6}
    if stage_index >= 2:
        cfg["data"]["append_missing_mask"] = True
    if stage_index >= 3:
        cfg["augmentation"] = copy.deepcopy(main["augmentation"])
        cfg["augmentation"]["seed"] = int(seed)
    if stage_index >= 4:
        for key in ("loss", "huber_delta", "late_life_threshold", "late_life_weight", "late_over_weight"):
            cfg["training"][key] = main["training"][key]
    if stage_index >= 5:
        cfg["training"]["selection_metric"] = main["training"]["selection_metric"]
        cfg["training"]["selection_lpr_weight"] = main["training"]["selection_lpr_weight"]
        cfg["training"]["selection_severe_late_weight"] = main["training"]["selection_severe_late_weight"]
    return cfg


def source_run(stage: str, model: str, seed: int) -> Path | None:
    if stage == "global_scaling":
        return PROJECT_ROOT / "results" / "standard_protocol_track" / f"standard_protocol_fd004_seed{seed}" / "FD004" / model
    if stage == "plus_risk_checkpoint":
        return PROJECT_ROOT / "results" / f"paper_main_v3_seed{seed}" / "FD004" / model
    return None


def materialize_reference(source: Path, target: Path, config: dict) -> None:
    required = ("metrics.json", "test_predictions.csv", "best_model.pt")
    missing = [name for name in required if not (source / name).exists()]
    if missing:
        raise FileNotFoundError(f"Cannot reuse {source}; missing {missing}")
    target.mkdir(parents=True, exist_ok=True)
    for item in source.iterdir():
        if item.is_file() and item.name != "run_config.yaml":
            shutil.copy2(item, target / item.name)
    # The staged config is the provenance record; the original source is retained explicitly.
    from yaml import safe_dump

    (target / "run_config.yaml").write_text(
        safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8"
    )
    (target / "reused_source.txt").write_text(
        str(source.relative_to(PROJECT_ROOT)).replace("\\", "/") + "\n", encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a cumulative FD004 protocol build-up on RAST-GRU and Transformer-lite."
    )
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(STAGES))
    parser.add_argument("--seeds", nargs="+", type=int, default=FORMAL_BENCHMARK_SEEDS)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Override the fixed batch size for a separately declared attribution grid.",
    )
    parser.add_argument(
        "--fast-cudnn",
        action="store_true",
        help="Use the same non-deterministic cuDNN autotuning policy for every stage; provenance is stored in each run config.",
    )
    parser.add_argument("--rerun-complete", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    main_cfg = load_config(PROJECT_ROOT / "configs" / "paper_main.yaml")
    conventional_cfg = load_config(PROJECT_ROOT / "configs" / "standard_protocol_fd004.yaml")
    jobs = []
    for model in args.models:
        for seed in args.seeds:
            for stage in args.stages:
                cfg = staged_config(main_cfg, conventional_cfg, model, stage, seed)
                if args.epochs is not None:
                    cfg["training"]["epochs"] = int(args.epochs)
                if args.batch_size is not None:
                    cfg["training"]["batch_size"] = int(args.batch_size)
                if args.fast_cudnn:
                    cfg["training"]["deterministic_cudnn"] = False
                run_dir = (
                    PROJECT_ROOT
                    / cfg["project"]["results_dir"]
                    / cfg["project"]["experiment_name"]
                    / "FD004"
                    / model
                )
                jobs.append((model, seed, stage, cfg, run_dir))

    print(f"[CROSS-BACKBONE BUILDUP] jobs={len(jobs)}", flush=True)
    if args.dry_run:
        for model, seed, stage, _, run_dir in jobs:
            reuse = source_run(stage, model, seed)
            print(f"- model={model} seed={seed} stage={stage} output={run_dir} reuse={reuse}")
        print("CROSS_BACKBONE_BUILDUP_DRY_RUN_PASS")
        return

    rows = []
    for index, (model, seed, stage, cfg, run_dir) in enumerate(jobs, 1):
        reference = (
            source_run(stage, model, seed)
            if args.epochs is None and args.batch_size is None and not args.fast_cudnn
            else None
        )
        if reference is not None and not args.rerun_complete:
            print(f"[BUILDUP REUSE {index}/{len(jobs)}] {model} seed={seed} stage={stage}", flush=True)
            materialize_reference(reference, run_dir, cfg)
            row = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8-sig"))
        else:
            existing = existing_formal_benchmark_run_metrics(run_dir, cfg)
            if existing is not None and not args.rerun_complete and args.epochs is None:
                print(f"[BUILDUP SKIP {index}/{len(jobs)}] {model} seed={seed} stage={stage}", flush=True)
                row = existing
            else:
                print(f"[BUILDUP RUN {index}/{len(jobs)}] {model} seed={seed} stage={stage}", flush=True)
                row = train_model(cfg, subset="FD004", model_name=model, run_index=index, total_runs=len(jobs))
        row.update({"buildup_model": model, "buildup_seed": seed, "buildup_stage": stage})
        rows.append(row)

    output = PROJECT_ROOT / "results" / "cross_backbone_protocol_buildup.json"
    output.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"CROSS_BACKBONE_BUILDUP_READY runs={len(rows)} output={output}")


if __name__ == "__main__":
    main()
