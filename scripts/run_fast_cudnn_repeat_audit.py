from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rul.config import load_config
from rul.training import train_model
from run_cross_backbone_protocol_buildup import MODELS, STAGES, staged_config
from run_formal_benchmark import existing_formal_benchmark_run_metrics


DEFAULT_STAGES = ("condition_normalization", "plus_risk_checkpoint")
DEFAULT_SEEDS = (42, 2024)


def repeat_run_dir(model: str, stage: str, seed: int, replicate: int) -> Path:
    if replicate == 0:
        return (
            PROJECT_ROOT
            / "results"
            / "cross_backbone_protocol_buildup"
            / f"protocol_buildup_{stage}_seed{seed}"
            / "FD004"
            / model
        )
    return (
        PROJECT_ROOT
        / "results"
        / "fast_cudnn_repeat_audit"
        / f"{model}_{stage}_seed{seed}_rep{replicate}"
        / "FD004"
        / model
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Repeat selected fast-cuDNN attribution cells at fixed seeds.")
    parser.add_argument("--models", nargs="+", choices=MODELS, default=list(MODELS))
    parser.add_argument("--stages", nargs="+", choices=STAGES, default=list(DEFAULT_STAGES))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--replicates", type=int, default=3)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun-complete", action="store_true")
    args = parser.parse_args()
    if args.replicates < 2:
        raise ValueError("At least two replicates are required for a rerun audit.")

    main_cfg = load_config(PROJECT_ROOT / "configs" / "paper_main.yaml")
    conventional_cfg = load_config(PROJECT_ROOT / "configs" / "standard_protocol_fd004.yaml")
    jobs = []
    for model in args.models:
        for stage in args.stages:
            for seed in args.seeds:
                for replicate in range(args.replicates):
                    cfg = staged_config(main_cfg, conventional_cfg, model, stage, seed)
                    cfg["training"]["deterministic_cudnn"] = False
                    cfg["project"]["results_dir"] = "results/fast_cudnn_repeat_audit"
                    cfg["project"]["experiment_name"] = f"{model}_{stage}_seed{seed}_rep{replicate}"
                    jobs.append((model, stage, seed, replicate, cfg, repeat_run_dir(model, stage, seed, replicate)))

    print(f"[FAST-CUDNN REPEAT AUDIT] trajectories={len(jobs)} new_runs={sum(r > 0 for *_, r, __, ___ in jobs)}")
    if args.dry_run:
        for model, stage, seed, replicate, _, path in jobs:
            print(f"- model={model} stage={stage} seed={seed} replicate={replicate} path={path}")
        print("FAST_CUDNN_REPEAT_DRY_RUN_PASS")
        return

    rows = []
    train_jobs = [job for job in jobs if job[3] > 0]
    train_index = 0
    for model, stage, seed, replicate, cfg, path in jobs:
        if replicate == 0:
            metrics_path = path / "metrics.json"
            if not metrics_path.exists():
                raise FileNotFoundError(f"Missing original attribution trajectory: {metrics_path}")
            row = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
            source = "original_fast_cudnn_grid"
        else:
            train_index += 1
            existing = existing_formal_benchmark_run_metrics(path, cfg)
            if existing is not None and not args.rerun_complete:
                print(
                    f"[FAST-CUDNN SKIP {train_index}/{len(train_jobs)}] "
                    f"{model} {stage} seed={seed} replicate={replicate}",
                    flush=True,
                )
                row = existing
            else:
                print(
                    f"[FAST-CUDNN RUN {train_index}/{len(train_jobs)}] "
                    f"{model} {stage} seed={seed} replicate={replicate}",
                    flush=True,
                )
                row = train_model(
                    cfg,
                    subset="FD004",
                    model_name=model,
                    run_index=train_index,
                    total_runs=len(train_jobs),
                )
            source = "fixed_seed_rerun"
        rows.append(
            {
                **row,
                "audit_model": model,
                "audit_stage": stage,
                "audit_seed": seed,
                "audit_replicate": replicate,
                "audit_source": source,
                "audit_run_dir": str(path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
            }
        )

    output = PROJECT_ROOT / "results" / "fast_cudnn_repeat_audit.json"
    output.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"FAST_CUDNN_REPEAT_READY trajectories={len(rows)} output={output}")


if __name__ == "__main__":
    main()
