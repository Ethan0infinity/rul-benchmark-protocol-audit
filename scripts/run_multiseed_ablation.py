from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rul.config import load_config
from rul.experiments import metrics_rows_to_csv
from rul.training import train_model
from run_ablation import make_variant
from run_formal_benchmark import FORMAL_BENCHMARK_SEEDS


MAJOR_FD004_VARIANTS = [
    "full",
    "no_condition_norm",
    "no_missing_mask",
    "no_reliability_gate",
    "no_channel_gate",
    "no_temporal_attention",
    "no_local_trend",
    "no_degradation_aug",
    "weighted_huber_no_asymmetry",
    "no_smooth_late_risk",
    "no_reliability_supervision",
    "no_consistency_regularization",
    "no_quantile_calibration",
]

MAIN_EXPERIMENT_PREFIX = "paper_main_v3_seed"
PROTOCOL_SECTIONS = ("data", "model", "training", "augmentation")
IGNORED_PROTOCOL_KEYS = {"data": {"subsets", "subset"}}


def _canonical_protocol_section(section: str, values: dict) -> dict:
    ignored = IGNORED_PROTOCOL_KEYS.get(section, set())
    return {key: value for key, value in values.items() if key not in ignored}


def _protocol_matches(run_dir: Path, expected_config: dict) -> bool:
    config_path = run_dir / "run_config.yaml"
    if not config_path.exists():
        return False
    actual = load_config(config_path)
    return all(
        _canonical_protocol_section(section, actual.get(section, {}))
        == _canonical_protocol_section(section, expected_config.get(section, {}))
        for section in PROTOCOL_SECTIONS
    )


def _existing_metrics(run_dir: Path, expected_config: dict) -> dict | None:
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        return None
    try:
        row = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
    except json.JSONDecodeError:
        return None
    if row.get("result_status") not in {"formal", "supplementary"}:
        return None
    if row.get("allowed_for_paper") is False:
        return None
    if not _protocol_matches(run_dir, expected_config):
        return None
    return row


def _main_full_metrics(results_root: Path, seed: int, subset: str, expected_config: dict) -> dict:
    run_dir = results_root / f"{MAIN_EXPERIMENT_PREFIX}{seed}" / subset / "rast_gru_v2"
    metrics_path = run_dir / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(f"Missing formal main-model metrics for full ablation reference: {metrics_path}")
    row = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
    if row.get("result_status") != "formal" or row.get("allowed_for_paper") is not True:
        raise ValueError(f"Main-model result is not an allowed formal result: {metrics_path}")
    if not _protocol_matches(run_dir, expected_config):
        raise ValueError(f"Main-model protocol does not match the full ablation protocol: {run_dir}")
    row["run_dir"] = str(run_dir)
    row["source_experiment_name"] = row.get("experiment_name")
    row["ablation_full_reference"] = True
    return row


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run FD004 multi-seed ablation experiments for the major OCM-MST-GRU modules.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "paper_ablation.yaml"))
    parser.add_argument("--subset", default="FD004")
    parser.add_argument("--seeds", nargs="+", type=int, default=FORMAL_BENCHMARK_SEEDS)
    parser.add_argument("--variants", nargs="+", default=MAJOR_FD004_VARIANTS)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--rerun-complete", action="store_true")
    parser.add_argument("--no-batch-bar", action="store_true")
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    base = load_config(args.config)
    results_root = PROJECT_ROOT / base["project"].get("results_dir", "results")
    total_runs = len(args.seeds) * len(args.variants)
    rows = []
    run_number = 0
    print(
        f"[MULTISEED ABLATION START] subset={args.subset} total_runs={total_runs} "
        f"seeds={','.join(map(str, args.seeds))} variants={','.join(args.variants)}",
        flush=True,
    )
    for seed in args.seeds:
        for variant in args.variants:
            run_number += 1
            cfg, model_name = make_variant(base, variant)
            cfg = copy.deepcopy(cfg)
            cfg["project"]["seed"] = int(seed)
            cfg["project"]["experiment_name"] = f"ablation_fd004_seed{seed}_{variant}"
            cfg.setdefault("augmentation", {})["seed"] = int(seed)
            cfg.setdefault("protocol", {})["ablation_variant"] = variant
            cfg.setdefault("protocol", {})["ablation_scope"] = "FD004 multi-seed major modules"
            if args.epochs is not None:
                cfg["training"]["epochs"] = int(args.epochs)
            if args.no_batch_bar:
                cfg.setdefault("progress", {})["batch_bar"] = False

            if variant == "full":
                row = _main_full_metrics(results_root, int(seed), args.subset, cfg)
                row["ablation_variant"] = variant
                row["ablation_scope"] = "FD004 multi-seed major modules"
                rows.append(row)
                print(
                    f"[MULTISEED ABLATION REFERENCE {run_number}/{total_runs}] "
                    f"seed={seed} variant=full source={row['run_dir']}",
                    flush=True,
                )
                continue

            run_dir = results_root / cfg["project"]["experiment_name"] / args.subset / model_name
            existing = None if args.rerun_complete else _existing_metrics(run_dir, cfg)
            if existing is not None:
                existing["ablation_variant"] = variant
                existing["ablation_scope"] = "FD004 multi-seed major modules"
                rows.append(existing)
                print(
                    f"[MULTISEED ABLATION SKIP {run_number}/{total_runs}] "
                    f"seed={seed} variant={variant} existing={run_dir}",
                    flush=True,
                )
                continue

            print(f"[MULTISEED ABLATION RUN {run_number}/{total_runs}] seed={seed} variant={variant}", flush=True)
            row = train_model(cfg, subset=args.subset, model_name=model_name, run_index=run_number, total_runs=total_runs)
            row["ablation_variant"] = variant
            row["ablation_scope"] = "FD004 multi-seed major modules"
            row["result_status"] = "supplementary"
            rows.append(row)
            print(
                f"[MULTISEED ABLATION DONE {run_number}/{total_runs}] seed={seed} variant={variant} "
                f"rmse={row.get('test_rmse', 0.0):.4f} lpr30={row.get('test_critical_30_late_prediction_ratio', 0.0):.4f}",
                flush=True,
            )

    out = results_root / "ablation_fd004_multiseed_runs.csv"
    metrics_rows_to_csv(rows, out)
    print(f"[MULTISEED ABLATION FINISH] saved summary to {out}", flush=True)


if __name__ == "__main__":
    main()
