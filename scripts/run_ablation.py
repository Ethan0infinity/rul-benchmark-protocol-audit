from __future__ import annotations

import argparse
import copy
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rul.config import load_config
from rul.experiments import metrics_rows_to_csv
from rul.training import train_model


DEFAULT_ABLATION_VARIANTS = [
    "full",
    "all_sensors",
    "no_condition_norm",
    "no_missing_mask",
    "no_reliability_gate",
    "no_channel_gate",
    "no_temporal_attention",
    "no_local_trend",
    "no_degradation_aug",
    "normal_huber",
    "weighted_huber_no_asymmetry",
    "no_smooth_late_risk",
    "no_reliability_supervision",
    "no_consistency_regularization",
    "no_quantile_calibration",
]


def make_variant(base: dict, variant: str) -> tuple[dict, str]:
    cfg = copy.deepcopy(base)
    model_name = "rast_gru_v2"
    cfg["project"]["experiment_name"] = f"ablation_{variant}"
    if variant == "full":
        pass
    elif variant == "all_sensors":
        cfg["data"]["feature_mode"] = "all"
    elif variant == "no_condition_norm":
        cfg["data"]["scaler"] = "standard"
        cfg["data"].pop("condition_clusters", None)
    elif variant == "no_missing_mask":
        cfg["data"]["append_missing_mask"] = False
    elif variant == "no_reliability_gate":
        cfg["model"]["use_reliability_gate"] = False
    elif variant == "no_channel_gate":
        cfg["model"]["use_channel_gate"] = False
    elif variant == "no_temporal_attention":
        cfg["model"]["use_temporal_attention"] = False
    elif variant == "no_local_trend":
        cfg["model"]["use_local_trend"] = False
    elif variant == "no_degradation_aug":
        cfg["augmentation"]["profile"] = "none"
        cfg["augmentation"]["train_noise_std"] = 0.0
        cfg["augmentation"]["train_missing_rate"] = 0.0
        cfg["augmentation"]["train_block_missing_rate"] = 0.0
        cfg["augmentation"]["train_drift_rate"] = 0.0
        cfg["augmentation"]["train_sensor_dropout_prob"] = 0.0
    elif variant == "normal_huber":
        cfg["training"]["loss"] = "huber"
        cfg["training"]["late_life_weight"] = 1.0
        cfg["training"]["late_over_weight"] = 1.0
        cfg["training"]["smooth_late_risk_weight"] = 0.0
    elif variant == "weighted_huber_no_asymmetry":
        cfg["training"]["loss"] = "asymmetric_weighted_huber"
        cfg["training"]["late_over_weight"] = 1.0
    elif variant == "no_smooth_late_risk":
        cfg["training"]["smooth_late_risk_weight"] = 0.0
    elif variant == "no_reliability_supervision":
        cfg["training"]["reliability_supervision_weight"] = 0.0
    elif variant == "no_consistency_regularization":
        cfg["training"]["consistency_weight"] = 0.0
    elif variant == "no_quantile_calibration":
        cfg["training"]["quantile_calibration_weight"] = 0.0
        cfg["model"]["use_uncertainty_head"] = False
    else:
        raise ValueError(f"Unknown ablation variant: {variant}")
    return cfg, model_name


def main() -> None:
    parser = argparse.ArgumentParser(description="Run RAST-GRU ablation experiments.")
    parser.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "paper_ablation.yaml"))
    parser.add_argument("--subsets", nargs="+", default=["FD001", "FD004"])
    parser.add_argument(
        "--variants",
        nargs="+",
        default=DEFAULT_ABLATION_VARIANTS,
    )
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--no-batch-bar", action="store_true", help="Hide tqdm batch bars and keep one clear console line per epoch.")
    args = parser.parse_args()

    base = load_config(args.config)
    rows = []
    total_runs = len(args.subsets) * len(args.variants)
    run_number = 0
    print(
        f"[ABLATION START] total_runs={total_runs} "
        f"subsets={','.join(args.subsets)} variants={','.join(args.variants)}",
        flush=True,
    )
    for subset in args.subsets:
        for variant in args.variants:
            run_number += 1
            print(f"[ABLATION {run_number:03d}/{total_runs:03d}] subset={subset} variant={variant}", flush=True)
            cfg, model_name = make_variant(base, variant)
            if args.epochs is not None:
                cfg["training"]["epochs"] = args.epochs
            if args.no_batch_bar:
                cfg.setdefault("progress", {})["batch_bar"] = False
            row = train_model(cfg, subset=subset, model_name=model_name, run_index=run_number, total_runs=total_runs)
            row["ablation_variant"] = variant
            rows.append(row)
            print(
                f"[ABLATION {run_number:03d}/{total_runs:03d} DONE] "
                f"variant={variant} model={model_name} "
                f"test_rmse={row.get('test_rmse', 0.0):.4f} "
                f"test_mae={row.get('test_mae', 0.0):.4f} "
                f"run_dir={row.get('run_dir')}",
                flush=True,
            )

    out = PROJECT_ROOT / base["project"].get("results_dir", "results") / "ablation_runs.csv"
    metrics_rows_to_csv(rows, out)
    print(f"[ABLATION FINISH] saved ablation summary to {out}", flush=True)


if __name__ == "__main__":
    main()
