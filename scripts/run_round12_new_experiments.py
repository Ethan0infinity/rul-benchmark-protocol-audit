from __future__ import annotations

import argparse
import copy
import csv
import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import pandas as pd
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rul.config import load_config
from rul.experiments import evaluate_robustness
from rul.ncmapss import load_prepared_cache, prepare_ncmapss_ds02, save_prepared_cache
from rul.training import train_model
from rul.utils import get_device
from run_ncmapss_benchmark import train_one


RESULTS_ROOT = PROJECT_ROOT / "results" / "round12_new_evidence"
OUTPUT_ROOT = PROJECT_ROOT / "paper_outputs" / "round12_new_evidence"
RAW_NCMAPSS = PROJECT_ROOT / "data" / "external" / "raw" / "N-CMAPSS_DS02-006.h5"
NCMAPSS_CACHE_ROOT = PROJECT_ROOT / "data" / "external" / "processed"

DEFAULT_STREAMS = (
    31415,
    27182,
    16180,
    14142,
    17320,
    42,
    123,
    2024,
    2025,
    2026,
)
PREFERENCE_SPLITS = (42, 123, 2024)
PREFERENCE_STREAMS = (31415, 27182, 16180)
FACTORIAL_STREAMS = (42, 123, 2024, 2025, 2026)
NCMAPSS_SEEDS = (42, 123, 2024, 2025, 2026)
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
PREFERENCE_LIFE_WEIGHTS = (1.25, 1.50, 2.00)
PREFERENCE_LATE_WEIGHTS = (1.00, 1.25, 1.50)
LOFO_FAMILIES = ("noise", "random_missingness", "block_missingness", "drift")
NCMAPSS_SAMPLING = (50, 100, 200)
NCMAPSS_WINDOWS = (30, 50, 70)
NCMAPSS_HORIZON_MATCHED = ((50, 121), (100, 61), (200, 31))
CROSSED_SPLITS = (42, 123, 2024)
CROSSED_STREAMS = (31415, 27182, 16180)
PREPROCESS_SEEDS = (42, 123, 2024)
PREPROCESS_SUBSETS = ("FD004",)
PREPROCESS_WINDOWS = (20, 30, 50)
PREPROCESS_SENSOR_BUDGETS = (8, 14, 21)


@dataclass(frozen=True)
class TrainingJob:
    family: str
    label: str
    subset: str
    model: str
    config: dict
    metadata: dict[str, object]


def configure_stream(
    cfg: dict,
    *,
    split_seed: int,
    stream_seed: int,
) -> None:
    cfg["project"]["seed"] = int(stream_seed)
    cfg["project"]["split_seed"] = int(split_seed)
    cfg["project"]["initialization_seed"] = int(stream_seed)
    cfg["project"]["shuffle_seed"] = int(stream_seed)
    cfg["augmentation"]["seed"] = int(stream_seed)


def safe_token(value: float) -> str:
    return f"{value:.2f}".replace(".", "p")


def configured_run_dir(job: TrainingJob) -> Path:
    return (
        PROJECT_ROOT
        / job.config["project"].get("results_dir", "results")
        / job.config["project"]["experiment_name"]
        / job.subset
        / job.model
    )


def artifact_id(run_dir: Path) -> str:
    relative = run_dir.relative_to(PROJECT_ROOT).as_posix()
    return "artifact-" + hashlib.sha256(relative.encode("utf-8")).hexdigest()[:16]


def run_complete(run_dir: Path, expected: dict, *, epochs: int) -> bool:
    required = (
        run_dir / "metrics.json",
        run_dir / "run_config.yaml",
        run_dir / "best_model.pt",
        run_dir / "test_predictions.csv",
        run_dir / "epoch_history.csv",
    )
    if not all(path.exists() for path in required):
        return False
    try:
        metrics = json.loads(
            (run_dir / "metrics.json").read_text(encoding="utf-8-sig")
        )
        actual = load_config(run_dir / "run_config.yaml")
        checkpoint = torch.load(
            run_dir / "best_model.pt",
            map_location="cpu",
            weights_only=False,
        )
    except Exception:
        return False
    if int(metrics.get("planned_epochs", 0)) != int(epochs):
        return False
    if not isinstance(checkpoint, dict) or "model_state" not in checkpoint:
        return False
    actual_for_compare = copy.deepcopy(actual)
    expected_for_compare = copy.deepcopy(expected)
    if expected.get("model", {}).get("name") == "rast_gru":
        # The plain RAST implementation never constructs the OCM consistency
        # second view. Normalize these inactive fields before cache comparison.
        for cfg in (actual_for_compare, expected_for_compare):
            cfg.setdefault("training", {})["consistency_weight"] = 0.0
            cfg.setdefault("training", {})["consistency_noise_std"] = 0.0
    return all(
        actual_for_compare.get(section, {}) == expected_for_compare.get(section, {})
        for section in ("project", "data", "model", "training", "augmentation")
    )


def write_manifest(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def execute_jobs(
    jobs: list[TrainingJob],
    *,
    family: str,
    epochs: int,
    dry_run: bool,
    rerun_complete: bool,
    replace_manifest: bool = False,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    manifest = OUTPUT_ROOT / f"{family}_run_manifest.csv"
    existing_rows: list[dict[str, object]] = []
    if manifest.exists() and not dry_run and not replace_manifest:
        existing_rows = pd.read_csv(manifest).to_dict("records")
    replaced_run_dirs = {
        configured_run_dir(job).relative_to(PROJECT_ROOT).as_posix()
        for job in jobs
    }
    retained_rows = [
        row for row in existing_rows
        if str(row.get("run_dir", "")) not in replaced_run_dirs
    ]
    print(f"[ROUND12 {family.upper()}] jobs={len(jobs)} epochs={epochs}", flush=True)
    for index, job in enumerate(jobs, start=1):
        run_dir = configured_run_dir(job)
        status = "planned"
        if dry_run:
            print(
                f"- {index:03d}/{len(jobs)} {job.label}: "
                f"{run_dir.relative_to(PROJECT_ROOT).as_posix()}",
                flush=True,
            )
        elif not rerun_complete and run_complete(
            run_dir,
            job.config,
            epochs=epochs,
        ):
            status = "reused"
            print(
                f"[ROUND12 SKIP {index}/{len(jobs)}] {job.label}",
                flush=True,
            )
        else:
            print(
                f"[ROUND12 RUN {index}/{len(jobs)}] {job.label}",
                flush=True,
            )
            train_model(
                job.config,
                subset=job.subset,
                model_name=job.model,
                run_index=index,
                total_runs=len(jobs),
            )
            status = "completed"
        row = {
            "family": family,
            "label": job.label,
            "subset": job.subset,
            "model": job.model,
            "run_dir": run_dir.relative_to(PROJECT_ROOT).as_posix(),
            "artifact_id": artifact_id(run_dir),
            "status": status,
            "planned_epochs": epochs,
            **job.metadata,
        }
        rows.append(row)
        if not dry_run:
            write_manifest(retained_rows + rows, manifest)
    if dry_run:
        print(f"ROUND12_{family.upper()}_DRY_RUN_PASS jobs={len(jobs)}")
    else:
        print(f"ROUND12_{family.upper()}_READY jobs={len(rows)} manifest={manifest}")
    return rows


def independent_jobs(base: dict, streams: tuple[int, ...], epochs: int) -> list[TrainingJob]:
    jobs = []
    for stream_seed in streams:
        for subset in SUBSETS:
            for point, model in (("rast", "rast_gru"), ("asym", "rast_gru_v2")):
                cfg = copy.deepcopy(base)
                cfg["project"]["results_dir"] = "results/round12_new_evidence"
                cfg["project"]["experiment_name"] = (
                    f"independent_{point}_{subset.lower()}_split42_stream{stream_seed}"
                )
                configure_stream(cfg, split_seed=42, stream_seed=stream_seed)
                cfg["model"]["name"] = model
                cfg["training"]["epochs"] = epochs
                cfg.setdefault("progress", {})["batch_bar"] = False
                jobs.append(
                    TrainingJob(
                        "independent",
                        f"{point} {subset} split=42 stream={stream_seed}",
                        subset,
                        model,
                        cfg,
                        {
                            "point": point,
                            "split_seed": 42,
                            "training_stream_seed": stream_seed,
                        },
                    )
                )
    return jobs


def independent_experiment_name(point: str, subset: str, stream_seed: int) -> str:
    return f"independent_{point}_{subset.lower()}_split42_stream{stream_seed}"


def crossed_split_stream_jobs(
    base: dict,
    split_seeds: tuple[int, ...],
    stream_seeds: tuple[int, ...],
    epochs: int,
    subsets: tuple[str, ...] = SUBSETS,
    points: tuple[str, ...] = ("rast", "asym"),
) -> list[TrainingJob]:
    jobs = []
    for split_seed in split_seeds:
        for stream_seed in stream_seeds:
            for subset in subsets:
                for point, model in (("rast", "rast_gru"), ("asym", "rast_gru_v2")):
                    if point not in points:
                        continue
                    reuse_source = "new_crossed_training"
                    if split_seed == 42 and stream_seed in DEFAULT_STREAMS:
                        cfg = copy.deepcopy(base)
                        cfg["project"]["results_dir"] = "results/round12_new_evidence"
                        cfg["project"]["experiment_name"] = independent_experiment_name(
                            point, subset, stream_seed
                        )
                        configure_stream(
                            cfg,
                            split_seed=split_seed,
                            stream_seed=stream_seed,
                        )
                        cfg["model"]["name"] = model
                        cfg["training"]["epochs"] = epochs
                        cfg.setdefault("progress", {})["batch_bar"] = False
                        reuse_source = "fixed_split_independent_family"
                    elif subset == "FD004":
                        shared_experiment = (
                            f"preference_rast_reference_split{split_seed}_stream{stream_seed}"
                            if point == "rast"
                            else (
                                f"preference_ll1p25_lo1p50_split{split_seed}_"
                                f"stream{stream_seed}"
                            )
                        )
                        shared_config = (
                            PROJECT_ROOT
                            / "results"
                            / "round12_new_evidence"
                            / shared_experiment
                            / "FD004"
                            / model
                            / "run_config.yaml"
                        )
                        if shared_config.exists():
                            cfg = load_config(shared_config)
                            reuse_source = "preference_fixed_candidate_family"
                        else:
                            cfg = copy.deepcopy(base)
                            cfg["project"]["results_dir"] = "results/round12_new_evidence"
                            cfg["project"]["experiment_name"] = (
                                f"crossed_{point}_{subset.lower()}_split{split_seed}_"
                                f"stream{stream_seed}"
                            )
                            configure_stream(
                                cfg,
                                split_seed=split_seed,
                                stream_seed=stream_seed,
                            )
                            cfg["model"]["name"] = model
                            cfg["training"]["epochs"] = epochs
                            cfg.setdefault("progress", {})["batch_bar"] = False
                    else:
                        cfg = copy.deepcopy(base)
                        cfg["project"]["results_dir"] = "results/round12_new_evidence"
                        cfg["project"]["experiment_name"] = (
                            f"crossed_{point}_{subset.lower()}_split{split_seed}_"
                            f"stream{stream_seed}"
                        )
                        configure_stream(
                            cfg,
                            split_seed=split_seed,
                            stream_seed=stream_seed,
                        )
                        cfg["model"]["name"] = model
                        cfg["training"]["epochs"] = epochs
                        cfg.setdefault("progress", {})["batch_bar"] = False
                    jobs.append(
                        TrainingJob(
                            "crossed",
                            (
                                f"{point} {subset} split={split_seed} "
                                f"stream={stream_seed}"
                            ),
                            subset,
                            model,
                            cfg,
                            {
                                "point": point,
                                "split_seed": split_seed,
                                "training_stream_seed": stream_seed,
                                "design_role": (
                                    "selected-level 3x3 finite split-by-training-stream audit"
                                ),
                                "artifact_reuse_source": reuse_source,
                            },
                        )
                    )
    return jobs


def cmapss_preprocessing_jobs(
    base: dict,
    streams: tuple[int, ...],
    epochs: int,
    subsets: tuple[str, ...] = PREPROCESS_SUBSETS,
    points: tuple[str, ...] = ("rast", "asym"),
) -> list[TrainingJob]:
    jobs = []
    design_points = (
        ("window", 20, 20, 14),
        ("reference", 30, 30, 14),
        ("window", 50, 50, 14),
        ("sensor_budget", 8, 30, 8),
        ("sensor_budget", 21, 30, 21),
    )
    for stream_seed in streams:
        for subset in subsets:
            for point, model in (("rast", "rast_gru"), ("asym", "rast_gru_v2")):
                if point not in points:
                    continue
                for audit_axis, audit_level, window_size, sensor_budget in design_points:
                    reuse_source = "new_preprocessing_training"
                    if audit_axis == "reference":
                        shared_config = (
                            PROJECT_ROOT
                            / "results"
                            / f"paper_main_v3_seed{stream_seed}"
                            / subset
                            / model
                            / "run_config.yaml"
                        )
                        if not shared_config.exists():
                            raise FileNotFoundError(shared_config)
                        cfg = load_config(shared_config)
                        reuse_source = "formal_composite_seed_reference"
                    elif audit_axis == "window" and point == "asym":
                        shared_config = (
                            PROJECT_ROOT
                            / "results"
                            / (
                                f"design_sensitivity_v3_seed{stream_seed}_"
                                f"window_w{window_size}_cap125"
                            )
                            / subset
                            / model
                            / "run_config.yaml"
                        )
                        if not shared_config.exists():
                            raise FileNotFoundError(shared_config)
                        cfg = load_config(shared_config)
                        reuse_source = "existing_window_sensitivity"
                    else:
                        cfg = copy.deepcopy(base)
                        cfg["project"]["results_dir"] = "results/round12_new_evidence"
                        cfg["project"]["experiment_name"] = (
                            f"cmapss_preprocess_{audit_axis}{audit_level}_{point}_"
                            f"{subset.lower()}_composite{stream_seed}"
                        )
                        cfg["project"]["seed"] = int(stream_seed)
                        cfg["augmentation"]["seed"] = int(stream_seed)
                        cfg["data"]["window_size"] = int(window_size)
                        cfg["data"]["max_selected_sensors"] = int(sensor_budget)
                        cfg["model"]["name"] = model
                        cfg["training"]["epochs"] = epochs
                        cfg.setdefault("progress", {})["batch_bar"] = False
                    jobs.append(
                        TrainingJob(
                            "cmapss_preprocessing",
                            (
                                f"{point} {subset} {audit_axis}={audit_level} "
                                f"stream={stream_seed}"
                            ),
                            subset,
                            model,
                            cfg,
                            {
                                "point": point,
                                "split_seed": stream_seed,
                                "training_stream_seed": stream_seed,
                                "audit_axis": audit_axis,
                                "audit_level": audit_level,
                                "window_size": window_size,
                                "sensor_budget": sensor_budget,
                                "design_role": (
                                    "one-factor-at-a-time composite-seed C-MAPSS preprocessing sensitivity"
                                ),
                                "artifact_reuse_source": reuse_source,
                            },
                        )
                    )
    return jobs


def preference_jobs(
    base: dict,
    split_seeds: tuple[int, ...],
    stream_seeds: tuple[int, ...],
    epochs: int,
    life_weights: tuple[float, ...] = PREFERENCE_LIFE_WEIGHTS,
    late_weights: tuple[float, ...] = PREFERENCE_LATE_WEIGHTS,
) -> list[TrainingJob]:
    jobs = []
    for split_seed in split_seeds:
        for stream_seed in stream_seeds:
            reference = copy.deepcopy(base)
            reference["project"]["results_dir"] = "results/round12_new_evidence"
            if split_seed == 42 and stream_seed in DEFAULT_STREAMS:
                reference["project"]["experiment_name"] = (
                    f"independent_rast_fd004_split42_stream{stream_seed}"
                )
            else:
                reference["project"]["experiment_name"] = (
                    f"preference_rast_reference_split{split_seed}_stream{stream_seed}"
                )
            configure_stream(
                reference,
                split_seed=split_seed,
                stream_seed=stream_seed,
            )
            reference["model"]["name"] = "rast_gru"
            reference["training"]["epochs"] = epochs
            reference.setdefault("progress", {})["batch_bar"] = False
            jobs.append(
                TrainingJob(
                    "preference",
                    f"RAST reference split={split_seed} stream={stream_seed}",
                    "FD004",
                    "rast_gru",
                    reference,
                    {
                        "candidate": "rast_reference",
                        "late_life_weight": reference["training"][
                            "late_life_weight"
                        ],
                        "late_over_weight": reference["training"][
                            "late_over_weight"
                        ],
                        "point": "rast_reference",
                        "split_seed": split_seed,
                        "training_stream_seed": stream_seed,
                    },
                )
            )
            for life_weight in life_weights:
                for late_weight in late_weights:
                    cfg = copy.deepcopy(base)
                    cfg["project"]["results_dir"] = "results/round12_new_evidence"
                    candidate = (
                        f"ll{safe_token(life_weight)}_lo{safe_token(late_weight)}"
                    )
                    cfg["project"]["experiment_name"] = (
                        f"preference_{candidate}_split{split_seed}_stream{stream_seed}"
                    )
                    configure_stream(
                        cfg,
                        split_seed=split_seed,
                        stream_seed=stream_seed,
                    )
                    cfg["model"]["name"] = "rast_gru_v2"
                    cfg["training"]["late_life_weight"] = float(life_weight)
                    cfg["training"]["late_over_weight"] = float(late_weight)
                    cfg["training"]["epochs"] = epochs
                    cfg.setdefault("progress", {})["batch_bar"] = False
                    jobs.append(
                        TrainingJob(
                            "preference",
                            (
                                f"{candidate} split={split_seed} "
                                f"stream={stream_seed}"
                            ),
                            "FD004",
                            "rast_gru_v2",
                            cfg,
                            {
                                "candidate": candidate,
                                "late_life_weight": life_weight,
                                "late_over_weight": late_weight,
                                "point": "asym",
                                "split_seed": split_seed,
                                "training_stream_seed": stream_seed,
                            },
                        )
                    )
    return jobs


def factorial_jobs(base: dict, streams: tuple[int, ...], epochs: int) -> list[TrainingJob]:
    jobs = []
    for stream_seed in streams:
        for backbone, model in (("rast", "rast_gru"), ("ocm", "rast_gru_v2")):
            for risk_factor in (0, 1):
                for consistency_factor in (0, 1):
                    objective = {
                        (0, 0): "base",
                        (1, 0): "risk_only",
                        (0, 1): "consistency_only",
                        (1, 1): "risk_consistency",
                    }[(risk_factor, consistency_factor)]
                    cfg = copy.deepcopy(base)
                    cfg["project"]["results_dir"] = "results/round12_new_evidence"
                    cfg["project"]["experiment_name"] = (
                        f"factorial_{backbone}_{objective}_split42_stream{stream_seed}"
                    )
                    configure_stream(cfg, split_seed=42, stream_seed=stream_seed)
                    cfg["model"]["name"] = model
                    cfg["model"]["use_uncertainty_head"] = False
                    cfg["training"]["epochs"] = epochs
                    cfg["training"]["reliability_supervision_weight"] = 0.0
                    cfg["training"]["quantile_calibration_weight"] = 0.0
                    cfg["training"]["loss"] = (
                        "asymmetric_weighted_huber" if risk_factor else "huber"
                    )
                    cfg["training"]["late_life_weight"] = (
                        1.25 if risk_factor else 1.0
                    )
                    cfg["training"]["late_over_weight"] = (
                        1.50 if risk_factor else 1.0
                    )
                    cfg["training"]["smooth_late_risk_weight"] = (
                        0.10 if risk_factor else 0.0
                    )
                    cfg["training"]["consistency_weight"] = (
                        0.05 if consistency_factor else 0.0
                    )
                    cfg["training"]["allow_shared_risk_regularization"] = bool(
                        risk_factor or consistency_factor
                    )
                    cfg.setdefault("progress", {})["batch_bar"] = False
                    jobs.append(
                        TrainingJob(
                            "factorial",
                            (
                                f"backbone={backbone} objective={objective} "
                                f"stream={stream_seed}"
                            ),
                            "FD004",
                            model,
                            cfg,
                            {
                                "backbone": backbone,
                                "objective": objective,
                                "risk_factor": risk_factor,
                                "consistency_factor": consistency_factor,
                                "split_seed": 42,
                                "training_stream_seed": stream_seed,
                                "objective_scope": (
                                    "shared-objective 2x2x2 backbone-by-risk-by-"
                                    "consistency factorial; OCM-specific reliability and "
                                    "quantile terms disabled"
                                ),
                            },
                        )
                    )
    return jobs


def apply_lofo(cfg: dict, family: str) -> None:
    if family == "noise":
        cfg["augmentation"]["train_noise_std"] = 0.0
        if cfg["model"]["name"] == "rast_gru_v2":
            cfg["training"]["consistency_weight"] = 0.0
            cfg["training"]["consistency_noise_std"] = 0.0
    elif family == "random_missingness":
        cfg["augmentation"]["train_missing_rate"] = 0.0
    elif family == "block_missingness":
        cfg["augmentation"]["train_block_missing_rate"] = 0.0
    elif family == "drift":
        cfg["augmentation"]["train_drift_rate"] = 0.0
    else:
        raise ValueError(family)


def lofo_exposure_metadata(cfg: dict) -> dict[str, object]:
    training = cfg["training"]
    augmentation = cfg["augmentation"]
    consistency_active = cfg["model"]["name"] == "rast_gru_v2"
    return {
        "primary_gaussian_noise_std": float(augmentation.get("train_noise_std", 0.0)),
        "consistency_weight": (
            float(training.get("consistency_weight", 0.0))
            if consistency_active
            else 0.0
        ),
        "consistency_gaussian_noise_std": float(
            training.get("consistency_noise_std", 0.0)
            if consistency_active
            else 0.0
        ),
        "random_missingness_rate": float(
            augmentation.get("train_missing_rate", 0.0)
        ),
        "block_missingness_rate": float(
            augmentation.get("train_block_missing_rate", 0.0)
        ),
        "drift_rate": float(augmentation.get("train_drift_rate", 0.0)),
    }


def validate_lofo_exposure_matrix(jobs: list[TrainingJob]) -> None:
    for job in jobs:
        family = str(job.metadata["held_out_family"])
        exposure = lofo_exposure_metadata(job.config)
        if family == "noise" and any(
            float(exposure[name]) != 0.0
            for name in (
                "primary_gaussian_noise_std",
                "consistency_weight",
                "consistency_gaussian_noise_std",
            )
        ):
            raise ValueError(
                f"noise LOFO retains Gaussian training exposure: {job.label} {exposure}"
            )
        held_out_field = {
            "random_missingness": "random_missingness_rate",
            "block_missingness": "block_missingness_rate",
            "drift": "drift_rate",
        }.get(family)
        if held_out_field is not None and float(exposure[held_out_field]) != 0.0:
            raise ValueError(
                f"{family} LOFO retains held-out exposure: {job.label} {exposure}"
            )


def write_lofo_exposure_matrix(jobs: list[TrainingJob]) -> None:
    rows = []
    for job in jobs:
        rows.append(
            {
                "artifact_id": artifact_id(configured_run_dir(job)),
                "held_out_family": job.metadata["held_out_family"],
                "point": job.metadata["point"],
                "training_stream_seed": job.metadata["training_stream_seed"],
                **lofo_exposure_metadata(job.config),
            }
        )
    path = OUTPUT_ROOT / "lofo_active_exposure_matrix.csv"
    retained = pd.DataFrame()
    if path.exists():
        retained = pd.read_csv(path)
        held_out = {str(job.metadata["held_out_family"]) for job in jobs}
        retained = retained[~retained["held_out_family"].isin(held_out)]
    pd.concat([retained, pd.DataFrame(rows)], ignore_index=True).to_csv(path, index=False)


def lofo_jobs(
    base: dict,
    streams: tuple[int, ...],
    epochs: int,
    families: tuple[str, ...] = LOFO_FAMILIES,
    points: tuple[str, ...] = ("rast", "asym"),
) -> list[TrainingJob]:
    jobs = []
    for stream_seed in streams:
        for family in families:
            for point, model in (("rast", "rast_gru"), ("asym", "rast_gru_v2")):
                if point not in points:
                    continue
                cfg = copy.deepcopy(base)
                cfg["project"]["results_dir"] = "results/round12_new_evidence"
                cfg["project"]["experiment_name"] = (
                    f"lofo_{family}_{point}_split42_stream{stream_seed}"
                )
                configure_stream(cfg, split_seed=42, stream_seed=stream_seed)
                cfg["model"]["name"] = model
                cfg["training"]["epochs"] = epochs
                apply_lofo(cfg, family)
                cfg.setdefault("progress", {})["batch_bar"] = False
                jobs.append(
                    TrainingJob(
                        "lofo",
                        f"family={family} point={point} stream={stream_seed}",
                        "FD004",
                        model,
                        cfg,
                        {
                            "held_out_family": family,
                            "point": point,
                            "split_seed": 42,
                            "training_stream_seed": stream_seed,
                            **lofo_exposure_metadata(cfg),
                        },
                    )
                )
    return jobs


def lofo_override(family: str) -> dict[str, list[float]]:
    empty = {
        "noise_levels": [],
        "missing_rates": [],
        "block_missing_rates": [],
        "drift_rates": [],
        "bias_levels": [],
        "stuck_at_rates": [],
        "burst_noise_levels": [],
        "correlated_missing_rates": [],
    }
    if family == "noise":
        empty["noise_levels"] = [0.01, 0.03, 0.05, 0.10]
    elif family == "random_missingness":
        empty["missing_rates"] = [0.10, 0.20, 0.30, 0.40]
    elif family == "block_missingness":
        empty["block_missing_rates"] = [0.10, 0.20, 0.30]
    elif family == "drift":
        empty["drift_rates"] = [0.02, 0.05, 0.10]
    return empty


def evaluate_lofo(
    jobs: list[TrainingJob],
    *,
    perturbation_seeds: tuple[int, ...],
    dry_run: bool,
) -> None:
    total = len(jobs) * len(perturbation_seeds)
    if dry_run:
        print(
            f"ROUND12_LOFO_EVALUATION_DRY_RUN_PASS evaluations={total}",
            flush=True,
        )
        return
    rows: list[dict[str, object]] = []
    engine_rows: list[dict[str, object]] = []
    output = OUTPUT_ROOT / "lofo_held_out_family_perturbation_rows.csv"
    engine_output = OUTPUT_ROOT / "lofo_held_out_family_engine_rows.csv"
    retained = pd.DataFrame()
    retained_engines = pd.DataFrame()
    if output.exists():
        retained = pd.read_csv(output)
        held_out = {str(job.metadata["held_out_family"]) for job in jobs}
        retained = retained[~retained["held_out_family"].isin(held_out)]
    if engine_output.exists():
        retained_engines = pd.read_csv(engine_output)
        held_out = {str(job.metadata["held_out_family"]) for job in jobs}
        retained_engines = retained_engines[
            ~retained_engines["held_out_family"].isin(held_out)
        ]
    evaluation_index = 0
    for index, job in enumerate(jobs, start=1):
        run_dir = configured_run_dir(job)
        family = str(job.metadata["held_out_family"])
        for perturbation_seed in perturbation_seeds:
            evaluation_index += 1
            print(
                f"[ROUND12 LOFO EVAL {evaluation_index}/{total}] "
                f"{job.label} perturb={perturbation_seed}",
                flush=True,
            )
            evaluation_engine_rows: list[dict[str, object]] = []
            evaluated = evaluate_robustness(
                run_dir,
                perturbation_seed=perturbation_seed,
                save=False,
                engine_rows=evaluation_engine_rows,
                robustness_override=lofo_override(family),
            )
            run_dir_relative = run_dir.relative_to(PROJECT_ROOT).as_posix()
            for row in evaluated:
                rows.append(
                    {
                        **row,
                        **job.metadata,
                        "run_dir": run_dir_relative,
                    }
                )
            for row in evaluation_engine_rows:
                engine_rows.append(
                    {
                        **row,
                        **job.metadata,
                        "run_dir": run_dir_relative,
                    }
                )
        pd.concat([retained, pd.DataFrame(rows)], ignore_index=True).to_csv(
            output,
            index=False,
        )
        pd.concat(
            [retained_engines, pd.DataFrame(engine_rows)],
            ignore_index=True,
        ).to_csv(engine_output, index=False)
    total_rows = len(retained) + len(rows)
    total_engine_rows = len(retained_engines) + len(engine_rows)
    print(
        "ROUND12_LOFO_EVALUATION_READY "
        f"rows={total_rows} engine_rows={total_engine_rows} "
        f"output={output} engine_output={engine_output}"
    )


def ncmapss_cache_path(sampling: int, window: int) -> Path:
    return NCMAPSS_CACHE_ROOT / (
        f"ncmapss_ds02_smp{sampling}_win{window}_k4_round12.npz"
    )


def ensure_ncmapss_cache(
    sampling: int,
    window: int,
    *,
    dry_run: bool,
) -> Path:
    path = ncmapss_cache_path(sampling, window)
    if path.exists() or dry_run:
        return path
    print(
        f"[ROUND12 NCMAPSS CACHE] sampling={sampling} window={window}",
        flush=True,
    )
    prepared = prepare_ncmapss_ds02(
        RAW_NCMAPSS,
        sampling=sampling,
        window_size=window,
        stride=1,
        condition_clusters=4,
    )
    save_prepared_cache(prepared, path)
    return path


def ncmapss_run_complete(
    run_dir: Path,
    *,
    sampling: int,
    window: int,
    epochs: int,
) -> bool:
    metrics_path = run_dir / "metrics.json"
    checkpoint_path = run_dir / "best_model.pt"
    if not metrics_path.exists() or not checkpoint_path.exists():
        return False
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=False,
        )
    except Exception:
        return False
    return bool(
        int(metrics.get("sampling", -1)) == sampling
        and int(metrics.get("window_size", -1)) == window
        and int(metrics.get("planned_epochs", -1)) == epochs
        and isinstance(checkpoint, dict)
        and "model_state" in checkpoint
    )


def run_ncmapss_sensitivity(
    base: dict,
    *,
    samplings: tuple[int, ...],
    windows: tuple[int, ...],
    seeds: tuple[int, ...],
    epochs: int,
    dry_run: bool,
    rerun_complete: bool,
    cell_pairs: tuple[tuple[int, int], ...] | None = None,
    output_stem: str = "ncmapss_sampling_window",
    experiment_prefix: str = "ncmapss",
    display_label: str = "NCMAPSS",
) -> None:
    pairs = (
        cell_pairs
        if cell_pairs is not None
        else tuple((sampling, window) for sampling in samplings for window in windows)
    )
    jobs = [
        (sampling, window, seed)
        for sampling, window in pairs
        for seed in seeds
    ]
    print(f"[ROUND12 {display_label}] jobs={len(jobs)} epochs={epochs}", flush=True)
    if dry_run:
        for index, (sampling, window, seed) in enumerate(jobs, start=1):
            print(
                f"- {index:03d}/{len(jobs)} sampling={sampling} "
                f"window={window} seed={seed}",
                flush=True,
            )
        print(f"ROUND12_{display_label}_DRY_RUN_PASS jobs={len(jobs)}")
        return

    if not RAW_NCMAPSS.exists():
        raise FileNotFoundError(RAW_NCMAPSS)
    device = get_device("auto")
    rows: list[dict[str, object]] = []
    manifest_rows: list[dict[str, object]] = []
    current_key: tuple[int, int] | None = None
    prepared = None
    for index, (sampling, window, seed) in enumerate(jobs, start=1):
        key = (sampling, window)
        cache = ensure_ncmapss_cache(sampling, window, dry_run=False)
        if current_key != key:
            prepared = load_prepared_cache(cache)
            current_key = key
        assert prepared is not None
        experiment = f"{experiment_prefix}_smp{sampling}_win{window}_seed{seed}"
        run_dir = RESULTS_ROOT / experiment / "DS02" / "rast_gru_v2"
        if (
            not rerun_complete
            and ncmapss_run_complete(
                run_dir,
                sampling=sampling,
                window=window,
                epochs=epochs,
            )
        ):
            row = json.loads(
                (run_dir / "metrics.json").read_text(encoding="utf-8-sig")
            )
            status = "reused"
            print(
                f"[ROUND12 NCMAPSS SKIP {index}/{len(jobs)}] {experiment}",
                flush=True,
            )
        else:
            print(
                f"[ROUND12 NCMAPSS RUN {index}/{len(jobs)}] {experiment}",
                flush=True,
            )
            cfg = copy.deepcopy(base)
            cfg["training"]["epochs"] = epochs
            row = train_one(
                prepared,
                model_name="rast_gru_v2",
                seed=seed,
                run_dir=run_dir,
                epochs=epochs,
                patience=int(cfg["training"]["patience"]),
                batch_size=256,
                device=device,
                protocol_config=cfg,
                experiment_name=experiment,
                result_status="computational_proxy_sensitivity",
            )
            status = "completed"
        row["sampling_interval"] = sampling
        row["sensitivity_window_size"] = window
        row["source_record_span"] = 1 + (window - 1) * sampling
        rows.append(row)
        manifest_rows.append(
            {
                "sampling": sampling,
                "window_size": window,
                "seed": seed,
                "status": status,
                "planned_epochs": epochs,
                "source_record_span": 1 + (window - 1) * sampling,
                "cache": cache.relative_to(PROJECT_ROOT).as_posix(),
                "run_dir": run_dir.relative_to(PROJECT_ROOT).as_posix(),
                "artifact_id": artifact_id(run_dir),
            }
        )
        write_manifest(
            manifest_rows,
            OUTPUT_ROOT / f"{output_stem}_run_manifest.csv",
        )
        pd.DataFrame(rows).to_csv(
            OUTPUT_ROOT / f"{output_stem}_seed_metrics.csv",
            index=False,
        )
    print(
        f"ROUND12_{display_label}_READY runs={len(rows)} "
        f"output={OUTPUT_ROOT / f'{output_stem}_seed_metrics.csv'}"
    )


def parse_int_tuple(values: list[int] | tuple[int, ...]) -> tuple[int, ...]:
    return tuple(int(value) for value in values)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the Round-12 independent-stream, preference-retraining, "
            "factorial, LOFO, and N-CMAPSS sensitivity experiments."
        )
    )
    parser.add_argument(
        "--families",
        nargs="+",
        choices=(
            "independent",
            "crossed",
            "cmapss_preprocessing",
            "preference",
            "factorial",
            "lofo",
            "ncmapss",
            "ncmapss_horizon",
        ),
        default=(
            "independent",
            "crossed",
            "cmapss_preprocessing",
            "preference",
            "factorial",
            "lofo",
            "ncmapss",
            "ncmapss_horizon",
        ),
    )
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--rerun-complete", action="store_true")
    parser.add_argument(
        "--replace-manifest",
        action="store_true",
        help=(
            "Rewrite each selected family manifest from the scheduled jobs. "
            "Use after disjoint parallel workers finish to remove stale rows."
        ),
    )
    parser.add_argument("--independent-streams", nargs="+", type=int, default=DEFAULT_STREAMS)
    parser.add_argument("--crossed-splits", nargs="+", type=int, default=CROSSED_SPLITS)
    parser.add_argument("--crossed-streams", nargs="+", type=int, default=CROSSED_STREAMS)
    parser.add_argument(
        "--crossed-subsets",
        nargs="+",
        choices=SUBSETS,
        default=SUBSETS,
    )
    parser.add_argument(
        "--crossed-points",
        nargs="+",
        choices=("rast", "asym"),
        default=("rast", "asym"),
        help="Restrict crossed scheduling to named configuration points.",
    )
    parser.add_argument(
        "--preprocessing-streams",
        nargs="+",
        type=int,
        default=PREPROCESS_SEEDS,
    )
    parser.add_argument(
        "--preprocessing-subsets",
        nargs="+",
        choices=SUBSETS,
        default=PREPROCESS_SUBSETS,
    )
    parser.add_argument(
        "--preprocessing-points",
        nargs="+",
        choices=("rast", "asym"),
        default=("rast", "asym"),
        help="Restrict preprocessing scheduling to named configuration points.",
    )
    parser.add_argument("--preference-splits", nargs="+", type=int, default=PREFERENCE_SPLITS)
    parser.add_argument("--preference-streams", nargs="+", type=int, default=PREFERENCE_STREAMS)
    parser.add_argument(
        "--preference-life-weights",
        nargs="+",
        type=float,
        default=PREFERENCE_LIFE_WEIGHTS,
    )
    parser.add_argument(
        "--preference-late-weights",
        nargs="+",
        type=float,
        default=PREFERENCE_LATE_WEIGHTS,
    )
    parser.add_argument("--factorial-streams", nargs="+", type=int, default=FACTORIAL_STREAMS)
    parser.add_argument("--lofo-streams", nargs="+", type=int, default=FACTORIAL_STREAMS)
    parser.add_argument(
        "--lofo-families",
        nargs="+",
        choices=LOFO_FAMILIES,
        default=LOFO_FAMILIES,
    )
    parser.add_argument(
        "--lofo-points",
        nargs="+",
        choices=("rast", "asym"),
        default=("rast", "asym"),
        help="Restrict LOFO scheduling to named configuration points.",
    )
    parser.add_argument(
        "--lofo-evaluation-only",
        action="store_true",
        help="Reuse complete LOFO checkpoints and rebuild held-out-family evaluations.",
    )
    parser.add_argument("--perturbation-seeds", nargs="+", type=int, default=FACTORIAL_STREAMS)
    parser.add_argument("--ncmapss-samplings", nargs="+", type=int, default=NCMAPSS_SAMPLING)
    parser.add_argument("--ncmapss-windows", nargs="+", type=int, default=NCMAPSS_WINDOWS)
    parser.add_argument("--ncmapss-seeds", nargs="+", type=int, default=NCMAPSS_SEEDS)
    args = parser.parse_args()

    if args.epochs <= 0:
        raise SystemExit("--epochs must be positive")
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    base = load_config(PROJECT_ROOT / "configs" / "paper_main.yaml")

    if "independent" in args.families:
        execute_jobs(
            independent_jobs(
                base,
                parse_int_tuple(args.independent_streams),
                args.epochs,
            ),
            family="independent",
            epochs=args.epochs,
            dry_run=args.dry_run,
            rerun_complete=args.rerun_complete,
            replace_manifest=args.replace_manifest,
        )
    if "crossed" in args.families:
        execute_jobs(
            crossed_split_stream_jobs(
                base,
                parse_int_tuple(args.crossed_splits),
                parse_int_tuple(args.crossed_streams),
                args.epochs,
                tuple(args.crossed_subsets),
                tuple(args.crossed_points),
            ),
            family="crossed",
            epochs=args.epochs,
            dry_run=args.dry_run,
            rerun_complete=args.rerun_complete,
            replace_manifest=args.replace_manifest,
        )
    if "cmapss_preprocessing" in args.families:
        execute_jobs(
            cmapss_preprocessing_jobs(
                base,
                parse_int_tuple(args.preprocessing_streams),
                args.epochs,
                tuple(args.preprocessing_subsets),
                tuple(args.preprocessing_points),
            ),
            family="cmapss_preprocessing",
            epochs=args.epochs,
            dry_run=args.dry_run,
            rerun_complete=args.rerun_complete,
            replace_manifest=args.replace_manifest,
        )
    if "preference" in args.families:
        execute_jobs(
            preference_jobs(
                base,
                parse_int_tuple(args.preference_splits),
                parse_int_tuple(args.preference_streams),
                args.epochs,
                tuple(float(value) for value in args.preference_life_weights),
                tuple(float(value) for value in args.preference_late_weights),
            ),
            family="preference",
            epochs=args.epochs,
            dry_run=args.dry_run,
            rerun_complete=args.rerun_complete,
            replace_manifest=args.replace_manifest,
        )
    if "factorial" in args.families:
        execute_jobs(
            factorial_jobs(
                base,
                parse_int_tuple(args.factorial_streams),
                args.epochs,
            ),
            family="factorial",
            epochs=args.epochs,
            dry_run=args.dry_run,
            rerun_complete=args.rerun_complete,
            replace_manifest=args.replace_manifest,
        )
    if "lofo" in args.families:
        jobs = lofo_jobs(
            base,
            parse_int_tuple(args.lofo_streams),
            args.epochs,
            tuple(args.lofo_families),
            tuple(args.lofo_points),
        )
        validate_lofo_exposure_matrix(jobs)
        if not args.dry_run:
            write_lofo_exposure_matrix(jobs)
        if args.lofo_evaluation_only and not args.dry_run:
            incomplete = [
                job.label
                for job in jobs
                if not run_complete(
                    configured_run_dir(job), job.config, epochs=args.epochs
                )
            ]
            if incomplete:
                raise RuntimeError(
                    "LOFO evaluation-only requested with incomplete checkpoints: "
                    + ", ".join(incomplete)
                )
            print(f"ROUND12_LOFO_CHECKPOINTS_READY jobs={len(jobs)}")
        else:
            execute_jobs(
                jobs,
                family="lofo",
                epochs=args.epochs,
                dry_run=args.dry_run,
                rerun_complete=args.rerun_complete,
                replace_manifest=args.replace_manifest,
            )
        evaluate_lofo(
            jobs,
            perturbation_seeds=parse_int_tuple(args.perturbation_seeds),
            dry_run=args.dry_run,
        )
    if "ncmapss" in args.families:
        run_ncmapss_sensitivity(
            base,
            samplings=parse_int_tuple(args.ncmapss_samplings),
            windows=parse_int_tuple(args.ncmapss_windows),
            seeds=parse_int_tuple(args.ncmapss_seeds),
            epochs=args.epochs,
            dry_run=args.dry_run,
            rerun_complete=args.rerun_complete,
        )
    if "ncmapss_horizon" in args.families:
        run_ncmapss_sensitivity(
            base,
            samplings=(),
            windows=(),
            seeds=parse_int_tuple(args.ncmapss_seeds),
            epochs=args.epochs,
            dry_run=args.dry_run,
            rerun_complete=args.rerun_complete,
            cell_pairs=NCMAPSS_HORIZON_MATCHED,
            output_stem="ncmapss_horizon_matched",
            experiment_prefix="ncmapss_horizon",
            display_label="NCMAPSS_HORIZON",
        )
    print("ROUND12_NEW_EXPERIMENTS_READY", flush=True)


if __name__ == "__main__":
    main()
