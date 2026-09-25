from __future__ import annotations

import argparse
from itertools import combinations
import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT = PROJECT_ROOT / "paper_outputs" / "round12_new_evidence"
OUTPUT = INPUT
METRICS = (
    "test_rmse",
    "test_nasa_per_engine",
    "test_critical_30_late_prediction_ratio",
)


def trapezoidal_integral(values: np.ndarray, levels: np.ndarray) -> float:
    if hasattr(np, "trapezoid"):
        return float(np.trapezoid(values, levels))
    return float(np.trapz(values, levels))


def response_curve_summaries(
    levels: np.ndarray,
    absolute: np.ndarray,
    clean_value: float,
) -> dict[str, float]:
    """Return distinct nonzero-range and clean-origin curve summaries."""
    levels = np.asarray(levels, dtype=float)
    absolute = np.asarray(absolute, dtype=float)
    order = np.argsort(levels)
    levels = levels[order]
    absolute = absolute[order]
    degradation = absolute - float(clean_value)
    if len(levels) < 2 or np.isclose(levels.max(), levels.min()):
        nonzero_absolute = float(absolute.mean())
        nonzero_degradation = float(degradation.mean())
    else:
        nonzero_width = float(levels.max() - levels.min())
        nonzero_absolute = trapezoidal_integral(absolute, levels) / nonzero_width
        nonzero_degradation = (
            trapezoidal_integral(degradation, levels) / nonzero_width
        )
    anchored_levels = np.concatenate(([0.0], levels))
    anchored_absolute = np.concatenate(([float(clean_value)], absolute))
    anchored_degradation = np.concatenate(([0.0], degradation))
    maximum = float(anchored_levels.max())
    if maximum <= 0:
        zero_anchored_absolute = float(clean_value)
        zero_anchored_degradation = 0.0
    else:
        zero_anchored_absolute = (
            trapezoidal_integral(anchored_absolute, anchored_levels) / maximum
        )
        zero_anchored_degradation = (
            trapezoidal_integral(anchored_degradation, anchored_levels) / maximum
        )
    return {
        "nonzero_range_absolute_mean_response": nonzero_absolute,
        "nonzero_range_clean_corrected_mean_response": nonzero_degradation,
        "zero_anchored_absolute_auc": zero_anchored_absolute,
        "zero_anchored_clean_corrected_degradation_auc": zero_anchored_degradation,
    }


def _effect_rows(frame: pd.DataFrame, index: list[str]) -> pd.DataFrame:
    paired = frame.pivot(index=index, columns="point", values=list(METRICS))
    rows = []
    for keys, values in paired.iterrows():
        key_values = keys if isinstance(keys, tuple) else (keys,)
        for metric in METRICS:
            rows.append(
                {
                    **dict(zip(index, key_values, strict=True)),
                    "metric": metric,
                    "asym_minus_rast": float(
                        values[(metric, "asym")] - values[(metric, "rast")]
                    ),
                }
            )
    return pd.DataFrame(rows)


def _formal_composite_effects() -> pd.DataFrame:
    rows = []
    for seed in (42, 123, 2024, 2025, 2026):
        for subset in ("FD001", "FD002", "FD003", "FD004"):
            values = {}
            for point, model in (("rast", "rast_gru"), ("asym", "rast_gru_v2")):
                path = (
                    PROJECT_ROOT
                    / "results"
                    / f"paper_main_v3_seed{seed}"
                    / subset
                    / model
                    / "metrics.json"
                )
                metrics = json.loads(path.read_text(encoding="utf-8-sig"))
                metrics["test_nasa_per_engine"] = (
                    float(metrics["test_nasa_score"]) / int(metrics["n_test_units"])
                )
                values[point] = metrics
            for metric in METRICS:
                rows.append(
                    {
                        "subset": subset,
                        "composite_seed": seed,
                        "metric": metric,
                        "asym_minus_rast": float(values["asym"][metric])
                        - float(values["rast"][metric]),
                    }
                )
    return pd.DataFrame(rows)


def _enumerate_level_subsets(
    effects: pd.DataFrame,
    *,
    level_column: str,
    selected_levels: tuple[int, ...],
    source_design: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    detail_rows = []
    summary_rows = []
    for (subset, metric), group in effects.groupby(["subset", "metric"]):
        available = tuple(sorted(int(value) for value in group[level_column].unique()))
        choices = list(combinations(available, len(selected_levels)))
        for choice in choices:
            part = group[group[level_column].isin(choice)]
            detail_rows.append(
                {
                    "source_design": source_design,
                    "level_type": level_column,
                    "subset": subset,
                    "metric": metric,
                    "levels": ";".join(str(value) for value in choice),
                    "is_reported_subset": tuple(choice) == tuple(selected_levels),
                    "cell_count": len(part),
                    "mean_effect": float(part["asym_minus_rast"].mean()),
                    "negative_count": int((part["asym_minus_rast"] < 0).sum()),
                    "positive_count": int((part["asym_minus_rast"] > 0).sum()),
                    "zero_count": int((part["asym_minus_rast"] == 0).sum()),
                }
            )
        detail = pd.DataFrame(detail_rows)
        current = detail[
            (detail["source_design"] == source_design)
            & (detail["subset"] == subset)
            & (detail["metric"] == metric)
        ]
        selected = current[current["is_reported_subset"]]
        summary_rows.append(
            {
                "source_design": source_design,
                "level_type": level_column,
                "subset": subset,
                "metric": metric,
                "available_level_count": len(available),
                "subset_size": len(selected_levels),
                "combination_count": len(choices),
                "mean_effect_min": float(current["mean_effect"].min()),
                "mean_effect_median": float(current["mean_effect"].median()),
                "mean_effect_max": float(current["mean_effect"].max()),
                "negative_count_min": int(current["negative_count"].min()),
                "negative_count_max": int(current["negative_count"].max()),
                "reported_subset_mean": (
                    float(selected.iloc[0]["mean_effect"]) if len(selected) else np.nan
                ),
                "reported_subset_negative_count": (
                    int(selected.iloc[0]["negative_count"]) if len(selected) else -1
                ),
                "reported_levels": ";".join(str(value) for value in selected_levels),
            }
        )
    return pd.DataFrame(detail_rows), pd.DataFrame(summary_rows)


def read_manifest(family: str) -> pd.DataFrame:
    path = INPUT / f"{family}_run_manifest.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def load_metrics(manifest: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for item in manifest.to_dict("records"):
        path = PROJECT_ROOT / str(item["run_dir"]) / "metrics.json"
        if not path.exists():
            continue
        metrics = json.loads(path.read_text(encoding="utf-8-sig"))
        # Manifest paths are the portable authority. Some historical metrics
        # files contain host-specific absolute run_dir values.
        rows.append({**metrics, **item})
    return pd.DataFrame(rows)


def add_nasa_per_engine(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    if "test_nasa_per_engine" not in result:
        result["test_nasa_per_engine"] = (
            pd.to_numeric(result["test_nasa_score"])
            / pd.to_numeric(result["n_test_units"])
        )
    return result


def independent_summary() -> dict[str, int]:
    frame = add_nasa_per_engine(load_metrics(read_manifest("independent")))
    keys = ["subset", "training_stream_seed"]
    paired = frame.pivot(index=keys, columns="point", values=list(METRICS))
    rows = []
    for subset in sorted(frame["subset"].unique()):
        part = paired.loc[subset]
        for metric in METRICS:
            difference = part[(metric, "asym")] - part[(metric, "rast")]
            rows.append(
                {
                    "subset": subset,
                    "metric": metric,
                    "stream_count": int(difference.notna().sum()),
                    "asym_minus_rast_mean": float(difference.mean()),
                    "asym_minus_rast_sd": float(difference.std(ddof=1)),
                    "negative_count": int((difference < 0).sum()),
                    "positive_count": int((difference > 0).sum()),
                    "zero_count": int((difference == 0).sum()),
                }
            )
    pd.DataFrame(rows).to_csv(
        OUTPUT / "independent_stream_paired_summary.csv",
        index=False,
    )
    frame.to_csv(OUTPUT / "independent_stream_metrics.csv", index=False)
    return {"runs": len(frame), "paired_rows": len(rows)}


def crossed_split_stream_summary() -> dict[str, int]:
    frame = add_nasa_per_engine(load_metrics(read_manifest("crossed")))
    keys = ["subset", "split_seed", "training_stream_seed"]
    effects = _effect_rows(frame, keys)
    summary_rows = []
    for (subset, metric), group in effects.groupby(["subset", "metric"]):
        grand = float(group["asym_minus_rast"].mean())
        split_means = group.groupby("split_seed")["asym_minus_rast"].mean()
        stream_means = group.groupby("training_stream_seed")["asym_minus_rast"].mean()
        centered = group.copy()
        centered["split_mean"] = centered["split_seed"].map(split_means)
        centered["stream_mean"] = centered["training_stream_seed"].map(stream_means)
        interaction = (
            centered["asym_minus_rast"]
            - centered["split_mean"]
            - centered["stream_mean"]
            + grand
        )
        summary_rows.append(
            {
                "subset": subset,
                "metric": metric,
                "split_count": int(group["split_seed"].nunique()),
                "training_stream_count": int(
                    group["training_stream_seed"].nunique()
                ),
                "cell_count": int(len(group)),
                "asym_minus_rast_mean": grand,
                "asym_minus_rast_sd": float(group["asym_minus_rast"].std(ddof=1)),
                "negative_count": int((group["asym_minus_rast"] < 0).sum()),
                "positive_count": int((group["asym_minus_rast"] > 0).sum()),
                "zero_count": int((group["asym_minus_rast"] == 0).sum()),
                "split_mean_min": float(split_means.min()),
                "split_mean_max": float(split_means.max()),
                "stream_mean_min": float(stream_means.min()),
                "stream_mean_max": float(stream_means.max()),
                "finite_split_rms": float(
                    np.sqrt(np.mean(np.square(split_means.to_numpy() - grand)))
                ),
                "finite_stream_rms": float(
                    np.sqrt(np.mean(np.square(stream_means.to_numpy() - grand)))
                ),
                "finite_interaction_rms": float(
                    np.sqrt(np.mean(np.square(interaction.to_numpy())))
                ),
                "finite_additive_residual_rms": float(
                    np.sqrt(np.mean(np.square(interaction.to_numpy())))
                ),
            }
        )
    independent = add_nasa_per_engine(load_metrics(read_manifest("independent")))
    independent_effects = _effect_rows(
        independent, ["subset", "training_stream_seed"]
    )
    stream_detail, stream_summary = _enumerate_level_subsets(
        independent_effects,
        level_column="training_stream_seed",
        selected_levels=(16180, 27182, 31415),
        source_design="fixed_split_42_ten_streams",
    )
    composite_effects = _formal_composite_effects()
    composite_detail, composite_summary = _enumerate_level_subsets(
        composite_effects,
        level_column="composite_seed",
        selected_levels=(42, 123, 2024),
        source_design="five_coupled_composite_levels",
    )
    leave_one_rows = []
    for (subset, metric), group in effects.groupby(["subset", "metric"]):
        for level_type in ("split_seed", "training_stream_seed"):
            for omitted in sorted(group[level_type].unique()):
                part = group[group[level_type] != omitted]
                leave_one_rows.append(
                    {
                        "subset": subset,
                        "metric": metric,
                        "omitted_level_type": level_type,
                        "omitted_level": int(omitted),
                        "cell_count": len(part),
                        "mean_effect": float(part["asym_minus_rast"].mean()),
                        "negative_count": int((part["asym_minus_rast"] < 0).sum()),
                        "positive_count": int((part["asym_minus_rast"] > 0).sum()),
                        "zero_count": int((part["asym_minus_rast"] == 0).sum()),
                    }
                )
    frame.to_csv(OUTPUT / "crossed_split_stream_metrics.csv", index=False)
    effects.to_csv(OUTPUT / "crossed_split_stream_effects.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(
        OUTPUT / "crossed_split_stream_summary.csv", index=False
    )
    pd.concat([stream_detail, composite_detail], ignore_index=True).to_csv(
        OUTPUT / "crossed_level_subset_effects.csv", index=False
    )
    pd.concat([stream_summary, composite_summary], ignore_index=True).to_csv(
        OUTPUT / "crossed_level_subset_sensitivity.csv", index=False
    )
    pd.DataFrame(leave_one_rows).to_csv(
        OUTPUT / "crossed_leave_one_level_sensitivity.csv", index=False
    )
    return {"runs": len(frame), "paired_effects": len(effects)}


def cmapss_preprocessing_summary() -> dict[str, int]:
    frame = add_nasa_per_engine(load_metrics(read_manifest("cmapss_preprocessing")))
    keys = [
        "subset",
        "audit_axis",
        "audit_level",
        "window_size",
        "sensor_budget",
        "training_stream_seed",
    ]
    paired = frame.pivot(index=keys, columns="point", values=list(METRICS))
    effect_rows = []
    for index, values in paired.iterrows():
        for metric in METRICS:
            effect_rows.append(
                {
                    **dict(zip(keys, index, strict=True)),
                    "metric": metric,
                    "asym_minus_rast": float(
                        values[(metric, "asym")] - values[(metric, "rast")]
                    ),
                }
            )
    effects = pd.DataFrame(effect_rows)
    paired_summary = (
        effects.groupby(
            [
                "subset",
                "audit_axis",
                "audit_level",
                "window_size",
                "sensor_budget",
                "metric",
            ],
            as_index=False,
        )
        .agg(
            training_stream_count=("training_stream_seed", "nunique"),
            asym_minus_rast_mean=("asym_minus_rast", "mean"),
            asym_minus_rast_sd=("asym_minus_rast", "std"),
            negative_count=("asym_minus_rast", lambda x: int((x < 0).sum())),
            positive_count=("asym_minus_rast", lambda x: int((x > 0).sum())),
            zero_count=("asym_minus_rast", lambda x: int((x == 0).sum())),
        )
    )
    change_rows = []
    for (subset, point, stream_seed), group in frame.groupby(
        ["subset", "point", "training_stream_seed"], sort=False
    ):
        reference = group[group["audit_axis"] == "reference"]
        if len(reference) != 1:
            raise ValueError(
                f"Expected one preprocessing reference for {subset}/{point}/{stream_seed}"
            )
        for metric_name in METRICS:
            reference_value = float(reference.iloc[0][metric_name])
            for row in group[group["audit_axis"] != "reference"].to_dict("records"):
                change_rows.append(
                    {
                        "subset": subset,
                        "point": point,
                        "metric": metric_name,
                        "training_stream_seed": stream_seed,
                        "audit_axis": row["audit_axis"],
                        "audit_level": row["audit_level"],
                        "window_size": row["window_size"],
                        "sensor_budget": row["sensor_budget"],
                        "setting_minus_reference": float(row[metric_name])
                        - reference_value,
                    }
                )
    changes = pd.DataFrame(change_rows)
    change_summary = (
        changes.groupby(
            [
                "subset",
                "point",
                "metric",
                "audit_axis",
                "audit_level",
                "window_size",
                "sensor_budget",
            ],
            as_index=False,
        )
        .agg(
            training_stream_count=("training_stream_seed", "nunique"),
            setting_minus_reference_mean=("setting_minus_reference", "mean"),
            setting_minus_reference_sd=("setting_minus_reference", "std"),
            negative_count=("setting_minus_reference", lambda x: int((x < 0).sum())),
            positive_count=("setting_minus_reference", lambda x: int((x > 0).sum())),
        )
    )
    frame.to_csv(OUTPUT / "cmapss_preprocessing_metrics.csv", index=False)
    effects.to_csv(OUTPUT / "cmapss_preprocessing_paired_effects.csv", index=False)
    paired_summary.to_csv(
        OUTPUT / "cmapss_preprocessing_paired_summary.csv", index=False
    )
    changes.to_csv(OUTPUT / "cmapss_preprocessing_reference_changes.csv", index=False)
    change_summary.to_csv(
        OUTPUT / "cmapss_preprocessing_reference_change_summary.csv", index=False
    )
    resource = frame.copy()
    resource["estimated_optimization_steps"] = (
        np.ceil(pd.to_numeric(resource["n_train_windows"]) / 128.0)
        * pd.to_numeric(resource["completed_epochs"])
    ).astype(int)
    resource_summary = resource.groupby(
        ["subset", "audit_axis", "audit_level", "window_size", "sensor_budget", "point"],
        as_index=False,
    ).agg(
        training_stream_count=("training_stream_seed", "nunique"),
        parameters=("parameters", "first"),
        train_windows_mean=("n_train_windows", "mean"),
        completed_epochs_mean=("completed_epochs", "mean"),
        estimated_optimization_steps_mean=("estimated_optimization_steps", "mean"),
        wall_clock_seconds_mean=("training_elapsed_sec", "mean"),
    )
    resource_summary.to_csv(
        OUTPUT / "cmapss_preprocessing_resource_summary.csv", index=False
    )
    leave_one_rows = []
    for group_keys, group in effects.groupby(
        ["subset", "audit_axis", "audit_level", "window_size", "sensor_budget", "metric"]
    ):
        for omitted in sorted(group["training_stream_seed"].unique()):
            retained = group[group["training_stream_seed"] != omitted]
            leave_one_rows.append(
                {
                    **dict(
                        zip(
                            ["subset", "audit_axis", "audit_level", "window_size", "sensor_budget", "metric"],
                            group_keys,
                            strict=True,
                        )
                    ),
                    "omitted_training_stream_seed": int(omitted),
                    "retained_seed_count": len(retained),
                    "mean_effect": float(retained["asym_minus_rast"].mean()),
                }
            )
    pd.DataFrame(leave_one_rows).to_csv(
        OUTPUT / "cmapss_preprocessing_leave_one_seed.csv", index=False
    )
    return {"runs": len(frame), "paired_effects": len(effects)}


def preference_summary() -> dict[str, int]:
    frame = add_nasa_per_engine(load_metrics(read_manifest("preference")))
    candidate_frame = frame[frame["point"] == "asym"].copy()
    candidate_frame["validation_score"] = pd.to_numeric(
        candidate_frame["best_selection_score"]
    )
    group_keys = ["split_seed", "training_stream_seed"]
    candidate_frame["validation_rank"] = candidate_frame.groupby(group_keys)[
        "validation_score"
    ].rank(method="min")
    candidate_frame["validation_regret"] = candidate_frame[
        "validation_score"
    ] - candidate_frame.groupby(group_keys)["validation_score"].transform("min")
    candidate_frame["selected"] = candidate_frame["validation_rank"] == 1
    candidate = (
        candidate_frame.groupby(
            ["candidate", "late_life_weight", "late_over_weight"],
            as_index=False,
        )
        .agg(
            fitted_cells=("selected", "size"),
            selection_count=("selected", "sum"),
            mean_validation_rank=("validation_rank", "mean"),
            mean_validation_regret=("validation_regret", "mean"),
            max_validation_regret=("validation_regret", "max"),
        )
        .sort_values(
            ["selection_count", "mean_validation_rank"],
            ascending=[False, True],
        )
    )
    candidate["selection_frequency"] = (
        candidate["selection_count"] / candidate["fitted_cells"]
    )
    selected = candidate_frame[candidate_frame["selected"]].copy()
    references = frame[frame["point"] == "rast_reference"][
        group_keys + list(METRICS)
    ].copy()
    downstream = selected.merge(
        references,
        on=group_keys,
        how="left",
        suffixes=("_selected_asym", "_rast_reference"),
        validate="one_to_one",
    )
    for metric in METRICS:
        downstream[f"{metric}_selected_asym_minus_rast"] = (
            downstream[f"{metric}_selected_asym"]
            - downstream[f"{metric}_rast_reference"]
        )
    downstream.to_csv(
        OUTPUT / "preference_retraining_downstream_effects.csv",
        index=False,
    )
    selected.to_csv(OUTPUT / "preference_retraining_selected_cells.csv", index=False)
    candidate.to_csv(
        OUTPUT / "preference_retraining_candidate_summary.csv",
        index=False,
    )
    frame.to_csv(OUTPUT / "preference_retraining_all_runs.csv", index=False)
    candidate_frame.to_csv(
        OUTPUT / "preference_retraining_all_candidates.csv",
        index=False,
    )
    return {
        "runs": len(frame),
        "candidate_runs": len(candidate_frame),
        "reference_runs": len(references),
        "selected_cells": len(selected),
    }


def factorial_summary() -> dict[str, int]:
    frame = add_nasa_per_engine(load_metrics(read_manifest("factorial")))
    rows = []
    for metric in METRICS:
        pivot = frame.pivot(
            index="training_stream_seed",
            columns=["backbone", "risk_factor", "consistency_factor"],
            values=metric,
        )
        for stream, values in pivot.iterrows():
            cell = {
                (backbone, risk, consistency): float(
                    values[(backbone, risk, consistency)]
                )
                for backbone in ("rast", "ocm")
                for risk in (0, 1)
                for consistency in (0, 1)
            }
            effects = {
                "backbone_main": np.mean(
                    [
                        cell[("ocm", risk, consistency)]
                        - cell[("rast", risk, consistency)]
                        for risk in (0, 1)
                        for consistency in (0, 1)
                    ]
                ),
                "risk_main": np.mean(
                    [
                        cell[(backbone, 1, consistency)]
                        - cell[(backbone, 0, consistency)]
                        for backbone in ("rast", "ocm")
                        for consistency in (0, 1)
                    ]
                ),
                "consistency_main": np.mean(
                    [
                        cell[(backbone, risk, 1)]
                        - cell[(backbone, risk, 0)]
                        for backbone in ("rast", "ocm")
                        for risk in (0, 1)
                    ]
                ),
                "backbone_x_risk": np.mean(
                    [
                        (
                            cell[("ocm", 1, consistency)]
                            - cell[("ocm", 0, consistency)]
                        )
                        - (
                            cell[("rast", 1, consistency)]
                            - cell[("rast", 0, consistency)]
                        )
                        for consistency in (0, 1)
                    ]
                ),
                "backbone_x_consistency": np.mean(
                    [
                        (
                            cell[("ocm", risk, 1)]
                            - cell[("ocm", risk, 0)]
                        )
                        - (
                            cell[("rast", risk, 1)]
                            - cell[("rast", risk, 0)]
                        )
                        for risk in (0, 1)
                    ]
                ),
                "risk_x_consistency": np.mean(
                    [
                        (
                            cell[(backbone, 1, 1)]
                            - cell[(backbone, 1, 0)]
                        )
                        - (
                            cell[(backbone, 0, 1)]
                            - cell[(backbone, 0, 0)]
                        )
                        for backbone in ("rast", "ocm")
                    ]
                ),
                "backbone_x_risk_x_consistency": (
                    (
                        cell[("ocm", 1, 1)]
                        - cell[("ocm", 1, 0)]
                        - cell[("ocm", 0, 1)]
                        + cell[("ocm", 0, 0)]
                    )
                    - (
                        cell[("rast", 1, 1)]
                        - cell[("rast", 1, 0)]
                        - cell[("rast", 0, 1)]
                        + cell[("rast", 0, 0)]
                    )
                ),
            }
            rows.extend(
                {
                    "metric": metric,
                    "training_stream_seed": stream,
                    "effect": effect,
                    "estimate": estimate,
                }
                for effect, estimate in effects.items()
            )
    effects = pd.DataFrame(rows)
    summary = (
        effects.groupby(["metric", "effect"], as_index=False)
        .agg(
            stream_count=("training_stream_seed", "nunique"),
            estimate_mean=("estimate", "mean"),
            estimate_sd=("estimate", "std"),
            negative_count=("estimate", lambda x: int((x < 0).sum())),
            positive_count=("estimate", lambda x: int((x > 0).sum())),
            zero_count=("estimate", lambda x: int((x == 0).sum())),
        )
    )
    design = frame[
        ["backbone", "risk_factor", "consistency_factor", "objective", "model"]
    ].drop_duplicates().sort_values(
        ["backbone", "risk_factor", "consistency_factor"]
    )
    design["factor_coding"] = "0=off; 1=on; effects use 0/1 differences"
    design["risk_loss"] = np.where(
        design["risk_factor"] == 1,
        "asymmetric weighted Huber + smooth late-risk",
        "symmetric Huber",
    )
    design["consistency_objective"] = np.where(
        design["consistency_factor"] == 1,
        "weight 0.05; Gaussian second view std 0.01",
        "off",
    )
    design["ocm_specific_reliability_quantile"] = "off in all cells"
    design["backbone_bundled_components"] = np.where(
        design["backbone"] == "ocm",
        "mask parsing; local trend; fusion; ordered heads",
        "RAST recurrent backbone",
    )
    frame.to_csv(OUTPUT / "factorial_cell_metrics.csv", index=False)
    design.to_csv(OUTPUT / "factorial_design_matrix.csv", index=False)
    effects.to_csv(OUTPUT / "factorial_stream_effects.csv", index=False)
    summary.to_csv(OUTPUT / "factorial_effect_summary.csv", index=False)
    return {"runs": len(frame), "effect_rows": len(effects)}


def build_lofo_tolerance_rows(
    engine_rows: pd.DataFrame,
    epsilons: tuple[float, ...] = (0.0, 5.0, 10.0),
) -> pd.DataFrame:
    """Compute tolerance-specific late-event support from engine-level errors."""
    required = {
        "held_out_family",
        "point",
        "scenario",
        "level",
        "training_stream_seed",
        "perturbation_seed",
        "unit_id",
        "true_rul",
        "error",
    }
    missing = sorted(required.difference(engine_rows.columns))
    if missing:
        raise ValueError(f"LOFO engine rows are missing columns: {missing}")
    perturbed = engine_rows[engine_rows["scenario"] != "clean"].copy()
    key = [
        "held_out_family",
        "point",
        "scenario",
        "level",
        "training_stream_seed",
        "perturbation_seed",
    ]
    duplicate_key = key + ["unit_id"]
    if perturbed.duplicated(duplicate_key).any():
        duplicate = perturbed.loc[
            perturbed.duplicated(duplicate_key, keep=False), duplicate_key
        ].head(5)
        raise ValueError(
            "LOFO engine rows contain duplicate evaluation-engine keys: "
            + duplicate.to_dict("records").__repr__()
        )
    rows: list[dict[str, object]] = []
    for keys, group in perturbed.groupby(key, sort=False):
        eligible = group[pd.to_numeric(group["true_rul"]) <= 30.0]
        errors = pd.to_numeric(eligible["error"]).to_numpy(dtype=float)
        if not len(errors):
            raise ValueError(f"LOFO evaluation has no eligible engines: {keys}")
        for epsilon in epsilons:
            excess = np.maximum(errors - float(epsilon), 0.0)
            event = errors > float(epsilon)
            event_count = int(event.sum())
            rows.append(
                {
                    **dict(zip(key, keys, strict=True)),
                    "epsilon": float(epsilon),
                    "eligible_engine_count": int(len(errors)),
                    "late_event_count": event_count,
                    "late_ratio": float(event_count / len(errors)),
                    "zimle": float(excess.mean()),
                    "cmle": float(excess[event].mean()) if event_count else np.nan,
                    "cmle_defined": bool(event_count),
                }
            )
    return pd.DataFrame(rows)


def summarize_lofo_tolerance_rows(tolerance: pd.DataFrame) -> pd.DataFrame:
    """Summarize integer event support without hiding zero-event evaluations."""
    keys = ["held_out_family", "point", "scenario", "level", "epsilon"]
    rows: list[dict[str, object]] = []
    for values, group in tolerance.groupby(keys, sort=False):
        counts = pd.to_numeric(group["late_event_count"])
        cmle = pd.to_numeric(group["cmle"], errors="coerce")
        evaluation_count = int(len(group))
        zero_count = int((counts == 0).sum())
        undefined_count = int(cmle.isna().sum())
        rows.append(
            {
                **dict(zip(keys, values, strict=True)),
                "training_stream_count": int(group["training_stream_seed"].nunique()),
                "perturbation_seed_count": int(group["perturbation_seed"].nunique()),
                "evaluation_count": evaluation_count,
                "eligible_engines_per_evaluation_median": float(
                    group["eligible_engine_count"].median()
                ),
                "eligible_engines_per_evaluation_min": int(
                    group["eligible_engine_count"].min()
                ),
                "eligible_engines_per_evaluation_max": int(
                    group["eligible_engine_count"].max()
                ),
                "late_event_count_mean": float(counts.mean()),
                "late_event_count_median": float(counts.median()),
                "late_event_count_q1": float(counts.quantile(0.25)),
                "late_event_count_q3": float(counts.quantile(0.75)),
                "late_event_count_min": int(counts.min()),
                "late_event_count_max": int(counts.max()),
                "zero_event_evaluation_count": zero_count,
                "zero_event_evaluation_rate": float(zero_count / evaluation_count),
                "late_ratio_mean": float(group["late_ratio"].mean()),
                "late_ratio_sd": float(group["late_ratio"].std(ddof=1)),
                "zimle_mean": float(group["zimle"].mean()),
                "zimle_median": float(group["zimle"].median()),
                "cmle_mean": float(cmle.mean()) if cmle.notna().any() else np.nan,
                "cmle_median": float(cmle.median()) if cmle.notna().any() else np.nan,
                "undefined_cmle_evaluation_count": undefined_count,
                "undefined_cmle_evaluation_rate": float(
                    undefined_count / evaluation_count
                ),
            }
        )
    return pd.DataFrame(rows)


def summarize_lofo_tolerance_by_stream(tolerance: pd.DataFrame) -> pd.DataFrame:
    """Keep the five training streams visible while pooling perturbation seeds."""
    keys = [
        "held_out_family",
        "point",
        "scenario",
        "level",
        "epsilon",
        "training_stream_seed",
    ]
    rows: list[dict[str, object]] = []
    for values, group in tolerance.groupby(keys, sort=False):
        counts = pd.to_numeric(group["late_event_count"])
        cmle = pd.to_numeric(group["cmle"], errors="coerce")
        evaluation_count = int(len(group))
        rows.append(
            {
                **dict(zip(keys, values, strict=True)),
                "perturbation_seed_count": int(group["perturbation_seed"].nunique()),
                "evaluation_count": evaluation_count,
                "eligible_engine_count": int(group["eligible_engine_count"].median()),
                "late_event_count_mean": float(counts.mean()),
                "late_event_count_min": int(counts.min()),
                "late_event_count_max": int(counts.max()),
                "zero_event_evaluation_count": int((counts == 0).sum()),
                "late_ratio_mean": float(group["late_ratio"].mean()),
                "zimle_mean": float(group["zimle"].mean()),
                "cmle_mean": float(cmle.mean()) if cmle.notna().any() else np.nan,
                "undefined_cmle_evaluation_count": int(cmle.isna().sum()),
            }
        )
    return pd.DataFrame(rows)


def lofo_summary() -> dict[str, int]:
    training = add_nasa_per_engine(load_metrics(read_manifest("lofo")))
    perturbation_path = INPUT / "lofo_held_out_family_perturbation_rows.csv"
    if not perturbation_path.exists():
        raise FileNotFoundError(perturbation_path)
    perturbation = pd.read_csv(perturbation_path)
    unit_counts = training[["run_dir", "n_test_units"]].drop_duplicates("run_dir")
    perturbation = perturbation.drop(columns=["n_test_units"], errors="ignore").merge(
        unit_counts,
        on="run_dir",
        how="left",
        validate="many_to_one",
    )
    if perturbation["n_test_units"].isna().any():
        raise ValueError("LOFO perturbation rows could not be matched to test-engine counts")
    perturbation["nasa_per_engine"] = (
        pd.to_numeric(perturbation["nasa_score"])
        / pd.to_numeric(perturbation["n_test_units"])
    )
    clean = (
        perturbation[perturbation["scenario"] == "clean"]
        .sort_values("perturbation_seed")
        .drop_duplicates(
            ["held_out_family", "point", "training_stream_seed"],
            keep="first",
        )
    )
    clean_lookup = clean.set_index(
        ["held_out_family", "point", "training_stream_seed"]
    )
    perturbed = perturbation[perturbation["scenario"] != "clean"].copy()
    metrics = [
        "rmse",
        "nasa_per_engine",
        "critical_30_late_prediction_ratio",
    ]
    summary = (
        perturbed.groupby(
            ["held_out_family", "point", "scenario", "level"],
            as_index=False,
        )
        .agg(
            training_stream_count=("training_stream_seed", "nunique"),
            perturbation_seed_count=("perturbation_seed", "nunique"),
            **{f"{metric}_mean": (metric, "mean") for metric in metrics},
            **{f"{metric}_sd": (metric, "std") for metric in metrics},
        )
    )
    paired_rows = []
    keys = [
        "held_out_family",
        "scenario",
        "level",
        "training_stream_seed",
        "perturbation_seed",
    ]
    for metric in metrics:
        pivot = perturbed.pivot_table(
            index=keys,
            columns="point",
            values=metric,
            aggfunc="first",
        ).dropna(subset=["asym", "rast"])
        difference = pivot["asym"] - pivot["rast"]
        for index, value in difference.items():
            paired_rows.append(
                {
                    **dict(zip(keys, index, strict=True)),
                    "metric": metric,
                    "asym_minus_rast": value,
                }
            )
    paired = pd.DataFrame(paired_rows)
    stream_effects = (
        paired.groupby(
            [
                "held_out_family",
                "scenario",
                "level",
                "metric",
                "training_stream_seed",
            ],
            as_index=False,
        )
        .agg(
            perturbation_seed_count=("perturbation_seed", "nunique"),
            asym_minus_rast=("asym_minus_rast", "mean"),
        )
    )
    paired_summary = (
        stream_effects.groupby(
            ["held_out_family", "scenario", "level", "metric"],
            as_index=False,
        )
        .agg(
            training_stream_count=("training_stream_seed", "nunique"),
            asym_minus_rast_mean=("asym_minus_rast", "mean"),
            asym_minus_rast_sd=("asym_minus_rast", "std"),
            negative_count=("asym_minus_rast", lambda x: int((x < 0).sum())),
            positive_count=("asym_minus_rast", lambda x: int((x > 0).sum())),
            zero_count=("asym_minus_rast", lambda x: int((x == 0).sum())),
        )
    )
    curve_rows: list[dict[str, object]] = []
    point_group_keys = [
        "held_out_family",
        "point",
        "scenario",
        "training_stream_seed",
    ]
    for keys, group in perturbed.groupby(point_group_keys, sort=False):
        family, point, scenario, stream_seed = keys
        for metric in metrics:
            by_level = group.groupby("level", as_index=False)[metric].mean().sort_values("level")
            levels = by_level["level"].to_numpy(dtype=float)
            absolute = by_level[metric].to_numpy(dtype=float)
            clean_metric = {
                "rmse": "rmse",
                "nasa_per_engine": "nasa_per_engine",
                "critical_30_late_prediction_ratio": (
                    "critical_30_late_prediction_ratio"
                ),
            }[metric]
            clean_value = float(
                clean_lookup.loc[(family, point, stream_seed), clean_metric]
            )
            curve = response_curve_summaries(levels, absolute, clean_value)
            curve_rows.append(
                {
                    "held_out_family": family,
                    "point": point,
                    "scenario": scenario,
                    "metric": metric,
                    "training_stream_seed": stream_seed,
                    "nonzero_level_count": len(levels),
                    "minimum_level": float(levels.min()),
                    "maximum_level": float(levels.max()),
                    "clean_metric": clean_value,
                    **curve,
                    # Canonical AUC columns now use the declared clean-origin domain.
                    "normalized_absolute_corrupted_auc": curve[
                        "zero_anchored_absolute_auc"
                    ],
                    "normalized_clean_corrected_degradation_auc": curve[
                        "zero_anchored_clean_corrected_degradation_auc"
                    ],
                }
            )
    point_curves = pd.DataFrame(curve_rows)
    scenario_auc_rows = []
    for keys, group in point_curves.groupby(
        ["held_out_family", "scenario", "metric", "training_stream_seed"],
        sort=False,
    ):
        values = group.set_index("point")
        scenario_auc_rows.append(
            {
                "held_out_family": keys[0],
                "scenario": keys[1],
                "metric": keys[2],
                "training_stream_seed": keys[3],
                "nonzero_level_count": int(values["nonzero_level_count"].max()),
                "minimum_level": float(values["minimum_level"].min()),
                "maximum_level": float(values["maximum_level"].max()),
                "absolute_auc_asym_minus_rast": float(
                    values.loc["asym", "normalized_absolute_corrupted_auc"]
                    - values.loc["rast", "normalized_absolute_corrupted_auc"]
                ),
                "degradation_auc_asym_minus_rast": float(
                    values.loc[
                        "asym", "normalized_clean_corrected_degradation_auc"
                    ]
                    - values.loc[
                        "rast", "normalized_clean_corrected_degradation_auc"
                    ]
                ),
                "nonzero_absolute_mean_asym_minus_rast": float(
                    values.loc["asym", "nonzero_range_absolute_mean_response"]
                    - values.loc["rast", "nonzero_range_absolute_mean_response"]
                ),
                "nonzero_degradation_mean_asym_minus_rast": float(
                    values.loc[
                        "asym", "nonzero_range_clean_corrected_mean_response"
                    ]
                    - values.loc[
                        "rast", "nonzero_range_clean_corrected_mean_response"
                    ]
                ),
            }
        )
    scenario_auc = pd.DataFrame(scenario_auc_rows)
    family_stream = (
        scenario_auc.groupby(
            ["held_out_family", "metric", "training_stream_seed"],
            as_index=False,
        )
        .agg(
            scenario_count=("scenario", "nunique"),
            absolute_auc_asym_minus_rast=("absolute_auc_asym_minus_rast", "mean"),
            degradation_auc_asym_minus_rast=(
                "degradation_auc_asym_minus_rast",
                "mean",
            ),
            nonzero_absolute_mean_asym_minus_rast=(
                "nonzero_absolute_mean_asym_minus_rast",
                "mean",
            ),
            nonzero_degradation_mean_asym_minus_rast=(
                "nonzero_degradation_mean_asym_minus_rast",
                "mean",
            ),
        )
    )
    family_summary_rows = []
    for (family, metric), group in family_stream.groupby(
        ["held_out_family", "metric"]
    ):
        row = {
            "held_out_family": family,
            "metric": metric,
            "training_stream_count": int(group["training_stream_seed"].nunique()),
        }
        for estimand in (
            "absolute_auc_asym_minus_rast",
            "degradation_auc_asym_minus_rast",
            "nonzero_absolute_mean_asym_minus_rast",
            "nonzero_degradation_mean_asym_minus_rast",
        ):
            values = group[estimand]
            row[f"{estimand}_mean"] = float(values.mean())
            row[f"{estimand}_sd"] = float(values.std(ddof=1))
            row[f"{estimand}_negative_count"] = int((values < 0).sum())
            row[f"{estimand}_positive_count"] = int((values > 0).sum())
            row[f"{estimand}_zero_count"] = int((values == 0).sum())
        family_summary_rows.append(row)
    family_summary = pd.DataFrame(family_summary_rows)

    clean_values = clean[
        [
            "held_out_family",
            "point",
            "training_stream_seed",
            "rmse",
            "nasa_per_engine",
            "critical_30_late_prediction_ratio",
        ]
    ].melt(
        id_vars=["held_out_family", "point", "training_stream_seed"],
        var_name="metric",
        value_name="clean_metric",
    )
    level_degradation = perturbed.groupby(
        [
            "held_out_family",
            "point",
            "scenario",
            "level",
            "training_stream_seed",
        ],
        as_index=False,
    )[metrics].mean().melt(
        id_vars=[
            "held_out_family",
            "point",
            "scenario",
            "level",
            "training_stream_seed",
        ],
        var_name="metric",
        value_name="corrupted_metric",
    ).merge(
        clean_values,
        on=["held_out_family", "point", "training_stream_seed", "metric"],
        how="left",
        validate="many_to_one",
    )
    level_degradation["clean_corrected_degradation"] = (
        level_degradation["corrupted_metric"] - level_degradation["clean_metric"]
    )
    level_paired = level_degradation.pivot_table(
        index=[
            "held_out_family",
            "scenario",
            "level",
            "metric",
            "training_stream_seed",
        ],
        columns="point",
        values="clean_corrected_degradation",
        aggfunc="first",
    ).reset_index()
    level_paired["degradation_asym_minus_rast"] = (
        level_paired["asym"] - level_paired["rast"]
    )
    level_paired_summary = level_paired.groupby(
        ["held_out_family", "scenario", "level", "metric"], as_index=False
    ).agg(
        training_stream_count=("training_stream_seed", "nunique"),
        degradation_asym_minus_rast_mean=("degradation_asym_minus_rast", "mean"),
        degradation_asym_minus_rast_sd=("degradation_asym_minus_rast", "std"),
        negative_count=("degradation_asym_minus_rast", lambda x: int((x < 0).sum())),
        positive_count=("degradation_asym_minus_rast", lambda x: int((x > 0).sum())),
        zero_count=("degradation_asym_minus_rast", lambda x: int((x == 0).sum())),
    )
    engine_path = INPUT / "lofo_held_out_family_engine_rows.csv"
    if not engine_path.exists():
        raise FileNotFoundError(
            f"{engine_path} is required for tolerance-specific LOFO severity. "
            "Run run_round12_new_experiments.py --families lofo "
            "--lofo-evaluation-only first."
        )
    engine_frame = pd.read_csv(engine_path)
    tolerance = build_lofo_tolerance_rows(engine_frame)
    support_summary = summarize_lofo_tolerance_rows(tolerance)
    support_by_stream = summarize_lofo_tolerance_by_stream(tolerance)
    tolerance_stream = tolerance.groupby(
        [
            "held_out_family",
            "point",
            "scenario",
            "level",
            "epsilon",
            "training_stream_seed",
        ],
        as_index=False,
    )["late_ratio"].mean()
    tolerance_paired = tolerance_stream.pivot_table(
        index=[
            "held_out_family",
            "scenario",
            "level",
            "epsilon",
            "training_stream_seed",
        ],
        columns="point",
        values="late_ratio",
        aggfunc="first",
    ).reset_index()
    tolerance_paired["asym_minus_rast"] = (
        tolerance_paired["asym"] - tolerance_paired["rast"]
    )
    tolerance_paired_summary = tolerance_paired.groupby(
        ["held_out_family", "scenario", "level", "epsilon"], as_index=False
    ).agg(
        training_stream_count=("training_stream_seed", "nunique"),
        asym_minus_rast_mean=("asym_minus_rast", "mean"),
        asym_minus_rast_sd=("asym_minus_rast", "std"),
        negative_count=("asym_minus_rast", lambda x: int((x < 0).sum())),
        positive_count=("asym_minus_rast", lambda x: int((x > 0).sum())),
        zero_count=("asym_minus_rast", lambda x: int((x == 0).sum())),
    )
    training.to_csv(OUTPUT / "lofo_training_metrics.csv", index=False)
    clean.to_csv(OUTPUT / "lofo_clean_context_deduplicated.csv", index=False)
    summary.to_csv(OUTPUT / "lofo_perturbation_summary.csv", index=False)
    paired.to_csv(OUTPUT / "lofo_paired_effect_rows.csv", index=False)
    stream_effects.to_csv(
        OUTPUT / "lofo_training_stream_effects.csv",
        index=False,
    )
    paired_summary.to_csv(OUTPUT / "lofo_paired_effect_summary.csv", index=False)
    scenario_auc.to_csv(OUTPUT / "lofo_scenario_auc_stream_effects.csv", index=False)
    point_curves.to_csv(OUTPUT / "lofo_point_scenario_auc.csv", index=False)
    family_stream.to_csv(OUTPUT / "lofo_family_stream_effects.csv", index=False)
    family_summary.to_csv(OUTPUT / "lofo_family_effect_summary.csv", index=False)
    level_degradation.to_csv(OUTPUT / "lofo_level_degradation.csv", index=False)
    level_paired.to_csv(OUTPUT / "lofo_level_paired_degradation_effects.csv", index=False)
    level_paired_summary.to_csv(
        OUTPUT / "lofo_level_paired_degradation_summary.csv", index=False
    )
    tolerance.to_csv(OUTPUT / "lofo_late_event_rows.csv", index=False)
    support_summary.to_csv(OUTPUT / "lofo_late_event_support.csv", index=False)
    support_by_stream.to_csv(
        OUTPUT / "lofo_late_event_support_by_training_stream.csv", index=False
    )
    tolerance_paired.to_csv(
        OUTPUT / "lofo_lpr_tolerance_stream_effects.csv", index=False
    )
    tolerance_paired_summary.to_csv(
        OUTPUT / "lofo_lpr_tolerance_summary.csv", index=False
    )
    pd.DataFrame(
        [
            {
                "metric": "rmse",
                "unit": "RUL cycles",
                "aggregation": (
                    "zero-anchored normalized AUC over level 0 to family maximum; "
                    "nonzero-range mean response retained separately"
                ),
            },
            {
                "metric": "nasa_per_engine",
                "unit": "NASA score per official test engine",
                "aggregation": "raw NASA score divided by n_test_units",
            },
            {
                "metric": "critical_30_late_prediction_ratio",
                "unit": "proportion among engines with true RUL <= 30",
                "aggregation": "engine-level binary late-overprediction mean",
            },
        ]
    ).to_csv(OUTPUT / "lofo_metric_schema.csv", index=False)
    return {
        "runs": len(training),
        "stored_rows": len(perturbation),
        "engine_rows": len(engine_frame),
        "perturbed_rows": len(perturbed),
        "deduplicated_clean_rows": len(clean),
        "stream_support_rows": len(support_by_stream),
    }


def retrospective_governance_dashboard() -> dict[str, int]:
    """Build a single, machine-readable map of all retrospective directional families."""
    rows: list[dict[str, object]] = []

    def add_rows(
        family_id: str,
        frame: pd.DataFrame,
        *,
        metric_column: str,
        value_column: str,
        context_columns: list[str],
        statistical_unit: str,
        design_status: str,
        headline_impact: str,
        source_file: str,
        directional: bool = True,
    ) -> None:
        for metric, group in frame.groupby(metric_column, sort=False):
            values = pd.to_numeric(group[value_column], errors="raise").to_numpy(float)
            contexts = group[context_columns].astype(str).agg(" | ".join, axis=1)
            minimum_index = int(np.argmin(values))
            maximum_index = int(np.argmax(values))
            rows.append(
                {
                    "family_id": family_id,
                    "metric": str(metric),
                    "design_status": design_status,
                    "statistical_unit": statistical_unit,
                    "completed_direction_count": int(len(values)),
                    "negative_count": int((values < 0).sum()) if directional else 0,
                    "positive_count": int((values > 0).sum()) if directional else 0,
                    "zero_count": int((values == 0).sum()) if directional else 0,
                    "directional_contrast": bool(directional),
                    "estimate_min": float(values[minimum_index]),
                    "estimate_max": float(values[maximum_index]),
                    "most_negative_context": contexts.iloc[minimum_index],
                    "most_positive_context": contexts.iloc[maximum_index],
                    "headline_impact": headline_impact,
                    "source_file": source_file,
                }
            )

    fixed_path = PROJECT_ROOT / "paper_outputs" / "advanced_evidence" / "paired_fixed_task_bootstrap.csv"
    fixed = pd.read_csv(fixed_path)
    add_rows(
        "E-FIXED-12",
        fixed,
        metric_column="metric",
        value_column="mean_difference",
        context_columns=["subset"],
        statistical_unit="five coupled composite-seed levels and matched test engines",
        design_status="result-informed fixed-task estimation",
        headline_impact="descriptive accuracy-late-overprediction trade-off; no calibrated confirmatory role",
        source_file=str(fixed_path.relative_to(PROJECT_ROOT)).replace("\\", "/"),
    )

    independent = pd.read_csv(OUTPUT / "independent_stream_metrics.csv")
    independent_effects = _effect_rows(
        independent, ["subset", "training_stream_seed"]
    )
    add_rows(
        "D-STREAM-10",
        independent_effects,
        metric_column="metric",
        value_column="asym_minus_rast",
        context_columns=["subset", "training_stream_seed"],
        statistical_unit="training stream at fixed engine split 42",
        design_status="retrospective fixed-split stream sensitivity",
        headline_impact="direction changes across independently seeded training streams",
        source_file="paper_outputs/round12_new_evidence/independent_stream_metrics.csv",
    )

    crossed = pd.read_csv(OUTPUT / "crossed_split_stream_effects.csv")
    add_rows(
        "D-CROSSED-3X3",
        crossed,
        metric_column="metric",
        value_column="asym_minus_rast",
        context_columns=["subset", "split_seed", "training_stream_seed"],
        statistical_unit="one selected split-level by stream-level cell",
        design_status="result-informed finite 3x3 crossing",
        headline_impact="finite-cell dependence; does not identify population split or stream effects",
        source_file="paper_outputs/round12_new_evidence/crossed_split_stream_effects.csv",
    )

    preprocessing = pd.read_csv(OUTPUT / "cmapss_preprocessing_paired_effects.csv")
    add_rows(
        "D-PREPROCESS-FD004",
        preprocessing,
        metric_column="metric",
        value_column="asym_minus_rast",
        context_columns=["audit_axis", "audit_level", "training_stream_seed"],
        statistical_unit="composite-seed level within one OFAT configuration",
        design_status="retrospective joint protocol-and-training-dynamics sensitivity",
        headline_impact="window and sensor-budget choices reverse some configuration contrasts",
        source_file="paper_outputs/round12_new_evidence/cmapss_preprocessing_paired_effects.csv",
    )

    preference = pd.read_csv(OUTPUT / "preference_retraining_downstream_effects.csv")
    preference_long = preference.melt(
        id_vars=["split_seed", "training_stream_seed", "candidate"],
        value_vars=[
            "test_rmse_selected_asym_minus_rast",
            "test_nasa_per_engine_selected_asym_minus_rast",
            "test_critical_30_late_prediction_ratio_selected_asym_minus_rast",
        ],
        var_name="metric",
        value_name="asym_minus_rast",
    )
    preference_long["metric"] = preference_long["metric"].map(
        {
            "test_rmse_selected_asym_minus_rast": "test_rmse",
            "test_nasa_per_engine_selected_asym_minus_rast": "test_nasa_per_engine",
            "test_critical_30_late_prediction_ratio_selected_asym_minus_rast": "test_critical_30_late_prediction_ratio",
        }
    )
    add_rows(
        "D-PREFERENCE-3X3",
        preference_long,
        metric_column="metric",
        value_column="asym_minus_rast",
        context_columns=["split_seed", "training_stream_seed", "candidate"],
        statistical_unit="validation-selected split-level by stream-level cell",
        design_status="retrospective selection-operation audit",
        headline_impact="no candidate is selected in more than two of nine cells",
        source_file="paper_outputs/round12_new_evidence/preference_retraining_downstream_effects.csv",
    )

    factorial = pd.read_csv(OUTPUT / "factorial_stream_effects.csv")
    add_rows(
        "D-FACTORIAL-SHARED",
        factorial,
        metric_column="metric",
        value_column="estimate",
        context_columns=["effect", "training_stream_seed"],
        statistical_unit="training stream within restricted shared-objective design",
        design_status="limited bundled-architecture by shared-objective factorial",
        headline_impact="only declared shared-factor contrasts are identified; OCM-specific paths remain bundled",
        source_file="paper_outputs/round12_new_evidence/factorial_stream_effects.csv",
    )

    lofo = pd.read_csv(OUTPUT / "lofo_family_stream_effects.csv")
    add_rows(
        "D-LOFO-FD004",
        lofo,
        metric_column="metric",
        value_column="degradation_auc_asym_minus_rast",
        context_columns=["held_out_family", "training_stream_seed"],
        statistical_unit="training stream after averaging five perturbation seeds",
        design_status="retrospective zero-anchored clean-corrected augmentation-overlap sensitivity",
        headline_impact="direction depends on held-out family and estimand; no unseen-family transfer claim",
        source_file="paper_outputs/round12_new_evidence/lofo_family_stream_effects.csv",
    )

    for family_id, file_name, status in (
        (
            "D-NCMAPSS-GRID",
            "ncmapss_sampling_window_summary.csv",
            "strongly downsampled sampling-window computational proxy",
        ),
        (
            "D-NCMAPSS-SPAN",
            "ncmapss_horizon_matched_summary.csv",
            "fixed-source-span computational proxy",
        ),
    ):
        frame = pd.read_csv(OUTPUT / file_name)
        metric_columns = {
            "unit_macro_rmse_mean": "unit_macro_rmse",
            "unit_macro_nasa_per_window_mean": "unit_macro_nasa_per_window",
            "unit_macro_lpr30_mean": "unit_macro_lpr30",
        }
        long = frame.melt(
            id_vars=["sampling_interval", "sensitivity_window_size"],
            value_vars=list(metric_columns),
            var_name="metric",
            value_name="observed_value",
        )
        long["metric"] = long["metric"].map(metric_columns)
        add_rows(
            family_id,
            long,
            metric_column="metric",
            value_column="observed_value",
            context_columns=["sampling_interval", "sensitivity_window_size"],
            statistical_unit="same three official N-CMAPSS test units",
            design_status=status,
            headline_impact="representation dependence only; not an external validation or paired model contrast",
            source_file=f"paper_outputs/round12_new_evidence/{file_name}",
            directional=False,
        )

    dashboard = pd.DataFrame(rows)
    dashboard.to_csv(OUTPUT / "retrospective_family_dashboard.csv", index=False)
    workflow = pd.DataFrame(
        [
            {
                "step": 1,
                "component": "Claim",
                "mandatory_rule": "state the bounded empirical claim before selecting an exhibit",
                "recommended_record": "claim identifier and prohibited extrapolations",
                "retrospective_only_boundary": "label result-informed timing explicitly",
            },
            {
                "step": 2,
                "component": "Estimand",
                "mandatory_rule": "define outcome, aggregation unit, pairing, threshold, and direction",
                "recommended_record": "machine-readable metric schema",
                "retrospective_only_boundary": "do not relabel a post-result estimand as confirmatory",
            },
            {
                "step": 3,
                "component": "Design status",
                "mandatory_rule": "separate prospective, result-informed, sensitivity, and runtime roles",
                "recommended_record": "analysis-family registry and chronology",
                "retrospective_only_boundary": "completed levels define a finite design, not a sampled population",
            },
            {
                "step": 4,
                "component": "Uncertainty role",
                "mandatory_rule": "match uncertainty language to the number and independence of design levels",
                "recommended_record": "seed-level effects, sign counts, event support, and calibration audit",
                "retrospective_only_boundary": "uncalibrated intervals and p-like values remain supplementary diagnostics",
            },
            {
                "step": 5,
                "component": "Artifact",
                "mandatory_rule": "map each claim to source rows, code, environment, and immutable checksums",
                "recommended_record": "exhibit-to-script map and disposable-workspace verification",
                "retrospective_only_boundary": "local checksums do not imply a public timestamp or DOI",
            },
            {
                "step": 6,
                "component": "Allowed wording",
                "mandatory_rule": "report the complete direction range and state what the design cannot identify",
                "recommended_record": "dashboard-derived most-negative and most-positive contexts",
                "retrospective_only_boundary": "use protocol-conditional audit language, not dominance or generalization",
            },
        ]
    )
    workflow.to_csv(OUTPUT / "governance_workflow.csv", index=False)
    return {"dashboard_rows": len(dashboard), "workflow_steps": len(workflow)}


def ncmapss_summary_for(stem: str) -> dict[str, int]:
    path = INPUT / f"{stem}_seed_metrics.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    frame = pd.read_csv(path)
    frame["source_record_span"] = 1 + (
        pd.to_numeric(frame["sensitivity_window_size"]) - 1
    ) * pd.to_numeric(frame["sampling_interval"])
    metrics = [
        "unit_macro_rmse",
        "unit_macro_mae",
        "unit_macro_nasa_per_window",
        "unit_macro_lpr30",
    ]
    summary = (
        frame.groupby(
            ["sampling_interval", "sensitivity_window_size"],
            as_index=False,
        )
        .agg(
            seed_count=("seed", "nunique"),
            train_windows=("n_train_windows", "first"),
            test_windows=("n_test_windows", "first"),
            source_record_span=("source_record_span", "first"),
            **{f"{metric}_mean": (metric, "mean") for metric in metrics},
            **{f"{metric}_sd": (metric, "std") for metric in metrics},
        )
    )
    summary["sampled_window_overlap_fraction"] = (
        pd.to_numeric(summary["sensitivity_window_size"]) - 1.0
    ) / pd.to_numeric(summary["sensitivity_window_size"])
    summary["train_nonoverlap_equivalent"] = (
        pd.to_numeric(summary["train_windows"])
        / pd.to_numeric(summary["sensitivity_window_size"])
    )
    summary["optimization_exposure_note"] = (
        "descriptive only: sampling changes retained records, overlapping windows, "
        "and optimizer updates"
    )
    summary.to_csv(
        OUTPUT / f"{stem}_summary.csv",
        index=False,
    )
    return {"runs": len(frame), "cells": len(summary)}


def ncmapss_summary() -> dict[str, int]:
    return ncmapss_summary_for("ncmapss_sampling_window")


def ncmapss_horizon_summary() -> dict[str, int]:
    return ncmapss_summary_for("ncmapss_horizon_matched")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Analyze completed Round-12 experiment families."
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
            "dashboard",
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
            "dashboard",
        ),
    )
    args = parser.parse_args()
    functions = {
        "independent": independent_summary,
        "crossed": crossed_split_stream_summary,
        "cmapss_preprocessing": cmapss_preprocessing_summary,
        "preference": preference_summary,
        "factorial": factorial_summary,
        "lofo": lofo_summary,
        "ncmapss": ncmapss_summary,
        "ncmapss_horizon": ncmapss_horizon_summary,
        "dashboard": retrospective_governance_dashboard,
    }
    report = {}
    for family in args.families:
        report[family] = functions[family]()
        print(f"[ROUND12 ANALYZE] {family}: {report[family]}", flush=True)
    (OUTPUT / "round12_analysis_manifest.json").write_text(
        json.dumps(report, indent=2),
        encoding="utf-8",
    )
    print("ROUND12_NEW_EVIDENCE_ANALYSIS_PASS")


if __name__ == "__main__":
    main()
