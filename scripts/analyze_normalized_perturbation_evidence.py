"""Build normalized-coordinate perturbation and paired benchmark diagnostics."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def trapezoidal_integral(
    y: np.ndarray,
    x: np.ndarray | None = None,
    dx: float = 1.0,
    axis: int = -1,
) -> np.ndarray | np.floating:
    if hasattr(np, "trapezoid"):
        return np.trapezoid(y, x=x, dx=dx, axis=axis)
    return np.trapz(y, x=x, dx=dx, axis=axis)


TRAPEZOID = trapezoidal_integral
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_advanced_evidence import MODEL_DISPLAY, nasa_contribution
from build_stress_pair_extension import SCENARIOS
from run_formal_benchmark import FORMAL_BENCHMARK_SEEDS, FORMAL_BENCHMARK_SUBSETS


PROPOSED = "rast_gru_v2"
DIRECT_PREDECESSOR = "rast_gru"
PRIMARY_METRICS = ("rmse", "nasa_per_engine", "lpr30")
TOLERANCES = (0.0, 0.5, 1.0, 2.0, 5.0)
RUL_THRESHOLDS = (20.0, 30.0, 40.0, 50.0)
STRESS_METRICS = (
    "rmse",
    "critical_30_late_prediction_ratio",
    "critical_30_mean_late_excess",
)


def holm_adjust(p_values: np.ndarray) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    order = np.argsort(values)
    adjusted = np.empty(len(values), dtype=float)
    running = 0.0
    total = len(values)
    for rank, index in enumerate(order):
        running = max(running, (total - rank) * values[index])
        adjusted[index] = min(running, 1.0)
    return adjusted


def engine_count_matrix(draws: np.ndarray, engine_count: int) -> np.ndarray:
    counts = np.empty((engine_count, draws.shape[0]), dtype=np.int16)
    for rep, draw in enumerate(draws):
        counts[:, rep] = np.bincount(draw, minlength=engine_count)
    return counts


def centered_bootstrap_p(samples: np.ndarray, observed: float) -> float:
    centered = np.asarray(samples, dtype=float) - float(observed)
    return float((np.sum(np.abs(centered) >= abs(float(observed))) + 1) / (len(centered) + 1))


def fixed_task_arrays(results_dir: Path, subset: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    truths = []
    proposed = []
    predecessor = []
    expected_units: np.ndarray | None = None
    for seed in FORMAL_BENCHMARK_SEEDS:
        root = results_dir / f"paper_main_v3_seed{seed}" / subset
        left = pd.read_csv(root / PROPOSED / "test_predictions.csv")
        right = pd.read_csv(root / DIRECT_PREDECESSOR / "test_predictions.csv")
        merged = left.merge(
            right,
            on=["unit_id", "true_rul"],
            suffixes=("_proposed", "_predecessor"),
            validate="one_to_one",
        ).sort_values("unit_id")
        units = merged["unit_id"].to_numpy(int)
        if expected_units is None:
            expected_units = units
        elif not np.array_equal(expected_units, units):
            raise ValueError(f"Official test-engine order differs across seeds for {subset}")
        truths.append(merged["true_rul"].to_numpy(float))
        proposed.append(merged["pred_rul_proposed"].to_numpy(float))
        predecessor.append(merged["pred_rul_predecessor"].to_numpy(float))
    truth = np.stack(truths)
    if not np.allclose(truth, truth[0]):
        raise ValueError(f"True RUL differs across seeds for {subset}")
    return truth[0], np.stack(proposed), np.stack(predecessor)


def sampled_seed_metrics(
    truth: np.ndarray,
    predictions: np.ndarray,
    counts: np.ndarray,
    metric: str,
    *,
    rul_threshold: float = 30.0,
    tolerance: float = 0.0,
) -> np.ndarray:
    error = predictions - truth[None, :]
    if metric == "rmse":
        return np.sqrt((error**2) @ counts / counts.sum(axis=0)[None, :])
    if metric == "nasa_per_engine":
        return nasa_contribution(error) @ counts / counts.sum(axis=0)[None, :]
    if metric == "lpr30":
        critical = (truth <= float(rul_threshold)).astype(float)
        denominator = np.maximum(critical @ counts, 1.0)
        late = ((error > float(tolerance)) & (critical[None, :] > 0)).astype(float)
        return late @ counts / denominator[None, :]
    raise ValueError(metric)


def crossed_fixed_task_bootstrap(
    results_dir: Path,
    *,
    reps: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    tolerance_rows: list[dict[str, object]] = []
    for subset_index, subset in enumerate(FORMAL_BENCHMARK_SUBSETS):
        truth, proposed, predecessor = fixed_task_arrays(results_dir, subset)
        rng = np.random.default_rng(random_seed + subset_index * 1009)
        seed_draw = rng.integers(0, len(FORMAL_BENCHMARK_SEEDS), size=(reps, len(FORMAL_BENCHMARK_SEEDS)))
        engine_draw = rng.integers(0, len(truth), size=(reps, len(truth)))
        counts = engine_count_matrix(engine_draw, len(truth))
        rep_index = np.arange(reps)[:, None]

        for metric in PRIMARY_METRICS:
            proposed_by_seed = sampled_seed_metrics(truth, proposed, counts, metric)
            predecessor_by_seed = sampled_seed_metrics(truth, predecessor, counts, metric)
            difference = (
                proposed_by_seed[seed_draw, rep_index].mean(axis=1)
                - predecessor_by_seed[seed_draw, rep_index].mean(axis=1)
            )
            observed = float(
                np.mean(
                    [
                        sampled_seed_metrics(
                            truth,
                            proposed[index : index + 1],
                            np.ones((len(truth), 1), dtype=int),
                            metric,
                        )[0, 0]
                        - sampled_seed_metrics(
                            truth,
                            predecessor[index : index + 1],
                            np.ones((len(truth), 1), dtype=int),
                            metric,
                        )[0, 0]
                        for index in range(len(FORMAL_BENCHMARK_SEEDS))
                    ]
                )
            )
            low, high = np.quantile(difference, [0.025, 0.975])
            rows.append(
                {
                    "evidence_level": "MAIN FIXED-BENCHMARK ESTIMATION",
                    "subset": subset,
                    "comparison": "OCM-Asym minus RAST-GRU",
                    "metric": metric,
                    "mean_difference": observed,
                    "crossed_bootstrap_ci95_low": float(low),
                    "crossed_bootstrap_ci95_high": float(high),
                    "bootstrap_sign_proportion_ocm_lower": float(np.mean(difference < 0.0)),
                    "centered_bootstrap_p": centered_bootstrap_p(difference, observed),
                    "composite_seed_count": len(FORMAL_BENCHMARK_SEEDS),
                    "matched_test_engine_count": len(truth),
                    "bootstrap_repetitions": reps,
                    "resampling_design": "crossed composite-seed x matched-test-engine bootstrap",
                }
            )

        for rul_threshold in RUL_THRESHOLDS:
            for tolerance in TOLERANCES:
                proposed_by_seed = sampled_seed_metrics(
                    truth,
                    proposed,
                    counts,
                    "lpr30",
                    rul_threshold=rul_threshold,
                    tolerance=tolerance,
                )
                predecessor_by_seed = sampled_seed_metrics(
                    truth,
                    predecessor,
                    counts,
                    "lpr30",
                    rul_threshold=rul_threshold,
                    tolerance=tolerance,
                )
                difference = (
                    proposed_by_seed[seed_draw, rep_index].mean(axis=1)
                    - predecessor_by_seed[seed_draw, rep_index].mean(axis=1)
                )
                observed = float(
                    np.mean(
                        (
                            (
                                proposed - truth[None, :] > tolerance
                            ).astype(float)
                            - (
                                predecessor - truth[None, :] > tolerance
                            ).astype(float)
                        )[:, truth <= rul_threshold]
                    )
                )
                low, high = np.quantile(difference, [0.025, 0.975])
                tolerance_rows.append(
                    {
                        "evidence_level": "RETROSPECTIVE SENSITIVITY",
                        "subset": subset,
                        "rul_threshold_tau": rul_threshold,
                        "late_tolerance_epsilon": tolerance,
                        "metric": f"LPR@{rul_threshold:g},epsilon={tolerance:g}",
                        "mean_difference_ocm_minus_rast": observed,
                        "crossed_bootstrap_ci95_low": float(low),
                        "crossed_bootstrap_ci95_high": float(high),
                        "bootstrap_sign_proportion_ocm_lower": float(np.mean(difference < 0.0)),
                        "composite_seed_count": len(FORMAL_BENCHMARK_SEEDS),
                        "matched_test_engine_count": int(np.sum(truth <= rul_threshold)),
                        "bootstrap_repetitions": reps,
                    }
                )

    primary = pd.DataFrame(rows)
    primary["holm_adjusted_p"] = np.nan
    for metric, index in primary.groupby("metric").groups.items():
        primary.loc[index, "holm_adjusted_p"] = holm_adjust(
            primary.loc[index, "centered_bootstrap_p"].to_numpy(float)
        )
    primary["adjusted_evidence_status"] = np.where(
        primary["holm_adjusted_p"] < 0.05,
        np.where(primary["mean_difference"] < 0.0, "Holm: lower OCM-Asym", "Holm: lower RAST-GRU"),
        "unresolved after Holm",
    )
    return primary, pd.DataFrame(tolerance_rows)


def _stress_tensor(
    frame: pd.DataFrame,
    *,
    point: str,
    scenario: str,
    training_seeds: list[int],
    perturbation_seeds: list[int],
    units: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    scenario_levels = sorted(
        frame.loc[frame["scenario"] == scenario, "level"].astype(float).unique().tolist()
    )
    levels = np.asarray([0.0, *scenario_levels], dtype=float)
    arrays = []
    full_index = pd.MultiIndex.from_product(
        [training_seeds, perturbation_seeds, units],
        names=["training_seed", "perturbation_seed", "unit_id"],
    )
    for level in levels:
        source_scenario = "clean" if level == 0.0 else scenario
        part = frame[
            (frame["operating_point"] == point)
            & (frame["scenario"] == source_scenario)
            & np.isclose(frame["level"].astype(float), level)
        ]
        series = (
            part.set_index(["training_seed", "perturbation_seed", "unit_id"])["error"]
            .reindex(full_index)
        )
        if series.isna().any():
            raise ValueError(f"Incomplete stress grid for point={point}, scenario={scenario}, level={level}")
        arrays.append(series.to_numpy(float).reshape(len(training_seeds), len(perturbation_seeds), len(units)))
    return levels, np.stack(arrays, axis=2)


def _stress_curve_values(
    error: np.ndarray,
    truth: np.ndarray,
    levels: np.ndarray,
    counts: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    seed_count, perturb_count, level_count, engine_count = error.shape
    flat_count = seed_count * perturb_count
    denominator_all = counts.sum(axis=0)
    critical = (truth <= 30.0).astype(float)
    denominator_critical = np.maximum(critical @ counts, 1.0)
    sampled: dict[str, list[np.ndarray]] = {metric: [] for metric in STRESS_METRICS}
    observed: dict[str, list[np.ndarray]] = {metric: [] for metric in STRESS_METRICS}
    for level_index in range(level_count):
        level_error = error[:, :, level_index, :].reshape(flat_count, engine_count)
        sampled["rmse"].append(np.sqrt((level_error**2) @ counts / denominator_all[None, :]))
        observed["rmse"].append(np.sqrt(np.mean(level_error**2, axis=1)))
        late = ((level_error > 0.0) & (critical[None, :] > 0)).astype(float)
        sampled["critical_30_late_prediction_ratio"].append(
            late @ counts / denominator_critical[None, :]
        )
        observed["critical_30_late_prediction_ratio"].append(
            late.sum(axis=1) / max(float(critical.sum()), 1.0)
        )
        positive = np.maximum(level_error, 0.0) * critical[None, :]
        sampled["critical_30_mean_late_excess"].append(
            positive @ counts / denominator_critical[None, :]
        )
        observed["critical_30_mean_late_excess"].append(
            positive.sum(axis=1) / max(float(critical.sum()), 1.0)
        )
    width = float(levels[-1] - levels[0])
    sampled_curve = {
        metric: TRAPEZOID(np.stack(values, axis=1), levels, axis=1).reshape(
            seed_count, perturb_count, -1
        )
        / width
        for metric, values in sampled.items()
    }
    observed_curve = {
        metric: TRAPEZOID(np.stack(values, axis=1), levels, axis=1).reshape(
            seed_count, perturb_count
        )
        / width
        for metric, values in observed.items()
    }
    return sampled_curve, observed_curve


def crossed_stress_bootstrap(
    prediction_path: Path,
    *,
    reps: int,
    random_seed: int,
) -> pd.DataFrame:
    frame = pd.read_csv(prediction_path)
    training_seeds = sorted(frame["training_seed"].astype(int).unique().tolist())
    perturbation_seeds = sorted(frame["perturbation_seed"].astype(int).unique().tolist())
    units = sorted(frame["unit_id"].astype(int).unique().tolist())
    truth_series = (
        frame[(frame["operating_point"] == "RAST") & (frame["scenario"] == "clean")]
        .drop_duplicates("unit_id")
        .set_index("unit_id")["true_rul"]
        .reindex(units)
    )
    truth = truth_series.to_numpy(float)
    rows: list[dict[str, object]] = []
    for scenario_index, scenario in enumerate(SCENARIOS):
        rng = np.random.default_rng(random_seed + scenario_index * 7919)
        seed_draw = rng.integers(0, len(training_seeds), size=(reps, len(training_seeds)))
        perturb_draw = rng.integers(0, len(perturbation_seeds), size=(reps, len(perturbation_seeds)))
        engine_draw = rng.integers(0, len(units), size=(reps, len(units)))
        counts = engine_count_matrix(engine_draw, len(units))
        rep_index = np.arange(reps)[:, None, None]
        point_samples: dict[str, dict[str, np.ndarray]] = {}
        point_observed: dict[str, dict[str, np.ndarray]] = {}
        reference_levels: np.ndarray | None = None
        for point in ("RAST", "Core", "Asym"):
            levels, error = _stress_tensor(
                frame,
                point=point,
                scenario=scenario,
                training_seeds=training_seeds,
                perturbation_seeds=perturbation_seeds,
                units=units,
            )
            if reference_levels is None:
                reference_levels = levels
            elif not np.allclose(reference_levels, levels):
                raise ValueError(f"Stress levels differ across operating points for {scenario}")
            sampled, observed = _stress_curve_values(error, truth, levels, counts)
            point_samples[point] = sampled
            point_observed[point] = observed

        for point in ("Core", "Asym"):
            for metric in STRESS_METRICS:
                point_selected = point_samples[point][metric][
                    seed_draw[:, :, None],
                    perturb_draw[:, None, :],
                    rep_index,
                ].mean(axis=(1, 2))
                rast_selected = point_samples["RAST"][metric][
                    seed_draw[:, :, None],
                    perturb_draw[:, None, :],
                    rep_index,
                ].mean(axis=(1, 2))
                difference = point_selected - rast_selected
                observed = float(
                    point_observed[point][metric].mean()
                    - point_observed["RAST"][metric].mean()
                )
                low, high = np.quantile(difference, [0.025, 0.975])
                rows.append(
                    {
                        "evidence_level": "RETROSPECTIVE SENSITIVITY",
                        "operating_point": point,
                        "comparator": "RAST",
                        "scenario": scenario,
                        "metric": metric,
                        "mean_curve_difference_point_minus_rast": observed,
                        "crossed_bootstrap_ci95_low": float(low),
                        "crossed_bootstrap_ci95_high": float(high),
                        "bootstrap_sign_proportion_point_lower": float(np.mean(difference < 0.0)),
                        "centered_bootstrap_p": centered_bootstrap_p(difference, observed),
                        "training_seed_count": len(training_seeds),
                        "perturbation_seed_count": len(perturbation_seeds),
                        "matched_test_engine_count": len(units),
                        "bootstrap_repetitions": reps,
                        "resampling_design": "crossed training-seed x perturbation-seed x matched-engine bootstrap",
                        "multiplicity_family": "2 operating points x 9 algorithmic perturbation families x 3 metrics",
                    }
                )
    result = pd.DataFrame(rows)
    result["holm_adjusted_p_54"] = holm_adjust(result["centered_bootstrap_p"].to_numpy(float))
    result["adjusted_evidence_status"] = np.where(
        result["holm_adjusted_p_54"] < 0.05,
        np.where(
            result["mean_curve_difference_point_minus_rast"] < 0.0,
            "Holm: lower operating point",
            "Holm: lower RAST",
        ),
        "unresolved after Holm",
    )
    return result


def ranking_sensitivity(results_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in FORMAL_BENCHMARK_SEEDS:
        for subset in FORMAL_BENCHMARK_SUBSETS:
            root = results_dir / f"paper_main_v3_seed{seed}" / subset
            for model_dir in root.iterdir():
                metrics_path = model_dir / "metrics.json"
                if not metrics_path.exists():
                    continue
                metrics = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
                rmse = float(metrics["test_rmse"])
                nasa = float(metrics["test_nasa_score"]) / float(metrics["n_test_units"])
                flagged = model_dir.name == "tcn_gru" and rmse > 50.0
                rows.append(
                    {
                        "seed": seed,
                        "subset": subset,
                        "model": model_dir.name,
                        "rmse": rmse,
                        "nasa_per_engine": nasa,
                        "lpr30": float(metrics["test_critical_30_late_prediction_ratio"]),
                        "flagged_training_failure": flagged,
                    }
                )
    frame = pd.DataFrame(rows)
    output: list[dict[str, object]] = []
    for rule, selected in (
        ("all completed cells", frame),
        ("exclude frozen RMSE>50 training-failure flags", frame[~frame["flagged_training_failure"]]),
    ):
        for metric in ("rmse", "nasa_per_engine", "lpr30"):
            ranked = selected.copy()
            ranked["rank"] = ranked.groupby(["subset", "seed"])[metric].rank(method="average")
            summary = ranked.groupby("model", as_index=False).agg(
                mean_rank=("rank", "mean"),
                median_rank=("rank", "median"),
                cells=("rank", "size"),
            )
            for record in summary.to_dict("records"):
                output.append({"sensitivity_rule": rule, "metric": metric, **record})
    return pd.DataFrame(output)


def plot_primary_forest(frame: pd.DataFrame, output: Path) -> None:
    labels = {
        "rmse": "RMSE difference (cycles)",
        "nasa_per_engine": "NASA/engine difference",
        "lpr30": "LPR@30 difference",
    }
    fig, axes = plt.subplots(1, 3, figsize=(9.2, 3.5), sharey=True)
    for ax, metric in zip(axes, PRIMARY_METRICS):
        part = frame[frame["metric"] == metric].copy()
        part["subset"] = pd.Categorical(
            part["subset"], categories=list(reversed(FORMAL_BENCHMARK_SUBSETS)), ordered=True
        )
        part = part.sort_values("subset")
        y = np.arange(len(part))
        effect = part["mean_difference"].to_numpy(float)
        low = part["crossed_bootstrap_ci95_low"].to_numpy(float)
        high = part["crossed_bootstrap_ci95_high"].to_numpy(float)
        significant = part["holm_adjusted_p"].to_numpy(float) < 0.05
        colors = np.where(significant, "#2a9d8f", "#7d8590")
        ax.axvline(0.0, color="#222222", linewidth=1.0, linestyle="--")
        for index in range(len(part)):
            ax.errorbar(
                effect[index],
                y[index],
                xerr=[[effect[index] - low[index]], [high[index] - effect[index]]],
                fmt="o",
                color=colors[index],
                capsize=3,
                linewidth=1.5,
            )
        ax.set_xlabel(labels[metric])
        ax.grid(axis="x", linestyle=":", alpha=0.35)
        ax.set_title(metric.replace("_", " ").upper(), loc="left", fontsize=10.5)
        if ax is axes[0]:
            ax.set_yticks(y, part["subset"].astype(str))
        else:
            ax.tick_params(axis="y", labelleft=False)
    fig.suptitle("Fixed-task OCM-Asym minus RAST-GRU estimates", fontsize=11.5, y=1.02)
    fig.text(
        0.5,
        -0.01,
        "Crossed composite-seed x matched-engine percentile intervals; all markers are gray because no contrast survives within-metric Holm correction.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_figure_manifest(output_dir: Path) -> None:
    source = output_dir / "paired_fixed_task_bootstrap.csv"
    figures = []
    for suffix in (".pdf", ".png"):
        figure = output_dir / "figures" / f"fig_paired_fixed_task_forest{suffix}"
        figures.append(
            {
                "figure": figure.name,
                "path": figure.relative_to(PROJECT_ROOT).as_posix(),
                "figure_sha256": sha256_file(figure),
                "source_files": [
                    {
                        "name": source.name,
                        "path": source.relative_to(PROJECT_ROOT).as_posix(),
                        "sha256": sha256_file(source),
                        "rows": int(len(pd.read_csv(source))),
                    }
                ],
            }
        )
    payload = {
        "protocol_version": "normalized-perturbation-evidence-v1",
        "scope": "Primary crossed-factor forest plot only.",
        "figures": figures,
    }
    (output_dir / "figure_data_manifest.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )


def write_evidence_latex(
    primary: pd.DataFrame,
    stress: pd.DataFrame,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    primary_rows = []
    for row in primary.to_dict("records"):
        primary_rows.append(
            f"{row['subset']} & {row['metric'].replace('_', ' ')} & "
            f"{row['mean_difference']:.3f} & "
            f"[{row['crossed_bootstrap_ci95_low']:.3f}, {row['crossed_bootstrap_ci95_high']:.3f}] & "
            f"{row['holm_adjusted_p']:.3f} & {row['adjusted_evidence_status']} \\\\"
        )
    primary_text = "\n".join(
        [
            r"\begin{table*}[!t]",
            r"\centering",
            r"\small",
            r"\caption{Fixed-task OCM-Asym minus RAST-GRU estimates under crossed composite-seed $\times$ matched-engine diagnostics. Negative effects favor OCM-Asym. Holm adjustment is performed across the declared 12-contrast family.}",
            r"\label{tab:paired-fixed-task}",
            r"\begin{tabular}{llrrrl}",
            r"\toprule",
            r"Task & Metric & Mean $\Delta$ & Crossed diagnostic interval & Holm $p$ & Status \\",
            r"\midrule",
            *primary_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    (output_dir / "table_paired_fixed_task.tex").write_text(primary_text, encoding="utf-8")
    primary_zh_rows = []
    metric_zh = {
        "rmse": "RMSE",
        "nasa_per_engine": "NASA/engine",
        "lpr30": "LPR@30",
    }
    status_zh = {
        "unresolved after Holm": "Holm 校正后未解决",
        "Holm: lower OCM-Asym": "Holm：OCM-Asym 更低",
        "Holm: lower RAST-GRU": "Holm：RAST-GRU 更低",
    }
    for row in primary.to_dict("records"):
        primary_zh_rows.append(
            f"{row['subset']} & {metric_zh[row['metric']]} & "
            f"{row['mean_difference']:.3f} & "
            f"[{row['crossed_bootstrap_ci95_low']:.3f}, {row['crossed_bootstrap_ci95_high']:.3f}] & "
            f"{row['holm_adjusted_p']:.3f} & {status_zh[row['adjusted_evidence_status']]} \\\\"
        )
    primary_zh_text = "\n".join(
        [
            r"\begin{table*}[!t]",
            r"\centering",
            r"\small",
            r"\caption{[主要冻结证据] 交叉“复合种子 $\times$ 匹配发动机”推断下的 OCM-Asym 减 RAST-GRU。负值支持 OCM-Asym；Holm 校正在每个覆盖四任务的指标家族内执行。}",
            r"\label{tab:paired-fixed-task-zh}",
            r"\begin{tabular}{llrrrl}",
            r"\toprule",
            r"任务 & 指标 & 均值 $\Delta$ & 95\% 交叉区间 & Holm $p$ & 状态 \\",
            r"\midrule",
            *primary_zh_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    (output_dir / "table_paired_fixed_task_zh.tex").write_text(
        primary_zh_text, encoding="utf-8"
    )

    counts = (
        stress.groupby(["operating_point", "metric", "adjusted_evidence_status"], as_index=False)
        .size()
        .pivot_table(
            index=["operating_point", "metric"],
            columns="adjusted_evidence_status",
            values="size",
            fill_value=0,
        )
        .reset_index()
    )
    count_rows = []
    for row in counts.to_dict("records"):
        count_rows.append(
            f"{row['operating_point']} & {row['metric'].replace('_', ' ')} & "
            f"{int(row.get('Holm: lower operating point', 0))} & "
            f"{int(row.get('Holm: lower RAST', 0))} & "
            f"{int(row.get('unresolved after Holm', 0))} \\\\"
        )
    stress_text = "\n".join(
        [
            r"\begin{table}[!t]",
            r"\centering",
            r"\small",
            r"\caption{[POST-FREEZE SENSITIVITY] Normalized-perturbation-family classifications after one joint Holm correction across $2\times9\times3=54$ contrasts. Counts are adjusted decisions, not unadjusted interval directions.}",
            r"\label{tab:normalized-perturbation-holm}",
            r"\begin{tabular}{llrrr}",
            r"\toprule",
            r"Point & Metric & Lower point & Lower RAST & Unresolved \\",
            r"\midrule",
            *count_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    (output_dir / "table_normalized_perturbation_holm.tex").write_text(stress_text, encoding="utf-8")
    stress_zh_rows = []
    stress_metric_zh = {
        "rmse": "RMSE",
        "critical_30_late_prediction_ratio": "LPR@30",
        "critical_30_mean_late_excess": "MLE@30",
    }
    for row in counts.to_dict("records"):
        stress_zh_rows.append(
            f"{row['operating_point']} & {stress_metric_zh[row['metric']]} & "
            f"{int(row.get('Holm: lower operating point', 0))} & "
            f"{int(row.get('Holm: lower RAST', 0))} & "
            f"{int(row.get('unresolved after Holm', 0))} \\\\"
        )
    stress_zh_text = "\n".join(
        [
            r"\begin{table}[!t]",
            r"\centering",
            r"\small",
            r"\caption{[冻结后敏感性] 对 $2\times9\times3=54$ 个对照联合实施 Holm 校正后的归一化空间扰动族分类。计数表示校正后决策，而不是未校正区间方向。}",
            r"\label{tab:normalized-perturbation-holm-zh}",
            r"\begin{tabular}{llrrr}",
            r"\toprule",
            r"运行点 & 指标 & 运行点更低 & RAST 更低 & 未解决 \\",
            r"\midrule",
            *stress_zh_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    (output_dir / "table_normalized_perturbation_holm_zh.tex").write_text(
        stress_zh_text, encoding="utf-8"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build paired fixed-task and normalized-coordinate perturbation evidence.")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument(
        "--advanced-dir",
        default=str(PROJECT_ROOT / "paper_outputs" / "advanced_evidence"),
    )
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--stress-bootstrap-reps", type=int, default=3000)
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    output = Path(args.advanced_dir)
    output.mkdir(parents=True, exist_ok=True)

    primary, tolerance = crossed_fixed_task_bootstrap(
        results_dir,
        reps=args.bootstrap_reps,
        random_seed=20260726,
    )
    primary.to_csv(output / "paired_fixed_task_bootstrap.csv", index=False)
    tolerance.to_csv(output / "lpr_tolerance_sensitivity.csv", index=False)

    stress = crossed_stress_bootstrap(
        output / "stress_engine_level_predictions.csv",
        reps=args.stress_bootstrap_reps,
        random_seed=20260727,
    )
    stress.to_csv(output / "normalized_perturbation_contrasts.csv", index=False)

    ranking = ranking_sensitivity(results_dir)
    ranking.to_csv(output / "training_failure_ranking_sensitivity.csv", index=False)

    plot_primary_forest(primary, output / "figures" / "fig_paired_fixed_task_forest")
    write_figure_manifest(output)
    write_evidence_latex(primary, stress, output)
    print(
        "NORMALIZED_PERTURBATION_EVIDENCE_READY "
        f"primary={len(primary)} tolerance={len(tolerance)} stress={len(stress)} ranking={len(ranking)}"
    )


if __name__ == "__main__":
    main()
