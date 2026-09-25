from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap


PROJECT_ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_stress_pair_extension import METRICS, SCENARIOS, SCENARIO_DISPLAY, add_bias_diagnostics, normalized_curve_mean
from rul.experiments import evaluate_robustness
from run_formal_benchmark import FORMAL_BENCHMARK_SEEDS
from run_multiseed_stress_tests import PERTURBATION_SEEDS


def core_run(seed: int) -> Path:
    return (
        PROJECT_ROOT
        / "results"
        / f"ablation_fd004_seed{seed}_weighted_huber_no_asymmetry"
        / "FD004"
        / "rast_gru_v2"
    )


def curve_rows(raw: pd.DataFrame, operating_point: str) -> list[dict]:
    rows = []
    for (training_seed, perturbation_seed, scenario), group in raw.groupby(
        ["training_seed", "perturbation_seed", "scenario"]
    ):
        if scenario == "clean":
            continue
        clean = raw[
            (raw.training_seed == training_seed)
            & (raw.perturbation_seed == perturbation_seed)
            & (raw.scenario == "clean")
        ]
        combined = pd.concat([clean, group], ignore_index=True)
        row = {
            "operating_point": operating_point,
            "training_seed": int(training_seed),
            "perturbation_seed": int(perturbation_seed),
            "scenario": scenario,
        }
        for metric in METRICS:
            row[f"{metric}_curve_mean"] = normalized_curve_mean(combined, metric)
        rows.append(row)
    return rows


def paired_uncertainty(curves: pd.DataFrame, reps: int) -> pd.DataFrame:
    rows = []
    training_seeds = list(FORMAL_BENCHMARK_SEEDS)
    for point_index, point in enumerate(("Core", "Asym")):
        for scenario_index, scenario in enumerate(SCENARIOS):
            part = curves[curves.scenario == scenario]
            for metric_index, metric in enumerate(METRICS):
                column = f"{metric}_curve_mean"
                pivot = part.pivot(
                    index=["training_seed", "perturbation_seed"],
                    columns="operating_point",
                    values=column,
                )
                difference = (pivot[point] - pivot["RAST"]).unstack("perturbation_seed").loc[training_seeds]
                matrix = difference.to_numpy(float)
                # Preserve the exact formal Asym-vs-RAST bootstrap stream used by
                # build_stress_pair_extension.py; Core receives a separate,
                # structurally identical stream.
                rng_seed = (
                    20260716 + scenario_index * 101 + metric_index
                    if point == "Asym"
                    else 20260717 + scenario_index * 101 + metric_index
                )
                rng = np.random.default_rng(rng_seed)
                training_draw = rng.integers(0, matrix.shape[0], size=(reps, matrix.shape[0]))
                perturbation_draw = rng.integers(0, matrix.shape[1], size=(reps, matrix.shape[0], matrix.shape[1]))
                samples = matrix[training_draw[:, :, None], perturbation_draw].mean(axis=(1, 2))
                low, high = np.quantile(samples, [0.025, 0.975])
                classification = "unresolved"
                if high < 0:
                    classification = f"CI supports lower {point}"
                elif low > 0:
                    classification = "CI supports lower RAST"
                rows.append(
                    {
                        "operating_point": point,
                        "comparator": "RAST",
                        "scenario": scenario,
                        "metric": metric,
                        "mean_difference_point_minus_rast": float(matrix.mean()),
                        "nested_bootstrap_ci95_low": float(low),
                        "nested_bootstrap_ci95_high": float(high),
                        "probability_point_lower": float(np.mean(samples < 0.0)),
                        "ci_classification": classification,
                        "training_seed_count": matrix.shape[0],
                        "perturbation_seed_count": matrix.shape[1],
                        "bootstrap_repetitions": reps,
                    }
                )
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Add OCM-Core to the matched nine-family stress evidence.")
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "paper_outputs" / "advanced_evidence"))
    parser.add_argument("--bootstrap-reps", type=int, default=5000)
    parser.add_argument("--reuse", action="store_true")
    args = parser.parse_args()
    output = Path(args.output_dir)
    figure_dir = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    core_raw_path = output / "stress_core_seed_level.csv"

    if args.reuse:
        core_raw = pd.read_csv(core_raw_path)
    else:
        rows = []
        total = len(FORMAL_BENCHMARK_SEEDS) * len(PERTURBATION_SEEDS)
        index = 0
        for training_seed in FORMAL_BENCHMARK_SEEDS:
            for perturbation_seed in PERTURBATION_SEEDS:
                index += 1
                print(f"[CORE STRESS {index}/{total}] train={training_seed} perturb={perturbation_seed}", flush=True)
                evaluated = evaluate_robustness(core_run(training_seed), perturbation_seed=perturbation_seed, save=False)
                for row in evaluated:
                    row["training_seed"] = int(training_seed)
                    row["perturbation_seed"] = int(perturbation_seed)
                rows.extend(evaluated)
        core_raw = add_bias_diagnostics(pd.DataFrame(rows))
        core_raw.to_csv(core_raw_path, index=False)

    existing = add_bias_diagnostics(pd.read_csv(output / "stress_pair_extended_seed_level.csv"))
    curves = []
    curves.extend(curve_rows(core_raw, "Core"))
    curves.extend(curve_rows(existing[existing.model == "rast_gru_v2"], "Asym"))
    curves.extend(curve_rows(existing[existing.model == "rast_gru"], "RAST"))
    curve_frame = pd.DataFrame(curves)
    curve_frame.to_csv(output / "stress_operating_points_curve_means.csv", index=False)
    uncertainty = paired_uncertainty(curve_frame, args.bootstrap_reps)
    uncertainty.to_csv(output / "stress_operating_points_uncertainty.csv", index=False)

    primary = uncertainty[uncertainty.metric.isin(["rmse", "critical_30_late_prediction_ratio"])].copy()
    primary["row"] = primary["operating_point"] + "\n" + primary["metric"].map(
        {"rmse": "RMSE", "critical_30_late_prediction_ratio": "LPR@30"}
    )
    values = primary.pivot(index="row", columns="scenario", values="mean_difference_point_minus_rast")
    values = values[[scenario for scenario in SCENARIOS]]
    values.columns = [SCENARIO_DISPLAY[scenario] for scenario in values.columns]
    colors = np.sign(values)
    fig, ax = plt.subplots(figsize=(11.4, 6.2))
    sns.heatmap(
        colors,
        annot=values,
        fmt="+.3f",
        center=0,
        vmin=-1,
        vmax=1,
        cmap=ListedColormap(["#4c78a8", "#f2f2f2", "#c44e52"]),
        ax=ax,
        linewidths=.5,
        annot_kws={"fontsize": 10},
        cbar_kws={"label": "Direction only", "ticks": [-1, 0, 1], "shrink": 0.82},
    )
    ax.collections[0].colorbar.set_ticklabels(["Named OCM point lower", "equal", "RAST lower"])
    ax.set_xlabel("Controlled sensor-degradation family", fontsize=11)
    ax.set_ylabel("")
    ax.tick_params(axis="x", labelrotation=35, labelsize=9)
    ax.tick_params(axis="y", labelrotation=0, labelsize=10)
    ax.set_title("Core/Asym minus RAST normalized degradation-curve means", loc="left", fontsize=12)
    fig.tight_layout()
    fig.savefig(figure_dir / "fig_stress_operating_points.pdf", bbox_inches="tight")
    fig.savefig(figure_dir / "fig_stress_operating_points.png", dpi=400, bbox_inches="tight")
    plt.close(fig)
    print(f"CORE_STRESS_EXTENSION_READY raw={len(core_raw)} uncertainty={len(uncertainty)}")


if __name__ == "__main__":
    main()
