from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rul.augmentations import apply_block_missing_with_mask, apply_sensor_missing_with_mask
from rul.experiments import _load_run
from rul.preprocessing import WindowData
from rul.training import evaluate_window_data
from run_formal_benchmark import FORMAL_BENCHMARK_SEEDS


MODELS = ("rast_gru", "rast_gru_v2")
PERTURBATION_SEEDS = (7, 17, 27, 37, 47)
FALSE_NEGATIVE_RATES = (0.0, 0.10, 0.25, 0.50)
FALSE_POSITIVE_RATES = (0.0, 0.05, 0.10, 0.20)
DELAY_STEPS = (0, 2, 5, 10)
MODEL_DISPLAY = {"rast_gru": "RAST-GRU", "rast_gru_v2": "OCM-Asym"}
SCENARIO_DISPLAY = {
    "global_missing_false_negative": "False-negative mask rate",
    "global_missing_false_positive": "False-positive mask rate",
    "block_missing_indicator_delay": "Indicator delay (cycles)",
}


def corrupt_mask(
    mask: np.ndarray,
    *,
    false_negative_rate: float,
    false_positive_rate: float,
    seed: int,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    out = mask.copy()
    missing = out == 0.0
    observed = out == 1.0
    if false_negative_rate > 0.0:
        flip = missing & (rng.random(out.shape) < false_negative_rate)
        out[flip] = 1.0
    if false_positive_rate > 0.0:
        flip = observed & (rng.random(out.shape) < false_positive_rate)
        out[flip] = 0.0
    return out


def delay_mask(mask: np.ndarray, steps: int) -> np.ndarray:
    if steps <= 0:
        return mask.copy()
    out = mask.copy()
    for window in range(out.shape[0]):
        for sensor in range(out.shape[2]):
            missing = np.flatnonzero(out[window, :, sensor] == 0.0)
            if len(missing):
                onset = int(missing[0])
                out[window, onset : min(onset + steps, out.shape[1]), sensor] = 1.0
    return out


def plot_summary(summary: pd.DataFrame, output: Path) -> None:
    scenarios = list(SCENARIO_DISPLAY)
    fig, axes = plt.subplots(2, 3, figsize=(9.2, 5.7))
    styles = {
        "rast_gru": ("#365f8d", "o"),
        "rast_gru_v2": ("#b05a48", "s"),
    }
    for column, scenario in enumerate(scenarios):
        for model_name in MODELS:
            part = summary[
                (summary["scenario"] == scenario) & (summary["model"] == model_name)
            ].sort_values("level")
            color, marker = styles[model_name]
            axes[0, column].plot(
                part["level"],
                part["rmse_mean"],
                color=color,
                marker=marker,
                label=MODEL_DISPLAY[model_name],
            )
            axes[1, column].plot(
                part["level"],
                part["lpr30_mean"],
                color=color,
                marker=marker,
                label=MODEL_DISPLAY[model_name],
            )
        axes[0, column].set_title(SCENARIO_DISPLAY[scenario], fontsize=10)
        axes[1, column].set_xlabel("Declared mask-error level")
        for row in (0, 1):
            axes[row, column].grid(alpha=0.25, linestyle=":")
    axes[0, 0].set_ylabel("RMSE")
    axes[1, 0].set_ylabel("LPR@30")
    axes[0, 2].legend(frameon=False, fontsize=8)
    fig.suptitle("FD004 sensitivity to imperfect missingness indicators", fontsize=11.5)
    fig.text(
        0.5,
        0.01,
        "Means over five training seeds and five matched perturbation seeds; retrospective descriptive probes.",
        ha="center",
        fontsize=8.5,
    )
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def write_latex(summary: pd.DataFrame, output: Path) -> None:
    rows = []
    for scenario in SCENARIO_DISPLAY:
        for model_name in MODELS:
            part = summary[
                (summary["scenario"] == scenario) & (summary["model"] == model_name)
            ].sort_values("level")
            first = part.iloc[0]
            last = part.iloc[-1]
            rows.append(
                f"{SCENARIO_DISPLAY[scenario]} & {MODEL_DISPLAY[model_name]} & "
                f"{first['level']:.2f}$\\rightarrow${last['level']:.2f} & "
                f"{first['rmse_mean']:.3f}$\\rightarrow${last['rmse_mean']:.3f} & "
                f"{first['lpr30_mean']:.3f}$\\rightarrow${last['lpr30_mean']:.3f} \\\\"
            )
    text = "\n".join(
        [
            r"\begin{table}[!t]",
            r"\centering",
            r"\scriptsize",
            r"\caption{[RETROSPECTIVE MASK-OBSERVATION SENSITIVITY] Endpoint changes under false-negative masks, false-positive masks, and delayed block-missing indicators at a fixed injected missing rate of 0.20. Values average five training seeds and five matched perturbation seeds. Nonmonotone directions are descriptive and do not establish calibrated fault detection.}",
            r"\label{tab:mask-observation-sensitivity}",
            r"\begin{tabular}{llrrr}",
            r"\toprule",
            r"Probe & Point & Level range & RMSE & LPR@30 \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    output.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate imperfect or delayed missing-indicator sensitivity at fixed FD004 checkpoints."
    )
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "paper_outputs" / "analysis_working"))
    parser.add_argument("--missing-rate", type=float, default=0.20)
    args = parser.parse_args()
    results_dir = Path(args.results_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []

    for training_seed in FORMAL_BENCHMARK_SEEDS:
        loaded = {}
        for model_name in MODELS:
            run_dir = results_dir / f"paper_main_v3_seed{training_seed}" / "FD004" / model_name
            config, metrics, prepared, model, device = _load_run(run_dir)
            loaded[model_name] = (config, metrics, prepared, model, device)
        reference = loaded["rast_gru_v2"][2]
        for perturbation_seed in PERTURBATION_SEEDS:
            global_x, global_mask = apply_sensor_missing_with_mask(
                reference.test.x,
                args.missing_rate,
                reference.sensor_indices,
                seed=perturbation_seed,
            )
            block_x, block_mask = apply_block_missing_with_mask(
                reference.test.x,
                args.missing_rate,
                reference.sensor_indices,
                seed=perturbation_seed,
            )
            scenarios = []
            for rate in FALSE_NEGATIVE_RATES:
                scenarios.append(
                    (
                        "global_missing_false_negative",
                        rate,
                        global_x,
                        corrupt_mask(
                            global_mask,
                            false_negative_rate=rate,
                            false_positive_rate=0.0,
                            seed=perturbation_seed + 1000,
                        ),
                    )
                )
            for rate in FALSE_POSITIVE_RATES:
                scenarios.append(
                    (
                        "global_missing_false_positive",
                        rate,
                        global_x,
                        corrupt_mask(
                            global_mask,
                            false_negative_rate=0.0,
                            false_positive_rate=rate,
                            seed=perturbation_seed + 2000,
                        ),
                    )
                )
            for steps in DELAY_STEPS:
                scenarios.append(
                    (
                        "block_missing_indicator_delay",
                        float(steps),
                        block_x,
                        delay_mask(block_mask, steps),
                    )
                )

            for scenario, level, values, observed_mask in scenarios:
                data = WindowData(
                    x=values,
                    y=reference.test.y,
                    unit_ids=reference.test.unit_ids,
                    end_cycles=reference.test.end_cycles,
                    observed_mask=observed_mask,
                )
                for model_name in MODELS:
                    config, _, prepared, model, device = loaded[model_name]
                    metrics, _ = evaluate_window_data(
                        model,
                        data,
                        device,
                        batch_size=int(config["training"].get("batch_size", 128)),
                        sensor_indices=prepared.sensor_indices,
                        append_missing_mask=bool(config["data"].get("append_missing_mask", False)),
                    )
                    rows.append(
                        {
                            "training_seed": training_seed,
                            "perturbation_seed": perturbation_seed,
                            "model": model_name,
                            "scenario": scenario,
                            "level": level,
                            "injected_missing_rate": args.missing_rate,
                            "test_rmse": metrics["rmse"],
                            "test_nasa_per_engine": metrics["nasa_score"] / len(data.y),
                            "test_lpr30": metrics["critical_30_late_prediction_ratio"],
                            "evidence_level": "RETROSPECTIVE MASK-OBSERVATION SENSITIVITY",
                        }
                    )
    detail = pd.DataFrame(rows)
    detail.to_csv(output / "mask_observation_sensitivity_seed_level.csv", index=False)
    summary = (
        detail.groupby(["model", "scenario", "level"], as_index=False)
        .agg(
            rmse_mean=("test_rmse", "mean"),
            rmse_sd=("test_rmse", "std"),
            nasa_mean=("test_nasa_per_engine", "mean"),
            nasa_sd=("test_nasa_per_engine", "std"),
            lpr30_mean=("test_lpr30", "mean"),
            lpr30_sd=("test_lpr30", "std"),
            cells=("test_rmse", "size"),
        )
    )
    summary.to_csv(output / "mask_observation_sensitivity_summary.csv", index=False)
    plot_summary(summary, output / "figures" / "fig_mask_observation_sensitivity")
    write_latex(summary, output / "table_mask_observation_sensitivity.tex")
    print(
        f"MASK_OBSERVATION_SENSITIVITY_READY rows={len(detail)} summary={len(summary)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
