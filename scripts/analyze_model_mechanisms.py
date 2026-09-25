from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mpl_cache"))
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

from rul.augmentations import apply_sensor_missing_with_mask
from rul.experiments import _load_run
from rul.preprocessing import RULWindowDataset, WindowData
from run_formal_benchmark import FORMAL_BENCHMARK_SEEDS

plt.rcParams.update({"font.size": 10.5, "axes.titlesize": 12, "axes.labelsize": 11, "legend.fontsize": 9.5})


PERTURBATION_SEEDS = [42, 123, 2024, 2025, 2026]


def _input_tensor(data: WindowData, sensor_indices: list[int], batch_size: int) -> torch.Tensor:
    loader = DataLoader(
        RULWindowDataset(data, sensor_indices=sensor_indices, append_missing_mask=True),
        batch_size=batch_size,
        shuffle=False,
    )
    return torch.cat([batch[0] for batch in loader], dim=0)


@torch.no_grad()
def _reliability_weights(model: torch.nn.Module, inputs: torch.Tensor, device: torch.device, batch_size: int) -> np.ndarray:
    rows = []
    for start in range(0, len(inputs), batch_size):
        batch = inputs[start : start + batch_size].to(device)
        rows.append(model.reliability_gate.weights(batch).cpu().numpy())
    return np.concatenate(rows, axis=0)


@torch.no_grad()
def _predictions(model: torch.nn.Module, inputs: torch.Tensor, device: torch.device, batch_size: int) -> np.ndarray:
    rows = []
    for start in range(0, len(inputs), batch_size):
        batch = inputs[start : start + batch_size].to(device)
        rows.append(model(batch).cpu().numpy())
    return np.maximum(np.concatenate(rows, axis=0), 0.0)


@torch.no_grad()
def _attention_weights(model: torch.nn.Module, inputs: torch.Tensor, device: torch.device, batch_size: int) -> np.ndarray:
    rows = []
    for start in range(0, len(inputs), batch_size):
        batch = inputs[start : start + batch_size].to(device)
        z = model.reliability_gate(batch)
        z = model.channel_gate(z)
        z = model.local_trend(z)
        sequence, _ = model.gru(z)
        rows.append(model.attention_pool.weights(sequence).squeeze(-1).cpu().numpy())
    return np.concatenate(rows, axis=0)


def analyze_seed(
    run_dir: Path,
    *,
    train_seed: int,
    perturbation_seeds: list[int],
    missing_rates: list[float],
    batch_size: int,
) -> tuple[
    list[dict[str, float | int]],
    list[dict[str, float | int | str]],
    list[dict[str, float | int]],
]:
    _, _, prepared, model, device = _load_run(run_dir)
    if not hasattr(model.reliability_gate, "weights"):
        raise TypeError("The loaded model does not expose reliability weights.")
    if not hasattr(model.attention_pool, "weights"):
        raise TypeError("The loaded model does not expose temporal-attention weights.")

    clean_data = WindowData(
        x=prepared.test.x,
        y=prepared.test.y,
        unit_ids=prepared.test.unit_ids,
        end_cycles=prepared.test.end_cycles,
    )
    clean_inputs = _input_tensor(clean_data, prepared.sensor_indices, batch_size)
    clean_reliability = _reliability_weights(model, clean_inputs, device, batch_size)
    clean_sensor_reliability = clean_reliability[:, prepared.sensor_indices]

    reliability_rows: list[dict[str, float | int]] = []
    calibration_rows: list[dict[str, float | int]] = []
    for perturbation_seed in perturbation_seeds:
        for missing_rate in missing_rates:
            corrupted_x, observed_mask = apply_sensor_missing_with_mask(
                prepared.test.x,
                missing_rate,
                prepared.sensor_indices,
                seed=perturbation_seed,
            )
            corrupted_data = WindowData(
                x=corrupted_x,
                y=prepared.test.y,
                unit_ids=prepared.test.unit_ids,
                end_cycles=prepared.test.end_cycles,
                observed_mask=observed_mask,
            )
            inputs = _input_tensor(corrupted_data, prepared.sensor_indices, batch_size)
            reliability = _reliability_weights(model, inputs, device, batch_size)[:, prepared.sensor_indices]
            # The returned mask is already ordered over selected sensor positions only.
            labels = (observed_mask.mean(axis=1) < 0.5).astype(int)
            scores = 1.0 - reliability
            label_flat = labels.reshape(-1)
            score_flat = scores.reshape(-1)
            predictions = _predictions(model, inputs, device, batch_size)
            absolute_error = np.abs(predictions - prepared.test.y)
            mean_reliability = reliability.mean(axis=1)
            correlation = spearmanr(mean_reliability, absolute_error).statistic
            corrupted_values = reliability[labels > 0]
            observed_values = reliability[labels == 0]
            bin_ids = np.clip(np.digitize(score_flat, np.linspace(0.0, 1.0, 11), right=False) - 1, 0, 9)
            ece = 0.0
            for bin_id in range(10):
                in_bin = bin_ids == bin_id
                if not np.any(in_bin):
                    continue
                bin_count = int(np.sum(in_bin))
                bin_prediction = float(np.mean(score_flat[in_bin]))
                bin_observed = float(np.mean(label_flat[in_bin]))
                ece += (bin_count / len(label_flat)) * abs(bin_prediction - bin_observed)
                calibration_rows.append(
                    {
                        "training_seed": train_seed,
                        "perturbation_seed": perturbation_seed,
                        "missing_rate": missing_rate,
                        "bin_id": bin_id,
                        "bin_lower": bin_id / 10.0,
                        "bin_upper": (bin_id + 1) / 10.0,
                        "count": bin_count,
                        "mean_predicted_unavailability": bin_prediction,
                        "observed_missing_fraction": bin_observed,
                    }
                )
            clipped = np.clip(score_flat, 1e-6, 1.0 - 1e-6)
            logit_score = np.log(clipped / (1.0 - clipped)).reshape(-1, 1)
            calibration_model = LogisticRegression(C=1e6, solver="lbfgs", max_iter=1000)
            calibration_model.fit(logit_score, label_flat)
            reliability_rows.append(
                {
                    "training_seed": train_seed,
                    "perturbation_seed": perturbation_seed,
                    "missing_rate": missing_rate,
                    "fault_auroc": float(roc_auc_score(label_flat, score_flat)),
                    "fault_auprc": float(average_precision_score(label_flat, score_flat)),
                    "brier_score": float(np.mean((score_flat - label_flat) ** 2)),
                    "expected_calibration_error_10bin": float(ece),
                    "calibration_intercept": float(calibration_model.intercept_[0]),
                    "calibration_slope": float(calibration_model.coef_[0, 0]),
                    "corrupted_reliability_mean": float(np.mean(corrupted_values)),
                    "observed_reliability_mean": float(np.mean(observed_values)),
                    "corrupted_reliability_drop_from_clean": float(
                        np.mean(clean_sensor_reliability[labels > 0] - corrupted_values)
                    ),
                    "observed_reliability_drop_from_clean": float(
                        np.mean(clean_sensor_reliability[labels == 0] - observed_values)
                    ),
                    "reliability_abs_error_spearman": float(correlation),
                    "engine_count": int(len(prepared.test.y)),
                    "sensor_score_count": int(len(label_flat)),
                }
            )

    attention = _attention_weights(model, clean_inputs, device, batch_size)
    time_positions = np.linspace(0.0, 1.0, attention.shape[1], dtype=float)
    center_of_mass = np.sum(attention * time_positions[None, :], axis=1)
    end_third_start = int(math_floor_two_thirds(attention.shape[1]))
    end_third_mass = attention[:, end_third_start:].sum(axis=1)
    critical = prepared.test.y <= 30.0
    attention_rows: list[dict[str, float | int | str]] = []
    for zone_name, mask in [("critical_rul_le_30", critical), ("noncritical_rul_gt_30", ~critical)]:
        attention_rows.append(
            {
                "training_seed": train_seed,
                "zone": zone_name,
                "engine_count": int(mask.sum()),
                "attention_center_of_mass_mean": float(np.mean(center_of_mass[mask])),
                "attention_end_third_mass_mean": float(np.mean(end_third_mass[mask])),
                "attention_end_third_mass_std": float(np.std(end_third_mass[mask], ddof=1)),
            }
        )
    return reliability_rows, attention_rows, calibration_rows


def math_floor_two_thirds(length: int) -> int:
    return max(0, int(np.floor(length * 2.0 / 3.0)))


def summarize(
    reliability: pd.DataFrame,
    attention: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    numeric = [
        "fault_auroc", "fault_auprc", "brier_score", "expected_calibration_error_10bin",
        "calibration_intercept", "calibration_slope", "corrupted_reliability_mean",
        "observed_reliability_mean", "corrupted_reliability_drop_from_clean",
        "observed_reliability_drop_from_clean", "reliability_abs_error_spearman",
    ]
    seed_level = reliability.groupby(["training_seed", "missing_rate"], as_index=False)[numeric].mean()
    reliability_summary = seed_level.groupby("missing_rate", as_index=False).agg(
        training_seed_count=("training_seed", "nunique"),
        fault_auroc_mean=("fault_auroc", "mean"),
        fault_auroc_std=("fault_auroc", lambda x: x.std(ddof=1)),
        fault_auprc_mean=("fault_auprc", "mean"),
        fault_auprc_std=("fault_auprc", lambda x: x.std(ddof=1)),
        brier_score_mean=("brier_score", "mean"),
        brier_score_std=("brier_score", lambda x: x.std(ddof=1)),
        ece_10bin_mean=("expected_calibration_error_10bin", "mean"),
        ece_10bin_std=("expected_calibration_error_10bin", lambda x: x.std(ddof=1)),
        calibration_intercept_mean=("calibration_intercept", "mean"),
        calibration_intercept_std=("calibration_intercept", lambda x: x.std(ddof=1)),
        calibration_slope_mean=("calibration_slope", "mean"),
        calibration_slope_std=("calibration_slope", lambda x: x.std(ddof=1)),
        corrupted_reliability_mean=("corrupted_reliability_mean", "mean"),
        observed_reliability_mean=("observed_reliability_mean", "mean"),
        corrupted_reliability_drop_mean=("corrupted_reliability_drop_from_clean", "mean"),
        observed_reliability_drop_mean=("observed_reliability_drop_from_clean", "mean"),
        reliability_abs_error_spearman_mean=("reliability_abs_error_spearman", "mean"),
    )
    attention_summary = attention.groupby("zone", as_index=False).agg(
        seed_count=("training_seed", "nunique"),
        engine_count_mean=("engine_count", "mean"),
        attention_center_of_mass_mean=("attention_center_of_mass_mean", "mean"),
        attention_center_of_mass_std=("attention_center_of_mass_mean", lambda x: x.std(ddof=1)),
        attention_end_third_mass_mean=("attention_end_third_mass_mean", "mean"),
        attention_end_third_mass_std=("attention_end_third_mass_mean", lambda x: x.std(ddof=1)),
    )
    return seed_level, reliability_summary, attention_summary


def plot_mechanisms(
    reliability_summary: pd.DataFrame,
    attention: pd.DataFrame,
    calibration: pd.DataFrame,
    output: Path,
) -> None:
    fig = plt.figure(figsize=(7.2, 5.8), layout="constrained")
    grid = fig.add_gridspec(2, 2, height_ratios=(1.0, 0.9), hspace=0.28, wspace=0.38)
    axes = (
        fig.add_subplot(grid[0, 0]),
        fig.add_subplot(grid[1, :]),
        fig.add_subplot(grid[0, 1]),
    )
    ax = axes[0]
    x = reliability_summary["missing_rate"].to_numpy(float)
    for mean_column, std_column, label, color in [
        ("fault_auroc_mean", "fault_auroc_std", "AUROC", "#1f77b4"),
        ("fault_auprc_mean", "fault_auprc_std", "AUPRC", "#d95f02"),
    ]:
        mean = reliability_summary[mean_column].to_numpy(float)
        std = reliability_summary[std_column].fillna(0.0).to_numpy(float)
        ax.plot(x, mean, marker="o", linewidth=2.2, label=label, color=color)
        ax.fill_between(x, mean - std, mean + std, alpha=0.16, color=color)
    ax.axhline(0.5, color="black", linestyle=":", linewidth=1.0)
    ax.set_xlabel("Missing-sensor fraction")
    ax.set_ylabel("Missingness-localization score")
    ax.set_ylim(0.0, 1.02)
    ax.set_title("(a) Synthetic missingness localization", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(linestyle=":", alpha=0.3)

    ax = axes[1]
    plot_data = attention.copy()
    order = ["critical_rul_le_30", "noncritical_rul_gt_30"]
    sns.boxplot(
        data=plot_data,
        x="zone",
        y="attention_end_third_mass_mean",
        order=order,
        color="#66c2a5",
        width=0.5,
        ax=ax,
    )
    sns.stripplot(
        data=plot_data,
        x="zone",
        y="attention_end_third_mass_mean",
        order=order,
        color="black",
        size=5,
        ax=ax,
    )
    ax.set_xticks([0, 1], ["True RUL <= 30", "True RUL > 30"])
    ax.set_xlabel("")
    ax.set_ylabel("Mean final-third attention mass")
    ax.set_title("(c) Temporal-attention concentration", loc="left", fontsize=10)
    ax.grid(axis="y", linestyle=":", alpha=0.3)

    ax = axes[2]
    weighted_rows = []
    for bin_id, group in calibration.groupby("bin_id", sort=True):
        weighted_rows.append(
            {
                "bin_id": bin_id,
                "mean_predicted_unavailability": np.average(
                    group["mean_predicted_unavailability"], weights=group["count"]
                ),
                "observed_missing_fraction": np.average(
                    group["observed_missing_fraction"], weights=group["count"]
                ),
                "count": group["count"].sum(),
            }
        )
    weighted = pd.DataFrame(weighted_rows)
    ax.plot([0, 1], [0, 1], color="#555555", linestyle="--", linewidth=1.2, label="Ideal")
    ax.plot(
        weighted["mean_predicted_unavailability"],
        weighted["observed_missing_fraction"],
        color="#7570b3",
        label="Observed fraction",
    )
    ax.scatter(
        weighted["mean_predicted_unavailability"],
        weighted["observed_missing_fraction"],
        s=np.clip(np.sqrt(weighted["count"].to_numpy(float)) * 0.8, 20.0, 100.0),
        color="#7570b3",
        edgecolor="white",
        linewidth=0.5,
        zorder=3,
    )
    ax.set_xlabel("Mean predicted\nsensor-unavailability probability")
    ax.set_ylabel("Observed missing-sensor fraction")
    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(0.0, 1.0)
    ax.set_title("(b) Observed-fraction diagnostic", loc="left", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5)
    ax.grid(linestyle=":", alpha=0.3)
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit reliability-gate and temporal-attention mechanisms on FD004.")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "paper_outputs" / "advanced_evidence"))
    parser.add_argument("--training-seeds", nargs="+", type=int, default=FORMAL_BENCHMARK_SEEDS)
    parser.add_argument("--perturbation-seeds", nargs="+", type=int, default=PERTURBATION_SEEDS)
    parser.add_argument("--missing-rates", nargs="+", type=float, default=[0.1, 0.2, 0.3, 0.4])
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--reuse", action="store_true", help="Reuse stored mechanism CSVs and rebuild only summaries/figures.")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    figure_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    reliability_path = output_dir / "reliability_fault_localization_seed_level.csv"
    attention_path = output_dir / "attention_mechanism_seed_level.csv"
    calibration_path = output_dir / "reliability_calibration_bins.csv"
    if args.reuse and all(path.exists() for path in (reliability_path, attention_path, calibration_path)):
        reliability = pd.read_csv(reliability_path)
        attention = pd.read_csv(attention_path)
        calibration = pd.read_csv(calibration_path)
    else:
        reliability_rows = []
        attention_rows = []
        calibration_rows = []
        for index, train_seed in enumerate(args.training_seeds, start=1):
            run_dir = Path(args.results_dir) / f"paper_main_v3_seed{train_seed}" / "FD004" / "rast_gru_v2"
            print(f"[MECHANISM {index}/{len(args.training_seeds)}] training_seed={train_seed}", flush=True)
            reliability_seed, attention_seed, calibration_seed = analyze_seed(
                run_dir,
                train_seed=train_seed,
                perturbation_seeds=args.perturbation_seeds,
                missing_rates=args.missing_rates,
                batch_size=args.batch_size,
            )
            reliability_rows.extend(reliability_seed)
            attention_rows.extend(attention_seed)
            calibration_rows.extend(calibration_seed)
        reliability = pd.DataFrame(reliability_rows)
        attention = pd.DataFrame(attention_rows)
        calibration = pd.DataFrame(calibration_rows)
    seed_level, reliability_summary, attention_summary = summarize(reliability, attention)
    reliability.to_csv(reliability_path, index=False)
    seed_level.to_csv(output_dir / "reliability_observed_fraction_seed_summary.csv", index=False)
    reliability_summary.to_csv(output_dir / "reliability_fault_localization_summary.csv", index=False)
    attention.to_csv(attention_path, index=False)
    attention_summary.to_csv(output_dir / "attention_mechanism_summary.csv", index=False)
    calibration.to_csv(calibration_path, index=False)
    plot_mechanisms(reliability_summary, attention, calibration, figure_dir / "fig_mechanism_diagnostics")
    print(f"MECHANISM_ANALYSIS_READY {output_dir}")


if __name__ == "__main__":
    main()
