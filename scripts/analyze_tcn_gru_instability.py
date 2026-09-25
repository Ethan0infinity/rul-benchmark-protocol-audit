from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 123, 2024, 2025, 2026)
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
MODELS = (
    "gru",
    "lstm",
    "tcn",
    "tcn_gru",
    "rast_gru",
    "cnn_lstm",
    "attention_gru",
    "bigru_attention",
    "transformer_lite",
    "dual_attention_tcn",
    "sensor_graph_gru",
    "quantile_gru",
    "rast_gru_v2",
)
DISPLAY_NAMES = {
    "gru": "GRU",
    "lstm": "LSTM",
    "tcn": "TCN",
    "tcn_gru": "TCN-GRU",
    "rast_gru": "RAST-GRU",
    "cnn_lstm": "CNN-LSTM",
    "attention_gru": "Attention-GRU",
    "bigru_attention": "BiGRU-Attention",
    "transformer_lite": "Transformer-lite",
    "dual_attention_tcn": "Dual-Attention-TCN",
    "sensor_graph_gru": "Sensor-Graph-GRU",
    "quantile_gru": "Quantile-GRU",
    "rast_gru_v2": "OCM-Asym",
}


def collect() -> pd.DataFrame:
    rows: list[dict] = []
    for subset in SUBSETS:
        for seed in SEEDS:
            run_dir = PROJECT_ROOT / "results" / f"paper_main_v3_seed{seed}" / subset / "tcn_gru"
            with (run_dir / "metrics.json").open("r", encoding="utf-8-sig") as handle:
                metrics = json.load(handle)
            history = pd.read_csv(run_dir / "epoch_history.csv")
            predictions = pd.read_csv(run_dir / "test_predictions.csv")
            error = predictions["pred_rul"].to_numpy(float) - predictions["true_rul"].to_numpy(float)
            finite = np.isfinite(predictions[["true_rul", "pred_rul"]].to_numpy(float)).all()
            min_val_loss_epoch = int(history.loc[history["val_loss"].idxmin(), "epoch"])
            min_val_last_rmse_epoch = int(history.loc[history["val_last_rmse"].idxmin(), "epoch"])
            rows.append(
                {
                    "subset": subset,
                    "seed": seed,
                    "test_rmse": float(metrics["test_rmse"]),
                    "test_nasa_per_engine": float(metrics["test_nasa_score"]) / int(metrics["n_test_units"]),
                    "test_lpr30": float(metrics["test_critical_30_late_prediction_ratio"]),
                    "best_checkpoint_epoch": int(metrics["best_epoch"]),
                    "completed_epochs": int(metrics["completed_epochs"]),
                    "training_complete_reason": metrics["training_complete_reason"],
                    "minimum_validation_loss_epoch": min_val_loss_epoch,
                    "minimum_validation_endpoint_rmse_epoch": min_val_last_rmse_epoch,
                    "checkpoint_selection_reversal": int(metrics["best_epoch"]) != min_val_loss_epoch,
                    "initial_train_loss": float(history.iloc[0]["train_loss"]),
                    "final_train_loss": float(history.iloc[-1]["train_loss"]),
                    "initial_validation_loss": float(history.iloc[0]["val_loss"]),
                    "minimum_validation_loss": float(history["val_loss"].min()),
                    "prediction_min": float(predictions["pred_rul"].min()),
                    "prediction_max": float(predictions["pred_rul"].max()),
                    "error_min": float(error.min()),
                    "error_max": float(error.max()),
                    "prediction_has_nan_or_inf": not bool(finite),
                    "gradient_clip_max_norm": 5.0,
                    "gradient_norm_logged": False,
                    "deterministic_training": bool(metrics.get("deterministic_training", False)),
                    "checkpoint_selection_metric": metrics.get("checkpoint_selection_metric", "unknown"),
                    "run_dir": str(run_dir),
                }
            )
    frame = pd.DataFrame(rows)
    frame["catastrophic_cell"] = False
    for subset, indices in frame.groupby("subset").groups.items():
        values = frame.loc[indices, "test_rmse"]
        median = float(values.median())
        frame.loc[indices, "catastrophic_cell"] = (values > 50.0) & (values > 2.0 * median)
    return frame


def summarize(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.groupby("subset", as_index=False).agg(
        seed_count=("seed", "nunique"),
        rmse_mean=("test_rmse", "mean"),
        rmse_sd=("test_rmse", lambda x: x.std(ddof=1)),
        rmse_median=("test_rmse", "median"),
        rmse_q25=("test_rmse", lambda x: x.quantile(0.25)),
        rmse_q75=("test_rmse", lambda x: x.quantile(0.75)),
        nasa_per_engine_median=("test_nasa_per_engine", "median"),
        catastrophic_cell_count=("catastrophic_cell", "sum"),
        checkpoint_selection_reversal_count=("checkpoint_selection_reversal", "sum"),
        nonfinite_cell_count=("prediction_has_nan_or_inf", "sum"),
    )


def trimmed_mean(values: pd.Series, proportion: float = 0.1) -> float:
    ordered = np.sort(values.to_numpy(float))
    trim = int(np.floor(len(ordered) * proportion))
    retained = ordered[trim : len(ordered) - trim] if trim else ordered
    return float(retained.mean())


def collect_run_quality() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for seed in SEEDS:
        for subset in SUBSETS:
            for model in MODELS:
                run_dir = (
                    PROJECT_ROOT
                    / "results"
                    / f"paper_main_v3_seed{seed}"
                    / subset
                    / model
                )
                metrics = json.loads(
                    (run_dir / "metrics.json").read_text(
                        encoding="utf-8-sig"
                    )
                )
                history = pd.read_csv(run_dir / "epoch_history.csv")
                predictions = pd.read_csv(
                    run_dir / "test_predictions.csv"
                )
                selected_epoch = int(metrics["best_epoch"])
                minimum_composite_epoch = int(
                    history.loc[
                        history["selection_score"].idxmin(),
                        "epoch",
                    ]
                )
                minimum_endpoint_epoch = int(
                    history.loc[
                        history["val_last_rmse"].idxmin(),
                        "epoch",
                    ]
                )
                completed_epochs = int(metrics["completed_epochs"])
                finite = bool(
                    np.isfinite(
                        predictions[["true_rul", "pred_rul"]].to_numpy(
                            float
                        )
                    ).all()
                )
                selected_row = history.loc[
                    history["epoch"] == selected_epoch
                ].iloc[0]
                minimum_composite_row = history.loc[
                    history["epoch"] == minimum_composite_epoch
                ].iloc[0]
                minimum_rmse_row = history.loc[
                    history["epoch"] == minimum_endpoint_epoch
                ].iloc[0]
                objective_mismatch = bool(
                    selected_epoch != minimum_composite_epoch
                    or not np.isclose(
                        float(metrics["best_selection_score"]),
                        float(history["selection_score"].min()),
                        rtol=1e-7,
                        atol=1e-9,
                    )
                )
                # Result-informed sensitivity marker, applied uniformly. It
                # describes early selection plus a later RMSE improvement; it
                # is not a checkpoint-failure or exclusion rule.
                early_selection_sensitivity = bool(
                    selected_epoch <= 2
                    and completed_epochs >= 10
                    and minimum_endpoint_epoch - selected_epoch >= 5
                )
                rows.append(
                    {
                        "model": model,
                        "model_display": DISPLAY_NAMES[model],
                        "subset": subset,
                        "seed": seed,
                        "selected_checkpoint_epoch": selected_epoch,
                        "minimum_composite_score_epoch": minimum_composite_epoch,
                        "minimum_validation_endpoint_rmse_epoch": (
                            minimum_endpoint_epoch
                        ),
                        "completed_epochs": completed_epochs,
                        "checkpoint_selection_metric": metrics.get(
                            "checkpoint_selection_metric",
                            "unknown",
                        ),
                        "prediction_finite": finite,
                        "objective_aligned_selection_mismatch": objective_mismatch,
                        "early_selection_sensitivity_flag": early_selection_sensitivity,
                        "selected_validation_rmse": float(
                            selected_row["val_last_rmse"]
                        ),
                        "selected_validation_lpr30": float(
                            selected_row["val_last_lpr30"]
                        ),
                        "selected_validation_slpr30_10": float(
                            selected_row["val_last_slpr30_10"]
                        ),
                        "selected_composite_score": float(
                            selected_row["selection_score"]
                        ),
                        "minimum_composite_score": float(
                            minimum_composite_row["selection_score"]
                        ),
                        "later_minimum_rmse": float(
                            minimum_rmse_row["val_last_rmse"]
                        ),
                        "later_minimum_rmse_lpr30": float(
                            minimum_rmse_row["val_last_lpr30"]
                        ),
                        "later_minimum_rmse_slpr30_10": float(
                            minimum_rmse_row["val_last_slpr30_10"]
                        ),
                        "later_minimum_rmse_composite_score": float(
                            minimum_rmse_row["selection_score"]
                        ),
                        "test_rmse": float(metrics["test_rmse"]),
                        "test_nasa_per_engine": (
                            float(metrics["test_nasa_score"])
                            / int(metrics["n_test_units"])
                        ),
                        "test_lpr30": float(
                            metrics[
                                "test_critical_30_late_prediction_ratio"
                            ]
                        ),
                    }
                )
    return pd.DataFrame(rows)


def summarize_run_quality(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for model, part in frame.groupby("model", sort=False):
        retained = part[
            ~part["early_selection_sensitivity_flag"]
        ]
        row: dict[str, object] = {
            "model": model,
            "model_display": DISPLAY_NAMES[model],
            "cell_count": len(part),
            "nonfinite_count": int((~part["prediction_finite"]).sum()),
            "early_selection_sensitivity_count": int(
                part["early_selection_sensitivity_flag"].sum()
            ),
            "objective_aligned_selection_mismatch_count": int(
                part["objective_aligned_selection_mismatch"].sum()
            ),
        }
        for metric in (
            "test_rmse",
            "test_nasa_per_engine",
            "test_lpr30",
        ):
            row[f"{metric}_mean"] = float(part[metric].mean())
            row[f"{metric}_median"] = float(part[metric].median())
            row[f"{metric}_trimmed_mean_10pct"] = trimmed_mean(
                part[metric]
            )
            row[f"{metric}_policy_excluded_mean"] = (
                float(retained[metric].mean())
                if not retained.empty
                else np.nan
            )
        rows.append(row)
    summary = pd.DataFrame(rows)
    aggregation_columns = {
        "mean": "_mean",
        "median": "_median",
        "trimmed_mean_10pct": "_trimmed_mean_10pct",
        "policy_excluded_mean": "_policy_excluded_mean",
    }
    for label, suffix in aggregation_columns.items():
        rank_columns = []
        for metric in (
            "test_rmse",
            "test_nasa_per_engine",
            "test_lpr30",
        ):
            column = f"{metric}{suffix}"
            rank_column = f"{metric}_{label}_rank"
            summary[rank_column] = summary[column].rank(
                method="average",
                ascending=True,
            )
            rank_columns.append(rank_column)
        summary[f"three_metric_average_rank_{label}"] = summary[
            rank_columns
        ].mean(axis=1)
    return summary.sort_values(
        "three_metric_average_rank_mean"
    ).reset_index(drop=True)


def leave_one_seed_out_influence(
    frame: pd.DataFrame,
    full_summary: pd.DataFrame,
) -> pd.DataFrame:
    full_ranks = full_summary.set_index("model")[
        "three_metric_average_rank_mean"
    ].to_dict()
    rows: list[dict[str, object]] = []
    for omitted_seed in SEEDS:
        retained = frame[frame["seed"] != omitted_seed]
        aggregate = retained.groupby(
            ["model", "model_display"],
            as_index=False,
        ).agg(
            test_rmse=("test_rmse", "mean"),
            test_nasa_per_engine=("test_nasa_per_engine", "mean"),
            test_lpr30=("test_lpr30", "mean"),
        )
        rank_columns = []
        for metric in (
            "test_rmse",
            "test_nasa_per_engine",
            "test_lpr30",
        ):
            column = f"{metric}_rank"
            aggregate[column] = aggregate[metric].rank(
                method="average",
                ascending=True,
            )
            rank_columns.append(column)
        aggregate["three_metric_average_rank"] = aggregate[
            rank_columns
        ].mean(axis=1)
        for row in aggregate.itertuples(index=False):
            full_rank = float(full_ranks[row.model])
            rows.append(
                {
                    "omitted_seed": omitted_seed,
                    "model": row.model,
                    "model_display": row.model_display,
                    "retained_cell_count": len(
                        retained[retained["model"] == row.model]
                    ),
                    "test_rmse_mean": row.test_rmse,
                    "test_nasa_per_engine_mean": row.test_nasa_per_engine,
                    "test_lpr30_mean": row.test_lpr30,
                    "three_metric_average_rank": row.three_metric_average_rank,
                    "full_mean_rank": full_rank,
                    "rank_change_from_full": (
                        row.three_metric_average_rank - full_rank
                    ),
                }
            )
    return pd.DataFrame(rows)


def collect_component_trajectories(frame: pd.DataFrame) -> pd.DataFrame:
    selected_rows = []
    grouped = frame.groupby(["model", "subset"], sort=True)
    for _, part in grouped:
        flagged = part[part["early_selection_sensitivity_flag"]].sort_values(
            "seed"
        )
        if flagged.empty:
            continue
        controls = part[
            ~part["early_selection_sensitivity_flag"]
        ].sort_values("seed")
        for index, (_, row) in enumerate(flagged.iterrows()):
            selected_rows.append(
                (row, "early_selection_sensitivity")
            )
            if not controls.empty:
                control = controls.iloc[index % len(controls)]
                selected_rows.append(
                    (control, "matched_nonflagged_control")
                )

    rows: list[dict[str, object]] = []
    for item, role in selected_rows:
        run_dir = (
            PROJECT_ROOT
            / "results"
            / f"paper_main_v3_seed{int(item['seed'])}"
            / str(item["subset"])
            / str(item["model"])
        )
        history = pd.read_csv(run_dir / "epoch_history.csv")
        for epoch in history.to_dict("records"):
            rows.append(
                {
                    "audit_role": role,
                    "model": item["model"],
                    "model_display": item["model_display"],
                    "subset": item["subset"],
                    "seed": int(item["seed"]),
                    "selected_checkpoint_epoch": int(
                        item["selected_checkpoint_epoch"]
                    ),
                    "objective_aligned_selection_mismatch": bool(
                        item["objective_aligned_selection_mismatch"]
                    ),
                    "epoch": int(epoch["epoch"]),
                    "val_last_rmse": float(epoch["val_last_rmse"]),
                    "val_last_lpr30": float(epoch["val_last_lpr30"]),
                    "val_last_slpr30_10": float(
                        epoch["val_last_slpr30_10"]
                    ),
                    "selection_score": float(epoch["selection_score"]),
                    "best_selection_score": float(
                        epoch["best_selection_score"]
                    ),
                    "patience_wait": int(epoch["patience_wait"]),
                }
            )
    return pd.DataFrame(rows)


def write_run_quality_tables(
    frame: pd.DataFrame,
    summary: pd.DataFrame,
    output: Path,
) -> None:
    robust_rows = []
    for row in summary.itertuples(index=False):
        robust_rows.append(
            f"{row.model_display} & "
            f"{int(row.early_selection_sensitivity_count)} & "
            f"{int(row.objective_aligned_selection_mismatch_count)} & "
            f"{row.test_rmse_mean:.3f} & "
            f"{row.test_rmse_median:.3f} & "
            f"{row.test_rmse_trimmed_mean_10pct:.3f} & "
            f"{row.three_metric_average_rank_mean:.2f} & "
            f"{row.three_metric_average_rank_median:.2f} & "
            f"{row.three_metric_average_rank_trimmed_mean_10pct:.2f} & "
            f"{row.three_metric_average_rank_policy_excluded_mean:.2f} \\\\"
        )
    robust_table = "\n".join(
        [
            r"\begin{table*}[!t]",
            r"\centering",
            r"\scriptsize",
            r"\caption{Cross-model run-quality and heavy-tail sensitivity over the 20 fixed task--seed cells per configuration. The objective-aligned audit compares the selected epoch with the minimum recorded composite score using RMSE, LPR, and SLPR components. A separate result-informed early-selection marker records selected epoch at most 2, at least 10 completed epochs, and a later endpoint-RMSE minimum at least 5 epochs away. It is not a checkpoint-failure or exclusion rule. All finite runs remain in the main mean.}",
            r"\label{tab:run-quality-robust-summary}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{lrrrrrrrrr}",
            r"\toprule",
            r"Configuration & Early flags & Objective mismatch & RMSE mean & RMSE median & RMSE trim$_{10\%}$ & Mean-rank & Median-rank & Trim-rank & Sensitivity-excluded rank \\",
            r"\midrule",
            *robust_rows,
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table*}",
            "",
        ]
    )
    flagged = frame[
        frame["early_selection_sensitivity_flag"]
    ]
    flagged_rows = [
        f"{row.model_display} & {row.subset} & {int(row.seed)} & "
        f"{int(row.selected_checkpoint_epoch)} & "
        f"{int(row.minimum_validation_endpoint_rmse_epoch)} & "
        f"{row.selected_composite_score:.4f} & "
        f"{row.later_minimum_rmse_composite_score:.4f} & "
        f"{row.test_rmse:.3f} & {row.test_nasa_per_engine:.3f} \\\\"
        for row in flagged.itertuples(index=False)
    ]
    if not flagged_rows:
        flagged_rows = [r"None & -- & -- & -- & -- & -- & -- & -- & -- \\"]
    flagged_table = "\n".join(
        [
            r"\begin{table*}[!t]",
            r"\centering",
            r"\scriptsize",
            r"\caption{Cells marked by the uniform validation-only early-selection sensitivity diagnostic. Selected and later-minimum-RMSE composite scores contain the formal RMSE, LPR, and SLPR components. Test metrics only describe retained finite outcomes and do not enter either marker.}",
            r"\label{tab:run-quality-flagged-cells}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{lrrrrrrrr}",
            r"\toprule",
            r"Configuration & Task & Seed & Selected epoch & Min RMSE epoch & Selected score & Later score & Test RMSE & NASA/engine \\",
            r"\midrule",
            *flagged_rows,
            r"\bottomrule",
            r"\end{tabular}}",
            r"\end{table*}",
            "",
        ]
    )
    (output / "table_run_quality_robust_summary.tex").write_text(
        robust_table,
        encoding="utf-8",
        newline="\n",
    )
    (output / "table_run_quality_flagged_cells.tex").write_text(
        flagged_table,
        encoding="utf-8",
        newline="\n",
    )


def plot(frame: pd.DataFrame, output: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))
    colors = {False: "#2F6690", True: "#C44E52"}
    for ax, subset in zip(axes, ("FD001", "FD003"), strict=True):
        part = frame[frame["subset"] == subset].sort_values("seed")
        for item in part.itertuples(index=False):
            ax.scatter(item.seed, item.test_rmse, s=42, color=colors[item.catastrophic_cell], zorder=3)
            ax.annotate(str(item.seed), (item.seed, item.test_rmse), xytext=(0, 5), textcoords="offset points",
                        ha="center", fontsize=7)
        ax.axhline(part["test_rmse"].median(), color="#333333", linestyle="--", linewidth=1.1,
                   label="Five-seed median")
        ax.set_title(subset, fontsize=9, fontweight="bold")
        ax.set_xlabel("Composite training seed")
        ax.set_ylabel("RMSE (cycles)")
        ax.set_xticks([])
        ax.grid(axis="y", color="0.88", linewidth=0.6)
    axes[0].legend(frameon=False, fontsize=7)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    fig.savefig(output.with_suffix(".png"), dpi=400, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    output = PROJECT_ROOT / "paper_outputs" / "advanced_evidence"
    figure_dir = output / "figures"
    output.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)
    frame = collect()
    summary = summarize(frame)
    frame.to_csv(output / "tcn_gru_seed_audit.csv", index=False)
    summary.to_csv(output / "tcn_gru_subset_robust_summary.csv", index=False)
    run_quality = collect_run_quality()
    run_quality_summary = summarize_run_quality(run_quality)
    leave_one_seed_out = leave_one_seed_out_influence(
        run_quality,
        run_quality_summary,
    )
    component_trajectories = collect_component_trajectories(run_quality)
    run_quality.to_csv(
        output / "run_quality_cell_audit.csv",
        index=False,
    )
    run_quality_summary.to_csv(
        output / "run_quality_model_robust_summary.csv",
        index=False,
    )
    leave_one_seed_out.to_csv(
        output / "run_quality_leave_one_seed_out.csv",
        index=False,
    )
    component_trajectories.to_csv(
        output / "run_quality_component_trajectories.csv",
        index=False,
    )
    write_run_quality_tables(
        run_quality,
        run_quality_summary,
        output,
    )
    plot(frame, figure_dir / "fig_tcn_gru_seed_instability")
    print(
        "TCN_GRU_INSTABILITY_AUDIT_READY "
        f"cells={len(frame)} catastrophic={int(frame['catastrophic_cell'].sum())} "
        f"nonfinite={int(frame['prediction_has_nan_or_inf'].sum())} "
        f"all_model_cells={len(run_quality)} "
        "early_selection_sensitivity_flags="
        f"{int(run_quality['early_selection_sensitivity_flag'].sum())} "
        "objective_aligned_mismatches="
        f"{int(run_quality['objective_aligned_selection_mismatch'].sum())}"
    )


if __name__ == "__main__":
    main()
