from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEEDS = (42, 123, 2024, 2025, 2026)
SUBSETS = ("FD001", "FD002", "FD003", "FD004")
BATTERY_MODELS = ("core", "asym")
NORMALIZATIONS = ("global", "temperature")
Q_GRID = (0.05, 0.10, 0.20)
EPSILON_GRID = (0.0, 0.02, 0.05)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prediction_path(project: Path, point: str, subset: str, seed: int) -> Path:
    if point == "asym":
        return project / "results" / f"paper_main_v3_seed{seed}" / subset / "rast_gru_v2" / "test_predictions.csv"
    if point == "rast":
        return project / "results" / f"paper_main_v3_seed{seed}" / subset / "rast_gru" / "test_predictions.csv"
    if point != "core":
        raise ValueError(f"Unknown point: {point}")
    if subset == "FD004":
        return (
            project
            / "results"
            / f"ablation_fd004_seed{seed}_weighted_huber_no_asymmetry"
            / subset
            / "rast_gru_v2"
            / "test_predictions.csv"
        )
    return project / "results" / f"core_ocm_seed{seed}" / subset / "rast_gru_v2" / "test_predictions.csv"


def load_cmapss_predictions(project: Path, point: str, subset: str, seed: int) -> pd.DataFrame:
    path = prediction_path(project, point, subset, seed)
    frame = pd.read_csv(path)
    required = {"unit_id", "true_rul", "pred_rul"}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} lacks {sorted(required - set(frame.columns))}")
    if frame["unit_id"].duplicated().any():
        raise ValueError(f"Duplicate engine endpoints in {path}")
    return frame.sort_values("unit_id").reset_index(drop=True)


def decision_metrics(frame: pd.DataFrame, lead: float, late_cost_ratio: float) -> dict[str, float]:
    truth = frame["true_rul"].to_numpy(dtype=float) <= lead
    action = frame["pred_rul"].to_numpy(dtype=float) <= lead
    false_late = truth & ~action
    false_early = ~truth & action
    return {
        "n_engines": int(len(frame)),
        "service_rate": float(np.mean(action)),
        "false_late_rate": float(np.mean(false_late)),
        "false_early_rate": float(np.mean(false_early)),
        "decision_cost_per_engine": float(np.mean(late_cost_ratio * false_late + false_early)),
    }


def build_maintenance_decision_layer(project: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, object]] = []
    comparisons = (("asym", "rast"), ("core", "rast"), ("asym", "core"))
    for subset in SUBSETS:
        for seed in SEEDS:
            loaded = {point: load_cmapss_predictions(project, point, subset, seed) for point in ("asym", "core", "rast")}
            ids = loaded["asym"]["unit_id"].tolist()
            if any(frame["unit_id"].tolist() != ids for frame in loaded.values()):
                raise ValueError(f"Unmatched engine IDs for {subset}, seed {seed}")
            if any(not np.array_equal(frame["true_rul"], loaded["asym"]["true_rul"]) for frame in loaded.values()):
                raise ValueError(f"Unmatched true RUL for {subset}, seed {seed}")
            for lead in (10.0, 20.0, 30.0):
                actions = {
                    point: frame["pred_rul"].to_numpy(dtype=float) <= lead for point, frame in loaded.items()
                }
                for late_cost_ratio in (2.0, 5.0, 10.0):
                    metrics = {
                        point: decision_metrics(frame, lead, late_cost_ratio) for point, frame in loaded.items()
                    }
                    for left, right in comparisons:
                        row: dict[str, object] = {
                            "subset": subset,
                            "seed": seed,
                            "left_point": left,
                            "right_point": right,
                            "maintenance_lead_cycles": lead,
                            "late_to_early_cost_ratio": late_cost_ratio,
                            "independent_unit": "test engine endpoint",
                            "design_role": "post-result exploratory static maintenance-triage sensitivity",
                        }
                        for name in ("service_rate", "false_late_rate", "false_early_rate", "decision_cost_per_engine"):
                            row[f"left_{name}"] = metrics[left][name]
                            row[f"right_{name}"] = metrics[right][name]
                            row[f"delta_{name}"] = metrics[left][name] - metrics[right][name]
                        row["action_disagreement_rate"] = float(np.mean(actions[left] != actions[right]))
                        row["n_engines"] = metrics[left]["n_engines"]
                        rows.append(row)
    cells = pd.DataFrame(rows)
    summary_rows: list[dict[str, object]] = []
    group_columns = ["left_point", "right_point", "maintenance_lead_cycles", "late_to_early_cost_ratio"]
    for keys, group in cells.groupby(group_columns, sort=True):
        delta = group["delta_decision_cost_per_engine"].to_numpy(dtype=float)
        summary_rows.append(
            {
                **dict(zip(group_columns, keys)),
                "equal_task_seed_mean_delta_cost": float(np.mean(delta)),
                "median_delta_cost": float(np.median(delta)),
                "left_lower_cells": int(np.sum(delta < -1.0e-12)),
                "ties": int(np.sum(np.abs(delta) <= 1.0e-12)),
                "right_lower_cells": int(np.sum(delta > 1.0e-12)),
                "mean_action_disagreement_rate": float(group["action_disagreement_rate"].mean()),
                "mean_delta_false_late_rate": float(group["delta_false_late_rate"].mean()),
                "mean_delta_false_early_rate": float(group["delta_false_early_rate"].mean()),
                "task_seed_cells": int(len(group)),
                "interpretation": "finite-design policy sensitivity; lower cost is preferable",
            }
        )
    return cells, pd.DataFrame(summary_rows)


def build_cmapss_equivalence(project: Path) -> pd.DataFrame:
    source = project / "paper_outputs" / "release_v1" / "primary_seed_level_effects.csv"
    effects = pd.read_csv(source)
    margins = {"rmse": (0.5, 1.0, 2.0), "lpr30": (0.01, 0.025, 0.05)}
    rows: list[dict[str, object]] = []
    for metric, metric_margins in margins.items():
        current = effects[effects["metric"] == metric].copy()
        for margin in metric_margins:
            values = current["ocm_asym_minus_rast"].to_numpy(dtype=float)
            rows.append(
                {
                    "domain": "C-MAPSS",
                    "contrast": "OCM-Asym minus RAST-GRU",
                    "metric": metric,
                    "candidate_margin": margin,
                    "finite_design_mean": float(np.mean(values)),
                    "within_margin_cells": int(np.sum(np.abs(values) <= margin)),
                    "negative_beyond_margin_cells": int(np.sum(values < -margin)),
                    "positive_beyond_margin_cells": int(np.sum(values > margin)),
                    "total_task_seed_cells": int(len(values)),
                    "status": "exploratory transparent band; not a validated SESOI or equivalence test",
                }
            )
    return pd.DataFrame(rows)


def load_battery_bundle(path: Path, labels: bool) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        required = {"sample_id", "unit_id", "cycle_index"}
        if labels:
            required |= {"y", "lifetime"}
        if not required.issubset(payload.files):
            raise ValueError(f"{path} lacks {sorted(required - set(payload.files))}")
        return {key: np.asarray(payload[key]) for key in payload.files}


def battery_paths(project: Path) -> Path:
    return project / "studies" / "prospective_battery_freeze_v1"


def load_battery_predictions(root: Path, normalization: str, model: str, seed: int) -> pd.DataFrame:
    path = root / "predictions" / normalization / model / f"seed_{seed}.csv"
    frame = pd.read_csv(path)
    labels = load_battery_bundle(root / "firewalled" / f"test_labels_{normalization}.npz", labels=True)
    label_frame = pd.DataFrame(
        {
            "sample_id": labels["sample_id"].astype(str),
            "unit_id": labels["unit_id"].astype(str),
            "cycle_index": labels["cycle_index"].astype(int),
            "y": labels["y"].astype(float),
            "J": labels["lifetime"].astype(float),
        }
    )
    joined = frame.merge(label_frame, on=["sample_id", "unit_id", "cycle_index"], how="inner", validate="one_to_one")
    if len(joined) != len(frame) or len(joined) != len(label_frame):
        raise ValueError(f"Prediction/label mismatch for {normalization}/{model}/seed {seed}")
    return joined


def battery_unit_metrics(frame: pd.DataFrame, q: float, epsilon: float, denominator: str) -> dict[str, float]:
    if denominator not in {"J", "J_plus_1"}:
        raise ValueError(denominator)
    denom = frame["J"].to_numpy(dtype=float) + (1.0 if denominator == "J_plus_1" else 0.0)
    y = frame["y"].to_numpy(dtype=float)
    pred = frame["prediction_rul_cycles"].to_numpy(dtype=float)
    eligible = y / denom <= q
    late = pred - y > epsilon * denom
    return {
        "rmse": float(np.sqrt(np.mean(np.square(pred - y)))),
        "eligible": int(np.sum(eligible)),
        "late": int(np.sum(eligible & late)),
        "lpr": float(np.mean(late[eligible])) if np.any(eligible) else float("nan"),
    }


def build_battery_index_sensitivity(project: Path) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    root = battery_paths(project)
    rows: list[dict[str, object]] = []
    for normalization in NORMALIZATIONS:
        for model in BATTERY_MODELS:
            for seed in SEEDS:
                frame = load_battery_predictions(root, normalization, model, seed)
                for denominator in ("J", "J_plus_1"):
                    for q in Q_GRID:
                        for epsilon in EPSILON_GRID:
                            for unit_id, unit in frame.groupby("unit_id", sort=True):
                                metrics = battery_unit_metrics(unit, q, epsilon, denominator)
                                rows.append(
                                    {
                                        "normalization": normalization,
                                        "model": model,
                                        "seed": seed,
                                        "unit_id": unit_id,
                                        "denominator_definition": denominator,
                                        "q": q,
                                        "epsilon_fraction": epsilon,
                                        "rmse_cycles": metrics["rmse"],
                                        "eligible_windows": metrics["eligible"],
                                        "late_events": metrics["late"],
                                        "lpr": metrics["lpr"],
                                    }
                                )
    unit = pd.DataFrame(rows)
    seed = (
        unit.groupby(["normalization", "model", "seed", "denominator_definition", "q", "epsilon_fraction"], as_index=False)
        .agg(
            macro_rmse_cycles=("rmse_cycles", "mean"),
            macro_lpr=("lpr", "mean"),
            pooled_eligible_windows=("eligible_windows", "sum"),
            pooled_late_events=("late_events", "sum"),
        )
    )
    paired_rows: list[dict[str, object]] = []
    for keys, group in seed.groupby(["normalization", "seed", "denominator_definition", "q", "epsilon_fraction"]):
        by_model = group.set_index("model")
        if set(by_model.index) != set(BATTERY_MODELS):
            raise ValueError(f"Incomplete battery pair: {keys}")
        paired_rows.append(
            {
                "normalization": keys[0],
                "seed": keys[1],
                "denominator_definition": keys[2],
                "q": keys[3],
                "epsilon_fraction": keys[4],
                "delta_macro_rmse_cycles": float(by_model.loc["asym", "macro_rmse_cycles"] - by_model.loc["core", "macro_rmse_cycles"]),
                "delta_macro_lpr": float(by_model.loc["asym", "macro_lpr"] - by_model.loc["core", "macro_lpr"]),
                "core_pooled_events": int(by_model.loc["core", "pooled_late_events"]),
                "asym_pooled_events": int(by_model.loc["asym", "pooled_late_events"]),
                "core_pooled_eligible": int(by_model.loc["core", "pooled_eligible_windows"]),
                "asym_pooled_eligible": int(by_model.loc["asym", "pooled_eligible_windows"]),
                "role": "post-registration definition sensitivity; four batteries are the independent units",
            }
        )
    paired = pd.DataFrame(paired_rows)
    pivot = paired.pivot_table(
        index=["normalization", "seed", "q", "epsilon_fraction"],
        columns="denominator_definition",
        values="delta_macro_lpr",
    ).reset_index()
    pivot["J_plus_1_minus_J_delta_lpr"] = pivot["J_plus_1"] - pivot["J"]
    return unit, paired, pivot


def ridge_predict(x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, alpha: float) -> np.ndarray:
    center = x_train.mean(axis=0)
    scale = x_train.std(axis=0)
    scale[scale < 1.0e-12] = 1.0
    train = (x_train - center) / scale
    test = (x_test - center) / scale
    design = np.column_stack([np.ones(len(train)), train])
    test_design = np.column_stack([np.ones(len(test)), test])
    penalty = np.eye(design.shape[1]) * alpha
    penalty[0, 0] = 0.0
    coefficients = np.linalg.solve(design.T @ design + penalty, design.T @ y_train)
    return np.maximum(test_design @ coefficients, 0.0)


def evaluate_battery_prediction(
    prediction: np.ndarray, labels: dict[str, np.ndarray], q: float = 0.10, epsilon: float = 0.0
) -> dict[str, float]:
    frame = pd.DataFrame(
        {
            "unit_id": labels["unit_id"].astype(str),
            "y": labels["y"].astype(float),
            "J": labels["lifetime"].astype(float),
            "prediction_rul_cycles": prediction.astype(float),
        }
    )
    units = [battery_unit_metrics(unit, q, epsilon, "J") for _, unit in frame.groupby("unit_id", sort=True)]
    return {
        "macro_rmse_cycles": float(np.mean([item["rmse"] for item in units])),
        "macro_lpr_q10_e0": float(np.mean([item["lpr"] for item in units])),
        "pooled_eligible_windows": int(np.sum([item["eligible"] for item in units])),
        "pooled_late_events": int(np.sum([item["late"] for item in units])),
        "pooled_late_frequency": float(np.sum([item["late"] for item in units]) / np.sum([item["eligible"] for item in units])),
    }


def build_battery_baselines(project: Path) -> pd.DataFrame:
    root = battery_paths(project)
    train = load_battery_bundle(root / "prepared" / "training_global.npz", labels=True)
    test = load_battery_bundle(root / "prepared" / "test_inputs_global.npz", labels=False)
    labels = load_battery_bundle(root / "firewalled" / "test_labels_global.npz", labels=True)
    if not np.array_equal(test["sample_id"].astype(str), labels["sample_id"].astype(str)):
        raise ValueError("Battery test inputs and labels are not aligned")
    train_x = train["x"].astype(float)
    test_x = test["x"].astype(float)
    y_train = train["y"].astype(float)
    median_j = float(
        np.median(
            pd.DataFrame({"unit": train["unit_id"].astype(str), "J": train["lifetime"]})
            .drop_duplicates("unit")["J"]
        )
    )
    predictions = {
        "training-life-prior": np.maximum(median_j - test["cycle_index"].astype(float), 0.0),
        "capacity-index-linear": ridge_predict(
            np.column_stack([train_x[:, -1, 0], train_x[:, -1, 7]]),
            y_train,
            np.column_stack([test_x[:, -1, 0], test_x[:, -1, 7]]),
            alpha=1.0,
        ),
        "window-summary-ridge": ridge_predict(
            np.column_stack([train_x[:, -1, :], train_x.mean(axis=1), train_x[:, -1, :] - train_x[:, 0, :]]),
            y_train,
            np.column_stack([test_x[:, -1, :], test_x.mean(axis=1), test_x[:, -1, :] - test_x[:, 0, :]]),
            alpha=1.0,
        ),
    }
    rows = []
    for name, prediction in predictions.items():
        rows.append(
            {
                "model": name,
                **evaluate_battery_prediction(prediction, labels),
                "training_seed": "deterministic",
                "analysis_status": "post-registration exploratory baseline; excluded from registered hypothesis tests",
            }
        )
    for model in BATTERY_MODELS:
        seed_metrics = []
        for seed in SEEDS:
            frame = load_battery_predictions(root, "global", model, seed)
            label_dict = {
                "unit_id": frame["unit_id"].to_numpy(),
                "y": frame["y"].to_numpy(),
                "lifetime": frame["J"].to_numpy(),
            }
            seed_metrics.append(evaluate_battery_prediction(frame["prediction_rul_cycles"].to_numpy(), label_dict))
        rows.append(
            {
                "model": f"Battery-GRU-{model.capitalize()} (five-stream mean)",
                **{
                    key: float(np.mean([item[key] for item in seed_metrics]))
                    for key in (
                        "macro_rmse_cycles",
                        "macro_lpr_q10_e0",
                        "pooled_eligible_windows",
                        "pooled_late_events",
                        "pooled_late_frequency",
                    )
                },
                "training_seed": "mean over 42,123,2024,2025,2026",
                "analysis_status": "registered model reference; baseline comparison itself is post-registration exploratory",
            }
        )
    return pd.DataFrame(rows)


def build_battery_equivalence(paired: pd.DataFrame) -> pd.DataFrame:
    primary = paired[
        (paired["normalization"] == "global")
        & (paired["denominator_definition"] == "J")
        & np.isclose(paired["q"], 0.10)
        & np.isclose(paired["epsilon_fraction"], 0.0)
    ]
    rows: list[dict[str, object]] = []
    for metric, column, margins in (
        ("rmse", "delta_macro_rmse_cycles", (0.5, 1.0, 2.0)),
        ("lpr", "delta_macro_lpr", (0.01, 0.025, 0.05)),
    ):
        values = primary[column].to_numpy(dtype=float)
        for margin in margins:
            rows.append(
                {
                    "domain": "NASA battery",
                    "contrast": "Battery-GRU-Asym minus Battery-GRU-Core",
                    "metric": metric,
                    "candidate_margin": margin,
                    "finite_design_mean": float(np.mean(values)),
                    "within_margin_streams": int(np.sum(np.abs(values) <= margin)),
                    "negative_beyond_margin_streams": int(np.sum(values < -margin)),
                    "positive_beyond_margin_streams": int(np.sum(values > margin)),
                    "total_streams": int(len(values)),
                    "independent_test_units": 4,
                    "status": "post-registration exploratory transparent band; streams are repeated fitting variations",
                }
            )
    return pd.DataFrame(rows)


def build_timeline() -> pd.DataFrame:
    return pd.DataFrame(
        [
            ("2026-07-04", "C-MAPSS project scaffold and initial experiment design", "local project records", "precedes later governance analyses; no external timestamp"),
            ("before 2026-07-20", "Common-protocol C-MAPSS training grid completed", "analysis chronology", "results later inspected; retrospective evidence"),
            ("2026-07-26", "Crossed seed-engine reanalysis and threshold diagnostics", "analysis chronology", "result-informed reanalysis; no retraining"),
            ("2026-07-26/27", "Normalization, perturbation, mask, and latency audits", "analysis chronology", "post-result sensitivity families"),
            ("2026-07-29 to 2026-08-08", "Retrospective extension and release-governance revisions", "round 10--19 response records", "completed after earlier result access"),
            ("2026-08-09", "Evidence snapshot 2026-08-09-r21 assembled", "workflow state", "local evidence snapshot; not a prospective protocol"),
            ("2026-08-12T09:10:33Z", "Battery protocol externally registered", "external timestamp record", "precedes battery training and test-label evaluation"),
            ("after 2026-08-12 registration", "Registered battery pair trained and evaluated", "battery results audit", "prospectively frozen worked execution; same investigator"),
            ("2026-08-16", "Source provenance correction verified in v5.3.1", "Git tag v5.3.1 and repository history", "corrected source snapshot; does not alter evidence timing"),
            ("2026-08-16", "RESS major-revision exploratory analyses generated", "major-revision evidence generator", "post-result maintenance, equivalence, baseline, and target-definition sensitivities"),
            ("2026-08-16", "RESS submission object rebuilt", "Git tag v5.4.0 and submission_identity.json", "release synchronization and journal-format revision; no evidence-status upgrade"),
        ],
        columns=["date_or_period", "event", "source_record", "evidence_status"],
    )


def tex_escape(text: str) -> str:
    return text.replace("_", r"\_").replace("%", r"\%")


def write_tex_tables(
    maintenance: pd.DataFrame,
    cmapss_equivalence: pd.DataFrame,
    battery_equivalence: pd.DataFrame,
    baselines: pd.DataFrame,
    index_paired: pd.DataFrame,
    timeline: pd.DataFrame,
) -> None:
    generated = PROJECT_ROOT / "manuscript" / "RESS" / "generated"
    generated.mkdir(parents=True, exist_ok=True)

    focus = maintenance[(maintenance["left_point"] == "asym") & (maintenance["right_point"] == "rast")].copy()
    lines = [
        r"\begin{table}[!t]", r"\centering", r"\caption{Exploratory static maintenance-triage sensitivity for OCM-Asym minus RAST-GRU.}",
        r"\label{tab:maintenance-decision}", r"\scriptsize", r"\setlength{\tabcolsep}{3.5pt}", r"\begin{tabular}{rrrrrr}", r"\toprule",
        r"Lead $h$ & Ratio $\rho$ & $\Delta C$/engine & Cost $-/0/+$ & Action disagree. & $\Delta$ false-late \\", r"\midrule",
    ]
    for row in focus.itertuples():
        lines.append(
            f"{row.maintenance_lead_cycles:.0f} & {row.late_to_early_cost_ratio:.0f} & {row.equal_task_seed_mean_delta_cost:+.4f} & "
            f"{row.left_lower_cells}/{row.ties}/{row.right_lower_cells} & {row.mean_action_disagreement_rate:.4f} & {row.mean_delta_false_late_rate:+.4f} \\\\"
        )
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\begin{minipage}{0.98\linewidth}\footnotesize Notes: Service is triggered when predicted RUL is no greater than the declared lead. Cost equals $C_L$ for a deferred action when true RUL is within the lead and $C_E$ for premature service otherwise. Each summary gives an equal-weight mean over four tasks and five composite seeds. Negative differences favor OCM-Asym. This is a post-result static decision sensitivity, not an optimized fleet policy, failure-probability model, or field utility estimate.\end{minipage}",
        r"\end{table}", "",
    ]
    (generated / "table_maintenance_decision.tex").write_text("\n".join(lines), encoding="utf-8")

    combined = pd.concat(
        [
            cmapss_equivalence.rename(columns={"total_task_seed_cells": "total", "within_margin_cells": "within", "negative_beyond_margin_cells": "negative", "positive_beyond_margin_cells": "positive"})[
                ["domain", "metric", "candidate_margin", "finite_design_mean", "within", "negative", "positive", "total"]
            ],
            battery_equivalence.rename(columns={"total_streams": "total", "within_margin_streams": "within", "negative_beyond_margin_streams": "negative", "positive_beyond_margin_streams": "positive"})[
                ["domain", "metric", "candidate_margin", "finite_design_mean", "within", "negative", "positive", "total"]
            ],
        ],
        ignore_index=True,
    )
    lines = [
        r"\begin{table}[!t]", r"\centering", r"\caption{Transparent practical-equivalence-band sensitivity.}", r"\label{tab:practical-equivalence}",
        r"\small", r"\begin{tabular}{llrrrrr}", r"\toprule",
        r"Domain & Metric & Margin & Mean contrast & Within & Negative beyond & Positive beyond \\", r"\midrule",
    ]
    for row in combined.itertuples():
        lines.append(f"{tex_escape(row.domain)} & {row.metric.upper()} & {row.candidate_margin:g} & {row.finite_design_mean:+.4f} & {row.within}/{row.total} & {row.negative} & {row.positive} \\\\")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\begin{minipage}{0.98\linewidth}\footnotesize Notes: Negative/positive are first-minus-second contrasts beyond the displayed symmetric margin. Margins were introduced after outcome access and are not application-validated smallest effects of interest. The table separates sign sensitivity from magnitude sensitivity; it is not a formal equivalence test.\end{minipage}",
        r"\end{table}", "",
    ]
    (generated / "table_practical_equivalence.tex").write_text("\n".join(lines), encoding="utf-8")

    lines = [
        r"\begin{table}[!t]", r"\centering", r"\caption{Post-registration exploratory battery baselines under global training-only normalization.}",
        r"\label{tab:battery-baselines}", r"\scriptsize", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{p{0.34\linewidth}rrrr}", r"\toprule",
        r"Estimator & Macro RMSE & Macro LPR$_{10\%}$ & Pooled late/eligible & Pooled frequency \\", r"\midrule",
    ]
    for row in baselines.itertuples():
        lines.append(
            f"{tex_escape(row.model)} & {row.macro_rmse_cycles:.3f} & {row.macro_lpr_q10_e0:.4f} & "
            f"{row.pooled_late_events:.1f}/{row.pooled_eligible_windows:.1f} & {row.pooled_late_frequency:.4f} \\\\"
        )
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\begin{minipage}{0.98\linewidth}\footnotesize Notes: The three simple estimators were added after registration and do not enter the registered hypothesis or wording rule. GRU rows are five-training-stream means. LPR is an equal-battery macro average; pooled counts use 48 eligible windows per stream for the registered models and must not be reverse-calculated from macro LPR.\end{minipage}",
        r"\end{table}", "",
    ]
    (generated / "table_battery_exploratory_baselines.tex").write_text("\n".join(lines), encoding="utf-8")

    primary = index_paired[
        (index_paired["normalization"] == "global")
        & np.isclose(index_paired["q"], 0.10)
        & np.isclose(index_paired["epsilon_fraction"], 0.0)
    ]
    grouped = primary.groupby("denominator_definition", as_index=False).agg(
        delta_rmse=("delta_macro_rmse_cycles", "mean"),
        delta_lpr=("delta_macro_lpr", "mean"),
        core_events=("core_pooled_events", "sum"),
        asym_events=("asym_pooled_events", "sum"),
        eligible=("core_pooled_eligible", "sum"),
    )
    lines = [
        r"\begin{table}[!t]", r"\centering", r"\caption{Battery target-denominator definition sensitivity at the primary $q=0.10$, $\epsilon=0$ setting.}",
        r"\label{tab:battery-index-sensitivity}", r"\small", r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Denominator & $\Delta$ macro RMSE & $\Delta$ macro LPR & Core/Asym pooled events & Eligible windows \\", r"\midrule",
    ]
    for row in grouped.itertuples():
        label = "$J_i$" if row.denominator_definition == "J" else "$J_i+1$"
        lines.append(f"{label} & {row.delta_rmse:+.4f} & {row.delta_lpr:+.4f} & {row.core_events}/{row.asym_events} & {row.eligible} \\\\")
    lines += [
        r"\bottomrule", r"\end{tabular}",
        r"\begin{minipage}{0.98\linewidth}\footnotesize Notes: $J_i$ is the final zero-based discharge index and $J_i+1$ is the observed discharge-count denominator. The trained target and predictions remain $J_i-j$; only the life-fraction denominator changes. RMSE is therefore invariant. Event counts are pooled over four batteries and five streams, whereas LPR is first averaged equally across batteries and then across streams.\end{minipage}",
        r"\end{table}", "",
    ]
    (generated / "table_battery_index_sensitivity.tex").write_text("\n".join(lines), encoding="utf-8")

    lines = [
        r"\begin{longtable}{@{}>{\raggedright\arraybackslash}p{0.13\textwidth}>{\raggedright\arraybackslash}p{0.29\textwidth}>{\raggedright\arraybackslash}p{0.22\textwidth}>{\raggedright\arraybackslash}p{0.27\textwidth}@{}}",
        r"\caption{Complete analysis and release timeline. Later artifact repair does not change earlier design timing.}\label{tab:analysis-timeline} \\",
        r"\toprule", r"Date/period & Event & Source record & Evidence status \\", r"\midrule", r"\endfirsthead",
        r"\toprule", r"Date/period & Event & Source record & Evidence status \\", r"\midrule", r"\endhead",
    ]
    for row in timeline.itertuples():
        lines.append(
            f"{tex_escape(row.date_or_period)} & {tex_escape(row.event)} & {tex_escape(row.source_record)} & {tex_escape(row.evidence_status)} \\\\"
        )
    lines += [r"\bottomrule", r"\end{longtable}", ""]
    (generated / "table_analysis_timeline.tex").write_text("\n".join(lines), encoding="utf-8")


def plot_maintenance(summary: pd.DataFrame) -> None:
    focus = summary[(summary["left_point"] == "asym") & (summary["right_point"] == "rast")]
    pivot = focus.pivot(index="late_to_early_cost_ratio", columns="maintenance_lead_cycles", values="equal_task_seed_mean_delta_cost")
    span = np.max(np.abs(pivot.to_numpy()))
    fig, ax = plt.subplots(figsize=(6.6, 3.8))
    image = ax.imshow(pivot.to_numpy(), cmap="RdBu_r", aspect="auto", vmin=-span, vmax=span)
    ax.set_xticks(range(len(pivot.columns)), [f"{value:.0f}" for value in pivot.columns])
    ax.set_yticks(range(len(pivot.index)), [f"{value:.0f}" for value in pivot.index])
    ax.set_xlabel("Maintenance lead (cycles)")
    ax.set_ylabel(r"Late-to-early cost ratio $C_L/C_E$")
    for i, ratio in enumerate(pivot.index):
        for j, lead in enumerate(pivot.columns):
            value = pivot.loc[ratio, lead]
            ax.text(j, i, f"{value:+.3f}", ha="center", va="center", color="black", fontsize=9)
    colorbar = fig.colorbar(image, ax=ax, shrink=0.88)
    colorbar.set_label("OCM-Asym minus RAST cost per engine")
    ax.set_title("Post-result static maintenance-triage sensitivity", loc="left", fontweight="bold")
    fig.tight_layout()
    figure_root = PROJECT_ROOT / "manuscript" / "RESS" / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(figure_root / f"fig_maintenance_decision.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build post-result RESS major-revision evidence from stored predictions.")
    parser.add_argument("--project-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--output-dir", type=Path, default=PROJECT_ROOT / "paper_outputs" / "ress_major_revision")
    args = parser.parse_args()
    project = args.project_root.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    maintenance_cells, maintenance_summary = build_maintenance_decision_layer(project)
    cmapss_equivalence = build_cmapss_equivalence(project)
    battery_unit, battery_paired, battery_definition_change = build_battery_index_sensitivity(project)
    battery_baselines = build_battery_baselines(project)
    battery_equivalence = build_battery_equivalence(battery_paired)
    timeline = build_timeline()

    outputs = {
        "maintenance_decision_cell_level.csv": maintenance_cells,
        "maintenance_decision_summary.csv": maintenance_summary,
        "cmapss_practical_equivalence.csv": cmapss_equivalence,
        "battery_practical_equivalence.csv": battery_equivalence,
        "battery_exploratory_baselines.csv": battery_baselines,
        "battery_index_definition_unit_level.csv": battery_unit,
        "battery_index_definition_paired.csv": battery_paired,
        "battery_index_definition_change.csv": battery_definition_change,
        "analysis_timeline.csv": timeline,
    }
    for name, frame in outputs.items():
        frame.to_csv(output / name, index=False)

    write_tex_tables(
        maintenance_summary,
        cmapss_equivalence,
        battery_equivalence,
        battery_baselines,
        battery_paired,
        timeline,
    )
    plot_maintenance(maintenance_summary)

    input_paths = [project / "paper_outputs" / "release_v1" / "primary_seed_level_effects.csv"]
    input_paths.extend(prediction_path(project, point, subset, seed) for point in ("asym", "core", "rast") for subset in SUBSETS for seed in SEEDS)
    battery_root = battery_paths(project)
    input_paths.extend(battery_root.glob("predictions/*/*/seed_*.csv"))
    input_paths.extend(battery_root.glob("prepared/*.npz"))
    input_paths.extend(battery_root.glob("firewalled/*.npz"))
    manifest = {
        "analysis_status": "post-result exploratory major-revision evidence",
        "generator": str(Path(__file__).resolve().relative_to(project)).replace("\\", "/"),
        "inputs": [
            {"path": str(path.resolve().relative_to(project)).replace("\\", "/"), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in sorted(set(input_paths))
        ],
        "outputs": [
            {"path": str(path.resolve().relative_to(project)).replace("\\", "/"), "sha256": sha256(path), "bytes": path.stat().st_size}
            for path in sorted(output.glob("*"))
            if path.is_file()
        ],
        "claim_boundary": "No output is confirmatory, a field-validated maintenance policy, a reliability probability, or an independent comparator-study result.",
    }
    (output / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"RESS_MAJOR_REVISION_EVIDENCE_PASS output={output}")
    print(f"maintenance_cells={len(maintenance_cells)} battery_definition_rows={len(battery_paired)}")


if __name__ == "__main__":
    main()
