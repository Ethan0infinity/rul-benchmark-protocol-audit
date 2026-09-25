"""Audit FD004 preference selection under validation-endpoint resampling.

This analysis reuses the nine already fitted development-search candidates.
It resamples the 50 matched validation engines and therefore diagnoses
endpoint-sample instability only. It does not repeat model fitting across
training streams or engine splits.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SEARCH_GLOB = "tuning_risk_v3_ll*_lo*"
RUL_CAP = 125.0
LPR_WEIGHT = 0.25
SEVERE_WEIGHT = 0.25


def candidate_metadata(run_dir: Path) -> dict[str, object]:
    metrics = json.loads((run_dir / "metrics.json").read_text(encoding="utf-8-sig"))
    config = yaml.safe_load((run_dir / "run_config.yaml").read_text(encoding="utf-8-sig"))
    return {
        "candidate": run_dir.parents[1].name,
        "late_life_weight": float(config["training"]["late_life_weight"]),
        "late_over_weight": float(config["training"]["late_over_weight"]),
        "development_seed": int(metrics["seed"]),
        "reported_selection_score": float(metrics["best_selection_score"]),
    }


def endpoint_score(frame: pd.DataFrame) -> float:
    truth = frame["true_rul"].to_numpy(float)
    prediction = frame["pred_rul"].to_numpy(float)
    error = prediction - truth
    critical = truth <= 30.0
    rmse = float(np.sqrt(np.mean(error**2)))
    lpr = float(np.mean(error[critical] > 0.0)) if np.any(critical) else 0.0
    severe = float(np.mean(error[critical] > 10.0)) if np.any(critical) else 0.0
    return rmse / RUL_CAP + LPR_WEIGHT * lpr + SEVERE_WEIGHT * severe


def load_candidates(results_dir: Path) -> tuple[list[dict[str, object]], list[pd.DataFrame]]:
    metadata: list[dict[str, object]] = []
    predictions: list[pd.DataFrame] = []
    for experiment in sorted(results_dir.glob(SEARCH_GLOB)):
        run_dir = experiment / "FD004" / "rast_gru_v2"
        path = run_dir / "val_last_predictions.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path).sort_values("unit_id").reset_index(drop=True)
        if frame["unit_id"].duplicated().any():
            raise ValueError(f"Expected one endpoint per validation engine: {path}")
        item = candidate_metadata(run_dir)
        item["recomputed_selection_score"] = endpoint_score(frame)
        metadata.append(item)
        predictions.append(frame)
    if len(metadata) != 9:
        raise ValueError(f"Expected nine fitted preference candidates, found {len(metadata)}")
    reference_units = predictions[0]["unit_id"].tolist()
    for frame in predictions[1:]:
        if frame["unit_id"].tolist() != reference_units:
            raise ValueError("Preference candidates do not share matched validation engines")
        if not np.allclose(frame["true_rul"], predictions[0]["true_rul"]):
            raise ValueError("Preference candidates do not share validation targets")
    return metadata, predictions


def bootstrap_selection(
    metadata: list[dict[str, object]],
    predictions: list[pd.DataFrame],
    *,
    repetitions: int,
    random_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_seed)
    engine_count = len(predictions[0])
    candidate_count = len(predictions)
    selected = np.zeros(candidate_count, dtype=int)
    ranks = np.zeros((repetitions, candidate_count), dtype=float)
    oob_regret: list[float] = []
    oob_selected: list[int] = []

    for repetition in range(repetitions):
        draw = rng.integers(0, engine_count, size=engine_count)
        scores = np.asarray(
            [endpoint_score(frame.iloc[draw]) for frame in predictions],
            dtype=float,
        )
        order = np.argsort(scores, kind="stable")
        ranks[repetition, order] = np.arange(1, candidate_count + 1)
        winner = int(order[0])
        selected[winner] += 1

        in_bag = np.zeros(engine_count, dtype=bool)
        in_bag[np.unique(draw)] = True
        oob = np.flatnonzero(~in_bag)
        if len(oob):
            oob_scores = np.asarray(
                [endpoint_score(frame.iloc[oob]) for frame in predictions],
                dtype=float,
            )
            oob_regret.append(float(oob_scores[winner] - oob_scores.min()))
            oob_selected.append(winner)

    summary_rows: list[dict[str, object]] = []
    for index, item in enumerate(metadata):
        summary_rows.append(
            {
                **item,
                "endpoint_bootstrap_selection_count": int(selected[index]),
                "endpoint_bootstrap_selection_frequency": float(
                    selected[index] / repetitions
                ),
                "bootstrap_mean_rank": float(ranks[:, index].mean()),
                "bootstrap_rank_q05": float(np.quantile(ranks[:, index], 0.05)),
                "bootstrap_rank_q95": float(np.quantile(ranks[:, index], 0.95)),
                "matched_validation_engines": engine_count,
                "bootstrap_repetitions": repetitions,
                "rng_seed": random_seed,
                "scope": (
                    "same fitted candidates and development split; endpoint-resampling "
                    "diagnostic, not training-stream selection stability"
                ),
            }
        )

    oob_frame = pd.DataFrame(
        {
            "oob_regret": oob_regret,
            "selected_candidate_index": oob_selected,
        }
    )
    return pd.DataFrame(summary_rows), oob_frame


def leave_one_engine_out(
    metadata: list[dict[str, object]],
    predictions: list[pd.DataFrame],
) -> pd.DataFrame:
    engine_count = len(predictions[0])
    selected = np.zeros(len(metadata), dtype=int)
    for omitted in range(engine_count):
        keep = np.asarray([index for index in range(engine_count) if index != omitted])
        scores = np.asarray(
            [endpoint_score(frame.iloc[keep]) for frame in predictions],
            dtype=float,
        )
        selected[int(np.argmin(scores))] += 1
    return pd.DataFrame(
        [
            {
                "candidate": item["candidate"],
                "late_life_weight": item["late_life_weight"],
                "late_over_weight": item["late_over_weight"],
                "leave_one_engine_out_selection_count": int(selected[index]),
                "leave_one_engine_out_selection_frequency": float(
                    selected[index] / engine_count
                ),
                "omitted_engine_folds": engine_count,
            }
            for index, item in enumerate(metadata)
        ]
    )


def write_table(
    summary: pd.DataFrame,
    leave_one_out: pd.DataFrame,
    oob: pd.DataFrame,
    output: Path,
) -> None:
    merged = summary.merge(
        leave_one_out[
            [
                "candidate",
                "leave_one_engine_out_selection_frequency",
            ]
        ],
        on="candidate",
        how="left",
    ).sort_values("endpoint_bootstrap_selection_frequency", ascending=False)
    rows = [
        f"{row['late_life_weight']:.2f} & {row['late_over_weight']:.2f} & "
        f"{row['recomputed_selection_score']:.4f} & "
        f"{row['endpoint_bootstrap_selection_frequency']:.3f} & "
        f"{row['leave_one_engine_out_selection_frequency']:.3f} & "
        f"{row['bootstrap_mean_rank']:.2f} \\\\"
        for row in merged.to_dict("records")
    ]
    regret_mean = float(oob["oob_regret"].mean())
    regret_q95 = float(oob["oob_regret"].quantile(0.95))
    text = "\n".join(
        [
            r"\begin{table}[!t]",
            r"\centering",
            r"\scriptsize",
            r"\caption{FD004 preference-selection endpoint-resampling audit. The nine candidates are the same seed-31415 fitted models used in the original development search. Frequencies resample or omit the 50 matched validation engines; they do not repeat training across splits or training streams.}",
            r"\label{tab:preference-selection-stability}",
            r"\begin{tabular}{rrrrrr}",
            r"\toprule",
            r"$\lambda_{\mathrm{life}}$ & $\lambda_{\mathrm{late}}$ & Full score & Boot. select & LOO select & Mean rank \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\par\footnotesize OOB score regret, selected minus OOB-best: mean "
                f"{regret_mean:.4f}, 95th percentile {regret_q95:.4f}. "
                r"This is endpoint-sample instability only; the operating point remains a researcher-specified illustrative preference."
            ),
            r"\end{table}",
            "",
        ]
    )
    output.write_text(text, encoding="utf-8", newline="\n")
    chinese = "\n".join(
        [
            r"\begin{table}[!t]",
            r"\centering",
            r"\scriptsize",
            r"\caption{FD004 偏好选择的端点重采样审计。九个候选均为原开发搜索中种子 31415 的已拟合模型；频率只重采样或遗漏 50 台匹配验证发动机，不跨划分或训练流重训。}",
            r"\label{tab:preference-selection-stability-zh}",
            r"\begin{tabular}{rrrrrr}",
            r"\toprule",
            r"$\lambda_{\mathrm{life}}$ & $\lambda_{\mathrm{late}}$ & 全样本分数 & Bootstrap 入选率 & LOO 入选率 & 平均名次 \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\par\footnotesize OOB 分数 regret（所选减 OOB 最优）：均值 "
                f"{regret_mean:.4f}，95 分位数 {regret_q95:.4f}。"
                r"该审计只识别端点样本不稳定性；该工作点仍属于研究者选择。"
            ),
            r"\end{table}",
            "",
        ]
    )
    output.with_name(output.stem + "_zh.tex").write_text(
        chinese, encoding="utf-8", newline="\n"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Audit FD004 preference selection under endpoint resampling."
    )
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument(
        "--output-dir",
        default=str(PROJECT_ROOT / "paper_outputs" / "analysis_working"),
    )
    parser.add_argument("--bootstrap-repetitions", type=int, default=10000)
    args = parser.parse_args()

    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    metadata, predictions = load_candidates(Path(args.results_dir))
    summary, oob = bootstrap_selection(
        metadata,
        predictions,
        repetitions=args.bootstrap_repetitions,
        random_seed=2026072808,
    )
    leave_one_out = leave_one_engine_out(metadata, predictions)
    summary.to_csv(output / "preference_selection_stability.csv", index=False)
    leave_one_out.to_csv(output / "preference_selection_leave_one_out.csv", index=False)
    oob.to_csv(output / "preference_selection_oob_regret.csv", index=False)
    write_table(
        summary,
        leave_one_out,
        oob,
        output / "table_preference_selection_stability.tex",
    )
    print(
        "PREFERENCE_SELECTION_STABILITY_READY "
        f"candidates={len(summary)} engines={len(predictions[0])} "
        f"bootstrap={args.bootstrap_repetitions}"
    )


if __name__ == "__main__":
    main()
