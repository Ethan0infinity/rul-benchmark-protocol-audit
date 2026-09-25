from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))


def _read_csv_if_exists(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path)
    return df if not df.empty else None


def _compact_markdown(
    df: pd.DataFrame,
    columns: list[str],
    *,
    sort_by: str | None = None,
    ascending: bool = True,
    max_rows: int | None = None,
) -> str:
    available = [column for column in columns if column in df.columns]
    compact = df[available].copy()
    if sort_by and sort_by in compact.columns:
        try:
            compact[sort_by] = pd.to_numeric(compact[sort_by])
        except (TypeError, ValueError):
            pass
        compact = compact.sort_values(sort_by, ascending=ascending)
    if max_rows is not None:
        compact = compact.head(max_rows)
    return compact.to_markdown(index=False)


def build_result_brief(tables_dir: str | Path) -> str:
    tables_dir = Path(tables_dir)
    lines = ["# Result Brief", ""]

    main_df = _read_csv_if_exists(tables_dir / "table_main_accuracy.csv")
    if main_df is not None:
        lines.extend(["## Main Accuracy", "", main_df.to_markdown(index=False), ""])
        if "model" in main_df.columns and "test_rmse" in main_df.columns:
            best = main_df.sort_values("test_rmse").groupby("subset").head(1)
            lines.extend(
                [
                    "### Best RMSE by subset",
                    "",
                    _compact_markdown(best, ["subset", "model", "test_rmse", "test_mae", "test_nasa_score"]),
                    "",
                ]
            )

    complexity_df = _read_csv_if_exists(tables_dir / "table_complexity.csv")
    if complexity_df is not None:
        lines.extend(["## Complexity", "", complexity_df.to_markdown(index=False), ""])

    critical_df = _read_csv_if_exists(tables_dir / "table_critical_zone.csv")
    if critical_df is not None:
        lines.extend(
            [
                "## Critical-Zone Late-Prediction Risk",
                "",
                _compact_markdown(
                    critical_df,
                    [
                        "subset",
                        "model",
                        "test_critical_30_rmse",
                        "test_critical_30_late_prediction_ratio",
                        "test_critical_50_rmse",
                        "test_critical_50_late_prediction_ratio",
                    ],
                ),
                "",
            ]
        )

    ablation_df = _read_csv_if_exists(tables_dir / "table_ablation.csv")
    if ablation_df is not None:
        lines.extend(
            [
                "## Ablation Evidence",
                "",
                _compact_markdown(
                    ablation_df,
                    [
                        "subset",
                        "model",
                        "ablation_variant",
                        "test_rmse",
                        "test_mae",
                        "test_nasa_score",
                        "test_critical_30_late_prediction_ratio",
                    ],
                ),
                "",
            ]
        )

    robustness_df = _read_csv_if_exists(tables_dir / "table_robustness.csv")
    if robustness_df is not None:
        stress = robustness_df.copy()
        if "model" in stress.columns:
            stress = stress[stress["model"] == "rast_gru_v2"]
        if "scenario" in stress.columns:
            stress = stress[stress["scenario"] != "clean"]
        lines.extend(
            [
                "## Robustness Stress Cases",
                "",
                _compact_markdown(
                    stress if not stress.empty else robustness_df,
                    [
                        "subset",
                        "model",
                        "scenario",
                        "level",
                        "rmse",
                        "relative_rmse_increase",
                        "nasa_score",
                        "relative_nasa_score_increase",
                        "critical_30_late_prediction_ratio",
                    ],
                    sort_by="relative_rmse_increase",
                    ascending=False,
                    max_rows=12,
                ),
                "",
            ]
        )

    stats_df = _read_csv_if_exists(tables_dir / "table_statistical_tests.csv")
    if stats_df is not None:
        lines.extend(
            [
                "## Statistical Tests",
                "",
                _compact_markdown(
                    stats_df,
                    [
                        "subset",
                        "baseline_model",
                        "proposed_model",
                        "metric",
                        "p_value",
                        "effect_size",
                        "mean_baseline_metric",
                        "mean_proposed_metric",
                    ],
                    sort_by="p_value",
                    ascending=True,
                    max_rows=20,
                ),
                "",
            ]
        )

    multiseed_df = _read_csv_if_exists(tables_dir / "table_multiseed_summary.csv")
    if multiseed_df is not None:
        lines.extend(
            [
                "## Multi-Seed Evidence",
                "",
                _compact_markdown(
                    multiseed_df,
                    [
                        "subset",
                        "model",
                        "model_display",
                        "seed_count",
                        "seeds",
                        "test_rmse_mean",
                        "test_rmse_std",
                        "test_mae_mean",
                        "test_mae_std",
                        "test_nasa_score_mean",
                        "test_nasa_score_std",
                        "test_critical_30_late_prediction_ratio_mean",
                        "test_critical_30_late_prediction_ratio_std",
                    ],
                ),
                "",
            ]
        )

    multiseed_overall_df = _read_csv_if_exists(tables_dir / "table_multiseed_overall.csv")
    if multiseed_overall_df is not None:
        lines.extend(
            [
                "## Multi-Seed Overall Evidence",
                "",
                _compact_markdown(
                    multiseed_overall_df,
                    [
                        "model",
                        "model_display",
                        "subset_count",
                        "seed_count",
                        "seeds",
                        "test_rmse_mean",
                        "test_rmse_std",
                        "test_mae_mean",
                        "test_mae_std",
                        "test_nasa_score_mean",
                        "test_nasa_score_std",
                        "test_critical_30_late_prediction_ratio_mean",
                        "test_critical_30_late_prediction_ratio_std",
                    ],
                    sort_by="test_rmse_mean",
                ),
                "",
            ]
        )

    severe_probe_df = _read_csv_if_exists(tables_dir / "table_severe_missing_probe.csv")
    if severe_probe_df is not None:
        lines.extend(
            [
                "## Severe Missing Draft Probe",
                "",
                "This probe is not part of the formal main-result table because it is marked as draft and allowed_for_paper=false.",
                "",
                _compact_markdown(
                    severe_probe_df,
                    [
                        "experiment_name",
                        "subset",
                        "model",
                        "model_display",
                        "result_status",
                        "allowed_for_paper",
                        "test_rmse",
                        "test_mae",
                        "test_nasa_score",
                        "test_critical_30_late_prediction_ratio",
                    ],
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Claim Boundary",
            "",
            "- Recommended main line: a reproducible, risk-aware RUL prediction framework under complex operating conditions and controlled sensor degradation.",
            "- Supported claim: OCM-MST-GRU provides a decision-dependent accuracy-risk-efficiency trade-off across the controlled C-MAPSS tasks; it is not uniformly best in RMSE and does not establish field safety or robustness.",
            "- Do not claim that OCM-MST-GRU is best on every subset, every metric, or FD004 clean-data RMSE.",
            "- Do not claim universal robustness to all missing-sensor scenarios; severe random/global missing remains a limitation unless an additional mitigation experiment is completed.",
            "- FD004 local data note: in the validated local C-MAPSS files, FD004 contains 249 training engines and 248 test engines, with 248 aligned RUL rows.",
            "",
            "## Writing Notes",
            "",
            "- Current formal C-MAPSS claim level is five-composite-seed evidence across FD001--FD004; the fixed N-CMAPSS task and controlled stress tests have narrower external-validity boundaries.",
            "- Treat baselines as comparison methods, not contribution.",
            "- Emphasize the reproducible protocol, data validation, critical-zone maintenance-risk proxies, ablation, and controlled degradation sensitivity.",
            "- Treat severe missing results as a stress-test limitation unless an additional mitigation experiment is added.",
            "- Discuss FD004 separately because it combines multiple operating conditions and multiple fault modes.",
            "- Avoid leaderboard-style wording; use evidence-backed phrasing such as competitive, lowers selected late-risk proxies, or exhibits a decision-dependent trade-off under declared perturbations.",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Write a markdown result brief from paper tables.")
    parser.add_argument("--tables-dir", default=str(PROJECT_ROOT / "paper_outputs" / "tables"))
    parser.add_argument("--output", default=str(PROJECT_ROOT / "paper_outputs" / "result_brief.md"))
    args = parser.parse_args()

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(build_result_brief(Path(args.tables_dir)), encoding="utf-8")
    print(f"Saved result brief to {output}")


if __name__ == "__main__":
    main()
