from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from build_manuscript_evidence import write_booktabs_table
from check_extended_review_evidence import check_extended_review_evidence


DEFAULT_SOURCE = PROJECT_ROOT / "paper_outputs" / "advanced_evidence"
DEFAULT_OUTPUT = PROJECT_ROOT / "paper_outputs" / "manuscript_v3"


POINT_LABELS = {"RAST": "RAST-GRU", "Core": "OCM-Core", "Asym": "OCM-Asym"}
CONDITION_LABELS = {
    "p0": "$p=0$",
    "p025": "$p=0.25$",
    "p05": "$p=0.5$",
    "p1": "$p=1$ (locked)",
    "mix50": "50/50 mixture",
}
METRIC_LABELS = {
    "rmse": "RMSE",
    "nasa_per_engine": "NASA/engine",
    "lpr30": "LPR@30",
    "late_cvar95_30": "Late CVaR95",
    "unit_macro_rmse": "Unit RMSE",
    "unit_macro_nasa_per_window": "NASA/window",
    "unit_macro_lpr30": "Unit LPR@30",
}


def mean_sd(mean: float, sd: float, digits: int = 3) -> str:
    return f"{float(mean):.{digits}f} $\\pm$ {float(sd):.{digits}f}"


def effect_ci(row: pd.Series, low: str, high: str, effect: str) -> str:
    return (
        f"{float(row[effect]):.3f} "
        f"[{float(row[low]):.3f}, {float(row[high]):.3f}]"
    )


def interval_counts(frame: pd.DataFrame, low: str, high: str) -> tuple[int, int, int]:
    lower = int((frame[high] < 0.0).sum())
    higher = int((frame[low] > 0.0).sum())
    unresolved = int(len(frame) - lower - higher)
    return lower, higher, unresolved


def write_augmentation_tables(source: Path, output: Path) -> dict[str, int]:
    summary = pd.read_csv(source / "augmentation_distribution_summary.csv")
    intervals = pd.read_csv(source / "augmentation_distribution_paired_intervals.csv")
    display = summary.copy()
    display["point_display"] = display["point"].map(POINT_LABELS)
    display["condition_display"] = display["condition"].map(CONDITION_LABELS)
    for metric in (
        "clean_rmse",
        "clean_lpr30",
        "clean_critical30_signed_bias",
        "stress_rmse_nine_family_mean",
        "stress_lpr30_nine_family_mean",
    ):
        display[f"{metric}_mean_sd"] = display.apply(
            lambda row, metric=metric: mean_sd(row[f"{metric}_mean"], row[f"{metric}_sd"]),
            axis=1,
        )
    write_booktabs_table(
        display,
        output / "table_augmentation_distribution_summary.tex",
        columns=[
            ("point_display", "Operating point", "raw"),
            ("condition_display", "Training distribution", "raw"),
            ("clean_rmse_mean_sd", "Clean RMSE", "raw"),
            ("clean_lpr30_mean_sd", "Clean LPR", "raw"),
            ("clean_critical30_signed_bias_mean_sd", "Critical bias", "raw"),
            ("stress_rmse_nine_family_mean_mean_sd", "Stress RMSE", "raw"),
            ("stress_lpr30_nine_family_mean_mean_sd", "Stress LPR", "raw"),
        ],
        caption="FD004 sensitivity to the training augmentation distribution.",
        label="tab:v3-augmentation-distribution",
        note="Values are mean $\\pm$ sample SD over five matched training seeds. Stress columns average normalized curve means over nine controlled algorithmic perturbation families and five perturbation seeds. They condition on the fixed FD004 test engines and declared perturbation streams. At $p=0$, study augmentation is disabled while the OCM-specific consistency view remains fixed. The study-reference $p=1$ composite distribution is not an assumed optimum.",
        font_command=r"\scriptsize",
        fit_width=True,
    )
    write_booktabs_table(
        display,
        output / "table_augmentation_distribution_summary_zh.tex",
        columns=[
            ("point_display", "运行点", "raw"),
            ("condition_display", "训练分布", "raw"),
            ("clean_rmse_mean_sd", "干净 RMSE", "raw"),
            ("clean_lpr30_mean_sd", "干净 LPR", "raw"),
            ("clean_critical30_signed_bias_mean_sd", "临界区偏差", "raw"),
            ("stress_rmse_nine_family_mean_mean_sd", "压力 RMSE", "raw"),
            ("stress_lpr30_nine_family_mean_mean_sd", "压力 LPR", "raw"),
        ],
        caption="FD004 对训练增强分布的敏感性。",
        label="tab:v3-augmentation-distribution-zh",
        note="数值为五个匹配训练种子的均值 $\\pm$ 样本标准差。压力列对九类受控故障的归一化曲线均值及五个扰动种子取平均，并以固定 FD004 测试发动机和声明的扰动流为条件。$p=0$ 关闭主退化增强，但保持 OCM 特有的一致性视图不变。锁定的 $p=1$ 复合分布只是参照，不预设其最优。",
        font_command=r"\scriptsize",
        fit_width=True,
    )

    selected = intervals[
        intervals["metric"].isin(["clean_rmse", "stress_lpr30_nine_family_mean"])
        & (intervals["condition"] != "p1")
    ].copy()
    selected["point_display"] = selected["point"].map(POINT_LABELS)
    selected["condition_display"] = selected["condition"].map(CONDITION_LABELS)
    selected["metric_display"] = selected["metric"].map(
        {"clean_rmse": "Clean RMSE", "stress_lpr30_nine_family_mean": "Stress LPR curve mean"}
    )
    selected["effect_ci"] = selected.apply(
        lambda row: effect_ci(
            row,
            "paired_seed_bootstrap_ci95_low",
            "paired_seed_bootstrap_ci95_high",
            "mean_difference_condition_minus_p1",
        ),
        axis=1,
    )
    write_booktabs_table(
        selected,
        output / "table_augmentation_distribution_intervals.tex",
        columns=[
            ("point_display", "Point", "raw"),
            ("condition_display", "Condition", "raw"),
            ("metric_display", "Metric", "text"),
            ("effect_ci", "Condition$-p=1$ [95\\% CI]", "raw"),
            ("probability_condition_lower", "Pr(condition lower)", ".3f"),
        ],
        caption="Paired-seed augmentation-distribution contrasts against the locked composite distribution.",
        label="tab:v3-augmentation-distribution-ci",
        note="Diagnostic intervals resample the five matched training seeds and condition on the fixed FD004 test engines and five declared perturbation streams. Negative effects favor the named alternative for lower-is-better metrics. These retrospective contrasts do not select or validate the researcher-specified illustrative preference.",
        font_command=r"\scriptsize",
        fit_width=True,
    )
    write_booktabs_table(
        selected,
        output / "table_augmentation_distribution_intervals_zh.tex",
        columns=[
            ("point_display", "运行点", "raw"),
            ("condition_display", "条件", "raw"),
            ("metric_display", "指标", "text"),
            ("effect_ci", "条件减 $p=1$ [95\\% CI]", "raw"),
            ("probability_condition_lower", "条件更低概率", ".3f"),
        ],
        caption="相对锁定复合增强分布的配对种子敏感性对照。",
        label="tab:v3-augmentation-distribution-ci-zh",
        note="区间对五个匹配训练种子重采样，并以固定 FD004 测试发动机和五条声明扰动流为条件。对越低越好的指标，负值支持相应替代条件；这些锁定后敏感性结果不用于重新选择主运行点。",
        font_command=r"\scriptsize",
        fit_width=True,
    )
    counts: dict[str, int] = {}
    for metric, prefix in (
        ("clean_rmse", "aug_clean_rmse"),
        ("stress_lpr30_nine_family_mean", "aug_stress_lpr"),
    ):
        lower, higher, unresolved = interval_counts(
            selected[selected["metric"] == metric],
            "paired_seed_bootstrap_ci95_low",
            "paired_seed_bootstrap_ci95_high",
        )
        counts[f"{prefix}_alternative_lower"] = lower
        counts[f"{prefix}_p1_lower"] = higher
        counts[f"{prefix}_unresolved"] = unresolved
    return counts


def write_factorial_tables(source: Path, output: Path) -> dict[str, int]:
    decomposition = pd.read_csv(source / "seed_factorial_fd004_decomposition.csv")
    intervals = pd.read_csv(source / "seed_factorial_fd004_paired_intervals.csv")
    display = decomposition.copy()
    display["point_display"] = display["point"].map(POINT_LABELS)
    display["metric_display"] = display["metric"].map(METRIC_LABELS)
    write_booktabs_table(
        display,
        output / "table_seed_factorial_decomposition.tex",
        columns=[
            ("point_display", "Point", "raw"),
            ("metric_display", "Metric", "text"),
            ("grand_mean", "Grand mean", ".3f"),
            ("total_cell_sd", "Cell SD", ".3f"),
            ("split_mean_sd", "Split SD", ".3f"),
            ("training_mean_sd", "Training SD", ".3f"),
            ("nonadditive_residual_rms", "Residual RMS", ".3f"),
        ],
        caption="Descriptive 3-by-3 split-seed by training-seed variance-source audit on FD004.",
        label="tab:v3-seed-factorial-decomposition",
        note="The training seed jointly controls initialization, shuffling, and augmentation and is not a pure initialization factor. SD columns summarize three factor-level means; residual RMS records nonadditivity. This small fixed-level design is descriptive rather than a random-effects population model.",
        font_command=r"\scriptsize",
        fit_width=True,
    )
    write_booktabs_table(
        display,
        output / "table_seed_factorial_decomposition_zh.tex",
        columns=[
            ("point_display", "运行点", "raw"),
            ("metric_display", "指标", "text"),
            ("grand_mean", "总均值", ".3f"),
            ("total_cell_sd", "单元 SD", ".3f"),
            ("split_mean_sd", "划分 SD", ".3f"),
            ("training_mean_sd", "训练 SD", ".3f"),
            ("nonadditive_residual_rms", "非加性 RMS", ".3f"),
        ],
        caption="FD004 上 $3\\times3$ 划分种子与训练种子的方差来源审计。",
        label="tab:v3-seed-factorial-decomposition-zh",
        note="训练种子同时控制初始化、打乱和增强流，不能解释为纯初始化因子。SD 列概括三个因子水平的均值，残差 RMS 反映非加性；该固定水平小网格只作描述性审计。",
        font_command=r"\scriptsize",
        fit_width=True,
    )
    paired = intervals.copy()
    paired["comparison"] = paired.apply(
        lambda row: f"{POINT_LABELS[row['left_point']]} vs {POINT_LABELS[row['right_point']]}", axis=1
    )
    paired["metric_display"] = paired["metric"].map(METRIC_LABELS)
    paired["effect_ci"] = paired.apply(
        lambda row: effect_ci(
            row,
            "crossed_factor_bootstrap_ci95_low",
            "crossed_factor_bootstrap_ci95_high",
            "mean_difference_left_minus_right",
        ),
        axis=1,
    )
    for suffix, caption, note, headers in (
        (
            "",
            "Crossed-factor paired contrasts over the FD004 3-by-3 seed grid.",
            "Effects are left minus right; negative values favor the left point. Bootstrap draws resample split and training-factor levels. The nine cells do not identify pure initialization effects.",
            ("Comparison", "Metric", "Left$-$right [95\\% CI]", "Pr(left lower)"),
        ),
        (
            "_zh",
            "FD004 的 $3\\times3$ 种子网格交叉因子配对对照。",
            "效应为左减右，负值支持左侧运行点；bootstrap 分别重采样划分和训练因子水平。九个单元不能识别纯初始化效应。",
            ("比较", "指标", "左减右 [95\\% CI]", "左侧更低概率"),
        ),
    ):
        write_booktabs_table(
            paired,
            output / f"table_seed_factorial_intervals{suffix}.tex",
            columns=[
                ("comparison", headers[0], "text"),
                ("metric_display", headers[1], "text"),
                ("effect_ci", headers[2], "raw"),
                ("probability_left_lower", headers[3], ".3f"),
            ],
            caption=caption,
            label=f"tab:v3-seed-factorial-ci{suffix}",
            note=note,
            font_command=r"\scriptsize",
            fit_width=True,
        )
    return {
        "factor_split_larger": int((decomposition["split_mean_sd"] > decomposition["training_mean_sd"]).sum()),
        "factor_training_larger": int((decomposition["training_mean_sd"] > decomposition["split_mean_sd"]).sum()),
    }


def write_rotation_tables(source: Path, output: Path) -> dict[str, int]:
    summary = pd.read_csv(source / "ncmapss_dev_rotation_summary.csv")
    intervals = pd.read_csv(source / "ncmapss_dev_rotation_paired_intervals.csv")
    display = summary.copy()
    display["point_display"] = display["point"].map(POINT_LABELS)
    for metric in ("unit_macro_rmse", "unit_macro_nasa_per_window", "unit_macro_lpr30"):
        display[f"{metric}_mean_sd"] = display.apply(
            lambda row, metric=metric: mean_sd(row[f"{metric}_mean"], row[f"{metric}_sd"]), axis=1
        )
    for suffix, caption, note, headers in (
        (
            "",
            "N-CMAPSS DS02 development-unit rotation across six held-out units and three training seeds.",
            "Values are mean $\\pm$ sample SD over 18 unit-seed cells. Official test units 11, 14, and 15 are never used in this rotation. The result is within-task split sensitivity, not independent field validation.",
            ("Point", "Units", "Seeds", "Unit RMSE", "NASA/window", "Unit LPR"),
        ),
        (
            "_zh",
            "N-CMAPSS DS02 六个开发单元轮换与三个训练种子结果。",
            "数值为 18 个单元--种子单元的均值 $\\pm$ 样本标准差。官方测试单元 11、14、15 不参与该轮换；结果反映任务内划分敏感性，不是独立现场验证。",
            ("运行点", "单元", "种子", "单元 RMSE", "NASA/窗口", "单元 LPR"),
        ),
    ):
        write_booktabs_table(
            display,
            output / f"table_ncmapss_dev_rotation{suffix}.tex",
            columns=[
                ("point_display", headers[0], "raw"),
                ("unit_count", headers[1], "int"),
                ("seed_count", headers[2], "int"),
                ("unit_macro_rmse_mean_sd", headers[3], "raw"),
                ("unit_macro_nasa_per_window_mean_sd", headers[4], "raw"),
                ("unit_macro_lpr30_mean_sd", headers[5], "raw"),
            ],
            caption=caption,
            label=f"tab:v3-ncmapss-dev-rotation{suffix}",
            note=note,
            font_command=r"\scriptsize",
            fit_width=True,
        )
    paired = intervals.copy()
    paired["comparison"] = paired.apply(
        lambda row: f"{POINT_LABELS[row['left_point']]} vs {POINT_LABELS[row['right_point']]}", axis=1
    )
    paired["metric_display"] = paired["metric"].map(METRIC_LABELS)
    paired["effect_ci"] = paired.apply(
        lambda row: effect_ci(
            row,
            "unit_seed_bootstrap_ci95_low",
            "unit_seed_bootstrap_ci95_high",
            "mean_difference_left_minus_right",
        ),
        axis=1,
    )
    for suffix, caption, note, headers in (
        (
            "",
            "Crossed unit-by-seed intervals for the N-CMAPSS development-unit rotation.",
            "Effects are left minus right; negative values favor the left point. The bootstrap resamples held-out development units and training seeds. It does not extend inference beyond the declared DS02 task.",
            ("Comparison", "Metric", "Left$-$right [95\\% CI]", "Pr(left lower)"),
        ),
        (
            "_zh",
            "N-CMAPSS 开发单元轮换的单元--种子交叉区间。",
            "效应为左减右，负值支持左侧运行点；bootstrap 重采样留出开发单元和训练种子，推断范围不超出声明的 DS02 任务。",
            ("比较", "指标", "左减右 [95\\% CI]", "左侧更低概率"),
        ),
    ):
        write_booktabs_table(
            paired,
            output / f"table_ncmapss_dev_rotation_intervals{suffix}.tex",
            columns=[
                ("comparison", headers[0], "text"),
                ("metric_display", headers[1], "text"),
                ("effect_ci", headers[2], "raw"),
                ("probability_left_lower", headers[3], ".3f"),
            ],
            caption=caption,
            label=f"tab:v3-ncmapss-dev-rotation-ci{suffix}",
            note=note,
            font_command=r"\scriptsize",
            fit_width=True,
        )
    asym_rast = intervals[(intervals["left_point"] == "Asym") & (intervals["right_point"] == "RAST")]
    core_rast = intervals[(intervals["left_point"] == "Core") & (intervals["right_point"] == "RAST")]
    a_lower, a_higher, a_unresolved = interval_counts(
        asym_rast, "unit_seed_bootstrap_ci95_low", "unit_seed_bootstrap_ci95_high"
    )
    c_lower, c_higher, c_unresolved = interval_counts(
        core_rast, "unit_seed_bootstrap_ci95_low", "unit_seed_bootstrap_ci95_high"
    )
    return {
        "rotation_asym_lower": a_lower,
        "rotation_asym_higher": a_higher,
        "rotation_asym_unresolved": a_unresolved,
        "rotation_core_lower": c_lower,
        "rotation_core_higher": c_higher,
        "rotation_core_unresolved": c_unresolved,
    }


def write_result_fragments(output: Path, values: dict[str, int]) -> None:
    english = f"""
\\paragraph{{Retrospective distribution and randomness audits.}}
Across the 12 non-reference augmentation contrasts, paired intervals support a lower clean RMSE for an alternative to the locked $p=1$ distribution in {values['aug_clean_rmse_alternative_lower']} cases, support $p=1$ in {values['aug_clean_rmse_p1_lower']}, and leave {values['aug_clean_rmse_unresolved']} unresolved. For the nine-family LPR curve mean, the corresponding counts are {values['aug_stress_lpr_alternative_lower']}, {values['aug_stress_lpr_p1_lower']}, and {values['aug_stress_lpr_unresolved']}. The locked composite augmentation is therefore treated as a study-defined operating distribution rather than a generally optimal corruption model. In the FD004 $3\\times3$ audit, split-level SD exceeds training-randomness SD in {values['factor_split_larger']} of 12 point--metric cells, while the reverse occurs in {values['factor_training_larger']}; the five composite seeds cannot be reduced to a pure initialization interpretation. In the N-CMAPSS development-unit rotation, Asym-versus-RAST intervals support lower Asym in {values['rotation_asym_lower']} of three metrics, support lower RAST in {values['rotation_asym_higher']}, and leave {values['rotation_asym_unresolved']} unresolved; this remains within-task exploratory evidence.
""".lstrip()
    chinese = f"""
\\paragraph{{锁定后的增强分布与随机性审计。}}
在 12 个相对锁定 $p=1$ 的非参照增强对照中，配对区间有 {values['aug_clean_rmse_alternative_lower']} 个支持替代条件具有更低的干净 RMSE，{values['aug_clean_rmse_p1_lower']} 个支持 $p=1$，其余 {values['aug_clean_rmse_unresolved']} 个未解决；九故障族 LPR 曲线均值的对应计数为 {values['aug_stress_lpr_alternative_lower']}、{values['aug_stress_lpr_p1_lower']} 和 {values['aug_stress_lpr_unresolved']}。因此锁定的复合增强只被视为本文定义的训练分布，而不是普遍最优的退化模型。FD004 的 $3\\times3$ 审计中，12 个“运行点--指标”单元有 {values['factor_split_larger']} 个的划分层 SD 大于训练随机性层 SD，反向情形有 {values['factor_training_larger']} 个，故五个复合种子不能简化解释为纯初始化效应。N-CMAPSS 开发单元轮换中，Asym 相对 RAST 的三个指标有 {values['rotation_asym_lower']} 个区间支持 Asym 更低、{values['rotation_asym_higher']} 个支持 RAST 更低、{values['rotation_asym_unresolved']} 个未解决；该结果仍只属于任务内探索性证据。
""".lstrip()
    (output / "extended_review_results_en.tex").write_text(english, encoding="utf-8", newline="\n")
    (output / "extended_review_results_zh.tex").write_text(chinese, encoding="utf-8", newline="\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build LaTeX tables for the final reviewer-requested experiments.")
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--copy-to", action="append", default=[])
    args = parser.parse_args()

    errors = check_extended_review_evidence(PROJECT_ROOT)
    if errors:
        raise RuntimeError("Extended evidence is incomplete:\n- " + "\n- ".join(errors))
    source = Path(args.source_dir)
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    values: dict[str, int] = {}
    values.update(write_augmentation_tables(source, output))
    values.update(write_factorial_tables(source, output))
    values.update(write_rotation_tables(source, output))
    write_result_fragments(output, values)
    for raw in args.copy_to:
        destination = Path(raw)
        destination.mkdir(parents=True, exist_ok=True)
        for path in output.glob("*extended_review*.tex"):
            shutil.copy2(path, destination / path.name)
        for path in output.glob("table_augmentation_distribution*.tex"):
            shutil.copy2(path, destination / path.name)
        for path in output.glob("table_seed_factorial*.tex"):
            shutil.copy2(path, destination / path.name)
        for path in output.glob("table_ncmapss_dev_rotation*.tex"):
            shutil.copy2(path, destination / path.name)
    print(f"EXTENDED_REVIEW_TABLES_READY {output}")


if __name__ == "__main__":
    main()
