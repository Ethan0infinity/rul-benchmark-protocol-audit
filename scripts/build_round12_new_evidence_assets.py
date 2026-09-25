from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE = PROJECT_ROOT / "paper_outputs" / "round12_new_evidence"
TABLE_OUTPUT = PROJECT_ROOT / "paper_outputs" / "manuscript_v3"
FIGURE_OUTPUT = SOURCE / "figures"
MPL_CACHE = PROJECT_ROOT / ".mpl_cache"
os.environ.setdefault("MPLCONFIGDIR", str(MPL_CACHE))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, Rectangle


METRIC_LABELS = {
    "test_rmse": "RMSE",
    "test_nasa_per_engine": "NASA/engine",
    "test_critical_30_late_prediction_ratio": "LPR@30",
    "rmse": "RMSE",
    "nasa_per_engine": "NASA/engine",
    "critical_30_late_prediction_ratio": "LPR@30",
}


def require_csv(name: str) -> pd.DataFrame:
    path = SOURCE / name
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def tex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
    }
    return "".join(replacements.get(character, character) for character in text)


def write_table(
    path: Path,
    *,
    caption: str,
    label: str,
    headers: list[str],
    rows: list[list[object]],
    align: str,
    resize: bool = False,
) -> None:
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\small",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        *([r"\resizebox{\textwidth}{!}{%"] if resize else []),
        rf"\begin{{tabular}}{{{align}}}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    lines.extend(
        " & ".join(tex_escape(value) for value in row) + r" \\" for row in rows
    )
    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            *([r"}"] if resize else []),
            r"\end{table*}",
            "",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def build_independent_table() -> None:
    frame = require_csv("independent_stream_paired_summary.csv")
    rows = []
    for item in frame.to_dict("records"):
        rows.append(
            [
                item["subset"],
                METRIC_LABELS.get(item["metric"], item["metric"]),
                int(item["stream_count"]),
                f"{item['asym_minus_rast_mean']:.4f} "
                f"({item['asym_minus_rast_sd']:.4f})",
                f"{int(item['negative_count'])}/"
                f"{int(item['positive_count'])}/"
                f"{int(item['zero_count'])}",
            ]
        )
    write_table(
        TABLE_OUTPUT / "table_round12_independent_streams.tex",
        caption=(
            "Fixed-split paired estimates across ten separately seeded training "
            "streams. Differences are OCM-Asym minus RAST-GRU; the final "
            "column reports negative/positive/zero stream counts. These are "
            "retrospective descriptive estimates."
        ),
        label="tab:round12-independent",
        headers=[
            "Task",
            "Metric",
            "Streams",
            r"Mean difference (SD)",
            r"$-/+ /0$",
        ],
        rows=rows,
        align="llccc",
        resize=True,
    )
    write_table(
        TABLE_OUTPUT / "table_round12_independent_streams_zh.tex",
        caption=(
            "固定数据划分下十个分别设定随机种子的训练流配对估计。差值为 OCM-Asym "
            "减 RAST-GRU；末列依次为负、正和零差值的训练流数量。该分析为"
            "回顾性描述估计。"
        ),
        label="tab:round12-independent-zh",
        headers=["任务", "指标", "训练流", "平均差值（标准差）", "负/正/零"],
        rows=rows,
        align="llccc",
        resize=True,
    )


def build_crossed_table() -> None:
    frame = require_csv("crossed_split_stream_summary.csv")
    rows = []
    for item in frame.to_dict("records"):
        rows.append(
            [
                item["subset"],
                METRIC_LABELS.get(item["metric"], item["metric"]),
                int(item["cell_count"]),
                f"{item['asym_minus_rast_mean']:.4f} ({item['asym_minus_rast_sd']:.4f})",
                f"{int(item['negative_count'])}/{int(item['positive_count'])}/"
                f"{int(item['zero_count'])}",
                f"{item['finite_split_rms']:.4f}",
                f"{item['finite_stream_rms']:.4f}",
                f"{item['finite_additive_residual_rms']:.4f}",
            ]
        )
    write_table(
        TABLE_OUTPUT / "table_round12_crossed_split_stream.tex",
        caption=(
            "Result-informed finite-design audit over all combinations of three selected engine "
            "splits and three selected training streams. Differences are OCM-Asym "
            "minus RAST-GRU. Split, "
            "stream, and additive-residual RMS values are arithmetic summaries "
            "of the nine completed cells. With one observation per cell, the "
            "residual cannot be separated from optimization noise and is not a "
            "population variance component; the full 5-by-10 direct grid was not run."
        ),
        label="tab:round12-crossed",
        headers=[
            "Task",
            "Metric",
            "Cells",
            "Mean difference (SD)",
            r"$-/+/0$",
            "Split RMS",
            "Stream RMS",
            "Cell-residual RMS",
        ],
        rows=rows,
        align="llcccccc",
        resize=True,
    )
    write_table(
        TABLE_OUTPUT / "table_round12_crossed_split_stream_zh.tex",
        caption=(
            "结果知情的三个指定发动机划分与三条指定训练流全部组合的有限设计审计。差值为 OCM-Asym 减 RAST-GRU。"
            "划分 RMS、训练流 RMS 和加性残差 RMS 只描述九个已完成单元；每格仅有一个观测，"
            "残差无法与优化噪声分离，也不是总体方差分量；完整 5×10 直接网格未运行。"
        ),
        label="tab:round12-crossed-zh",
        headers=["任务", "指标", "单元", "平均差值（标准差）", "负/正/零", "划分 RMS", "训练流 RMS", "单元残差 RMS"],
        rows=rows,
        align="llcccccc",
        resize=True,
    )


def build_crossed_selection_tables() -> None:
    frame = require_csv("crossed_level_subset_sensitivity.csv")
    frame = frame[frame["metric"] == "test_critical_30_late_prediction_ratio"]
    labels = {
        "fixed_split_42_ten_streams": "10 fixed-split streams",
        "five_coupled_composite_levels": "5 coupled composite levels",
    }
    rows = []
    for item in frame.sort_values(["source_design", "subset"]).to_dict("records"):
        rows.append(
            [
                labels[item["source_design"]],
                item["subset"],
                int(item["combination_count"]),
                f"[{item['mean_effect_min']:.4f}, {item['mean_effect_max']:.4f}]",
                f"{int(item['negative_count_min'])}--{int(item['negative_count_max'])}/3",
                f"{item['reported_subset_mean']:.4f}",
            ]
        )
    caption = (
        "Alternative three-level subset sensitivity for LPR@30. The fixed-split "
        "analysis enumerates all 120 three-of-ten training-stream subsets. The "
        "composite analysis enumerates all ten three-of-five subsets, but each "
        "composite level jointly changes split and training randomness. Ranges "
        "are finite-set arithmetic, not uncertainty intervals."
    )
    write_table(
        TABLE_OUTPUT / "table_round16_crossed_level_selection.tex",
        caption=caption,
        label="tab:round16-crossed-level-selection",
        headers=["Available design", "Task", "Subsets", "Mean-effect range", "Negative-count range", "Reported triple"],
        rows=rows,
        align="llcccc",
        resize=True,
    )
    write_table(
        TABLE_OUTPUT / "table_round16_crossed_level_selection_zh.tex",
        caption=(
            "LPR@30 的三水平替代子集敏感性。固定划分分析枚举十条训练流中全部 120 个三流子集；"
            "复合水平分析枚举五个复合水平中的全部十个三水平子集，但每个复合水平同时改变划分与训练随机性。"
            "所列范围只是有限集合算术，不是不确定性区间。"
        ),
        label="tab:round16-crossed-level-selection-zh",
        headers=["可用设计", "任务", "子集数", "均值范围", "负方向数范围", "已报告三水平"],
        rows=rows,
        align="llcccc",
        resize=True,
    )


def build_preprocessing_resource_table() -> None:
    frame = require_csv("cmapss_preprocessing_resource_summary.csv")
    rows = []
    for item in frame.sort_values(["audit_axis", "audit_level", "point"]).to_dict("records"):
        setting = (
            "W30/B14"
            if item["audit_axis"] == "reference"
            else (
                f"W{int(item['window_size'])}"
                if item["audit_axis"] == "window"
                else f"B{int(item['sensor_budget'])}"
            )
        )
        rows.append(
            [
                setting,
                item["point"],
                int(item["parameters"]),
                f"{item['train_windows_mean']:.0f}",
                f"{item['completed_epochs_mean']:.1f}",
                f"{item['estimated_optimization_steps_mean']:.0f}",
                f"{item['wall_clock_seconds_mean'] / 60.0:.1f}",
            ]
        )
    write_table(
        TABLE_OUTPUT / "table_round16_preprocessing_resources.tex",
        caption=(
            "Training-resource ledger for the FD004 three-seed OFAT audit. "
            "Optimizer steps are reconstructed as completed epochs times the "
            "ceiling of training windows divided by batch size 128; wall time is "
            "descriptive and hardware-specific."
        ),
        label="tab:round16-preprocessing-resources",
        headers=["Setting", "Point", "Parameters", "Train windows", "Epochs", "Estimated steps", "Minutes"],
        rows=rows,
        align="llrrrrr",
        resize=True,
    )
    write_table(
        TABLE_OUTPUT / "table_round16_preprocessing_resources_zh.tex",
        caption=(
            "FD004 三种子单因素预处理审计的训练资源台账。优化步数按完成 epoch 数乘以训练窗口数除以批量 128 的向上取整重建；"
            "运行时间仅为本机描述值。"
        ),
        label="tab:round16-preprocessing-resources-zh",
        headers=["设置", "配置", "参数量", "训练窗口", "Epoch", "估计步数", "分钟"],
        rows=rows,
        align="llrrrrr",
        resize=True,
    )
def build_preprocessing_table() -> None:
    frame = require_csv("cmapss_preprocessing_paired_summary.csv")
    rows = []
    for item in frame.to_dict("records"):
        setting = (
            "W=30, B=14"
            if item["audit_axis"] == "reference"
            else (
                f"W={int(item['window_size'])}"
                if item["audit_axis"] == "window"
                else f"B={int(item['sensor_budget'])}"
            )
        )
        rows.append(
            [
                item["subset"],
                setting,
                METRIC_LABELS.get(item["metric"], item["metric"]),
                f"{item['asym_minus_rast_mean']:.4f} ({item['asym_minus_rast_sd']:.4f})",
                f"{int(item['negative_count'])}/{int(item['positive_count'])}/"
                f"{int(item['zero_count'])}",
            ]
        )
    write_table(
        TABLE_OUTPUT / "table_round12_cmapss_preprocessing.tex",
        caption=(
            "One-factor-at-a-time C-MAPSS preprocessing sensitivity on FD004 "
            "under three composite-seed repetitions. $W$ is window length "
            "and $B$ is the training-core-ranked sensor budget. Differences are "
            "OCM-Asym minus RAST-GRU; this retrospective audit does not optimize "
            "either setting."
        ),
        label="tab:round12-cmapss-preprocessing",
        headers=["Task", "Setting", "Metric", "Mean difference (SD)", r"$-/+/0$"],
        rows=rows,
        align="lllcc",
        resize=True,
    )
    write_table(
        TABLE_OUTPUT / "table_round12_cmapss_preprocessing_zh.tex",
        caption=(
            "FD004 上基于三个复合种子重复的 C-MAPSS 单因素预处理敏感性。"
            "$W$ 表示窗口长度，$B$ 表示仅由训练核心排序得到的传感器预算。差值为 "
            "OCM-Asym 减 RAST-GRU；该回顾性审计不用于优化设置。"
        ),
        label="tab:round12-cmapss-preprocessing-zh",
        headers=["任务", "设置", "指标", "平均差值（标准差）", "负/正/零"],
        rows=rows,
        align="lllcc",
        resize=True,
    )
def build_preference_table() -> None:
    frame = require_csv("preference_retraining_candidate_summary.csv")
    frame = frame.sort_values(
        ["selection_count", "mean_validation_rank"],
        ascending=[False, True],
    )
    rows = [
        [
            item["candidate"],
            f"{item['late_life_weight']:.2f}",
            f"{item['late_over_weight']:.2f}",
            f"{int(item['selection_count'])}/{int(item['fitted_cells'])}",
            f"{item['mean_validation_rank']:.2f}",
            f"{item['mean_validation_regret']:.4f}",
        ]
        for item in frame.to_dict("records")
    ]
    write_table(
        TABLE_OUTPUT / "table_round12_preference_retraining.tex",
        caption=(
            "Nine-point Asym preference grid retrained across three engine "
            "splits and three separately seeded training streams. Selection counts "
            "and regret are conditional on this retrospective grid."
        ),
        label="tab:round12-preference",
        headers=[
            "Candidate",
            r"$\lambda_{\mathrm{life}}$",
            r"$\lambda_{\mathrm{late}}$",
            "Selected",
            "Mean rank",
            "Mean regret",
        ],
        rows=rows,
        align="lccccc",
    )
    write_table(
        TABLE_OUTPUT / "table_round12_preference_retraining_zh.tex",
        caption=(
            "Asym 九格偏好参数在三个发动机划分和三个独立训练流上的重新"
            "训练结果。入选次数和 regret 仅描述该回顾性选择操作。"
        ),
        label="tab:round12-preference-zh",
        headers=[
            "候选点",
            r"$\lambda_{\mathrm{life}}$",
            r"$\lambda_{\mathrm{late}}$",
            "入选次数",
            "平均秩",
            "平均 regret",
        ],
        rows=rows,
        align="lccccc",
    )
    downstream = require_csv("preference_retraining_downstream_effects.csv")
    downstream_rows = [
        [
            int(item["split_seed"]),
            int(item["training_stream_seed"]),
            item["candidate"],
            f"{item['test_rmse_selected_asym_minus_rast']:.3f}",
            f"{item['test_nasa_per_engine_selected_asym_minus_rast']:.3f}",
            "{:.3f}".format(
                item[
                    "test_critical_30_late_prediction_ratio_"
                    "selected_asym_minus_rast"
                ]
            ),
        ]
        for item in downstream.to_dict("records")
    ]
    write_table(
        TABLE_OUTPUT / "table_round12_preference_downstream.tex",
        caption=(
            "Downstream paired direction for the validation-selected Asym "
            "candidate in each retrained split-by-stream grid. Differences "
            "are selected Asym minus the matched RAST-GRU reference and are "
            "not selection-adjusted performance estimates."
        ),
        label="tab:round12-preference-downstream",
        headers=[
            "Split",
            "Training stream",
            "Selected candidate",
            r"$\Delta$RMSE",
            r"$\Delta$NASA/engine",
            r"$\Delta$LPR@30",
        ],
        rows=downstream_rows,
        align="rrlrrr",
    )
    write_table(
        TABLE_OUTPUT / "table_round12_preference_downstream_zh.tex",
        caption=(
            "每个重新训练的 split×stream 网格中，验证集入选 Asym 候选的"
            "下游配对方向。差值为入选 Asym 减匹配 RAST-GRU；这些数值没有"
            "进行选择偏差校正。"
        ),
        label="tab:round12-preference-downstream-zh",
        headers=[
            "划分",
            "训练流",
            "入选候选",
            r"$\Delta$RMSE",
            r"$\Delta$NASA/engine",
            r"$\Delta$LPR@30",
        ],
        rows=downstream_rows,
        align="rrlrrr",
    )


def build_factorial_table() -> None:
    frame = require_csv("factorial_effect_summary.csv")
    effect_labels = {
        "backbone_main": "Bundled architecture",
        "risk_main": "Risk",
        "consistency_main": "Consistency",
        "backbone_x_risk": "Bundled architecture x Risk",
        "backbone_x_consistency": "Bundled architecture x Consistency",
        "risk_x_consistency": "Risk x Consistency",
        "backbone_x_risk_x_consistency": "Bundled architecture x Risk x Consistency",
    }
    rows = [
        [
            METRIC_LABELS.get(item["metric"], item["metric"]),
            effect_labels[item["effect"]],
            int(item["stream_count"]),
            f"{item['estimate_mean']:.4f} ({item['estimate_sd']:.4f})",
            f"{int(item['negative_count'])}/{int(item['positive_count'])}/"
            f"{int(item['zero_count'])}",
        ]
        for item in frame.to_dict("records")
    ]
    write_table(
        TABLE_OUTPUT / "table_round12_factorial.tex",
        caption=(
            "Shared-objective 2-by-2-by-2 factorial estimates on "
            "FD004. OCM-specific reliability and quantile objectives are "
            "disabled, so the table identifies only the bundled RAST-side/OCM-side architecture contrast, shared-risk, "
            "and consistency effects within this controlled design."
        ),
        label="tab:round12-factorial",
        headers=[
            "Metric",
            "Factorial effect",
            "Streams",
            "Mean estimate (SD)",
            r"$-/+/0$",
        ],
        rows=rows,
        align="llccc",
        resize=True,
    )
    write_table(
        TABLE_OUTPUT / "table_round12_factorial_zh.tex",
        caption=(
            "FD004 上共享目标 2×2×2 因子估计。OCM 特有的可靠性和"
            "分位数目标在该设计中关闭，因此表中只识别受控设计内 RAST 侧/OCM 侧捆绑架构、"
            "共享风险目标和一致性目标效应。"
        ),
        label="tab:round12-factorial-zh",
        headers=["指标", "因子效应", "训练流", "平均估计（标准差）", "负/正/零"],
        rows=rows,
        align="llccc",
        resize=True,
    )
    design = require_csv("factorial_design_matrix.csv")
    design_rows = [
        [
            item["backbone"],
            int(item["risk_factor"]),
            int(item["consistency_factor"]),
            item["risk_loss"],
            item["consistency_objective"],
            item["backbone_bundled_components"],
        ]
        for item in design.to_dict("records")
    ]
    write_table(
        TABLE_OUTPUT / "table_round12_factorial_design.tex",
        caption=(
            "Cell matrix for the shared-objective bundled-architecture-by-risk-by-consistency "
            "factorial. Binary factors use 0=off and 1=on; main effects are the "
            "mean 1-minus-0 difference over the other factors. OCM-specific "
            "reliability and quantile objectives are off in every cell."
        ),
        label="tab:round12-factorial-design",
        headers=["Architecture package", "Risk", "Consistency", "Point loss", "Consistency path", "Bundled content"],
        rows=design_rows,
        align="lccp{0.18\\textwidth}p{0.19\\textwidth}p{0.24\\textwidth}",
        resize=True,
    )
    write_table(
        TABLE_OUTPUT / "table_round12_factorial_design_zh.tex",
        caption=(
            "共享目标捆绑架构×risk×consistency 因子设计的单元矩阵。二元因子"
            "编码为 0=关闭、1=开启；主效应为对其余因子平均后的 1 减 0 差值。"
            "所有单元均关闭 OCM 特有的可靠性和分位数目标。"
        ),
        label="tab:round12-factorial-design-zh",
        headers=["架构包", "Risk", "一致性", "点损失", "一致性路径", "捆绑内容"],
        rows=design_rows,
        align="lccp{0.18\\textwidth}p{0.19\\textwidth}p{0.24\\textwidth}",
        resize=True,
    )


def build_lofo_table() -> None:
    family = require_csv("lofo_family_effect_summary.csv").sort_values(
        ["held_out_family", "metric"]
    )
    rows = [
        [
            item["held_out_family"],
            METRIC_LABELS.get(item["metric"], item["metric"]),
            int(item["training_stream_count"]),
            f"{item['absolute_auc_asym_minus_rast_mean']:.4f} "
            f"({item['absolute_auc_asym_minus_rast_sd']:.4f})",
            f"{item['degradation_auc_asym_minus_rast_mean']:.4f} "
            f"({item['degradation_auc_asym_minus_rast_sd']:.4f})",
            f"{int(item['absolute_auc_asym_minus_rast_negative_count'])}/"
            f"{int(item['absolute_auc_asym_minus_rast_positive_count'])}/"
            f"{int(item['absolute_auc_asym_minus_rast_zero_count'])}",
            f"{int(item['degradation_auc_asym_minus_rast_negative_count'])}/"
            f"{int(item['degradation_auc_asym_minus_rast_positive_count'])}/"
            f"{int(item['degradation_auc_asym_minus_rast_zero_count'])}",
        ]
        for item in family.to_dict("records")
    ]
    write_table(
        TABLE_OUTPUT / "table_round12_lofo.tex",
        caption=(
            "Leave-one-perturbation-family-out retraining on FD004. "
            "Differences are OCM-Asym minus RAST-GRU. Perturbation seeds are "
            "averaged at each level. Each curve includes level zero: the absolute "
            "curve starts at the model's clean metric and the degradation curve at "
            "zero. Trapezoidal area is divided by the family maximum intensity, "
            "then scenarios are equally averaged."
        ),
        label="tab:round12-lofo",
        headers=[
            "Held-out family",
            "Metric",
            "Streams",
            "Absolute AUC diff. (SD)",
            r"$\Delta$AUC diff. (SD)",
            r"Abs. $-/+/0$",
            r"$\Delta$ $-/+/0$",
        ],
        rows=rows,
        align="llccccc",
        resize=True,
    )
    write_table(
        TABLE_OUTPUT / "table_round12_lofo_zh.tex",
        caption=(
            "FD004 上逐算法扰动族留一重新训练，差值为 OCM-Asym 减 RAST-GRU。"
            "先在各强度平均扰动种子。每条曲线都加入强度零点：绝对曲线以模型自身 clean 指标起始，"
            "退化曲线以零起始；梯形面积除以该族最大强度，再对同族情景等权平均。"
        ),
        label="tab:round12-lofo-zh",
        headers=[
            "留出扰动族",
            "指标",
            "训练流",
            "绝对 AUC 差（标准差）",
            "退化 AUC 差（标准差）",
            "绝对负/正/零",
            "退化负/正/零",
        ],
        rows=rows,
        align="llccccc",
        resize=True,
    )

    lpr = family[
        family["metric"] == "critical_30_late_prediction_ratio"
    ].sort_values("held_out_family")
    estimand_rows = [
        [
            item["held_out_family"],
            f"{item['nonzero_absolute_mean_asym_minus_rast_mean']:.4f}",
            f"{item['absolute_auc_asym_minus_rast_mean']:.4f}",
            f"{item['nonzero_degradation_mean_asym_minus_rast_mean']:.4f}",
            f"{item['degradation_auc_asym_minus_rast_mean']:.4f}",
        ]
        for item in lpr.to_dict("records")
    ]
    write_table(
        TABLE_OUTPUT / "table_round16_lofo_estimand_comparison.tex",
        caption=(
            "LPR@30 sensitivity to LOFO curve definition. Nonzero mean integrates "
            "only from the minimum tested nonzero intensity to the maximum; "
            "zero-anchored AUC additionally includes the declared clean origin and "
            "normalizes over zero to the family maximum."
        ),
        label="tab:round16-lofo-estimands",
        headers=["Held-out family", "Nonzero absolute", "Zero-anchored absolute", "Nonzero degradation", "Zero-anchored degradation"],
        rows=estimand_rows,
        align="lrrrr",
        resize=True,
    )
    write_table(
        TABLE_OUTPUT / "table_round16_lofo_estimand_comparison_zh.tex",
        caption=(
            "LOFO 曲线定义对 LPR@30 的敏感性。非零均值只在最小非零强度至最大强度积分；"
            "零锚定 AUC 另加入声明的 clean 原点，并在零至该族最大强度上归一化。"
        ),
        label="tab:round16-lofo-estimands-zh",
        headers=["留出扰动族", "非零绝对", "零锚定绝对", "非零退化", "零锚定退化"],
        rows=estimand_rows,
        align="lrrrr",
        resize=True,
    )

    support = require_csv("lofo_late_event_support.csv")
    maximum = support.groupby(
        ["held_out_family", "point", "scenario"], as_index=False
    )["level"].max().rename(columns={"level": "maximum_level"})
    support = support.merge(
        maximum,
        on=["held_out_family", "point", "scenario"],
        how="left",
        validate="many_to_one",
    )
    support = support[np.isclose(support["level"], support["maximum_level"])]
    support_rows = []
    for item in support.sort_values(
        ["held_out_family", "scenario", "level", "epsilon", "point"]
    ).to_dict("records"):
        cmle = "NA" if pd.isna(item["cmle_median"]) else f"{item['cmle_median']:.3f}"
        support_rows.append(
            [
                item["held_out_family"],
                item["scenario"],
                f"{float(item['level']):.3g}",
                item["point"],
                f"{item['epsilon']:.0f}",
                int(item["eligible_engines_per_evaluation_median"]),
                f"{item['late_event_count_median']:.0f} "
                f"[{item['late_event_count_q1']:.0f}, {item['late_event_count_q3']:.0f}]",
                f"{int(item['zero_event_evaluation_count'])}/"
                f"{int(item['evaluation_count'])}",
                f"{item['late_ratio_mean']:.4f}",
                f"{item['zimle_median']:.3f}",
                f"{cmle} [{int(item['undefined_cmle_evaluation_count'])}/"
                f"{int(item['evaluation_count'])}]",
            ]
        )
    write_table(
        TABLE_OUTPUT / "table_round16_lofo_event_support.tex",
        caption=(
            "Near-failure event support at each held-out family's maximum tested "
            "intensity. Each scenario and level has a unique row. Event counts are "
            "median [interquartile range] across the 25 stream--perturbation-seed "
            "evaluations; zero-event and undefined-CMLE evaluations are counts out "
            "of 25. Epsilon 0, 5, and 10 use tolerance-specific event, ZIMLE, and "
            "CMLE definitions."
        ),
        label="tab:round16-lofo-event-support",
        headers=[
            "Family",
            "Scenario",
            "Level",
            "Point",
            r"$\epsilon$",
            "Eligible",
            "Events med. [IQR]",
            "Zero eval.",
            "Ratio",
            "ZIMLE med.",
            "CMLE med. [undef.]",
        ],
        rows=support_rows,
        align="lllrrrrrrrr",
        resize=True,
    )
    write_table(
        TABLE_OUTPUT / "table_round16_lofo_event_support_zh.tex",
        caption=(
            "各留出扰动族最大测试强度下的临近失效事件支持。每个场景和强度对应唯一行。事件数为 25 次"
            "训练流--扰动种子评价的中位数[四分位距]；零事件和 CMLE 未定义评价以占 25 次的计数报告。"
            "epsilon 为 0、5、10 时，事件、ZIMLE 与 CMLE 均按相应容差重新计算。"
        ),
        label="tab:round16-lofo-event-support-zh",
        headers=["扰动族", "场景", "强度", "配置", r"$\epsilon$", "Eligible", "事件中位[IQR]", "零事件", "比例", "ZIMLE中位", "CMLE中位[未定义]"],
        rows=support_rows,
        align="lllrrrrrrrr",
        resize=True,
    )

    stream_support = require_csv("lofo_late_event_support_by_training_stream.csv")
    maximum = stream_support.groupby(
        ["held_out_family", "point", "scenario"], as_index=False
    )["level"].max().rename(columns={"level": "maximum_level"})
    stream_support = stream_support.merge(
        maximum,
        on=["held_out_family", "point", "scenario"],
        how="left",
        validate="many_to_one",
    )
    stream_support = stream_support[
        np.isclose(stream_support["level"], stream_support["maximum_level"])
        & np.isclose(stream_support["epsilon"], 10.0)
    ]
    stream_rows: list[list[object]] = []
    for keys, group in stream_support.groupby(
        ["held_out_family", "scenario", "level", "training_stream_seed"],
        sort=True,
    ):
        by_point = group.set_index("point")

        def event_cell(point: str) -> str:
            row = by_point.loc[point]
            return (
                f"{row['late_event_count_mean']:.1f} "
                f"[{int(row['late_event_count_min'])}, {int(row['late_event_count_max'])}]"
            )

        def severity_cell(point: str) -> str:
            row = by_point.loc[point]
            cmle = "NA" if pd.isna(row["cmle_mean"]) else f"{row['cmle_mean']:.2f}"
            return f"{row['zimle_mean']:.2f}/{cmle}"

        stream_rows.append(
            [
                keys[0],
                keys[1],
                f"{float(keys[2]):.3g}",
                int(keys[3]),
                event_cell("rast"),
                event_cell("asym"),
                severity_cell("rast"),
                severity_cell("asym"),
            ]
        )
    write_table(
        TABLE_OUTPUT / "table_round19_lofo_stream_support.tex",
        caption=(
            "Training-stream-resolved FD004 LOFO event support at each scenario's "
            "maximum intensity and tolerance $\\epsilon=10$. Event entries are mean "
            "[minimum, maximum] counts across five perturbation seeds; severity entries "
            "are ZIMLE/CMLE means. Each stream reuses the same 53 eligible engines across "
            "perturbation seeds, so the five seed evaluations within a stream are clustered, "
            "not independent repetitions. The machine-readable companion retains every "
            "level, point, and $\\epsilon\\in\\{0,5,10\\}$."
        ),
        label="tab:round19-lofo-stream-support",
        headers=[
            "Family",
            "Scenario",
            "Level",
            "Stream",
            "RAST events",
            "Asym events",
            "RAST ZI/CM",
            "Asym ZI/CM",
        ],
        rows=stream_rows,
        align="lllrrrrr",
        resize=True,
    )
    write_table(
        TABLE_OUTPUT / "table_round19_lofo_stream_support_zh.tex",
        caption=(
            "FD004 逐族留出分析在各场景最大强度、容差 $\\epsilon=10$ 下按训练流展开的事件支持。"
            "事件项为五个扰动种子上的均值[最小值, 最大值]，严重度项为 ZIMLE/CMLE 均值。"
            "每个训练流中的五次评价复用相同 53 台 eligible 发动机，属于聚类重复而非独立重复。"
            "机器可读配套文件保留所有强度、配置及 $\\epsilon\\in\\{0,5,10\\}$。"
        ),
        label="tab:round19-lofo-stream-support-zh",
        headers=["扰动族", "场景", "强度", "训练流", "RAST事件", "Asym事件", "RAST ZI/CM", "Asym ZI/CM"],
        rows=stream_rows,
        align="lllrrrrr",
        resize=True,
    )

    level = require_csv("lofo_level_paired_degradation_summary.csv")
    level_rows: list[list[object]] = []
    for (held_out, scenario, intensity), group in level.groupby(
        ["held_out_family", "scenario", "level"], sort=True
    ):
        values = {item["metric"]: item for item in group.to_dict("records")}

        def effect_cell(metric: str) -> str:
            item = values[metric]
            return (
                f"{item['degradation_asym_minus_rast_mean']:.4f} "
                f"({item['degradation_asym_minus_rast_sd']:.4f})"
            )

        level_rows.append(
            [
                held_out,
                scenario,
                f"{float(intensity):.3g}",
                effect_cell("rmse"),
                effect_cell("nasa_per_engine"),
                effect_cell("critical_30_late_prediction_ratio"),
            ]
        )

    write_table(
        TABLE_OUTPUT / "table_round12_lofo_levels.tex",
        caption=(
            "Intensity-level clean-corrected degradation effects for the FD004 "
            "leave-one-family-out audit. Entries are mean (training-stream SD) "
            "of OCM-Asym minus RAST-GRU after subtracting each model's own clean "
            "metric. Stream-level sign counts are retained in the accompanying "
            "machine-readable CSV."
        ),
        label="tab:round12-lofo-levels",
        headers=[
            "Held-out family",
            "Scenario",
            "Level",
            r"$\Delta$RMSE",
            r"$\Delta$NASA/engine",
            r"$\Delta$LPR@30",
        ],
        rows=level_rows,
        align="lllccc",
        resize=True,
    )
    write_table(
        TABLE_OUTPUT / "table_round12_lofo_levels_zh.tex",
        caption=(
            "FD004 逐扰动族留一审计的逐强度 clean 校正退化效应。单元格为 OCM-Asym "
            "减 RAST-GRU 的均值（训练流标准差），并已减去各模型自身 clean 指标。"
            "训练流级符号计数保存在配套机器可读 CSV 中。"
        ),
        label="tab:round12-lofo-levels-zh",
        headers=[
            "留出扰动族",
            "情景",
            "强度",
            r"$\Delta$RMSE",
            r"$\Delta$NASA/engine",
            r"$\Delta$LPR@30",
        ],
        rows=level_rows,
        align="lllccc",
        resize=True,
    )


def build_ncmapss_table() -> None:
    frame = require_csv("ncmapss_sampling_window_summary.csv")
    rows = [
        [
            int(item["sampling_interval"]),
            int(item["sensitivity_window_size"]),
            int(item["seed_count"]),
            int(item["train_windows"]),
            int(item["source_record_span"]),
            f"{item['unit_macro_rmse_mean']:.3f} "
            f"({item['unit_macro_rmse_sd']:.3f})",
            f"{item['unit_macro_lpr30_mean']:.3f} "
            f"({item['unit_macro_lpr30_sd']:.3f})",
        ]
        for item in frame.to_dict("records")
    ]
    write_table(
        TABLE_OUTPUT / "table_round12_ncmapss_sampling.tex",
        caption=(
            "N-CMAPSS DS02 computational-proxy sensitivity to source-record "
            "sampling interval and sampled-record window length. Source-record "
            "span is $1+(W-1)s$ and is not a calibrated physical duration. Values are "
            "five-seed means (SD); the three official test units are unchanged."
        ),
        label="tab:round12-ncmapss-sampling",
        headers=[
            "Sampling",
            "Window",
            "Seeds",
            "Train windows",
            "Source-record span",
            "Macro RMSE",
            "Macro LPR@30",
        ],
        rows=rows,
        align="rrrrrcc",
        resize=True,
    )
    write_table(
        TABLE_OUTPUT / "table_round12_ncmapss_sampling_zh.tex",
        caption=(
            "N-CMAPSS DS02 计算代理对源记录采样间隔和采样后窗口长度的敏感性。"
            "源记录跨度按 $1+(W-1)s$ 计算，不等同于已校准的物理时长。"
            "数值为五 seed 均值（标准差），三个正式测试单元保持不变。"
        ),
        label="tab:round12-ncmapss-sampling-zh",
        headers=[
            "采样间隔",
            "窗口",
            "Seed",
            "训练窗口",
            "源记录跨度",
            "宏观 RMSE",
            "宏观 LPR@30",
        ],
        rows=rows,
        align="rrrrrcc",
        resize=True,
    )
    matched = require_csv("ncmapss_horizon_matched_summary.csv")
    matched_rows = [
        [
            int(item["sampling_interval"]),
            int(item["sensitivity_window_size"]),
            int(item["source_record_span"]),
            int(item["train_windows"]),
            f"{item['unit_macro_rmse_mean']:.3f} ({item['unit_macro_rmse_sd']:.3f})",
            f"{item['unit_macro_lpr30_mean']:.3f} ({item['unit_macro_lpr30_sd']:.3f})",
        ]
        for item in matched.to_dict("records")
    ]
    write_table(
        TABLE_OUTPUT / "table_round12_ncmapss_horizon_matched.tex",
        caption=(
            "N-CMAPSS density sensitivity at an exactly matched 6,001-source-record "
            "span. Sampling density changes while the discrete source-record horizon "
            "is fixed; the same three official test units remain in every cell."
        ),
        label="tab:round12-ncmapss-horizon",
        headers=["Sampling", "Window", "Span", "Train windows", "Macro RMSE", "Macro LPR@30"],
        rows=matched_rows,
        align="rrrrcc",
    )
    write_table(
        TABLE_OUTPUT / "table_round12_ncmapss_horizon_matched_zh.tex",
        caption=(
            "固定为 6,001 个源记录跨度的 N-CMAPSS 密度敏感性。采样密度改变而"
            "离散源记录回看跨度保持一致；所有单元仍共享三个官方测试 unit。"
        ),
        label="tab:round12-ncmapss-horizon-zh",
        headers=["采样", "窗口", "跨度", "训练窗口", "宏观 RMSE", "宏观 LPR@30"],
        rows=matched_rows,
        align="rrrrcc",
    )


def build_round19_governance_tables() -> None:
    workflow = require_csv("governance_workflow.csv")
    workflow_rows = [
        [
            int(item["step"]),
            item["component"],
            item["mandatory_rule"],
            item["recommended_record"],
            item["retrospective_only_boundary"],
        ]
        for item in workflow.to_dict("records")
    ]
    write_table(
        TABLE_OUTPUT / "table_round19_governance_workflow.tex",
        caption=(
            "Benchmark-claim calibration checklist instantiated in this retrospective "
            "case study. Mandatory rules define the minimum claim-to-artifact chain; "
            "the final column prevents result-informed work from being presented as "
            "prospective validation."
        ),
        label="tab:round19-governance-workflow",
        headers=["Step", "Component", "Mandatory rule", "Recommended record", "Retrospective boundary"],
        rows=workflow_rows,
        align="rlp{4.2cm}p{3.6cm}p{4.0cm}",
        resize=True,
    )
    workflow_rows_zh = [
        [1, "主张", "在选择图表前先写出边界明确的经验主张", "主张编号及禁止外推项", "明确标记结果知情的时间顺序"],
        [2, "估计量", "定义结果、聚合单位、配对、阈值与方向", "机器可读指标 schema", "不得把事后估计量改称确认性"],
        [3, "设计状态", "区分前瞻、结果知情、敏感性和运行时角色", "分析族注册表与时间线", "已完成水平只定义有限设计"],
        [4, "不确定性", "表述强度必须匹配设计水平的数量和独立性", "逐 seed 效应、符号数、事件支持与校准审计", "未校准区间和类 p 量仅作补充诊断"],
        [5, "制品", "把主张映射到源数据、代码、环境与不可变校验和", "图表到脚本索引及一次性工作区验证", "本地校验和不等于公开时间戳或 DOI"],
        [6, "允许措辞", "报告完整方向范围及设计无法识别的内容", "最负向和最正向场景的仪表板", "使用协议条件化审计，不使用支配性或泛化措辞"],
    ]
    write_table(
        TABLE_OUTPUT / "table_round19_governance_workflow_zh.tex",
        caption=(
            "本回顾性案例实例化的基准主张校准清单。强制规则定义最小的主张--制品链，"
            "推荐记录增强可审计性，最后一列防止把结果知情分析包装为前瞻性验证。"
        ),
        label="tab:round19-governance-workflow-zh",
        headers=["步骤", "组件", "强制规则", "推荐记录", "回顾性边界"],
        rows=workflow_rows_zh,
        align="rlp{4.2cm}p{3.6cm}p{4.0cm}",
        resize=True,
    )

    dashboard = require_csv("retrospective_family_dashboard.csv")
    dashboard_rows = []
    for item in dashboard.to_dict("records"):
        is_directional = str(item.get("directional_contrast", True)).lower() == "true"
        dashboard_rows.append(
            [
                item["family_id"],
                METRIC_LABELS.get(item["metric"], item["metric"]),
                int(item["completed_direction_count"]),
                (
                    f"{int(item['negative_count'])}/{int(item['positive_count'])}/{int(item['zero_count'])}"
                    if is_directional
                    else "N/A"
                ),
                f"[{item['estimate_min']:.4g}, {item['estimate_max']:.4g}]",
                item["most_negative_context"],
                item["most_positive_context"],
                item["headline_impact"],
            ]
        )
    write_table(
        TABLE_OUTPUT / "table_round19_retrospective_dashboard.tex",
        caption=(
            "Complete retrospective-family direction dashboard. N is the number "
            "of completed finite-design cells and signs are negative/positive/zero for "
            "paired contrasts. N-CMAPSS proxy rows are not paired model contrasts and "
            "therefore show N/A. "
            "Ranges are descriptive and metric-specific, not confidence intervals. "
            "Every registered family and metric is included before the most negative "
            "and most positive completed contexts are selected."
        ),
        label="tab:round19-retrospective-dashboard",
        headers=["Family", "Metric", "N", "-/+/0", "Range", "Minimum context", "Maximum context", "Headline consequence"],
        rows=dashboard_rows,
        align="llrrllll",
        resize=True,
    )
    write_table(
        TABLE_OUTPUT / "table_round19_retrospective_dashboard_zh.tex",
        caption=(
            "完整回顾性分析族方向仪表板。N 为已完成有限设计方向数，符号依次为负/正/零。"
            "范围均为指标特定描述量，不是置信区间。提取规则先覆盖每个已注册分析族和指标，"
            "再报告最负向与最正向的已完成场景。"
        ),
        label="tab:round19-retrospective-dashboard-zh",
        headers=["分析族", "指标", "N", "负/正/零", "范围", "最负向场景", "最正向场景", "对主结论影响"],
        rows=dashboard_rows,
        align="llrrllll",
        resize=True,
    )


def build_figures() -> None:
    FIGURE_OUTPUT.mkdir(parents=True, exist_ok=True)
    fig, axis = plt.subplots(figsize=(12.4, 5.7), constrained_layout=True)
    axis.set_xlim(0, 12.4)
    axis.set_ylim(0, 6.0)
    axis.axis("off")
    top_nodes = [
        (0.35, 3.9, 3.05, "Compressed benchmark claim\n'Configuration A is better'", "#d9e8f5"),
        (4.05, 3.9, 4.30, "Hidden design degrees of freedom\nTask and split | preprocessing | training stream\nmetric and aggregation | perturbation protocol", "#fce5cd"),
        (9.00, 3.9, 3.05, "Observed instability\nranking shifts, direction reversals,\nand negative results", "#f4cccc"),
    ]
    box_height = 1.35
    for x, y, width, label, color in top_nodes:
        axis.add_patch(
            Rectangle(
                (x, y),
                width,
                box_height,
                facecolor=color,
                edgecolor="#3f4a54",
                linewidth=1.0,
            )
        )
        axis.text(
            x + width / 2,
            y + box_height / 2,
            label,
            ha="center",
            va="center",
            fontsize=9.0,
        )
    arrow_pairs = [
        ((3.40, 4.575), (4.05, 4.575)),
        ((8.35, 4.575), (9.00, 4.575)),
    ]
    for start, end in arrow_pairs:
        axis.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.1,
                color="#3f4a54",
            )
        )
    axis.text(
        6.2,
        5.70,
        "Calibrating a benchmark-based scientific claim",
        ha="center",
        va="center",
        fontsize=12.5,
        fontweight="bold",
    )

    axis.add_patch(
        FancyArrowPatch(
            (10.52, 3.90),
            (10.52, 3.25),
            arrowstyle="-|>",
            mutation_scale=13,
            linewidth=1.1,
            color="#3f4a54",
        )
    )
    axis.text(6.2, 3.25, "Six auditable links", ha="center", va="center", fontsize=9.3, fontweight="bold")

    chain = [
        (0.35, "1  Claim\nBound the statement", "#d9e8f5"),
        (2.35, "2  Estimand\nUnit and direction", "#e2f0d9"),
        (4.35, "3  Design status\nProspective or informed", "#fce5cd"),
        (6.35, "4  Evidence role\nEstimate or diagnostic", "#eadcf0"),
        (8.35, "5  Artifact\nRows, code, checksum", "#d9ead3"),
        (10.35, "6  Wording\nState the supported scope", "#e8efcf"),
    ]
    chain_width, chain_height, chain_y = 1.70, 1.05, 1.75
    for x, label, color in chain:
        axis.add_patch(
            Rectangle(
                (x, chain_y),
                chain_width,
                chain_height,
                facecolor=color,
                edgecolor="#3f4a54",
                linewidth=0.9,
            )
        )
        axis.text(x + chain_width / 2, chain_y + chain_height / 2, label, ha="center", va="center", fontsize=7.9)
    for x in [2.05, 4.05, 6.05, 8.05, 10.05]:
        axis.add_patch(
            FancyArrowPatch(
                (x, chain_y + chain_height / 2),
                (x + 0.30, chain_y + chain_height / 2),
                arrowstyle="-|>",
                mutation_scale=10,
                linewidth=0.9,
                color="#3f4a54",
            )
        )
    axis.add_patch(
        Rectangle((2.35, 0.45), 7.70, 0.72, facecolor="#eef4ea", edgecolor="#4f6b4f", linewidth=1.0)
    )
    axis.text(
        6.2,
        0.81,
        "Calibrated output: a protocol-conditional statement whose strength does not exceed its design or provenance.",
        ha="center",
        va="center",
        fontsize=8.9,
    )
    fig.savefig(FIGURE_OUTPUT / "round19_governance_workflow.png", dpi=300)
    fig.savefig(FIGURE_OUTPUT / "round19_governance_workflow.pdf")
    plt.close(fig)

    paired = require_csv("independent_stream_metrics.csv")
    pivot = paired.pivot(
        index=["subset", "training_stream_seed"],
        columns="point",
        values=list(METRIC_LABELS)[:3],
    )
    fig, axes = plt.subplots(1, 3, figsize=(12.0, 3.7), constrained_layout=True)
    colors = {"FD001": "#2166ac", "FD002": "#b2182b", "FD003": "#1b7837", "FD004": "#7b3294"}
    stream_ids = sorted(paired["training_stream_seed"].unique())
    markers = ("o", "s", "^", "v", "D", "P", "X", "<", ">", "h")
    marker_map = dict(zip(stream_ids, markers, strict=True))
    for axis, metric in zip(axes, list(METRIC_LABELS)[:3], strict=True):
        for subset_index, subset in enumerate(("FD001", "FD002", "FD003", "FD004")):
            values = (
                pivot.loc[subset][(metric, "asym")]
                - pivot.loc[subset][(metric, "rast")]
            )
            x = np.full(len(values), subset_index, dtype=float)
            x += np.linspace(-0.12, 0.12, len(values))
            for position, (stream, value) in enumerate(values.items()):
                axis.scatter(
                    x[position],
                    value,
                    s=31,
                    color=colors[subset],
                    marker=marker_map[stream],
                    alpha=0.9,
                )
            axis.plot(
                [subset_index - 0.19, subset_index + 0.19],
                [values.mean(), values.mean()],
                color="black",
                linewidth=2,
            )
        axis.axhline(0.0, color="#555555", linewidth=0.9, linestyle="--")
        axis.set_xticks(range(4), ("FD001", "FD002", "FD003", "FD004"))
        axis.set_title(METRIC_LABELS[metric], fontsize=10)
        axis.set_ylabel("OCM-Asym minus RAST-GRU", fontsize=9)
        axis.tick_params(labelsize=8)
    handles = [
        plt.Line2D(
            [],
            [],
            color="#444444",
            marker=marker_map[stream],
            linestyle="None",
            markersize=5,
            label=str(stream),
        )
        for stream in stream_ids
    ]
    fig.legend(
        handles=handles,
        title="Training-stream seed",
        loc="outside lower center",
        ncol=5,
        fontsize=7,
        title_fontsize=8,
    )
    fig.savefig(FIGURE_OUTPUT / "round12_independent_stream_effects.png", dpi=300)
    plt.close(fig)

    crossed = require_csv("crossed_split_stream_effects.csv")
    crossed_metrics = ("test_rmse", "test_critical_30_late_prediction_ratio")
    crossed_subsets = ("FD001", "FD002", "FD003", "FD004")
    split_ids = sorted(crossed["split_seed"].unique())
    stream_ids = sorted(crossed["training_stream_seed"].unique())
    for metric in crossed_metrics:
        metric_values = crossed.loc[
            crossed["metric"] == metric, "asym_minus_rast"
        ].to_numpy(dtype=float)
        limit = max(abs(metric_values.min()), abs(metric_values.max()), 1e-9)
        fig, axes = plt.subplots(1, 4, figsize=(13.2, 4.1), constrained_layout=True)
        metric_title = METRIC_LABELS[metric]
        fig.suptitle(
            f"{metric_title}: result-informed selected 3x3 finite grid",
            fontsize=12,
        )
        for column_index, subset in enumerate(crossed_subsets):
            axis = axes[column_index]
            part = crossed[
                (crossed["metric"] == metric) & (crossed["subset"] == subset)
            ]
            heat = part.pivot(
                index="split_seed",
                columns="training_stream_seed",
                values="asym_minus_rast",
            ).reindex(index=split_ids, columns=stream_ids)
            image = axis.imshow(
                heat.to_numpy(dtype=float),
                cmap="RdBu_r",
                vmin=-limit,
                vmax=limit,
                aspect="equal",
            )
            axis.set_xticks(range(len(stream_ids)), [str(x) for x in stream_ids])
            axis.set_yticks(range(len(split_ids)), [str(x) for x in split_ids])
            axis.tick_params(labelsize=8.5)
            axis.set_title(subset, fontsize=10)
            axis.set_xlabel("Selected training-stream level", fontsize=9)
            if column_index == 0:
                axis.set_ylabel("Selected engine-split level", fontsize=9)
            for i in range(heat.shape[0]):
                for j in range(heat.shape[1]):
                    value = float(heat.iloc[i, j])
                    label = f"{value:+.2f}" if metric == "test_rmse" else f"{value:+.3f}"
                    axis.text(
                        j,
                        i,
                        label,
                        ha="center",
                        va="center",
                        fontsize=8,
                        color="white" if abs(value) > 0.55 * limit else "black",
                    )
        colorbar = fig.colorbar(image, ax=axes, shrink=0.78, pad=0.015)
        colorbar.set_label("OCM-Asym minus RAST-GRU", fontsize=9)
        colorbar.ax.tick_params(labelsize=8)
        suffix = "rmse" if metric == "test_rmse" else "lpr"
        fig.savefig(
            FIGURE_OUTPUT / f"round12_crossed_split_stream_{suffix}.png",
            dpi=300,
            bbox_inches="tight",
        )
        fig.savefig(
            FIGURE_OUTPUT / f"round12_crossed_split_stream_{suffix}.pdf",
            bbox_inches="tight",
        )
        plt.close(fig)

    preprocess = require_csv("cmapss_preprocessing_paired_effects.csv")
    order = [
        ("window", 20, "W20"),
        ("reference", 30, "W30/B14"),
        ("window", 50, "W50"),
        ("sensor_budget", 8, "B8"),
        ("sensor_budget", 21, "B21"),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(12.2, 3.8), constrained_layout=True)
    for column_index, metric in enumerate(list(METRIC_LABELS)[:3]):
        subset = "FD004"
        axis = axes[column_index]
        part = preprocess[(preprocess["subset"] == subset) & (preprocess["metric"] == metric)]
        for x, (audit_axis, level, label) in enumerate(order):
            values = part[
                (part["audit_axis"] == audit_axis)
                & (part["audit_level"] == level)
            ]["asym_minus_rast"].to_numpy(dtype=float)
            axis.scatter(
                np.full(len(values), x) + np.linspace(-0.08, 0.08, len(values)),
                values,
                color="#2166ac" if audit_axis != "sensor_budget" else "#b2182b",
                s=28,
                alpha=0.8,
            )
            axis.plot(x, values.mean(), marker="_", markersize=16, color="#111111")
        axis.axhline(0.0, color="#333333", linewidth=0.9)
        axis.set_xticks(range(len(order)), [item[2] for item in order])
        axis.set_title(f"{subset}: {METRIC_LABELS.get(metric, metric)}")
        axis.set_ylabel("OCM-Asym minus RAST-GRU")
        axis.grid(axis="y", color="#dddddd", linewidth=0.6)
    fig.savefig(FIGURE_OUTPUT / "round12_cmapss_preprocessing.png", dpi=300)
    plt.close(fig)

    preference = require_csv("preference_retraining_candidate_summary.csv")
    heat = preference.pivot(
        index="late_life_weight",
        columns="late_over_weight",
        values="selection_count",
    ).sort_index(ascending=False)
    fig, axis = plt.subplots(figsize=(5.3, 4.2), constrained_layout=True)
    image = axis.imshow(heat.to_numpy(), cmap="cividis", vmin=0.0, vmax=2.0)
    axis.set_xticks(range(len(heat.columns)), [f"{x:.2f}" for x in heat.columns])
    axis.set_yticks(range(len(heat.index)), [f"{x:.2f}" for x in heat.index])
    axis.set_xlabel(r"$\lambda_{\mathrm{late}}$")
    axis.set_ylabel(r"$\lambda_{\mathrm{life}}$")
    axis.set_title("Retrained preference selection frequency", fontsize=10)
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            axis.text(
                j,
                i,
                f"{int(heat.iloc[i, j])}/9",
                ha="center",
                va="center",
                color="white" if heat.iloc[i, j] < 1.5 else "black",
                fontsize=9,
            )
    colorbar = fig.colorbar(image, ax=axis, label="Selection count (of 9)")
    colorbar.set_ticks([0, 1, 2])
    fig.savefig(FIGURE_OUTPUT / "round12_preference_retraining.png", dpi=300)
    plt.close(fig)

    ncmapss = require_csv("ncmapss_sampling_window_summary.csv")
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 4.0), constrained_layout=True)
    for axis, metric, title, colorbar_label in (
        (axes[0], "unit_macro_rmse_mean", "Macro RMSE", "RMSE"),
        (axes[1], "unit_macro_lpr30_mean", "Macro LPR@30", "LPR@30"),
    ):
        heat = ncmapss.pivot(
            index="sampling_interval",
            columns="sensitivity_window_size",
            values=metric,
        ).sort_index()
        image = axis.imshow(heat.to_numpy(), cmap="cividis")
        axis.set_xticks(range(len(heat.columns)), [str(x) for x in heat.columns])
        axis.set_yticks(range(len(heat.index)), [str(x) for x in heat.index])
        axis.set_xlabel("Window length after sampling")
        axis.set_ylabel("Source-record sampling interval")
        axis.set_title(f"N-CMAPSS proxy: {title}", fontsize=10)
        for i in range(heat.shape[0]):
            for j in range(heat.shape[1]):
                axis.text(
                    j,
                    i,
                    f"{heat.iloc[i, j]:.3f}",
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=8,
                )
        fig.colorbar(image, ax=axis, label=colorbar_label)
    fig.savefig(FIGURE_OUTPUT / "round12_ncmapss_sampling_window.png", dpi=300)
    plt.close(fig)


def copy_outputs(table_targets: list[Path], figure_targets: list[Path]) -> None:
    table_files = sorted(
        set(TABLE_OUTPUT.glob("table_round12_*.tex"))
        | set(TABLE_OUTPUT.glob("table_round16_*.tex"))
        | set(TABLE_OUTPUT.glob("table_round19_*.tex"))
    )
    figure_files = sorted(
        set(FIGURE_OUTPUT.glob("round12_*.png"))
        | set(FIGURE_OUTPUT.glob("round12_*.pdf"))
        | set(FIGURE_OUTPUT.glob("round19_*.png"))
        | set(FIGURE_OUTPUT.glob("round19_*.pdf"))
    )
    for target in table_targets:
        target.mkdir(parents=True, exist_ok=True)
        for source in table_files:
            shutil.copy2(source, target / source.name)
    for target in figure_targets:
        target.mkdir(parents=True, exist_ok=True)
        for source in figure_files:
            shutil.copy2(source, target / source.name)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build manuscript tables and figures from Round-12 evidence."
    )
    parser.add_argument("--copy-tables-to", nargs="*", type=Path, default=[])
    parser.add_argument("--copy-figures-to", nargs="*", type=Path, default=[])
    args = parser.parse_args()
    build_independent_table()
    build_crossed_table()
    build_crossed_selection_tables()
    build_preprocessing_resource_table()
    build_preprocessing_table()
    build_preference_table()
    build_factorial_table()
    build_lofo_table()
    build_ncmapss_table()
    build_round19_governance_tables()
    build_figures()
    copy_outputs(args.copy_tables_to, args.copy_figures_to)
    print(
        "ROUND12_NEW_EVIDENCE_ASSETS_PASS "
        f"tables={len(list(TABLE_OUTPUT.glob('table_round*_*.tex')))} "
        f"figures={len(list(FIGURE_OUTPUT.glob('round1*_*.*')))}"
    )


if __name__ == "__main__":
    main()
