from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rul.reporting import model_display_name


DEFAULT_SOURCE = PROJECT_ROOT / "paper_outputs" / "advanced_evidence"
DEFAULT_OUTPUT = PROJECT_ROOT / "paper_outputs" / "manuscript_v3"
PROPOSED = "rast_gru_v2"


def latex_escape(value: object) -> str:
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


def require_csv(directory: Path, filename: str, minimum_rows: int) -> pd.DataFrame:
    path = directory / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing manuscript evidence source: {path}")
    frame = pd.read_csv(path)
    if len(frame) < minimum_rows:
        raise ValueError(f"{filename} has {len(frame)} rows; expected at least {minimum_rows}")
    return frame


def ensure_model_display(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach canonical manuscript labels to sources that store only model IDs."""
    if "model" not in frame.columns:
        raise ValueError("A model table must contain a 'model' column.")
    result = frame.copy()
    if "model_display" not in result.columns:
        result["model_display"] = result["model"].map(model_display_name)
    if result["model_display"].isna().any():
        missing = sorted(result.loc[result["model_display"].isna(), "model"].astype(str).unique())
        raise ValueError(f"Missing canonical display labels for models: {missing}")
    return result


def macro(name: str, value: object) -> str:
    return rf"\providecommand{{\{name}}}{{{value}}}"


def write_booktabs_table(
    frame: pd.DataFrame,
    output: Path,
    *,
    columns: list[tuple[str, str, str]],
    caption: str,
    label: str,
    note: str,
    font_command: str = r"\small",
    fit_width: bool = False,
    column_spec: str | None = None,
) -> None:
    align = column_spec or ("l" + "r" * (len(columns) - 1))
    lines = [
        r"\begin{table}[!htbp]",
        r"\centering",
        rf"\caption{{{caption}}}",
        rf"\label{{{label}}}",
        font_command,
        r"\setlength{\tabcolsep}{3.5pt}",
    ]
    if fit_width:
        lines.append(r"\resizebox{\linewidth}{!}{%")
    lines.extend(
        [
            rf"\begin{{tabular}}{{{align}}}",
            r"\toprule",
            " & ".join(header for _, header, _ in columns) + r" \\",
            r"\midrule",
        ]
    )
    for row in frame.to_dict("records"):
        values = []
        for column, _, fmt in columns:
            value = row[column]
            if fmt == "text":
                values.append(latex_escape(value))
            elif fmt == "raw":
                values.append(str(value))
            elif fmt == "int":
                values.append(f"{int(round(float(value))):,}")
            else:
                values.append(format(float(value), fmt))
        lines.append(" & ".join(values) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    if fit_width:
        lines.extend([r"}%", r"\par"])
    lines.extend(
        [
            rf"\begin{{minipage}}{{0.98\linewidth}}\footnotesize Notes: {note}\end{{minipage}}",
            r"\end{table}",
            "",
        ]
    )
    output.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def write_stress_family_longtable(frame: pd.DataFrame, output: Path) -> None:
    caption = "Family-level paired resampling diagnostics for controlled-degradation accuracy and late-overprediction metrics."
    note = (
        "The crossed seed--perturbation-seed--engine bootstrap diagnostic resamples five coupled composite-seed levels, five matched "
        "perturbation seeds within each sampled training seed, and 248 matched FD004 test engines "
        "within each sampled seed pair. It is used for RMSE, LPR@30, MLE@30, late CVaR95, and "
        "asymmetric error cost at kappa=5. The direction label only records whether the uncalibrated "
        "diagnostic envelope excludes zero; it is not a confidence statement or decision rule."
    )
    header = (
        r"OCM point & Algorithmic perturbation family & Metric & OCM$-$RAST curve mean [diagnostic interval] "
        r"& Bootstrap fraction OCM lower & Diagnostic interval direction \\"
    )
    lines = [
        r"\begingroup\footnotesize",
        r"\setlength{\tabcolsep}{2.5pt}",
        r"\begin{longtable}{@{}p{0.075\linewidth}p{0.145\linewidth}p{0.135\linewidth}p{0.225\linewidth}p{0.105\linewidth}p{0.205\linewidth}@{}}",
        rf"\caption{{{caption}}}",
        r"\label{tab:v3-stress-family-ci}\\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endfirsthead",
        rf"\caption[]{{{caption} (continued)}}\\",
        r"\toprule",
        header,
        r"\midrule",
        r"\endhead",
        r"\midrule",
        r"\multicolumn{6}{r}{\footnotesize Continued on next page}\\",
        r"\endfoot",
        r"\bottomrule",
        rf"\multicolumn{{6}}{{@{{}}p{{0.97\linewidth}}@{{}}}}{{\footnotesize Notes: {note}}}\\",
        r"\endlastfoot",
    ]
    for row in frame.to_dict("records"):
        lines.append(
            " & ".join(
                [
                    latex_escape(row["operating_point"]),
                    latex_escape(str(row["scenario"]).replace("_", " ")),
                    latex_escape(row["metric_display"]),
                    str(row["effect_ci"]),
                    f"{float(row['probability_point_lower']):.3f}",
                    latex_escape(row["ci_classification"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\end{longtable}", r"\endgroup", ""])
    output.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def best_competitor(frame: pd.DataFrame, column: str) -> pd.Series:
    competitors = frame[frame["model"] != PROPOSED]
    return competitors.loc[competitors[column].astype(float).idxmin()]


def add_ledger(rows: list[dict], macro_name: str, source: str, selector: str, value: object) -> None:
    rows.append({"macro": macro_name, "source_file": source, "selector": selector, "value": value})


def write_training_algorithm(output: Path) -> None:
    lines = [
        r"\begin{algorithm}[t]",
        r"\caption{Fixed-protocol training, checkpoint selection, and endpoint evaluation for OCM-MST-GRU}",
        r"\label{alg:v3-training}",
        r"\begin{algorithmic}[1]",
        r"\Require engine-level splits $\mathcal{D}_{\mathrm{tr}},\mathcal{D}_{\mathrm{val}},\mathcal{D}_{\mathrm{te}}$; declared composite seed; objective weights; augmentation mechanisms",
        r"\State Fit feature selection, regime centroids, and regime-specific scalers on $\mathcal{D}_{\mathrm{tr}}$ only; retain them unchanged",
        r"\For{each training epoch}",
        r"\State Construct the augmented primary value/mask view and a secondary Gaussian-noise view derived from it",
        r"\State Predict both views; stop-gradient the primary prediction in the consistency term; predict an ordered lower/point/upper triplet",
        r"\State Optimize the declared base, smooth late-risk, reliability, consistency, and q10/q90 auxiliary losses",
        r"\State Evaluate one endpoint per validation engine and update the risk-score checkpoint",
        r"\EndFor",
        r"\State Retain the selected checkpoint and evaluate each clean test-engine endpoint once",
        r"\State Apply matched normalized-coordinate perturbations only for algorithmic sensitivity probes",
        r"\Ensure checkpoint, point predictions, engine-level metrics, sensitivity diagnostics, and traceable artifacts",
        r"\end{algorithmic}",
        r"\end{algorithm}",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def write_training_algorithm_zh(output: Path) -> None:
    lines = [
        r"\begin{algorithm}[t]",
        r"\caption{OCM-MST-GRU 的固定协议训练、检查点选择与端点评估流程}",
        r"\label{alg:v3-training}",
        r"\begin{algorithmic}[1]",
        r"\Require 发动机级划分 $\mathcal{D}_{\mathrm{tr}},\mathcal{D}_{\mathrm{val}},\mathcal{D}_{\mathrm{te}}$；声明的复合种子、目标权重与增强机制",
        r"\State 仅在 $\mathcal{D}_{\mathrm{tr}}$ 上拟合特征选择、工况中心和分工况缩放器，并保持不变",
        r"\For{每个训练轮次}",
        r"\State 构造增强后的主数值/掩码视图，并从主视图派生第二高斯噪声视图",
        r"\State 同时预测两视图；一致性项对主预测停止梯度，并输出有序的下界/点预测/上界三元组",
        r"\State 优化声明的基础、平滑晚预测风险、可靠性、一致性及 q10/q90 辅助损失",
        r"\State 在验证发动机端点评估并更新风险评分最优检查点",
        r"\EndFor",
        r"\State 保留选定检查点，并对每台干净测试发动机端点仅评估一次",
        r"\State 仅在算法敏感性探针中施加匹配的归一化坐标扰动",
        r"\Ensure 检查点、点预测、发动机级指标、敏感性诊断和可追溯产物",
        r"\end{algorithmic}",
        r"\end{algorithm}",
        "",
    ]
    output.write_text("\n".join(lines), encoding="utf-8", newline="\n")


def current_results_section_text() -> tuple[str, str]:
    english = r"""\input{generated/numbers.tex}
\FloatBarrier
\section{Results}

\subsection{RQ1: Is there a protocol-independent architecture advantage?}

OCM-MST-GRU-Asym yields macro RMSE \OCMMacroRMSE, MAE \OCMMacroMAE, NASA score per engine \OCMMacroNASAperEngine, and LPR@30 \OCMMacroLPRThirty\ across FD001--FD004. The lowest locked-baseline values are \BestBaselineMacroRMSE\ (\BestBaselineMacroRMSEName), \BestBaselineMacroMAE\ (\BestBaselineMacroMAEName), \BestBaselineMacroNASAperEngine\ (\BestBaselineMacroNASAperEngineName), and \BestBaselineMacroLPRThirty\ (\BestBaselineMacroLPRThirtyName). Asym does not minimize point error, but it enters the descriptive accuracy--risk Pareto set in \OCMParetoSeedCount\ of five seed-level analyses. Table~\ref{tab:v3-cmapss-macro} labels the post-lock Core confirmation separately from the locked 260-run grid.

The primary uncertainty target is the engine-level paired effect within each fixed C-MAPSS subset. Table~\ref{tab:v3-fixed-subset-direct} shows that the direct-predecessor RMSE and NASA/engine intervals cross zero in all four tasks, as do LPR intervals in FD001--FD003. Only the FD004 LPR difference is resolved in favor of OCM-MST-GRU. The equal-subset macro and the cross-subset hierarchical interval are therefore descriptive summaries rather than population-level task inference.

\input{generated/table_cmapss_macro.tex}
\input{generated/table_fixed_subset_direct.tex}
\input{generated/table_core_rast_fixed_summary.tex}
The symmetric summary gives Core and Asym equal statistical status. Core has a resolved lower FD004 LPR, whereas RAST-GRU has a resolved lower FD003 NASA/engine; the other ten Core contrasts are unresolved. For Asym, only FD004 LPR resolves in its favor. Thus neither operating point has a fixed-task architecture advantage across metrics.
\FloatBarrier

\subsubsection{Checkpoint-rule sensitivity}

With the locked risk-score rule, FD004 RMSE/LPR@30 are \RiskCheckpointRMSE/\RiskCheckpointLPRThirty; pure validation-RMSE checkpointing changes them to \RMSECheckpointRMSE/\RMSECheckpointLPRThirty. The paired risk-minus-RMSE-checkpoint difference is \CheckpointRMSEDifference\ cycles for RMSE (95\% CI \CheckpointRMSECILow\ to \CheckpointRMSECIHigh) and \CheckpointLPRThirtyDifference\ for LPR@30 (\CheckpointLPRThirtyCILow\ to \CheckpointLPRThirtyCIHigh). The endpoint audit exposes small $N_{30}$ in some subset--seed cells; $N_{10}$ is not the denominator of SLPR@30,10. Thus the risk terms can be discrete and noisy. The three-design confirmation reported in the Supplementary Information evaluates whether conclusions persist after engine-balanced critical support is increased. The risk rule is treated as a sensitivity-tested selection option, not an established advantage or an optimized decision rule.

The endpoint-distribution audit further shows that uniform-single is closest to the test endpoint distribution by Wasserstein distance (\EndpointUniformWFDTwo{} on FD002 and \EndpointUniformWFDFour{} on FD004). Fixed-multi and critical-stratified place approximately half of validation endpoints at RUL $\leq30$, compared with test fractions \EndpointTestCriticalFDTwo{} and \EndpointTestCriticalFDFour{}. They are therefore deliberate critical-zone reweighting designs, not better estimates of the natural test-endpoint distribution. Supplementary interval-direction counts summarize all six fixed-task metric contrasts for each model pair and endpoint design; mean changes alone are not treated as confirmation.

The completed 30-trajectory confirmation sharpens, but does not overturn, this limitation. For each named Core--RAST or Asym--RAST pair, one of six fixed-task contrasts is resolved under uniform-single endpoints and two of six are resolved under each reweighted design; every resolved contrast favors the named OCM operating point, principally for LPR@30, while the majority remain unresolved. In the Core--Asym comparison, RMSE resolves in favor of Core on FD004 under all three endpoint designs and on FD002 under the two reweighted designs. Only critical-stratified FD004 LPR@30 resolves in favor of Asym. Thus increased critical-zone support makes some accuracy--risk contrasts more identifiable, but does not establish a universally superior endpoint design or operating point.

The complete checkpoint-sensitivity table is reported in the Supplementary Information.
\FloatBarrier

\subsubsection{Common-risk versus standardized conventional protocol}

Track B changes a declared bundle of preprocessing, input, loss, augmentation, and checkpoint choices while retaining the split, target, model family, optimizer, budget, and seed. Across the \ProtocolTotalComparisonCount{} paired model--metric comparisons, the 95\% intervals resolve \ProtocolConventionalResolvedCount{} in favor of the standardized conventional protocol and \ProtocolCommonResolvedCount{} in favor of the common-risk protocol; \ProtocolUnresolvedCount{} remain unresolved. The absolute results materially narrow the architecture claim. Under Common Base, OCM backbone minus RAST is \CommonBaseArchitectureRMSEDelta{} for RMSE, \CommonBaseArchitectureLPRDelta{} for LPR, and \CommonBaseArchitectureCVaRDelta{} for CVaR95: the backbone pays an accuracy cost while modestly reducing late-risk measures. Under Track B, OCM ranks \TrackBOCMRMSERank/6 for RMSE, \TrackBOCMNASARank/6 for NASA/engine, and \TrackBOCMLPRRank/6 for LPR. Thus, the evidence does not support a protocol-independent architecture advantage; it supports an integrated common-risk accuracy--risk configuration. Because Track B changes several settings together and is not a set of tuned native reproductions, neither track is treated as a protocol-neutral native leaderboard.

The cumulative two-backbone audit is retained only as a descriptive sensitivity analysis. Its mean trajectories are non-monotone, showing that protocol components interact with the backbone and objective; the final bundle cannot be decomposed into a claim that every added component is independently beneficial. The transition intervals in the Supplementary Information are conditional diagnostics from one fast-cuDNN runtime per composite seed, not confirmation intervals. A fixed-seed repeat audit on two backbones, two representative stages, and two seeds shows why: the largest RAST-GRU repeat-to-cross-seed SD ratio is \FastCudnnRASTMaxRatio{}, whereas the audited Transformer-lite cells have a maximum fixed-seed repeat SD of \FastCudnnTransformerMaxRepeatSD{}. Because the error bars do not include a full runtime-replicate level, no transition is labeled ``resolved'' and the trajectories are not used for inferential attribution.

\begin{figure*}[t]
\centering
\bestgraphic[width=0.96\textwidth]{figures/fig_cross_backbone_protocol_buildup}
\caption{Descriptive five-seed cumulative protocol build-up on RAST-GRU and Transformer-lite for FD004. Open markers show the five composite-seed observations and solid lines connect their means. Each transition changes one declared protocol component while retaining the same split, sensors, window, optimizer, and budget. The points condition on one fast-cuDNN runtime per seed and do not cover the runtime nondeterminism found by the repeat audit; non-monotone trajectories are retained.}
\label{fig:cross-backbone-protocol-buildup}
\end{figure*}
\FloatBarrier

\subsection{RQ2: How do Core and Asym encode decision preference?}

With the OCM backbone and the common protocol held fixed, the Base-only stage has FD004 RMSE/NASA/engine/LPR \OCMBaseOnlyRMSE/\OCMBaseOnlyNASA/\OCMBaseOnlyLPR. Adding the smooth risk term changes these values to \OCMPlusRiskRMSE/\OCMPlusRiskNASA/\OCMPlusRiskLPR; adding consistency gives \OCMPlusRiskConsRMSE/\OCMPlusRiskConsNASA/\OCMPlusRiskConsLPR; and the locked Full model records \OCMFullRMSE/\OCMFullNASA/\OCMFullLPR. Reliability supervision and the quantile terms enter together in the final sequential step, so that step is a combined incremental block rather than separate evidence for either objective. Paired intervals, rather than monotone mean improvement, govern interpretation. This build-up complements one-at-a-time removal and exposes interactions among the OCM-specific objectives.

Asymmetric-error-cost effects also separate the two operating points. Over the six $\kappa\in\{5,10\}$ contrasts against RAST-GRU, Transformer-lite, and Dual-Mixer, Core has \CoreHighCostLowerCount{} intervals supporting lower error cost, \CoreHighCostWorseCount{} supporting higher error cost, and \CoreHighCostUnresolvedCount{} unresolved. Asym has \AsymHighCostLowerCount{}, \AsymHighCostWorseCount{}, and \AsymHighCostUnresolvedCount{}, respectively. Because the Dual-Mixer contrasts also change architecture and recommended protocol, and $A_\kappa$ contains no maintenance action model, these counts describe preference-dependent predictive operating points rather than a pure model or deployment-cost effect.

\input{generated/table_core_asym_cmapss.tex}
\FloatBarrier

\subsubsection{Late-error tails and module evidence}

The Asym frequency result is accompanied by lower late-error magnitude. OCM-MST-GRU-Asym records conditional MLE \OCMConditionalMLE\ and late-error CVaR$_{0.95}$ \OCMLateCVAR, compared with \RASTConditionalMLE\ and \RASTLateCVAR\ for RAST-GRU. These are the lowest macro tail values in the controlled model suite.

The five-seed FD004 ablation identifies two clear mechanisms and several mixed ones. Removing condition normalization increases RMSE by \NoConditionRMSEDelta\ cycles and SLPR@30,10 by \NoConditionSLPRTenDelta. Removing the mask changes RMSE by \NoMaskRMSEDelta\ cycles but raises LPR@30 by \NoMaskLPRDelta, exposing an accuracy--risk exchange. Removing the smooth late-risk term raises LPR@30 by \NoSmoothRiskLPRDelta. The no-asymmetry row improves all four reported FD004 core means, so the asymmetric multiplier is not claimed as beneficial and is subjected to a separately labeled simplification audit. Channel gating, reliability supervision, and the quantile auxiliary loss have metric-dependent effects and are not described as independently beneficial.

The complete 13-row ablation table, including reliability and quantile auxiliaries as separate switches, is reported in the Supplementary Information.
\FloatBarrier

\subsection{RQ3: How does the declared error preference behave under simulated sensor faults?}

Across nine standardized-space perturbation generators, the Supplementary Information reports crossed seed--perturbation-seed--engine resampling diagnostics. The manuscript interpretation rests on observed family-level point directions and their heterogeneity, not on diagnostic-envelope exclusion. These are controlled perturbation-time accuracy--late-overprediction contrasts, not evidence of physical-fault robustness. Global and window-random missingness remain difficult. Descriptive within-margin probabilities are retained only as threshold sensitivity and do not define or test engineering equivalence.

\input{generated/table_stress_operating_points.tex}

The LPR result does not imply a uniform asymmetric-error-cost advantage. At $\kappa=5$, no simulated fault-family interval supports lower Asym error cost, correlated-group missingness, global missingness, and window-random missingness support lower RAST-GRU error cost, and six families remain unresolved. The same direction count is three RAST-lower and six unresolved at $\kappa=2$; at $\kappa=10$, two families support Asym, two support RAST-GRU, and five remain unresolved. Signed bias and early-error magnitude show that reduced late-event frequency can be offset by conservative underestimation or by a smaller number of large positive errors. Core is more balanced but is also scenario-dependent.

\input{generated/table_stress_decision_cost_rho.tex}

\subsection{RQ4: How stable are the findings on another complex-condition C-MAPSS task and an exploratory N-CMAPSS task?}

The post-lock FD002 protocol-transfer grid holds the official endpoint task, split, target, sensors, window, optimizer, budget, and five composite seeds fixed. It compares RAST-GRU and OCM-MST-GRU under both Core and Asym objectives and adds Transformer-lite under Core. Table~\ref{tab:v3-fd002-protocol-transfer} reports every operating-point mean; the Supplementary Information reports all 20 engine-clustered paired intervals rather than selecting favorable metrics. Because this confirmation was designed after inspection of the primary study, it tests a declared transfer question but does not retroactively make FD002 an independent preregistered validation task.

\input{generated/table_fd002_protocol_transfer.tex}
Across the eight within-backbone Core--Asym metric contrasts, \FDTwoProtocolCoreLowerCount{} intervals support Core, \FDTwoProtocolAsymLowerCount{} support Asym, and \FDTwoProtocolUnresolvedCount{} are unresolved. Across the 12 OCM--reference architecture/protocol contrasts, \FDTwoArchitectureOCMLowerCount{} support OCM, \FDTwoArchitectureReferenceLowerCount{} support the matched reference, and \FDTwoArchitectureUnresolvedCount{} are unresolved. These metric-specific directions are interpreted as transfer of an accuracy--risk tension only when the full interval table supports it; they are not collapsed into a win rate or a universal protocol claim.
Only two of the 20 nominal percentile intervals exclude zero, and neither concerns RMSE, NASA/engine, or LPR@30. For asymmetric error cost at $\kappa=5$, OCM-Core minus OCM-Asym is \FDTwoOCMCoreMinusAsymCost{} [\FDTwoOCMCoreMinusAsymCostLow{}, \FDTwoOCMCoreMinusAsymCostHigh{}], whereas OCM-Core minus RAST-Core is \FDTwoOCMCoreMinusRASTCoreCost{} [\FDTwoOCMCoreMinusRASTCoreCostLow{}, \FDTwoOCMCoreMinusRASTCoreCostHigh{}]. After joint Holm correction over all 20 exploratory tests, \FDTwoHolmSignificantCount{} contrast remains significant. Thus FD002 does not confirm stable transfer of the FD004 late-frequency pattern; it only shows that task and the declared error utility can reorder selected operating points.
\FloatBarrier

On the fixed N-CMAPSS DS02 task, the complete supplementary table reports both OCM-Asym and post-lock OCM-Core. Relative to RAST-GRU, Asym has lower unit-macro RMSE and LPR point estimates, but neither operating point is uniformly best among all controlled baselines. More importantly, validation-only selection chooses $K=4$, whereas the post-lock test-best RMSE occurs at $K=8$. DS02 is therefore exploratory evidence of a decision-dependent trade-off, not stable external confirmation.

The Supplementary Information per-unit audit exposes heterogeneity rather than treating 12,391 overlapping windows as independent evidence. A second supplementary check rotates all six development units (2, 5, 10, 16, 18, and 20): one unit is held out for exploratory testing, the next declared unit is used for validation, and the remaining four are used for training. Official test units 11, 14, and 15 are excluded from every rotation split. This is within-task development-split sensitivity, not an expansion of the independent external test sample.

\IfFileExists{generated/extended_review_results_en.tex}{\input{generated/extended_review_results_en.tex}}{}

\FloatBarrier

\subsection{Supporting mechanism and design diagnostics}

Mechanism diagnostics are informative but bounded. Across missing rates, the soft reliability weighting has fault-localization AUROC \ReliabilityAUROCLow--\ReliabilityAUROCHigh, Brier score \ReliabilityBrierLow--\ReliabilityBrierHigh, and ten-bin ECE \ReliabilityECELow--\ReliabilityECEHigh. The slope/intercept and bin counts are reported in the Supplementary Information. This observed-fraction diagnostic is insufficient to interpret the weights as calibrated physical fault probabilities or as a stand-alone detector. Temporal attention places \CriticalAttentionMass\ of its mass in the final third of critical windows versus \NoncriticalAttentionMass\ outside the critical region. Under engine-grouped linear probes, condition normalization reduces regime balanced accuracy from 1.000 to 0.153 while increasing critical-stage balanced accuracy from 0.885 to 0.939 and continuous-RUL Ridge $R^2$ from 0.542 to 0.673. These are representation diagnostics, not causal module effects.

The predeclared 30-cycle window is not uniquely optimal. A 50-cycle window changes FD004 RMSE from \ReferenceDesignRMSE\ to \WindowFiftyRMSE\ and LPR@30 from \ReferenceDesignLPR\ to \WindowFiftyLPR. This sensitivity is retained rather than used for post-hoc retuning of the main grid.

The TCN-GRU baseline audit retains all 20 subset--seed cells. FD001 seed 2025 and FD003 seed 2026 are flagged because RMSE exceeds 50 cycles and twice the corresponding five-seed median. Neither cell contains NaN or infinite predictions, and gradient clipping at max norm 5 was active. In both cells, the locked risk-score rule selected an early conservative checkpoint (epochs 1 and 2) while ordinary validation loss continued to decrease. The failure is therefore reported as an architecture--selection interaction rather than removed as a bad seed; medians, interquartile ranges, and complete seed values are provided in the Supplementary Information.


\subsection{Closest-prior context}

The architecture-adapted FD002 regime dual-attention comparator has higher mean error than OCM-MST-GRU, but its paired MAE interval crosses zero and its LPR difference is negligible. It is a targeted common-protocol comparator, not an exact reproduction of third-party code. The result narrows the novelty claim to the integrated risk, missingness, auxiliary interval-diagnostic, and evidence protocol.

The official-code-derived Dual-Mixer reference in Table~\ref{tab:v3-official-dual-mixer} follows the authors' public layer equations and recommended window, sensors, scaling, width, depth, dropout, loss, and early stopping while retaining this study's locked engine split. The table names Core and Asym explicitly and applies Holm adjustment over the 12 subset--metric contrasts within each operating point. For the predeclared Asym comparison, RMSE intervals support lower Dual-Mixer in \OfficialDualRMSELowerCount{} task, lower Asym in \OfficialOCMRMSELowerCount{}, and are unresolved in \OfficialRMSEUnresolvedCount{}; Asym has lower LPR@30 in all \OfficialOCMLPRLowerCount{} tasks, while NASA/engine is resolved for Asym in only \OfficialOCMNASALowerCount{}. This is a stronger published-code reference than a locally invented recurrent variant, but the paired comparison includes both architecture and recommended-protocol differences and cannot isolate an architecture effect.

\input{generated/table_official_dual_mixer_summary.tex}

\FloatBarrier
"""
    chinese = r"""\input{generated/numbers.tex}
\FloatBarrier
\section{实验结果}

\subsection{RQ1：是否存在脱离协议仍成立的架构优势？}

在 FD001--FD004 上，OCM-MST-GRU-Asym 的宏平均 RMSE、MAE、每发动机 NASA Score 和 LPR@30 分别为 \OCMMacroRMSE、\OCMMacroMAE、\OCMMacroNASAperEngine 和 \OCMMacroLPRThirty。相应锁定基线的最低值为 \BestBaselineMacroRMSE（\BestBaselineMacroRMSEName）、\BestBaselineMacroMAE（\BestBaselineMacroMAEName）、\BestBaselineMacroNASAperEngine（\BestBaselineMacroNASAperEngineName）和 \BestBaselineMacroLPRThirty（\BestBaselineMacroLPRThirtyName）。Asym 没有取得最低点预测误差，但在五次种子级分析中有 \OCMParetoSeedCount 次进入描述性的精度--风险 Pareto 集；主表将锁定后的 Core 确认结果单独标明。

本文的主要不确定性对象是每个固定 C-MAPSS 子集内部的发动机级配对效应。表~\ref{tab:v3-fixed-subset-direct-zh} 显示，直接前身比较的 RMSE 与 NASA/engine 区间在四个任务中均跨 0，FD001--FD003 的 LPR 区间也跨 0；只有 FD004 的 LPR 差异明确有利于 OCM-MST-GRU。因此，四子集等权宏平均和跨子集分层区间只作为描述性汇总，不解释为任务总体推断。

\input{generated/table_cmapss_macro_zh.tex}
\input{generated/table_fixed_subset_direct_zh.tex}
\input{generated/table_core_rast_fixed_summary_zh.tex}
对称摘要赋予 Core 和 Asym 相同的统计地位：Core 仅在 FD004 LPR 上明确更低，RAST-GRU 仅在 FD003 NASA/engine 上明确更低，其余十个 Core 对照未解决；Asym 也只有 FD004 LPR 明确有利。因此，两个运行点都没有形成跨指标的固定任务架构优势。
\FloatBarrier

\subsubsection{检查点规则敏感性}

锁定风险评分规则下，FD004 的 RMSE/LPR@30 为 \RiskCheckpointRMSE/\RiskCheckpointLPRThirty；改用纯验证 RMSE 检查点后变为 \RMSECheckpointRMSE/\RMSECheckpointLPRThirty。端点审计揭示部分 subset--seed 单元的 $N_{30}$ 很小；$N_{10}$ 不是 SLPR@30,10 的分母。补充材料中的三设计确认通过增加发动机平衡的临界端点检验结论对端点分布的依赖。风险检查点只解释为经过敏感性检验的可选规则，不解释为已确认优势或最优决策规则。

端点分布审计进一步表明，uniform-single 与测试端点分布的 Wasserstein 距离最小（FD002 为 \EndpointUniformWFDTwo，FD004 为 \EndpointUniformWFDFour）。fixed-multi 和 critical-stratified 将约一半验证端点置于 RUL $\leq30$，而测试端点对应比例仅为 \EndpointTestCriticalFDTwo 和 \EndpointTestCriticalFDFour。因此，它们是有意加强临界区权重的设计，不是对自然测试端点分布的更好估计。补充材料按模型对和端点设计汇总六个固定任务--指标对照的区间方向；均值变化本身不视为确认性证据。

完整的 30 条训练轨迹确认使这一限制更清楚，但没有推翻它。对每个 OCM--RAST 模型对，uniform-single 下六个固定任务对照仅有一个区间得到解决，两种重加权设计下各有两个；所有得到解决的 OCM--RAST 对照均支持 OCM 运行点，且主要体现在 LPR@30，但多数对照仍未解决。在 Core--Asym 比较中，FD004 的 RMSE 在三种端点设计下均明确支持 Core，FD002 的 RMSE 则仅在两种重加权设计下明确支持 Core；只有 FD004 的 critical-stratified LPR@30 明确支持 Asym。因此，增加临界区支持能够提高部分精度--风险差异的可辨识度，却不能证明存在普遍最优的端点设计或运行点。

完整检查点敏感性表移至补充材料。
\FloatBarrier

\subsubsection{共同风险协议与标准化常规协议}

Track B 在保持划分、目标、模型族、优化器、预算和种子不变的情况下，同时改变预处理、输入、损失、增强和检查点选择这一组已声明设置。在 \ProtocolTotalComparisonCount 个“模型--指标”配对比较中，95\% 区间有 \ProtocolConventionalResolvedCount 个支持标准化常规协议、\ProtocolCommonResolvedCount 个支持共同风险协议，其余 \ProtocolUnresolvedCount 个尚未解析。绝对结果进一步收紧了架构结论：Common Base 下，OCM backbone 减 RAST 的 RMSE、LPR 和 CVaR95 差值分别为 \CommonBaseArchitectureRMSEDelta、\CommonBaseArchitectureLPRDelta 和 \CommonBaseArchitectureCVaRDelta，即该主干付出点误差代价并小幅降低晚预测风险；Track B 内 OCM 的 RMSE、NASA/engine 和 LPR 排名分别为 \TrackBOCMRMSERank/6、\TrackBOCMNASARank/6 和 \TrackBOCMLPRRank/6。因此证据不支持脱离协议的独立架构优势，而支持共同风险协议下的集成精度--风险配置。由于 Track B 同时改变多项设置，任一轨道都不能被当作与协议无关的原生排行榜。

双骨干累计审计仅保留为描述性敏感性分析。均值轨迹呈非单调，说明协议组件会与骨干和目标交互，不能把最终组合拆解成“每个新增组件都独立有益”的结论。补充材料中的逐转变区间以每个复合种子的一次 fast-cuDNN 运行作为条件，只是诊断量而非确认性区间。固定 seed 重跑审计说明了原因：RAST-GRU 最大“固定 seed 重跑标准差/原五 seed 标准差”为 \FastCudnnRASTMaxRatio{}，所审计 Transformer-lite 单元的最大固定 seed 重跑标准差为 \FastCudnnTransformerMaxRepeatSD{}。由于误差条没有纳入完整的 runtime replicate 层级，本文不再把任何转变标记为“明确改善/恶化”，也不据此作推断归因。

\begin{figure*}[t]
\centering
\bestgraphic[width=0.96\textwidth]{figures/fig_cross_backbone_protocol_buildup}
\caption{RAST-GRU 与 Transformer-lite 在 FD004 上的五种子累计协议构建。空心点给出五个复合种子的观测，实线连接其均值。每次转变只改变一个声明的协议成分，并保持数据划分、传感器、窗口、优化器和预算一致。各点仅对应每个种子的一次 fast-cuDNN 运行，不覆盖固定种子重跑审计发现的运行时波动；非单调轨迹被完整保留。}
\label{fig:cross-backbone-protocol-buildup-zh}
\end{figure*}
\FloatBarrier

\subsection{RQ2：Core 与 Asym 如何编码决策偏好？}

保持 OCM 主干和共同协议不变时，Base-only 阶段的 FD004 RMSE/NASA/engine/LPR 为 \OCMBaseOnlyRMSE/\OCMBaseOnlyNASA/\OCMBaseOnlyLPR；加入平滑风险项后为 \OCMPlusRiskRMSE/\OCMPlusRiskNASA/\OCMPlusRiskLPR；继续加入一致性项后为 \OCMPlusRiskConsRMSE/\OCMPlusRiskConsNASA/\OCMPlusRiskConsLPR；锁定 Full 模型为 \OCMFullRMSE/\OCMFullNASA/\OCMFullLPR。可靠性监督与分位数项在最后一步共同加入，因此该步是联合增量块，不能分别归因给任一目标。解释以配对区间为准，不要求均值随模块加入单调改善。该逐步构建实验与 one-at-a-time removal 互补，用于暴露 OCM 特有目标项之间的交互。

非对称误差成本进一步区分两个运行点。在相对 RAST-GRU、Transformer-lite 和 Dual-Mixer 的六个 $\kappa\in\{5,10\}$ 对照中，Core 有 \CoreHighCostLowerCount{} 个区间支持更低误差成本、\CoreHighCostWorseCount{} 个支持更高误差成本、\CoreHighCostUnresolvedCount{} 个未解决；Asym 对应为 \AsymHighCostLowerCount{}、\AsymHighCostWorseCount{} 和 \AsymHighCostUnresolvedCount{}。由于 Dual-Mixer 对照同时改变架构和推荐协议，且 $A_\kappa$ 不包含维修动作模型，这些计数只描述依赖误差偏好的预测运行点，不解释为纯模型效应或部署成本。

\input{generated/table_core_asym_cmapss_zh.tex}
\FloatBarrier

\subsubsection{晚预测尾部与模块证据}

Asym 的比例改善同时伴随晚预测幅度下降。OCM-MST-GRU-Asym 的条件 MLE 和晚误差 CVaR$_{0.95}$ 分别为 \OCMConditionalMLE 和 \OCMLateCVAR，RAST-GRU 对应为 \RASTConditionalMLE 和 \RASTLateCVAR；前者是统一模型集合中最低的宏平均尾部值。

FD004 五种子消融给出了两项清晰机制和若干混合效应。去掉工况归一化后，RMSE 增加 \NoConditionRMSEDelta cycles，SLPR@30,10 增加 \NoConditionSLPRTenDelta。去掉缺失掩码会使 RMSE 变化 \NoMaskRMSEDelta cycles，却使 LPR@30 增加 \NoMaskLPRDelta，表明点误差与晚预测风险存在交换。去掉平滑晚预测风险项使 LPR@30 增加 \NoSmoothRiskLPRDelta。去除非对称项后，FD004 四个核心均值全部改善，因此本文不声称该 multiplier 有益，并对简化候选作单独审计。通道门控、可靠性监督和分位数辅助损失的效应随指标变化，不分别描述为稳定增益。

完整 13 行消融表移至补充材料，其中可靠性监督与分位数辅助损失作为独立开关报告。
\FloatBarrier

\subsection{RQ3：声明的误差偏好在模拟传感器故障下如何变化？}

九类标准化空间故障的主要三级配对区间同时重采样训练种子、匹配扰动种子和匹配 FD004 测试发动机。Asym 的 LPR 中，\AsymStressLPROCMLowerCount 类支持 Asym 更低、\AsymStressLPRRASTLowerCount 类支持 RAST-GRU 更低、\AsymStressLPRUnresolvedCount 类未解决；Asym 的 RMSE 对应为 \AsymStressRMSEOCMLowerCount、\AsymStressRMSERASTLowerCount 和 \AsymStressRMSEUnresolvedCount。Core 在同一主表中对称报告。因此证据表明的是退化条件下的精度--风险折中，而不是鲁棒性优势；候选等效边界只作敏感性分析。

\input{generated/table_stress_operating_points_zh.tex}

LPR 结果并不等价于统一的非对称误差成本优势。在 $\kappa=5$ 时，没有模拟故障族区间支持 Asym 误差成本更低；相关传感器组缺失、全局缺失和窗口随机缺失三类支持 RAST-GRU 更低，其余六类未解决。$\kappa=2$ 时同样为三类支持 RAST、六类未解决；$\kappa=10$ 时两类支持 Asym、两类支持 RAST、五类未解决。有符号偏差和提前误差幅度表明，较低的晚预测事件频率可能被更保守的低估或少量更大的正误差抵消。Core 更为平衡，但同样依赖具体场景。

\input{generated/table_stress_decision_cost_rho_zh.tex}

\subsection{RQ4：结论在另一复杂工况 C-MAPSS 任务和探索性 N-CMAPSS 任务上有多稳定？}

\IfFileExists{generated/extended_review_results_zh.tex}{\input{generated/extended_review_results_zh.tex}}{}

锁定后的 FD002 协议迁移网格固定官方端点任务、数据划分、目标、传感器、窗口、优化器、预算和五个复合种子，在 Core 与 Asym 两种目标下对照 RAST-GRU 与 OCM-MST-GRU，并在 Core 下加入 Transformer-lite。表~\ref{tab:v3-fd002-protocol-transfer-zh} 完整报告五个运行点均值；补充材料保留全部 20 个发动机簇配对区间，而不筛选有利指标。由于该确认是在查看主研究后设计的，它只能检验明确声明的迁移问题，不能追溯地把 FD002 变成独立预注册验证任务。

\input{generated/table_fd002_protocol_transfer_zh.tex}
在 8 个同主干 Core--Asym 指标对照中，\FDTwoProtocolCoreLowerCount{} 个区间支持 Core 更低，\FDTwoProtocolAsymLowerCount{} 个支持 Asym 更低，\FDTwoProtocolUnresolvedCount{} 个未解决。在 12 个 OCM--参考点架构/协议对照中，\FDTwoArchitectureOCMLowerCount{} 个支持 OCM 更低，\FDTwoArchitectureReferenceLowerCount{} 个支持匹配参考点更低，\FDTwoArchitectureUnresolvedCount{} 个未解决。只有完整区间表支持时，这些随指标变化的方向才被解释为精度--风险张力的迁移；本文不把它们压缩成胜率或普遍协议结论。
20 个名义百分位区间中只有 2 个排除 0，且都不涉及 RMSE、NASA/engine 或 LPR@30。在 $\kappa=5$ 非对称误差成本上，OCM-Core 减 OCM-Asym 为 \FDTwoOCMCoreMinusAsymCost{} [\FDTwoOCMCoreMinusAsymCostLow{}, \FDTwoOCMCoreMinusAsymCostHigh{}]，OCM-Core 减 RAST-Core 为 \FDTwoOCMCoreMinusRASTCoreCost{} [\FDTwoOCMCoreMinusRASTCoreCostLow{}, \FDTwoOCMCoreMinusRASTCoreCostHigh{}]。对全部 20 个探索性检验联合实施 Holm 校正后，有 \FDTwoHolmSignificantCount{} 个对照仍显著。因此 FD002 没有确认 FD004 晚预测频率模式能够稳定迁移，只能说明任务与声明的误差效用可能改变所选运行点。
\FloatBarrier

固定 N-CMAPSS DS02 任务上，补充材料的完整表同时报告 OCM-Asym 与锁定后的 OCM-Core。Asym 相对 RAST-GRU 的 unit-macro RMSE 和 LPR 点估计更低，但两个运行点都没有在所有受控基线上形成统一优势。更重要的是，仅基于验证证据的规则选择 $K=4$，而锁定后测试 RMSE 最优值出现在 $K=8$。因此 DS02 只能作为探索性的决策偏好折中证据，不能作为稳定外部确认。

补充材料的逐 unit 表主动展示异质性，不把 12,391 个重叠窗口当作独立证据。另一项补充检查轮换全部六个开发单元（2、5、10、16、18 和 20）：每次留出一个开发单元作探索性测试，序列中的下一个声明单元作验证，其余四个用于训练；官方测试单元 11、14 和 15 在所有轮换划分中均被排除。该实验只检验任务内开发划分敏感性，不扩充独立外部测试样本。


表~\ref{tab:v3-official-dual-mixer} 给出官方代码派生的 Dual-Mixer 参考结果，明确区分 Core 与 Asym，并在每个运行点内部对 12 个“子集--指标”对照实施 Holm 校正。该运行保留作者公开的层结构与推荐窗口、传感器、缩放、宽深度、损失和早停设置，同时使用本文锁定的发动机划分。对于预声明的 Asym，RMSE 区间在 \OfficialDualRMSELowerCount{} 个任务支持 Dual-Mixer 更低、\OfficialOCMRMSELowerCount{} 个任务支持 Asym 更低，其余 \OfficialRMSEUnresolvedCount{} 个未解决；Asym 的 LPR@30 在全部 \OfficialOCMLPRLowerCount{} 个任务明确更低，而 NASA/engine 仅在 \OfficialOCMNASALowerCount{} 个任务明确支持 Asym。它提供了比自定义循环变体更强的公开代码参考，但配对差异同时包含架构与推荐协议差异，不能解释为纯架构归因。
\input{generated/table_official_dual_mixer_summary_zh.tex}
\FloatBarrier

\subsection{补充机制与设计诊断}

机制诊断提供的是有限支持。不同缺失率下，软可靠性加权的故障定位 AUROC 为 \ReliabilityAUROCLow 至 \ReliabilityAUROCHigh，Brier score 为 \ReliabilityBrierLow 至 \ReliabilityBrierHigh，十箱 ECE 为 \ReliabilityECELow 至 \ReliabilityECEHigh；斜率、截距和每箱样本数在补充材料报告。该观测比例诊断不足以把软权重解释为已校准的物理故障概率或独立故障检测器。按发动机分组的线性探针显示，工况归一化使工况 balanced accuracy 从 1.000 降至 0.153，同时使临界退化阶段 balanced accuracy 从 0.885 升至 0.939，连续 RUL Ridge $R^2$ 从 0.542 升至 0.673；这些只是表示诊断，不是因果模块效应。

预先声明的 30-cycle 窗口并非唯一最优设计。将窗口改为 50 后，FD004 RMSE 从 \ReferenceDesignRMSE 变为 \WindowFiftyRMSE，LPR@30 从 \ReferenceDesignLPR 变为 \WindowFiftyLPR。本文保留这一敏感性，不据此对主实验做事后调参。

TCN-GRU 基线审计保留全部 20 个“子集--种子”单元。FD001 seed 2025 与 FD003 seed 2026 因 RMSE 超过 50 cycles 且高于对应五种子中位数的两倍而被标记；两者均无 NaN/Inf，且训练始终启用 max norm 5 的梯度裁剪。两个异常单元中，锁定风险评分规则分别选择 epoch 1 和 2 的早期保守检查点，而普通验证损失此后仍继续下降。因此本文将其报告为“架构与选择规则的交互失稳”，不把它们删除为坏种子；完整种子值、中位数与四分位距见补充材料。


\subsection{近期最接近方法的对照}

FD002 工况双注意力比较器的平均误差高于 OCM-MST-GRU，但配对 MAE 区间跨 0，LPR 差异也很小。该结果是统一协议下的结构适配比较，不是对第三方代码的精确复现；它进一步把本文的新颖性边界限定为风险、缺失感知、辅助区间诊断和证据协议的集成。

\FloatBarrier
"""
    return english, chinese


def write_results_sections(output: Path) -> None:
    release_primary = PROJECT_ROOT / "paper_outputs" / "release_v1" / "primary_fixed_task_estimates.csv"
    release_templates = (
        PROJECT_ROOT / "templates" / "results_section_en_release_v1.tex",
        PROJECT_ROOT / "templates" / "results_section_zh_release_v1.tex",
    )
    if release_primary.exists() and all(path.exists() for path in release_templates):
        for language, template in zip(("en", "zh"), release_templates):
            (output / f"results_section_{language}.tex").write_text(
                template.read_text(encoding="utf-8"),
                encoding="utf-8",
                newline="\n",
            )
        return

    english, chinese = current_results_section_text()
    (output / "results_section_en.tex").write_text(english, encoding="utf-8", newline="\n")
    (output / "results_section_zh.tex").write_text(chinese, encoding="utf-8", newline="\n")


def write_round2_condensed_results(output: Path, advanced: Path) -> None:
    """Write the semantic-release Results section when release evidence is available."""

    release_primary = PROJECT_ROOT / "paper_outputs" / "release_v1" / "primary_fixed_task_estimates.csv"
    release_templates = (
        PROJECT_ROOT / "templates" / "results_section_en_release_v1.tex",
        PROJECT_ROOT / "templates" / "results_section_zh_release_v1.tex",
    )
    if release_primary.exists() and all(path.exists() for path in release_templates):
        for language, template in zip(("en", "zh"), release_templates):
            (output / f"results_section_{language}.tex").write_text(
                template.read_text(encoding="utf-8"),
                encoding="utf-8",
                newline="\n",
            )
        return

    required = (
        "table_round2_primary_crossed.tex",
        "table_round2_primary_crossed_zh.tex",
        "table_round2_stress_holm.tex",
        "table_round2_stress_holm_zh.tex",
    )
    for name in required:
        source = advanced / name
        if not source.exists():
            raise FileNotFoundError(
                f"Missing round-2 evidence table: {source}. "
                "Run scripts/analyze_normalized_perturbation_evidence.py first."
            )
        shutil.copy2(source, output / name)

    english = r"""\input{generated/numbers.tex}
\FloatBarrier
\section{Results}

\subsection{[MAIN RESULT-INFORMED FIXED-TASK ESTIMATION] Fixed-task accuracy and late-overprediction}

The result-informed comparison is OCM-Asym minus its direct predecessor RAST-GRU on the four official C-MAPSS test-engine endpoint tasks. Training seed and engine identity are crossed factors: every composite seed predicts the same official engines. The revised diagnostic therefore draws composite-seed levels and matched engine IDs independently, applies the same engine draw to every sampled seed, and preserves model pairing. It does not pool repeated engine outcomes into an exact McNemar table.

\input{generated/table_round2_primary_crossed.tex}

The mean pattern remains an accuracy--late-overprediction exchange: relative to RAST-GRU, OCM-Asym changes equal-task macro RMSE from 14.989 to 15.112 cycles, NASA/engine from 4.418 to 4.064, and LPR@30 from 0.176 to 0.137. All multiplicity-adjusted diagnostic summaries remain unresolved; this does not accept a null effect or establish equivalence. FD004 LPR@30 has five same-sign seed effects and is reported descriptively, without a separate evidence category.

\begin{figure*}[!t]
\centering
\bestgraphic[width=0.92\textwidth]{figures/fig_round2_primary_crossed_forest}
\caption{[MAIN RESULT-INFORMED FIXED-TASK ESTIMATION] OCM-Asym minus RAST-GRU on four fixed tasks. Lines use crossed composite-seed $\times$ matched-engine resampling as diagnostic envelopes. Negative effects favor OCM-Asym; seed-level effects remain the primary visual information.}
\label{fig:round2-primary-crossed}
\end{figure*}
\FloatBarrier

\subsection{[SENSITIVITY] Predictive error preference}

OCM-Core and OCM-Asym share the same architecture and training protocol; only the late-error multiplier differs. Core has lower mean macro RMSE (14.732 versus 15.112 cycles), while Asym has lower mean NASA/engine (4.064 versus 4.530), LPR@30 (0.137 versus 0.159), and late-error CVaR$_{0.95}$ (3.999 versus 4.577). Their paired macro intervals include zero. These are researcher-declared predictive-error operating points, not maintenance-policy decisions or independently selected optima.

\input{generated/table_core_asym_cmapss.tex}

The LPR conclusion is not tied only to floating-point positivity. A crossed sensitivity grid evaluates $\epsilon\in\{0,0.5,1,2,5\}$ cycles and $\tau\in\{20,30,40,50\}$. On FD004 at $\tau=30$, the OCM-Asym minus RAST-GRU mean changes from $-0.117$ at $\epsilon=0$ to $-0.038$ at $\epsilon=5$; intervals become increasingly unresolved as the tolerance grows. LPR is therefore interpreted jointly with SLPR and late-error magnitude, not as a stand-alone safety endpoint. The complete 80-row grid is stored in \texttt{round2\_lpr\_tolerance\_sensitivity.csv}.

\subsection{[SENSITIVITY] Normalized-space perturbations}

The stress design crosses five training seeds, five perturbation seeds, and the same 248 FD004 test engines. Revised inference resamples all three factor levels independently and preserves matched engine and perturbation identities across operating points. One joint Holm family covers two OCM points, nine perturbation families, and three declared metrics (54 contrasts).

\input{generated/table_round2_stress_holm.tex}

After joint correction, no contrast supports a lower OCM response. Three mean-late-excess contrasts support lower RAST-GRU and the remaining 51 are unresolved. Unadjusted direction counts and heat-map colors are no longer used as support statements. The complete effect estimates, crossed intervals, centered-bootstrap values, adjusted $p$ values, and status labels are the machine-readable inferential record in \texttt{round2\_stress\_crossed\_holm.csv}. Because amplitudes are imposed after normalization, this section evaluates algorithmic sensitivity to normalized-space perturbations, not physical sensor faults or deployment robustness.

\subsection{[EXPLORATORY] Protocol, transfer, and representation audits}

The remaining analyses delimit rather than extend the primary claim. FD002 does not confirm the FD004 late-frequency pattern: none of its RMSE, NASA/engine, or LPR contrasts is resolved, and one asymmetric-error-utility contrast survives a separate 20-test Holm family. The fixed N-CMAPSS DS02 task has only three official test units; validation-only selection chooses $K=4$ while the observed test-best RMSE occurs at $K=8$. It is retained as a single-task failure analysis, not cross-dataset confirmation.

Condition normalization removes most linearly decodable operating-regime information and is the strongest removal ablation, but $K$-means exactly recovers the rounded six-state setting partition on this benchmark. Without matched direct-state and continuous-correction training controls, the gain is attributed only to condition-specific normalization, not to $K$-means itself. Reliability and attention outputs are likewise representation diagnostics: their modest localization/calibration values do not establish physical fault detection or causal mechanism.

A result-informed validation-only checkpoint diagnostic flags six finite cells across all configurations, including two TCN-GRU cells. All finite runs remain in the main benchmark; mean, median, 10\% trimmed mean, and diagnostic policy-excluded summaries are reported in the Supplementary Information. The failure documents checkpoint-policy sensitivity rather than a corrected ranking.
\FloatBarrier
"""

    chinese = r"""\input{generated/numbers.tex}
\FloatBarrier
\section{实验结果}

\subsection{[主要冻结证据] 固定任务的精度与晚高估}

冻结后报告的主要比较是在四个 C-MAPSS 官方测试发动机端点任务上计算 OCM-Asym 减直接前身 RAST-GRU。每个复合种子都预测同一组官方发动机，因此训练种子与发动机 ID 是交叉因子。修订后的推断分别抽取复合种子水平和匹配发动机 ID，并把同一发动机抽样应用于所有入选种子；不再把跨种子重复出现的发动机结果合并进 exact McNemar 表。

\input{generated/table_round2_primary_crossed_zh.tex}

均值仍表现为精度--晚高估交换：相对 RAST-GRU，OCM-Asym 使等任务宏平均 RMSE 从 14.989 变为 15.112 cycles、NASA/engine 从 4.418 变为 4.064、LPR@30 从 0.176 变为 0.137。但按声明的“每指标四任务”Holm 家族校正后，12 个固定任务对照全部未解决。FD004 LPR@30 的交叉 95\% 区间为负，但校正后 $p=0.058$，故只报告为提示性结果，不作为确认性发现。

\begin{figure*}[!t]
\centering
\bestgraphic[width=0.92\textwidth]{figures/fig_round2_primary_crossed_forest}
\caption{[主要冻结证据] 四个固定任务上的 OCM-Asym 减 RAST-GRU。区间采用交叉“复合种子 $\times$ 匹配发动机”重采样；负值支持 OCM-Asym，标记状态由指标内 Holm 决策确定，而不是只依据未校正区间。}
\label{fig:round2-primary-crossed-zh}
\end{figure*}
\FloatBarrier

\subsection{[锁定后敏感性] 预测误差偏好}

OCM-Core 与 OCM-Asym 使用相同架构和训练协议，仅晚高估 multiplier 不同。Core 的宏平均 RMSE 更低（14.732 对 15.112 cycles）；Asym 的 NASA/engine（4.064 对 4.530）、LPR@30（0.137 对 0.159）和晚误差 CVaR$_{0.95}$（3.999 对 4.577）更低，但相应宏平均配对区间均跨 0。它们是研究者声明的预测误差运行点，不是维修策略决策或经独立数据选择的最优点。

\input{generated/table_core_asym_cmapss_zh.tex}

LPR 结论不只依赖浮点数正误差判定。交叉敏感性网格覆盖 $\epsilon\in\{0,0.5,1,2,5\}$ cycles 与 $\tau\in\{20,30,40,50\}$。FD004、$\tau=30$ 时，OCM-Asym 减 RAST-GRU 的均值从 $\epsilon=0$ 的 $-0.117$ 变为 $\epsilon=5$ 的 $-0.038$，容差增大后区间更趋于未解决。因此，LPR 必须与 SLPR 和晚误差幅度共同解释，不能单独作为安全终点。完整 80 行结果位于 \texttt{round2\_lpr\_tolerance\_sensitivity.csv}。

\subsection{[锁定后敏感性] 归一化空间扰动}

压力测试交叉五个训练种子、五个扰动种子和相同的 248 台 FD004 测试发动机。修订后的推断独立重采样三个因子水平，同时维持运行点之间匹配的发动机与扰动身份。一个联合 Holm 家族覆盖两个 OCM 运行点、九类扰动和三个声明指标，共 54 项。

\input{generated/table_round2_stress_holm_zh.tex}

联合校正后，没有对照支持 OCM 响应更低；3 个平均晚超量对照支持 RAST-GRU 更低，其余 51 项未解决。未校正方向计数和热图颜色不再承担支持性结论。完整效应量、交叉区间、中心化 bootstrap 值、校正 $p$ 值和状态标签存于 \texttt{round2\_stress\_crossed\_holm.csv}。由于扰动在归一化后施加，本节只评价算法对归一化空间扰动的敏感性，不评价物理传感器故障或部署鲁棒性。

\subsection{[探索性] 协议、迁移与表示审计}

其余分析用于限定而非扩展主要结论。FD002 未确认 FD004 的晚高估频率模式：RMSE、NASA/engine 和 LPR 对照均未解决，只有一个非对称误差效用对照通过独立 20 项 Holm 家族。固定 N-CMAPSS DS02 任务只有三个官方测试 unit；仅验证集规则选择 $K=4$，而观察到的测试 RMSE 最优值在 $K=8$。因此它只作为单任务失败分析，不作为跨数据集确认。

工况归一化移除了大部分可线性解码的工况信息，也是最强 removal ablation，但 $K$-means 在该基准上与四舍五入的六状态 setting 划分完全一致。在缺少匹配的直接状态分组和连续校正训练对照时，收益只能归于“按工况归一化”，不能归于 $K$-means 本身。可靠性和注意力输出也只作表示诊断；现有定位与校准数值不足以证明物理故障检测或因果机制。

两个 TCN-GRU 单元满足预先声明的 RMSE $>50$ cycles 诊断标记。保留它们时，TCN-GRU 在 RMSE、NASA/engine 和 LPR 上的平均秩为 10.20、9.05 和 6.20；仅排除标记单元后变为 9.89、8.61 和 5.61。总体基准背景变化有限，但该失稳仍说明架构与检查点规则不兼容。完整消融、端点设计、协议构建、官方代码派生对照、运行时审计、tidy 行级输出与 schema 均移至补充材料和复现包。
\FloatBarrier
"""

    chinese = (PROJECT_ROOT / "templates" / "results_section_zh_round2.tex").read_text(
        encoding="utf-8"
    )
    (output / "results_section_en.tex").write_text(english, encoding="utf-8", newline="\n")
    (output / "results_section_zh.tex").write_text(chinese, encoding="utf-8", newline="\n")
    return


def write_main_analysis_family_registry(output: Path, analysis_working: Path) -> None:
    registry_path = analysis_working / "multiplicity_registry.csv"
    if not registry_path.exists():
        raise FileNotFoundError(
            "Main analysis-family registry requires "
            "paper_outputs/analysis_working/multiplicity_registry.csv"
        )
    registry = pd.read_csv(registry_path)
    expected = {
        "E-FIXED-12",
        "S-ENDPOINT-80",
        "S-PERTURB-54",
        "D-NCMAPSS-ROT",
        "D-STREAM10-FIXED",
        "D-CROSSED-3X3",
        "D-CMAPSS-PREP5",
        "D-PREF-REFIT-3X3",
        "D-SHARED-FACT-8",
        "D-LOFO-4",
        "D-NCMAPSS-SW9",
        "D-NCMAPSS-H3",
    }
    missing = expected - set(registry["family_id"])
    if missing:
        raise ValueError(f"Multiplicity registry is missing main families: {sorted(missing)}")

    rows = [
        (
            "E-FIXED-12",
            "formed after earlier result review; result-informed",
            "5 coupled composite-seed levels $\\times$ matched engines; fixed tasks",
            "paired finite-design effects, signs, and event support; resampling diagnostics in Supplement",
            "fixed-task estimates; no confirmatory language",
        ),
        (
            "S-ENDPOINT-80",
            "retrospective; result-informed",
            "5 coupled composite-seed levels $\\times$ matched engines; fixed tasks",
            "threshold grid, signs, and event support; resampling diagnostics in Supplement",
            "threshold sensitivity only",
        ),
        (
            "S-PERTURB-54",
            "retrospective; result-informed",
            "5 coupled composite-seed levels $\\times$ 5 perturbation seeds $\\times$ engines",
            "complete normalized-coordinate direction family; resampling diagnostics in Supplement",
            "normalized-coordinate sensitivity conditional on the declared mixture",
        ),
        (
            "D-NCMAPSS-ROT",
            "retrospective; result-informed",
            "fixed DS02 development/test units",
            "descriptive rotation and prior checks",
            "strongly downsampled computational proxy only",
        ),
        (
            "D-STREAM10-FIXED",
            "retrospective; result-informed",
            "one fixed split $\\times$ 10 separately seeded streams per task",
            "finite-design paired effects and sign counts; no multiplicity decision",
            "fixed-design stability description only",
        ),
        (
            "D-CROSSED-3X3",
            "retrospective; result-informed",
            r"3 engine splits $\times$ 3 training streams per task",
            "9 paired effects, finite-cell additive-residual RMS, and all available 3-level subset sensitivities",
            "selected-grid arithmetic only; no population variance components",
        ),
        (
            "D-CMAPSS-PREP5",
            "retrospective; result-informed",
            r"FD004 $\times$ 3 composite levels; one-factor protocol settings",
            "paired effects and setting-minus-reference changes",
            "joint window--training-dynamics and sensor-budget--capacity sensitivity only",
        ),
        (
            "D-PREF-REFIT-3X3",
            "retrospective; result-informed",
            "3 splits $\\times$ 3 separately seeded streams; reused test engines",
            "conditional selection counts over a fixed nine-point grid",
            "selection-instability diagnostic; not adjusted performance evidence",
        ),
        (
            "D-SHARED-FACT-8",
            "retrospective; result-informed",
            "8 cells $\\times$ 5 streams on fixed FD004",
            "bundled-architecture--risk--consistency finite factorial",
            "shared-objective effects only; OCM-specific components remain bundled",
        ),
        (
            "D-LOFO-4",
            "retrospective; result-informed",
            "4 held-out families $\\times$ 5 streams; 5 clustered perturbation-seed evaluations per stream",
            "zero-anchored AUC, nonzero-range response, and threshold-event support",
            "augmentation-overlap sensitivity; not unseen-family or physical-fault robustness",
        ),
        (
            "D-NCMAPSS-SW9",
            "retrospective; result-informed",
            "9 sampling--window cells $\\times$ 5 streams; same 3 test units",
            "joint density--horizon representation sensitivity",
            "computational-proxy dependence only",
        ),
        (
            "D-NCMAPSS-H3",
            "retrospective; result-informed",
            "3 intervals $\\times$ 5 streams at a 6001-record source span",
            "fixed-source-span sampling-density control",
            "computational-proxy sensitivity; not external validation",
        ),
    ]
    main_family_ids = {
        "E-FIXED-12",
        "S-ENDPOINT-80",
        "S-PERTURB-54",
        "D-CROSSED-3X3",
        "D-CMAPSS-PREP5",
        "D-LOFO-4",
        "D-NCMAPSS-H3",
    }
    rows = [row for row in rows if row[0] in main_family_ids]
    english_rows = [
        f"{family} & {formation}; {unit} & {rule} & {role} \\\\"
        for family, formation, unit, rule, role in rows
    ]
    english = "\n".join(
        [
            r"\begin{table*}[!t]",
            r"\centering",
            r"\scriptsize",
            r"\caption{Focal analysis-family registry. Manuscript focus does not imply prospective confirmation. Resampling and multiplicity diagnostics remain in the Supplementary Information; the complete retrospective registry is reported there.}",
            r"\label{tab:main-analysis-family-registry}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{p{0.12\linewidth}p{0.30\linewidth}p{0.34\linewidth}p{0.24\linewidth}}",
            r"\toprule",
            r"Family & Formation/status and unit & Evidence record & Allowed wording \\",
            r"\midrule",
            *english_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table*}",
            "",
        ]
    )
    chinese_values = {
        "E-FIXED-12": (
            "早期结果审阅后形成；受结果信息影响",
            "5 个耦合复合种子水平 $\\times$ 匹配发动机；固定任务",
            "有限设计配对效应、符号与事件支持；重采样诊断置于补充材料",
            "固定任务估计；不得作确认性表述",
        ),
        "S-ENDPOINT-80": (
            "回顾性；受结果信息影响",
            "5 个耦合复合种子水平 $\\times$ 匹配发动机；固定任务",
            "阈值网格、符号与事件支持；重采样诊断置于补充材料",
            "仅阈值敏感性",
        ),
        "S-PERTURB-54": (
            "回顾性；受结果信息影响",
            "5 个耦合复合种子水平 $\\times$ 5 扰动种子 $\\times$ 发动机",
            "完整归一化坐标方向族；重采样诊断置于补充材料",
            "仅适用于声明增强混合分布的归一化坐标敏感性",
        ),
        "D-NCMAPSS-ROT": (
            "回顾性；受结果信息影响",
            "固定 DS02 开发/测试 unit",
            "描述性轮换与先验检查",
            "仅强下采样计算代理",
        ),
        "D-STREAM10-FIXED": (
            "回顾性；受结果信息影响",
            "每个任务一个固定划分 $\\times$ 10 条分别设种子的训练流",
            "有限设计配对效应与方向计数；不作多重性决策",
            "仅描述固定设计稳定性",
        ),
        "D-CROSSED-3X3": (
            "回顾性；受既有结果影响",
            "每个任务 3 个发动机划分乘 3 个训练流",
            "9 个有限设计配对效应、方向计数及划分/训练流/交互 RMS",
            "仅作有限设计分解；不解释为总体方差分量",
        ),
        "D-CMAPSS-PREP5": (
            "回顾性；受既有结果影响",
            "FD004 乘 3 个复合水平；单因素协议设置",
            "配置配对效应及相对参考点变化",
            "仅说明窗口--训练动态及传感器预算--容量联合敏感性",
        ),
        "D-PREF-REFIT-3X3": (
            "回顾性；受结果信息影响",
            "3 划分 $\\times$ 3 条分别设种子的训练流；重复使用测试发动机",
            "固定九点网格上的条件选择计数",
            "仅诊断选择不稳定性；不是选择校正后的性能证据",
        ),
        "D-SHARED-FACT-8": (
            "回顾性；受结果信息影响",
            "固定 FD004 上 8 单元 $\\times$ 5 训练流",
            "捆绑架构--risk--consistency 有限因子设计",
            "仅解释共享目标；OCM 专属部件仍被捆绑",
        ),
        "D-LOFO-4": (
            "回顾性；受结果信息影响",
            "4 个留出扰动族 $\\times$ 5 训练流；每流 5 个聚类扰动种子评价",
            "归一化梯形场景 AUC，再对场景等权平均",
            "仅算法扰动族敏感性；不是物理故障鲁棒性",
        ),
        "D-NCMAPSS-SW9": (
            "回顾性；受结果信息影响",
            "9 个采样--窗口单元 $\\times$ 5 训练流；共享 3 个测试 unit",
            "采样密度与时间跨度联合表示敏感性",
            "仅计算代理依赖性",
        ),
        "D-NCMAPSS-H3": (
            "回顾性；受结果信息影响",
            "固定 6001 源记录跨度下 3 间隔 $\\times$ 5 训练流",
            "固定源跨度采样密度控制",
            "仅计算代理敏感性；不是外部验证",
        ),
    }
    chinese_rows = [
        f"{family} & {chinese_values[family][0]}；"
        f"{chinese_values[family][1]} & {chinese_values[family][2]} & "
        f"{chinese_values[family][3]} \\\\"
        for family, *_ in rows
    ]
    chinese = "\n".join(
        [
            r"\begin{table*}[!t]",
            r"\centering",
            r"\scriptsize",
            r"\caption{正文关注分析族简表。正文关注不表示前瞻性确认；重采样和多重性诊断置于补充材料，完整回顾性注册表亦见补充材料。}",
            r"\label{tab:main-analysis-family-registry-zh}",
            r"\resizebox{\textwidth}{!}{%",
            r"\begin{tabular}{p{0.12\linewidth}p{0.30\linewidth}p{0.34\linewidth}p{0.24\linewidth}}",
            r"\toprule",
            r"证据族 & 形成状态与统计单位 & 证据记录 & 允许表述 \\",
            r"\midrule",
            *chinese_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"}",
            r"\end{table*}",
            "",
        ]
    )
    (output / "table_main_analysis_family_registry.tex").write_text(
        english, encoding="utf-8", newline="\n"
    )
    (output / "table_main_analysis_family_registry_zh.tex").write_text(
        chinese, encoding="utf-8", newline="\n"
    )


def write_evidence_navigation(output: Path) -> None:
    rows = [
        (
            "E-FIXED-12",
            "Main Fig. 2; Supp. fixed-task and sign tables",
            "paper_outputs/release_v1/primary_fixed_task_estimates.csv",
            "scripts/analyze_release_statistics.py; scripts/build_release_evidence.py",
        ),
        (
            "S-ENDPOINT-80",
            "Supp. endpoint-sensitivity figure and late-event table",
            "paper_outputs/release_v1/endpoint_tolerance_sensitivity.csv",
            "scripts/analyze_release_statistics.py; scripts/build_release_evidence.py",
        ),
        (
            "S-PERTURB-54",
            "Supp. normalized-coordinate perturbation table",
            "paper_outputs/release_v1/normalized_perturbation_contrasts.csv",
            "scripts/analyze_normalized_perturbation_evidence.py; scripts/build_release_evidence.py",
        ),
        (
            "D-NCMAPSS-ROT",
            "Supp. N-CMAPSS rotation tables and figure",
            "paper_outputs/advanced_evidence/ncmapss_dev_rotation_summary.csv",
            "scripts/analyze_ncmapss_dev_rotation.py; scripts/build_manuscript_evidence.py",
        ),
        (
            "D-STREAM10-FIXED",
            "Main/Supp. fixed-split ten-stream effects",
            "paper_outputs/round12_new_evidence/independent_stream_effects.csv",
            "scripts/analyze_round12_new_experiments.py --families independent",
        ),
        (
            "D-CROSSED-3X3",
            "Main/Supp. crossed split-by-training-stream audit",
            "paper_outputs/round12_new_evidence/crossed_split_stream_summary.csv",
            "scripts/analyze_round12_new_experiments.py --families crossed",
        ),
        (
            "D-CMAPSS-PREP5",
            "Main/Supp. C-MAPSS preprocessing sensitivity",
            "paper_outputs/round12_new_evidence/cmapss_preprocessing_paired_summary.csv",
            "scripts/analyze_round12_new_experiments.py --families cmapss_preprocessing",
        ),
        (
            "D-PREF-REFIT-3X3",
            "Supp. preference refit grid and selection counts",
            "paper_outputs/round12_new_evidence/preference_cell_summary.csv",
            "scripts/analyze_round12_new_experiments.py --families preference",
        ),
        (
            "D-SHARED-FACT-8",
            "Main/Supp. limited shared-objective factorial",
            "paper_outputs/round12_new_evidence/factorial_effect_summary.csv",
            "scripts/analyze_round12_new_experiments.py --families factorial",
        ),
        (
            "D-LOFO-4",
            "Main/Supp. leave-one-perturbation-family-out audit",
            "paper_outputs/round12_new_evidence/lofo_family_effect_summary.csv",
            "scripts/analyze_round12_new_experiments.py --families lofo",
        ),
        (
            "D-NCMAPSS-SW9",
            "Main/Supp. N-CMAPSS sampling--window grid",
            "paper_outputs/round12_new_evidence/ncmapss_sampling_window_summary.csv",
            "scripts/analyze_round12_new_experiments.py --families ncmapss",
        ),
        (
            "D-NCMAPSS-H3",
            "Main/Supp. fixed-source-span N-CMAPSS control",
            "paper_outputs/round12_new_evidence/ncmapss_horizon_matched_summary.csv",
            "scripts/analyze_round12_new_experiments.py --families ncmapss_horizon",
        ),
    ]
    english_rows = [
        f"{family} & {latex_escape(exhibit)} & \\path{{{source}}} & "
        f"\\path{{{command}}} \\\\"
        for family, exhibit, source, command in rows
    ]
    english = "\n".join(
        [
            r"\begin{table*}[!t]",
            r"\centering",
            r"\tiny",
            r"\setlength{\tabcolsep}{2pt}",
            r"\caption{Evidence navigation map from analysis family to reader-facing exhibit, machine-readable source, and generating command. Allowed wording remains governed by Table~\ref{tab:multiplicity-registry}.}",
            r"\label{tab:evidence-navigation}",
            r"\begin{tabular}{p{0.10\linewidth}p{0.22\linewidth}p{0.30\linewidth}p{0.30\linewidth}}",
            r"\toprule",
            r"Family & PDF exhibit & Machine-readable source & Generator \\",
            r"\midrule",
            *english_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    zh_exhibits = {
        "E-FIXED-12": "正文图 2；补充材料固定任务与符号表",
        "S-ENDPOINT-80": "补充材料端点敏感性图与晚事件表",
        "S-PERTURB-54": "补充材料归一化坐标扰动表",
        "D-NCMAPSS-ROT": "补充材料 N-CMAPSS 轮换表与图",
        "D-STREAM10-FIXED": "正文/补充材料固定划分十流效应",
        "D-CROSSED-3X3": "正文/补充材料划分与训练流完全交叉审计",
        "D-CMAPSS-PREP5": "正文/补充材料 C-MAPSS 预处理敏感性",
        "D-PREF-REFIT-3X3": "补充材料偏好重训网格与选择计数",
        "D-SHARED-FACT-8": "正文/补充材料有限共享目标因子实验",
        "D-LOFO-4": "正文/补充材料逐扰动族留出审计",
        "D-NCMAPSS-SW9": "正文/补充材料 N-CMAPSS 采样--窗口网格",
        "D-NCMAPSS-H3": "正文/补充材料 N-CMAPSS 固定源跨度控制",
    }
    chinese_rows = [
        f"{family} & {zh_exhibits[family]} & \\path{{{source}}} & "
        f"\\path{{{command}}} \\\\"
        for family, _, source, command in rows
    ]
    chinese = "\n".join(
        [
            r"\begin{table*}[!t]",
            r"\centering",
            r"\tiny",
            r"\setlength{\tabcolsep}{2pt}",
            r"\caption{从分析族到 PDF 展示、机器可读来源和生成命令的证据导航。允许表述仍以分析族登记表为准。}",
            r"\label{tab:evidence-navigation-zh}",
            r"\begin{tabular}{p{0.10\linewidth}p{0.22\linewidth}p{0.30\linewidth}p{0.30\linewidth}}",
            r"\toprule",
            r"分析族 & PDF 展示 & 机器可读来源 & 生成器 \\",
            r"\midrule",
            *chinese_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table*}",
            "",
        ]
    )
    (output / "table_evidence_navigation.tex").write_text(
        english, encoding="utf-8", newline="\n"
    )
    (output / "table_evidence_navigation_zh.tex").write_text(
        chinese, encoding="utf-8", newline="\n"
    )


def write_chinese_table_copies(output: Path) -> None:
    captions = {
        "table_design_sensitivity.tex": "FD004 五种子窗口长度与 RUL 截断敏感性分析。",
        "table_threshold_paired_ci.tex": "相对 RAST-GRU 的配对阈值敏感性区间。",
        "table_cmapss_macro.tex": "C-MAPSS FD001--FD004 五复合种子宏平均结果。",
        "table_late_tail_risk.tex": "C-MAPSS 临近失效正晚误差尾部风险。",
        "table_ablation_fd004.tex": "FD004 五复合种子模块与目标函数消融。",
        "table_ncmapss.tex": "固定 N-CMAPSS DS02 协议的探索性转移评估。",
        "table_conformal_calibration.tex": "名义 80\\% RUL 区间的后选择经验残差分位数调整。",
        "table_efficiency.tex": "FD004 模型复杂度与实测推理效率。",
        "table_close_prior_fd002.tex": "FD002 近期工况双注意力设计族的五复合种子比较。",
        "table_hierarchical_direct_predecessor.tex": "相对于直接前身 RAST-GRU 的分层配对证据。",
        "table_checkpoint_sensitivity.tex": "OCM-MST-GRU 在 FD004 上的五种子检查点规则敏感性。",
        "table_protocol_track_sensitivity.tex": "FD004 共同风险协议与标准化常规协议的敏感性比较。",
        "table_hyperparameter_sensitivity.tex": "FD004 工况簇数量与辅助损失整体尺度的五种子敏感性。",
        "table_condition_k_diagnostics.tex": "仅使用训练核心集的 FD004 工况聚类诊断。",
        "table_track_b_absolute.tex": "标准化常规 Track B 内的 FD004 绝对结果。",
        "table_architecture_attribution_absolute.tex": "含绝对结果的 FD004 架构与协议归因。",
        "table_architecture_selected_comparators.tex": "匹配协议下的 FD004 架构配对证据。",
        "table_core_asym_cmapss.tex": "Core 与非对称晚误差偏好的 C-MAPSS 五种子宏平均比较。",
        "table_validation_endpoint_audit.tex": "五个复合种子下的验证端点构成。",
        "table_stress_family_uncertainty.tex": "受控退化 RMSE 与晚预测风险的故障族级配对不确定性。",
        "table_metric_best_comparators.tex": "相对各指标最优受控基线的预声明比较。",
        "table_ncmapss_k_sensitivity.tex": "N-CMAPSS DS02 对训练侧工况簇数量的敏感性。",
        "table_core_asym_paired.tex": "Core 与非对称偏好在固定 C-MAPSS 任务上的五种子配对证据。",
        "table_endpoint_selection_confirmation.tex": "三种验证端点设计下的检查点选择敏感性。",
        "table_validation_endpoint_distribution.tex": "验证端点设计与自然测试端点分布的比较。",
        "table_endpoint_confirmation_interval_directions.tex": "验证端点敏感性审计的配对诊断方向汇总。",
        "table_ncmapss_validation_k_selection.tex": "仅基于验证证据的 N-CMAPSS 工况簇数量选择审计。",
        "table_training_augmentation_protocol.tex": "声明的训练期归一化坐标增强混合机制。",
        "table_stress_generator_protocol.tex": "九类归一化坐标扰动生成机制。",
        "table_stress_equivalence_sensitivity.tex": "候选等效边界下的描述性概率敏感性。",
        "table_official_dual_mixer.tex": "官方代码派生 Dual-Mixer 与 OCM-MST-GRU 的锁定划分比较。",
        "table_official_dual_mixer_summary.tex": "官方代码派生 Dual-Mixer 与 Core/Asym 的固定任务区间方向摘要。",
        "table_cross_backbone_protocol_buildup.tex": "RAST-GRU 与 Transformer-lite 上的 FD004 五种子累计协议构建。",
        "table_cross_backbone_transitions.tex": "双骨干累计协议构建的逐转变配对区间。",
        "table_fast_cudnn_repeat_audit.tex": "fast-cuDNN 固定种子重跑波动与跨种子波动审计。",
        "table_reliability_observed_fraction.tex": "软可靠性权重的观测比例诊断。",
        "table_core_asym_decision_cost.tex": "Core 与 Asym 相对主要对比方法的配对决策成本敏感性。",
        "table_endpoint_epoch_agreement.tex": "不同验证端点设计下最佳检查点 epoch 的敏感性。",
        "table_stress_operating_points.tex": "Core 与 Asym 相对 RAST-GRU 的九类退化区间方向计数。",
        "table_tcn_gru_robust_summary.tex": "TCN-GRU 种子失稳的稳健汇总。",
        "table_tcn_gru_seed_audit.tex": "TCN-GRU 五种子完整失败审计。",
        "table_run_quality_robust_summary.tex": "跨模型极早选择与重尾汇总敏感性。",
        "table_run_quality_flagged_cells.tex": "统一验证侧极早选择敏感性规则标记的运行单元。",
        "table_augmentation_distribution_cost_intervals.tex": "相对锁定增强分布的配对种子敏感性区间。",
    }
    for filename, caption in captions.items():
        source = output / filename
        target = output / filename.replace(".tex", "_zh.tex")
        text = source.read_text(encoding="utf-8-sig")
        if filename == "table_training_augmentation_protocol.tex":
            text = (
                text.replace("Mechanism", "机制")
                .replace("Declared level", "声明强度")
                .replace("Redraw/window", "逐窗口重绘")
                .replace("Baseline view", "基线视图")
                .replace("OCM view", "OCM 视图")
            )
        elif filename == "table_stress_generator_protocol.tex":
            text = (
                text.replace("Perturbation family", "扰动类别")
                .replace("Intensity", "强度参数")
                .replace("Nonzero grid", "非零网格")
                .replace("Scope", "作用范围")
                .replace("Mask", "更新掩码")
            )
        lines = text.splitlines()
        for index, line in enumerate(lines):
            if line.startswith(r"\caption{"):
                lines[index] = rf"\caption{{{caption}}}"
            if "Notes:" in lines[index]:
                lines[index] = lines[index].replace("Notes:", "注：")
        target.write_text(
            "\n".join(lines) + "\n", encoding="utf-8", newline="\n"
        )


def normalize_paper_display_names(output: Path) -> None:
    """Keep historical implementation identifiers out of paper-facing tables."""
    replacements = {
        "OCM-RAST-GRU": "OCM-MST-GRU",
        "OCM backbone": "complete OCM configuration",
        "Architecture attribution": "Configuration-axis comparison",
        "architecture attribution": "configuration-axis comparison",
        "[POST-FREEZE SENSITIVITY]": "[SENSITIVITY]",
        "[POST-FREEZE;": "[RETROSPECTIVE SENSITIVITY;",
        "POST-FREEZE": "RETROSPECTIVE DIAGNOSTIC",
        "PRIMARY FROZEN": "MAIN RESULT-INFORMED FIXED-TASK ESTIMATION",
        "PRIMARY LOCKED": "MAIN RESULT-INFORMED FIXED-TASK ESTIMATION",
        "POST-LOCK NORMALIZATION CONTROL": "RETROSPECTIVE NORMALIZATION SENSITIVITY",
        "POST-LOCK": "RETROSPECTIVE",
        "Post-lock": "Retrospective",
        "post-lock": "retrospective",
        "confirmation experiment": "endpoint-design sensitivity audit",
        "Checkpoint-selection confirmation": "Checkpoint-selection sensitivity",
        "checkpoint-selection confirmation": "checkpoint-selection sensitivity",
        "retrospective symmetric confirmation": "retrospective Core sensitivity point",
        "locked reference": "study-defined reference",
        "locked engine split": "fixed engine split",
        "$p=1$ (locked)": "$p=1$ (study reference)",
        "p=1 (locked)": "p=1 (study reference)",
        "Simultaneous 95\\% CI": "Max-$t$ diagnostic envelope",
        "Nominal 95\\% CI": "Nominal diagnostic interval",
        "95\\% CI low": "Diagnostic lower",
        "95\\% CI high": "Diagnostic upper",
        "[95\\% CI]": "[diagnostic interval]",
        "Pr(left lower)": "Boot. frac. left lower",
        "Pr(Asym better)": "Boot. frac. Asym lower",
        "Pr(condition lower)": "Boot. frac. condition lower",
        "Pr(alternative lower)": "Boot. frac. alternative lower",
        "Pr(Core lower)": "Boot. frac. Core lower",
        "Pr(Asym lower)": "Boot. frac. Asym lower",
        "Pr(right lower)": "Boot. frac. right lower",
        "Pr(OCM lower)": "Boot. frac. OCM lower",
        "locked 260-run grid": "fixed 260-run grid",
        "locked common": "fixed common",
        "locked operating point": "study-reference operating point",
        "locked formal checkpoints": "fixed formal checkpoints",
        "locked composite distribution": "study-reference composite distribution",
        "deterministic locked benchmark": "deterministic fixed benchmark",
        "locally frozen value": "researcher-specified illustrative value",
        "frozen sensor-budget constraint": "fixed sensor-budget constraint",
        "then frozen before test evaluation": "then held fixed before test evaluation",
        "retrospective confirmation": "retrospective sensitivity point",
        "confirmation intervals": "diagnostic intervals",
        "confirmation interval": "diagnostic interval",
        "confirmation tests": "diagnostic tests",
        "confirmation test": "diagnostic test",
        "validation-endpoint confirmation": "validation-endpoint sensitivity audit",
        "The confirmation does not replace": "This sensitivity audit does not replace",
        "Core confirmation": "Core sensitivity point",
        "95\\% CI": "diagnostic interval",
        "predeclared selection rule": "study-defined validation-only selection rule",
        "CI supports lower Asym": "nominal interval lower for Asym; not adjusted evidence",
        "CI supports lower asym": "nominal interval lower for Asym; not adjusted evidence",
        "three-level paired bootstrap": "crossed seed--perturbation-seed--engine bootstrap diagnostic",
    }
    for path in output.glob("*.tex"):
        text = path.read_text(encoding="utf-8-sig")
        normalized = text
        for old, new in replacements.items():
            normalized = normalized.replace(old, new)
        if normalized != text:
            path.write_text(normalized, encoding="utf-8", newline="\n")


def write_semantic_table_aliases(output: Path) -> None:
    aliases = {
        "table_endpoint_confirmation_interval_directions.tex":
            "table_validation_endpoint_paired_directions.tex",
        "table_endpoint_confirmation_interval_directions_zh.tex":
            "table_validation_endpoint_paired_directions_zh.tex",
    }
    for historical_name, semantic_name in aliases.items():
        source = output / historical_name
        if source.exists():
            shutil.copy2(source, output / semantic_name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build protocol-v3 LaTeX macros, tables, and a claim-to-source ledger.")
    parser.add_argument("--source-dir", default=str(DEFAULT_SOURCE))
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--copy-to", default=None, help="Optional manuscript-local mirror directory.")
    args = parser.parse_args()

    source = Path(args.source_dir)
    output = Path(args.output_dir)
    manifest_path = source / "advanced_evidence_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Run scripts/build_advanced_evidence.py before building manuscript evidence.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    if manifest.get("protocol_version") != "3.0" or int(manifest.get("formal_run_count", 0)) != 260:
        raise ValueError("Refusing to build manuscript evidence from a non-v3 or incomplete manifest.")

    aggregation = require_csv(source, "aggregation_summary.csv", 13)
    fairness = require_csv(source, "benchmark_fairness.csv", 13)
    late_tail = require_csv(source, "late_tail_risk_summary.csv", 13)
    pareto = require_csv(source, "pareto_seed_stability_summary.csv", 13)
    intervals = require_csv(source, "interval_calibration_summary.csv", 2)
    conformal = require_csv(source, "conformal_interval_summary.csv", 2)
    external = ensure_model_display(require_csv(source, "ncmapss_ds02_summary.csv", 13))
    design = require_csv(source, "design_sensitivity_summary.csv", 5)
    close_prior = require_csv(source, "close_prior_fd002_summary.csv", 2)
    bootstrap = require_csv(source, "hierarchical_bootstrap_bca.csv", 48)
    reliability = require_csv(source, "reliability_fault_localization_summary.csv", 4)
    attention = require_csv(source, "attention_mechanism_summary.csv", 2)
    risk = require_csv(source, "risk_hyperparameter_search.csv", 9)
    threshold_paired = require_csv(source, "threshold_paired_bootstrap.csv", 768)
    ablation = require_csv(source, "ablation_fd004_multiseed_summary.csv", 13)
    checkpoint_summary = require_csv(source, "checkpoint_sensitivity_summary.csv", 2)
    checkpoint_bootstrap = require_csv(source, "checkpoint_sensitivity_paired_bootstrap.csv", 4)
    protocol_track = require_csv(source, "standard_protocol_track_comparison.csv", 6)
    protocol_track_bootstrap = require_csv(source, "standard_protocol_track_paired_bootstrap.csv", 18)
    fixed_subset = require_csv(source, "subset_engine_paired_bootstrap.csv", 192)
    ncmapss_units = require_csv(source, "ncmapss_ds02_per_unit_summary.csv", 39)
    attribution_summary = require_csv(source, "ocm_attribution_summary.csv", 4)
    attribution_bootstrap = require_csv(source, "ocm_attribution_paired_bootstrap.csv", 12)
    hyperparameter_sensitivity = require_csv(source, "hyperparameter_sensitivity_summary.csv", 8)
    cluster_k_diagnostics = require_csv(source, "condition_cluster_k_diagnostics_summary.csv", 5)
    stress_pair = require_csv(source, "stress_pair_extended_auc_comparison.csv", 45)
    ncmapss_split_sensitivity = require_csv(source, "ncmapss_split_sensitivity_summary.csv", 6)
    architecture_absolute = require_csv(source, "architecture_attribution_absolute.csv", 5)
    architecture_bootstrap = require_csv(source, "architecture_selected_comparator_bootstrap.csv", 8)
    track_b_absolute = require_csv(source, "track_b_absolute_ranking.csv", 6)
    core_asym = require_csv(source, "ocm_core_asym_summary.csv", 2)
    core_asym_bootstrap = require_csv(source, "ocm_core_asym_paired_bootstrap.csv", 4)
    endpoint_audit = require_csv(source, "validation_endpoint_audit.csv", 20)
    stress_uncertainty = require_csv(source, "stress_family_uncertainty.csv", 99)
    stress_equivalence = require_csv(source, "stress_equivalence_margin_sensitivity.csv", 54)
    efficiency_enhanced = require_csv(source, "benchmark_efficiency_enhanced.csv", 13)
    selected_comparators = require_csv(source, "selected_comparator_evidence.csv", 12)
    ncmapss_k_sensitivity = require_csv(source, "ncmapss_k_sensitivity_summary.csv", 3)
    endpoint_confirmation = require_csv(source, "endpoint_selection_confirmation_seed_metrics.csv", 90)
    endpoint_confirmation_bootstrap = require_csv(source, "endpoint_selection_confirmation_bootstrap.csv", 54)
    endpoint_distribution = require_csv(source, "validation_endpoint_distribution_summary.csv", 6)
    ncmapss_validation_k = require_csv(source, "ncmapss_validation_only_k_selection.csv", 3)
    official_dual_summary = require_csv(source, "official_dual_mixer_summary.csv", 4)
    official_dual_paired = require_csv(source, "official_dual_mixer_paired_bootstrap.csv", 12)
    official_dual_operating = require_csv(source, "official_dual_mixer_operating_points.csv", 24)
    major_comparators = require_csv(source, "core_asym_major_comparator_bootstrap.csv", 18)
    decision_cost_pairs = require_csv(source, "core_asym_decision_cost_bootstrap.csv", 24)
    endpoint_task_summary = require_csv(source, "endpoint_selection_task_summary.csv", 18)
    endpoint_epoch_agreement = require_csv(source, "endpoint_selection_epoch_agreement.csv", 6)
    cross_backbone_summary = require_csv(source, "cross_backbone_protocol_buildup_summary.csv", 12)
    cross_backbone_transitions = require_csv(source, "cross_backbone_protocol_buildup_transitions.csv", 40)
    stress_operating_uncertainty = require_csv(source, "stress_operating_points_uncertainty.csv", 198)
    stress_engine_bootstrap = require_csv(source, "stress_operating_points_engine_bootstrap.csv", 162)
    fast_cudnn_repeat = require_csv(source, "fast_cudnn_repeat_summary.csv", 16)
    ncmapss_core = require_csv(source, "ncmapss_ds02_symmetric_candidate_summary.csv", 1)
    core_rast_subset = require_csv(source, "core_asym_rast_subset_bootstrap.csv", 24)
    augmentation_distribution = require_csv(source, "augmentation_distribution_summary.csv", 15)
    augmentation_intervals = require_csv(source, "augmentation_distribution_paired_intervals.csv", 150)
    endpoint_cluster_audit = require_csv(source, "validation_endpoint_cluster_audit.csv", 6)
    tcn_gru_audit = require_csv(source, "tcn_gru_seed_audit.csv", 20)
    tcn_gru_robust = require_csv(source, "tcn_gru_subset_robust_summary.csv", 4)
    fd002_transfer_summary = require_csv(source, "fd002_protocol_transfer_summary.csv", 5)
    fd002_transfer_bootstrap = require_csv(source, "fd002_protocol_transfer_paired_bootstrap.csv", 20)

    output.mkdir(parents=True, exist_ok=True)
    proposed = aggregation.loc[aggregation["model"] == PROPOSED].iloc[0]
    ledger: list[dict] = []
    macros = [
        macro("FormalRunCount", 260),
        macro("EvaluatedModelCount", 13),
        macro("BaselineModelCount", 12),
        macro("CMapssSubsetCount", 4),
        macro("MainSeedCount", 5),
        macro("StressPerturbSeedCount", 5),
        macro("NCMapssRunCount", 65),
    ]

    fd002_protocol_labels = {"RAST protocol effect", "OCM protocol effect"}
    fd002_protocol_rows = fd002_transfer_bootstrap[
        fd002_transfer_bootstrap["comparison"].isin(fd002_protocol_labels)
    ]
    fd002_architecture_rows = fd002_transfer_bootstrap[
        ~fd002_transfer_bootstrap["comparison"].isin(fd002_protocol_labels)
    ]
    fd002_ocm_cost = fd002_transfer_bootstrap[
        (fd002_transfer_bootstrap["comparison"] == "OCM protocol effect")
        & (fd002_transfer_bootstrap["metric"] == "cost_5")
    ].iloc[0]
    fd002_core_rast_cost = fd002_transfer_bootstrap[
        (fd002_transfer_bootstrap["comparison"] == "Core architecture contrast")
        & (fd002_transfer_bootstrap["metric"] == "cost_5")
    ].iloc[0]
    macros.extend(
        [
            macro("FDTwoTransferResolvedCount", int((fd002_transfer_bootstrap["classification_nominal_ci"] != "unresolved").sum())),
            macro("FDTwoTransferUnresolvedCount", int((fd002_transfer_bootstrap["classification_nominal_ci"] == "unresolved").sum())),
            macro("FDTwoHolmSignificantCount", int(fd002_transfer_bootstrap["holm_significant_0_05"].sum())),
            macro("FDTwoProtocolCoreLowerCount", int((fd002_protocol_rows["ci95_high"] < 0).sum())),
            macro("FDTwoProtocolAsymLowerCount", int((fd002_protocol_rows["ci95_low"] > 0).sum())),
            macro("FDTwoProtocolUnresolvedCount", int((fd002_protocol_rows["classification_nominal_ci"] == "unresolved").sum())),
            macro("FDTwoArchitectureOCMLowerCount", int((fd002_architecture_rows["ci95_high"] < 0).sum())),
            macro("FDTwoArchitectureReferenceLowerCount", int((fd002_architecture_rows["ci95_low"] > 0).sum())),
            macro("FDTwoArchitectureUnresolvedCount", int((fd002_architecture_rows["classification_nominal_ci"] == "unresolved").sum())),
            macro("FDTwoOCMCoreMinusAsymCost", f"{fd002_ocm_cost['mean_left_minus_right']:.3f}"),
            macro("FDTwoOCMCoreMinusAsymCostLow", f"{fd002_ocm_cost['ci95_low']:.3f}"),
            macro("FDTwoOCMCoreMinusAsymCostHigh", f"{fd002_ocm_cost['ci95_high']:.3f}"),
            macro("FDTwoOCMCoreMinusRASTCoreCost", f"{fd002_core_rast_cost['mean_left_minus_right']:.3f}"),
            macro("FDTwoOCMCoreMinusRASTCoreCostLow", f"{fd002_core_rast_cost['ci95_low']:.3f}"),
            macro("FDTwoOCMCoreMinusRASTCoreCostHigh", f"{fd002_core_rast_cost['ci95_high']:.3f}"),
        ]
    )
    for name, selector, value in [
        ("FDTwoTransferResolvedCount", "all comparisons; nominal interval excludes zero", int((fd002_transfer_bootstrap["classification_nominal_ci"] != "unresolved").sum())),
        ("FDTwoTransferUnresolvedCount", "all comparisons; nominal interval includes zero", int((fd002_transfer_bootstrap["classification_nominal_ci"] == "unresolved").sum())),
        ("FDTwoHolmSignificantCount", "all 20 exploratory comparisons; Holm-adjusted p < 0.05", int(fd002_transfer_bootstrap["holm_significant_0_05"].sum())),
        ("FDTwoProtocolCoreLowerCount", "protocol comparisons; upper CI < 0", int((fd002_protocol_rows["ci95_high"] < 0).sum())),
        ("FDTwoProtocolAsymLowerCount", "protocol comparisons; lower CI > 0", int((fd002_protocol_rows["ci95_low"] > 0).sum())),
        ("FDTwoProtocolUnresolvedCount", "protocol comparisons; nominal interval includes zero", int((fd002_protocol_rows["classification_nominal_ci"] == "unresolved").sum())),
        ("FDTwoArchitectureOCMLowerCount", "architecture comparisons; upper CI < 0", int((fd002_architecture_rows["ci95_high"] < 0).sum())),
        ("FDTwoArchitectureReferenceLowerCount", "architecture comparisons; lower CI > 0", int((fd002_architecture_rows["ci95_low"] > 0).sum())),
        ("FDTwoArchitectureUnresolvedCount", "architecture comparisons; nominal interval includes zero", int((fd002_architecture_rows["classification_nominal_ci"] == "unresolved").sum())),
        ("FDTwoOCMCoreMinusAsymCost", "comparison=OCM protocol effect; metric=cost_5; mean", f"{fd002_ocm_cost['mean_left_minus_right']:.3f}"),
        ("FDTwoOCMCoreMinusAsymCostLow", "comparison=OCM protocol effect; metric=cost_5; CI low", f"{fd002_ocm_cost['ci95_low']:.3f}"),
        ("FDTwoOCMCoreMinusAsymCostHigh", "comparison=OCM protocol effect; metric=cost_5; CI high", f"{fd002_ocm_cost['ci95_high']:.3f}"),
        ("FDTwoOCMCoreMinusRASTCoreCost", "comparison=Core architecture contrast; metric=cost_5; mean", f"{fd002_core_rast_cost['mean_left_minus_right']:.3f}"),
        ("FDTwoOCMCoreMinusRASTCoreCostLow", "comparison=Core architecture contrast; metric=cost_5; CI low", f"{fd002_core_rast_cost['ci95_low']:.3f}"),
        ("FDTwoOCMCoreMinusRASTCoreCostHigh", "comparison=Core architecture contrast; metric=cost_5; CI high", f"{fd002_core_rast_cost['ci95_high']:.3f}"),
    ]:
        add_ledger(ledger, name, "fd002_protocol_transfer_paired_bootstrap.csv", selector, value)

    endpoint_index = endpoint_distribution.set_index(["subset", "endpoint_strategy"])
    fd002_uniform = endpoint_index.loc[("FD002", "uniform_single")]
    fd004_uniform = endpoint_index.loc[("FD004", "uniform_single")]
    endpoint_macros = [
        ("EndpointUniformWFDTwo", fd002_uniform["wasserstein_distance_to_test"], ".2f"),
        ("EndpointUniformWFDFour", fd004_uniform["wasserstein_distance_to_test"], ".2f"),
        ("EndpointTestCriticalFDTwo", fd002_uniform["test_critical_30_fraction"], ".3f"),
        ("EndpointTestCriticalFDFour", fd004_uniform["test_critical_30_fraction"], ".3f"),
    ]
    for name, raw_value, fmt in endpoint_macros:
        value = format(float(raw_value), fmt)
        macros.append(macro(name, value))
        add_ledger(
            ledger,
            name,
            "validation_endpoint_distribution_summary.csv",
            "uniform_single row for the named subset",
            value,
        )

    official_rmse = official_dual_paired[official_dual_paired["metric"] == "rmse"]
    official_lpr = official_dual_paired[official_dual_paired["metric"] == "lpr30"]
    official_nasa = official_dual_paired[official_dual_paired["metric"] == "nasa_per_engine"]
    official_count_macros = [
        ("OfficialDualRMSELowerCount", int((official_rmse["percentile_ci95_high"] < 0.0).sum())),
        ("OfficialOCMRMSELowerCount", int((official_rmse["percentile_ci95_low"] > 0.0).sum())),
        (
            "OfficialRMSEUnresolvedCount",
            int(
                (
                    (official_rmse["percentile_ci95_low"] <= 0.0)
                    & (official_rmse["percentile_ci95_high"] >= 0.0)
                ).sum()
            ),
        ),
        ("OfficialOCMLPRLowerCount", int((official_lpr["percentile_ci95_low"] > 0.0).sum())),
        ("OfficialOCMNASALowerCount", int((official_nasa["percentile_ci95_low"] > 0.0).sum())),
    ]
    for name, value in official_count_macros:
        macros.append(macro(name, value))
        add_ledger(
            ledger,
            name,
            "official_dual_mixer_paired_bootstrap.csv",
            "95% paired interval direction count across fixed C-MAPSS tasks",
            value,
        )

    high_cost = decision_cost_pairs[decision_cost_pairs["late_to_early_cost_ratio"].isin([5, 10])]
    for variant, prefix in (("core", "Core"), ("asym", "Asym")):
        part = high_cost[high_cost["ocm_variant"] == variant]
        cost_counts = {
            f"{prefix}HighCostLowerCount": int((part["percentile_ci95_high"] < 0.0).sum()),
            f"{prefix}HighCostWorseCount": int((part["percentile_ci95_low"] > 0.0).sum()),
            f"{prefix}HighCostUnresolvedCount": int(
                ((part["percentile_ci95_low"] <= 0.0) & (part["percentile_ci95_high"] >= 0.0)).sum()
            ),
        }
        for name, value in cost_counts.items():
            macros.append(macro(name, value))
            add_ledger(
                ledger,
                name,
                "core_asym_decision_cost_bootstrap.csv",
                f"paired 95% interval direction over kappa=5/10 and three comparators for {variant}",
                value,
            )

    metric_specs = [
        ("MacroRMSE", "macro_rmse_mean", ".3f"),
        ("MacroMAE", "macro_mae_mean", ".3f"),
        ("MacroNASAperEngine", "macro_nasa_per_engine_mean", ".3f"),
        ("MacroLPRThirty", "macro_lpr30_mean", ".3f"),
    ]
    for suffix, column, fmt in metric_specs:
        value = format(float(proposed[column]), fmt)
        competitor = best_competitor(aggregation, column)
        macros.extend(
            [
                macro(f"OCM{suffix}", value),
                macro(f"BestBaseline{suffix}", format(float(competitor[column]), fmt)),
                macro(f"BestBaseline{suffix}Name", latex_escape(competitor["model_display"])),
            ]
        )
        add_ledger(ledger, f"OCM{suffix}", "aggregation_summary.csv", f"model={PROPOSED}; column={column}", value)

    pareto_row = pareto.loc[pareto["model"] == PROPOSED].iloc[0]
    pareto_seed_count = int(round(float(pareto_row["pareto_seed_frequency"]) * 5))
    macros.append(macro("OCMParetoSeedFrequency", f"{float(pareto_row['pareto_seed_frequency']):.2f}"))
    macros.append(macro("OCMParetoSeedCount", pareto_seed_count))
    add_ledger(
        ledger,
        "OCMParetoSeedFrequency",
        "pareto_seed_stability_summary.csv",
        f"model={PROPOSED}",
        pareto_row["pareto_seed_frequency"],
    )

    interval_row = intervals.loc[intervals["model"] == PROPOSED].iloc[0]
    macros.extend(
        [
            macro("OCMIntervalCoverage", f"{float(interval_row['empirical_coverage_mean']):.3f}"),
            macro("OCMIntervalWidth", f"{float(interval_row['mean_interval_width_mean']):.2f}"),
            macro("OCMIntervalScore", f"{float(interval_row['mean_interval_score_mean']):.2f}"),
        ]
    )
    for name, column in (
        ("OCMIntervalCoverage", "empirical_coverage_mean"),
        ("OCMIntervalWidth", "mean_interval_width_mean"),
        ("OCMIntervalScore", "mean_interval_score_mean"),
    ):
        add_ledger(ledger, name, "interval_calibration_summary.csv", f"model={PROPOSED}", interval_row[column])

    conformal_row = conformal.loc[conformal["model"] == PROPOSED].iloc[0]
    macros.extend(
        [
            macro("OCMConformalCoverage", f"{float(conformal_row['conformal_coverage_mean']):.3f}"),
            macro("OCMConformalWidth", f"{float(conformal_row['conformal_mean_width_mean']):.2f}"),
            macro("OCMConformalIntervalScore", f"{float(conformal_row['conformal_interval_score_mean']):.2f}"),
        ]
    )
    for name, column in (
        ("OCMConformalCoverage", "conformal_coverage_mean"),
        ("OCMConformalWidth", "conformal_mean_width_mean"),
        ("OCMConformalIntervalScore", "conformal_interval_score_mean"),
    ):
        add_ledger(ledger, name, "conformal_interval_summary.csv", f"model={PROPOSED}", conformal_row[column])

    external_row = external.loc[external["model"] == PROPOSED].iloc[0]
    external_competitor = best_competitor(external, "unit_macro_rmse_mean")
    external_lpr_competitor = best_competitor(external, "unit_macro_lpr30_mean")
    external_predecessor = external.loc[external["model"] == "rast_gru"].iloc[0]
    macros.extend(
        [
            macro("OCMNCMapssRMSE", f"{float(external_row['unit_macro_rmse_mean']):.3f}"),
            macro("BestBaselineNCMapssRMSE", f"{float(external_competitor['unit_macro_rmse_mean']):.3f}"),
            macro("BestBaselineNCMapssRMSEName", latex_escape(external_competitor["model_display"])),
            macro("OCMNCMapssLPR", f"{float(external_row['unit_macro_lpr30_mean']):.3f}"),
            macro("BestBaselineNCMapssLPR", f"{float(external_lpr_competitor['unit_macro_lpr30_mean']):.3f}"),
            macro("BestBaselineNCMapssLPRName", latex_escape(external_lpr_competitor["model_display"])),
            macro("RASTNCMapssRMSE", f"{float(external_predecessor['unit_macro_rmse_mean']):.3f}"),
            macro("RASTNCMapssLPR", f"{float(external_predecessor['unit_macro_lpr30_mean']):.3f}"),
        ]
    )
    add_ledger(
        ledger,
        "OCMNCMapssRMSE",
        "ncmapss_ds02_summary.csv",
        f"model={PROPOSED}; column=unit_macro_rmse_mean",
        external_row["unit_macro_rmse_mean"],
    )
    for name, column in (
        ("OCMNCMapssLPR", "unit_macro_lpr30_mean"),
        ("RASTNCMapssRMSE", "unit_macro_rmse_mean"),
        ("RASTNCMapssLPR", "unit_macro_lpr30_mean"),
    ):
        row = external_row if name.startswith("OCM") else external_predecessor
        add_ledger(ledger, name, "ncmapss_ds02_summary.csv", f"model={row['model']}; column={column}", row[column])

    direct = bootstrap.loc[bootstrap["baseline_model"] == "rast_gru"].set_index("metric")
    for metric, suffix in (("rmse", "RMSE"), ("nasa_per_engine", "NASAperEngine"), ("lpr30", "LPRThirty")):
        row = direct.loc[metric]
        values = {
            f"Direct{suffix}Difference": float(row["mean_paired_difference_baseline_minus_proposed"]),
            f"Direct{suffix}CILow": float(row["bca_ci95_low"]),
            f"Direct{suffix}CIHigh": float(row["bca_ci95_high"]),
        }
        for name, value in values.items():
            macros.append(macro(name, f"{value:.3f}"))
            add_ledger(ledger, name, "hierarchical_bootstrap_bca.csv", f"baseline=rast_gru; metric={metric}", value)

    proposed_tail = late_tail.loc[late_tail["model"] == PROPOSED].iloc[0]
    predecessor_tail = late_tail.loc[late_tail["model"] == "rast_gru"].iloc[0]
    for name, row, column in (
        ("OCMLateCVAR", proposed_tail, "macro_late_cvar95_30_mean"),
        ("RASTLateCVAR", predecessor_tail, "macro_late_cvar95_30_mean"),
        ("OCMConditionalMLE", proposed_tail, "macro_conditional_mle30_mean"),
        ("RASTConditionalMLE", predecessor_tail, "macro_conditional_mle30_mean"),
    ):
        value = float(row[column])
        macros.append(macro(name, f"{value:.3f}"))
        add_ledger(ledger, name, "late_tail_risk_summary.csv", f"model={row['model']}; column={column}", value)

    ablation_index = ablation.set_index("ablation_variant")
    full_ablation = ablation_index.loc["full"]
    for variant, metric, name in (
        ("no_condition_norm", "test_rmse_mean", "NoConditionRMSEDelta"),
        ("no_condition_norm", "test_critical_30_severe_late_10_ratio_mean", "NoConditionSLPRTenDelta"),
        ("no_missing_mask", "test_rmse_mean", "NoMaskRMSEDelta"),
        ("no_missing_mask", "test_critical_30_late_prediction_ratio_mean", "NoMaskLPRDelta"),
        ("no_smooth_late_risk", "test_critical_30_late_prediction_ratio_mean", "NoSmoothRiskLPRDelta"),
    ):
        value = float(ablation_index.loc[variant, metric] - full_ablation[metric])
        macros.append(macro(name, f"{value:.3f}"))
        add_ledger(ledger, name, "ablation_fd004_multiseed_summary.csv", f"variant={variant}; delta_vs_full; column={metric}", value)

    reference_design = design.loc[design["design"] == "reference_w30_cap125"].iloc[0]
    window50_design = design.loc[design["design"] == "window_w50_cap125"].iloc[0]
    for name, row, column in (
        ("ReferenceDesignRMSE", reference_design, "test_rmse_mean"),
        ("ReferenceDesignLPR", reference_design, "test_critical_30_late_prediction_ratio_mean"),
        ("WindowFiftyRMSE", window50_design, "test_rmse_mean"),
        ("WindowFiftyLPR", window50_design, "test_critical_30_late_prediction_ratio_mean"),
    ):
        value = float(row[column])
        macros.append(macro(name, f"{value:.3f}"))
        add_ledger(ledger, name, "design_sensitivity_summary.csv", f"design={row['design']}; column={column}", value)

    lpr_stress = stress_pair.loc[stress_pair["metric"] == "critical_30_late_prediction_ratio"]
    absolute_lower_count = int((lpr_stress["ocm_minus_rast"] < 0.0).sum())
    macros.append(macro("StressLowerAbsoluteLPRAUCCount", absolute_lower_count))
    add_ledger(
        ledger,
        "StressLowerAbsoluteLPRAUCCount",
        "stress_pair_extended_auc_comparison.csv",
        "OCM normalized LPR degradation-curve mean < RAST-GRU across nine scenarios",
        absolute_lower_count,
    )
    stress_macro_specs = [
        (
            "StressLPRResolvedLowerCount",
            "critical_30_late_prediction_ratio",
            lambda frame: int((frame["nested_bootstrap_ci95_high"] < 0.0).sum()),
        ),
        (
            "StressRMSEHigherCount",
            "rmse",
            lambda frame: int((frame["ocm_minus_rast_auc"] > 0.0).sum()),
        ),
        (
            "StressCostOneLowerCount",
            "critical_30_decision_cost_1",
            lambda frame: int((frame["ocm_minus_rast_auc"] < 0.0).sum()),
        ),
        (
            "StressCostTenLowerCount",
            "critical_30_decision_cost_10",
            lambda frame: int((frame["ocm_minus_rast_auc"] < 0.0).sum()),
        ),
    ]
    for name, metric, reducer in stress_macro_specs:
        metric_frame = stress_uncertainty[stress_uncertainty["metric"] == metric]
        value = reducer(metric_frame)
        macros.append(macro(name, value))
        add_ledger(ledger, name, "stress_family_uncertainty.csv", f"metric={metric}", value)

    for point in ("Core", "Asym"):
        for metric, metric_name in (("rmse", "RMSE"), ("critical_30_late_prediction_ratio", "LPR")):
            subset = stress_engine_bootstrap[
                (stress_engine_bootstrap["operating_point"] == point)
                & (stress_engine_bootstrap["metric"] == metric)
            ]
            counts = subset["ci_classification"].value_counts()
            for suffix, classification in (
                ("OCMLower", f"CI supports lower {point}"),
                ("RASTLower", "CI supports lower RAST"),
                ("Unresolved", "unresolved"),
            ):
                name = f"{point}Stress{metric_name}{suffix}Count"
                value = int(counts.get(classification, 0))
                macros.append(macro(name, value))
                add_ledger(
                    ledger,
                    name,
                    "stress_operating_points_engine_bootstrap.csv",
                    f"operating_point={point}; metric={metric}; classification={classification}",
                    value,
                )

    macros.extend(
        [
            macro("ReliabilityAUROCLow", f"{float(reliability['fault_auroc_mean'].min()):.3f}"),
            macro("ReliabilityAUROCHigh", f"{float(reliability['fault_auroc_mean'].max()):.3f}"),
            macro("ReliabilityBrierLow", f"{float(reliability['brier_score_mean'].min()):.3f}"),
            macro("ReliabilityBrierHigh", f"{float(reliability['brier_score_mean'].max()):.3f}"),
            macro("ReliabilityECELow", f"{float(reliability['ece_10bin_mean'].min()):.3f}"),
            macro("ReliabilityECEHigh", f"{float(reliability['ece_10bin_mean'].max()):.3f}"),
        ]
    )
    add_ledger(
        ledger,
        "ReliabilityAUROCLow",
        "reliability_fault_localization_summary.csv",
        "minimum fault_auroc_mean across missing-rate levels",
        reliability["fault_auroc_mean"].min(),
    )
    add_ledger(
        ledger,
        "ReliabilityAUROCHigh",
        "reliability_fault_localization_summary.csv",
        "maximum fault_auroc_mean across missing-rate levels",
        reliability["fault_auroc_mean"].max(),
    )
    for name, column, reducer in (
        ("ReliabilityBrierLow", "brier_score_mean", "min"),
        ("ReliabilityBrierHigh", "brier_score_mean", "max"),
        ("ReliabilityECELow", "ece_10bin_mean", "min"),
        ("ReliabilityECEHigh", "ece_10bin_mean", "max"),
    ):
        value = getattr(reliability[column], reducer)()
        add_ledger(ledger, name, "reliability_fault_localization_summary.csv", f"{reducer} {column}", value)
    critical_attention = attention.loc[attention["zone"] == "critical_rul_le_30"].iloc[0]
    noncritical_attention = attention.loc[attention["zone"] == "noncritical_rul_gt_30"].iloc[0]
    for name, row in (("CriticalAttentionMass", critical_attention), ("NoncriticalAttentionMass", noncritical_attention)):
        value = float(row["attention_end_third_mass_mean"])
        macros.append(macro(name, f"{value:.3f}"))
        add_ledger(ledger, name, "attention_mechanism_summary.csv", f"zone={row['zone']}", value)

    selected_risk = risk.sort_values("best_val_risk_score").iloc[0]
    macros.extend(
        [
            macro("SelectedLateLifeWeight", f"{float(selected_risk['late_life_weight']):.2f}"),
            macro("SelectedLateOverWeight", f"{float(selected_risk['late_over_weight']):.2f}"),
        ]
    )
    add_ledger(ledger, "SelectedLateLifeWeight", "risk_hyperparameter_search.csv", "minimum selection_score", selected_risk["late_life_weight"])
    add_ledger(ledger, "SelectedLateOverWeight", "risk_hyperparameter_search.csv", "minimum selection_score", selected_risk["late_over_weight"])

    checkpoint_index = checkpoint_summary.set_index("checkpoint_variant")
    for prefix, variant in (("RiskCheckpoint", "risk_score_checkpoint"), ("RMSECheckpoint", "rmse_checkpoint")):
        row = checkpoint_index.loc[variant]
        for suffix, column in (
            ("RMSE", "rmse_mean"),
            ("NASAperEngine", "nasa_per_engine_mean"),
            ("LPRThirty", "lpr30_mean"),
            ("SLPRTen", "slpr30_10_mean"),
        ):
            value = float(row[column])
            macros.append(macro(f"{prefix}{suffix}", f"{value:.3f}"))
            add_ledger(
                ledger,
                f"{prefix}{suffix}",
                "checkpoint_sensitivity_summary.csv",
                f"checkpoint_variant={variant}; column={column}",
                value,
            )
    checkpoint_bootstrap_index = checkpoint_bootstrap.set_index("metric")
    for metric_name, suffix in (("rmse", "RMSE"), ("lpr30", "LPRThirty")):
        row = checkpoint_bootstrap_index.loc[metric_name]
        for macro_suffix, column in (
            ("Difference", "mean_difference_risk_minus_rmse_checkpoint"),
            ("CILow", "percentile_ci95_low"),
            ("CIHigh", "percentile_ci95_high"),
        ):
            value = float(row[column])
            name = f"Checkpoint{suffix}{macro_suffix}"
            macros.append(macro(name, f"{value:.3f}"))
            add_ledger(
                ledger,
                name,
                "checkpoint_sensitivity_paired_bootstrap.csv",
                f"metric={metric_name}; column={column}",
                value,
            )

    protocol_ci_low = protocol_track_bootstrap["percentile_ci95_low"].astype(float)
    protocol_ci_high = protocol_track_bootstrap["percentile_ci95_high"].astype(float)
    protocol_total = int(len(protocol_track_bootstrap))
    protocol_conventional = int((protocol_ci_low > 0.0).sum())
    protocol_common = int((protocol_ci_high < 0.0).sum())
    protocol_unresolved = protocol_total - protocol_conventional - protocol_common
    for name, value in (
        ("ProtocolTotalComparisonCount", protocol_total),
        ("ProtocolConventionalResolvedCount", protocol_conventional),
        ("ProtocolCommonResolvedCount", protocol_common),
        ("ProtocolUnresolvedCount", protocol_unresolved),
    ):
        macros.append(macro(name, str(value)))
        add_ledger(
            ledger,
            name,
            "standard_protocol_track_paired_bootstrap.csv",
            "count based on the sign of the paired 95% interval",
            value,
        )

    rast_protocol = protocol_track.loc[protocol_track["model"] == "rast_gru"].iloc[0]
    for suffix, column in (
        ("RMSEDelta", "rmse_conventional_minus_common"),
        ("NASADelta", "nasa_per_engine_conventional_minus_common"),
        ("LPRDelta", "lpr30_conventional_minus_common"),
    ):
        value = float(rast_protocol[column])
        name = f"RASTProtocol{suffix}"
        macros.append(macro(name, f"{value:.3f}"))
        add_ledger(
            ledger,
            name,
            "standard_protocol_track_comparison.csv",
            f"model=rast_gru; column={column}",
            value,
        )

    ocm_protocol = protocol_track.loc[protocol_track["model"] == PROPOSED].iloc[0]
    for suffix, column in (
        ("RMSEDelta", "rmse_conventional_minus_common"),
        ("NASADelta", "nasa_per_engine_conventional_minus_common"),
        ("LPRDelta", "lpr30_conventional_minus_common"),
    ):
        value = float(ocm_protocol[column])
        name = f"OCMProtocol{suffix}"
        macros.append(macro(name, f"{value:.3f}"))
        add_ledger(
            ledger,
            name,
            "standard_protocol_track_comparison.csv",
            f"model={PROPOSED}; column={column}",
            value,
        )

    attribution_index = attribution_summary.set_index("variant")
    for variant, prefix in (("base_only", "OCMBaseOnly"), ("plus_risk", "OCMPlusRisk"),
                            ("plus_risk_cons", "OCMPlusRiskCons"), ("full", "OCMFull")):
        row = attribution_index.loc[variant]
        for suffix, column in (("RMSE", "rmse_mean"), ("NASA", "nasa_per_engine_mean"), ("LPR", "lpr30_mean")):
            value = float(row[column])
            name = f"{prefix}{suffix}"
            macros.append(macro(name, f"{value:.3f}"))
            add_ledger(ledger, name, "ocm_attribution_summary.csv", f"variant={variant}; column={column}", value)

    architecture_index = architecture_absolute.set_index(["model_display", "protocol"])
    rast_common = architecture_index.loc[("RAST-GRU", "Common Base")]
    ocm_common = architecture_index.loc[("OCM backbone", "Common Base")]
    for name, column in (
        ("CommonBaseArchitectureRMSEDelta", "rmse_mean"),
        ("CommonBaseArchitectureLPRDelta", "lpr30_mean"),
        ("CommonBaseArchitectureCVaRDelta", "late_cvar95_30_mean"),
    ):
        value = float(ocm_common[column] - rast_common[column])
        macros.append(macro(name, f"{value:.3f}"))
        add_ledger(
            ledger,
            name,
            "architecture_attribution_absolute.csv",
            f"OCM backbone minus RAST-GRU under Common Base; column={column}",
            value,
        )
    track_b_ocm = track_b_absolute.loc[track_b_absolute["model"] == PROPOSED].iloc[0]
    for name, column in (
        ("TrackBOCMRMSERank", "rmse_rank"),
        ("TrackBOCMNASARank", "nasa_per_engine_rank"),
        ("TrackBOCMLPRRank", "lpr30_rank"),
    ):
        value = int(round(float(track_b_ocm[column])))
        macros.append(macro(name, value))
        add_ledger(ledger, name, "track_b_absolute_ranking.csv", f"model={PROPOSED}; column={column}", value)

    rast_repeat_max = float(
        fast_cudnn_repeat.loc[
            fast_cudnn_repeat["model"] == "rast_gru", "repeat_to_cross_seed_sd_ratio"
        ].max()
    )
    transformer_repeat_sd_max = float(
        fast_cudnn_repeat.loc[
            fast_cudnn_repeat["model"] == "transformer_lite", "fixed_seed_repeat_sd_mean"
        ].max()
    )
    macros.append(macro("FastCudnnRASTMaxRatio", f"{rast_repeat_max:.3f}"))
    macros.append(macro("FastCudnnTransformerMaxRepeatSD", f"{transformer_repeat_sd_max:.4f}"))
    add_ledger(
        ledger,
        "FastCudnnRASTMaxRatio",
        "fast_cudnn_repeat_summary.csv",
        "maximum repeat_to_cross_seed_sd_ratio where model=rast_gru",
        rast_repeat_max,
    )
    add_ledger(
        ledger,
        "FastCudnnTransformerMaxRepeatSD",
        "fast_cudnn_repeat_summary.csv",
        "maximum fixed_seed_repeat_sd_mean where model=transformer_lite",
        transformer_repeat_sd_max,
    )

    (output / "numbers.tex").write_text("\n".join(macros) + "\n", encoding="utf-8", newline="\n")
    pd.DataFrame(ledger).to_csv(output / "evidence_ledger.csv", index=False)
    write_training_algorithm(output / "algorithm_training_inference.tex")
    write_training_algorithm_zh(output / "algorithm_training_inference_zh.tex")
    write_results_sections(output)

    benchmark_table = aggregation.copy()
    benchmark_table.loc[benchmark_table["model"] == PROPOSED, "model_display"] = "OCM-MST-GRU-Asym"
    core_macro = core_asym.loc[core_asym["variant"] == "core"].iloc[0]
    benchmark_table = pd.concat(
        [
            benchmark_table,
            pd.DataFrame(
                [
                    {
                        "model": "rast_gru_v2_core",
                        "model_display": "OCM-MST-GRU-Core [post-lock]",
                        "macro_rmse_mean": core_macro["rmse_mean"],
                        "macro_rmse_std": core_macro["rmse_std"],
                        "macro_nasa_per_engine_mean": core_macro["nasa_per_engine_mean"],
                        "macro_nasa_per_engine_std": core_macro["nasa_per_engine_std"],
                        "macro_lpr30_mean": core_macro["lpr30_mean"],
                        "macro_lpr30_std": core_macro["lpr30_std"],
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    benchmark_table["evidence_order"] = (benchmark_table["model"] == "rast_gru_v2_core").astype(int)
    benchmark_table = benchmark_table.sort_values(["evidence_order", "macro_rmse_mean"])
    for stem in ("macro_rmse", "macro_nasa_per_engine", "macro_lpr30"):
        benchmark_table[f"{stem}_display"] = benchmark_table.apply(
            lambda row, s=stem: f"{row[f'{s}_mean']:.3f} $\\pm$ {row[f'{s}_std']:.3f}", axis=1
        )
    write_booktabs_table(
        benchmark_table,
        output / "table_cmapss_macro.tex",
        columns=[
            ("model_display", "Model", "text"),
            ("macro_rmse_display", "RMSE", "raw"),
            ("macro_nasa_per_engine_display", "NASA/engine", "raw"),
            ("macro_lpr30_display", "LPR@30", "raw"),
        ],
        caption="Five-seed macro-averaged performance across C-MAPSS FD001--FD004.",
        label="tab:v3-cmapss-macro",
        note="Values are mean $\\pm$ sample standard deviation over five composite seeds after equal weighting of FD001--FD004. Asym belongs to the fixed 260-run grid; Core is a retrospective preference sensitivity and is labeled separately. Lower is better. MAE is retained in the source table and supplement.",
    )

    fairness_table = efficiency_enhanced.sort_values("parameters_fd004").copy()
    fairness_table["gpu_ms_display"] = fairness_table.apply(
        lambda row: f"{row['gpu_ms_sample_mean']:.3f} $\\pm$ {row['gpu_ms_sample_std']:.3f}", axis=1
    )
    fairness_table["cpu_ms_display"] = fairness_table.apply(
        lambda row: f"{row['cpu_ms_sample_mean']:.3f} $\\pm$ {row['cpu_ms_sample_std']:.3f}", axis=1
    )
    fairness_table["training_min_display"] = fairness_table.apply(
        lambda row: f"{row['training_minutes_mean']:.1f} $\\pm$ {row['training_minutes_std']:.1f}", axis=1
    )
    fairness_table["peak_mb_display"] = fairness_table.apply(
        lambda row: f"{row['peak_gpu_memory_mb_mean']:.1f} $\\pm$ {row['peak_gpu_memory_mb_std']:.1f}", axis=1
    )
    fairness_table["checkpoint_epoch_display"] = fairness_table.apply(
        lambda row: f"{row['checkpoint_epoch_mean']:.1f} $\\pm$ {row['checkpoint_epoch_std']:.1f}", axis=1
    )
    fairness_table["executed_epoch_display"] = fairness_table.apply(
        lambda row: f"{row['executed_epochs_mean']:.1f} $\\pm$ {row['executed_epochs_std']:.1f}", axis=1
    )
    fairness_table["seconds_epoch_display"] = fairness_table.apply(
        lambda row: f"{row['seconds_per_epoch_mean']:.2f} $\\pm$ {row['seconds_per_epoch_std']:.2f}", axis=1
    )
    write_booktabs_table(
        fairness_table,
        output / "table_efficiency.tex",
        columns=[
            ("model_display", "Model", "text"),
            ("parameters_fd004", "Parameters", "int"),
            ("gpu_ms_display", "GPU batch-1 ms", "raw"),
            ("checkpoint_epoch_display", "Checkpoint epoch", "raw"),
            ("executed_epoch_display", "Executed epochs", "raw"),
            ("seconds_epoch_display", "Seconds/epoch", "raw"),
            ("training_min_display", "Training min", "raw"),
        ],
        caption="FD004 parameter count, measured inference latency, and training cost.",
        label="tab:v3-efficiency",
        note="Values are mean $\\pm$ sample standard deviation over five FD004 runs. Seconds/epoch separates per-epoch computation from convergence and early-stopping effects; training minutes is the measured end-to-end optimization time. GPU latency excludes preprocessing and transfer.",
        fit_width=True,
    )

    fairness_protocol = fairness.sort_values("parameters_fd004").copy()
    for column in (
        "receives_mask_channels",
        "explicit_mask_processing_module",
        "uses_shared_degradation_augmentation",
    ):
        fairness_protocol[column] = fairness_protocol[column].map({True: "Yes", False: "No"})
    write_booktabs_table(
        fairness_protocol,
        output / "table_benchmark_fairness.tex",
        columns=[
            ("model_display", "Model", "text"),
            ("receives_mask_channels", "Mask input", "text"),
            ("explicit_mask_processing_module", "Explicit mask module", "text"),
            ("objective_code", "Objective", "text"),
        ],
        caption="Controlled-protocol benchmark fairness and model-specific objective terms.",
        label="tab:v3-benchmark-fairness",
        note="Every model receives the same value and observed-mask input channels, training-only degradation augmentation, base loss, data split, and checkpoint budget. Only OCM-MST-GRU contains an explicit mask-conditioned reliability module. Objective codes: Q, quantile pinball; Risk, smooth late-risk; Rel, reliability supervision; Cons, two-view consistency. Seeds are composite and jointly control split, initialization, data order, and augmentation. This is a controlled common-protocol comparison, not a native reproduction of every baseline paper.",
    )

    checkpoint_display = checkpoint_summary.copy()
    checkpoint_display["variant_display"] = checkpoint_display["checkpoint_variant"].map(
        {
            "risk_score_checkpoint": "Risk-score checkpoint",
            "rmse_checkpoint": "Validation-RMSE checkpoint",
        }
    ).fillna(checkpoint_display["checkpoint_variant"])
    write_booktabs_table(
        checkpoint_display,
        output / "table_checkpoint_sensitivity.tex",
        columns=[
            ("variant_display", "Checkpoint rule", "text"),
            ("rmse_mean", "RMSE", ".3f"),
            ("nasa_per_engine_mean", "NASA/engine", ".3f"),
            ("lpr30_mean", "LPR@30", ".3f"),
            ("slpr30_10_mean", "SLPR@30,10", ".3f"),
            ("seed_count", "Seeds", "int"),
        ],
        caption="Five-seed FD004 checkpoint-rule sensitivity for OCM-MST-GRU-Asym.",
        label="tab:v3-checkpoint-sensitivity",
        note="Only checkpoint selection and its associated early stopping change. The training objective, data split, initialization, augmentation stream, model, and budget remain matched. Paired seed-and-engine bootstrap intervals are generated separately and govern the interpretation.",
    )

    protocol_display = protocol_track.copy()
    protocol_display["model_display"] = protocol_display["model"].map(model_display_name)
    write_booktabs_table(
        protocol_display,
        output / "table_protocol_track_sensitivity.tex",
        columns=[
            ("model_display", "Model", "text"),
            ("rmse_conventional_minus_common", r"$\Delta$ RMSE", ".3f"),
            ("nasa_per_engine_conventional_minus_common", r"$\Delta$ NASA/engine", ".3f"),
            ("lpr30_conventional_minus_common", r"$\Delta$ LPR@30", ".3f"),
        ],
        caption="FD004 common-risk versus standardized conventional protocol sensitivity.",
        label="tab:v3-protocol-track",
        note="Each entry is standardized conventional minus common-risk protocol, averaged over five matched composite seeds. Negative values favor the conventional protocol for a lower-is-better metric. Removing mask channels changes the input-layer parameter count, so this is an end-to-end protocol sensitivity rather than an isolated training-rule effect. The paired seed-and-engine bootstrap file, not the sign alone, determines whether a difference is resolved.",
    )

    track_b_display = track_b_absolute.copy()
    track_b_display["model_display"] = track_b_display["model"].map(model_display_name)
    track_b_display["mean_rank"] = track_b_display[
        ["rmse_rank", "nasa_per_engine_rank", "lpr30_rank"]
    ].mean(axis=1)
    for stem in ("rmse", "nasa_per_engine", "lpr30"):
        track_b_display[f"{stem}_display"] = track_b_display.apply(
            lambda row, s=stem: f"{row[f'{s}_mean']:.3f} $\\pm$ {row[f'{s}_std']:.3f}", axis=1
        )
    write_booktabs_table(
        track_b_display.sort_values("mean_rank"),
        output / "table_track_b_absolute.tex",
        columns=[
            ("model_display", "Model", "text"),
            ("rmse_display", "RMSE", "raw"),
            ("nasa_per_engine_display", "NASA/engine", "raw"),
            ("lpr30_display", "LPR@30", "raw"),
            ("mean_rank", "Mean rank", ".2f"),
        ],
        caption="Absolute FD004 results within the standardized conventional Track B.",
        label="tab:v3-track-b-absolute",
        note="Values are mean $\\pm$ sample standard deviation over five matched composite seeds. Mean rank averages RMSE, NASA/engine, and LPR@30 ranks within this six-model track. Track B is a controlled protocol-sensitivity analysis, not a native-method leaderboard.",
    )

    architecture_display = architecture_absolute.copy()
    architecture_display["model_display"] = architecture_display["model_display"].replace(
        {"Full OCM": "OCM-Asym (full objectives)", "OCM": "OCM backbone"}
    )
    for stem in ("rmse", "nasa_per_engine", "lpr30", "late_cvar95_30"):
        architecture_display[f"{stem}_display"] = architecture_display.apply(
            lambda row, s=stem: f"{row[f'{s}_mean']:.3f} $\\pm$ {row[f'{s}_std']:.3f}", axis=1
        )
    write_booktabs_table(
        architecture_display,
        output / "table_architecture_attribution_absolute.tex",
        columns=[
            ("model_display", "Architecture", "text"),
            ("protocol", "Protocol", "text"),
            ("rmse_display", "RMSE", "raw"),
            ("nasa_per_engine_display", "NASA/engine", "raw"),
            ("lpr30_display", "LPR@30", "raw"),
            ("late_cvar95_30_display", "CVaR95", "raw"),
        ],
        caption="FD004 architecture and protocol attribution with absolute results.",
        label="tab:v3-architecture-absolute",
        note="Common Base holds the study-defined common-risk preprocessing, asymmetric base loss, augmentation, budget, and checkpoint rule fixed while excluding OCM-specific auxiliary objectives. Track B uses the standardized conventional bundle. OCM-Asym (full objectives) adds the declared OCM objectives. Neither track is a protocol-neutral architecture leaderboard. Lower is better.",
        fit_width=True,
    )

    architecture_ci = architecture_bootstrap.copy()
    architecture_ci["comparison_display"] = architecture_ci["comparison"].map(
        {
            "common_base_architecture": "Common Base: OCM backbone - RAST",
            "track_b_strongest_competitor": "Track B: OCM backbone - strongest competitor",
        }
    )
    architecture_ci["ci_display"] = architecture_ci.apply(
        lambda row: f"{row['mean_difference_left_minus_right']:.3f} "
        f"[{row['percentile_ci95_low']:.3f}, {row['percentile_ci95_high']:.3f}]",
        axis=1,
    )
    write_booktabs_table(
        architecture_ci,
        output / "table_architecture_selected_comparators.tex",
        columns=[
            ("comparison_display", "Configuration comparison", "text"),
            ("metric", "Metric", "text"),
            ("ci_display", "Left $-$ right [95\\% CI]", "raw"),
            ("probability_left_lower", "Pr(left lower)", ".3f"),
        ],
        caption="Paired FD004 configuration evidence under matched protocols.",
        label="tab:v3-architecture-ci",
        note="Intervals jointly resample matched composite seeds and engine endpoints. Negative differences favor the left configuration for all reported lower-is-better metrics. The strongest Track B competitor is selected from the same test summary by mean rank across RMSE, NASA/engine, and LPR@30; that comparison is exploratory post-selection evidence, not confirmatory inference.",
        font_command=r"\footnotesize",
        fit_width=True,
    )
    write_booktabs_table(
        architecture_ci,
        output / "table_architecture_selected_comparators_zh.tex",
        columns=[
            ("comparison_display", "配置对照", "text"),
            ("metric", "指标", "text"),
            ("ci_display", "左项$-$右项 [95\\% CI]", "raw"),
            ("probability_left_lower", "Pr(左项更低)", ".3f"),
        ],
        caption="匹配协议下的 FD004 配置配对证据。",
        label="tab:v3-architecture-ci-zh",
        note="区间联合重采样匹配的复合种子与发动机端点。对于所有越低越好的指标，负差值支持左侧配置。Track B 的最强竞争者依据同一测试摘要中 RMSE、NASA/engine 与 LPR@30 的平均秩选出，因此该对照属于探索性的选择后证据，不作验证性推断。",
        font_command=r"\footnotesize",
        fit_width=True,
    )

    core_display = core_asym.copy()
    core_display["variant_display"] = core_display["variant"].map(
        {"core": "OCM-MST-GRU-Core", "asym": "OCM-MST-GRU-Asym"}
    )
    for stem in ("rmse", "nasa_per_engine", "lpr30", "late_cvar95_30"):
        core_display[f"{stem}_display"] = core_display.apply(
            lambda row, s=stem: f"{row[f'{s}_mean']:.3f} $\\pm$ {row[f'{s}_std']:.3f}", axis=1
        )
    write_booktabs_table(
        core_display,
        output / "table_core_asym_cmapss.tex",
        columns=[
            ("variant_display", "Variant", "text"),
            ("rmse_display", "RMSE", "raw"),
            ("nasa_per_engine_display", "NASA/engine", "raw"),
            ("lpr30_display", "LPR@30", "raw"),
            ("late_cvar95_30_display", "CVaR95", "raw"),
        ],
        caption="Five-seed C-MAPSS macro comparison of the Core and asymmetric late-error preferences.",
        label="tab:v3-core-asym",
        note="Core sets only the late-overprediction multiplier to 1.0; Asym uses the researcher-specified illustrative value 1.5. All four subsets receive equal weight within each seed. This is a fixed-task preference sensitivity, not a selection-stability or generalization test.",
    )

    core_ci = core_asym_bootstrap.copy()
    core_ci["effect_ci"] = core_ci.apply(
        lambda row: f"{row['mean_difference_core_minus_asym']:.3f} "
        f"[{row['percentile_ci95_low']:.3f}, {row['percentile_ci95_high']:.3f}]",
        axis=1,
    )
    write_booktabs_table(
        core_ci,
        output / "table_core_asym_paired.tex",
        columns=[
            ("metric", "Metric", "text"),
            ("effect_ci", "Core$-$Asym [95\\% CI]", "raw"),
            ("probability_core_lower", "Pr(Core lower)", ".3f"),
            ("probability_asym_lower", "Pr(Asym lower)", ".3f"),
        ],
        caption="Paired five-seed evidence for Core versus asymmetric preference across fixed C-MAPSS tasks.",
        label="tab:v3-core-asym-ci",
        note="Each bootstrap replicate resamples composite seeds and paired engines within each of the four fixed subsets, then gives subsets equal weight. Negative differences favor Core. The subsets are not treated as random draws from a task population.",
    )

    endpoint_rows = []
    for subset, group in endpoint_audit.groupby("subset", sort=True):
        endpoint_rows.append(
            {
                "subset": subset,
                "validation_engines": f"{int(group.validation_engines.min())}-{int(group.validation_engines.max())}",
                "n30": f"{int(group.n30.min())}-{int(group.n30.max())}",
                "n10": f"{int(group.n10.min())}-{int(group.n10.max())}",
                "rul_mean_range": f"{group.truncation_rul_mean.min():.1f}-{group.truncation_rul_mean.max():.1f}",
                "rul_range": f"{group.truncation_rul_min.min():.0f}-{group.truncation_rul_max.max():.0f}",
            }
        )
    write_booktabs_table(
        pd.DataFrame(endpoint_rows),
        output / "table_validation_endpoint_audit.tex",
        columns=[
            ("subset", "Subset", "text"),
            ("validation_engines", "Val engines", "raw"),
            ("n30", r"$N_{30}$", "raw"),
            ("n10", r"$N_{10}$", "raw"),
            ("rul_mean_range", "Mean RUL range", "raw"),
            ("rul_range", "Observed RUL range", "raw"),
        ],
        caption="Validation-endpoint composition across five composite seeds.",
        label="tab:v3-validation-endpoints",
        note="Ranges are across the five seeds. One cutoff is sampled once per validation engine from eligible cycles with RUL 1--125 using the composite seed; the resulting endpoint set is fixed for every epoch. Full unit/cycle fingerprints are provided in the artifact.",
    )

    attribution_display = attribution_summary.copy()
    attribution_order = ["base_only", "plus_risk", "plus_risk_cons", "full"]
    attribution_display["variant"] = pd.Categorical(
        attribution_display["variant"], categories=attribution_order, ordered=True
    )
    attribution_display = attribution_display.sort_values("variant").reset_index(drop=True)
    attribution_display["variant_display"] = attribution_display["variant"].map(
        {"base_only": "OCM backbone + Base", "plus_risk": "+ Risk", "plus_risk_cons": "+ Risk + Cons",
         "full": "OCM-Asym full (+ Rel + Q)"}
    )
    write_booktabs_table(
        attribution_display,
        output / "table_ocm_attribution.tex",
        columns=[
            ("variant_display", "Build-up stage", "text"),
            ("rmse_mean", "RMSE", ".3f"),
            ("nasa_per_engine_mean", "NASA/engine", ".3f"),
            ("lpr30_mean", "LPR@30", ".3f"),
            ("late_cvar95_30_mean", "Late CVaR95", ".3f"),
            ("seed_count", "Seeds", "int"),
        ],
        caption="Five-seed FD004 build-up attribution for OCM-specific objectives.",
        label="tab:v3-ocm-attribution",
        note="The OCM backbone, common preprocessing, input, augmentation, asymmetric base loss, budget, and risk-score checkpoint remain fixed. Stages add the smooth late-risk term, consistency regularization, and finally reliability supervision plus q10/q90 pinball terms. Sequential and Base-to-Full paired intervals are stored separately; mean changes alone are not interpreted as causal component effects.",
    )
    write_booktabs_table(
        attribution_display,
        output / "table_ocm_attribution_zh.tex",
        columns=[
            ("variant_display", "逐步构建阶段", "text"),
            ("rmse_mean", "RMSE", ".3f"),
            ("nasa_per_engine_mean", "NASA/engine", ".3f"),
            ("lpr30_mean", "LPR@30", ".3f"),
            ("late_cvar95_30_mean", "晚误差 CVaR95", ".3f"),
            ("seed_count", "种子", "int"),
        ],
        caption="OCM 特有目标项的 FD004 五种子逐步归因。",
        label="tab:v3-ocm-attribution-zh",
        note="OCM 主干、共同预处理、输入、增强、非对称基础损失、预算和风险检查点保持不变；各阶段依次加入平滑晚预测风险、一致性正则，最后加入可靠性监督与 q10/q90 pinball 项。顺序比较及 Base-to-Full 配对区间单独保存，不能只按均值变化解释组件因果效应。",
    )

    hyper_display = hyperparameter_sensitivity.copy()
    hyper_display["family_display"] = hyper_display["family"].map(
        {"cluster_k": "Condition clusters K", "aux_scale": "Auxiliary scale"}
    )
    write_booktabs_table(
        hyper_display,
        output / "table_hyperparameter_sensitivity.tex",
        columns=[
            ("family_display", "Sensitivity family", "text"),
            ("value", "Value", ".1f"),
            ("rmse_mean", "RMSE", ".3f"),
            ("nasa_per_engine_mean", "NASA/engine", ".3f"),
            ("lpr30_mean", "LPR@30", ".3f"),
            ("seed_count", "Seeds", "int"),
        ],
        caption="Five-seed FD004 sensitivity to condition-cluster count and joint auxiliary-loss scale.",
        label="tab:v3-hyperparameter-sensitivity",
        note="K=6 and auxiliary scale 1.0 are the study-defined reference. Alternative values are retrospective sensitivity checks and are not used to retune the main benchmark.",
    )

    write_booktabs_table(
        cluster_k_diagnostics,
        output / "table_condition_k_diagnostics.tex",
        columns=[
            ("k", "K", "int"),
            ("silhouette_mean", "Silhouette", ".3f"),
            ("silhouette_std", "Silhouette SD", ".3f"),
            ("ari_vs_k6_mean", "ARI vs K=6", ".3f"),
            ("ari_vs_declared_setting_combinations_mean", "ARI vs settings", ".3f"),
            ("cross_seed_ari_mean", "Cross-seed ARI", ".3f"),
            ("minimum_cluster_fraction_mean", "Min. cluster share", ".3f"),
        ],
        caption="Training-core-only FD004 operating-condition clustering diagnostics.",
        label="tab:v3-condition-k-diagnostics",
        note="Diagnostics use only the three operating-setting variables from the training core. ARI vs settings compares cluster labels with the six discrete combinations obtained by rounding those variables to the nearest integer; it is a descriptive correspondence audit, not a selection target. Silhouette uses a deterministic sample of at most 10,000 rows per seed.",
    )

    write_booktabs_table(
        late_tail.sort_values("macro_late_cvar95_30_mean"),
        output / "table_late_tail_risk.tex",
        columns=[
            ("model_display", "Model", "text"),
            ("macro_mle30_mean", "MLE@30", ".3f"),
            ("macro_conditional_mle30_mean", "Cond. MLE@30", ".3f"),
            ("macro_late_q90_30_mean", "Late Q90", ".3f"),
            ("macro_late_q95_30_mean", "Late Q95", ".3f"),
            ("macro_late_cvar95_30_mean", "Late CVaR95", ".3f"),
        ],
        caption="Near-failure positive-error tail risk across C-MAPSS subsets.",
        label="tab:v3-late-tail-risk",
        note="MLE@30 averages max(predicted RUL minus true RUL, 0) over engines with true RUL at most 30 cycles. Conditional MLE and Q90, Q95, and CVaR95 condition on strictly positive late errors. Subsets receive equal weight within each of five composite seeds; lower is better.",
    )

    threshold_direct = threshold_paired[
        (threshold_paired["subset"] == "FD004")
        & (threshold_paired["baseline_model"] == "rast_gru")
    ].copy()
    write_booktabs_table(
        threshold_direct.sort_values(["rul_threshold", "late_error_threshold"]),
        output / "table_threshold_paired_ci.tex",
        columns=[
            ("metric_name", "Metric", "text"),
            ("rul_threshold", "RUL threshold", ".0f"),
            ("late_error_threshold", "Late margin", ".0f"),
            ("paired_difference_baseline_minus_proposed", "Paired difference", ".4f"),
            ("percentile_ci95_low", "CI low", ".4f"),
            ("percentile_ci95_high", "CI high", ".4f"),
            ("probability_proposed_lower", r"Pr($\Delta>0$)", ".3f"),
            ("probability_tie", r"Pr($\Delta=0$)", ".3f"),
            ("probability_proposed_higher", r"Pr($\Delta<0$)", ".3f"),
        ],
        caption="FD004 threshold sensitivity with paired uncertainty against RAST-GRU.",
        label="tab:v3-threshold-paired-ci",
        note=r"Here $\Delta$ is baseline minus OCM, so positive values favor OCM. Exact bootstrap ties are reported separately instead of being counted as evidence against either model. This is a sensitivity analysis over predeclared thresholds, not a search for a favorable threshold.",
    )

    ablation_display = ablation.copy()
    ablation_display["test_nasa_per_engine_mean"] = (
        ablation_display["test_nasa_score_mean"].astype(float) / 248.0
    )
    ablation_display["variant_display"] = ablation_display["ablation_variant"].replace(
        {
            "full": "OCM-Asym (full)",
            "weighted_huber_no_asymmetry": r"OCM-Core ($\lambda_{\mathrm{late}}=1$)",
            "no_condition_norm": "w/o condition normalization",
            "no_missing_mask": "w/o missing-mask channels",
            "no_reliability_gate": "w/o reliability gate",
            "no_reliability_supervision": "w/o reliability supervision",
            "no_channel_gate": "w/o channel gate",
            "no_local_trend": "w/o local-trend branch",
            "no_temporal_attention": "w/o temporal attention",
            "no_degradation_aug": "w/o degradation augmentation",
            "no_smooth_late_risk": "w/o smooth late-risk term",
            "no_consistency_regularization": "w/o consistency regularization",
            "no_quantile_calibration": "w/o quantile auxiliary loss",
        }
    )
    write_booktabs_table(
        ablation_display.sort_values("test_rmse_mean"),
        output / "table_ablation_fd004.tex",
        columns=[
            ("variant_display", "Variant", "text"),
            ("test_rmse_mean", "RMSE", ".3f"),
            ("test_nasa_per_engine_mean", "NASA/engine", ".3f"),
            ("test_critical_30_late_prediction_ratio_mean", "LPR@30", ".3f"),
            ("test_critical_30_severe_late_10_ratio_mean", "SLPR@30,10", ".3f"),
            ("seed_count", "Seeds", "int"),
        ],
        caption="Five-seed FD004 module and objective ablation.",
        label="tab:v3-ablation",
        note="The full row directly references the five formal OCM-MST-GRU checkpoints. Every non-full variant changes only its named switch and otherwise retains the formal budget, composite seed, split, augmentation stream, and checkpoint rule. Lower is better; mixed effects are interpreted by module function rather than by RMSE alone.",
    )

    stress_metric_names = {
        "rmse": "RMSE",
        "critical_30_late_prediction_ratio": "LPR@30",
        "critical_30_mean_late_excess": "MLE@30",
        "critical_30_late_cvar95": "Late CVaR95",
        "critical_30_decision_cost_5": r"Asymmetric error cost ($\kappa=5$)",
    }
    stress_metric_names_zh = {
        "rmse": "RMSE",
        "critical_30_late_prediction_ratio": "LPR@30",
        "critical_30_mean_late_excess": "MLE@30",
        "critical_30_late_cvar95": "晚误差 CVaR95",
        "critical_30_decision_cost_5": r"非对称误差成本 ($\kappa=5$)",
    }
    stress_summary_rows = []
    for metric, group in stress_pair[stress_pair["metric"].isin(stress_metric_names)].groupby("metric", sort=False):
        differences = group["ocm_minus_rast"].astype(float)
        stress_summary_rows.append(
            {
                "metric_display": stress_metric_names[metric],
                "metric_display_zh": stress_metric_names_zh[metric],
                "ocm_lower_count": int((differences < 0.0).sum()),
                "rast_lower_count": int((differences > 0.0).sum()),
                "median_difference": float(differences.median()),
                "worst_difference": float(differences.max()),
            }
        )
    stress_selected = pd.DataFrame(stress_summary_rows)
    write_booktabs_table(
        stress_selected,
        output / "table_stress_auc_direct_predecessor.tex",
        columns=[
            ("metric_display", "Curve-mean metric", "raw"),
            ("ocm_lower_count", "Mean favors Asym", "int"),
            ("rast_lower_count", "Mean favors RAST", "int"),
            ("median_difference", r"Median $\Delta$", ".3f"),
            ("worst_difference", r"Worst $\Delta$", ".3f"),
        ],
        caption="Descriptive FD004 sign counts for nine normalized simulated fault-response curve means.",
        label="tab:v3-stress-auc",
        note=r"These are descriptive directions of mean point estimates, not interval-supported classifications. Each curve mean uses trapezoidal integration over the declared family-specific intensity grid, including clean. $\Delta$ is Asym minus RAST, so negative values favor Asym; the separate interval table provides inferential classifications.",
    )
    write_booktabs_table(
        stress_selected,
        output / "table_stress_auc_direct_predecessor_zh.tex",
        columns=[
            ("metric_display_zh", "归一化曲线均值指标", "raw"),
            ("ocm_lower_count", "OCM 较低", "int"),
            ("rast_lower_count", "RAST 较低", "int"),
            ("median_difference", r"中位 $\Delta$", ".3f"),
            ("worst_difference", r"最差 $\Delta$", ".3f"),
        ],
        caption="相对直接前身的 FD004 九类故障归一化退化曲线汇总。",
        label="tab:v3-stress-auc-zh",
        note=r"曲线均值是在各场景声明的强度区间上进行梯形积分并除以该区间长度，且包含强度为零的 clean 点。每行汇总九类故障、五个训练种子与五个匹配扰动种子；$\Delta$ 为 OCM 减 RAST，负值有利于 OCM。",
    )

    family_primary = stress_engine_bootstrap.copy()
    family_primary["metric_display"] = family_primary["metric"].map(
        {
            "rmse": "RMSE",
            "critical_30_late_prediction_ratio": "LPR@30",
            "critical_30_signed_error_mean": "Signed error mean@30",
            "critical_30_early_error_magnitude": "Early-error magnitude@30",
            "critical_30_mean_late_excess": "MLE@30",
            "critical_30_late_cvar95": "Late CVaR95",
            "critical_30_decision_cost_2": "Asymmetric error cost (kappa=2)",
            "critical_30_decision_cost_5": "Asymmetric error cost (kappa=5)",
            "critical_30_decision_cost_10": "Asymmetric error cost (kappa=10)",
        }
    )
    family_primary["effect_ci"] = family_primary.apply(
        lambda row: f"{row['mean_difference_point_minus_rast']:.3f} "
        f"[{row['hierarchical_bootstrap_ci95_low']:.3f}, {row['hierarchical_bootstrap_ci95_high']:.3f}]",
        axis=1,
    )
    write_stress_family_longtable(
        family_primary.sort_values(["operating_point", "scenario", "metric"]),
        output / "table_stress_family_uncertainty.tex",
    )

    equivalence_summary = (
        stress_equivalence.groupby(["metric", "practical_equivalence_margin"], as_index=False)
        .agg(
            minimum_probability=("probability_within_margin", "min"),
            median_probability=("probability_within_margin", "median"),
            maximum_probability=("probability_within_margin", "max"),
        )
    )
    equivalence_summary["metric_display"] = equivalence_summary["metric"].map(
        {"rmse": "RMSE", "critical_30_late_prediction_ratio": "LPR@30"}
    )
    write_booktabs_table(
        equivalence_summary,
        output / "table_stress_equivalence_sensitivity.tex",
        columns=[
            ("metric_display", "Metric", "text"),
            ("practical_equivalence_margin", "Candidate margin", ".3f"),
            ("minimum_probability", "Min Pr(within)", ".3f"),
            ("median_probability", "Median Pr(within)", ".3f"),
            ("maximum_probability", "Max Pr(within)", ".3f"),
        ],
        caption="Descriptive sensitivity of within-margin probabilities to arbitrary reporting margins.",
        label="tab:v3-within-margin-sensitivity",
        note="Bootstrap fractions are summarized across the nine declared algorithmic perturbation families. These arbitrary reporting margins (RMSE: 0.25, 0.50, 1.00 cycles; LPR: 0.01, 0.02, 0.05) are not engineering equivalence bounds, do not define practical equivalence, and are not used for confirmatory classification.",
    )

    external_table = external.copy()
    external_table.loc[external_table["model"] == PROPOSED, "model_display"] = "OCM-MST-GRU-Asym"
    core_external = ncmapss_core.iloc[0].copy()
    core_external["model"] = "rast_gru_v2_core"
    core_external["model_display"] = "OCM-MST-GRU-Core"
    external_table = pd.concat([external_table, core_external.to_frame().T], ignore_index=True)
    external_table = external_table.sort_values("unit_macro_rmse_mean")
    for stem in ("unit_macro_rmse", "unit_macro_lpr30"):
        external_table[f"{stem}_display"] = external_table.apply(
            lambda row, s=stem: f"{row[f'{s}_mean']:.3f} $\\pm$ {row[f'{s}_std']:.3f}", axis=1
        )
    write_booktabs_table(
        external_table,
        output / "table_ncmapss.tex",
        columns=[
            ("model_display", "Model", "text"),
            ("unit_macro_rmse_display", "Unit RMSE", "raw"),
            ("unit_macro_lpr30_display", "LPR@30", "raw"),
            ("seed_count", "Seeds", "int"),
        ],
        caption="Exploratory transfer evaluation on the fixed N-CMAPSS DS02 protocol.",
        label="tab:v3-ncmapss",
        note="Values are mean $\\pm$ sample standard deviation across five seeds. Asym is the researcher-specified illustrative point and Core is its retrospective neutral-preference sensitivity. All rows use the same fixed unit split, training-only normalizer, window cache, and budget. Claims are limited to this strongly downsampled DS02 computational proxy.",
    )

    ncmapss_k_display = ncmapss_k_sensitivity.copy()
    for stem in ("unit_macro_rmse", "unit_macro_nasa_per_window", "unit_macro_lpr30"):
        ncmapss_k_display[f"{stem}_display"] = ncmapss_k_display.apply(
            lambda row, s=stem: f"{row[f'{s}_mean']:.3f} $\\pm$ {row[f'{s}_std']:.3f}", axis=1
        )
    write_booktabs_table(
        ncmapss_k_display.sort_values("condition_clusters"),
        output / "table_ncmapss_k_sensitivity.tex",
        columns=[
            ("condition_clusters", "$K$", "int"),
            ("unit_macro_rmse_display", "Unit RMSE", "raw"),
            ("unit_macro_nasa_per_window_display", "NASA/window", "raw"),
            ("unit_macro_lpr30_display", "LPR@30", "raw"),
            ("seed_count", "Seeds", "int"),
        ],
        caption="N-CMAPSS DS02 sensitivity to training-only condition-cluster count.",
        label="tab:v3-ncmapss-k",
        note="$K\\in\\{4,6,8\\}$ changes only the training-fitted condition clustering and normalizer. The fixed unit split, sampled stream, channel set, architecture, objective, budget, and five seeds remain unchanged. This post-lock analysis is not used to retune the main result.",
    )

    fixed_direct = fixed_subset[
        (fixed_subset["baseline_model"] == "rast_gru")
        & (fixed_subset["metric"].isin(["rmse", "nasa_per_engine", "lpr30"]))
    ].copy()
    fixed_direct["metric_display"] = fixed_direct["metric"].map(
        {"rmse": "RMSE", "nasa_per_engine": "NASA/engine", "lpr30": "LPR@30"}
    )
    write_booktabs_table(
        fixed_direct,
        output / "table_fixed_subset_direct.tex",
        columns=[
            ("subset", "Fixed task", "text"),
            ("metric_display", "Metric", "text"),
            ("mean_paired_difference_baseline_minus_proposed", r"$\Delta$ RAST$-$Asym", ".3f"),
            ("percentile_ci95_low", "95\\% CI low", ".3f"),
            ("percentile_ci95_high", "95\\% CI high", ".3f"),
        ],
        caption="Main result-informed fixed-subset engine-level paired estimates for the direct predecessor.",
        label="tab:v3-fixed-subset-direct",
        note="Positive differences favor OCM-MST-GRU-Asym because all metrics are lower-is-better. Each interval resamples matched engines within one fixed C-MAPSS subset and matched composite seeds; it does not treat the four subsets as random draws from a task population.",
    )

    selected_display = selected_comparators.copy()
    selected_display["effect_ci"] = selected_display.apply(
        lambda row: f"{row['mean_paired_difference_baseline_minus_proposed']:.3f} "
        f"[{row['percentile_ci95_low']:.3f}, {row['percentile_ci95_high']:.3f}]",
        axis=1,
    )
    selected_display.sort_values(["selection_metric", "subset"]).to_csv(
        output / "metric_best_comparators.csv", index=False
    )
    write_booktabs_table(
        selected_display.sort_values(["selection_metric", "subset"]),
        output / "table_metric_best_comparators.tex",
        columns=[
            ("selection_metric", "Selection metric", "text"),
            ("baseline_display", "Metric-best baseline", "text"),
            ("subset", "Fixed task", "text"),
            ("effect_ci", "Baseline$-$Asym [95\\% CI]", "raw"),
            ("probability_proposed_better", "Pr(Asym better)", ".3f"),
        ],
        caption="Exploratory post-selection comparisons against metric-best controlled baselines.",
        label="tab:v3-metric-best",
        note="The comparator is selected from the same five-seed test macro table separately for RMSE, NASA/engine, and LPR@30, excluding the predeclared Asym point. The resulting intervals are descriptive post-selection analyses, not confirmatory inference. Positive differences favor Asym.",
    )
    write_booktabs_table(
        fixed_direct,
        output / "table_fixed_subset_direct_zh.tex",
        columns=[
            ("subset", "固定任务", "text"),
            ("metric_display", "指标", "text"),
            ("mean_paired_difference_baseline_minus_proposed", r"$\Delta$ RAST$-$Asym", ".3f"),
            ("percentile_ci95_low", "95\\% CI 下界", ".3f"),
            ("percentile_ci95_high", "95\\% CI 上界", ".3f"),
        ],
        caption="直接前身比较的固定子集发动机级主要配对证据。",
        label="tab:v3-fixed-subset-direct-zh",
        note="全部指标越低越好，因此正差值有利于 OCM-MST-GRU-Asym。每个区间只在一个固定 C-MAPSS 子集内重采样匹配发动机和复合种子，不把四个子集视为从任务总体随机抽取的样本。",
    )

    core_rast_display = core_rast_subset.copy()
    core_rast_display["point_display"] = core_rast_display["ocm_variant"].map(
        {"core": "Core", "asym": "Asym"}
    )
    core_rast_display["metric_display"] = core_rast_display["metric"].map(
        {"rmse": "RMSE", "nasa_per_engine": "NASA/engine", "lpr30": "LPR@30"}
    )
    core_rast_display["effect_ci"] = core_rast_display.apply(
        lambda row: f"{row['mean_difference_ocm_minus_rast']:.3f} "
        f"[{row['percentile_ci95_low']:.3f}, {row['percentile_ci95_high']:.3f}]",
        axis=1,
    )
    core_rast_counts = []
    for (variant, metric_name), group in core_rast_display.groupby(["ocm_variant", "metric"], sort=False):
        core_rast_counts.append(
            {
                "point_display": {"core": "Core", "asym": "Asym"}[variant],
                "metric_display": {"rmse": "RMSE", "nasa_per_engine": "NASA/engine", "lpr30": "LPR@30"}[metric_name],
                "ocm_lower": int((group["ci_classification"] == f"CI supports lower {variant}").sum()),
                "rast_lower": int((group["ci_classification"] == "CI supports lower RAST").sum()),
                "unresolved": int((group["ci_classification"] == "unresolved").sum()),
            }
        )
    core_rast_counts = pd.DataFrame(core_rast_counts)
    for language, suffix in (("en", ""), ("zh", "_zh")):
        write_booktabs_table(
            core_rast_counts,
            output / f"table_core_rast_fixed_summary{suffix}.tex",
            columns=[
                ("point_display", "OCM point" if language == "en" else "OCM 运行点", "text"),
                ("metric_display", "Metric" if language == "en" else "指标", "text"),
                ("ocm_lower", "Diagnostic: OCM lower" if language == "en" else "诊断：OCM 更低", "int"),
                ("rast_lower", "Diagnostic: RAST lower" if language == "en" else "诊断：RAST 更低", "int"),
                ("unresolved", "Unresolved" if language == "en" else "未解决", "int"),
            ],
            caption=(
                "Symmetric fixed-task interval directions for Core and Asym against RAST-GRU."
                if language == "en" else "Core 与 Asym 相对 RAST-GRU 的对称固定任务区间方向。"
            ),
            label=f"tab:v3-core-rast-fixed-summary{suffix}",
            note=(
                "Each row counts four fixed C-MAPSS tasks. Full task-level effects and intervals are reported in the Supplementary Information."
                if language == "en" else "每行统计四个固定 C-MAPSS 任务；完整任务级效应和区间见补充材料。"
            ),
        )
        write_booktabs_table(
            core_rast_display.sort_values(["ocm_variant", "subset", "metric"]),
            output / f"table_core_rast_fixed_full{suffix}.tex",
            columns=[
                ("point_display", "OCM point" if language == "en" else "OCM 运行点", "text"),
                ("subset", "Task" if language == "en" else "任务", "text"),
                ("metric_display", "Metric" if language == "en" else "指标", "text"),
                ("effect_ci", "OCM$-$RAST [95\\% CI]", "raw"),
                ("ci_classification", "Classification" if language == "en" else "区间分类", "text"),
            ],
            caption=(
                "Complete fixed-task paired evidence for both OCM operating points against RAST-GRU."
                if language == "en" else "两个 OCM 运行点相对 RAST-GRU 的完整固定任务配对证据。"
            ),
            label=f"tab:v3-core-rast-fixed-full{suffix}",
            note=(
                "Effects are OCM minus RAST; negative values favor OCM. Engines are resampled within matched composite seeds and one fixed task."
                if language == "en" else "效应定义为 OCM 减 RAST；负值有利于 OCM。发动机在匹配复合种子和单一固定任务内部重采样。"
            ),
            font_command=r"\scriptsize",
            fit_width=True,
        )

    endpoint_cluster_display = endpoint_cluster_audit.copy()
    write_booktabs_table(
        endpoint_cluster_display,
        output / "table_endpoint_cluster_audit.tex",
        columns=[
            ("subset", "Task", "text"),
            ("endpoint_strategy", "Validation design", "text"),
            ("validation_engine_seed_clusters", "Engine-seed clusters", "int"),
            ("validation_endpoints_per_engine_min", "Min", "int"),
            ("validation_endpoints_per_engine_median", "Median", ".1f"),
            ("validation_endpoints_per_engine_max", "Max", "int"),
            ("test_endpoints_per_engine", "Test/engine", "int"),
        ],
        caption="Cluster structure of the validation-endpoint confirmation.",
        label="tab:v3-endpoint-cluster-audit",
        note="Validation cluster ID is composite seed plus engine ID. Multi-endpoint validation rows are used only for checkpoint selection. Inferential test intervals resample matched test engines, each contributing one official endpoint; validation endpoints are never resampled as independent test observations.",
        fit_width=True,
    )
    write_booktabs_table(
        endpoint_cluster_display,
        output / "table_endpoint_cluster_audit_zh.tex",
        columns=[
            ("subset", "任务", "text"),
            ("endpoint_strategy", "验证设计", "text"),
            ("validation_engine_seed_clusters", "发动机--种子簇", "int"),
            ("validation_endpoints_per_engine_min", "最小", "int"),
            ("validation_endpoints_per_engine_median", "中位数", ".1f"),
            ("validation_endpoints_per_engine_max", "最大", "int"),
            ("test_endpoints_per_engine", "测试端点/发动机", "int"),
        ],
        caption="验证端点确认实验的聚类结构。",
        label="tab:v3-endpoint-cluster-audit-zh",
        note="验证簇 ID 为复合种子加发动机 ID。多端点验证样本只用于检查点选择；推断区间重采样匹配测试发动机，每台发动机仅贡献一个官方端点，不把验证端点作为独立测试观测重采样。",
        fit_width=True,
    )

    unit_direct = ncmapss_units[ncmapss_units["model"].isin(["rast_gru", "rast_gru_v2"])].copy()
    unit_direct["model_display"] = unit_direct["model"].map(model_display_name)
    write_booktabs_table(
        unit_direct.sort_values(["test_unit", "model"]),
        output / "table_ncmapss_units_direct.tex",
        columns=[
            ("test_unit", "Test unit", "int"),
            ("model_display", "Model", "text"),
            ("rmse_mean", "RMSE", ".3f"),
            ("rmse_std", "RMSE SD", ".3f"),
            ("lpr30_mean", "LPR@30", ".3f"),
            ("lpr30_std", "LPR SD", ".3f"),
        ],
        caption="Per-unit N-CMAPSS DS02 audit for OCM-MST-GRU and its direct predecessor.",
        label="tab:v3-ncmapss-unit-direct",
        note="Values are five-seed means and sample standard deviations. Units 11, 14, and 15 are the complete held-out test set of the declared fixed DS02 task; overlapping windows are prediction instances, not independent inferential units.",
    )
    write_booktabs_table(
        unit_direct.sort_values(["test_unit", "model"]),
        output / "table_ncmapss_units_direct_zh.tex",
        columns=[
            ("test_unit", "测试 unit", "int"),
            ("model_display", "模型", "text"),
            ("rmse_mean", "RMSE", ".3f"),
            ("rmse_std", "RMSE SD", ".3f"),
            ("lpr30_mean", "LPR@30", ".3f"),
            ("lpr30_std", "LPR SD", ".3f"),
        ],
        caption="OCM-MST-GRU 与直接前身的 N-CMAPSS DS02 逐 unit 审计。",
        label="tab:v3-ncmapss-unit-direct-zh",
        note="数值为五种子均值和样本标准差。Unit 11、14、15 构成声明的固定 DS02 任务全部测试集；重叠窗口是预测实例，不是独立推断单位。",
    )

    split_display = ncmapss_split_sensitivity.copy()
    split_display["model_display"] = split_display["model"].map(model_display_name)
    write_booktabs_table(
        split_display.sort_values(["validation_unit", "model"]),
        output / "table_ncmapss_split_sensitivity.tex",
        columns=[
            ("validation_unit", "Validation unit", "int"),
            ("model_display", "Model", "text"),
            ("unit_macro_rmse_mean", "Unit RMSE", ".3f"),
            ("unit_macro_lpr30_mean", "LPR@30", ".3f"),
            ("seed_count", "Seeds", "int"),
        ],
        caption="N-CMAPSS DS02 development-split sensitivity with a fixed held-out test set.",
        label="tab:v3-ncmapss-split-sensitivity",
        note="Validation units 2, 10, and 20 are predeclared alternatives drawn from the six development units; the other five development units train the model. Test units 11, 14, and 15 remain fixed. This checks training/validation split dependence but does not create additional independent test units.",
    )
    write_booktabs_table(
        split_display.sort_values(["validation_unit", "model"]),
        output / "table_ncmapss_split_sensitivity_zh.tex",
        columns=[
            ("validation_unit", "验证 unit", "int"),
            ("model_display", "模型", "text"),
            ("unit_macro_rmse_mean", "Unit RMSE", ".3f"),
            ("unit_macro_lpr30_mean", "LPR@30", ".3f"),
            ("seed_count", "种子", "int"),
        ],
        caption="固定测试集下的 N-CMAPSS DS02 开发划分敏感性。",
        label="tab:v3-ncmapss-split-sensitivity-zh",
        note="验证 unit 2、10、20 是从六个开发 unit 中预先声明的三种选择，其余五个开发 unit 用于训练；测试 unit 11、14、15 始终固定。该实验检验训练/验证划分依赖，但不会创造额外独立测试 unit。",
    )

    interval_table = intervals.sort_values("mean_interval_score_mean").copy()
    write_booktabs_table(
        interval_table,
        output / "table_interval_calibration.tex",
        columns=[
            ("model_display", "Model", "text"),
            ("empirical_coverage_mean", "Coverage", ".3f"),
            ("absolute_coverage_error_mean", "|Cov.-0.8|", ".3f"),
            ("mean_interval_width_mean", "Width", ".2f"),
            ("mean_interval_score_mean", "Interval score", ".2f"),
            ("critical_coverage_mean", "Critical coverage", ".3f"),
        ],
        caption="Diagnostic performance of nominal 80\\% predictive RUL intervals.",
        label="tab:v3-interval-calibration",
        note="Coverage and width are macro-averaged across subsets within each seed and then summarized across five seeds. Lower coverage error, width, and interval score are preferable, but width must be interpreted jointly with coverage.",
    )

    conformal_table = conformal.copy()
    conformal_table["model_display"] = conformal_table["model"].map(
        {"quantile_gru": "Quantile-GRU", "rast_gru_v2": "OCM-MST-GRU"}
    ).fillna(conformal_table["model"])
    write_booktabs_table(
        conformal_table,
        output / "table_conformal_calibration.tex",
        columns=[
            ("model_display", "Model", "text"),
            ("raw_coverage_mean", "Raw cov.", ".3f"),
            ("conformal_coverage_mean", "Adjusted cov.", ".3f"),
            ("raw_mean_width_mean", "Raw width", ".2f"),
            ("conformal_mean_width_mean", "Adjusted width", ".2f"),
            ("conformal_interval_score_mean", "Score", ".2f"),
        ],
        caption="Post-selection empirical residual-quantile adjustment of nominal 80\\% RUL intervals.",
        label="tab:v3-conformal-calibration",
        note="For every subset and training seed, the residual quantile is estimated from validation-last engines after those endpoints have already participated in checkpoint selection, then frozen before test evaluation. This is an auxiliary post-selection adjustment; no split-conformal finite-sample coverage guarantee is claimed.",
    )

    design_table = design.sort_values(["window_size", "rul_cap"]).copy()
    write_booktabs_table(
        design_table,
        output / "table_design_sensitivity.tex",
        columns=[
            ("design", "Design", "text"),
            ("window_size", "Window", "int"),
            ("rul_cap", "RUL cap", "int"),
            ("test_rmse_mean", "RMSE", ".3f"),
            ("test_critical_30_late_prediction_ratio_mean", "LPR@30", ".3f"),
            ("seed_count", "Seeds", "int"),
        ],
        caption="FD004 input-window and RUL-cap design sensitivity.",
        label="tab:v3-design-sensitivity",
        note="The reference row reuses the fixed formal checkpoints; only non-reference designs are retrained. Different RUL caps redefine the capped target and are interpreted as design sensitivity, not as a common-target leaderboard.",
    )

    direct = bootstrap[(bootstrap["baseline_model"] == "rast_gru")].copy()
    write_booktabs_table(
        direct,
        output / "table_hierarchical_direct_predecessor.tex",
        columns=[
            ("metric", "Metric", "text"),
            ("mean_paired_difference_baseline_minus_proposed", "Paired difference", ".4f"),
            ("bca_ci95_low", "BCa low", ".4f"),
            ("bca_ci95_high", "BCa high", ".4f"),
            ("probability_proposed_better", "Pr(OCM better)", ".3f"),
            ("holm_p", "Holm p", ".4f"),
        ],
        caption="Hierarchical paired evidence against the direct RAST-GRU predecessor.",
        label="tab:v3-hierarchical",
        note="Positive paired differences favor OCM-MST-GRU for the lower-is-better metrics. Resampling respects subset, training-seed, and engine levels; the table is not a pooled window-level test.",
    )
    close_prior_display = close_prior.copy()
    close_prior_display["model_display"] = close_prior_display["model"].map(
        {
            "regime_dual_attention_cnn_gru": "Regime DA-CNN-GRU (adapted)",
            "rast_gru_v2": "OCM-MST-GRU",
        }
    ).fillna(close_prior_display["model"])
    write_booktabs_table(
        close_prior_display,
        output / "table_close_prior_fd002.tex",
        columns=[
            ("model_display", "Model", "text"),
            ("test_rmse_mean", "RMSE", ".3f"),
            ("test_mae_mean", "MAE", ".3f"),
            ("test_nasa_score_mean", "NASA score", ".1f"),
            ("test_critical_30_late_prediction_ratio_mean", "LPR@30", ".3f"),
            ("parameters_mean", "Parameters", "int"),
        ],
        caption="Targeted five-seed FD002 comparison with the closest regime-aware dual-attention design family.",
        label="tab:v3-close-prior",
        note="The comparator is an architecture-adapted implementation under the fixed common preprocessing, input, augmentation, loss, budget, and checkpoint protocol. It is not a claim of bitwise reproduction or a comparison with the original paper's reported values.",
    )

    buildup_display = cross_backbone_summary.copy()
    buildup_display["model_display"] = buildup_display["model"].map(
        {"rast_gru": "RAST-GRU", "transformer_lite": "Transformer-lite"}
    )
    buildup_display["stage_display"] = buildup_display["stage"].map(
        {
            "global_scaling": "Global scaling",
            "condition_normalization": "+ condition norm.",
            "plus_mask_channels": "+ mask channels",
            "plus_degradation_augmentation": "+ degradation aug.",
            "plus_asymmetric_base_loss": "+ asymmetric base loss",
            "plus_risk_checkpoint": "+ risk checkpoint",
        }
    )
    write_booktabs_table(
        buildup_display.sort_values(["model", "stage_order"]),
        output / "table_cross_backbone_protocol_buildup.tex",
        columns=[
            ("model_display", "Backbone", "text"),
            ("stage_display", "Cumulative stage", "text"),
            ("rmse_mean", "RMSE", ".3f"),
            ("nasa_per_engine_mean", "NASA/engine", ".3f"),
            ("lpr30_mean", "LPR@30", ".3f"),
            ("late_cvar95_30_mean", "Late CVaR95", ".3f"),
        ],
        caption="Five-seed FD004 cumulative protocol build-up on two fixed backbones.",
        label="tab:v3-cross-backbone-buildup",
        note="Each row adds exactly the named protocol component to the row above while retaining backbone, engine splits, selected sensors, window, optimizer, patience, and epoch budget. All 60 runs use one separately declared fast-cuDNN policy; they are not pooled with the deterministic fixed benchmark. Means are descriptive; the figure shows seed points rather than confirmatory intervals.",
        fit_width=True,
    )

    stage_short = {
        "global_scaling": "Global",
        "condition_normalization": "Condition",
        "plus_mask_channels": "+ mask",
        "plus_degradation_augmentation": "+ aug.",
        "plus_asymmetric_base_loss": "+ asym. base",
        "plus_risk_checkpoint": "+ risk ckpt.",
    }
    transition_display = cross_backbone_transitions.copy()
    transition_display["model_display"] = transition_display["model"].map(
        {"rast_gru": "RAST-GRU", "transformer_lite": "Transformer-lite"}
    )
    transition_display["transition_display"] = transition_display.apply(
        lambda row: f"{stage_short[row['left_stage']]} to {stage_short[row['right_stage']]}", axis=1
    )
    transition_display["metric_display"] = transition_display["metric"].map(
        {"rmse": "RMSE", "nasa_per_engine": "NASA/engine", "lpr30": "LPR@30", "late_cvar95_30": "CVaR95"}
    )
    transition_display["effect_ci"] = transition_display.apply(
        lambda row: f"{row['mean_difference_right_minus_left']:.3f} "
        f"[{row['percentile_ci95_low']:.3f}, {row['percentile_ci95_high']:.3f}]",
        axis=1,
    )
    write_booktabs_table(
        transition_display.sort_values(["model", "left_stage", "metric"]),
        output / "table_cross_backbone_transitions.tex",
        columns=[
            ("model_display", "Backbone", "text"),
            ("transition_display", "Cumulative transition", "text"),
            ("metric_display", "Metric", "text"),
            ("effect_ci", "Right $-$ left [95\\% CI]", "raw"),
            ("probability_right_lower", "Pr(right lower)", ".3f"),
        ],
        caption="Conditional transition diagnostics for the cumulative protocol audit.",
        label="tab:v3-cross-backbone-transitions",
        note="Each interval resamples matched FD004 test engines within composite seed but conditions on one fast-cuDNN runtime per seed. The repeat audit shows that omitted runtime variation can exceed cross-seed variation in some cells. These intervals are auxiliary diagnostics, not confirmation tests; their sign is not labeled resolved.",
        font_command=r"\scriptsize",
        fit_width=True,
    )

    fast_display = fast_cudnn_repeat.copy()
    fast_display["model_display"] = fast_display["model"].map(
        {"rast_gru": "RAST-GRU", "transformer_lite": "Transformer-lite"}
    )
    fast_display["stage_display"] = fast_display["stage"].map(
        {
            "condition_normalization": "Condition normalization",
            "plus_risk_checkpoint": "Full cumulative protocol",
        }
    )
    write_booktabs_table(
        fast_display.sort_values(["model", "stage", "metric"]),
        output / "table_fast_cudnn_repeat_audit.tex",
        columns=[
            ("model_display", "Backbone", "text"),
            ("stage_display", "Stage", "text"),
            ("metric", "Metric", "text"),
            ("fixed_seed_repeat_sd_mean", "Repeat SD", ".4f"),
            ("cross_seed_sd_original_grid", "Cross-seed SD", ".4f"),
            ("repeat_to_cross_seed_sd_ratio", "Ratio", ".3f"),
        ],
        caption="Fixed-seed fast-cuDNN rerun variability relative to original cross-seed variability.",
        label="tab:v3-fast-cudnn-repeat",
        note="Two backbones, two representative cumulative stages, and two fixed composite seeds were each evaluated in three trajectories (the original plus two reruns). Repeat SD measures kernel-level and residual runtime variability at fixed seed; cross-seed SD comes from the original five-seed attribution grid. The ratio is descriptive and the fast-cuDNN grid is not bitwise reproducible.",
        fit_width=True,
    )

    cost_display = decision_cost_pairs.copy()
    cost_display["variant_display"] = cost_display["ocm_variant"].map({"core": "Core", "asym": "Asym"})
    cost_display["comparator_display"] = cost_display["comparator"].map(
        {
            "rast_gru": "RAST-GRU",
            "transformer_lite": "Transformer-lite",
            "official_dual_mixer": "Dual-Mixer",
        }
    )
    cost_display["effect_ci"] = cost_display.apply(
        lambda row: f"{row['mean_difference_ocm_minus_comparator']:.3f} "
        f"[{row['percentile_ci95_low']:.3f}, {row['percentile_ci95_high']:.3f}]",
        axis=1,
    )
    write_booktabs_table(
        cost_display.sort_values(["ocm_variant", "comparator", "late_to_early_cost_ratio"]),
        output / "table_core_asym_decision_cost.tex",
        columns=[
            ("variant_display", "OCM point", "text"),
            ("comparator_display", "Comparator", "text"),
            ("late_to_early_cost_ratio", "$\\kappa$", ".0f"),
            ("effect_ci", "OCM$-$comparator [95\\% CI]", "raw"),
            ("probability_ocm_lower_cost", "$P(\\mathrm{OCM}<\\mathrm{comp.})$", ".3f"),
        ],
        caption="Paired asymmetric-error-cost sensitivity for Core and Asym against major comparators.",
        label="tab:v3-core-asym-decision-cost",
        note="The researcher-set error cost is mean $A_\\kappa(e)=\\max(-e,0)+\\kappa\\max(e,0)$ over test-engine endpoints. It is not a maintenance-policy cost model. Intervals resample engines within the four fixed tasks and composite seeds. Negative differences favor the named OCM operating point.",
        fit_width=True,
    )

    endpoint_agreement_display = endpoint_epoch_agreement.copy()
    endpoint_agreement_display["variant_display"] = endpoint_agreement_display["variant"].map(
        {"rast": "RAST-GRU", "ocm_core": "OCM-Core", "ocm_asym": "OCM-Asym"}
    )
    write_booktabs_table(
        endpoint_agreement_display.sort_values(["variant", "subset"]),
        output / "table_endpoint_epoch_agreement.tex",
        columns=[
            ("variant_display", "Operating point", "text"),
            ("subset", "Task", "text"),
            ("uniform_epoch_mean", "Uniform", ".1f"),
            ("fixed_multi_epoch_mean", "Fixed multi", ".1f"),
            ("critical_stratified_epoch_mean", "Critical strat.", ".1f"),
            ("same_epoch_all_three_count", "Exact agreement", "int"),
            ("max_pairwise_epoch_gap_mean", "Mean max gap", ".1f"),
        ],
        caption="Best-checkpoint epoch sensitivity across validation endpoint designs.",
        label="tab:v3-endpoint-epoch-agreement",
        note="Exact agreement counts seeds for which all three endpoint designs select the same epoch. Test evaluation remains one endpoint per engine; no sliding-window row is treated as an independent inferential unit.",
        fit_width=True,
    )

    stress_operating_primary = stress_engine_bootstrap.copy()
    stress_operating_primary = stress_operating_primary[
        stress_operating_primary["metric"].isin(
            [
                "rmse",
                "critical_30_late_prediction_ratio",
                "critical_30_mean_late_excess",
                "critical_30_late_cvar95",
                "critical_30_decision_cost_5",
            ]
        )
    ].copy()
    stress_operating_counts = (
        stress_operating_primary.groupby(["operating_point", "metric", "ci_classification"], as_index=False)
        .size()
        .pivot(index=["operating_point", "metric"], columns="ci_classification", values="size")
        .fillna(0)
        .reset_index()
    )
    for column in ("CI supports lower Core", "CI supports lower Asym", "CI supports lower RAST", "unresolved"):
        if column not in stress_operating_counts:
            stress_operating_counts[column] = 0
    stress_operating_counts["point_lower"] = stress_operating_counts.apply(
        lambda row: row.get(f"CI supports lower {row['operating_point']}", 0), axis=1
    )
    stress_operating_counts["metric_display"] = stress_operating_counts["metric"].map(
        {
            "rmse": "RMSE",
            "critical_30_late_prediction_ratio": "LPR@30",
            "critical_30_mean_late_excess": "MLE@30",
            "critical_30_late_cvar95": "Late CVaR95",
            "critical_30_decision_cost_5": "Asymmetric error cost (kappa=5)",
        }
    )
    write_booktabs_table(
        stress_operating_counts,
        output / "table_stress_operating_points.tex",
        columns=[
            ("operating_point", "OCM point", "text"),
            ("metric_display", "Metric", "text"),
            ("point_lower", "Diagnostic: OCM lower", "int"),
            ("CI supports lower RAST", "Diagnostic: RAST lower", "int"),
            ("unresolved", "Unresolved", "int"),
        ],
        caption="Nine-family diagnostic-envelope directions for Core and Asym against RAST-GRU.",
        label="tab:v3-stress-operating-points",
        note="Counts use an uncalibrated crossed seed--perturbation-seed--engine resampling diagnostic over five coupled composite-seed levels, five matched perturbation seeds, and 248 matched FD004 test engines. They summarize standardized-coordinate perturbation curves, not confidence decisions, physical degradation, or field robustness.",
    )

    stress_cost = stress_engine_bootstrap[
        stress_engine_bootstrap["metric"].isin(
            ["critical_30_decision_cost_2", "critical_30_decision_cost_5", "critical_30_decision_cost_10"]
        )
    ].copy()
    stress_cost["rho"] = stress_cost["metric"].str.rsplit("_", n=1).str[-1].astype(int)
    stress_cost_rows = []
    for (point, rho), group in stress_cost.groupby(["operating_point", "rho"], sort=True):
        stress_cost_rows.append(
            {
                "operating_point": point,
                "rho": rho,
                "point_lower": int((group["ci_classification"] == f"CI supports lower {point}").sum()),
                "rast_lower": int((group["ci_classification"] == "CI supports lower RAST").sum()),
                "unresolved": int((group["ci_classification"] == "unresolved").sum()),
            }
        )
    stress_cost_counts = pd.DataFrame(stress_cost_rows)
    for language, suffix in (("en", ""), ("zh", "_zh")):
        write_booktabs_table(
            stress_cost_counts,
            output / f"table_stress_decision_cost_rho{suffix}.tex",
            columns=[
                ("operating_point", "OCM point" if language == "en" else "OCM 运行点", "text"),
                ("rho", "$\\kappa$", "int"),
                ("point_lower", "Diagnostic: OCM lower" if language == "en" else "诊断：OCM 更低", "int"),
                ("rast_lower", "Diagnostic: RAST lower" if language == "en" else "诊断：RAST 更低", "int"),
                ("unresolved", "Unresolved" if language == "en" else "未解决", "int"),
            ],
            caption=(
                "Stress-test asymmetric-error-cost classifications across late-to-early error ratios."
                if language == "en" else "不同晚预测/早预测成本比下的压力测试决策成本分类。"
            ),
            label=f"tab:v3-stress-cost-rho{suffix}",
            note=(
                "Each row counts nine perturbation families using the same crossed resampling diagnostic. Lower cost is preferable; the descriptive direction label can differ from LPR because error magnitude and early underestimation also enter the cost."
                if language == "en" else "每行统计九类扰动，并使用同一交叉重采样诊断。成本越低越好；由于代价还包含误差幅度和提前低估，描述性方向可与 LPR 不同。"
            ),
        )

    augmentation_display = augmentation_distribution.copy()
    augmentation_display["point_display"] = augmentation_display["point"].map(
        {"RAST": "RAST", "Core": "Core", "Asym": "Asym"}
    )
    augmentation_display["condition_display"] = augmentation_display["condition"].map(
        {"p0": "$p=0$", "p025": "$p=0.25$", "p05": "$p=0.5$", "p1": "$p=1$ (study reference)", "mix50": "50/50 mix"}
    )
    for language, suffix in (("en", ""), ("zh", "_zh")):
        write_booktabs_table(
            augmentation_display,
            output / f"table_augmentation_distribution_complete{suffix}.tex",
            columns=[
                ("point_display", "Point" if language == "en" else "运行点", "text"),
                ("condition_display", "Training distribution" if language == "en" else "训练分布", "raw"),
                ("clean_rmse_mean", "Clean RMSE", ".3f"),
                ("clean_lpr30_mean", "Clean LPR", ".3f"),
                ("clean_critical30_signed_bias_mean", "Clean signed bias", ".3f"),
                ("stress_lpr30_nine_family_mean_mean", "Stress LPR", ".3f"),
                ("stress_critical_30_signed_error_mean_nine_family_mean_mean", "Stress signed bias", ".3f"),
                ("stress_critical_30_decision_cost_5_nine_family_mean_mean", "Stress error cost $\\kappa=5$", ".3f"),
            ],
            caption=(
                "Complete FD004 augmentation-distribution sensitivity."
                if language == "en" else "完整 FD004 增强分布敏感性。"
            ),
            label=f"tab:v3-augmentation-complete{suffix}",
            note=(
                "Values are five-seed means. Stress columns average normalized curves over nine algorithmic perturbation families and five perturbation seeds. The study-reference p=1 row is retained for protocol continuity and is not a universally optimal distribution."
                if language == "en" else "数值为五种子均值。压力列对九类故障及五个扰动种子的归一化曲线求平均；锁定 p=1 仅为保持协议连续性，不作为普遍最优推荐。"
            ),
            font_command=r"\scriptsize",
            fit_width=True,
        )

    augmentation_interval_display = augmentation_intervals[
        (augmentation_intervals["condition"] != "p1")
        & augmentation_intervals["metric"].isin(
            [
                "clean_rmse",
                "clean_lpr30",
                "stress_lpr30_nine_family_mean",
                "stress_critical_30_decision_cost_5_nine_family_mean",
            ]
        )
    ].copy()
    augmentation_interval_display["point_display"] = augmentation_interval_display["point"]
    augmentation_interval_display["condition_display"] = augmentation_interval_display["condition"].map(
        {"p0": "$p=0$", "p025": "$p=0.25$", "p05": "$p=0.5$", "mix50": "50/50 mix"}
    )
    augmentation_interval_display["metric_display"] = augmentation_interval_display["metric"].map(
        {
            "clean_rmse": "Clean RMSE",
            "clean_lpr30": "Clean LPR",
            "stress_lpr30_nine_family_mean": "Stress LPR",
            "stress_critical_30_decision_cost_5_nine_family_mean": "Stress error cost $\\kappa=5$",
        }
    )
    augmentation_interval_display["effect_ci"] = augmentation_interval_display.apply(
        lambda row: f"{row['mean_difference_condition_minus_p1']:.3f} "
        f"[{row['paired_seed_bootstrap_ci95_low']:.3f}, {row['paired_seed_bootstrap_ci95_high']:.3f}]",
        axis=1,
    )
    augmentation_table_parts = []
    for point in ("RAST", "Core", "Asym"):
        temporary = output / f"_augmentation_cost_{point.lower()}.tex"
        write_booktabs_table(
            augmentation_interval_display[augmentation_interval_display["point"] == point],
            temporary,
            columns=[
                ("condition_display", "Alternative", "raw"),
                ("metric_display", "Metric", "raw"),
                ("effect_ci", "Alternative$-p=1$ [95\\% CI]", "raw"),
                ("probability_condition_lower", "Pr(alternative lower)", ".3f"),
            ],
            caption=f"Paired-seed augmentation-distribution sensitivity for the {point} operating point.",
            label=f"tab:v3-augmentation-cost-intervals-{point.lower()}",
            note="Intervals resample five matched training seeds. They are post-lock sensitivity evidence and are not used to select a replacement training distribution.",
            font_command=r"\scriptsize",
            fit_width=True,
        )
        augmentation_table_parts.append(temporary.read_text(encoding="utf-8"))
        temporary.unlink()
    (output / "table_augmentation_distribution_cost_intervals.tex").write_text(
        "\n".join(augmentation_table_parts), encoding="utf-8", newline="\n"
    )

    tcn_display = tcn_gru_audit.copy()
    tcn_display["catastrophic_display"] = tcn_display["catastrophic_cell"].map({True: "Yes", False: "No"})
    tcn_display["finite_display"] = tcn_display["prediction_has_nan_or_inf"].map({True: "No", False: "Yes"})
    tcn_robust_display = tcn_gru_robust.copy()
    write_booktabs_table(
        tcn_robust_display,
        output / "table_tcn_gru_robust_summary.tex",
        columns=[
            ("subset", "Task", "text"),
            ("rmse_mean", "Mean", ".3f"),
            ("rmse_sd", "SD", ".3f"),
            ("rmse_median", "Median", ".3f"),
            ("rmse_q25", "Q25", ".3f"),
            ("rmse_q75", "Q75", ".3f"),
            ("catastrophic_cell_count", "Flagged", "int"),
        ],
        caption="TCN-GRU seed-instability audit with robust summaries.",
        label="tab:v3-tcn-gru-robust",
        note="A cell is flagged only when RMSE exceeds 50 cycles and twice the within-task five-seed median. All 20 cells are retained; the rule is an audit flag, not an exclusion criterion.",
    )
    write_booktabs_table(
        tcn_display,
        output / "table_tcn_gru_seed_audit.tex",
        columns=[
            ("subset", "Task", "text"),
            ("seed", "Seed", "int"),
            ("test_rmse", "RMSE", ".3f"),
            ("test_nasa_per_engine", "NASA/engine", ".3f"),
            ("best_checkpoint_epoch", "Selected epoch", "int"),
            ("minimum_validation_loss_epoch", "Min-val epoch", "int"),
            ("error_max", "Max +error", ".2f"),
            ("error_min", "Max -error", ".2f"),
            ("finite_display", "Finite", "text"),
            ("catastrophic_display", "Flag", "text"),
        ],
        caption="Complete five-seed TCN-GRU failure audit.",
        label="tab:v3-tcn-gru-seeds",
        note="Gradient clipping at max norm 5.0 was active in every run, but gradient norms were not logged. The two flagged cells contain no NaN/Inf; their risk-score checkpoint selected an early conservative epoch despite later reductions in ordinary validation loss.",
        font_command=r"\scriptsize",
        fit_width=True,
    )

    fd002_point_labels = {
        "rast_core": "RAST-GRU Core",
        "transformer_core": "Transformer-lite Core",
        "ocm_core": "OCM-MST-GRU Core",
        "rast_asym": "RAST-GRU Asym",
        "ocm_asym": "OCM-MST-GRU Asym",
    }
    fd002_summary_display = fd002_transfer_summary.copy()
    fd002_summary_display["point_display"] = fd002_summary_display["point"].map(fd002_point_labels)
    write_booktabs_table(
        fd002_summary_display,
        output / "table_fd002_protocol_transfer.tex",
        columns=[
            ("point_display", "Operating point", "text"),
            ("rmse_mean", "RMSE", ".3f"),
            ("nasa_per_engine_mean", "NASA/engine", ".3f"),
            ("lpr30_mean", "LPR@30", ".3f"),
            ("cost_5_mean", "Error cost $\\kappa=5$", ".3f"),
        ],
        caption="Post-lock five-seed FD002 exploratory protocol-transfer audit.",
        label="tab:v3-fd002-protocol-transfer",
        note="Values are five-seed means on the same 259 official FD002 test-engine endpoints. Core and Asym denote neutral and illustrative asymmetric late-error preferences under the study-defined common protocol. This retrospective grid is not used to retune or replace the main fixed-task benchmark.",
        font_command=r"\scriptsize",
        fit_width=True,
    )
    write_booktabs_table(
        fd002_summary_display,
        output / "table_fd002_protocol_transfer_zh.tex",
        columns=[
            ("point_display", "运行点", "text"),
            ("rmse_mean", "RMSE", ".3f"),
            ("nasa_per_engine_mean", "NASA/engine", ".3f"),
            ("lpr30_mean", "LPR@30", ".3f"),
            ("cost_5_mean", "误差成本 $\\kappa=5$", ".3f"),
        ],
        caption="FD002 协议迁移的锁定后五种子探索性审计。",
        label="tab:v3-fd002-protocol-transfer-zh",
        note="数值是在相同 259 个 FD002 官方测试发动机端点上的五种子均值。Core 与 Asym 分别表示研究定义共同协议下的对称与非对称晚预测风险目标。该锁定后网格不用于重新调参或替换主基准。",
        font_command=r"\scriptsize",
        fit_width=True,
    )

    fd002_bootstrap_display = fd002_transfer_bootstrap.copy()
    fd002_bootstrap_display["metric_display"] = fd002_bootstrap_display["metric"].map(
        {"rmse": "RMSE", "nasa_per_engine": "NASA/engine", "lpr30": "LPR@30", "cost_5": "Error cost $\\kappa=5$"}
    )
    fd002_bootstrap_display["effect_ci"] = fd002_bootstrap_display.apply(
        lambda row: f"{row['mean_left_minus_right']:.3f} [{row['ci95_low']:.3f}, {row['ci95_high']:.3f}]",
        axis=1,
    )
    fd002_bootstrap_display["comparison_display"] = fd002_bootstrap_display.apply(
        lambda row: f"{fd002_point_labels[row['left']]} $-$ {fd002_point_labels[row['right']]}", axis=1
    )
    fd002_bootstrap_display["classification_zh"] = fd002_bootstrap_display.apply(
        lambda row: (
            "Holm 校正后未解决"
            if not row["holm_significant_0_05"]
            else ("Holm 支持左侧更低" if row["ci95_high"] < 0 else "Holm 支持右侧更低")
        ),
        axis=1,
    )
    fd002_bootstrap_display["classification_display"] = fd002_bootstrap_display.apply(
        lambda row: (
            "unresolved after Holm"
            if not row["holm_significant_0_05"]
            else ("Holm supports lower left" if row["ci95_high"] < 0 else "Holm supports lower right")
        ),
        axis=1,
    )
    write_booktabs_table(
        fd002_bootstrap_display,
        output / "table_fd002_protocol_transfer_intervals.tex",
        columns=[
            ("comparison_display", "Left $-$ right", "raw"),
            ("metric_display", "Metric", "raw"),
            ("effect_ci", "Mean difference [95\\% CI]", "raw"),
            ("nominal_p_centered_bootstrap", "Nominal $p$", ".3f"),
            ("holm_p_20", "Holm $p$", ".3f"),
            ("classification_display", "Direction", "text"),
        ],
        caption="Engine-clustered paired intervals for the exploratory FD002 protocol-transfer audit.",
        label="tab:v3-fd002-protocol-transfer-intervals",
        note="The bootstrap resamples five matched composite seeds and, within each sampled seed, 259 matched test engines. Negative effects favor the left point. Two-sided centered-bootstrap p-values are adjusted jointly over all 20 exploratory contrasts by Holm's method. Percentile intervals remain estimation summaries rather than separate confirmatory tests.",
        font_command=r"\scriptsize",
        fit_width=True,
    )
    write_booktabs_table(
        fd002_bootstrap_display,
        output / "table_fd002_protocol_transfer_intervals_zh.tex",
        columns=[
            ("comparison_display", "左侧 $-$ 右侧", "raw"),
            ("metric_display", "指标", "raw"),
            ("effect_ci", "均值差 [95\\% CI]", "raw"),
            ("nominal_p_centered_bootstrap", "名义 $p$", ".3f"),
            ("holm_p_20", "Holm $p$", ".3f"),
            ("classification_zh", "方向", "text"),
        ],
        caption="FD002 探索性协议迁移审计的发动机簇配对区间。",
        label="tab:v3-fd002-protocol-transfer-intervals-zh",
        note="Bootstrap 先重采样五个匹配复合种子，再在每个抽中种子内重采样 259 台匹配测试发动机。负值有利于左侧运行点。双侧中心化 bootstrap p 值在全部 20 个探索性对照上采用 Holm 法联合校正；百分位区间仅作效应估计摘要，不作为 20 次独立确认性检验。",
        font_command=r"\scriptsize",
        fit_width=True,
    )

    evidence_levels = pd.DataFrame(
        [
            ("Main benchmark [E-FIXED-12]", "FD004 preference development; FD001--003 fixed transfer", "Matched engine + five seed streams", "Fixed-task estimation"),
            ("Core preference [D-CORE]", "Retrospective on FD001--004", "Matched engine", "Configuration sensitivity"),
            ("Endpoint design [D-ENDPOINT]", "Retrospective on FD002/FD004", "Matched engine", "Design sensitivity"),
            ("FD002 protocol [S-FD002-20]", "Exploratory", "Matched engine", "Protocol sensitivity"),
            ("Normalization [S-NORM-9]", "Retrospective FD004 control", "Seed + matched engine", "Mechanism sensitivity"),
            ("Perturbations [S-PERTURB-54]", "Retrospective FD004 normalized-coordinate probes", "Train seed + perturbation seed + engine", "Algorithmic sensitivity"),
            ("N-CMAPSS [D-NCMAPSS]", "Single DS02 computational proxy", "Held-out unit", "Exploratory failure analysis"),
        ],
        columns=["analysis", "development_use", "primary_unit", "status"],
    )
    write_booktabs_table(
        evidence_levels,
        output / "table_evidence_levels.tex",
        columns=[
            ("analysis", "Analysis", "text"),
            ("development_use", "Development use", "text"),
            ("primary_unit", "Primary unit", "text"),
            ("status", "Evidence role", "text"),
        ],
        caption="Task roles, development use, and statistical units used in this study.",
        label="tab:v3-evidence-levels",
        note="The manuscript's main family was formalized after earlier result review and is therefore an estimation family, not an externally timestamped prospective family. FD004 informed the illustrative asymmetric preference setting; FD001--FD003 are fixed-transfer tasks. No retrospective analysis replaces this family.",
        font_command=r"\footnotesize",
        column_spec=r"@{}p{0.22\linewidth}p{0.30\linewidth}p{0.22\linewidth}p{0.20\linewidth}@{}",
    )
    evidence_levels_zh = pd.DataFrame(
        [
            ("主要基准 [E-FIXED-12]", "FD004 偏好开发；FD001--003 固定迁移", "匹配发动机 + 五条种子流", "固定任务估计"),
            ("Core 偏好 [D-CORE]", "FD001--004 回顾性分析", "匹配发动机", "配置敏感性"),
            ("端点设计 [D-ENDPOINT]", "FD002/FD004 回顾性分析", "匹配发动机", "设计敏感性"),
            ("FD002 协议 [S-FD002-20]", "探索性分析", "匹配发动机", "协议敏感性"),
            ("归一化 [S-NORM-9]", "FD004 回顾性控制", "种子 + 匹配发动机", "机制敏感性"),
            ("扰动 [S-PERTURB-54]", "FD004 归一化坐标回顾性探针", "训练种子 + 扰动种子 + 发动机", "算法敏感性"),
            ("N-CMAPSS [D-NCMAPSS]", "单一 DS02 计算代理", "留出测试单元", "探索性失败分析"),
        ],
        columns=["analysis", "development_use", "primary_unit", "status"],
    )
    write_booktabs_table(
        evidence_levels_zh,
        output / "table_evidence_levels_zh.tex",
        columns=[
            ("analysis", "分析", "text"),
            ("development_use", "开发使用情况", "text"),
            ("primary_unit", "主要统计单位", "text"),
            ("status", "证据角色", "text"),
        ],
        caption="本研究的任务角色、开发使用情况与主要统计单位。",
        label="tab:v3-evidence-levels-zh",
        note="主文证据族是在早期结果审阅后形成的估计族，并非具有外部时间戳的前瞻性分析族。FD004 影响了示例性非对称偏好设置，FD001--FD003 为固定迁移任务；回顾性分析不替代该主证据族。",
        font_command=r"\footnotesize",
        column_spec=r"@{}p{0.22\linewidth}p{0.30\linewidth}p{0.22\linewidth}p{0.20\linewidth}@{}",
    )

    dual_ci = official_dual_operating.copy()
    dual_ci["metric_display"] = dual_ci["metric"].map(
        {"rmse": "RMSE", "nasa_per_engine": "NASA/engine", "lpr30": "LPR@30"}
    )
    dual_ci["dual_display"] = dual_ci["dual_mean"].map(lambda value: f"{value:.3f}")
    dual_ci["variant_display"] = dual_ci["ocm_variant"].map(
        {"ocm_core": "OCM-Core", "ocm_asym": "OCM-Asym"}
    )
    dual_ci["ocm_display"] = dual_ci["ocm_mean"].map(lambda value: f"{value:.3f}")
    dual_ci["effect_ci"] = dual_ci.apply(
        lambda row: f"{row['mean_paired_difference_dual_minus_ocm']:.3f} "
        f"[{row['percentile_ci95_low']:.3f}, {row['percentile_ci95_high']:.3f}]",
        axis=1,
    )
    write_booktabs_table(
        dual_ci.sort_values(["ocm_variant", "subset", "metric"]),
        output / "table_official_dual_mixer.tex",
        columns=[
            ("variant_display", "OCM point", "text"),
            ("subset", "Task", "text"),
            ("metric_display", "Metric", "text"),
            ("dual_display", "Dual-Mixer", "raw"),
            ("ocm_display", "OCM", "raw"),
            ("effect_ci", "Dual$-$OCM [95\\% CI]", "raw"),
        ],
        caption="Official-code-derived Dual-Mixer reference against both OCM operating points.",
        label="tab:v3-official-dual-mixer",
        note="Means use five matched seeds. Diagnostic intervals jointly resample matched seeds and engine endpoints; positive Dual$-$OCM differences favor the named OCM point. Holm adjustment is performed over the 12 task--metric diagnostics within each operating point and is stored with the source table. The adapter follows upstream commit 727c020 and the authors' recommended non-FSGRI settings, but uses this study's fixed engine split; it cannot isolate an architecture effect and is not a bitwise native reproduction.",
        fit_width=True,
    )

    dual_ci["ci_direction"] = np.select(
        [dual_ci["percentile_ci95_high"] < 0.0, dual_ci["percentile_ci95_low"] > 0.0],
        ["Dual lower", "OCM lower"],
        default="Unresolved",
    )
    dual_summary = (
        dual_ci.groupby(["ocm_variant", "variant_display", "metric", "metric_display"], as_index=False)
        .agg(
            dual_lower=("ci_direction", lambda values: int((values == "Dual lower").sum())),
            ocm_lower=("ci_direction", lambda values: int((values == "OCM lower").sum())),
            unresolved=("ci_direction", lambda values: int((values == "Unresolved").sum())),
            holm_significant=("holm_adjusted_p", lambda values: int((values < 0.05).sum())),
        )
    )
    write_booktabs_table(
        dual_summary.sort_values(["ocm_variant", "metric"]),
        output / "table_official_dual_mixer_summary.tex",
        columns=[
            ("variant_display", "OCM point", "text"),
            ("metric_display", "Metric", "text"),
            ("dual_lower", "CI: Dual lower", "int"),
            ("ocm_lower", "CI: OCM lower", "int"),
            ("unresolved", "Unresolved", "int"),
            ("holm_significant", "Holm sig.", "int"),
        ],
        caption="Fixed-task interval-direction summary for the official-code-derived Dual-Mixer reference.",
        label="tab:v3-official-dual-mixer",
        note="Each row counts four fixed C-MAPSS tasks. Diagnostic interval direction is based on paired nominal bootstrap intervals for Dual$-$OCM. Adjusted values over the 12 task--metric family are multiplicity diagnostics, not significance decisions. Full task-level estimates and diagnostics are reported in the Supplementary Information and source CSV.",
    )

    endpoint_seed_macro = endpoint_confirmation.groupby(
        ["variant", "endpoint_strategy", "seed"], as_index=False
    )[[
        "test_rmse",
        "test_nasa_per_engine",
        "test_critical_30_late_prediction_ratio",
        "test_critical_30_severe_late_10_ratio",
        "validation_critical_30_count",
    ]].mean()
    endpoint_table = endpoint_seed_macro.groupby(["variant", "endpoint_strategy"], as_index=False).agg(
        seed_count=("seed", "nunique"),
        rmse_mean=("test_rmse", "mean"),
        nasa_mean=("test_nasa_per_engine", "mean"),
        lpr_mean=("test_critical_30_late_prediction_ratio", "mean"),
        slpr10_mean=("test_critical_30_severe_late_10_ratio", "mean"),
        critical_validation_count_mean=("validation_critical_30_count", "mean"),
    )
    endpoint_table["variant_display"] = endpoint_table["variant"].map(
        {"rast": "RAST-GRU", "ocm_core": "OCM-Core", "ocm_asym": "OCM-Asym"}
    )
    endpoint_table["variant"] = pd.Categorical(
        endpoint_table["variant"], categories=["rast", "ocm_core", "ocm_asym"], ordered=True
    )
    endpoint_table["strategy_display"] = endpoint_table["endpoint_strategy"].map(
        {
            "uniform_single": "Uniform single",
            "fixed_multi": "Fixed multi",
            "critical_stratified": "Critical-stratified",
        }
    )
    endpoint_table["endpoint_strategy"] = pd.Categorical(
        endpoint_table["endpoint_strategy"],
        categories=["uniform_single", "fixed_multi", "critical_stratified"],
        ordered=True,
    )
    write_booktabs_table(
        endpoint_table.sort_values(["variant", "endpoint_strategy"]),
        output / "table_endpoint_selection_confirmation.tex",
        columns=[
            ("variant_display", "Operating point", "text"),
            ("strategy_display", "Endpoint design", "text"),
            ("critical_validation_count_mean", r"Mean $N_{30}$", ".1f"),
            ("rmse_mean", "RMSE", ".3f"),
            ("nasa_mean", "NASA/engine", ".3f"),
            ("lpr_mean", "LPR@30", ".3f"),
            ("slpr10_mean", "SLPR@30,10", ".3f"),
        ],
        caption="Checkpoint-selection confirmation under three validation endpoint designs.",
        label="tab:v3-endpoint-confirmation",
        note="The same training trajectory is evaluated under all three validation endpoint designs, and each design independently chooses its checkpoint. Results are equal-subset C-MAPSS macro means over FD002 and FD004 and five composite seeds. This retrospective sensitivity does not replace the fixed 260-run benchmark.",
    )

    endpoint_distribution_display = endpoint_distribution.copy()
    endpoint_distribution_display["strategy_display"] = endpoint_distribution_display[
        "endpoint_strategy"
    ].map(
        {
            "uniform_single": "Uniform single",
            "fixed_multi": "Fixed multi",
            "critical_stratified": "Critical-stratified",
        }
    )
    endpoint_distribution_display["endpoint_strategy"] = pd.Categorical(
        endpoint_distribution_display["endpoint_strategy"],
        categories=["uniform_single", "fixed_multi", "critical_stratified"],
        ordered=True,
    )
    write_booktabs_table(
        endpoint_distribution_display.sort_values(["subset", "endpoint_strategy"]),
        output / "table_validation_endpoint_distribution.tex",
        columns=[
            ("subset", "Task", "text"),
            ("strategy_display", "Endpoint design", "text"),
            ("endpoint_count", "$N$", "int"),
            ("critical_30_fraction", r"$\Pr(y\leq30)$", ".3f"),
            ("mean_rul", "Mean RUL", ".1f"),
            ("wasserstein_distance_to_test", "$W_1$ to test", ".2f"),
        ],
        caption="Validation-endpoint designs compared with the natural test-endpoint distribution.",
        label="tab:v3-endpoint-distribution",
        note="Statistics pool the five development splits within each subset. Test endpoints contain one endpoint per test engine. Fixed-multi and critical-stratified deliberately upweight the critical zone; their larger Wasserstein distances are not interpreted as improved distribution matching.",
    )

    endpoint_ci = endpoint_confirmation_bootstrap.copy()
    endpoint_ci["direction"] = np.select(
        [endpoint_ci["percentile_ci95_high"] < 0.0, endpoint_ci["percentile_ci95_low"] > 0.0],
        ["left_lower", "right_lower"],
        default="unresolved",
    )
    endpoint_ci_summary = (
        endpoint_ci.groupby(
            ["endpoint_strategy", "left_variant", "right_variant", "direction"], as_index=False
        )
        .size()
        .pivot(
            index=["endpoint_strategy", "left_variant", "right_variant"],
            columns="direction",
            values="size",
        )
        .fillna(0)
        .reset_index()
    )
    for column in ("left_lower", "right_lower", "unresolved"):
        if column not in endpoint_ci_summary:
            endpoint_ci_summary[column] = 0
        endpoint_ci_summary[column] = endpoint_ci_summary[column].astype(int)
    variant_labels = {"rast": "RAST", "ocm_core": "Core", "ocm_asym": "Asym"}
    endpoint_ci_summary["pair_display"] = endpoint_ci_summary.apply(
        lambda row: f"{variant_labels[row['left_variant']]} vs {variant_labels[row['right_variant']]}", axis=1
    )
    endpoint_ci_summary["strategy_display"] = endpoint_ci_summary["endpoint_strategy"].map(
        {
            "uniform_single": "Uniform single",
            "fixed_multi": "Fixed multi",
            "critical_stratified": "Critical-stratified",
        }
    )
    endpoint_ci_summary["endpoint_strategy"] = pd.Categorical(
        endpoint_ci_summary["endpoint_strategy"],
        categories=["uniform_single", "fixed_multi", "critical_stratified"],
        ordered=True,
    )
    write_booktabs_table(
        endpoint_ci_summary.sort_values(["endpoint_strategy", "left_variant", "right_variant"]),
        output / "table_endpoint_confirmation_interval_directions.tex",
        columns=[
            ("strategy_display", "Endpoint design", "text"),
            ("pair_display", "Left vs right", "text"),
            ("left_lower", "Diagnostic envelope: left lower", "int"),
            ("right_lower", "Diagnostic envelope: right lower", "int"),
            ("unresolved", "Unresolved", "int"),
        ],
        caption="Paired interval-direction counts for checkpoint-selection confirmation.",
        label="tab:v3-endpoint-confirmation-ci",
        note="Each row classifies six contrasts: FD002 and FD004 crossed with RMSE, LPR@30, and SLPR@30,10. Intervals jointly resample matched composite seeds and engine endpoints. Lower is better for every metric; a zero-crossing interval is unresolved.",
        fit_width=True,
    )

    k_display = ncmapss_validation_k.copy()
    k_display["selected_display"] = k_display["selected_by_validation_only"].map({True: "yes", False: "no"})
    write_booktabs_table(
        k_display.sort_values("condition_clusters"),
        output / "table_ncmapss_validation_k_selection.tex",
        columns=[
            ("condition_clusters", "$K$", "int"),
            ("validation_risk_score_mean", "Validation risk", ".3f"),
            ("test_unit_macro_rmse_mean", "Test unit RMSE", ".3f"),
            ("test_unit_macro_lpr30_mean", "Test LPR@30", ".3f"),
            ("selected_display", "Val.-selected", "text"),
        ],
        caption="Validation-only selection audit for the N-CMAPSS condition count.",
        label="tab:v3-ncmapss-validation-k",
        note="The predeclared selection rule minimizes the five-seed mean validation risk score without test access. It selects $K=4$, whereas the post-lock test-best RMSE occurs at $K=8$; this mismatch limits DS02 to exploratory external evidence.",
    )

    reliability_display = reliability.copy()
    for metric in ("fault_auroc", "brier_score", "ece_10bin", "calibration_intercept", "calibration_slope"):
        reliability_display[f"{metric}_mean_sd"] = reliability_display.apply(
            lambda row, metric=metric: (
                f"{float(row[f'{metric}_mean']):.3f} $\\pm$ {float(row[f'{metric}_std']):.3f}"
            ),
            axis=1,
        )
    write_booktabs_table(
        reliability_display.sort_values("missing_rate"),
        output / "table_reliability_observed_fraction.tex",
        columns=[
            ("missing_rate", "Missing rate", ".2f"),
            ("training_seed_count", "Seeds", "int"),
            ("fault_auroc_mean_sd", "AUROC", "raw"),
            ("brier_score_mean_sd", "Brier", "raw"),
            ("ece_10bin_mean_sd", "ECE", "raw"),
            ("calibration_intercept_mean_sd", "Intercept", "raw"),
            ("calibration_slope_mean_sd", "Slope", "raw"),
        ],
        caption="Observed-fraction diagnostic for the soft reliability weights.",
        label="tab:v3-reliability-observed-fraction",
        note="Values are mean $\\pm$ sample SD across five training-seed means; each seed mean averages five matched perturbation seeds. ECE uses ten fixed bins, whose counts are stored in reliability\\_calibration\\_bins.csv. The target is synthetic mask missingness. These diagnostics do not establish calibrated physical fault probabilities.",
        fit_width=True,
    )

    training_augmentation = pd.DataFrame(
        [
            {"mechanism": "Gaussian noise", "declared_level": "std=0.01", "application_probability": "1.0", "per_window_redraw": "yes", "baseline_view": "corrupted primary", "ocm_view": "primary + noisy consistency"},
            {"mechanism": "Full-window channel missingness", "declared_level": "10% channels", "application_probability": "1.0", "per_window_redraw": "yes", "baseline_view": "values + updated mask", "ocm_view": "same + reliability target"},
            {"mechanism": "Contiguous block missingness", "declared_level": "8% time, 8% channels", "application_probability": "1.0", "per_window_redraw": "yes", "baseline_view": "values + updated mask", "ocm_view": "same + reliability target"},
            {"mechanism": "Signed linear drift", "declared_level": "end amplitude=0.03", "application_probability": "1.0", "per_window_redraw": "yes", "baseline_view": "corrupted primary", "ocm_view": "same primary"},
        ]
    )
    write_booktabs_table(
        training_augmentation,
        output / "table_training_augmentation_protocol.tex",
        columns=[
            ("mechanism", "Mechanism", "text"),
            ("declared_level", "Declared level", "text"),
            ("application_probability", "$p_{apply}$", "text"),
            ("per_window_redraw", "Redraw/window", "text"),
            ("baseline_view", "Baseline view", "text"),
            ("ocm_view", "OCM view", "text"),
        ],
        caption="Declared training-time normalized-coordinate augmentation mixture.",
        label="tab:v3-training-augmentation",
        note="All four mechanisms are applied sequentially to every training window in standardized feature space; their random draws are renewed by a deterministic epoch--window seed derived from the composite seed. All controlled baselines receive the same corrupted primary value/mask input. OCM alone uses the mask as a reliability-supervision target and adds a second Gaussian-noise view (std=0.01) for consistency. Targets are never perturbed.",
        fit_width=True,
    )

    stress_protocol = pd.DataFrame(
        [
            ("Gaussian noise", "std", "0.01, 0.03, 0.05, 0.10", "all selected sensors", "no"),
            ("Global missing", "channel fraction", "0.10, 0.20, 0.30, 0.40", "same channels for all windows", "yes"),
            ("Window missing", "channel fraction", "0.10, 0.20, 0.30, 0.40", "channels redrawn per window", "yes"),
            ("Block missing", "time/channel fraction", "0.10, 0.20, 0.30", "shared contiguous block", "yes"),
            ("Group missing", "channel fraction", "0.10, 0.20, 0.30", "shared 50% time block", "yes"),
            ("Linear drift", "end amplitude", "0.02, 0.05, 0.10", "random sign by channel", "no"),
            ("Bias shift", "amplitude", "0.10, 0.25, 0.50", "random sign by channel", "no"),
            ("Stuck-at", "channel fraction", "0.10, 0.20, 0.30", "onset in middle third", "no"),
            ("Burst noise", "std", "0.10, 0.25, 0.50", "25% channels/time block", "no"),
        ],
        columns=["family", "intensity", "declared_grid", "generator_scope", "mask_update"],
    )
    write_booktabs_table(
        stress_protocol,
        output / "table_stress_generator_protocol.tex",
        columns=[("family", "Perturbation family", "text"), ("intensity", "Intensity", "text"),
                 ("declared_grid", "Nonzero grid", "text"), ("generator_scope", "Scope", "text"),
                 ("mask_update", "Mask", "text")],
        caption="Declared normalized-coordinate perturbation generators.",
        label="tab:v3-stress-generators",
        note="Generators are applied after normalization and before fixed-checkpoint inference. Five matched perturbation seeds are crossed with five training seeds, with identical perturbed values and masks across configurations. Every curve includes the clean point at intensity zero. These probes are algorithmic sensitivities, not raw-unit or physical fault models.",
        fit_width=True,
    )
    preference_table = (
        PROJECT_ROOT
        / "paper_outputs"
        / "release_v1"
        / "tables"
        / "table_fd004_preference_development_search.tex"
    )
    if preference_table.exists():
        shutil.copy2(preference_table, output / preference_table.name)
    analysis_working = PROJECT_ROOT / "paper_outputs" / "analysis_working"
    for name in (
        "table_condition_normalization_controls.tex",
        "table_mask_observation_sensitivity.tex",
        "table_primary_fixed_task_estimates.tex",
        "table_primary_seed_sign_patterns.tex",
        "table_development_excluded_summary.tex",
        "table_small_cluster_calibration.tex",
        "table_max_t_family_calibration.tex",
        "table_late_event_support.tex",
        "table_multiplicity_registry.tex",
        "table_feature_selection_stability.tex",
        "table_preference_selection_stability.tex",
        "table_preference_selection_stability_zh.tex",
    ):
        source_table = analysis_working / name
        if source_table.exists():
            shutil.copy2(source_table, output / name)
    for name in (
        "table_run_quality_robust_summary.tex",
        "table_run_quality_flagged_cells.tex",
    ):
        source_table = source / name
        if source_table.exists():
            shutil.copy2(source_table, output / name)
    write_round2_condensed_results(output, source)
    write_main_analysis_family_registry(output, analysis_working)
    write_evidence_navigation(output)
    normalize_paper_display_names(output)
    write_chinese_table_copies(output)
    normalize_paper_display_names(output)
    write_semantic_table_aliases(output)
    if args.copy_to:
        copy_to = Path(args.copy_to)
        copy_to.mkdir(parents=True, exist_ok=True)
        for path in output.iterdir():
            if path.is_file():
                shutil.copy2(path, copy_to / path.name)
        normalize_paper_display_names(copy_to)
        write_semantic_table_aliases(copy_to)
    print(f"MANUSCRIPT_EVIDENCE_READY {output}")


if __name__ == "__main__":
    main()
