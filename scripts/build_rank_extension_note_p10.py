"""Create a standalone manuscript-ready addendum, preserving the frozen p9."""
from pathlib import Path
import shutil

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT/'paper_outputs/protocol_rank_extension_p10'
TARGET = ROOT.parent/'正文/新增实验_p10'


def main():
    TARGET.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(SOURCE/'task_rank_summary.csv')
    comparison = pd.read_csv(SOURCE/'rank_comparisons.csv')
    same_metric = comparison[(comparison.factor == 'aggregation') & (comparison.level == 'mean5')]
    rows = []
    for r in summary.itertuples():
        rows.append(f'{r.scope} & {r.top1_change_count}/15 & {r.top3_change_count}/15 & {r.mean_reversal_fraction:.3f} & [{r.min_tau:.3f}, {r.max_tau:.3f}] \\\\')
    aggregate_rows = []
    for r in same_metric.itertuples():
        metric = str(r.scope).replace('_', r'\_')
        aggregate_rows.append(f'{r.left} to {r.right} & {metric} & {r.strict_reversals}/{r.strict_comparable_pairs} & {r.kendall_tau_b:.3f} \\\\')
    section = r'''\section{Thirteen-model evaluation-ranking sensitivity}
\label{sec:p10-rank-extension}
This result-informed extension uses the complete stored benchmark: 13 controlled configurations,
four C-MAPSS tasks and five composite levels, giving 260 fitted runs. We recomputed metrics
from matched official-test-engine predictions and checked RMSE, MAE, NASA score and LPR against
the stored records. All finite runs, including the two extreme TCN-GRU runs, remain included.
No model was retrained for this analysis, and no significance test was performed.

The finite evaluation grid contains RMSE, MAE, NASA/engine, LPR@30, SLPR@30,10 and
zero-included mean late excess at RUL$\leq30$. Each endpoint is minimized. The eight scopes are
the four individual tasks, equal-task macro summaries over FD001--FD003 or FD001--FD004,
and the corresponding pooled-engine summaries. A macro summary averages calculated
task metrics; pooling instead concatenates task-specific engine errors and recalculates the metric.
The five-level mean is always a mean of level metrics, not pooled level observations.
There are 288 ranking cells: eight scopes, six endpoints, and five individual levels plus the five-level mean.

Ranks use average ties, with values rounded to twelve decimal places for ranking only to suppress
last-bit aggregation artifacts. Top-$k$ sets include all ties at the $k$th order statistic;
turnover is one minus Jaccard overlap. A strict reversal requires opposite orders under both
specifications; pairs tied in either are excluded from that denominator and counted separately.
Rank dispersion is the maximum minus minimum rank over the declared finite grid.
First-place counts include ties, with fractional credits also provided. These counts are not winning probabilities.

\begin{table}[htbp]\centering\small
\caption{Five-level-mean rankings across the 15 unordered pairs of six endpoints. Top-$k$
columns count set changes. Reversal fractions average the 15 comparisons, each with its own
strict-pair denominator. FD004 is development-informed.}
\begin{tabular}{lrrrr}\toprule
Task & Top-1 changes & Top-3 changes & Reversal fraction & Kendall $\tau_b$ range\\\midrule
''' + '\n'.join(rows) + r'''
\bottomrule\end{tabular}\end{table}

Changes extend beyond the focal OCM-Asym/RAST-GRU pair. However, changing an endpoint also
changes the estimand: a different winner under late overprediction than under RMSE is not
by itself evidence of unreliability at a fixed objective. Aggregation and task-composition
comparisons are therefore reported separately. These results do not demonstrate thirteen-model
sensitivity to preprocessing or checkpoint retraining, which were not completed for the full set.
They do not establish population ranking probabilities, calibrated intervals or model superiority.

\clearpage\begin{landscape}
\begin{figure}[p]\centering
\includegraphics[width=\linewidth,height=.82\textwidth,keepaspectratio]{rank_heatmap_FD001_FD002.pdf}
\caption{Ranks of five-level mean metrics for thirteen configurations. Panels hold the task fixed;
average ties are explicit. Lower rank means a lower metric.}
\end{figure}\clearpage\end{landscape}
\begin{landscape}
\begin{figure}[p]\centering
\includegraphics[width=\linewidth,height=.82\textwidth,keepaspectratio]{rank_heatmap_FD003_FD004.pdf}
\caption{Continuation of the thirteen-model rankings for FD003 and development-informed FD004.}
\end{figure}\clearpage\end{landscape}
\begin{landscape}
\begin{figure}[p]\centering
\includegraphics[width=\linewidth,height=.82\textwidth,keepaspectratio]{rank_range_turnover.pdf}
\caption{Left: observed macro4 rank range across six metrics and five individual levels; dots are
mean ranks. Right: fractions of 15 metric pairs with changed tie-inclusive Top-1/Top-3 sets.}
\end{figure}\clearpage\end{landscape}
\clearpage
\subsection{Same-endpoint aggregation comparisons}
\begin{longtable}{llrr}\toprule
Scope change & Endpoint & Strict reversals/pairs & Kendall $\tau_b$\\\midrule\endhead
''' + '\n'.join(aggregate_rows) + r'''
\bottomrule\end{longtable}
The three-task variants omit development-informed FD004. These are finite-grid descriptions,
not an outcome-selected recommendation to use one aggregation. Full level-specific rankings,
pair denominators, event support, run accounting, development-history caveats and 780 input
hashes are retained with the analysis.

\subsection{Status of new training studies}
A separate protocol specifies 400 C-MAPSS runs (five new engine splits crossed with ten
new training streams, four tasks and two configurations), 80 grouped battery GRU runs and
24 simple-baseline fits. No such real-data training has started; external protocol registration
is pending. These plans contribute no numerical evidence here. The battery design reuses sixteen
previously inspected eligible units and is a result-informed grouped extension, not independent
external confirmation.
'''
    (TARGET/'rank_extension_section.tex').write_text(section, encoding='utf-8')
    document = r'''\documentclass[11pt]{article}
\usepackage[a4paper,margin=22mm]{geometry}
\usepackage{graphicx,booktabs,longtable,amsmath,pdflscape}
\usepackage[hidelinks]{hyperref}
\title{Supplemental experiment: full-benchmark ranking sensitivity}
\author{Research working addendum to the p9 manuscript}
\date{13 September 2026}
\begin{document}\maketitle
\input{rank_extension_section.tex}
\end{document}
'''
    (TARGET/'rank_extension_report.tex').write_text(document, encoding='utf-8')
    for name in ('rank_heatmap.pdf', 'rank_heatmap_FD001_FD002.pdf', 'rank_heatmap_FD003_FD004.pdf',
                 'rank_range_turnover.pdf', '实验结果说明.md'):
        shutil.copy2(SOURCE/name, TARGET/name)
    (TARGET/'README.md').write_text('''# p10 新增实验入口

rank_extension_report.pdf：已完成的13模型排名实验报告。
rank_extension_section.tex：可整合进论文的英文章节。
这是p9基础上的新增工作稿。504项新训练尚待外部注册，不能将计划写成结果。
完整CSV、输入哈希、脚本位于 ../../rs_tcn_gru_rul/paper_outputs/protocol_rank_extension_p10。
新训练协议位于 ../../rs_tcn_gru_rul/studies/protocol_replication_p10。
本次新增分析设计和代码由Codex在用户授权下协助制定与实施；若采纳入主文，应由作者核查并相应更新AI使用披露，不能继续笼统声称AI未参与任何实验设计。
''', encoding='utf-8')
    print(f'RANK_ADDENDUM_WRITTEN {TARGET}')


if __name__ == '__main__':
    main()
