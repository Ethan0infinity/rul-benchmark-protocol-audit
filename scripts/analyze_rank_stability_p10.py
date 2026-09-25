"""Retrospective finite-grid rankings from all 260 stored prediction files."""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import kendalltau, spearmanr

ROOT = Path(__file__).resolve().parents[1]
MODELS = ('gru', 'lstm', 'tcn', 'tcn_gru', 'rast_gru', 'cnn_lstm', 'attention_gru',
          'bigru_attention', 'transformer_lite', 'dual_attention_tcn', 'sensor_graph_gru',
          'quantile_gru', 'rast_gru_v2')
SEEDS = (42, 123, 2024, 2025, 2026)
TASKS = ('FD001', 'FD002', 'FD003', 'FD004')
METRICS = ('rmse', 'mae', 'nasa_per_engine', 'lpr30', 'slpr30_10', 'zimle30')
LABELS = ('GRU', 'LSTM', 'TCN', 'TCN-GRU', 'RAST-GRU', 'CNN-LSTM', 'Attention-GRU',
          'BiGRU-Attn', 'Transformer-lite', 'Dual-Attn-TCN', 'Sensor-Graph-GRU',
          'Quantile-GRU', 'OCM-Asym')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def metrics(frame):
    error = frame.pred_rul.to_numpy(float) - frame.true_rul.to_numpy(float)
    zone = error[frame.true_rul.to_numpy(float) <= 30]
    nasa = np.expm1(np.where(error < 0, -error / 13, error / 10))
    return dict(rmse=float(np.sqrt(np.mean(error ** 2))), mae=float(np.mean(abs(error))),
                nasa_per_engine=float(nasa.mean()), lpr30=float(np.mean(zone > 0)) if len(zone) else np.nan,
                slpr30_10=float(np.mean(zone > 10)) if len(zone) else np.nan,
                zimle30=float(np.maximum(zone, 0).mean()) if len(zone) else np.nan,
                engines=len(error), eligible_engines=len(zone), late_events=int((zone > 0).sum()))


def compare_rankings(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    i, j = np.triu_indices(len(a), 1)
    left, right = np.sign(a[i] - a[j]), np.sign(b[i] - b[j])
    comparable = (left != 0) & (right != 0)
    reversal = int(np.sum(left * right < 0))
    out = dict(kendall_tau_b=float(kendalltau(a, b).statistic),
               spearman_rho=float(spearmanr(a, b).statistic),
               strict_reversals=reversal, strict_comparable_pairs=int(comparable.sum()),
               total_model_pairs=len(i), tied_in_either_pairs=int((~comparable).sum()),
               reversal_fraction=reversal / int(comparable.sum()) if comparable.any() else np.nan)
    for k in (1, 3):
        # Include all ties at the kth order statistic; no arbitrary tie breaking.
        sa = set(np.flatnonzero(a <= np.sort(a)[k - 1]))
        sb = set(np.flatnonzero(b <= np.sort(b)[k - 1]))
        out.update({f'top{k}_jaccard_turnover': 1 - len(sa & sb) / len(sa | sb),
                    f'top{k}_set_changed': int(sa != sb),
                    f'top{k}_left_size': len(sa), f'top{k}_right_size': len(sb)})
    return out


def collect(root):
    rows, predictions, provenance, accounting = [], {}, [], []
    for task, seed, model in itertools.product(TASKS, SEEDS, MODELS):
        run = root / 'results' / f'paper_main_v3_seed{seed}' / task / model
        required = ['test_predictions.csv', 'metrics.json', 'run_config.yaml']
        missing = [n for n in required if not (run / n).is_file()]
        if missing:
            raise ValueError(f'Missing planned run {run}: {missing}')
        df = pd.read_csv(run / required[0]).sort_values('unit_id').reset_index(drop=True)
        if not {'unit_id', 'end_cycle', 'true_rul', 'pred_rul'}.issubset(df):
            raise ValueError(f'Prediction schema: {run}')
        if df.unit_id.duplicated().any() or not np.isfinite(df.to_numpy(float)).all():
            raise ValueError(f'Nonfinite or repeated engine: {run}')
        reference = predictions.get((task, SEEDS[0], MODELS[0]))
        if reference is not None and not df[['unit_id', 'end_cycle', 'true_rul']].equals(reference[['unit_id', 'end_cycle', 'true_rul']]):
            raise ValueError(f'Matched engine/label mismatch: {run}')
        expected_n = dict(FD001=100, FD002=259, FD003=100, FD004=248)[task]
        if len(df) != expected_n:
            raise ValueError(f'Unexpected official engine count: {run}')
        meta = json.loads((run / 'metrics.json').read_text(encoding='utf-8-sig'))
        cfg = yaml.safe_load((run / 'run_config.yaml').read_text(encoding='utf-8-sig'))
        if meta.get('model') != model or meta.get('seed') != seed or meta.get('subset') != task:
            raise ValueError(f'Run identity mismatch: {run}')
        if meta.get('result_status') != 'formal' or not meta.get('allowed_for_paper'):
            raise ValueError(f'Not a formal run: {run}')
        values = metrics(df)
        for field in ('rmse', 'mae'):
            if not np.isclose(values[field], meta[f'test_{field}'], rtol=1e-7, atol=1e-7):
                raise ValueError(f'Stored metric differs from predictions: {run} {field}')
        for field, actual in [('test_nasa_score', values['nasa_per_engine']*len(df)),
                              ('test_critical_30_late_prediction_ratio', values['lpr30'])]:
            if not np.isclose(actual, meta[field], rtol=1e-7, atol=1e-7):
                raise ValueError(f'Stored metric differs from predictions: {run} {field}')
        if not np.isfinite([values[m] for m in METRICS]).all():
            raise ValueError(f'Undefined ranking endpoint: {run}')
        rows.append(dict(task=task, seed=seed, model=model, **values))
        predictions[task, seed, model] = df
        for name in required:
            path = run / name
            provenance.append(dict(path=path.relative_to(root).as_posix(), sha256=sha(path), bytes=path.stat().st_size))
        accounting.append(dict(task=task, seed=seed, model=model, status='included_finite',
            planned_epochs=meta.get('planned_epochs'), completed_epochs=meta.get('completed_epochs'),
            stop_reason=meta.get('training_complete_reason'), best_epoch=meta.get('best_epoch'),
            checkpoint_metric=meta.get('checkpoint_selection_metric'), parameters=meta.get('parameters'),
            training_seconds=meta.get('training_elapsed_sec'), peak_gpu_mb=meta.get('peak_gpu_memory_mb'),
            inference_ms=meta.get('single_sample_inference_ms'),
            recorded_loss=cfg['training'].get('loss'), recorded_late_weight=cfg['training'].get('late_over_weight'),
            extreme_rmse_over_50_descriptive_flag=values['rmse'] > 50))
    return pd.DataFrame(rows), predictions, pd.DataFrame(provenance), pd.DataFrame(accounting)


def protocol_grid(raw, predictions):
    scopes = {task: ((task,), 'single_task') for task in TASKS}
    scopes.update(macro4=(TASKS, 'mean_task_metric'), macro3=(TASKS[:3], 'mean_task_metric'),
                  pooled4=(TASKS, 'pooled_engine_metric'), pooled3=(TASKS[:3], 'pooled_engine_metric'))
    rows = []
    for scope, (tasks, aggregation) in scopes.items():
        for seed, model in itertools.product(SEEDS, MODELS):
            if aggregation == 'pooled_engine_metric':
                values = metrics(pd.concat([predictions[t, seed, model] for t in tasks]))
            else:
                cells = raw[(raw.task.isin(tasks)) & (raw.seed == seed) & (raw.model == model)]
                values = cells[list(METRICS)].mean().to_dict()
            for metric in METRICS:
                rows.append(dict(scope=scope, aggregation=aggregation, tasks='+'.join(tasks),
                                 level=str(seed), model=model, metric=metric, value=values[metric]))
    frame = pd.DataFrame(rows)
    mean = frame.groupby(['scope', 'aggregation', 'tasks', 'model', 'metric'], as_index=False).value.mean()
    mean['level'] = 'mean5'
    frame = pd.concat([frame, mean], ignore_index=True)
    # Suppress last-bit aggregation artifacts when rational event fractions tie.
    frame['rank_value'] = frame.value.round(12)
    frame['rank'] = frame.groupby(['scope', 'level', 'metric']).rank_value.rank(method='average')
    return frame


def comparisons(grid):
    rows = []
    vectors = {key: g.set_index('model').loc[list(MODELS), 'rank'].to_numpy()
               for key, g in grid.groupby(['scope', 'level', 'metric'])}
    for scope, level in itertools.product(grid.scope.unique(), grid.level.unique()):
        for left, right in itertools.combinations(METRICS, 2):
            rows.append(dict(factor='metric_choice', scope=scope, level=level, left=left, right=right,
                             **compare_rankings(vectors[scope, level, left], vectors[scope, level, right])))
    pairs = [('macro4', 'pooled4', 'aggregation'), ('macro3', 'pooled3', 'aggregation'),
             ('macro3', 'macro4', 'task_composition'), ('pooled3', 'pooled4', 'task_composition')]
    for left, right, factor in pairs:
        for level, metric in itertools.product(grid.level.unique(), METRICS):
            rows.append(dict(factor=factor, scope=metric, level=level, left=left, right=right,
                             **compare_rankings(vectors[left, level, metric], vectors[right, level, metric])))
    return pd.DataFrame(rows)


def plot(grid, ranges, comparisons_frame, output):
    os.environ.setdefault('MPLCONFIGDIR', str(output / '.mpl_cache'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    fig, axes = plt.subplots(2, 2, figsize=(14, 9), constrained_layout=True)
    for task, ax in zip(TASKS, axes.flat):
        frame = grid[(grid.scope == task) & (grid.level == 'mean5')]
        matrix = frame.pivot(index='metric', columns='model', values='rank').loc[list(METRICS), list(MODELS)]
        im = ax.imshow(matrix, vmin=1, vmax=13, cmap='viridis_r', aspect='auto')
        for i, j in itertools.product(range(6), range(13)):
            v = matrix.iloc[i, j]
            ax.text(j, i, f'{v:g}', ha='center', va='center', fontsize=8, color='white' if v > 7 else 'black')
        ax.set(title=task + (' (development-informed)' if task == 'FD004' else ''),
               xticks=np.arange(13), yticks=np.arange(6), yticklabels=METRICS)
        ax.set_xticklabels(LABELS, rotation=60, ha='right')
    fig.colorbar(im, ax=axes, label='Average-tie rank of five-level mean metric (1 = lowest)')
    for ext in ('pdf', 'png'):
        fig.savefig(output / f'rank_heatmap.{ext}', dpi=220)
    plt.close(fig)
    for pair in (TASKS[:2], TASKS[2:]):
        fig, axes = plt.subplots(2, 1, figsize=(13, 7.5), constrained_layout=True)
        for task, ax in zip(pair, axes):
            frame = grid[(grid.scope == task) & (grid.level == 'mean5')]
            matrix = frame.pivot(index='metric', columns='model', values='rank').loc[list(METRICS), list(MODELS)]
            im = ax.imshow(matrix, vmin=1, vmax=13, cmap='viridis_r', aspect='auto')
            for i, j in itertools.product(range(6), range(13)):
                value = matrix.iloc[i, j]
                ax.text(j, i, f'{value:g}', ha='center', va='center', fontsize=11,
                        color='white' if value > 7 else 'black')
            ax.set(title=task + (' (development-informed)' if task == 'FD004' else ''),
                   xticks=np.arange(13), yticks=np.arange(6), yticklabels=METRICS)
            ax.set_xticklabels(LABELS, rotation=35, ha='right', fontsize=11)
        fig.colorbar(im, ax=axes, label='Average-tie rank of five-level mean metric')
        fig.savefig(output/f'rank_heatmap_{pair[0]}_{pair[1]}.pdf')
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), constrained_layout=True)
    r = ranges[ranges.scope == 'macro4'].set_index('model').loc[list(MODELS)]
    for i, (_, row) in enumerate(r.iterrows()):
        axes[0].plot([row.min_rank, row.max_rank], [i, i], color='#26746b', linewidth=2.5)
        axes[0].scatter(row.mean_rank, i, color='#a43e43', s=25, zorder=3)
    axes[0].set(yticks=np.arange(13), yticklabels=LABELS, xticks=[1, 3, 5, 7, 9, 11, 13],
                xlabel='Rank across 6 metrics x 5 levels (macro4)', title='Observed rank ranges; dot = mean rank')
    axes[0].invert_yaxis()
    c = comparisons_frame[(comparisons_frame.factor == 'metric_choice') & (comparisons_frame.level == 'mean5')]
    for k, color, offset in [(1, '#26746b', -0.15), (3, '#a43e43', 0.15)]:
        vals = [c[c.scope == t][f'top{k}_set_changed'].mean() for t in TASKS]
        axes[1].bar(np.arange(4) + offset, vals, width=.3, label=f'Top-{k} set changes', color=color)
    axes[1].set(xticks=np.arange(4), xticklabels=TASKS, ylim=(0, 1),
                ylabel='Fraction of 15 declared metric pairs', title='Tie-inclusive top-k turnover')
    axes[1].legend()
    for ext in ('pdf', 'png'):
        fig.savefig(output / f'rank_range_turnover.{ext}', dpi=220)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', type=Path, default=ROOT/'paper_outputs/protocol_rank_extension_p10')
    args = parser.parse_args()
    out = args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    raw, predictions, provenance, accounting = collect(ROOT)
    grid = protocol_grid(raw, predictions)
    compare = comparisons(grid)
    indiv = grid[grid.level != 'mean5']
    ranges = indiv.groupby(['scope', 'model']).agg(min_rank=('rank', 'min'), max_rank=('rank', 'max'), mean_rank=('rank', 'mean')).reset_index()
    ranges['rank_dispersion'] = ranges.max_rank - ranges.min_rank
    wins = []
    for key, g in indiv.groupby(['scope', 'level', 'metric']):
        best = g['rank'].min()
        ties = int((g['rank'] == best).sum())
        for row in g.itertuples():
            wins.append(dict(scope=key[0], level=key[1], metric=key[2], model=row.model,
                             first_place_including_ties=int(row.rank == best),
                             fractional_first_credit=(1/ties if row.rank == best else 0)))
    wins = pd.DataFrame(wins).groupby(['scope', 'model'], as_index=False)[['first_place_including_ties', 'fractional_first_credit']].sum()
    outputs = {'run_metrics': raw, 'protocol_rankings': grid, 'rank_comparisons': compare,
               'rank_ranges': ranges, 'first_place_counts': wins,
               'source_manifest': provenance, 'run_accounting': accounting}
    ledger = []
    for model in MODELS:
        a = accounting[accounting.model == model]
        ledger.append(dict(model=model, implementation='local controlled implementation; not asserted official-code-derived',
            p9_role='focal developed configuration' if model in ('rast_gru', 'rast_gru_v2') else 'controlled reference',
            development_exposure='FD004 informed focal design; detailed per-baseline history not independently documented',
            hyperparameter_source='stored run_config.yaml; common controlled budget, not native tuning',
            test_outcome_use='p9 states no test-based checkpoint selection; retrospective analysis has outcome access',
            checkpoint_metric=';'.join(sorted(a.checkpoint_metric.unique())),
            max_epoch_budget='80', parameter_min=a.parameters.min(), parameter_max=a.parameters.max(),
            training_hours=a.training_seconds.sum()/3600, completed_runs=len(a)))
    outputs['development_history_matrix'] = pd.DataFrame(ledger)
    for name, frame in outputs.items():
        frame.to_csv(out/f'{name}.csv', index=False, encoding='utf-8-sig')
    plot(grid, ranges, compare, out)
    c = compare[(compare.factor == 'metric_choice') & (compare.level == 'mean5') & (compare.scope.isin(TASKS))]
    summary = c.groupby('scope').agg(mean_reversal_fraction=('reversal_fraction', 'mean'),
        min_tau=('kendall_tau_b', 'min'), max_tau=('kendall_tau_b', 'max'),
        top1_change_count=('top1_set_changed', 'sum'), top3_change_count=('top3_set_changed', 'sum')).reset_index()
    summary.to_csv(out/'task_rank_summary.csv', index=False)
    totals = {'status': 'RANK_ANALYSIS_PASS', 'planned_runs': 260, 'included_runs': len(raw),
        'protocol_cells': len(grid)//13, 'source_files_hashed': len(provenance),
        'comparison_rows': len(compare), 'source_training_hours': float(accounting.training_seconds.sum()/3600),
        'extreme_finite_runs_retained': int(accounting.extreme_rmse_over_50_descriptive_flag.sum()),
        'new_training_runs': 0, 'evidence_role': 'retrospective finite evaluation-grid description',
        'python': platform.python_version(), 'script_sha256': sha(Path(__file__)),
        'interpretation': 'Metric choice changes the estimand. No population reversal probability, significance test, or 13-model preprocessing retraining claim.'}
    (out/'analysis_summary.json').write_text(json.dumps(totals, indent=2), encoding='utf-8')
    lines = ['# 13 模型排名敏感性补充实验', '',
        '本分析重新读取 260 个正式 run 的测试预测，核对模型、任务、seed、匹配发动机和 RMSE/MAE。全部有限结果纳入，包括极端误差结果。',
        '', '## 已完成结果', '', '| 任务 | 15 个指标对中的 Top-1 集合变化次数 | Top-3 集合变化次数 | 平均严格排序反转比例 | Kendall tau-b 范围 |',
        '|---|---:|---:|---:|---|']
    for r in summary.itertuples():
        lines.append(f'| {r.scope} | {r.top1_change_count}/15 | {r.top3_change_count}/15 | {r.mean_reversal_fraction:.3f} | {r.min_tau:.3f} 至 {r.max_tau:.3f} |')
    lines += ['', '## 定义与边界', '',
        '- 6 个指标：RMSE、MAE、NASA/engine、LPR@30、SLPR@30,10、ZIMLE@30，均越低越好。LPR 按 true RUL<=30 且 error>0；SLPR 使用 error>10。',
        '- 8 个任务/聚合范围：四个单任务、macro3/macro4、pooled3/pooled4。macro 是先算每任务指标再平均；pooled 是合并不同任务的发动机后重新计算非线性指标。',
        '- 每范围有 5 个原 composite levels 和 1 个五级指标均值，合计 288 个排名单元。mean5 是逐级指标的均值，不是重新合并 seed 样本。',
        '- 指标值保留原精度；仅排序时取12位小数以消除浮点汇总末位噪声，并使用平均并列秩。Top-k 包含第 k 个位置处的全部并列模型。Top-k turnover=1-Jaccard；另报告集合是否改变。',
        '- reversal_fraction 分母仅为两协议下都不并列的模型对；同时报告全部78对、可比对和并列对数量。',
        '- 排名范围/第一名次数基于各范围的 6指标×5级=30格；第一名同时报告含并列次数与分摊信用，不能解释为概率。',
        '- 换指标意味着换研究目标；这些结果不能证明单一目标下模型不可靠。任务集合和聚合方法变化另表分列。',
        '- 没有声称这些模型都做过窗口、传感器或 checkpoint 重训；没有显著性检验或总体推断。',
        '- development_history_matrix.csv 中没有证据支撑的个体调参历史明确标为未独立记录。',
        '', '## 查看文件', '', '![排名热图](rank_heatmap.png)', '', '![排名范围与Top-k变化](rank_range_turnover.png)', '',
        '`run_metrics.csv`：260组重算指标；`protocol_rankings.csv`：全排名；`rank_comparisons.csv`：分因素相关与反转；',
        '`rank_ranges.csv` / `first_place_counts.csv`：范围和获胜次数；`run_accounting.csv`：停止原因、资源、极端值；',
        '`source_manifest.csv`：780份输入文件SHA-256；`analysis_summary.json`：本次计算状态。']
    (out/'实验结果说明.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(json.dumps(totals), flush=True)
    print(summary.to_string(index=False), flush=True)


if __name__ == '__main__':
    main()
