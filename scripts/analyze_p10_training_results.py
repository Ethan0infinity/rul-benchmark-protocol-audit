"""Finite-design summaries; incomplete families never become completed evidence."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd


def measure(frame):
    e = frame.pred_rul.to_numpy() - frame.true_rul.to_numpy()
    zone = e[frame.true_rul.to_numpy() <= 30]
    return {'rmse': float(np.sqrt(np.mean(e**2))), 'mae': float(np.mean(abs(e))),
            'nasa_per_engine': float(np.expm1(np.where(e < 0, -e/13, e/10)).mean()),
            'lpr30': float((zone > 0).mean()) if len(zone) else np.nan,
            'slpr30_10': float((zone > 10).mean()) if len(zone) else np.nan,
            'decision_error_cost2': float((np.maximum(-e, 0) + 2*np.maximum(e, 0)).mean()),
            'eligible_engines': len(zone), 'engines': len(e)}


def analyze_cmapss(jobs, completions, project, output):
    rows, identity = [], {}
    for job in jobs:
        saved = completions[job['job_id']]
        path = next(project/r['path'] for r in saved['artifacts'] if r['path'].endswith('/test_predictions.csv'))
        df = pd.read_csv(path).sort_values('unit_id').reset_index(drop=True)
        ids = df[['unit_id', 'end_cycle', 'true_rul']]
        if job['task'] in identity and not ids.equals(identity[job['task']]):
            raise ValueError('Unmatched official test engines')
        identity[job['task']] = ids
        rows.append({**{k: job[k] for k in ('task', 'split_seed', 'stream_seed', 'model')}, **measure(df)})
    raw = pd.DataFrame(rows)
    keys = ['task', 'split_seed', 'stream_seed']
    endpoints = ['rmse', 'mae', 'nasa_per_engine', 'lpr30', 'slpr30_10', 'decision_error_cost2']
    a = raw[raw.model == 'rast_gru_v2'].set_index(keys)[endpoints]
    b = raw[raw.model == 'rast_gru'].set_index(keys)[endpoints]
    effect = (a-b).reset_index()
    macro = effect.groupby(['split_seed', 'stream_seed'], as_index=False)[endpoints].mean()
    summaries, decomposition = [], []
    for task, group in [('macro4', macro), *list(effect.groupby('task'))]:
        for endpoint in endpoints:
            v = group[endpoint]
            summaries.append({'task': task, 'endpoint': endpoint, 'completed_cells': len(v), 'mean': v.mean(),
                'minimum': v.min(), 'maximum': v.max(), 'negative': int((v < 0).sum()),
                'zero': int((v == 0).sum()), 'positive': int((v > 0).sum())})
            if len(group.groupby(['split_seed', 'stream_seed'])) != 50:
                raise ValueError('Expected complete 5x10 design')
            sm = group.groupby('split_seed')[endpoint].transform('mean')
            tm = group.groupby('stream_seed')[endpoint].transform('mean')
            residual = v-sm-tm+v.mean()
            decomposition.append({'task': task, 'endpoint': endpoint, 'split_mean_range': sm.max()-sm.min(),
                'stream_mean_range': tm.max()-tm.min(), 'additive_residual_range': residual.max()-residual.min(),
                'interpretation': 'finite design; no replicated-cell interaction identification or population variance'})
    crossed = effect.merge(macro[['split_seed', 'stream_seed', 'rmse']], on=['split_seed', 'stream_seed'], suffixes=('', '_macro'))
    crossed['task_to_macro_strict_reversal'] = crossed.rmse * crossed.rmse_macro < 0
    bands = []
    for fraction in (.005, .01, .02):
        for task, group in effect.groupby('task'):
            bands.append({'task': task, 'cap_fraction': fraction, 'margin_cycles': fraction*125,
                'below': int((group.rmse < -fraction*125).sum()), 'within_inclusive': int((group.rmse.abs() <= fraction*125).sum()),
                'above': int((group.rmse > fraction*125).sum())})
    for name, frame in [('cmapss_run_metrics', raw), ('cmapss_paired_effects', effect), ('cmapss_macro_effects', macro),
                        ('cmapss_summary', pd.DataFrame(summaries)), ('finite_decomposition', pd.DataFrame(decomposition)),
                        ('task_macro_direction_changes', crossed), ('reporting_bands', pd.DataFrame(bands))]:
        frame.to_csv(output/f'{name}.csv', index=False)


def analyze_battery(jobs, completions, project, output):
    rows = []
    for job in jobs:
        saved = completions[job['job_id']]
        path = next(project/r['path'] for r in saved['artifacts'] if r['path'].endswith('/test_predictions.csv'))
        df = pd.read_csv(path)
        if df.sample_id.duplicated().any():
            raise ValueError('Duplicate battery sample')
        for unit, group in df.groupby('unit_id'):
            e = (group.pred_rul-group.true_rul).to_numpy()
            values = {'rmse': float(np.sqrt(np.mean(e**2))), 'mae': float(np.mean(abs(e)))}
            for q in (.05, .1, .2):
                zone = e[(group.true_rul/group.lifetime).to_numpy() <= q]
                lifetime = float(group.lifetime.iloc[0])
                values[f'eligible_{q}'] = len(zone)
                for epsilon in (0., .02, .05):
                    values[f'lpr_q{q}_eps{epsilon}'] = float((zone > epsilon*lifetime).mean()) if len(zone) else np.nan
            rows.append({'fold': job['fold'], 'stream_seed': job['stream_seed'], 'normalization': job['normalization'],
                         'model': job['model'], 'unit_id': unit, **values})
    raw = pd.DataFrame(rows)
    if raw.duplicated(['unit_id', 'stream_seed', 'normalization', 'model']).any():
        raise ValueError('A battery was held out more than once per model/stream/normalization')
    summaries = []
    endpoints = [c for c in raw if c.startswith('lpr_')] + ['rmse', 'mae']
    for (stream, normalization, model), g in raw.groupby(['stream_seed', 'normalization', 'model']):
        if g.unit_id.nunique() != 16:
            raise ValueError('Incomplete 16-battery OOF set')
        summaries.append({'stream_seed': stream, 'normalization': normalization, 'model': model,
                          'independent_units': 16, **g[endpoints].mean().to_dict()})
    summary = pd.DataFrame(summaries)
    a = summary[summary.model == 'asym'].set_index(['stream_seed', 'normalization'])[endpoints]
    b = summary[summary.model == 'core'].set_index(['stream_seed', 'normalization'])[endpoints]
    raw.to_csv(output/'battery_unit_metrics.csv', index=False)
    summary.to_csv(output/'battery_equal_unit_summary.csv', index=False)
    (a-b).reset_index().to_csv(output/'battery_paired_stream_contrasts.csv', index=False)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, required=True)
    args = parser.parse_args()
    project = args.project_root.resolve()
    frozen = Path(__file__).resolve().parent
    study = frozen.parent
    jobs = json.loads((frozen/'jobs.json').read_text(encoding='utf-8'))
    output = study/'analysis'
    output.mkdir(exist_ok=True)
    accounting, completions = [], {}
    for job in jobs:
        attempts = sorted((study/'execution'/job['job_id']).glob('attempt_*'))
        status = 'planned_not_started'
        for attempt in attempts:
            path = attempt/'completion.json'
            if path.exists():
                data = json.loads(path.read_text(encoding='utf-8'))
                for r in data['artifacts']:
                    if hashlib.sha256((project/r['path']).read_bytes()).hexdigest() != r['sha256']:
                        raise ValueError('Completed artifact hash mismatch')
                completions[job['job_id']] = data
                status = 'valid_included'
            elif status != 'valid_included':
                status = 'started_failed_or_incomplete'
        accounting.append({'job_id': job['job_id'], 'family': job['family'], 'status': status, 'attempts': len(attempts)})
    pd.DataFrame(accounting).to_csv(output/'planned_started_valid_included.csv', index=False)
    families = []
    for family in ('cmapss_crossed', 'battery_grouped', 'battery_baseline'):
        subset = [j for j in jobs if j['family'] == family]
        n = sum(j['job_id'] in completions for j in subset)
        families.append({'family': family, 'planned': len(subset), 'valid': n, 'complete': n == len(subset)})
        if n == len(subset):
            if family == 'cmapss_crossed':
                analyze_cmapss(subset, completions, project, output)
    battery_jobs = [j for j in jobs if j['family'].startswith('battery')]
    if all(j['job_id'] in completions for j in battery_jobs):
        analyze_battery(battery_jobs, completions, project, output)
    result = {'status': 'ALL_NEW_TRAINING_COMPLETE' if len(completions) == len(jobs) else 'NEW_TRAINING_PENDING',
              'families': families, 'note': 'No absent family is reported as completed evidence.'}
    (output/'completion_status.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
