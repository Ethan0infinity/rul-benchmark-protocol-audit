"""Prepare, but never execute, the new externally timestamped training studies."""
from __future__ import annotations

import hashlib
import argparse
import itertools
import json
import platform
import shutil
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT/'studies/protocol_replication_p10'
TASKS = ('FD001', 'FD002', 'FD003', 'FD004')
SPLITS = tuple(range(9300001, 9300006))
STREAMS = tuple(range(9400001, 9400011))


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(4*1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def write_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--refresh-unregistered', action='store_true')
    args = parser.parse_args()
    if args.refresh_unregistered and any((STUDY/p).exists() for p in ('execution_receipt.json', 'execution', 'results')):
        raise SystemExit('Cannot refresh a registered or started protocol')
    if (STUDY/'frozen').exists() and not args.refresh_unregistered:
        raise SystemExit('Existing frozen protocol retained. Do not silently regenerate it.')
    frozen = STUDY/'frozen'
    frozen.mkdir(parents=True, exist_ok=args.refresh_unregistered)
    sys.path.insert(0, str(ROOT/'src'))
    from rul.preprocessing import split_units
    prior_seeds, scanned = set(), []
    for path in (ROOT/'results').rglob('run_config.yaml'):
        cfg = yaml.safe_load(path.read_text(encoding='utf-8-sig'))
        project = cfg.get('project', {})
        seeds = [project.get(k) for k in ('seed', 'split_seed', 'initialization_seed', 'shuffle_seed')]
        seeds.append(cfg.get('augmentation', {}).get('seed'))
        prior_seeds.update(int(s) for s in seeds if s is not None)
        scanned.append({'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path)})
    if prior_seeds.intersection(SPLITS + STREAMS):
        raise ValueError('Proposed seed overlaps recorded prior seeds')
    pd.DataFrame(scanned).to_csv(frozen/'historical_seed_audit_sources.csv', index=False)
    write_json(frozen/'historical_seed_audit.json', {'configs_scanned': len(scanned), 'recorded_seeds': sorted(prior_seeds),
        'scope': 'Local results/run_config.yaml files only; cannot certify unrecorded experiments.',
        'new_split_seeds': SPLITS, 'new_stream_seeds': STREAMS, 'overlap': []})
    source = frozen/'source'
    for path in (ROOT/'src/rul').glob('*.py'):
        target = source/'src/rul'/path.name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, target)
    for name in ('paper_main.yaml', 'protocol_v3.lock.yaml'):
        target = source/'configs'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(ROOT/'configs'/name, target)
    lock_path = source/'configs/protocol_v3.lock.yaml'
    lock = yaml.safe_load(lock_path.read_text(encoding='utf-8-sig'))
    lock.update(protocol_name='P10-RUL-PROTOCOL-REPLICATION-V1', protocol_version='1.0',
                allowed_experiment_prefix='cmapss_')
    lock_path.write_text(yaml.safe_dump(lock, sort_keys=False), encoding='utf-8')
    battery_root = ROOT/'studies/prospective_battery_freeze_v1'
    for name in ('battery_protocol.py', 'prepare_frozen_inputs.py', 'train_registered_models.py'):
        target = source/'battery'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(battery_root/'scripts'/name, target)
    shutil.copy2(ROOT/'scripts/run_p10_training_protocol.py', frozen/'run_experiments.py')
    shutil.copy2(ROOT/'scripts/analyze_p10_training_results.py', frozen/'analyze_results.py')
    shutil.copy2(Path(__file__), frozen/'prepare_protocol_source.py')
    shutil.copy2(STUDY/'OSF_REGISTRATION_TEXT.md', frozen/'OSF_REGISTRATION_TEXT.md')
    data_manifest, splits, jobs = [], [], []
    for path in sorted((ROOT/'data/raw').glob('*.txt')):
        if any(t in path.name for t in TASKS):
            data_manifest.append({'path': path.relative_to(ROOT).as_posix(), 'sha256': sha(path), 'bytes': path.stat().st_size})
    base = yaml.safe_load((ROOT/'configs/paper_main.yaml').read_text(encoding='utf-8-sig'))
    for task in TASKS:
        units = pd.read_csv(ROOT/f'data/raw/train_{task}.txt', sep=r'\s+', header=None).iloc[:, 0].unique()
        old_splits = {tuple(sorted(split_units(units, .2, seed)[1])) for seed in prior_seeds}
        for split in SPLITS:
            train, val = split_units(units, .2, split)
            if tuple(sorted(val)) in old_splits:
                raise ValueError('Exact validation membership duplicates recorded-seed split')
            for assignment, members in [('training', train), ('development', val)]:
                splits.extend({'task': task, 'split_seed': split, 'unit_id': int(u), 'assignment': assignment} for u in members)
            for stream, model in itertools.product(STREAMS, ('rast_gru', 'rast_gru_v2')):
                cfg = json.loads(json.dumps(base))
                job_id = f'cmapss_{task}_split{split}_stream{stream}_{model}'
                cfg['project'].update(seed=stream, split_seed=split, initialization_seed=stream, shuffle_seed=stream,
                    experiment_name=job_id, results_dir='studies/protocol_replication_p10/results/cmapss')
                cfg['model']['name'] = model
                cfg['data']['subset'] = task
                cfg['augmentation']['seed'] = stream
                cfg['training'].update(selection_metric='val_last_rmse', deterministic_cudnn=True)
                cfg['progress'].update(console=True, batch_bar=False)
                path = frozen/'configs'/f'{job_id}.json'
                write_json(path, cfg)
                jobs.append({'job_id': job_id, 'family': 'cmapss_crossed', 'task': task, 'split_seed': split,
                             'stream_seed': stream, 'model': model, 'config': path.relative_to(frozen).as_posix()})
    pd.DataFrame(splits).to_csv(frozen/'cmapss_engine_splits.csv', index=False)
    inventory = pd.read_csv(battery_root/'BATTERY_UNIT_INVENTORY.csv', dtype=str).fillna('')
    eligible = inventory[inventory.analysis_eligible.str.lower() == 'true']
    if len(eligible) != 16:
        raise ValueError('Battery inventory differs from reviewed 16-unit design')
    eligible.to_csv(frozen/'battery_eligible_units.csv', index=False)
    assignments = []
    # One 24 C and three 4 C test batteries per fold. Development takes one
    # unit per temperature, retaining >=2 training batteries per temperature.
    grouped = {}
    for temperature, group in eligible.groupby('temperature'):
        ordered = sorted(group.unit_id, key=lambda u: hashlib.sha256(f'P10-BATTERY-V1:{u}'.encode()).hexdigest())
        if len(ordered) % 4:
            raise ValueError('Four-fold temperature stratification is not balanced')
        grouped[temperature] = [ordered[f::4] for f in range(4)]
    for fold in range(4):
        for temperature, folds in grouped.items():
            testing = folds[fold]
            remain = [u for i, items in enumerate(folds) if i != fold for u in items]
            development = [min(remain, key=lambda u: hashlib.sha256(f'P10-DEV:{fold}:{u}'.encode()).hexdigest())]
            training = [u for u in remain if u not in development]
            if len(training) < 2:
                raise ValueError('Insufficient training units for temperature scaler')
            for role, units in [('training', training), ('development', development), ('test', testing)]:
                assignments.extend({'fold': fold, 'unit_id': u, 'temperature': temperature, 'assignment': role} for u in units)
        for stream, normalization, model in itertools.product(STREAMS[:5], ('global', 'temperature'), ('core', 'asym')):
            jobs.append({'job_id': f'battery_f{fold}_{normalization}_{model}_s{stream}', 'family': 'battery_grouped',
                         'fold': fold, 'normalization': normalization, 'stream_seed': stream, 'model': model})
        for normalization, model in itertools.product(('global', 'temperature'), ('life_prior', 'ridge', 'hist_gradient_boosting')):
            jobs.append({'job_id': f'battery_f{fold}_{normalization}_{model}', 'family': 'battery_baseline',
                         'fold': fold, 'normalization': normalization, 'stream_seed': STREAMS[0], 'model': model})
    pd.DataFrame(assignments).to_csv(frozen/'battery_grouped_splits.csv', index=False)
    for row in eligible.itertuples():
        path = battery_root/row.source_file
        actual = sha(path)
        if actual != row.canonical_sha256:
            raise ValueError(f'Battery source hash mismatch: {row.unit_id}')
        data_manifest.append({'path': path.relative_to(ROOT).as_posix(), 'sha256': actual, 'bytes': path.stat().st_size})
    pd.DataFrame(data_manifest).to_csv(frozen/'data_manifest.csv', index=False)
    write_json(frozen/'jobs.json', jobs)
    protocol = {
        'protocol_id': 'P10-RUL-PROTOCOL-REPLICATION-V1', 'prepared_at_utc': datetime.now(timezone.utc).isoformat(),
        'status': 'LOCAL_PREPARATION_ONLY_EXTERNAL_TIMESTAMP_PENDING',
        'questions': ['How do fresh crossed split/stream levels change finite task and macro rankings?',
                      'How does equal-battery OOF evaluation over all 16 previously eligible units change Core/Asym and simple-baseline rankings?'],
        'planned_runs': {'cmapss_crossed': 400, 'battery_grouped': 80, 'battery_baseline': 24},
        'primary_cmapss_estimands': ['OCM-Asym minus RAST task RMSE means over 50 cells', 'equal-four-task mean RMSE contrast',
            'negative/zero/positive counts per task', 'task-to-macro direction changes across 50 matched split-stream cells'],
        'secondary_cmapss': ['MAE', 'NASA/engine', 'LPR@30', 'SLPR@30,10', 'mean absolute early error + 2*positive error'],
        'cmapss_checkpoint': 'validation simulated-last RMSE only; RMSE and late metrics remain development-informed, not causally independent outcomes',
        'reporting_bands': {'rmse_fraction_of_125_cap': [.005, .01, .02], 'status': 'predeclared reporting bands; not validated SESOI or equivalence tests'},
        'randomness': 'deterministic ordered unused seed IDs; five split seeds crossed with ten optimization streams; init/shuffle/augmentation share stream ID',
        'decomposition': 'finite split-mean and stream-mean ranges and additive-residual range; no population variance components',
        'battery_primary': 'equal-16-battery OOF RMSE and LPR at true_RUL/lifetime<=0.10; aggregate units before streams; each unit held out once per stream',
        'battery_sensitivity': 'global versus train-only temperature normalization; LPR epsilon fractions 0,.02,.05; q=.05,.10,.20',
        'battery_training': 'unchanged frozen-v1 BatteryGRU/Huber training implementation; 100 epochs, patience15, equal-battery dev RMSE, two dev batteries per fold',
        'battery_baselines': 'training mean lifetime minus current index, weighted ridge alpha1, weighted HistGradientBoosting max_iter100/max_leaf_nodes15/l2=1/learning_rate=.1/early_stopping=False; no hyperparameter search',
        'failure_policy': 'Record every planned/started/valid/failed run. Retain all finite high errors. No quality-based exclusions. Errors stop execution; exact-config infrastructure retry is logged, previous partial output retained. No partial-completeness success.',
        'stopping_policy': 'Stop after all504 design cells; no outcome-adaptive stopping or extra seeds. Resume verifies completion hashes.',
        'analysis_policy': 'Descriptive finite-design effects and ties; no p-values, multiple-testing decision, calibrated CI, population probability or directionally preregistered superiority',
        'prior_knowledge': 'All old C-MAPSS test outcomes and battery outcomes have already been inspected. New streams are unrun, but datasets are not new independent samples. This is a prospectively specified rerun only after valid external timestamp; battery is a result-informed grouped extension, not an independent confirmatory study.',
        'excluded_scopes': ['full non-thinned N-CMAPSS', 'human comparator recruitment', 'new UQ framework', 'maintenance optimization', 'new architecture claim'],
        'external_gate': 'Public frozen OSF registration must contain the exact protocol ZIP. Execution fetches registration metadata and the registered file to check hash before training.',
        'source_adaptation': 'Copied p9 RUL code is unchanged. The snapshot protocol lock labels cmapss_ runs with the new protocol identity. Reused battery trainer original protocol ID is retained as implementation_protocol_id and new checkpoints receive P10 identity.',
        'allowed_wording': 'finite-grid replication / result-informed grouped battery sensitivity; no external confirmation or deployment safety',
    }
    write_json(frozen/'analysis_registry.json', protocol)
    write_json(frozen/'environment.json', {'python': sys.version, 'platform': platform.platform(),
        'packages': subprocess.check_output([sys.executable, '-m', 'pip', 'list', '--format=json'], text=True),
        'gpu': subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'], text=True).strip()})
    readme = '''# 新增训练协议：待外部注册

该包冻结400组C-MAPSS（5 split×10 stream×4 task×2配置）、80组电池GRU和24组简单基线。
新种子和具体发动机/电池分配已写入清单。原始训练结果不覆盖。

## 必须如实披露

原数据及旧模型结果已经看过。本包只能冻结尚未执行的复训操作，不能把原数据变成未见外部样本。
电池覆盖全部16个合格单元，每个电池在每个stream下仅作为测试单元一次；这是同研究者的结果知情分组扩展。
不运行未抽稀N-CMAPSS，不执行人类评审招募、维护优化或新UQ研究。

## 外部时间戳后执行

将整个PROTOCOL_FOR_OSF.zip上传到新的OSF项目，再创建冻结注册。不要沿用原battery注册号证明本方案。
注册完成后，把新的注册URL交给Codex核查。execution_receipt.json必须提供registration_id与registered_filename。
脚本会从OSF注册API查验时间、注册文件归属，并下载核对精确ZIP哈希。

运行（在本frozen目录内）：
`C:\\DevTools\\Python311\\python.exe run_experiments.py --project-root D:\\论文\\RUL\\rs_tcn_gru_rul --dry-run`
`C:\\DevTools\\Python311\\python.exe run_experiments.py --project-root D:\\论文\\RUL\\rs_tcn_gru_rul --execute`
`--max-jobs 1`可用于注册后的首组运行；后续不带此参数自动继续。每次状态和每组日志保存在相邻execution目录。
全部完成后运行 `analyze_results.py --project-root D:\\论文\\RUL\\rs_tcn_gru_rul`。

首组真实训练必须在时间戳之后。本地锁定不等于外部预注册。synthetic-only单元测试不访问新实验结果。
'''
    (frozen/'README_注册与运行.md').write_text(readme, encoding='utf-8')
    paths = [p for p in sorted(frozen.rglob('*')) if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc' and p.name != 'freeze_manifest.json']
    checks = [{'path': p.relative_to(frozen).as_posix(), 'sha256': sha(p)} for p in paths]
    write_json(frozen/'freeze_manifest.json', checks)
    archive = STUDY/'PROTOCOL_FOR_OSF.zip'
    with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
        for p in paths + [frozen/'freeze_manifest.json']:
            z.write(p, p.relative_to(frozen).as_posix())
    write_json(STUDY/'prepared_identity.json', {'protocol_zip_sha256': sha(archive), 'protocol_zip': archive.name,
        'registration_status': 'PENDING', 'planned': len(jobs), 'training_started': False})
    print(f'PROTOCOL_PREPARED jobs={len(jobs)} new_training_started=0 archive={archive}', flush=True)


if __name__ == '__main__':
    main()
