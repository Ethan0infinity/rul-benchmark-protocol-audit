"""Execute frozen jobs with verified registration or an explicit recorded deviation."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
import time
import traceback
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
import pandas as pd


def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for data in iter(lambda: f.read(4*1024*1024), b''):
            h.update(data)
    return h.hexdigest()


def write_json(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.writing')
    temporary.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding='utf-8')
    temporary.replace(path)


def fetch_json(url):
    with urllib.request.urlopen(url, timeout=40) as response:
        return json.load(response)


def parse_osf_timestamp(value):
    stamp = datetime.fromisoformat(value.replace('Z', '+00:00'))
    # OSF v2 currently returns UTC date_registered without an explicit suffix.
    return stamp.replace(tzinfo=timezone.utc) if stamp.tzinfo is None else stamp


def acquire_queue_lock(path):
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open('a+b')
    if path.stat().st_size == 0:
        handle.write(b'0')
        handle.flush()
    handle.seek(0)
    try:
        if os.name == 'nt':
            import msvcrt
            msvcrt.locking(handle.fileno(), msvcrt.LK_NBLCK, 1)
        else:
            import fcntl
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        handle.close()
        raise RuntimeError('Another training queue is already running')
    return handle


def verify_registration(study):
    receipt_path = study/'execution_receipt.json'
    if not receipt_path.exists():
        raise RuntimeError('EXTERNAL_TIMESTAMP_REQUIRED: no new registered protocol receipt; no training started')
    receipt = json.loads(receipt_path.read_text(encoding='utf-8'))
    registration = receipt['registration_id']
    if not re.fullmatch('[a-z0-9]{5,12}', registration):
        raise ValueError('Invalid OSF registration ID')
    meta = fetch_json(f'https://api.osf.io/v2/registrations/{registration}/')['data']
    attributes = meta['attributes']
    if not attributes.get('public') or attributes.get('withdrawn') or attributes.get('pending_registration_approval'):
        raise ValueError('OSF registration is not public, frozen and active')
    stamp = parse_osf_timestamp(attributes['date_registered'])
    if stamp > datetime.now(timezone.utc):
        raise ValueError('Registration timestamp is in the future')
    url = f'https://api.osf.io/v2/registrations/{registration}/files/osfstorage/?page%5Bsize%5D=100'
    matches = []
    while url:
        if urlparse(url).hostname != 'api.osf.io':
            raise ValueError('Unexpected pagination host')
        listing = fetch_json(url)
        matches += [d for d in listing['data'] if d['attributes']['name'] == receipt['registered_filename']]
        url = listing.get('links', {}).get('next')
    if len(matches) != 1:
        raise ValueError('Exact protocol ZIP must be a unique root file of this registration')
    download = matches[0]['links']['download']
    if urlparse(download).scheme != 'https' or urlparse(download).hostname not in ('files.osf.io', 'api.osf.io', 'osf.io'):
        raise ValueError('Unexpected OSF download host')
    remote_hash = hashlib.sha256()
    with urllib.request.urlopen(download, timeout=40) as response:
        for data in iter(lambda: response.read(4*1024*1024), b''):
            remote_hash.update(data)
    expected = json.loads((study/'prepared_identity.json').read_text(encoding='utf-8'))['protocol_zip_sha256']
    if remote_hash.hexdigest() != expected or sha(study/'PROTOCOL_FOR_OSF.zip') != expected:
        raise ValueError('Registered protocol ZIP does not match prepared immutable ZIP')
    return {'registration_id': registration, 'date_registered': attributes['date_registered'],
            'verified_at_utc': datetime.now(timezone.utc).isoformat(), 'protocol_zip_sha256': expected,
            'file_id': matches[0]['id']}


def verify_local(frozen, project):
    study = frozen.parent
    expected = json.loads((study/'prepared_identity.json').read_text(encoding='utf-8'))['protocol_zip_sha256']
    if sha(study/'PROTOCOL_FOR_OSF.zip') != expected:
        raise ValueError('Prepared protocol ZIP changed')
    with zipfile.ZipFile(study/'PROTOCOL_FOR_OSF.zip') as z:
        for name in z.namelist():
            path = (frozen/name).resolve()
            path.relative_to(frozen)
            if sha(path) != hashlib.sha256(z.read(name)).hexdigest():
                raise ValueError(f'Working frozen file differs from registered ZIP: {name}')
    for row in json.loads((frozen/'freeze_manifest.json').read_text(encoding='utf-8')):
        path = (frozen/row['path']).resolve()
        path.relative_to(frozen)
        if sha(path) != row['sha256']:
            raise ValueError(f'Frozen source changed: {row["path"]}')
    for row in pd.read_csv(frozen/'data_manifest.csv').itertuples():
        path = (project/row.path).resolve()
        path.relative_to(project)
        if path.stat().st_size != row.bytes or sha(path) != row.sha256:
            raise ValueError(f'Input data changed: {row.path}')


def verify_execution_authority(study, authorization_file=None):
    if authorization_file is None:
        return verify_registration(study)
    authorization_file = authorization_file.resolve()
    authorization_file.relative_to(study.resolve())
    record = json.loads(authorization_file.read_text(encoding='utf-8'))
    expected = json.loads((study/'prepared_identity.json').read_text(encoding='utf-8'))['protocol_zip_sha256']
    if (record.get('mode') != 'local_lock_external_registration_pending'
            or record.get('user_authorized_start_before_external_registration') is not True
            or record.get('preregistered_before_training') is not False):
        raise ValueError('Explicit non-preregistered execution authorization required')
    if record.get('protocol_zip_sha256') != expected or sha(study/'PROTOCOL_FOR_OSF.zip') != expected:
        raise ValueError('Authorized protocol hash mismatch')
    if record.get('execution_runner_sha256') != sha(Path(__file__).resolve()):
        raise ValueError('Authorized execution runner changed')
    timestamp = parse_osf_timestamp(record['authorized_at_utc'])
    if timestamp > datetime.now(timezone.utc):
        raise ValueError('Authorization timestamp is in the future')
    return {**record, 'authorization_sha256': sha(authorization_file),
            'verified_at_utc': datetime.now(timezone.utc).isoformat(),
            'external_registration_verified': False}


def prepare_battery(frozen, project, fold, normalization, working):
    from prepare_frozen_inputs import extract_discharge_rows, fit_scaler, make_examples, merge_examples, save_bundle
    rows = pd.read_csv(frozen/'battery_grouped_splits.csv', dtype={'temperature': str})
    rows = rows[rows.fold == fold]
    inventory = pd.read_csv(frozen/'battery_eligible_units.csv').set_index('unit_id')
    original = project/'studies/prospective_battery_freeze_v1'
    data = {u: extract_discharge_rows(original/inventory.loc[u, 'source_file'], u) for u in rows.unit_id}
    train = rows[rows.assignment == 'training']
    scaler = fit_scaler([v for u in train.unit_id for v in data[u] if v is not None])
    per_temp = {t: fit_scaler([v for u in g.unit_id for v in data[u] if v is not None]) for t, g in train.groupby('temperature')}
    bundles = {}
    for role, group in rows.groupby('assignment'):
        parts = []
        for r in group.itertuples():
            mean, std = scaler if normalization == 'global' else per_temp[r.temperature]
            parts.append(make_examples(r.unit_id, data[r.unit_id], mean, std, 20))
        bundle = merge_examples(parts)
        if not len(bundle['y']) or not np.isfinite(bundle['x']).all():
            raise ValueError('Empty or nonfinite battery bundle')
        if set(bundle['unit_id']) != set(group.unit_id):
            raise ValueError('Planned battery lost all windows')
        bundles[role] = bundle
        if role != 'test':
            save_bundle(working/'prepared'/f'{role}_{normalization}.npz', bundle, True)
    write_json(working/'scaler_record.json', {'normalization': normalization, 'training_units': train.unit_id.tolist(),
               'global_mean': scaler[0].tolist(), 'global_std': scaler[1].tolist(),
               'temperature_scalers': {t: [x.tolist() for x in p] for t, p in per_temp.items()}})
    return bundles


def battery_job(job, frozen, project, working):
    import torch
    from battery_protocol import model_from_checkpoint
    from train_registered_models import train_one
    bundles = prepare_battery(frozen, project, job['fold'], job['normalization'], working)
    train, test = bundles['training'], bundles['test']
    x = train['x']
    counts = {u: int((train['unit_id'] == u).sum()) for u in set(train['unit_id'])}
    weights = np.array([1/counts[u] for u in train['unit_id']])
    weights *= len(weights)/weights.sum()
    model_name = job['model']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if job['family'] == 'battery_grouped':
        train_one(working, job['normalization'], model_name, job['stream_seed'], device)
        checkpoint = working/'checkpoints'/job['normalization']/model_name/f'seed_{job["stream_seed"]}.pt'
        # The reused trainer labels its old protocol internally. Correct this
        # new-run metadata explicitly; the original study/checkpoints are untouched.
        payload = torch.load(checkpoint, weights_only=False, map_location='cpu')
        payload['implementation_protocol_id'] = payload['protocol_id']
        payload['protocol_id'] = 'P10-RUL-PROTOCOL-REPLICATION-V1'
        payload['fold'] = job['fold']
        torch.save(payload, checkpoint)
        model, _ = model_from_checkpoint(checkpoint, device)
        with torch.no_grad():
            pred = np.concatenate([torch.clamp(model(torch.from_numpy(test['x'][i:i+1024]).to(device)), min=0).cpu().numpy()
                                   for i in range(0, len(test['x']), 1024)])
    elif model_name == 'life_prior':
        lifetimes = [train['lifetime'][train['unit_id'] == u][0] for u in sorted(counts)]
        pred = np.maximum(np.mean(lifetimes)-test['cycle_index'], 0)
    else:
        from sklearn.linear_model import Ridge
        from sklearn.ensemble import HistGradientBoostingRegressor
        def features(array):
            return np.concatenate([array.mean(1), array.std(1), array[:, -1], array[:, -1]-array[:, 0]], axis=1)
        estimator = Ridge(alpha=1.0) if model_name == 'ridge' else HistGradientBoostingRegressor(
            max_iter=100, max_leaf_nodes=15, learning_rate=.1, l2_regularization=1., early_stopping=False, random_state=job['stream_seed'])
        estimator.fit(features(x), train['y'], sample_weight=weights)
        pred = np.maximum(estimator.predict(features(test['x'])), 0)
        import pickle
        with (working/'baseline_model.pkl').open('wb') as f:
            pickle.dump(estimator, f)
    frame = pd.DataFrame({k: test[k] for k in ('sample_id', 'unit_id', 'cycle_index', 'lifetime')})
    frame['true_rul'], frame['pred_rul'] = test['y'], pred
    if not np.isfinite(pred).all():
        raise ValueError('Nonfinite prediction')
    frame.to_csv(working/'test_predictions.csv', index=False)
    return [working/'test_predictions.csv'] + list(working.rglob('*.pt')) + list(working.glob('*.pkl'))


def execute_job(job, frozen, project, working):
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    sys.path.insert(0, str(frozen/'source/src'))
    sys.path.insert(0, str(frozen/'source/battery'))
    if job['family'] != 'cmapss_crossed':
        return battery_job(job, frozen, project, working)
    from rul.training import train_model
    config = json.loads((frozen/job['config']).read_text(encoding='utf-8'))
    config['_project_root'] = str(project)
    config['_config_path'] = str(frozen/job['config'])
    result = train_model(config, subset=job['task'], model_name=job['model'])
    directory = Path(result['run_dir'])
    df = pd.read_csv(directory/'test_predictions.csv')
    if df.unit_id.duplicated().any() or not np.isfinite(df[['true_rul', 'pred_rul']].to_numpy()).all():
        raise ValueError('Invalid C-MAPSS test predictions')
    write_json(working/'result_location.json', {'path': directory.relative_to(project).as_posix()})
    return [directory/name for name in ('test_predictions.csv', 'metrics.json', 'best_model.pt', 'run_config.yaml', 'epoch_history.csv')]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--frozen-root', type=Path, help='Unchanged protocol snapshot; default is this script directory')
    parser.add_argument('--local-authorization', type=Path,
                        help='Explicit recorded deviation; results are NOT externally preregistered')
    parser.add_argument('--execute', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--max-jobs', type=int)
    parser.add_argument('--family', choices=('all', 'cmapss_crossed', 'battery_grouped', 'battery_baseline'), default='all')
    parser.add_argument('--worker-job', help=argparse.SUPPRESS)
    parser.add_argument('--attempt', type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    frozen = (args.frozen_root or Path(__file__).resolve().parent).resolve()
    project = args.project_root.resolve()
    study = frozen.parent
    jobs = json.loads((frozen/'jobs.json').read_text(encoding='utf-8'))
    if args.worker_job:
        # Workers independently verify both frozen inputs and execution authority.
        verify_local(frozen, project)
        authority = verify_execution_authority(study, args.local_authorization)
        job = next(j for j in jobs if j['job_id'] == args.worker_job)
        artifacts = execute_job(job, frozen, project, args.attempt)
        write_json(args.attempt/'completion.json', {'job_id': job['job_id'], 'status': 'valid',
            'execution_authority': authority,
            'artifacts': [{'path': p.relative_to(project).as_posix(), 'sha256': sha(p)} for p in artifacts]})
        return
    selected = [j for j in jobs if args.family == 'all' or j['family'] == args.family]
    verify_local(frozen, project)
    print(f'LOCAL_PROTOCOL_PASS total_planned={len(jobs)} selected={len(selected)}', flush=True)
    if not args.execute or args.dry_run:
        if args.local_authorization:
            verify_execution_authority(study, args.local_authorization)
            print('LOCAL_DEVIATION_VERIFIED; NOT externally preregistered', flush=True)
        print('DRY_RUN_ONLY; no training started', flush=True)
        return
    registration = verify_execution_authority(study, args.local_authorization)
    execution = study/'execution'
    execution.mkdir(exist_ok=True)
    queue_lock = acquire_queue_lock(execution/'queue.lock')
    write_json(execution/f'registration_check_{time.time_ns()}.json', registration)
    if not (execution/'first_execution.json').exists():
        write_json(execution/'first_execution.json', {
            'first_execution_at_utc': datetime.now(timezone.utc).isoformat(),
            'registration': registration})
    def progress(state, current_job=None):
        write_json(execution/'queue_progress.json', {
            'state': state, 'total_planned': len(jobs), 'selected': len(selected),
            'completion_records': sum(bool(list((execution/j['job_id']).glob('attempt_*/completion.json'))) for j in jobs),
            'current_job': current_job, 'pid': os.getpid(),
            'updated_at_utc': datetime.now(timezone.utc).isoformat(),
            'evidence_mode': registration.get('mode', 'verified_external_registration')})
    done = 0
    for index, job in enumerate(selected, 1):
        jobdir = execution/job['job_id']
        completions = sorted(jobdir.glob('attempt_*/completion.json'))
        if completions:
            saved = json.loads(completions[-1].read_text(encoding='utf-8'))
            if all((project/r['path']).is_file() and sha(project/r['path']) == r['sha256'] for r in saved['artifacts']):
                print(f'SKIP_VALID {index}/{len(selected)} {job["job_id"]}', flush=True)
                continue
            raise RuntimeError('Previously completed artifacts changed; explicit audit required')
        attempt = jobdir/f'attempt_{time.time_ns()}'
        attempt.mkdir(parents=True)
        if job['family'] == 'cmapss_crossed':
            existing = project/'studies/protocol_replication_p10/results/cmapss'/job['job_id']
            if existing.exists():
                import shutil
                shutil.copytree(existing, attempt/'previous_partial_output')
        start = time.monotonic()
        started_at = datetime.now(timezone.utc).isoformat()
        write_json(attempt/'status.json', {'state': 'started', 'job': job, 'started_at_utc': started_at})
        progress('running', job['job_id'])
        print(f'START {index}/{len(selected)} {job["job_id"]}', flush=True)
        env = os.environ.copy()
        env.update(PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1', CUBLAS_WORKSPACE_CONFIG=':4096:8')
        command = [sys.executable, str(Path(__file__).resolve()), '--project-root', str(project),
                   '--frozen-root', str(frozen), '--worker-job', job['job_id'], '--attempt', str(attempt)]
        if args.local_authorization:
            command += ['--local-authorization', str(args.local_authorization.resolve())]
        with (attempt/'console.log').open('w', encoding='utf-8') as log:
            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                                       encoding='utf-8', errors='replace', env=env)
            try:
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    print(line, end='', flush=True)
                code = process.wait()
            except BaseException:
                process.terminate()
                process.wait()
                write_json(attempt/'status.json', {'state': 'interrupted', 'job': job,
                           'started_at_utc': started_at, 'elapsed_seconds': time.monotonic()-start})
                raise
        write_json(attempt/'status.json', {'state': 'valid' if code == 0 else 'failed', 'job': job,
                   'elapsed_seconds': time.monotonic()-start, 'exit_code': code,
                   'started_at_utc': started_at, 'finished_at_utc': datetime.now(timezone.utc).isoformat()})
        if code:
            progress('failed', job['job_id'])
            raise SystemExit(f'JOB_FAILED retained logs/partial output: {attempt}')
        done += 1
        progress('between_jobs')
        if args.max_jobs is not None and done >= args.max_jobs:
            print('REQUESTED_BATCH_COMPLETE; remaining planned runs are still pending', flush=True)
            return
    print('SELECTED_JOB_SET_COMPLETE; use analyze_results.py for full-family accounting', flush=True)
    progress('selected_job_set_complete')


if __name__ == '__main__':
    main()
