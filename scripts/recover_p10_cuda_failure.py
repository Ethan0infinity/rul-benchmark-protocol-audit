"""Retry one recorded CUDA infrastructure failure in isolation, then resume p10."""
from __future__ import annotations

import argparse
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


def utc():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--job-id', required=True)
    args = parser.parse_args()
    project = args.project_root.resolve()
    study = project/'studies/protocol_replication_p10'
    frozen = study/'frozen'
    execution = study/'execution'
    worker = study/'amendments/start_before_osf_20260913/run_authorized_experiments.py'
    authorization = worker.parent/'authorization.json'
    spec = importlib.util.spec_from_file_location('p10_authorized_worker', worker)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.verify_local(frozen, project)
    authority = runner.verify_execution_authority(study, authorization)
    jobs = json.loads((frozen/'jobs.json').read_text(encoding='utf-8'))
    matches = [job for job in jobs if job['job_id'] == args.job_id]
    if len(matches) != 1:
        raise ValueError('Retry job must identify exactly one planned design cell')
    job = matches[0]
    old_attempts = sorted((execution/args.job_id).glob('attempt_*'))
    failures = []
    for old in old_attempts:
        status_path = old/'status.json'
        if status_path.exists():
            status = json.loads(status_path.read_text(encoding='utf-8-sig'))
            if status.get('state') == 'failed':
                failures.append({'attempt': old.relative_to(project).as_posix(), 'status': status})
        if (old/'completion.json').exists():
            raise RuntimeError('A valid completion exists; explicit retry is forbidden')
    if len(failures) != 1 or failures[0]['status'].get('exit_code') != 1:
        raise RuntimeError('Expected exactly one recorded failed attempt')
    console = project/failures[0]['attempt']/'console.log'
    if 'CUDA error: an illegal memory access was encountered' not in console.read_text(encoding='utf-8', errors='replace'):
        raise RuntimeError('Failure signature does not match authorized infrastructure retry')
    lock = runner.acquire_queue_lock(execution/'queue.lock')
    attempt = execution/args.job_id/f'attempt_{time.time_ns()}'
    attempt.mkdir(parents=True)
    prior_output = study/'results/cmapss'/args.job_id
    if prior_output.exists():
        shutil.copytree(prior_output, attempt/'previous_partial_output')
    recovery_dir = execution/'recovery_runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    recovery_dir.mkdir(parents=True)
    shutil.copy2(Path(__file__), recovery_dir/'recovery_supervisor_snapshot.py')
    metadata = {
        'job_id': args.job_id, 'started_at_utc': utc(),
        'reason': 'Exact-config isolated retry after one CUDA illegal-memory-access failure.',
        'classification': 'provisional infrastructure failure; retry outcome determines escalation',
        'previous_failure': failures[0]['attempt'], 'protocol_zip_sha256': authority['protocol_zip_sha256'],
        'scientific_configuration_changes': [], 'seed_changes': [], 'worker_concurrency': 1,
        'cpu_threads': 2, 'cuda_launch_blocking': False,
        'policy': 'One isolated exact-config retry. On success resume two workers; on failure stop.'}
    runner.write_json(recovery_dir/'retry_metadata.json', metadata)
    status = {'state': 'started_exact_config_retry', 'job': job, 'started_at_utc': metadata['started_at_utc'],
              'retry_metadata': (recovery_dir/'retry_metadata.json').relative_to(project).as_posix()}
    runner.write_json(attempt/'status.json', status)
    command = [sys.executable, '-u', str(worker), '--project-root', str(project), '--frozen-root', str(frozen),
               '--local-authorization', str(authorization), '--worker-job', args.job_id, '--attempt', str(attempt)]
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS='2', MKL_NUM_THREADS='2',
               OPENBLAS_NUM_THREADS='2', NUMEXPR_NUM_THREADS='2')
    start = time.monotonic()
    with (attempt/'console.log').open('w', encoding='utf-8') as log:
        process = subprocess.Popen(command, cwd=project, env=env, stdout=log, stderr=subprocess.STDOUT)
        code = process.wait()
    valid = False
    completion = attempt/'completion.json'
    if code == 0 and completion.exists():
        saved = json.loads(completion.read_text(encoding='utf-8'))
        valid = saved.get('job_id') == args.job_id and saved.get('status') == 'valid'
        for item in saved.get('artifacts', []):
            path = (project/item['path']).resolve()
            path.relative_to(project)
            valid = valid and path.is_file() and runner.sha(path) == item['sha256']
    finished = {**status, 'state': 'valid_exact_config_retry' if valid else 'failed_exact_config_retry',
                'finished_at_utc': utc(), 'elapsed_seconds': time.monotonic()-start, 'exit_code': code}
    runner.write_json(attempt/'status.json', finished)
    runner.write_json(recovery_dir/'retry_result.json', {'valid': valid, 'exit_code': code,
                      'finished_at_utc': finished['finished_at_utc'], 'attempt': attempt.relative_to(project).as_posix()})
    lock.close()
    if not valid:
        raise SystemExit('Exact-config isolated retry failed; queue remains stopped for diagnosis')
    resume = [sys.executable, '-u', str(project/'scripts/run_p10_parallel_queue.py'),
              '--project-root', str(project), '--workers', '2', '--cpu-threads', '2']
    runner.write_json(recovery_dir/'resume_handoff.json', {'resumed_at_utc': utc(), 'command': resume})
    raise SystemExit(subprocess.call(resume, cwd=project, env=env))


if __name__ == '__main__':
    main()
