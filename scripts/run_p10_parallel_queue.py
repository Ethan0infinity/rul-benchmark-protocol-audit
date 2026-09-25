"""Two-process scheduler over the unchanged, locally authorized p10 worker."""
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


def load_runner(project):
    path = project/'studies/protocol_replication_p10/amendments/start_before_osf_20260913/run_authorized_experiments.py'
    spec = importlib.util.spec_from_file_location('p10_authorized_worker', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, path


def validated_pending(jobs, execution, project, runner):
    if len({j['job_id'] for j in jobs}) != len(jobs):
        raise ValueError('Duplicate planned job identities')
    pending, completed = [], []
    for job in jobs:
        paths = sorted((execution/job['job_id']).glob('attempt_*/completion.json'))
        if not paths:
            pending.append(job)
            continue
        data = json.loads(paths[-1].read_text(encoding='utf-8'))
        if data.get('job_id') != job['job_id'] or data.get('status') != 'valid' or not data.get('artifacts'):
            raise ValueError('Invalid completion identity: '+job['job_id'])
        for item in data['artifacts']:
            path = (project/item['path']).resolve()
            path.relative_to(project)
            if not path.is_file() or runner.sha(path) != item['sha256']:
                raise ValueError('Completed artifact changed: '+str(path))
        completed.append(job['job_id'])
    return pending, completed


def thread_environment(count):
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1',
               CUBLAS_WORKSPACE_CONFIG=':4096:8', OMP_NUM_THREADS=str(count),
               MKL_NUM_THREADS=str(count), OPENBLAS_NUM_THREADS=str(count), NUMEXPR_NUM_THREADS=str(count))
    return env


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, required=True)
    parser.add_argument('--workers', type=int, choices=(1, 2), default=2)
    parser.add_argument('--cpu-threads', type=int, choices=(1, 2, 4), default=2)
    parser.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    project = args.project_root.resolve()
    study = project/'studies/protocol_replication_p10'
    frozen, execution = study/'frozen', study/'execution'
    runner, worker_script = load_runner(project)
    runner.verify_local(frozen.resolve(), project)
    authority = runner.verify_execution_authority(study, worker_script.parent/'authorization.json')
    jobs = json.loads((frozen/'jobs.json').read_text(encoding='utf-8'))
    pending, complete = validated_pending(jobs, execution, project, runner)
    print(f'PARALLEL_PREFLIGHT_PASS planned={len(jobs)} verified_complete={len(complete)} pending={len(pending)} workers={args.workers}', flush=True)
    if args.dry_run:
        return
    # Shares the original serial queue lock. Never run both schedulers together.
    lock = runner.acquire_queue_lock(execution/'queue.lock')
    # A completed worker can exit before its stopped serial parent updates status.
    for job in jobs:
        for status_path in (execution/job['job_id']).glob('attempt_*/status.json'):
            status = json.loads(status_path.read_text(encoding='utf-8-sig'))
            if status.get('state') != 'started':
                continue
            completion = status_path.parent/'completion.json'
            shutil.copy2(status_path, status_path.parent/'status_before_parallel_handoff.json')
            status.update(state='valid' if completion.exists() and job['job_id'] in complete else 'interrupted_before_resume',
                          reconciled_at_utc=utc(),
                          reconciliation='No live owning scheduler at resume; original partial output and status snapshot retained')
            runner.write_json(status_path, status)
    run = execution/'parallel_runs'/datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S_%fZ')
    run.mkdir(parents=True)
    shutil.copy2(Path(__file__), run/'scheduler_snapshot.py')
    metadata = {
        'started_at_utc': utc(), 'pid': os.getpid(), 'workers': args.workers,
        'cpu_threads_per_worker': args.cpu_threads, 'user_instruction': '可以并行跑实验',
        'worker_sha256': runner.sha(worker_script), 'scheduler_sha256': runner.sha(Path(__file__)),
        'protocol_zip_sha256': authority['protocol_zip_sha256'],
        'evidence_mode': authority['mode'], 'preregistered_before_training': False,
        'configuration_changes': [],
        'runtime_changes': ['Concurrent GPU execution', 'Per-process CPU numerical-library thread budget'],
        'limitations': ['Concurrent timing/peak memory is not comparable with isolated profiling.',
                       'CPU thread changes can affect floating-point reduction order; bitwise identity is not claimed.'],
        'previous_verified_complete': complete}
    runner.write_json(run/'runtime_amendment.json', metadata)
    for name in ('supervisor_status.json', 'queue_progress.json'):
        if (execution/name).exists():
            shutil.copy2(execution/name, run/('previous_'+name))
    active, failures = {}, []
    env = thread_environment(args.cpu_threads)
    state = 'running'

    def progress():
        now = utc()
        runner.write_json(execution/'queue_progress.json', {
            'state': state, 'total_planned': len(jobs), 'completion_records': len(complete),
            'workers': args.workers, 'pid': os.getpid(), 'updated_at_utc': now,
            'active_jobs': [{'job_id': entry['job']['job_id'], 'pid': process.pid,
                             'attempt': entry['attempt'].relative_to(project).as_posix(),
                             'elapsed_seconds': round(time.monotonic()-entry['clock'], 1)}
                            for process, entry in active.items()],
            'failed_jobs': failures, 'pending_not_dispatched': len(pending),
            'evidence_mode': authority['mode'], 'runtime_amendment': (run/'runtime_amendment.json').relative_to(project).as_posix()})
        runner.write_json(execution/'supervisor_status.json', {
            'state': state, 'pid': os.getpid(), 'started_at_utc': metadata['started_at_utc'],
            'updated_at_utc': now, 'workers': args.workers, 'mode': authority['mode']})

    try:
        while pending or active:
            while pending and len(active) < args.workers and not failures:
                job = pending.pop(0)
                attempt = execution/job['job_id']/f'attempt_{time.time_ns()}'
                attempt.mkdir(parents=True)
                if job['family'] == 'cmapss_crossed':
                    previous = study/'results/cmapss'/job['job_id']
                    if previous.exists():
                        shutil.copytree(previous, attempt/'previous_partial_output')
                record = {'state': 'started', 'job': job, 'started_at_utc': utc(),
                          'runtime_amendment': (run/'runtime_amendment.json').relative_to(project).as_posix()}
                runner.write_json(attempt/'status.json', record)
                runner.write_json(attempt/'runtime_conditions.json', {
                    'concurrent_limit': args.workers, 'cpu_threads': args.cpu_threads,
                    'timing_role': 'shared-resource execution, not isolated benchmark',
                    'runtime_amendment': record['runtime_amendment']})
                command = [sys.executable, '-u', str(worker_script), '--project-root', str(project),
                           '--frozen-root', str(frozen), '--local-authorization', str(worker_script.parent/'authorization.json'),
                           '--worker-job', job['job_id'], '--attempt', str(attempt)]
                log = (attempt/'console.log').open('w', encoding='utf-8')
                try:
                    process = subprocess.Popen(command, cwd=project, env=env, stdout=log, stderr=subprocess.STDOUT)
                except BaseException:
                    log.close()
                    runner.write_json(attempt/'status.json', {**record, 'state': 'launch_failed', 'finished_at_utc': utc()})
                    raise
                active[process] = {'job': job, 'attempt': attempt, 'record': record,
                                   'clock': time.monotonic(), 'log': log}
                print(f'START pid={process.pid} {job["job_id"]}', flush=True)
            for process, entry in list(active.items()):
                code = process.poll()
                if code is None:
                    continue
                entry['log'].close()
                job, attempt = entry['job'], entry['attempt']
                valid = False
                if code == 0:
                    remainder, verified = validated_pending([job], execution, project, runner)
                    valid = not remainder and len(verified) == 1
                record = {**entry['record'], 'state': 'valid' if valid else 'failed',
                          'finished_at_utc': utc(), 'exit_code': code,
                          'elapsed_seconds': time.monotonic()-entry['clock']}
                runner.write_json(attempt/'status.json', record)
                if valid:
                    complete.append(job['job_id'])
                else:
                    failures.append(job['job_id'])
                del active[process]
                print(f'{"VALID" if valid else "FAILED"} {len(complete)}/{len(jobs)} {job["job_id"]}', flush=True)
            if failures:
                state = 'failure_draining_active_jobs'
                if not active:
                    break
            progress()
            if pending or active:
                time.sleep(5)
        state = 'failed' if failures else 'training_complete_analysis_running'
        progress()
        code = subprocess.call([sys.executable, str(frozen/'analyze_results.py'), '--project-root', str(project)],
                               cwd=project, env=env)
        state = 'complete' if not failures and len(complete) == len(jobs) and code == 0 else 'failed_or_incomplete'
        progress()
        runner.write_json(run/'finished.json', {'state': state, 'finished_at_utc': utc(),
                          'completed': len(complete), 'analysis_exit_code': code, 'failures': failures})
    except BaseException:
        state = 'interrupted_or_failed'
        for process, entry in list(active.items()):
            if process.poll() is None:
                process.terminate()
            process.wait()
            entry['log'].close()
            runner.write_json(entry['attempt']/'status.json', {**entry['record'], 'state': 'interrupted',
                'finished_at_utc': utc(), 'elapsed_seconds': time.monotonic()-entry['clock']})
        active.clear()
        progress()
        raise
    finally:
        lock.close()
    if state != 'complete':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
