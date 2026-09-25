"""Run the full explicitly authorized queue, then reconcile all result accounting."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', type=Path, required=True)
    args = parser.parse_args()
    project = args.project_root.resolve()
    study = project/'studies/protocol_replication_p10'
    amendment = study/'amendments/start_before_osf_20260913'
    logdir = study/'execution'
    logdir.mkdir(exist_ok=True)
    started = datetime.now(timezone.utc).isoformat()
    state = {'pid': os.getpid(), 'started_at_utc': started,
             'mode': 'local_lock_external_registration_pending', 'state': 'running'}
    path = logdir/'supervisor_status.json'
    path.write_text(json.dumps(state, indent=2), encoding='utf-8')
    command = [sys.executable, '-u', str(amendment/'run_authorized_experiments.py'),
               '--project-root', str(project), '--frozen-root', str(study/'frozen'),
               '--local-authorization', str(amendment/'authorization.json'), '--execute']
    env = os.environ.copy()
    env.update(PYTHONUNBUFFERED='1', PYTHONIOENCODING='utf-8', PYTHONDONTWRITEBYTECODE='1')
    code = 1
    try:
        code = subprocess.call(command, cwd=project, env=env)
        state['training_exit_code'] = code
        # Account for successes and failures even when the queue stops early.
        analysis_code = subprocess.call([sys.executable, str(study/'frozen/analyze_results.py'),
                                        '--project-root', str(project)], cwd=project, env=env)
        state['analysis_exit_code'] = analysis_code
        if code == 0 and analysis_code == 0:
            summary = json.loads((study/'analysis/completion_status.json').read_text(encoding='utf-8'))
            state['state'] = 'complete' if summary['status'] == 'ALL_NEW_TRAINING_COMPLETE' else 'incomplete'
        else:
            state['state'] = 'failed'
    except BaseException as error:
        state.update(state='interrupted_or_failed', error=repr(error))
        raise
    finally:
        state['finished_at_utc'] = datetime.now(timezone.utc).isoformat()
        path.write_text(json.dumps(state, indent=2), encoding='utf-8')
    if state['state'] != 'complete':
        raise SystemExit(code or 1)


if __name__ == '__main__':
    main()
