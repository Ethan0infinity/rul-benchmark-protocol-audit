"""Record the user's explicit start-before-registration deviation without changing the ZIP."""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--project-root', required=True, type=Path)
    args = parser.parse_args()
    project = args.project_root.resolve()
    study = project/'studies/protocol_replication_p10'
    source = project/'scripts/run_p10_training_protocol.py'
    spec = importlib.util.spec_from_file_location('execution_runner', source)
    runner = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(runner)
    runner.verify_local((study/'frozen').resolve(), project)
    directory = study/'amendments/start_before_osf_20260913'
    authorization = directory/'authorization.json'
    if directory.exists():
        raise RuntimeError('Amendment directory exists; audit it instead of overwriting authorization')
    directory.mkdir(parents=True)
    snapshot = directory/'run_authorized_experiments.py'
    shutil.copy2(source, snapshot)
    protocol_hash = json.loads((study/'prepared_identity.json').read_text(encoding='utf-8'))['protocol_zip_sha256']
    record = {
        'mode': 'local_lock_external_registration_pending',
        'user_authorized_start_before_external_registration': True,
        'preregistered_before_training': False,
        'authorized_at_utc': datetime.now(timezone.utc).isoformat(),
        'timestamp_role': 'local recording of explicit user instruction; not independent attestation',
        'user_instruction': '请你先执行实验，稍后发给你，因为现在OSF官网好像出现了服务器问题。',
        'reason': 'User requested execution before OSF registration; server issue is user-reported, not independently confirmed.',
        'protocol_zip_sha256': protocol_hash,
        'execution_runner_sha256': runner.sha(snapshot),
        'planned_fits': 504,
        'scientific_configuration_changes': [],
        'execution_changes': ['Explicit local authorization in place of external registration gate',
                              'Per-job authority record and live queue progress'],
        'disclosure': 'External registration pending at execution start. Any later registration is retrospective for already-started fits.'}
    runner.write_json(authorization, record)
    print(json.dumps(record, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
