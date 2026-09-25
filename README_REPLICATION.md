# Replication and data integrity

The v5.5.0 source release contains code and derived evidence for the RUL studies. NASA source data are not redistributed. The local dataset audit used the archived files listed in the Supplement; `FD004: 249 train engines, 248 test engines`. A prior complete local validation reported `DATA_VALIDATION_PASS` with `88 total, 0 failed`. This record is not a substitute for rerunning validation after obtaining the source data.

## Data Integrity

From the project root, run:

```powershell
python scripts/validate_data.py
```

The validator checks file presence, row and engine counts, RUL labels, and sensor values. Do not change the FD004 counts to match a landing-page summary without inspecting the hashed files.

## Core tests

```powershell
python scripts/run_core_tests.py
```

The current local suite contains 133 tests. All 133 passed from the prepared v5.5.0 source tree on 2026-09-26; seven submission-text contract tests also passed. Dataset validation still requires the official NASA source files, which are not redistributed here.

## Evidence timing

The C-MAPSS crossing is retrospective. Training began before the analysis plan was locked; the lock preceded full-result aggregation. The battery protocol was fixed before training and test-label evaluation. Neither status should be described as preregistration of the C-MAPSS crossing.
