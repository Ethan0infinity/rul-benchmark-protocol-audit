# Benchmark-Claim Traceability and Maintenance-Decision Sensitivity for RUL Evaluation

Minimal source-code and compact C-MAPSS evidence repository accompanying the manuscript:

**Benchmark-Claim Traceability and Maintenance-Decision Sensitivity for Remaining-Useful-Life Evaluation: Two Worked Studies**

This study examines how remaining-useful-life (RUL) benchmark claims depend on the estimand, statistical unit, design history, evidence role, artifact provenance, completed design choices, and permitted wording. It adds post-result practical-equivalence bands and a static inspection-trigger sensitivity that links endpoint predictions to actions and normalized asymmetric error cost. The latter is not an optimized maintenance policy, field reliability model, or calibrated monetary utility.

## Evidence design

The manuscript contains two deliberately separated evidence layers:

1. **Retrospective C-MAPSS audit.** Fixed-task and sensitivity analyses retain adverse, null, mixed, and reversing outcomes. Excluding development-informed FD004 gives an equal-task RMSE contrast of -0.134 cycles; including FD004 changes it to +0.123 cycles. The FD004 LPR@30 contrast is -0.117 and attenuates to -0.038 at a five-cycle tolerance.
2. **Maintenance-decision sensitivity.** A static service trigger is evaluated over lead times 10/20/30 cycles and late-to-early cost ratios 2/5/10. OCM-Asym minus RAST-GRU mean normalized cost ranges from -0.0142 to +0.0048 per engine. The mixed direction establishes decision sensitivity, not stable policy benefit.
3. **Prospectively frozen NASA battery audit.** The protocol was externally registered before training and test-label evaluation. Under the registered primary normalization, Battery-GRU-Asym minus Battery-GRU-Core is +2.546 cycles for equal-battery macro RMSE and -0.0777 for LPR at 10% remaining life. Support is mixed, the LPR direction reverses under temperature-specific normalization, and post-registration simple baselines show that the registered GRUs are not competitively strong on this split.

These are finite-design estimates. They are not population-level probabilities of superiority, field reliability estimates, or evidence of safer maintenance operation.

## Evidence records

| Record | Purpose | Identifier |
|---|---|---|
| GitHub repository | Tagged source and compact C-MAPSS semantic evidence | <https://github.com/Ethan0infinity/rul-benchmark-protocol-audit> |
| Complete evidence archive | C-MAPSS derived evidence, prospective battery execution artifacts, manifests, and checksums | [OSF u76ze](https://osf.io/u76ze/), DOI [10.17605/OSF.IO/U76ZE](https://doi.org/10.17605/OSF.IO/U76ZE) |
| Frozen battery protocol | Pre-outcome protocol, split, code, hashes, and wording rule | [OSF xqsv8](https://osf.io/xqsv8/) |

Complete archive file: `RUL_Benchmark_Claim_Audit_Complete_Evidence_v1.zip`

Archive SHA-256:

```text
48ffbbccf2e853243e2dd34b88a12ec6561bba0c325e43c5bfa91cf9e11fdfad
```

The later public release improves artifact resolvability but does not change the retrospective status of the C-MAPSS analyses or constitute independent-team replication.

## Repository contents

- `src/rul/`: C-MAPSS data, model, metric, training, protocol, and reporting code.
- `configs/`: default, formal benchmark, and locked protocol configurations.
- `scripts/`: installation checks, data setup, training, formal benchmarking, readiness checks, and table generation.
- `paper_outputs/release_v1/`: compact machine-readable C-MAPSS evidence with a manifest and schema metadata.
- `paper_outputs/ress_major_revision/`: post-result maintenance, practical-equivalence, battery-baseline, target-definition, and chronology outputs.
- `studies/six_link_comparator_study/`: an unexecuted blinded independent-rater protocol and empty response schemas; it contains no participant data.
- `data/README.md`: dataset acquisition and redistribution boundaries.
- `results/README.md`: relationship between local run outputs and the public complete archive.

This GitHub repository intentionally stays compact. Checkpoints, per-engine predictions, runtime logs, the full retrospective extension, and prospective battery execution artifacts are distributed through the complete OSF evidence archive.

Generator paths recorded in `paper_outputs/release_v1/release_manifest.csv` refer to the complete evidence build tree. Those analysis generators and their recorded SHA-256 values are retained in the complete OSF archive and are intentionally not duplicated in this compact source profile.

## Provenance correction for descriptive latency

The machine-readable release record `paper_outputs/release_v1/inference_latency_protocol_summary.csv` and its manifest entry are authoritative for the descriptive FD004 latency audit. An earlier rendered Supplement table retained a superseded timing run (1.167 ms rather than 1.612 ms for OCM-Asym, CPU batch 1). The concise Supplement now regenerates this exhibit directly from the release-v1 values. The correction changes only system-state-sensitive descriptive timing and does not alter predictions, accuracy, late-overprediction results, or any scientific headline estimate. The immutable OSF v1 archive is not repacked; this note records the discrepancy and its resolution.

## Verify the major-revision evidence

The CSVs in `paper_outputs/ress_major_revision/` are directly inspectable in this compact repository. Rebuilding them requires the complete prediction and prospective-battery objects from the OSF evidence archive placed at their documented project paths:

```bash
python scripts/build_ress_major_revision_evidence.py
python scripts/prepare_comparator_study.py --check-only
```

The comparator command must report `COMPARATOR_STUDY_READY_NOT_EXECUTED` and zero independent responses. It does not run or simulate a human study.

## Install and verify

Python 3.11 is recommended.

```bash
python -m venv .venv
python -m pip install -r requirements.txt
python scripts/verify_installation.py
```

Activate the virtual environment using the command appropriate for the operating system before installing dependencies. The verification command imports the scientific stack, validates the published configurations, runs forward checks for supported models, and verifies compact evidence files against `release_manifest.csv`.

## Obtain and validate C-MAPSS

Raw NASA C-MAPSS files are not redistributed.

```bash
python scripts/download_cmapss.py --extract
python scripts/validate_data.py
```

Alternatively, place the public files under `data/raw/` before running the validator.

## Reproduce C-MAPSS training

Run a short smoke experiment first:

```bash
python scripts/train.py --config configs/default.yaml --subset FD001 --model gru --epochs 2 --no-batch-bar
```

Inspect and execute the formal benchmark matrix:

```bash
python scripts/run_formal_benchmark.py --dry-run
python scripts/run_formal_benchmark.py --no-batch-bar
python scripts/check_benchmark_readiness.py
python scripts/build_paper_tables.py --experiment-prefix paper_main_v3
```

Full retraining is computationally expensive. The compact files in `paper_outputs/release_v1/` support semantic inspection without downloading checkpoints; the complete OSF archive supports the broader artifact audit reported in the manuscript.

## Interpretation boundary

- The formal C-MAPSS matrix contains 13 configurations, four fixed tasks, and five composite seeds: 260 runs.
- OCM-Asym is an illustrative, development-informed operating point.
- OCM-Asym versus RAST-GRU is a complete-configuration comparison; independent module effects are not identified.
- Five composite seeds support finite-design summaries and diagnostics, not confirmatory population inference.
- Normalized-coordinate perturbations are algorithmic sensitivity probes, not physical fault validation.
- N-CMAPSS DS02 is a strongly downsampled computational proxy, not independent external validation.
- The battery audit demonstrates prospective execution of the reporting record under one frozen split, not comparative estimator efficacy.
- Candidate practical-equivalence margins are transparent post-result bands, not validated SESOIs or formal equivalence tests.
- Static maintenance-trigger costs are post-result decision sensitivities, not optimized policy utilities.
- The six-link record is a worked reporting template. Comparative efficacy against REFORMS, Evaluation Cards, Eval Factsheets, Model Cards, or a conventional checklist remains untested.

## Release

- Version: `5.4.0`
- Tag: `v5.4.0`
- Repository migration: `2026-08-20`; the scientific evidence content is unchanged from the preserved v5.4.0 source package.
- Complete evidence DOI: `10.17605/OSF.IO/U76ZE`
- License: MIT for original repository code; third-party materials retain their upstream terms.

`artifact_metadata.json` records the source/archive roles. The regenerated submission object additionally carries `submission_identity.json`, which records the resolved annotated tag, peeled commit, final manuscript and Supplement hashes, archive hash, and build time. The OSF record is the authority for the complete post-study evidence archive; this repository is the authority for readable source inspection and fresh C-MAPSS retraining.
