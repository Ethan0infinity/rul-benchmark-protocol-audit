# Full 5 x 10 Evidence Timeline

This chronology records protocol and execution identity. It does not contain
scientific results and does not authorize interim aggregation.

This file is a dated snapshot prepared on 2026-09-25, before the v5.5.0 OSF
evidence file was uploaded. Its statements below that public artifact assembly
was pending describe the status on that date. The upload was subsequently
verified as recorded in `docs/OSF_PUBLIC_UPLOAD_VERIFICATION.md`.

## Chronology note

Formal 5 x 10 execution began before the analysis-plan and Amendment 01
freezes. These documents therefore establish a **pre-aggregation analysis
lock**, not a pre-execution or preregistered design. No completed 5 x 10
scientific aggregate was generated before the lock.

## Timeline

| Event | Status | Timestamp or condition |
|---|---|---|
| Original full 5 x 10 analysis plan frozen | complete | 2026-09-22 23:35:43 Asia/Shanghai |
| Amendment 01 frozen | complete | 2026-09-22 23:46:51 Asia/Shanghai |
| First canonical run artifact observed | complete | 2026-09-22 03:36:13 Asia/Shanghai; filesystem creation time, auxiliary only |
| Formal 400-cell training started | complete | execution was underway before the 2026-09-22 plan freeze; no pre-execution preregistration claim |
| Final canonical run completed | complete | 400/400 rows, all status `completed`; manifest last written 2026-09-25 18:40:30 Asia/Shanghai |
| Canonical manifest QA | complete | 400 unique cells; all rows passed configuration, checkpoint, finite-output, metric-recomputation, and patience-based stopping checks |
| Raw canonical manifest hashed | complete | SHA-256 recorded below after QA; this is an analysis-input identity, not a release identity |
| Frozen aggregation | complete | finite-design aggregation passed on the locked manifest; code hash and provenance are recorded in the QA note |
| Aggregate outputs generated | complete | 400 run metrics, 200 matched contrasts, task summaries, factor marginals, candidate bands, pairwise rank support, and 12 heatmaps |
| Ranking integration | complete | same-family RAST-GRU vs OCM-Asym pairwise support; not pooled with the older 13-model benchmark |
| Observed-prefix policy analysis | complete | 400-run manifest; cutoff, unresolved denominator, conditional cost, and jointly-resolved pairing rules applied |
| Manuscript update | complete | local working manuscript and Supplement compiled and QA-checked; no final release identity assigned |
| Artifact freeze | pending | assemble source, figures, results, manifest, and provenance; assign identity only after package QA |

## Current execution snapshot

The canonical run manifest now contains 400 completed rows. The separate
P10 queue under `studies/protocol_replication_p10/` is a different evidence
family and is not used to fill or validate these cells.

No process PID is used as an execution identity. Long-term execution identity
uses the canonical run ID, configuration fingerprint, start/end timestamps,
code hash, manifest row, and artifact hashes. The first artifact timestamp
above came from the local filesystem and remains auxiliary evidence.

## Bound source identities at this snapshot

The following SHA-256 values bind the current protocol and execution sources.
They are source hashes, not the final submission identity:

| Object | SHA-256 |
|---|---|
| `docs/FULL_5X10_ANALYSIS_PLAN_FROZEN.md` | `02696177678D946625B4EDB62005877F80D0F433DB149F3BADF8F0E8629A02E9` |
| `docs/FULL_5X10_ANALYSIS_PLAN_AMENDMENT_01.md` | `745F4F40A014272CE062B37383009A257AB81B26BF642D9B06DDBD1AB1AF4411` |
| `scripts/run_full_5x10_crossing.py` | `76D9B649C273F9F7B2A5E8BB607A91B9F3E5E025058AB2D5001C406A490759AD` |
| `scripts/aggregate_full_5x10_crossing.py` | `B7CE52817D47DAAD1D69A1A0B7362D3534A2893DF5613645A3E3602E3F942AAE` |
| `scripts/run_dynamic_maintenance_policy.py` | `1AB69E24E837B29000FBFB1541E0514BFACCB4105AC58DA683C7F401CEBC1081` |
| `tests/test_core.py` | `98EDF80106BE61DCC3D14D0B77EC53A8A6AFC8825F87AE6F1E60F1F6D21AEBFF` |

## Canonical manifest lock

Audit recorded 2026-09-25 10:56:15 UTC. The 400-row canonical manifest
`paper_outputs/full_5x10_crossing/full_5x10_run_manifest.csv` has SHA-256
`BAE85A8CE7B463350C5751A7833EE5A68FFD7129CEAFDB03CF65D7C931789DAB`.
QA matched all rows to the frozen 400-job schedule in order, found no
duplicate cells, confirmed all statuses were `completed`, and validated the
per-run configuration fingerprint, expected 80-epoch plan, required artifact
presence, and checkpoint structure. No current-run outcome summary was used
to define these checks. This hash does not identify the manuscript or a
public release.

## Completed sequence and remaining release step

Completed locally:

`400/400 -> manifest QA -> final raw-manifest SHA-256 -> frozen aggregation ->
ranking integration -> observed-prefix policy analysis -> manuscript update`.

The canonical manifest remains locked. At the time this timeline snapshot was
prepared, public artifact assembly and a new immutable release identity were
still pending. The listed hashes identify local analysis outputs, not the
subsequently uploaded OSF archive or compiled PDFs.

## Local output hashes

These identify local analysis outputs only:

| Object | SHA-256 |
|---|---|
| `analysis/full_5x10_paired_contrasts.csv` | `724B02ACA59434FDCBB77CA904EA1674C54BB2029B7F173951EA06E8F4FA1A82` |
| `analysis/full_5x10_finite_design_summary.csv` | `29E617C82F4CEFD316DE2CDF7A9AA7085BFE878C17A30C9DA3E1BC61B7EF0C44` |
| `analysis/full_5x10_pairwise_rank_support.csv` | `34E69A569B737693D80A281B794EF33A995554ECF7CAFFB0EC436D40216E1D3E` |
| `dynamic_policy/dynamic_policy_protocol.json` | `C096213692781CACD997F88A1995AD372A2BC107DDB66CDB7CE4018296E5F743` |
| `dynamic_policy/dynamic_policy_paired_model_comparison.csv` | `DAB6B47718FE86B9AF6B6E584BA4A1AF4F1D1F0B624C9995D2F95D2BE4C5F1D9` |
| `dynamic_policy/dynamic_policy_summary.csv` | `AD201D0116ADBA6FC91342E74372AC150EC4A3ACF8FC0E4A98798EDF874197B0` |
| `dynamic_policy/dynamic_policy_engine_level.csv` | `B811F32CDA3E54A6CA42715479CD28E89ECE371F134B5326CF37E69A66711ED2` |

Local manuscript integration checks and PDF hashes are recorded in
`docs/FULL_5X10_MANUSCRIPT_INTEGRATION_QA_20260925.md`. Those are working-tree
identities from 2026-09-25; the evidence was uploaded to OSF after this snapshot
and is not identified by those PDF hashes.
