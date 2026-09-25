# Full 5 x 10 Canonical Manifest QA

Audit time: 2026-09-25 10:56:15 UTC; downstream checks completed 2026-09-25 Asia/Shanghai
Scope: `paper_outputs/full_5x10_crossing/full_5x10_run_manifest.csv` only.

## Result

- Planned and observed rows: 400 / 400.
- Unique canonical cells: 400 / 400.
- Task counts: FD001 100, FD002 100, FD003 100, FD004 100.
- Split levels: 42, 123, 2024, 2025, 2026.
- Training streams: 31415, 27182, 16180, 14142, 17320, 42, 123, 2024, 2025, 2026.
- Configuration counts: RAST 200, Asym 200.
- Manifest statuses: `completed` 400.
- Per-row configuration fingerprints match the frozen launcher schedule.
- Per-row run directory and experiment identity match the frozen launcher schedule.
- Each run passes the launcher's completion check for required files, planned epoch count, and checkpoint structure.
- All 400 runs use `val_last_risk_score` for checkpoint selection and contain finite metrics, histories, and test predictions.
- All 400 runs reached the declared patience-based early-stop condition with `patience=12`; stop epochs range from 13 to 61 under the 80-epoch cap. Histories are contiguous, the best-epoch identity matches the selected validation score, and each terminal history contains the required patience window.
- Metric values were recomputed from each run's stored test predictions by the frozen aggregation script; task-level test-engine identities agree across matched configurations.
- Duplicate, missing, extra, or reordered schedule identities: none.
- Scientific metric direction or magnitude was not used for inclusion decisions. Three earlier infrastructure/numerical attempts remain separately recorded; only an exact-configuration infrastructure retry may fill a canonical cell, with lineage retained. No failed run was silently replaced by changing its training configuration.

## Identity

Manifest SHA-256:

`BAE85A8CE7B463350C5751A7833EE5A68FFD7129CEAFDB03CF65D7C931789DAB`

The frozen analysis plan SHA-256 is
`02696177678D946625B4EDB62005877F80D0F433DB149F3BADF8F0E8629A02E9`.
Amendment 01 SHA-256 is
`745F4F40A014272CE062B37383009A257AB81B26BF642D9B06DDBD1AB1AF4411`.
These hashes bind the analysis input and source protocol, not a public release.

## Downstream analysis checks

- The aggregation script passed manifest, source-hash, configuration, early-stop, finite-output, prediction-metric-recomputation, and 200-pair completeness checks. It generated 400 run-metric rows, 200 matched contrasts, 24 task-endpoint finite-design summaries, pairwise rank-support rows, candidate-band summaries, factor marginals, and 12 heatmaps.
- The rank-support file compares only RAST-GRU with OCM-Asym inside this 5-by-10 family. It is not merged with the older 13-model/260-run ranking.
- The observed-prefix policy run completed from the same 400-row manifest. It generated 86,400 model/policy summaries, 21,600 paired model comparisons, and 15,271,200 engine-level rows across the declared grid.
- Policy QA checked that resolution and unresolved rates sum to one, resolved counts agree with rates, paired discordance counts sum to the eligible denominator, and paired support equals the jointly resolved count. Unresolved engines remain in the resolution denominator; conditional cost uses resolved engines; paired cost uses jointly resolved engines.
- `pytest tests/test_core.py -k dynamic_policy_pairing_reports_resolution_discordance -q`: 1 passed.
- These checks establish structural and arithmetic validity for the declared finite outputs. They do not establish calibrated population inference, an optimized policy, or field maintenance benefit.

## Analysis identities

- Aggregation script SHA-256: `B7CE52817D47DAAD1D69A1A0B7362D3534A2893DF5613645A3E3602E3F942AAE`.
- Aggregation provenance: `paper_outputs/full_5x10_crossing/analysis/aggregation_provenance.json`.
- Dynamic policy script SHA-256: `1AB69E24E837B29000FBFB1541E0514BFACCB4105AC58DA683C7F401CEBC1081`.
- Dynamic policy protocol: `paper_outputs/full_5x10_crossing/dynamic_policy/dynamic_policy_protocol.json`.
- At the time this QA report was written on 2026-09-25, aggregation and policy CSVs, figures, and prediction trajectories remained local. They were later included in the OSF file identified in `docs/OSF_PUBLIC_UPLOAD_VERIFICATION.md`; this report's hashes remain analysis-input/output checks, not the archive hash.

## Separate evidence families

The earlier P10 queue under `studies/protocol_replication_p10/` is separate.
Its 400 C-MAPSS plus battery jobs are not part of this manifest and were not
used to fill any canonical cell above.

## Next steps

Manifest QA, raw-manifest hashing, frozen endpoint aggregation, pairwise rank
integration, and observed-prefix policy analysis are complete. Manuscript
integration and a new public artifact freeze remain separate tasks. Preserve
the canonical manifest unchanged.
