# Full 5 x 10 Analysis Plan

This file freezes the aggregation rules for the formal full crossing before
the completed run is summarized. It does not convert the retrospective design
into a preregistration.

## Design

- Tasks: FD001, FD002, FD003, and FD004.
- Split levels: 42, 123, 2024, 2025, and 2026.
- Training streams: 31415, 27182, 16180, 14142, 17320, 42, 123, 2024, 2025, and 2026.
- Configurations: `rast_gru` and `rast_gru_v2`.
- Canonical cells: 4 tasks x 5 split levels x 10 training streams x 2 configurations = 400.
- Contrast: `rast_gru_v2 - rast_gru` within the same task, split level, and
  training stream.

## Endpoint hierarchy

Primary endpoints are RMSE, NASA score per engine, and LPR@30. MAE and
SLPR@30,5/SLPR@30,10 are secondary endpoints. No additional endpoint family
is promoted to primary after inspecting the completed results.

## Finite-design summaries

For each task and primary endpoint, report the 50 paired contrasts using:

- the number of negative, zero, and positive contrasts;
- mean, median, minimum, and maximum;
- a declared candidate magnitude-band occupancy count, described as a finite
  design summary rather than a practical-significance claim;
- a 5 x 10 split-level by training-stream heatmap;
- split-level and training-stream marginal summaries.

The summaries do not estimate a population p-value, a variance component, a
universal ranking, or a generalization probability. The five split levels and
ten training streams are composite experimental levels, not exchangeable
replicates from a sampled population.

## Inclusion and quality rules

1. A canonical row must match exactly one `(task, split_level,
   training_stream, model)` identity.
2. Duplicate canonical identities are an error, not an averaging opportunity.
3. A completed row requires the expected checkpoint, metrics, configuration
   fingerprint, selection metric, and provenance fields.
4. Retries are retained as provenance but are not silently substituted into
   the canonical cell.
5. Non-finite metrics, missing checkpoints, wrong selection metrics, stale
   configurations, and incomplete epochs are quality failures.
6. Old 3 x 3, P10, smoke, or exploratory outputs cannot fill a formal 5 x 10
   cell.
7. Finite completed runs are retained in the descriptive summaries. Any
   exclusion used for a diagnostic sensitivity display must be explicit and
   cannot replace the primary finite-design table.

## Evidence boundary

The full crossing is a result-informed retrospective finite-design analysis.
Freezing this file prevents post-aggregation changes to endpoint priority,
contrast direction, inclusion rules, or summary definitions. It does not
claim a preregistered or confirmatory population inference.
