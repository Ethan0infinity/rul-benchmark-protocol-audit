# Full 5 x 10 Analysis Plan Amendment 01

Date: 2026-09-22

This amendment supplements, and does not overwrite,
`FULL_5X10_ANALYSIS_PLAN_FROZEN.md`. The original plan remains the frozen
source document; this file records clarifications made before the complete
400-cell aggregate is unlocked.

## Candidate magnitude-band provenance

The candidate magnitude bands were introduced in the earlier retrospective
C-MAPSS analysis. Their reuse here is frozen before aggregation of the full
5 x 10 results, but the bands are not prospective SESOIs and are not claimed
to represent stakeholder-validated practical significance.

## Retry and failure taxonomy

- An infrastructure interruption may be retried with the exact same resolved
  configuration. A successful retry may fill the canonical cell only when its
  retry lineage, configuration fingerprint, and replacement event are retained
  in the manifest.
- A numerical or model failure is itself an observed run outcome. NaN/Inf,
  divergence, invalid checkpoint state, or an invalid training trace may not
  be replaced by changing the seed, learning rate, batch size, or any other
  training choice and then silently using the successful run.
- Failed canonical cells remain visible in the audit. They are not silently
  converted into completed evidence.

## Valid early stopping

`planned_epochs=80` does not imply that every valid run must contain 80
epochs. A run is complete when it either reaches the declared maximum or
reaches the declared patience-based early-stopping condition, and it contains
the stop reason, complete history, best checkpoint, selection metric, and
configuration fingerprint. A process interruption, truncated history without
a valid stop marker, non-finite training state, missing checkpoint, wrong
selection metric, or stale configuration is a quality failure.

## Dynamic observed-prefix denominator rule

The dynamic policy analysis is separate from the 5 x 10 endpoint aggregate.
For every policy cell:

1. Unresolved engines remain in the eligible denominator for the
   `resolution_rate` and `unresolved_rate` endpoints.
2. Model-based cost is reported as
   `mean_cost_among_resolved`, conditional on an in-prefix trigger.
3. Direct OCM-versus-RAST cost contrasts use the jointly resolved engine set
   and report its integer support.
4. Reference-only, comparison-only, both-resolved, and neither-resolved
   counts accompany every paired comparison.
5. An action scheduled after the observation cutoff is allowed when its trigger
   was fixed before the cutoff; no trigger after the cutoff may be inferred.

These rules are descriptive finite-design rules. They do not create a
run-to-failure prospective validation from the truncated official test
trajectories.
