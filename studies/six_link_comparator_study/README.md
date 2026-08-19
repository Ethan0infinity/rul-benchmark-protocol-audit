# Independent Comparator Study Package

This directory is a study-ready package for a future independent comparison
between the six-link worked reporting template and a compact baseline
checklist. It contains no participant responses and therefore supplies no
evidence that the six-link template improves reviewer agreement, defect
detection, completion time, reproduction success, or decision quality.

The current manuscript may cite this directory only as an openly specified
future validation protocol. It must not call the comparator study completed
until independent eligible raters have been recruited, any required ethics or
institutional determination has been obtained, the assignment has been frozen
before responses are opened, and the prespecified analysis has been executed.

Run the package self-check with:

python scripts/prepare_comparator_study.py --check-only

The study deliberately separates structural comparison from empirical
effectiveness. framework_construct_map.csv is a literature-based construct
map, not a performance result. COMPARATOR_STUDY_PROTOCOL.md defines the future
independent evaluation.
