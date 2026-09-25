# Protocol Sensitivity in RUL Model Comparisons

Source code and publication files for **Protocol Sensitivity in Remaining-Useful-Life Model Comparisons: Retrospective C-MAPSS Analysis and a Pre-Outcome-Frozen Battery Study**.

The paper reports two distinct studies. The C-MAPSS analysis is retrospective: the 5-by-10 design and analysis rules were locked after training had begun but before complete-result aggregation. The battery protocol was externally frozen before training and test-label evaluation. Neither study establishes a universal model winner, field maintenance benefit, or optimized maintenance policy.

## Evidence archive

The complete v5.5.0 evidence archive is available from [OSF](https://osf.io/download/6ab6a6735205d2cb18154961/). Its SHA-256 is:

```text
A390CA0216E9CA1238420CA3F76AAFDF536A18B07D1B0FF2631FAA7FDF50566B
```

The archive contains the 400-cell C-MAPSS run manifest and outputs, observed-prefix analyses, battery artifacts, manifests, and provenance records. It does not contain the third-party NASA source datasets. The OSF project DOI identifies the project record; the version-named ZIP does not have a separate version DOI.

## Reproduction

Python 3.11 is the supported environment. Install dependencies from `environment.yml` or `requirements.txt`. Obtain the NASA datasets from the official sources and verify them using `data/README.md` and the manuscript Supplement.

Run data and software checks:

```powershell
python scripts/validate_data.py
python scripts/run_core_tests.py
```

The full 400-run training route is computationally expensive. The stored-results route is described in `docs/FULL_5X10_EVIDENCE_TIMELINE.md` and the evidence archive README. The code supports reproducing the declared aggregation from stored results; the complete observed-prefix policy replay did not finish within the local 10-minute audit window and is not claimed as independently replayed here.

To compile the manuscript from `manuscript/` with TeX Live:

```powershell
latexmk -xelatex -interaction=nonstopmode -halt-on-error main_revised_RESS.tex
latexmk -xelatex -interaction=nonstopmode -halt-on-error supplement_en_RESS.tex
```

## Repository map

- `src/rul/`: data processing, models, training, metrics, and reporting.
- `configs/`: model and evaluation settings.
- `scripts/`: experiment, aggregation, policy-sensitivity, and verification scripts.
- `tests/`: 133 core tests and seven submission-text contract tests.
- `paper_outputs/release_v1/`: compact earlier benchmark exhibits and manifest.
- `paper_outputs/ress_major_revision/`: maintenance-decision, equivalence, battery, and timeline outputs.
- `manuscript/`: main paper, Supplement, bibliography, figures, generated tables, and compiled PDFs.
- `studies/six_link_comparator_study/`: unexecuted protocol and empty response schemas; no participant responses are included.

## Claims and limits

The primary C-MAPSS results are finite-design comparisons. The static and observed-prefix analyses are decision-sensitivity calculations, not field maintenance-policy evaluations. Normalized-coordinate perturbations do not establish physical fault robustness. The battery study has four held-out batteries. See the manuscript and Supplement for endpoint definitions, analysis timing, event support, and limitations.

## Citation and license

See `CITATION.cff` for citation metadata. Code is distributed under the MIT License; third-party notices are in `THIRD_PARTY_NOTICES.md`.
