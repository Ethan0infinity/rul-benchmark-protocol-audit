from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
STUDY = ROOT / "studies" / "six_link_comparator_study"


def check() -> None:
    required = {
        "README.md",
        "COMPARATOR_STUDY_PROTOCOL.md",
        "framework_construct_map.csv",
        "response_template.csv",
        "assignment_template.csv",
    }
    missing = sorted(name for name in required if not (STUDY / name).is_file())
    if missing:
        raise FileNotFoundError(f"Missing comparator-study files: {missing}")
    construct = pd.read_csv(STUDY / "framework_construct_map.csv")
    expected = {
        "framework",
        "primary_object",
        "claim_estimand_unit",
        "design_timing",
        "evidence_role",
        "artifact_provenance",
        "finite_choice_behavior",
        "wording_constraint",
        "empirical_effectiveness_evidence",
        "current_study_interpretation",
    }
    if set(construct.columns) != expected:
        raise ValueError("Construct-map schema differs from the study-ready schema")
    responses = pd.read_csv(STUDY / "response_template.csv")
    if "rater_id" not in responses or "arm" not in responses:
        raise ValueError("Response template lacks rater or arm fields")
    if len(responses):
        raise ValueError("Response template unexpectedly contains participant data")
    print("COMPARATOR_STUDY_READY_NOT_EXECUTED")
    print("Independent rater responses: 0")
    print("Permitted current-paper wording: worked reporting template; comparative efficacy untested")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the independent comparator-study package.")
    parser.add_argument("--check-only", action="store_true", help="Validate files without generating assignments.")
    args = parser.parse_args()
    if not args.check_only:
        raise SystemExit(
            "Participant assignment requires a frozen rater roster and any required ethics determination. "
            "Use --check-only for the current unexecuted package."
        )
    check()


if __name__ == "__main__":
    main()
