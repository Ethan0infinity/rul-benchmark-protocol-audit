from __future__ import annotations

import os
import re
import subprocess
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = REPO_ROOT / "submission_contract"
DEFAULT_MANUSCRIPT = (
    REPO_ROOT / "manuscript"
    if (REPO_ROOT / "manuscript").is_dir()
    else REPO_ROOT.parent / "manuscript"
)
SUBMISSION_ROOT = Path(os.environ.get("SUBMISSION_ROOT", DEFAULT_CONTRACT if DEFAULT_CONTRACT.is_dir() else DEFAULT_MANUSCRIPT))
if (SUBMISSION_ROOT / "latex_source").is_dir():
    LATEX_ROOT = SUBMISSION_ROOT / "latex_source"
    MAIN = LATEX_ROOT / "main_revised_RESS.tex"
    SUPPLEMENT = LATEX_ROOT / "supplementary" / "supplement_en_RESS.tex"
    GENERATED = LATEX_ROOT / "supplementary" / "generated"
else:
    LATEX_ROOT = SUBMISSION_ROOT
    MAIN = LATEX_ROOT / "main_revised_RESS.tex"
    SUPPLEMENT = LATEX_ROOT / "supplement_en_RESS.tex"
    GENERATED = LATEX_ROOT / "generated"


def abstract_plain_text() -> str:
    text = MAIN.read_text(encoding="utf-8")
    match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", text, re.S)
    assert match, "abstract environment missing"
    result = subprocess.run(
        ["detex"], input=match.group(1), text=True, capture_output=True, check=True
    )
    return re.sub(r"\s+", " ", result.stdout).strip()


def test_abstract_is_at_most_200_words() -> None:
    words = abstract_plain_text().split()
    assert len(words) <= 200, len(words)


def test_authoritative_latency_and_release_strings() -> None:
    corpus = "\n".join(
        [MAIN.read_text(encoding="utf-8"), SUPPLEMENT.read_text(encoding="utf-8")]
    )
    assert "2026-08-09-r21" not in corpus
    assert "prepared tag \\texttt{v5.2.0}" not in corpus
    assert "authoritative median is 1.612 ms, not 1.167 ms" in corpus


def test_six_part_traceability_scaffold_has_six_operational_fields_in_supplement() -> None:
    main = MAIN.read_text(encoding="utf-8")
    text = SUPPLEMENT.read_text(encoding="utf-8")
    assert re.search(r"six-part traceability scaffold", main, re.I)
    table_start = text.index(r"\label{tab:supp-sixpart}")
    table_end = text.index(r"\end{table}", table_start)
    table = text[table_start:table_end]
    expected_fields = [
        "Comparison and unit",
        "Design timing",
        "Evidence support",
        "Artifacts",
        "Completed sensitivities",
        "Claim wording",
    ]
    for field in expected_fields:
        assert field in table, field


def test_selected_grid_direction_counts_match_dashboard_bookkeeping() -> None:
    text = MAIN.read_text(encoding="utf-8")
    expected = {
        "FD001": "4/4/1",
        "FD002": "5/4/0",
        "FD003": "4/3/2",
        "FD004": "7/2/0",
    }
    for task, counts in expected.items():
        assert f"{task} &" in text
        table_region = text[text.index("\\label{tab:lpr-directions}") :]
        assert f"{task} &" in table_region and counts in table_region.split(f"{task} &", 1)[1].split("\\\\", 1)[0]


def test_supplement_battery_tables_are_fixed_in_reading_order() -> None:
    for name in (
        "table_battery_unit_support.tex",
        "table_battery_exploratory_baselines.tex",
        "table_battery_index_sensitivity.tex",
    ):
        text = (GENERATED / name).read_text(encoding="utf-8")
        assert "\\begin{table}[H]" in text, name
        assert "\\begin{table*}" not in text, name


def test_public_submission_source_excludes_confidential_contact_details() -> None:
    corpus = "\n".join(
        [MAIN.read_text(encoding="utf-8"), SUPPLEMENT.read_text(encoding="utf-8")]
    )
    assert "Tel.:" not in corpus
    assert re.search(r"\+86\s+\d", corpus) is None
    assert re.search(r"\bRoom\s+\d+", corpus, re.I) is None
    assert re.search(r"\bBuilding\s+\d+", corpus, re.I) is None
    assert re.search(r"\bUnit\s+\d+", corpus, re.I) is None
    main_text = MAIN.read_text(encoding="utf-8")
    assert r"\author[aff1]{Zhihao Liu}" in main_text
    assert r"\address[aff1]{Independent Researcher, China}" in main_text
    assert not re.search(r"\bORCID\b|[\w.+-]+@[\w.-]+\.[A-Za-z]{2,}|\b\d{6}\b", corpus, re.I)


def test_submission_contract_defaults_to_packaged_snapshot() -> None:
    assert MAIN.is_file()
    assert SUPPLEMENT.is_file()
