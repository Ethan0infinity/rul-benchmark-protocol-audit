from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rul.reporting import filter_metric_rows, collect_metrics, write_table_bundle


FORBIDDEN_TABLE_TOKENS = ["archive_", "smoke", "paper_main_seed42"]


def audit_generated_tables(output_dir: str | Path) -> list[str]:
    issues: list[str] = []
    root = Path(output_dir)
    if not root.exists():
        return issues
    for path in root.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {".csv", ".md", ".json", ".txt"}:
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        for token in FORBIDDEN_TABLE_TOKENS:
            if token.lower() in text.lower():
                issues.append(f"{path}: contains forbidden token {token!r}")
    return issues


def regenerate_tables_dry_run(
    rows: list[dict[str, Any]],
    *,
    output_dir: str | Path | None = None,
) -> tuple[dict[str, str], list[str]]:
    if output_dir is not None:
        out = Path(output_dir)
        outputs = write_table_bundle(rows, out)
        return outputs, audit_generated_tables(out)
    with tempfile.TemporaryDirectory() as tmp:
        outputs = write_table_bundle(rows, tmp)
        issues = audit_generated_tables(tmp)
    return outputs, issues


def main() -> None:
    parser = argparse.ArgumentParser(description="Regenerate and audit paper tables from metrics.json only.")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--experiment-name", action="append", dest="experiment_names", default=None)
    parser.add_argument("--experiment-prefix", default=None)
    parser.add_argument("--output-dir", default=None, help="Optional real output directory. Omit for temp dry-run.")
    parser.add_argument("--allow-empty", action="store_true")
    args = parser.parse_args()

    rows = filter_metric_rows(
        collect_metrics(args.results_dir),
        experiment_prefix=args.experiment_prefix,
        experiment_names=args.experiment_names,
    )
    if not rows and not args.allow_empty:
        raise FileNotFoundError("No metrics rows available for table audit")
    if not rows:
        print("TABLE_AUDIT_PASS empty result set")
        return
    outputs, issues = regenerate_tables_dry_run(rows, output_dir=args.output_dir)
    print(json.dumps(outputs, indent=2, ensure_ascii=False))
    if issues:
        print("[TABLE AUDIT FAIL]")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)
    print("TABLE_AUDIT_PASS")


if __name__ == "__main__":
    main()
