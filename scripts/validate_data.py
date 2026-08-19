from __future__ import annotations

import argparse
import hashlib
import json
import sys
import zipfile
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rul.cmapss import COLUMNS


EXPECTED = {
    "FD001": {"train_rows": 20631, "test_rows": 13096, "train_units": 100, "test_units": 100, "rul_rows": 100},
    "FD002": {"train_rows": 53759, "test_rows": 33991, "train_units": 260, "test_units": 259, "rul_rows": 259},
    "FD003": {"train_rows": 24720, "test_rows": 16596, "train_units": 100, "test_units": 100, "rul_rows": 100},
    "FD004": {"train_rows": 61249, "test_rows": 41214, "train_units": 249, "test_units": 248, "rul_rows": 248},
}


@dataclass
class CheckResult:
    name: str
    status: str
    detail: str


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_table(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS, engine="python")


def read_rul(path: Path) -> pd.Series:
    return pd.read_csv(path, sep=r"\s+", header=None, engine="python").iloc[:, 0].astype(float)


def check(condition: bool, name: str, ok_detail: str, fail_detail: str, results: list[CheckResult]) -> None:
    results.append(CheckResult(name=name, status="PASS" if condition else "FAIL", detail=ok_detail if condition else fail_detail))


def cycle_issues(df: pd.DataFrame) -> list[str]:
    issues = []
    for unit_id, group in df.groupby("unit_id"):
        cycles = group["cycle"].to_numpy()
        if len(cycles) == 0:
            issues.append(f"unit {unit_id}: empty")
            continue
        if cycles[0] != 1:
            issues.append(f"unit {unit_id}: first cycle {cycles[0]} != 1")
        if not np.all(np.diff(cycles) == 1):
            issues.append(f"unit {unit_id}: cycles not consecutive")
    return issues


def numeric_profile(df: pd.DataFrame) -> dict[str, dict[str, float]]:
    cols = [c for c in df.columns if c not in {"unit_id", "cycle"}]
    desc: dict[str, dict[str, float]] = {}
    for col in cols:
        values = df[col].astype(float)
        desc[col] = {
            "min": float(values.min()),
            "max": float(values.max()),
            "mean": float(values.mean()),
            "std": float(values.std()),
        }
    return desc


def validate_subset(data_dir: Path, subset: str, results: list[CheckResult]) -> dict[str, Any]:
    expected = EXPECTED[subset]
    train_path = data_dir / f"train_{subset}.txt"
    test_path = data_dir / f"test_{subset}.txt"
    rul_path = data_dir / f"RUL_{subset}.txt"
    train = read_table(train_path)
    test = read_table(test_path)
    rul = read_rul(rul_path)

    profile: dict[str, Any] = {"subset": subset}
    profile["train_rows"] = int(len(train))
    profile["test_rows"] = int(len(test))
    profile["train_units"] = int(train["unit_id"].nunique())
    profile["test_units"] = int(test["unit_id"].nunique())
    profile["rul_rows"] = int(len(rul))
    profile["train_cols"] = int(train.shape[1])
    profile["test_cols"] = int(test.shape[1])
    profile["train_cycle_min"] = int(train.groupby("unit_id")["cycle"].max().min())
    profile["train_cycle_max"] = int(train.groupby("unit_id")["cycle"].max().max())
    profile["test_cycle_min"] = int(test.groupby("unit_id")["cycle"].max().min())
    profile["test_cycle_max"] = int(test.groupby("unit_id")["cycle"].max().max())
    profile["rul_min"] = float(rul.min())
    profile["rul_max"] = float(rul.max())

    for key in ["train_rows", "test_rows", "train_units", "test_units", "rul_rows"]:
        check(
            profile[key] == expected[key],
            f"{subset} {key}",
            f"{profile[key]} matches expected",
            f"{profile[key]} != expected {expected[key]}",
            results,
        )

    check(train.shape[1] == 26 and test.shape[1] == 26, f"{subset} column count", "train/test have 26 columns", f"train={train.shape[1]}, test={test.shape[1]}", results)
    check(not train.isna().any().any(), f"{subset} train missing values", "no missing values", "missing values found", results)
    check(not test.isna().any().any(), f"{subset} test missing values", "no missing values", "missing values found", results)
    check(np.isfinite(train.to_numpy(dtype=float)).all(), f"{subset} train finite numeric", "all values finite", "non-finite value found", results)
    check(np.isfinite(test.to_numpy(dtype=float)).all(), f"{subset} test finite numeric", "all values finite", "non-finite value found", results)
    check(train.duplicated().sum() == 0, f"{subset} train exact duplicates", "0 duplicate rows", f"{int(train.duplicated().sum())} duplicate rows", results)
    check(test.duplicated().sum() == 0, f"{subset} test exact duplicates", "0 duplicate rows", f"{int(test.duplicated().sum())} duplicate rows", results)

    train_units = sorted(train["unit_id"].unique().astype(int).tolist())
    test_units = sorted(test["unit_id"].unique().astype(int).tolist())
    check(train_units == list(range(1, expected["train_units"] + 1)), f"{subset} train unit ids", "unit ids contiguous from 1", "unit ids are not contiguous", results)
    check(test_units == list(range(1, expected["test_units"] + 1)), f"{subset} test unit ids", "unit ids contiguous from 1", "unit ids are not contiguous", results)

    train_cycle_issues = cycle_issues(train)
    test_cycle_issues = cycle_issues(test)
    check(not train_cycle_issues, f"{subset} train cycle continuity", "all unit cycles start at 1 and increment by 1", "; ".join(train_cycle_issues[:5]), results)
    check(not test_cycle_issues, f"{subset} test cycle continuity", "all unit cycles start at 1 and increment by 1", "; ".join(test_cycle_issues[:5]), results)
    check((rul >= 0).all(), f"{subset} RUL non-negative", "all test RUL values are non-negative", "negative RUL value found", results)

    max_train_cycle = train.groupby("unit_id")["cycle"].max()
    derived_train_rul = max_train_cycle.reindex(train["unit_id"]).to_numpy() - train["cycle"].to_numpy()
    check(derived_train_rul.min() == 0, f"{subset} derived train RUL", "each training trajectory reaches failure RUL=0", f"min derived RUL={derived_train_rul.min()}", results)

    profile["numeric_profile_train"] = numeric_profile(train)
    profile["numeric_profile_test"] = numeric_profile(test)
    return profile


def write_markdown(results: list[CheckResult], profiles: list[dict[str, Any]], checksums: dict[str, str], output: Path) -> None:
    fails = [r for r in results if r.status != "PASS"]
    lines = [
        "# C-MAPSS Data Validation Report",
        "",
        f"Overall status: {'PASS' if not fails else 'FAIL'}",
        f"Checks: {len(results)} total, {len(fails)} failed",
        "",
        "## Important Note",
        "",
        "The local files contain FD004 with 249 training units and 248 test units. The NASA readme bundled with some downloads lists FD004 as 248 training and 249 test trajectories, but the actual file row counts, unit ids, and RUL vector align with 249 train / 248 test. This report validates the actual canonical files used by the code.",
        "",
        "## Dataset Summary",
        "",
        "| subset | train rows | train units | test rows | test units | RUL rows | train cycle min/max | test cycle min/max | RUL min/max |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for p in profiles:
        lines.append(
            f"| {p['subset']} | {p['train_rows']} | {p['train_units']} | {p['test_rows']} | {p['test_units']} | {p['rul_rows']} | "
            f"{p['train_cycle_min']}/{p['train_cycle_max']} | {p['test_cycle_min']}/{p['test_cycle_max']} | {p['rul_min']:.0f}/{p['rul_max']:.0f} |"
        )
    lines.extend(["", "## Check Results", "", "| status | check | detail |", "| --- | --- | --- |"])
    for r in results:
        lines.append(f"| {r.status} | {r.name} | {r.detail.replace('|', '/')} |")
    lines.extend(["", "## SHA256 Checksums", ""])
    for path, digest in checksums.items():
        lines.append(f"- `{digest}`  `{path}`")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate local NASA C-MAPSS data files.")
    parser.add_argument("--data-dir", default=str(PROJECT_ROOT / "data" / "raw"))
    parser.add_argument("--reports-dir", default=str(PROJECT_ROOT / "reports"))
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    results: list[CheckResult] = []
    required = ["readme.txt", "CMAPSSData.zip"]
    for subset in EXPECTED:
        required.extend([f"train_{subset}.txt", f"test_{subset}.txt", f"RUL_{subset}.txt"])

    for name in required:
        path = data_dir / name
        check(path.exists(), f"required file {name}", "file exists", f"missing {path}", results)

    zip_path = data_dir / "CMAPSSData.zip"
    if zip_path.exists():
        try:
            with zipfile.ZipFile(zip_path) as zf:
                bad = zf.testzip()
                names = set(Path(n).name for n in zf.namelist())
            check(bad is None, "CMAPSSData.zip integrity", "zipfile.testzip returned no corrupt member", f"corrupt member: {bad}", results)
            missing_in_zip = sorted(set(required) - {"CMAPSSData.zip"} - names)
            check(not missing_in_zip, "CMAPSSData.zip required members", "all required members present in zip", f"missing in zip: {missing_in_zip}", results)
        except Exception as exc:
            results.append(CheckResult("CMAPSSData.zip readable", "FAIL", str(exc)))

    checksums: dict[str, str] = {}
    for name in required:
        path = data_dir / name
        if path.exists():
            checksums[name] = sha256_file(path)
    (reports_dir / "raw_file_checksums.sha256").write_text(
        "\n".join(f"{digest}  {name}" for name, digest in checksums.items()) + "\n",
        encoding="utf-8",
    )

    profiles = []
    if all((data_dir / f"train_{subset}.txt").exists() and (data_dir / f"test_{subset}.txt").exists() and (data_dir / f"RUL_{subset}.txt").exists() for subset in EXPECTED):
        for subset in EXPECTED:
            profiles.append(validate_subset(data_dir, subset, results))

    pd.DataFrame([asdict(r) for r in results]).to_csv(reports_dir / "data_validation_checks.csv", index=False)
    pd.DataFrame([{k: v for k, v in p.items() if not isinstance(v, dict)} for p in profiles]).to_csv(reports_dir / "dataset_summary.csv", index=False)
    (reports_dir / "dataset_profile.json").write_text(json.dumps(profiles, indent=2), encoding="utf-8")
    write_markdown(results, profiles, checksums, reports_dir / "data_validation_report.md")

    fails = [r for r in results if r.status != "PASS"]
    print(f"Data validation checks: {len(results)} total, {len(fails)} failed")
    print(f"Report: {reports_dir / 'data_validation_report.md'}")
    if fails:
        for fail in fails:
            print(f"FAIL {fail.name}: {fail.detail}")
        raise SystemExit(1)
    print("DATA_VALIDATION_PASS")


if __name__ == "__main__":
    main()

