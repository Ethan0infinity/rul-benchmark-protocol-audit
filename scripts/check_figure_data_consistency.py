from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).absolute().parents[1]
OUTPUT = PROJECT_ROOT / "paper_outputs" / "advanced_evidence"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    manifest_path = OUTPUT / "figure_data_manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError("Run scripts/build_advanced_evidence.py to create figure_data_manifest.json.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    issues: list[str] = []
    for item in manifest.get("figures", []):
        figure = PROJECT_ROOT / item.get("path", f"paper_outputs/advanced_evidence/figures/{item['figure']}")
        if not figure.exists() or figure.stat().st_size < 1000:
            issues.append(f"missing/empty figure: {item['figure']}")
        elif sha256(figure) != item.get("figure_sha256"):
            issues.append(f"figure changed after manifest generation: {item['figure']}")
        for source in item.get("source_files", []):
            source_path = OUTPUT / source["name"]
            if not source_path.exists():
                issues.append(f"missing figure source: {source['name']}")
                continue
            if sha256(source_path) != source.get("sha256"):
                issues.append(f"source changed after figure generation: {source['name']}")
            actual_rows = len(pd.read_csv(source_path))
            if actual_rows != source.get("rows"):
                issues.append(f"row-count mismatch for {source['name']}: {actual_rows} != {source.get('rows')}")

    threshold = OUTPUT / "threshold_rank_stability_fd004.csv"
    if threshold.exists():
        frame = pd.read_csv(threshold)
        if not frame["rank"].between(1, 13).all() or not frame["ratio_mean"].between(0, 1).all():
            issues.append("threshold heatmap source contains out-of-range ranks or ratios")
    interval = OUTPUT / "interval_calibration_summary.csv"
    if interval.exists():
        frame = pd.read_csv(interval)
        if not frame["empirical_coverage_mean"].between(0, 1).all() or not (frame["mean_interval_width_mean"] >= 0).all():
            issues.append("interval figure source contains invalid coverage or width")
    mechanism = OUTPUT / "reliability_calibration_bins.csv"
    if mechanism.exists():
        frame = pd.read_csv(mechanism)
        calibration_columns = ("mean_predicted_unavailability", "observed_missing_fraction")
        missing_columns = set(calibration_columns) - set(frame.columns)
        if missing_columns:
            issues.append(f"mechanism calibration missing columns: {sorted(missing_columns)}")
        else:
            for column in calibration_columns:
                if not frame[column].between(0, 1).all():
                    issues.append(f"mechanism calibration contains invalid {column}")

    report = {"protocol_version": manifest.get("protocol_version"), "issues": issues}
    report_path = PROJECT_ROOT / "reports" / "figure_data_consistency.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if issues:
        print("FIGURE_DATA_CONSISTENCY_FAIL")
        for issue in issues:
            print(f"- {issue}")
        raise SystemExit(1)
    print(f"FIGURE_DATA_CONSISTENCY_PASS figures={len(manifest.get('figures', []))}")


if __name__ == "__main__":
    main()
