from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RELEASE = PROJECT_ROOT / "paper_outputs" / "release_v1"


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def validate_release(release: Path) -> list[str]:
    errors: list[str] = []
    required = {
        "primary_fixed_task_estimates.csv": 12,
        "primary_seed_level_effects.csv": 60,
        "fixed_transfer_aggregation_summary.csv": 12,
        "development_excluded_seed_effects.csv": 75,
        "preference_selection_stability.csv": 9,
        "preference_selection_leave_one_out.csv": 9,
        "preference_selection_oob_regret.csv": 10000,
        "endpoint_tolerance_sensitivity.csv": 80,
        "late_event_support.csv": 800,
        "small_cluster_calibration.csv": 16,
        "max_t_family_calibration.csv": 15,
        "multiplicity_registry.csv": 22,
        "normalized_perturbation_contrasts.csv": 54,
        "fd004_preference_development_search.csv": 9,
        "analysis_object_chronology.csv": 5,
        "no_overlap_augmentation_audit.csv": 30,
    }
    for name, expected_rows in required.items():
        path = release / name
        if not path.exists():
            errors.append(f"missing required artifact: {name}")
            continue
        rows = read_csv(path)
        if len(rows) != expected_rows:
            errors.append(f"{name}: expected {expected_rows} rows, found {len(rows)}")

    manifest_path = release / "release_manifest.csv"
    if not manifest_path.exists():
        errors.append("missing release_manifest.csv")
        return errors
    manifest = read_csv(manifest_path)
    if len(manifest) < 36:
        errors.append(f"manifest contains only {len(manifest)} artifacts; expected at least 36")
    required_manifest_fields = {
        "artifact_id",
        "release_path",
        "schema_version",
        "family_id",
        "generator",
        "generator_sha256",
        "primary_key",
        "release_sha256",
    }
    if manifest and not required_manifest_fields.issubset(manifest[0]):
        errors.append(
            "release manifest is missing fields: "
            f"{sorted(required_manifest_fields - set(manifest[0]))}"
        )
    for row in manifest:
        release_path = row["release_path"]
        if "round2" in release_path.lower() or "round3" in release_path.lower():
            errors.append(f"revision-round name leaked into semantic release: {release_path}")
        target = release / Path(release_path)
        if not target.exists():
            errors.append(f"manifest target missing: {release_path}")
            continue
        if sha256_file(target) != row["release_sha256"]:
            errors.append(f"release hash mismatch: {release_path}")
        if len(row.get("generator_sha256", "")) != 64:
            errors.append(f"invalid generator hash: {release_path}")
        primary_key = row.get("primary_key", "")
        if target.suffix.lower() == ".csv" and primary_key not in {"", "not applicable"}:
            rows = read_csv(target)
            key_fields = primary_key.split("|")
            if rows and not set(key_fields).issubset(rows[0]):
                errors.append(
                    f"{release_path}: declared primary key fields are missing: "
                    f"{sorted(set(key_fields) - set(rows[0]))}"
                )
            else:
                observed = [tuple(item[field] for field in key_fields) for item in rows]
                if len(observed) != len(set(observed)):
                    errors.append(f"{release_path}: declared primary key is not unique")

    schema_path = release / "metric_schema.json"
    if not schema_path.exists():
        errors.append("missing metric_schema.json")
    else:
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        metric_ids = {row["metric_id"] for row in schema.get("metrics", [])}
        expected_metrics = {
            "LPR@tau,epsilon",
            "ZIMLE@tau,epsilon",
            "CMLE@tau,epsilon",
            "late_CVaR95",
        }
        if metric_ids != expected_metrics:
            errors.append("metric_schema.json does not contain the required metric contracts")

    for path in release.rglob("*"):
        if path.is_file() and path.suffix.lower() in {".csv", ".json", ".md", ".tex"}:
            text = path.read_text(encoding="utf-8-sig", errors="replace")
            if "PRIMARY LOCKED" in text or "POST-LOCK" in text:
                errors.append(f"obsolete evidence terminology remains in {path.relative_to(release)}")

    registry_path = release / "multiplicity_registry.csv"
    if registry_path.exists():
        registry = read_csv(registry_path)
        family_ids = {row["family_id"] for row in registry}
        expected_ids = {
            "E-FIXED-12",
            "S-ENDPOINT-80",
            "S-PERTURB-54",
            "S-NORM-9",
            "S-FD002-20",
            "D-DUALMIXER-A12",
            "D-DUALMIXER-B12",
            "D-ENDPOINT-DESIGN",
            "D-CHECKPOINT",
            "D-ABLATION",
            "D-AUGDIST",
            "D-SEEDGRID-3X3",
            "D-NCMAPSS-ROT",
            "D-STREAM10-FIXED",
            "D-PREF-REFIT-3X3",
            "D-SHARED-FACT-8",
            "D-LOFO-4",
            "D-NCMAPSS-SW9",
            "D-NCMAPSS-H3",
            "D-CROSSED-3X3",
            "D-CMAPSS-PREP5",
            "D-LATE-SUPPORT",
        }
        if family_ids != expected_ids:
            errors.append(
                "multiplicity family mismatch: "
                f"missing={sorted(expected_ids - family_ids)} extra={sorted(family_ids - expected_ids)}"
            )
        for row in registry:
            role = row["decision_role"].lower()
            if "confirmatory" in role and "cannot" not in role and "no " not in role:
                errors.append(f"confirmatory role remains in {row['family_id']}: {row['decision_role']}")

    calibration_path = release / "max_t_family_calibration.csv"
    if calibration_path.exists():
        calibration = read_csv(calibration_path)
        family_rows = [
            row
            for row in calibration
            if row["estimand"] == "family-wise null coverage over all 12 contrasts"
        ]
        if len(family_rows) != 1:
            errors.append("max-t calibration must contain exactly one family-wise coverage row")
        elif family_rows[0]["interpretation"] != "calibration diagnostic, not confirmatory validation":
            errors.append("max-t family coverage is not labeled as diagnostic")

    metadata_path = release / "release_metadata.json"
    if not metadata_path.exists():
        errors.append("missing release_metadata.json")
    else:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("source_of_truth") != "paper_outputs/release_v1":
            errors.append("release_metadata source_of_truth is not paper_outputs/release_v1")
        archive = metadata.get("external_archive", {})
        if any(archive.get(key) for key in ("public_url", "immutable_release", "doi")):
            errors.append("external archive identifier must not be populated without author deposit")

    chronology_path = release / "analysis_object_chronology.csv"
    if chronology_path.exists():
        for row in read_csv(chronology_path):
            if row["external_timestamp"].lower() != "none":
                errors.append(
                    f"chronology unexpectedly claims external timestamp: {row['analysis_object']}"
                )
    return errors


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the semantic manuscript evidence release.")
    parser.add_argument("--release-dir", default=str(DEFAULT_RELEASE))
    args = parser.parse_args()
    release = Path(args.release_dir)
    errors = validate_release(release)
    if errors:
        print("RELEASE_EVIDENCE_FAIL")
        for error in errors:
            print(f"  - {error}")
        raise SystemExit(1)
    print(f"RELEASE_EVIDENCE_PASS release={release}")


if __name__ == "__main__":
    main()
