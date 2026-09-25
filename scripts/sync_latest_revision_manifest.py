from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = PROJECT_ROOT / "paper_outputs" / "advanced_evidence"

CSV_FILES = (
    "core_asym_rast_subset_bootstrap.csv",
    "validation_endpoint_cluster_audit.csv",
    "endpoint_selection_confirmation_bootstrap.csv",
    "stress_operating_points_engine_bootstrap.csv",
    "augmentation_distribution_summary.csv",
    "augmentation_distribution_paired_intervals.csv",
    "tcn_gru_seed_audit.csv",
    "tcn_gru_subset_robust_summary.csv",
    "fd002_protocol_transfer_seed_metrics.csv",
    "fd002_protocol_transfer_summary.csv",
    "fd002_protocol_transfer_paired_bootstrap.csv",
)
FIGURE_FILES = (
    "figures/fig_cross_backbone_protocol_buildup.pdf",
    "figures/fig_tcn_gru_seed_instability.pdf",
    "figures/fig_augmentation_distribution_sensitivity.pdf",
    "figures/fig_validation_endpoint_distribution.pdf",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    manifest_path = EVIDENCE / "advanced_evidence_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    artifact_rows = {}
    artifact_hashes = {}
    for filename in CSV_FILES:
        path = EVIDENCE / filename
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        artifact_rows[filename] = int(len(pd.read_csv(path)))
        artifact_hashes[filename] = sha256(path)
    for filename in FIGURE_FILES:
        path = EVIDENCE / filename
        if not path.exists() or path.stat().st_size == 0:
            raise FileNotFoundError(path)
        artifact_hashes[filename] = sha256(path)
    manifest["latest_revision_synced_at_utc"] = datetime.now(timezone.utc).isoformat()
    manifest["latest_revision_artifact_rows"] = artifact_rows
    manifest["latest_revision_sha256"] = artifact_hashes
    manifest["model_display_name"] = "OCM-MST-GRU"
    manifest["historical_implementation_key"] = "rast_gru_v2"
    manifest["endpoint_inference_unit"] = "matched official test engine endpoint"
    manifest["validation_multi_endpoint_role"] = "checkpoint selection only; clustered by composite seed and engine"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8", newline="\n")
    print(
        "LATEST_REVISION_MANIFEST_SYNCED "
        f"csv={len(CSV_FILES)} figures={len(FIGURE_FILES)} output={manifest_path}"
    )


if __name__ == "__main__":
    main()
