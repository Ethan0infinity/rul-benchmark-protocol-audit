from __future__ import annotations

import argparse
import csv
from functools import lru_cache
import hashlib
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).absolute().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rul.reporting import collect_metrics, filter_metric_rows


MANIFEST_COLUMNS = [
    "experiment_name",
    "subset",
    "model",
    "seed",
    "protocol_version",
    "run_dir",
    "metrics_path",
    "predictions_path",
    "config_path",
    "checkpoint_path",
    "allowed_for_paper",
    "sha256_metrics",
    "sha256_predictions",
    "sha256_checkpoint",
]

TEXT_SUFFIXES = {
    ".bib",
    ".cff",
    ".cmd",
    ".csv",
    ".json",
    ".log",
    ".md",
    ".ps1",
    ".py",
    ".sh",
    ".tex",
    ".toml",
    ".txt",
    ".yaml",
    ".yml",
}

EXCLUDED_DIR_NAMES = {
    ".git",
    ".idea",
    ".mpl_cache",
    ".vscode",
    "__pycache__",
    "editorial_report",
    "logs",
    "replication_package",
    "superpowers",
}
EXCLUDED_FILE_NAMES = {
    "finalize_review_revision.ps1",
    "run_review_revision_queue.ps1",
    "result_brief.md",
    "table_endpoint_selection_confirmation.tex",
    "table_endpoint_selection_confirmation_zh.tex",
    "table_core_asym_paired.tex",
    "table_core_asym_paired_zh.tex",
    "table_core_asym_decision_cost.tex",
    "table_core_asym_decision_cost_zh.tex",
    "table_stress_auc_direct_predecessor.tex",
    "table_stress_auc_direct_predecessor_zh.tex",
    "table_core_rast_fixed_full.tex",
    "table_core_rast_fixed_full_zh.tex",
    "table_stress_decision_cost_rho.tex",
    "table_stress_decision_cost_rho_zh.tex",
    "table_stress_family_uncertainty.tex",
    "table_stress_family_uncertainty_zh.tex",
    "table_stress_operating_points.tex",
    "table_stress_operating_points_zh.tex",
}
EXCLUDED_FILE_SUFFIXES = {".pyc", ".pyo"}
EXCLUDED_REL_PATHS = {
    "docs/editorial_report",
    "docs/superpowers",
}
TOP_LEVEL_DIRS = [
    "benchmark",
    "configs",
    "docs",
    "eval_harness",
    "paper_outputs",
    "reports",
    "scripts",
    "src",
    "tests",
    "templates",
]
TOP_LEVEL_FILES = [
    "CITATION.cff",
    "Dockerfile",
    "Dockerfile.texlive2026",
    "environment.yml",
    "LICENSE",
    "THIRD_PARTY_NOTICES.md",
    "pyproject.toml",
    "README.md",
    "README_REVISED.md",
    "references.bib",
    "reproduce_all.sh",
    "reproduce_all.ps1",
    "reproduce_minimal.sh",
    "reproduce_minimal.ps1",
    "reproduce_release_only.sh",
    "reproduce_release_only.ps1",
    "requirements.txt",
    "tex_environment.lock",
    "texlive.profile",
    "artifact_metadata.json",
    "author_metadata.json",
]
RELEASE_DOC_ALLOWLIST = {
    "BENCHMARK_CLAIM_CALIBRATION_CHECKLIST.md",
    "RETROSPECTIVE_GOVERNANCE_EXTENSION_PROTOCOL.md",
    "ROUND20_CONCEPTUAL_REVISION_20260809.md",
    "ROUND20_VALIDATION_RECORD.md",
    "ROUND19_REVISION_RESPONSE_20260809.md",
    "PORTABLE_RUNTIME_CONTRACT.md",
    "TEX_TOOLCHAIN_SETUP.md",
    "LITERATURE_SEARCH_LOG_20260808.md",
    "STRESS_AND_TRANSFER_DATA_DICTIONARY.md",
    "dataset_brief.md",
    "delivery_profile.schema.json",
    "evidence_vocabulary.md",
    "experiment_matrix.md",
    "formal_benchmark_run_guide.md",
    "method_figure_contract.md",
    "paper_outline.md",
    "research_protocol_v3.md",
    "source_hierarchy.md",
}


def relative_to_root(path: str | Path, root: str | Path) -> str:
    path = Path(path)
    root = Path(root).resolve()
    if not path.is_absolute():
        path = (root / path).resolve()
    else:
        path = path.resolve()
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.as_posix()


def resolve_project_path(path: str | Path, root: str | Path) -> Path:
    path = Path(path)
    root = Path(root).resolve()
    if path.is_absolute():
        return path
    return root / path


def sha256_file(path: str | Path) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sanitize_text(text: str, project_root: Path) -> str:
    sanitized = text
    known_roots = (
        (project_root, "."),
        (Path(r"R:\rs_tcn_gru_rul"), "."),
        (project_root.parent / "正文", "manuscript"),
    )
    for root, replacement in known_roots:
        variants = {
            str(root),
            root.as_posix(),
            json.dumps(str(root), ensure_ascii=True)[1:-1],
            json.dumps(str(root), ensure_ascii=False)[1:-1],
        }
        prefix = "" if replacement == "." else replacement + "/"
        for value in sorted(variants, key=len, reverse=True):
            for separator in ("\\\\", "\\", "/"):
                sanitized = sanitized.replace(value + separator, prefix)
            sanitized = sanitized.replace(value, replacement)
    local_python = "C:" + "\\DevTools\\Python311\\python.exe"
    for value in (local_python, local_python.replace("\\", "\\\\")):
        sanitized = sanitized.replace(value, "python")
    return sanitized


@lru_cache(maxsize=8)
def current_runtime_logs(reports_root: str) -> frozenset[str]:
    root = Path(reports_root)
    selected: set[str] = set()
    contract_root = root / "stored_results_pipeline"
    for name in ("runtime_contract_full.json", "runtime_contract_quick.json"):
        path = contract_root / name
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        for stage in payload.get("stages", []):
            log = str(stage.get("log", "")).replace("\\", "/")
            if log.startswith("reports/"):
                log = log[len("reports/") :]
            if log:
                selected.add(log)
    return frozenset(selected)


def should_skip(path: Path, root: Path) -> bool:
    rel = path.relative_to(root)
    rel_posix = rel.as_posix()
    parts = set(rel.parts)
    if any(not part.isascii() for part in rel.parts):
        return True
    runtime_log = (
        root.name.lower() == "reports"
        and rel_posix.startswith("stored_results_pipeline/logs/")
    )
    if parts & EXCLUDED_DIR_NAMES and not runtime_log:
        return True
    if path.name in EXCLUDED_FILE_NAMES:
        return True
    lower_name = path.name.lower()
    if re.search(r"round[23](?!\d)", lower_name):
        return True
    if rel_posix in EXCLUDED_REL_PATHS or any(rel_posix.startswith(prefix + "/") for prefix in EXCLUDED_REL_PATHS):
        return True
    if path.suffix.lower() in EXCLUDED_FILE_SUFFIXES:
        return True
    root_scope = root.name.lower()
    if root_scope == "docs":
        if rel.parts and (
            len(rel.parts) > 1
            or rel.parts[0] not in RELEASE_DOC_ALLOWLIST
        ):
            return True
    if root_scope == "reports":
        if rel.parts and (
            rel.parts[0].lower().startswith("round")
            or rel.parts[0].lower() == "visual_audit"
        ):
            return True
        if rel.parts and rel.parts[0].startswith("stored_results_pipeline_round"):
            return True
        if (
            len(rel.parts) > 1
            and rel.parts[0] == "stored_results_pipeline"
            and rel.parts[1] in {"contracts", "historical"}
        ):
            return True
        if runtime_log and rel_posix not in current_runtime_logs(str(root.resolve())):
            return True
        if (
            path.name.startswith("clean_extraction_")
            or path.name.startswith("review_revision_")
            or path.name.lower().startswith("round")
            or path.suffix.lower() in {".err", ".out"}
        ):
            return True
        if path.name in {
            "cross_document_release.json",
            "clean_extraction_authority.json",
            "clean_extraction_release_test.json",
            "evidence_vocabulary_gate.json",
            "release_authority_gate.json",
            "submission_manuscript_gate.json",
            "tex_source_build_gate.json",
        }:
            return True
        if path.suffix.lower() == ".log" and not rel_posix.startswith(
            "stored_results_pipeline/logs/"
        ):
            return True
        if path.suffix.lower() == ".md" and path.name not in {
            "data_validation_report.md",
            "dataset_sufficiency_report.md",
        }:
            return True
    if root_scope == "paper_outputs" and rel.parts and rel.parts[0] in {
        "manuscript_v3",
        "release_evidence_smoke",
        "round3_evidence",
    }:
        return True
    if root_scope == "docs" and rel.parts and (
        rel.parts[0] not in RELEASE_DOC_ALLOWLIST
        and (
            rel.parts[0].lower().startswith("round")
            or "review_response" in rel.parts[0].lower()
            or rel.parts[0].lower()
            in {"analysis_chronology.md", "provenance_and_governance.md"}
        )
    ):
        return True
    if "data" in parts and "raw" in parts:
        return True
    return False


def copy_file_clean(src: Path, dst: Path, project_root: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    release_root = (project_root / "paper_outputs" / "release_v1").resolve()
    if (
        src.resolve().is_relative_to(release_root)
        or is_figure_integrity_artifact(src, project_root)
    ):
        shutil.copy2(src, dst)
        return
    if src.suffix.lower() in TEXT_SUFFIXES:
        try:
            text = src.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            shutil.copy2(src, dst)
            return
        dst.write_text(sanitize_text(text, project_root), encoding="utf-8", newline="\n")
    else:
        shutil.copy2(src, dst)


@lru_cache(maxsize=8)
def figure_integrity_relative_paths(project_root_text: str) -> frozenset[str]:
    """Return byte-authoritative figure, manifest, and source paths."""
    project_root = Path(project_root_text)
    relative = {
        "paper_outputs/advanced_evidence/figure_data_manifest.json",
    }
    manifest_path = (
        project_root
        / "paper_outputs"
        / "advanced_evidence"
        / "figure_data_manifest.json"
    )
    if not manifest_path.exists():
        return frozenset(relative)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    for item in manifest.get("figures", []):
        figure_path = str(
            item.get(
                "path",
                f"paper_outputs/advanced_evidence/figures/{item['figure']}",
            )
        ).replace("\\", "/")
        relative.add(figure_path)
        for source in item.get("source_files", []):
            relative.add(
                "paper_outputs/advanced_evidence/" + str(source["name"])
            )
    return frozenset(relative)


def is_figure_integrity_artifact(
    source: Path,
    project_root: Path,
) -> bool:
    try:
        relative = source.resolve().relative_to(project_root.resolve()).as_posix()
    except ValueError:
        return False
    return relative in figure_integrity_relative_paths(
        str(project_root.resolve())
    )


def validate_packaged_figure_data(package_root: Path) -> None:
    command = [
        sys.executable,
        "scripts/check_figure_data_consistency.py",
    ]
    completed = subprocess.run(
        command,
        cwd=package_root,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        detail = (completed.stdout + completed.stderr).strip()
        raise RuntimeError(
            "Packaged figure-data consistency failed before release:\n" + detail
        )


def copy_tree_clean(src: Path, dst: Path, project_root: Path) -> None:
    if not src.exists():
        return
    root = src.resolve()
    for path in root.rglob("*"):
        if should_skip(path, root):
            continue
        rel = path.relative_to(root)
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        else:
            copy_file_clean(path, target, project_root)


def reset_output_dir(out: Path, project_root: Path) -> None:
    out = out.resolve()
    project_root = project_root.resolve()
    if out == project_root:
        raise ValueError("Refusing to use project root itself as replication package output.")
    if project_root not in out.parents:
        raise ValueError(f"Output directory must stay inside the project root: {out}")
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)


def build_results_manifest(
    results_dir: str | Path,
    *,
    experiment_names: list[str] | None = None,
    experiment_prefix: str | None = None,
) -> list[dict[str, Any]]:
    results_root = Path(results_dir).resolve()
    project_root = results_root.parent
    rows = filter_metric_rows(
        collect_metrics(results_dir),
        experiment_names=experiment_names,
        experiment_prefix=experiment_prefix,
    )
    manifest: list[dict[str, Any]] = []
    for row in rows:
        run_dir = resolve_project_path(str(row["run_dir"]), project_root)
        metrics_path = run_dir / "metrics.json"
        predictions_path = run_dir / "test_predictions.csv"
        config_path = run_dir / "run_config.yaml"
        checkpoint_path = run_dir / "best_model.pt"
        manifest.append(
            {
                "experiment_name": row.get("experiment_name"),
                "subset": row.get("subset"),
                "model": row.get("model"),
                "seed": row.get("seed"),
                "protocol_version": row.get("protocol_version"),
                "run_dir": relative_to_root(run_dir, project_root),
                "metrics_path": relative_to_root(metrics_path, project_root),
                "predictions_path": relative_to_root(predictions_path, project_root),
                "config_path": relative_to_root(config_path, project_root),
                "checkpoint_path": relative_to_root(checkpoint_path, project_root),
                "allowed_for_paper": row.get("allowed_for_paper", False),
                "sha256_metrics": sha256_file(metrics_path) if metrics_path.exists() else "",
                "sha256_predictions": sha256_file(predictions_path) if predictions_path.exists() else "",
                "sha256_checkpoint": sha256_file(checkpoint_path) if checkpoint_path.exists() else "",
            }
        )
    return manifest


def normalize_packaged_metrics(package_root: str | Path) -> None:
    package_root = Path(package_root).resolve()
    results_root = package_root / "results"
    if not results_root.exists():
        return
    for metrics_path in results_root.rglob("metrics.json"):
        try:
            payload = json.loads(metrics_path.read_text(encoding="utf-8-sig"))
        except json.JSONDecodeError:
            continue
        payload["run_dir"] = metrics_path.parent.relative_to(package_root).as_posix()
        metrics_path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
            newline="\n",
        )


def build_packaged_results_manifest(
    package_root: str | Path,
    source_results_dir: str | Path,
    *,
    experiment_names: list[str] | None = None,
    experiment_prefix: str | None = None,
) -> list[dict[str, Any]]:
    package_root = Path(package_root).resolve()
    source_results_root = Path(source_results_dir).resolve()
    project_root = source_results_root.parent
    rows = filter_metric_rows(
        collect_metrics(source_results_dir),
        experiment_names=experiment_names,
        experiment_prefix=experiment_prefix,
    )
    manifest: list[dict[str, Any]] = []
    for row in rows:
        source_run_dir = resolve_project_path(str(row["run_dir"]), project_root)
        run_dir_rel = relative_to_root(source_run_dir, project_root)
        run_dir = package_root / run_dir_rel
        metrics_path = run_dir / "metrics.json"
        predictions_path = run_dir / "test_predictions.csv"
        config_path = run_dir / "run_config.yaml"
        checkpoint_path = run_dir / "best_model.pt"
        manifest.append(
            {
                "experiment_name": row.get("experiment_name"),
                "subset": row.get("subset"),
                "model": row.get("model"),
                "seed": row.get("seed"),
                "protocol_version": row.get("protocol_version"),
                "run_dir": run_dir_rel,
                "metrics_path": metrics_path.relative_to(package_root).as_posix(),
                "predictions_path": predictions_path.relative_to(package_root).as_posix(),
                "config_path": config_path.relative_to(package_root).as_posix(),
                "checkpoint_path": checkpoint_path.relative_to(package_root).as_posix(),
                "allowed_for_paper": row.get("allowed_for_paper", False),
                "sha256_metrics": sha256_file(metrics_path) if metrics_path.exists() else "",
                "sha256_predictions": sha256_file(predictions_path) if predictions_path.exists() else "",
                "sha256_checkpoint": sha256_file(checkpoint_path) if checkpoint_path.exists() else "",
            }
        )
    return manifest


def write_manifest(rows: list[dict[str, Any]], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({col: row.get(col, "") for col in MANIFEST_COLUMNS})


def write_replication_readme(out: Path) -> None:
    (out / "README_REPLICATION.md").write_text(
        "# Replication Package\n\n"
        "This directory is a standalone clean replication handoff for the C-MAPSS RUL paper project. "
        "It includes source code, scripts, configurations, generated tables and figures, reports, "
        "stored formal result artifacts, result hashes, and run commands. It intentionally excludes "
        "raw NASA C-MAPSS data files, downloaded paper PDFs, IDE metadata, local logs, Python caches, "
        "and machine-specific absolute paths.\n\n"
        "## Evidence Level\n\n"
        "- Main benchmark: 13 controlled configurations x 4 fixed C-MAPSS subsets x 5 composite seeds = 260 formal runs.\n"
        "- OCM-Asym is a researcher-specified illustrative preference point informed by one FD004 development split. A retrospective 3x3 split-by-stream refit diagnoses preference instability but is not selection-adjusted validation.\n"
        "- OCM versus RAST is a complete-configuration comparison; architecture, pathways, heads, objectives, and their interactions are not independently identified.\n"
        "- Main fixed-task evidence: seed-level paired effects and finite task means. Crossed bootstrap, Holm, max-t, "
        "and exact five-seed sign-flip quantities are retained only as uncalibrated or small-cluster diagnostics; "
        "none defines a confirmatory decision.\n"
        "- Endpoint sensitivity: one 80-cell Holm family with explicit late-event counts and undefined "
        "conditional severity when a cell contains no late event.\n"
        "- Normalized-coordinate perturbation diagnostics: nine FD004 algorithmic perturbation generators "
        "with five composite training streams crossed with five perturbation seeds. Reported envelopes are "
        "descriptive diagnostics, not physical fault models or calibrated robustness guarantees.\n"
        "- Retrospective direct-comparator controls: a result-informed 3x3 split-by-stream grid with "
        "all available three-level subset sensitivities, an FD004 three-seed OFAT preprocessing audit, "
        "and four LOFO augmentation-overlap audits. LOFO distinguishes zero-anchored AUC from the "
        "nonzero-range response and reports tolerance-specific event support.\n"
        "- Condition-normalization evidence: global, official-state, K-means, and continuous corrections "
        "under the same FD004 split, architecture, five training seeds, and optimization budget.\n"
        "- Exploratory computational proxy: 13-model, five-seed N-CMAPSS DS02 benchmark; derived caches are included only when redistribution metadata permits.\n"
        "- Supplementary ablation evidence: FD004 five-seed ablations for the main OCM-MST-GRU modules.\n"
        "- Additional evidence: OCM protocol/objective attribution, condition-cluster and auxiliary-loss sensitivity, "
        "fixed-test and development-unit N-CMAPSS split sensitivity, post-selection interval diagnostics, exact normalized hypervolume, seed-level Pareto "
        "stability, mechanism diagnostics, and paired failure-case analysis.\n"
        "- Quality gates: data validation, benchmark completeness, protocol consistency, checkpoint integrity, and final artifact checks.\n\n"
        "## Public Artifact Release Status\n\n"
        "This package is deposit-ready, but the public repository identifier must be filled in only after "
        "the author uploads the package to a real repository such as GitHub Releases, Zenodo, OSF, "
        "an institutional repository, or another trusted archive. Do not cite a placeholder as a public "
        "artifact. Before journal submission, record all of the following in the manuscript and in this README:\n\n"
        "- Repository URL: unresolved until author deposit\n"
        "- Release tag or commit hash: unresolved until author deposit\n"
        "- DOI or repository record identifier: unresolved until author deposit\n"
        "- Archive date and uploader: unresolved until author deposit\n"
        "- Whether raw NASA C-MAPSS files are redistributed: no; users download them from the official source.\n\n"
        "## Computational Requirements\n\n"
        "- Python: 3.11 recommended; key dependencies are pinned in `requirements.txt` and `environment.yml`.\n"
        "- Stored-result quick validation: a standard laptop CPU is sufficient; expected runtime is usually under 5 minutes.\n"
        "- Full stored-result regeneration: stage-level wall time, peak resident memory, commands, and logs are written to "
        "`reports/stored_results_pipeline/runtime_contract_full.{json,csv}` and "
        "`runtime_contract_quick.{json,csv}`. The compatibility "
        "`runtime_contract.{json,csv}` points to the latest full run when one exists. "
        "Append-only historical copies remain in the author workspace and are excluded from the clean release. "
        "The verified machine-specific values in those "
        "files supersede generic runtime estimates.\n"
        "- Full retraining: a CUDA-capable GPU is recommended; the 260-run main grid and 373-unique-artifact retrospective extension can take many hours depending on GPU speed. The extension contains 412 registered family cells because exact reference and fixed-candidate artifacts are shared across compatible designs.\n"
        "- Randomness: training and perturbation seeds are independently crossed over 42, 123, 2024, 2025, and 2026 for the FD004 stress-test grid.\n\n"
        "## Use\n\n"
        "Run commands from this package root after installing dependencies. The stored-result route checks "
        "the formal results already included in the package and does not require raw C-MAPSS files. Full "
        "retraining requires downloading the public NASA C-MAPSS files into `data/raw/` first.\n\n"
        "Stored-result route:\n\n"
        "```bash\n"
        "bash reproduce_minimal.sh --stored-results --quick\n"
        "```\n\n"
        "On Windows PowerShell, use:\n\n"
        "```powershell\n"
        ".\\reproduce_minimal.ps1 -StoredResults -Quick\n"
        "```\n\n"
        "The quick route verifies `results_manifest.csv`, checks benchmark completeness, regenerates "
        "paper tables, and reruns statistical tests. Omit `--quick` / `-Quick` to also validate the "
        "stored external, ablation, stress, mechanism, calibration, and advanced-evidence artifacts.\n\n"
        "The full route is stage-aware. List stages with `python scripts/run_stored_results_pipeline.py "
        "--list-stages`; resume completed stages with `--resume` / `-Resume`. To restart from one stage, "
        "use `--from-stage NAME` / `-FromStage NAME`; this is accepted only when all predecessor stages "
        "are recorded PASS in `reports/stored_results_pipeline/pipeline_state.json`. Every stage has a "
        "separate log and a measured wall-time/peak-memory record. Silent stages emit a heartbeat every "
        "15 seconds by default; override it with `--heartbeat-seconds`. The supplied runtime contract is "
        "a deposited author-environment run report. A third party should preserve its own newly generated "
        "contract separately and must not describe the deposited report as an independent reproduction.\n\n"
        "Release-required commands are cross-platform Python or the supplied bash/PowerShell wrappers. "
        "Host finalization helpers are authoring-only and excluded from the portable release; "
        "`scripts/check_package_closure.py` verifies that tests and documented release commands do not "
        "depend on excluded host files.\n\n"
        "When the package is nested in a submission delivery, the route first verifies the delivery-root "
        "`checksums.sha256`. Regeneration then intentionally rewrites generated tables, figures, manifests, "
        "and reports. Use a disposable extraction and re-extract the ZIP before re-auditing its original "
        "checksum state.\n\n"
        "The canonical/mirror hierarchy is documented in `docs/source_hierarchy.md`. The fixed-benchmark "
        "semantic release is `paper_outputs/release_v1`; `paper_outputs/retrospective_governance_v1` is "
        "the stable index for retrospective evidence, while the historical `round12_new_evidence` name "
        "is retained for manifest compatibility. `analysis_working`, `advanced_evidence`, and `manuscript_v3` "
        "are generated working layers, while manuscript `generated/` directories are typesetting mirrors. "
        "`reports/stored_results_pipeline/post_regeneration_digest.csv` records regenerated hashes without "
        "replacing the deposit checksum.\n\n"
        "Full retraining route after downloading raw data:\n\n"
        "```bash\n"
        "bash reproduce_all.sh\n"
        "```\n\n"
        "For the complete retraining pipeline, use `bash reproduce_all.sh` or `./reproduce_all.ps1` on "
        "Windows PowerShell. These scripts check for raw data up front and print the download command if "
        "the files are missing.\n\n"
        "## Data Integrity\n\n"
        "Expected raw files:\n\n"
        "```text\n"
        "train_FD001.txt  test_FD001.txt  RUL_FD001.txt\n"
        "train_FD002.txt  test_FD002.txt  RUL_FD002.txt\n"
        "train_FD003.txt  test_FD003.txt  RUL_FD003.txt\n"
        "train_FD004.txt  test_FD004.txt  RUL_FD004.txt\n"
        "```\n\n"
        "Expected validation output:\n\n"
        "```text\n"
        "DATA_VALIDATION_PASS\n"
        "Data validation checks: 88 total, 0 failed\n"
        "```\n\n"
        "Validated local file structure:\n\n"
        "```text\n"
        "FD001: 100 train engines, 100 test engines\n"
        "FD002: 260 train engines, 259 test engines\n"
        "FD003: 100 train engines, 100 test engines\n"
        "FD004: 249 train engines, 248 test engines\n"
        "```\n\n"
        "FD004 is documented from the actual validated local files: 249 training engines, "
        "248 test engines, and 248 RUL rows. Use `reports/data_validation_report.md` and "
        "`reports/raw_file_checksums.sha256` as the audit trail for this file structure.\n\n"
        "## Manifest\n\n"
        "`results_manifest.csv` uses project-relative POSIX paths such as "
        "`results/paper_main_v3_seed42/FD001/gru/metrics.json`. SHA256 columns hash the packaged "
        "metrics and prediction files after path sanitization. Verify it with:\n\n"
        "```bash\n"
        "python scripts/verify_results_manifest.py --package-root . --expected-rows 260\n"
        "```\n"
        "`supplementary_results_manifest.csv` records the copied ablation, N-CMAPSS, design-sensitivity, "
        "and closest-prior checkpoints separately so that the fixed 260-row main manifest remains unambiguous.\n"
        "\n## Exhibit and Script Map\n\n"
        "This table is the source-of-truth map for the manuscript exhibits. The manuscript Data and Code "
        "Availability statement can stay short because the detailed directory structure and generation route "
        "are documented here. The qualitative scope map moved out of the main manuscript is stored at "
        "in this README and the release-level registry.\n\n"
        "| Manuscript item | Primary generating script | Main output |\n"
        "|---|---|---|\n"
        "| Recent-method scope context (`tab:recent_context`) | manuscript source and `references.bib` | `manuscript/main_revised.tex`, `references.bib` |\n"
        "| Dataset profile (`tab:dataset`) | `scripts/validate_data.py`, `scripts/analyze_dataset_sufficiency.py`, `scripts/build_revised_assets.py` | `reports/dataset_summary.csv`, `paper_outputs/revised_tables/dataset_summary_revised.csv` |\n"
        "| Inference-latency protocol (`tab:inference-latency-protocol`) | `scripts/benchmark_inference_protocol.py`, `scripts/build_release_evidence.py` | `paper_outputs/release_v1/inference_latency_protocol_summary.csv` |\n"
        "| Fixed-task diagnostic display (`fig:primary-fixed-diagnostic`) | `scripts/analyze_release_statistics.py`, `scripts/build_release_evidence.py` | `paper_outputs/release_v1/primary_fixed_task_estimates.csv`, `figures/fig_primary_fixed_task_diagnostic_envelope.*` |\n"
        "| FD004 ablation (`tab:v3-ablation`) | `scripts/run_multiseed_ablation.py`, `scripts/build_paper_tables.py`, `scripts/build_revised_assets.py` | `paper_outputs/revised_tables/ablation_*_revised.csv` |\n"
        "| Run-quality robust summary (`tab:run-quality-robust-summary`) | `scripts/analyze_tcn_gru_instability.py` | `paper_outputs/advanced_evidence/run_quality_model_robust_summary.csv`, `run_quality_cell_audit.csv` |\n"
        "| Dataset summary table | `scripts/validate_data.py`, `scripts/analyze_dataset_sufficiency.py`, `scripts/build_revised_assets.py` | `reports/dataset_summary.csv`, `paper_outputs/revised_tables/dataset_summary_revised.csv` |\n"
        "| Main benchmark tables | `scripts/build_paper_tables.py`, `scripts/build_revised_assets.py` | `paper_outputs/tables/table_multiseed_*.csv`, `paper_outputs/revised_tables/multiseed_*_revised.csv` |\n"
        "| FD004 Pareto and RMSE figures | `scripts/build_revised_assets.py` | `paper_outputs/revised_figures/fig_fd004_pareto_tradeoff.png`, `fig_multiseed_rmse_errorbar.png` |\n"
        "| Threshold-sensitivity table/figure | `scripts/analyze_threshold_sensitivity.py`, `scripts/build_revised_assets.py` | `paper_outputs/revised_tables/threshold_sensitivity_*.csv`, `fig_fd004_threshold_sensitivity.png` |\n"
        "| Ablation tables | `scripts/run_multiseed_ablation.py`, `scripts/build_paper_tables.py`, `scripts/build_revised_assets.py` | `paper_outputs/revised_tables/ablation_*_revised.csv` |\n"
        "| Multi-seed normalized-space perturbation evidence | `scripts/run_multiseed_stress_tests.py` | `paper_outputs/advanced_evidence/stress_*` |\n"
        "| Training-augmentation distribution sensitivity | `scripts/run_augmentation_distribution_sensitivity.py`, `scripts/analyze_augmentation_distribution_sensitivity.py` | `paper_outputs/advanced_evidence/augmentation_distribution_*`, `figures/fig_augmentation_distribution_sensitivity.*` |\n"
        "| FD004 split-by-training-seed audit | `scripts/run_split_training_seed_factorial.py`, `scripts/analyze_split_training_seed_factorial.py` | `paper_outputs/advanced_evidence/seed_factorial_fd004_*`, `figures/fig_seed_factorial_fd004.*` |\n"
        "| N-CMAPSS exploratory fixed-task benchmark | `scripts/run_ncmapss_benchmark.py` | `paper_outputs/advanced_evidence/ncmapss_ds02_*` |\n"
        "| N-CMAPSS per-unit audit | `scripts/analyze_ncmapss_units.py` | `paper_outputs/advanced_evidence/ncmapss_ds02_per_unit_*` |\n"
        "| N-CMAPSS development-unit rotation | `scripts/prepare_ncmapss_dev_rotation.py`, `scripts/run_ncmapss_dev_rotation.py`, `scripts/analyze_ncmapss_dev_rotation.py` | `paper_outputs/advanced_evidence/ncmapss_dev_rotation_*`, `figures/fig_ncmapss_dev_rotation.*` |\n"
        "| Ten-stream fixed-task retraining | `scripts/run_round12_new_experiments.py`, `scripts/analyze_round12_new_experiments.py` | `paper_outputs/round12_new_evidence/independent_stream_*` |\n"
        "| Result-informed split-by-training-stream audit and level-subset sensitivity | `scripts/run_round12_new_experiments.py`, `scripts/analyze_round12_new_experiments.py` | `paper_outputs/round12_new_evidence/crossed_split_stream_*`, `crossed_level_subset_*` |\n"
        "| FD004 three-seed OFAT window and sensor-budget sensitivity | `scripts/run_round12_new_experiments.py`, `scripts/analyze_round12_new_experiments.py` | `paper_outputs/round12_new_evidence/cmapss_preprocessing_*` |\n"
        "| Split-by-stream preference retraining | `scripts/run_round12_new_experiments.py`, `scripts/analyze_round12_new_experiments.py` | `paper_outputs/round12_new_evidence/preference_retraining_*` |\n"
        "| Shared-objective backbone factorial | `scripts/run_round12_new_experiments.py`, `scripts/analyze_round12_new_experiments.py` | `paper_outputs/round12_new_evidence/factorial_*` |\n"
        "| Leave-one-perturbation-family-out augmentation-overlap audit | `scripts/run_round12_new_experiments.py`, `scripts/analyze_round12_new_experiments.py` | `paper_outputs/round12_new_evidence/lofo_*` |\n"
        "| N-CMAPSS sampling--window sensitivity | `scripts/run_round12_new_experiments.py`, `scripts/analyze_round12_new_experiments.py` | `paper_outputs/round12_new_evidence/ncmapss_sampling_window_*` |\n"
        "| N-CMAPSS fixed-source-span sampling control | `scripts/run_round12_new_experiments.py`, `scripts/analyze_round12_new_experiments.py` | `paper_outputs/round12_new_evidence/ncmapss_horizon_*` |\n"
        "| Closest-prior FD002 comparison | `scripts/run_close_prior_baseline.py` | `paper_outputs/advanced_evidence/close_prior_fd002_*` |\n"
        "| Matched FD002 Core/Asym protocol transfer | `scripts/run_fd002_protocol_transfer.py`, `scripts/analyze_fd002_protocol_transfer.py` | `results/fd002_protocol_transfer/run_manifest.csv`, `paper_outputs/advanced_evidence/fd002_protocol_transfer_*` |\n"
        "| Official-code-derived Dual-Mixer reference | `scripts/run_official_dual_mixer_baseline.py`, `scripts/analyze_official_dual_mixer_baseline.py` | `paper_outputs/advanced_evidence/official_dual_mixer_*` |\n"
        "| Learned endpoint-width diagnostic | `scripts/build_advanced_evidence.py` | `paper_outputs/advanced_evidence/interval_calibration_*`, `figures/fig_interval_calibration.pdf`; no nominal-coverage interpretation |\n"
        "| Legacy post-selection residual adjustment | `scripts/calibrate_prediction_intervals.py` | Legacy artifact names retained for audit compatibility; excluded from inferential claims |\n"
        "| Legacy endpoint coverage audit | `scripts/analyze_interval_reliability.py` | Legacy files retained for provenance; nominal-coverage curves are excluded from the manuscript |\n"
        "| Pareto, rank, and hierarchical evidence | `scripts/build_advanced_evidence.py` | `paper_outputs/advanced_evidence/pareto_*`, `benchmark_rank_*`, `hierarchical_bootstrap_bca.csv` |\n"
        "| Fixed-subset paired uncertainty | `scripts/analyze_subset_paired_bootstrap.py` | `paper_outputs/advanced_evidence/subset_engine_paired_bootstrap.csv` |\n"
        "| Asymmetric-loss component audit | `scripts/analyze_asymmetry_evidence.py` | `paper_outputs/advanced_evidence/asymmetry_fd004_*` |\n"
        "| Computational-proxy simplification sensitivity | `scripts/analyze_simplification_confirmation.py` | `paper_outputs/advanced_evidence/simplification_ncmapss_*` |\n"
        "| Checkpoint-rule sensitivity | `scripts/analyze_checkpoint_sensitivity.py` | `paper_outputs/advanced_evidence/checkpoint_sensitivity_*` |\n"
        "| Validation-endpoint checkpoint sensitivity | `scripts/run_endpoint_selection_confirmation.py` | `paper_outputs/advanced_evidence/endpoint_selection_confirmation_*` (historical key) |\n"
        "| Validation/test endpoint-distribution audit | `scripts/analyze_validation_endpoint_distributions.py` | `paper_outputs/advanced_evidence/validation_endpoint_distribution_*`, `figures/fig_validation_endpoint_distribution.*` |\n"
        "| Endpoint statistical-unit and cluster audit | `scripts/run_endpoint_selection_confirmation.py`, `scripts/analyze_validation_endpoint_distributions.py` | `paper_outputs/advanced_evidence/endpoint_selection_confirmation_bootstrap.csv`, `validation_endpoint_cluster_audit.csv` |\n"
        "| Endpoint epochs, Core/Asym comparator intervals, and asymmetric error costs | `scripts/analyze_major_revision_evidence.py` | `paper_outputs/advanced_evidence/endpoint_selection_*`, `core_asym_major_*`, `core_asym_decision_cost_*`, `paper_outputs/figures/fig_core_asym_decision_cost.*` |\n"
        "| Symmetric fixed-subset Core/Asym versus RAST inference | `scripts/analyze_major_revision_evidence.py` | `paper_outputs/advanced_evidence/core_asym_rast_subset_bootstrap.csv` |\n"
        "| Common-risk versus conventional protocol track | `scripts/analyze_standard_protocol_track.py` | `paper_outputs/advanced_evidence/standard_protocol_track_*` |\n"
        "| OCM protocol and objective build-up attribution | `scripts/run_attribution_experiments.py`, `scripts/analyze_ocm_attribution.py` | `paper_outputs/advanced_evidence/ocm_attribution_*` |\n"
        "| Cross-backbone cumulative protocol build-up | `scripts/run_cross_backbone_protocol_buildup.py`, `scripts/analyze_cross_backbone_protocol_buildup.py` | `paper_outputs/advanced_evidence/cross_backbone_protocol_buildup_*`, `cross_backbone_protocol_buildup_transitions.csv`, `figures/fig_cross_backbone_protocol_buildup.*` |\n"
        "| Fixed-seed fast-cuDNN repeat audit | `scripts/run_fast_cudnn_repeat_audit.py`, `scripts/analyze_fast_cudnn_repeat_audit.py` | `results/fast_cudnn_repeat_audit.json`, `paper_outputs/advanced_evidence/fast_cudnn_repeat_*` |\n"
        "| OCM Core versus Asym preference sensitivity | `scripts/run_core_experiments.py`, `scripts/analyze_final_revision_evidence.py` | `paper_outputs/advanced_evidence/ocm_core_asym_*` |\n"
        "| Absolute architecture and selected-comparator attribution | `scripts/analyze_final_revision_evidence.py` | `paper_outputs/advanced_evidence/architecture_*`, `track_b_absolute_ranking.csv` |\n"
        "| Condition-cluster and auxiliary-loss sensitivity | `scripts/run_design_sensitivity.py`, `scripts/analyze_design_sensitivity.py` | `paper_outputs/advanced_evidence/hyperparameter_sensitivity_*`, `condition_cluster_k_*` |\n"
        "| Nine-family absolute stress summary and declared failure case | `scripts/build_stress_pair_extension.py` | `paper_outputs/advanced_evidence/stress_pair_extended_*`, `figures/fig_stress_*` |\n"
        "| Core/Asym nine-family stress extension | `scripts/extend_core_stress_evidence.py` | `paper_outputs/advanced_evidence/stress_operating_points_*`, `figures/fig_stress_operating_points.*` |\n"
        "| Engine-level crossed-factor stress bootstrap | `scripts/build_engine_level_stress_evidence.py` | `paper_outputs/advanced_evidence/stress_engine_level_predictions.csv`, `stress_operating_points_engine_bootstrap.csv`, `stress_global_missing_failure_curve.csv`, `figures/fig_stress_global_missing_failure.*` |\n"
        "| Fixed-task estimation, endpoint family, event support, and small-cluster calibration | `scripts/analyze_release_statistics.py`, `scripts/build_release_evidence.py` | `paper_outputs/release_v1/primary_fixed_task_estimates.csv`, `endpoint_tolerance_sensitivity.csv`, `late_event_support.csv`, `small_cluster_calibration.csv` |\n"
        "| Max-t family calibration and multiplicity registry | `scripts/analyze_release_statistics.py`, `scripts/build_release_evidence.py` | `paper_outputs/release_v1/max_t_family_calibration.csv`, `multiplicity_registry.csv` |\n"
        "| Matched condition-normalization controls | `scripts/run_condition_normalization_controls.py`, `scripts/analyze_condition_normalization_controls.py`, `scripts/build_release_evidence.py` | `paper_outputs/release_v1/condition_normalization_summary.csv`, `condition_normalization_contrasts.csv` |\n"
        "| Normalized-coordinate perturbation evidence | `scripts/analyze_normalized_perturbation_evidence.py`, `scripts/build_release_evidence.py` | `paper_outputs/release_v1/normalized_perturbation_contrasts.csv`, `normalized_perturbation_scale_calibration.csv` |\n"
        "| Mask-metadata and inference-latency diagnostics | `scripts/analyze_mask_observation_sensitivity.py`, `scripts/benchmark_inference_protocol.py`, `scripts/build_release_evidence.py` | `paper_outputs/release_v1/mask_metadata_sensitivity_summary.csv`, `inference_latency_protocol_summary.csv` |\n"
        "| Semantic evidence consistency gate | `scripts/release_evidence_gate.py` | console token `RELEASE_EVIDENCE_PASS` |\n"
        "| Stress decision-cost sensitivity at rho 2/5/10 | `scripts/build_engine_level_stress_evidence.py` | `paper_outputs/advanced_evidence/stress_operating_points_engine_bootstrap.csv` |\n"
        "| Normalized stress-curve means and reporting-margin sensitivity | `scripts/build_stress_pair_extension.py` | `paper_outputs/advanced_evidence/stress_pair_auc_comparison.csv`, `stress_equivalence_margin_sensitivity.csv` |\n"
        "| N-CMAPSS fixed-test split sensitivity | `scripts/prepare_ncmapss_split_sensitivity.py`, `scripts/analyze_ncmapss_split_sensitivity.py` | `paper_outputs/advanced_evidence/ncmapss_split_sensitivity_*` |\n"
        "| N-CMAPSS condition-cluster sensitivity | `scripts/run_ncmapss_k_sensitivity.py` | `paper_outputs/advanced_evidence/ncmapss_k_sensitivity_*` |\n"
        "| N-CMAPSS validation-only cluster selection audit | `scripts/analyze_ncmapss_validation_only_k_selection.py` | `paper_outputs/advanced_evidence/ncmapss_validation_only_k_selection.csv`, `ncmapss_validation_only_k_verdict.csv` |\n"
        "| Late-error tail risk and paired threshold intervals | `scripts/build_advanced_evidence.py`, `scripts/analyze_threshold_sensitivity.py` | `paper_outputs/advanced_evidence/late_tail_risk_summary.csv`, `threshold_paired_bootstrap.csv` |\n"
        "| Mechanism and condition-normalization evidence | `scripts/analyze_model_mechanisms.py`, `scripts/analyze_condition_normalization.py` | `paper_outputs/advanced_evidence/*mechanism*`, `condition_normalization_*`, `degradation_rul_*` |\n"
        "| Cross-model checkpoint and heavy-tail audit (`tab:run-quality-robust-summary`) | `scripts/analyze_tcn_gru_instability.py` | `paper_outputs/advanced_evidence/run_quality_cell_audit.csv`, `run_quality_model_robust_summary.csv`, `figures/fig_tcn_gru_seed_instability.*` |\n"
        "| Evidence-level classification and latest-revision hashes | `scripts/build_manuscript_evidence.py`, `scripts/sync_latest_revision_manifest.py` | `paper_outputs/manuscript_v3/table_evidence_levels*.tex`, `paper_outputs/advanced_evidence/advanced_evidence_manifest.json` |\n"
        "| Figure-data integrity | `scripts/check_figure_data_consistency.py` | `paper_outputs/advanced_evidence/figure_data_manifest.json`, `reports/figure_data_consistency.json` |\n"
        "| Extended-review evidence gate and manuscript tables | `scripts/check_extended_review_evidence.py`, `scripts/build_extended_review_tables.py` | `paper_outputs/manuscript_v3/table_augmentation_distribution_*`, `table_seed_factorial_*`, `table_ncmapss_dev_rotation_*` |\n"
        "| Vector method-workflow QA | `scripts/check_method_figure_qa.py` | `paper_outputs/advanced_evidence/figures/fig_method_architecture.*`, `docs/method_figure_contract.md` |\n"
        "| Design sensitivity and failure cases | `scripts/run_design_sensitivity.py`, `scripts/analyze_failure_cases.py` | `paper_outputs/advanced_evidence/design_sensitivity_*`, `failure_case_*` |\n"
        "| Statistical-test table | `scripts/run_statistical_tests.py`, `scripts/build_revised_assets.py` | `paper_outputs/tables/table_statistical_tests.csv`, `paper_outputs/revised_tables/statistical_*_revised.csv` |\n"
        "| Replication manifest | `scripts/create_replication_package.py`, `scripts/verify_results_manifest.py` | `results_manifest.csv` |\n",
        encoding="utf-8",
    )


def write_data_readme(out: Path) -> None:
    data_dir = out / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    (data_dir / "README.md").write_text(
        "# Data Availability\n\n"
        "This replication package does not redistribute the raw NASA C-MAPSS files. "
        "Download the public C-MAPSS dataset from the NASA Prognostics Center of Excellence "
        "or NASA Open Data source according to their terms, then place the files under "
        "`data/raw/` in the package root before running data validation or full retraining.\n\n"
        "Expected raw files:\n\n"
        "```text\n"
        "train_FD001.txt  test_FD001.txt  RUL_FD001.txt\n"
        "train_FD002.txt  test_FD002.txt  RUL_FD002.txt\n"
        "train_FD003.txt  test_FD003.txt  RUL_FD003.txt\n"
        "train_FD004.txt  test_FD004.txt  RUL_FD004.txt\n"
        "```\n\n"
        "Run `python scripts/validate_data.py` before reproducing tables or retraining models.\n\n"
        "## N-CMAPSS DS02 Computational Proxy\n\n"
        "The strongly downsampled computational-proxy task is derived from NASA N-CMAPSS DS02-006. The raw 2.45 GB HDF5 "
        "file and the derived NPZ cache are not redistributed under the conservative artifact policy. "
        "The package retains the official source URL, access date, fixed unit split, sampling factor, window "
        "definition, schema, SHA256, validation code, and deterministic preparation route. Users download "
        "the source from NASA under the source terms and rebuild the cache locally.\n",
        encoding="utf-8",
    )


def write_papers_readme(out: Path) -> None:
    papers_dir = out / "papers"
    papers_dir.mkdir(parents=True, exist_ok=True)
    (papers_dir / "README.md").write_text(
        "# Literature Files\n\n"
        "Downloaded article PDFs are not redistributed in this clean package. "
        "Use the bibliography and `download_links.md` to retrieve sources through legal "
        "publisher or repository access.\n",
        encoding="utf-8",
    )


def write_public_release_checklist(out: Path) -> None:
    (out / "PUBLIC_ARTIFACT_RELEASE_CHECKLIST.md").write_text(
        "# Public Artifact Release Checklist\n\n"
        "Complete this checklist before journal submission. The manuscript should cite only real, "
        "accessible artifact identifiers, never placeholders.\n\n"
        "## Required Release Fields\n\n"
        "- [ ] Public repository URL recorded.\n"
        "- [ ] Release tag, commit hash, or immutable archive version recorded.\n"
        "- [ ] DOI or repository record identifier recorded, if the host provides one.\n"
        "- [ ] The delivery-root `checksums.sha256` verifies after a clean extraction.\n"
        "- [ ] `results_manifest.csv` verifies with `MANIFEST_VERIFY_PASS rows=260`.\n"
        "- [ ] Minimal stored-result route tested from a fresh extracted archive.\n"
        "- [ ] Full retraining route documented with hardware and expected runtime.\n"
        "- [ ] Raw NASA C-MAPSS redistribution status stated clearly: not redistributed.\n"
        "- [ ] N-CMAPSS processed-cache redistribution terms checked; include the cache only if permitted.\n"
        "- [ ] Manuscript Data and Code Availability paragraph updated with the final URL and identifier.\n\n"
        "## Suggested Public Statement Template\n\n"
        "Code, processed stored-result artifacts, generated tables, generated figures, and replication "
        "instructions are available at `<PUBLIC_URL>`, release `<TAG_OR_COMMIT>`, DOI `<DOI_IF_ANY>`. "
        "The package excludes raw NASA C-MAPSS data; users should download the public dataset from NASA "
        "and place the files under `data/raw/` before full retraining.\n",
        encoding="utf-8",
        newline="\n",
    )


def copy_package_contents(root: Path, out: Path) -> None:
    for filename in TOP_LEVEL_FILES:
        src = root / filename
        if src.exists():
            copy_file_clean(src, out / filename, root)

    for dirname in TOP_LEVEL_DIRS:
        src = root / dirname
        if src.exists():
            copy_tree_clean(src, out / dirname, root)

    download_links = root / "papers" / "download_links.md"
    if download_links.exists():
        copy_file_clean(download_links, out / "papers" / "download_links.md", root)

    metadata_path = root / "artifact_metadata.json"
    artifact_metadata = json.loads(metadata_path.read_text(encoding="utf-8-sig")) if metadata_path.exists() else {}
    processed = root / "data" / "external" / "processed"
    if processed.exists() and artifact_metadata.get("ncmapss_cache_distribution") != "excluded":
        copy_tree_clean(processed, out / "data" / "external" / "processed", root)
    external_readme = root / "data" / "external" / "README.md"
    if external_readme.exists():
        copy_file_clean(external_readme, out / "data" / "external" / "README.md", root)


def copy_evidence_results(root: Path, out: Path) -> None:
    results = root / "results"
    patterns = (
        "paper_main_v3_seed*",
        "condition_control_global_fd004_seed*",
        "condition_control_official_state_fd004_seed*",
        "condition_control_continuous_fd004_seed*",
        "core_ocm_seed*",
        "ablation_fd004_seed*",
        "ncmapss_ds02_seed*",
        "ncmapss_ds02_val*_seed*",
        "design_sensitivity_v3_seed*",
        "design_sensitivity_k*_fd004_seed*",
        "design_sensitivity_aux*_fd004_seed*",
        "attribution_ocm_*_fd004_seed*",
        "standard_protocol_track",
        "close_prior_fd002_v3_seed*",
        "fd002_protocol_transfer",
        "augmentation_distribution_sensitivity",
        "augmentation_distribution_*",
        "seed_factorial_fd004",
        "seed_factorial_fd004_*",
        "tuning_risk_v3_ll*_lo*",
        "ncmapss_dev_rotation",
        "ncmapss_dev_rotation_*",
        "round12_new_evidence",
    )
    copied: set[Path] = set()
    for pattern in patterns:
        for source in results.glob(pattern):
            if not source.is_dir() or source in copied:
                continue
            copied.add(source)
            copy_tree_clean(source, out / "results" / source.name, root)
    for summary_name in (
        "ablation_fd004_multiseed_runs.csv",
        "attribution_experiments.json",
        "design_sensitivity_experiments.json",
        "condition_normalization_controls.json",
    ):
        summary = results / summary_name
        if summary.exists():
            copy_file_clean(summary, out / "results" / summary.name, root)


def generate_replication_package(
    project_root: str | Path = PROJECT_ROOT,
    output_dir: str | Path | None = None,
    *,
    experiment_names: list[str] | None = None,
    experiment_prefix: str | None = None,
) -> Path:
    root = Path(project_root).absolute()
    out = Path(output_dir).absolute() if output_dir is not None else root / "replication_package"
    reset_output_dir(out, root)
    copy_package_contents(root, out)
    copy_evidence_results(root, out)
    normalize_packaged_metrics(out)
    manifest = build_packaged_results_manifest(
        out,
        root / "results",
        experiment_names=experiment_names,
        experiment_prefix=experiment_prefix,
    )
    write_manifest(manifest, out / "results_manifest.csv")
    packaged_manifest = build_packaged_results_manifest(out, out / "results")
    supplementary_manifest = [
        row for row in packaged_manifest if not str(row.get("experiment_name", "")).startswith("paper_main_v3_seed")
    ]
    write_manifest(supplementary_manifest, out / "supplementary_results_manifest.csv")
    write_replication_readme(out)
    write_data_readme(out)
    write_papers_readme(out)
    write_public_release_checklist(out)
    validate_packaged_figure_data(out)
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a clean standalone replication package.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT))
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--experiment-name", action="append", dest="experiment_names", default=None)
    parser.add_argument("--experiment-prefix", default=None)
    args = parser.parse_args()

    out = generate_replication_package(
        args.project_root,
        args.output_dir,
        experiment_names=args.experiment_names,
        experiment_prefix=args.experiment_prefix,
    )
    print(f"REPLICATION_PACKAGE_READY {out}")


if __name__ == "__main__":
    main()
