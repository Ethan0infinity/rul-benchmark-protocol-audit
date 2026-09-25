from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

PROJECT_ROOT = Path(__file__).absolute().parents[1]
WORKSPACE_ROOT = PROJECT_ROOT.parent

sys.path.insert(0, str(Path(__file__).absolute().parent))
from create_replication_package import generate_replication_package  # noqa: E402


TEXT_SUFFIXES = {
    ".bib",
    ".cff",
    ".csv",
    ".json",
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

EXCLUDED_SUFFIXES = {".aux", ".bbl", ".blg", ".fdb_latexmk", ".fls", ".log", ".out", ".synctex.gz", ".toc"}
EXCLUDED_NAMES = {".DS_Store", "Thumbs.db"}
OBSOLETE_GENERATED_STEMS = {
    "table_endpoint_selection_confirmation",
    "table_core_asym_paired",
    "table_core_asym_decision_cost",
    "table_stress_auc_direct_predecessor",
    "table_core_rast_fixed_full",
    "table_stress_decision_cost_rho",
    "table_stress_family_uncertainty",
    "table_stress_operating_points",
}


def find_manuscript_dir() -> Path:
    candidates = [WORKSPACE_ROOT / "正文"]
    candidates.extend(path for path in WORKSPACE_ROOT.iterdir() if path.is_dir())
    for path in candidates:
        if (path / "main_revised.tex").exists():
            return path
    raise FileNotFoundError("Could not find manuscript directory containing main_revised.tex.")


def reset_dir(path: Path) -> None:
    path = path.resolve()
    root = WORKSPACE_ROOT.resolve()
    if path == root or root not in path.parents:
        raise ValueError(f"Refusing to reset path outside workspace root: {path}")
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_text_or_binary(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if src.suffix.lower() in TEXT_SUFFIXES:
        dst.write_text(src.read_text(encoding="utf-8", errors="replace"), encoding="utf-8", newline="\n")
        shutil.copystat(src, dst)
    else:
        shutil.copy2(src, dst)


def copy_tree_clean(src: Path, dst: Path) -> None:
    for path in src.rglob("*"):
        rel = path.relative_to(src)
        if any(not part.isascii() for part in rel.parts):
            continue
        if any(part in {"__pycache__", ".git", ".idea", ".vscode", ".mpl_cache"} for part in rel.parts):
            continue
        if "round2" in path.name.lower() or "round3" in path.name.lower():
            continue
        if path.name in EXCLUDED_NAMES or path.suffix.lower() in EXCLUDED_SUFFIXES:
            continue
        if path.name == "ARTIFACT_SHA256SUMS.txt":
            continue
        if path.stem.removesuffix("_zh") in OBSOLETE_GENERATED_STEMS:
            continue
        target = dst / rel
        if path.is_dir():
            target.mkdir(parents=True, exist_ok=True)
        elif len(rel.parts) >= 2 and rel.parts[:2] == ("paper_outputs", "release_v1"):
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(path, target)
        else:
            copy_text_or_binary(path, target)


INPUT_PATTERN = re.compile(r"\\(?:input|IfFileExists)\{generated/([^}]+)\}")
GRAPHIC_PATTERN = re.compile(
    r"\\(?:bestgraphic|includegraphics)(?:\[[^\]]*\])?\{figures/([^}]+)\}"
)


def copy_generated_dependencies(root_tex: Path, generated_src: Path, generated_dst: Path) -> None:
    pending = [root_tex]
    copied: set[str] = set()
    while pending:
        tex_path = pending.pop()
        text = tex_path.read_text(encoding="utf-8", errors="replace")
        for match in INPUT_PATTERN.finditer(text):
            name = match.group(1)
            if not Path(name).suffix:
                name += ".tex"
            if name in copied:
                continue
            source = generated_src / Path(name)
            if not source.exists():
                if "\\IfFileExists" in match.group(0):
                    continue
                raise FileNotFoundError(f"Missing generated LaTeX dependency: {source}")
            copied.add(name)
            target = generated_dst / Path(name)
            copy_text_or_binary(source, target)
            if source.suffix.lower() == ".tex":
                pending.append(source)


def copy_graphic_dependencies(
    root_tex: Path,
    generated_src: Path,
    figures_src: Path,
    figures_dst: Path,
) -> None:
    pending = [root_tex]
    visited: set[Path] = set()
    copied: set[str] = set()
    while pending:
        tex_path = pending.pop().resolve()
        if tex_path in visited:
            continue
        visited.add(tex_path)
        text = tex_path.read_text(encoding="utf-8", errors="replace")
        for match in INPUT_PATTERN.finditer(text):
            name = match.group(1)
            if not Path(name).suffix:
                name += ".tex"
            source = generated_src / Path(name)
            if source.exists() and source.suffix.lower() == ".tex":
                pending.append(source)
        for match in GRAPHIC_PATTERN.finditer(text):
            raw_name = match.group(1)
            raw_path = Path(raw_name)
            candidates = (
                [figures_src / raw_path]
                if raw_path.suffix
                else [
                    figures_src / raw_path.with_suffix(".pdf"),
                    figures_src / raw_path.with_suffix(".png"),
                ]
            )
            existing = [candidate for candidate in candidates if candidate.exists()]
            if not existing:
                raise FileNotFoundError(
                    f"Missing graphic dependency for {root_tex.name}: figures/{raw_name}"
                )
            for source in existing:
                key = source.relative_to(figures_src).as_posix()
                if key in copied:
                    continue
                copied.add(key)
                copy_text_or_binary(source, figures_dst / source.relative_to(figures_src))


def write_text_file(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8", newline="\n")


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_package_checksums(package_dir: Path) -> None:
    rows = []
    for path in sorted(p for p in package_dir.rglob("*") if p.is_file()):
        rel = path.relative_to(package_dir).as_posix()
        if rel == "checksums.sha256":
            continue
        rows.append(f"{sha256_file(path)}  {rel}")
    write_text_file(package_dir / "checksums.sha256", "\n".join(rows) + "\n")


def write_cross_document_release_report(package_dir: Path) -> None:
    main_root = package_dir / "manuscript"
    supplement_root = package_dir / "supplementary" / "manuscript"
    sync_path = main_root / "generated" / "release_sync.tex"
    sync_text = sync_path.read_text(encoding="utf-8-sig")
    release_match = re.search(
        r"\\EvidenceReleaseID\}\{([^}]+)\}",
        sync_text,
    )
    protocol_match = re.search(
        r"\\EvidenceProtocolVersion\}\{([^}]+)\}",
        sync_text,
    )
    report = {
        "layout": "submission-package",
        "release_id": release_match.group(1) if release_match else "",
        "protocol_version": protocol_match.group(1) if protocol_match else "",
        "main_source_sha256": sha256_file(
            main_root / "main_revised.tex"
        ),
        "supplement_source_sha256": sha256_file(
            supplement_root / "supplement_en.tex"
        ),
        "sync_file_sha256": sha256_file(sync_path),
    }
    write_text_file(
        package_dir / "cross_document_release.json",
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    )


def write_submission_readmes(package_dir: Path) -> None:
    write_text_file(
        package_dir / "README_SUBMISSION_PACKAGE.md",
        "# Submission Package\n\n"
        "This package contains the English manuscript, all figures required by the LaTeX source, "
        "references, and a clean supplementary replication package. ZIP entries are written with "
        "forward-slash POSIX paths so the archive is portable across Windows and Linux.\n\n"
        "## Compile Manuscript\n\n"
        "From `manuscript/`, run:\n\n"
        "```bash\n"
        "latexmk -xelatex -interaction=nonstopmode -halt-on-error main_revised.tex\n"
        "```\n\n"
        "The source expects figures under `manuscript/figures/`, which are included in this package.\n\n"
        "Generated numeric macros and booktabs tables required by the source are included under "
        "`manuscript/generated/`.\n\n"
        "## Supplementary Material\n\n"
        "The scientific Supplementary Information is under `supplementary/manuscript/`. From that "
        "directory, compile `supplement_en.tex` with the same XeLaTeX command.\n\n"
        "The supplementary replication package is under `supplementary/replication_package/`. "
        "It excludes raw NASA C-MAPSS data according to redistribution constraints, but includes "
        "stored formal results, generated tables, generated figures, configuration files, scripts, "
        "and a replication manifest. Before journal submission, complete the unresolved public-artifact "
        "fields in the manuscript and `PUBLIC_ARTIFACT_RELEASE_CHECKLIST.md` with the real repository "
        "URL, release tag or commit hash, and DOI if one is minted. The package-root "
        "`checksums.sha256` is the single checksum authority for this delivery. It authenticates a "
        "fresh extraction; stored-result regeneration intentionally rewrites generated evidence, so run "
        "that route in a disposable copy and re-extract the ZIP before auditing the deposited bytes again.\n",
    )
    write_text_file(
        package_dir / "supplementary" / "README_SUPPLEMENTARY.md",
        "# Supplementary Material\n\n"
        "The replication package contains stored formal C-MAPSS results and reproduction scripts. "
        "Use `reproduce_minimal.sh --stored-results --quick` on Linux/macOS or "
        "`reproduce_minimal.ps1 -StoredResults -Quick` on Windows to validate stored artifacts "
        "without retraining. The route checks the delivery checksum before writing regenerated outputs; "
        "use a disposable extraction when preservation of the original checksum state is required. Full "
        "retraining requires downloading the public NASA C-MAPSS files "
        "into `data/raw/` first. Public release metadata is tracked in "
        "`replication_package/PUBLIC_ARTIFACT_RELEASE_CHECKLIST.md`.\n",
    )


def create_zip(package_dir: Path, zip_path: Path) -> None:
    if zip_path.exists():
        zip_path.unlink()
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for path in sorted(p for p in package_dir.rglob("*") if p.is_file()):
            rel = path.relative_to(package_dir).as_posix()
            zf.write(path, arcname=rel)


def run_release_authority(package_dir: Path) -> None:
    command = [
        sys.executable,
        "run_release_authority.py",
        "--allow-pending-metadata",
        "--in-place",
    ]
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    completed = subprocess.run(
        command,
        cwd=package_dir,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    if completed.returncode != 0:
        raise RuntimeError(
            "Release authority gate failed before packaging:\n"
            + (completed.stdout + completed.stderr)
        )


def write_archive_release_report(
    package_dir: Path,
    zip_path: Path,
) -> Path:
    authority_path = (
        package_dir
        / "supplementary"
        / "replication_package"
        / "reports"
        / "release_authority_gate.json"
    )
    authority = json.loads(
        authority_path.read_text(encoding="utf-8-sig")
    )
    report = {
        **authority,
        "authority_scope": "archive",
        "archive_digest_bound": True,
        "archive_path": zip_path.name,
        "archive_sha256": sha256_file(zip_path),
        "archive_size_bytes": zip_path.stat().st_size,
    }
    output = zip_path.with_name(
        zip_path.stem + "_release_authority.json"
    )
    write_text_file(
        output,
        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
    )
    verified = json.loads(output.read_text(encoding="utf-8-sig"))
    if verified["archive_sha256"] != sha256_file(zip_path):
        raise RuntimeError("Archive release authority digest verification failed.")
    return output


def validate_zip(zip_path: Path) -> None:
    with zipfile.ZipFile(zip_path) as zf:
        names = zf.namelist()
        crlf_shell_entries = [
            name for name in names if name.endswith(".sh") and b"\r\n" in zf.read(name)
        ]
        profile = (
            json.loads(zf.read("delivery_profile.json").decode("utf-8"))["profile"]
            if "delivery_profile.json" in names
            else "full"
        )
    backslash_entries = [name for name in names if "\\" in name]
    if backslash_entries:
        raise RuntimeError(f"ZIP contains backslash paths: {backslash_entries[:5]}")
    revision_round_entries = [
        name for name in names if re.search(r"round[23](?!\d)", name.lower())
    ]
    if revision_round_entries:
        raise RuntimeError(
            f"ZIP contains revision-round paths: {revision_round_entries[:5]}"
        )
    if crlf_shell_entries:
        raise RuntimeError(f"ZIP contains CRLF shell scripts: {crlf_shell_entries[:5]}")
    required = [
        "run_release_authority.py",
        "manuscript/main_revised.tex",
        "manuscript/references.bib",
        "manuscript/generated/numbers.tex",
        "manuscript/generated/release_sync.tex",
        "manuscript/generated/results_section_en.tex",
        "manuscript/figures/fig_method_architecture.pdf",
        "manuscript/figures/fig_primary_fixed_task_diagnostic_envelope.pdf",
        "manuscript/figures/fig_condition_normalization_controls.pdf",
        "supplementary/replication_package/README_REPLICATION.md",
        "supplementary/manuscript/supplement_en.tex",
        "supplementary/manuscript/generated/numbers.tex",
        "supplementary/manuscript/generated/release_sync.tex",
        "supplementary/manuscript/generated/algorithm_training_inference.tex",
    ]
    if profile == "full":
        required.append("supplementary/replication_package/reproduce_minimal.sh")
    elif profile == "lightweight-release-only":
        required.extend(
            [
                "supplementary/replication_package/reproduce_release_only.sh",
                "supplementary/replication_package/lightweight_manifest.csv",
            ]
        )
    else:
        raise RuntimeError(f"ZIP has unknown delivery profile: {profile!r}")
    missing = [name for name in required if name not in names]
    if missing:
        raise RuntimeError(f"ZIP is missing required files: {missing}")


def create_lightweight_delivery(package_dir: Path) -> tuple[Path, Path]:
    lightweight_dir = WORKSPACE_ROOT / "lightweight_submission_package"
    lightweight_zip = WORKSPACE_ROOT / "cmapss_benchmark_governance_lightweight.zip"
    reset_dir(lightweight_dir)
    replication_prefix = Path("supplementary/replication_package")
    allowed_replication_roots = {
        "configs",
        "docs",
        "paper_outputs",
        "scripts",
        "src",
    }
    allowed_replication_files = {
        "CITATION.cff",
        "Dockerfile",
        "LICENSE",
        "PUBLIC_ARTIFACT_RELEASE_CHECKLIST.md",
        "README_REPLICATION.md",
        "artifact_metadata.json",
        "author_metadata.json",
        "environment.yml",
        "pyproject.toml",
        "reproduce_release_only.ps1",
        "reproduce_release_only.sh",
        "requirements.txt",
        "tex_environment.lock",
    }
    for source in package_dir.rglob("*"):
        if not source.is_file():
            continue
        relative = source.relative_to(package_dir)
        if "__pycache__" in relative.parts or source.suffix.lower() == ".pyc":
            continue
        relative_posix = relative.as_posix()
        include = (
            relative.parts[0] == "manuscript"
            or relative.parts[:2] == ("supplementary", "manuscript")
            or len(relative.parts) == 1
        )
        if relative.is_relative_to(replication_prefix):
            rep_relative = relative.relative_to(replication_prefix)
            if rep_relative.name in allowed_replication_files and len(rep_relative.parts) == 1:
                include = True
            elif rep_relative.parts and rep_relative.parts[0] in allowed_replication_roots:
                include = True
                if rep_relative.parts[0] == "paper_outputs" and rep_relative.parts[1:2] != ("release_v1",):
                    include = False
        if relative_posix == "checksums.sha256":
            include = False
        if include:
            target = lightweight_dir / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, target)
    write_text_file(
        lightweight_dir / "README_LIGHTWEIGHT_DELIVERY.md",
        "# Lightweight Submission Delivery\n\n"
        "This delivery contains the complete manuscript and Supplementary Information, the semantic "
        "`release_v1` evidence directory, source code required by its audit, configurations, and a "
        "release-only entry point. Large checkpoints, the 260-run result tree, full-result manifests, "
        "and full-suite tests are intentionally absent. From `supplementary/replication_package`, run "
        "`python scripts/verify_lightweight_package.py` or `./reproduce_release_only.sh`. Full table "
        "regeneration and retraining require the full archive.\n",
    )
    write_text_file(
        lightweight_dir / "supplementary" / "README_SUPPLEMENTARY.md",
        "# Supplementary Material: Lightweight Profile\n\n"
        "This delivery supports manuscript compilation and immutable `release_v1` evidence audit only. "
        "It intentionally does not support `--stored-results`, table regeneration, or retraining. Run "
        "`reproduce_release_only.sh` (Linux/macOS) or `reproduce_release_only.ps1` (Windows) from "
        "`supplementary/replication_package/`.\n",
    )
    replication = lightweight_dir / replication_prefix
    write_text_file(
        replication / "README_REPLICATION.md",
        "# Release-Only Replication Profile\n\n"
        "Capability: verify the semantic `paper_outputs/release_v1` evidence and its hashes. "
        "The package deliberately excludes the training-result tree and therefore must not be used "
        "with the full stored-results commands. Run `python scripts/verify_lightweight_package.py` and "
        "`python scripts/run_lightweight_tests.py`; full-only tests are reported as explicit skips.\n\n"
        "## Stable exhibit navigation\n\n"
        "The compiled manuscript and Supplementary Information in this delivery retain stable LaTeX "
        "labels. The full source-to-generator index is intentionally available only in the full archive.\n\n"
        "- Main estimand hierarchy: `tab:estimand-hierarchy`\n"
        "- Fixed-task paired-effect display: `fig:primary-fixed-diagnostic`\n"
        "- Cross-model run-quality audit: `tab:run-quality-robust-summary`\n",
    )
    profile = {
        "schema_version": "1.0",
        "profile": "lightweight-release-only",
        "supports_release_audit": True,
        "supports_stored_result_regeneration": False,
        "supports_retraining": False,
    }
    write_text_file(lightweight_dir / "delivery_profile.json", json.dumps(profile, indent=2) + "\n")
    write_text_file(replication / "delivery_profile.json", json.dumps(profile, indent=2) + "\n")
    release_root = replication / "paper_outputs" / "release_v1"
    manifest_path = replication / "lightweight_manifest.csv"
    with manifest_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "sha256", "capability"])
        writer.writeheader()
        for path in sorted(item for item in release_root.rglob("*") if item.is_file()):
            writer.writerow(
                {
                    "path": path.relative_to(replication).as_posix(),
                    "sha256": sha256_file(path),
                    "capability": "release-evidence-audit",
                }
            )
    run_release_authority(lightweight_dir)
    write_package_checksums(lightweight_dir)
    create_zip(lightweight_dir, lightweight_zip)
    validate_zip(lightweight_zip)
    write_archive_release_report(lightweight_dir, lightweight_zip)
    return lightweight_dir, lightweight_zip


def generate_submission_package(
    output_dir: str | Path | None = None,
    zip_path: str | Path | None = None,
    experiment_prefix: str = "paper_main_v3_seed",
) -> tuple[Path, Path]:
    package_dir = Path(output_dir).resolve() if output_dir else WORKSPACE_ROOT / "submission_package"
    archive_path = (
        Path(zip_path).resolve()
        if zip_path
        else WORKSPACE_ROOT / "cmapss_benchmark_governance_full_submission.zip"
    )
    reset_dir(package_dir)
    write_text_file(
        package_dir / "delivery_profile.json",
        json.dumps(
            {
                "schema_version": "1.0",
                "profile": "full",
                "supports_release_audit": True,
                "supports_stored_result_regeneration": True,
                "supports_retraining": True,
            },
            indent=2,
        )
        + "\n",
    )

    manuscript_dir = find_manuscript_dir()
    manuscript_out = package_dir / "manuscript"
    manuscript_out.mkdir(parents=True, exist_ok=True)
    for name in ["main_revised.tex", "main_revised.pdf", "references.bib"]:
        src = manuscript_dir / name
        if src.exists():
            copy_text_or_binary(src, manuscript_out / name)
    figures_src = manuscript_dir / "figures"
    generated_src = manuscript_dir / "generated"
    if not figures_src.exists():
        raise FileNotFoundError(f"Missing figures directory: {figures_src}")
    if not generated_src.exists():
        raise FileNotFoundError(f"Missing generated evidence directory: {generated_src}")
    copy_graphic_dependencies(
        manuscript_dir / "main_revised.tex",
        generated_src,
        figures_src,
        manuscript_out / "figures",
    )
    copy_generated_dependencies(
        manuscript_dir / "main_revised.tex",
        generated_src,
        manuscript_out / "generated",
    )

    replication_dir = generate_replication_package(
        output_dir=PROJECT_ROOT / "replication_package",
        experiment_prefix=experiment_prefix,
    )
    shutil.copytree(
        replication_dir,
        package_dir / "supplementary" / "replication_package",
    )

    supplement_src = PROJECT_ROOT / "supplementary"
    supplement_out = package_dir / "supplementary" / "manuscript"
    supplement_out.mkdir(parents=True, exist_ok=True)
    for name in ["supplement_en.tex", "supplement_en.pdf"]:
        src = supplement_src / name
        if src.exists():
            copy_text_or_binary(src, supplement_out / name)
    supplement_generated = supplement_src / "generated"
    supplement_figures = supplement_src / "figures"
    if not supplement_generated.exists() or not supplement_figures.exists():
        raise FileNotFoundError("Supplementary generated evidence and figures must be synchronized before packaging.")
    copy_generated_dependencies(
        supplement_src / "supplement_en.tex",
        supplement_generated,
        supplement_out / "generated",
    )
    copy_graphic_dependencies(
        supplement_src / "supplement_en.tex",
        supplement_generated,
        supplement_figures,
        supplement_out / "figures",
    )
    write_cross_document_release_report(package_dir)
    write_submission_readmes(package_dir)
    shutil.copy2(
        PROJECT_ROOT / "scripts" / "verify_submission_package.py",
        package_dir / "verify_submission_package.py",
    )
    shutil.copy2(
        PROJECT_ROOT / "scripts" / "run_package_release_authority.py",
        package_dir / "run_release_authority.py",
    )
    run_release_authority(package_dir)
    write_package_checksums(package_dir)
    create_zip(package_dir, archive_path)
    validate_zip(archive_path)
    write_archive_release_report(package_dir, archive_path)
    lightweight_dir, lightweight_zip = create_lightweight_delivery(package_dir)
    print(
        f"SUBMISSION_PACKAGE_PASS dir={package_dir} zip={archive_path} "
        f"lightweight_dir={lightweight_dir} lightweight_zip={lightweight_zip}"
    )
    return package_dir, archive_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Create a portable manuscript and supplementary submission package.")
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--zip-path", default=None)
    parser.add_argument("--experiment-prefix", default="paper_main_v3_seed")
    args = parser.parse_args()
    generate_submission_package(args.output_dir, args.zip_path, args.experiment_prefix)


if __name__ == "__main__":
    main()
