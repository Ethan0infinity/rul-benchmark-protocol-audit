from __future__ import annotations

import argparse
import ast
import importlib.util
import json
from pathlib import Path
import re
import sys


PROJECT_ROOT = Path(__file__).resolve().parents[1]
AUTHORING_ONLY = {
    "scripts/finalize_review_revision.ps1",
    "scripts/run_review_revision_queue.ps1",
}
RELEASE_PATH_PREFIXES = {"benchmark", "configs", "docs", "eval_harness", "scripts", "src", "tests"}
COMMAND_PATH = re.compile(
    r"(?<![A-Za-z0-9_.-])((?:benchmark|configs|docs|eval_harness|scripts|src|tests)/"
    r"[A-Za-z0-9_.\-/]+\.(?:py|ps1|sh|json|ya?ml|md|toml))"
)


def _load_pipeline(root: Path):
    path = root / "scripts" / "run_stored_results_pipeline.py"
    spec = importlib.util.spec_from_file_location("release_pipeline_for_closure", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot load pipeline: {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _path_parts(node: ast.AST) -> list[str]:
    if isinstance(node, ast.BinOp) and isinstance(node.op, ast.Div):
        return [*_path_parts(node.left), *_path_parts(node.right)]
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return [node.value]
    return []


def declared_test_paths(root: Path) -> set[str]:
    paths: set[str] = set()
    tests = root / "tests"
    if not tests.exists():
        return paths
    for source in tests.rglob("*.py"):
        tree = ast.parse(source.read_text(encoding="utf-8-sig"), filename=str(source))
        for node in ast.walk(tree):
            if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Div):
                continue
            parts = _path_parts(node)
            if len(parts) < 2 or parts[0] not in RELEASE_PATH_PREFIXES:
                continue
            candidate = Path(*parts).as_posix()
            if Path(candidate).suffix.lower() in {".py", ".ps1", ".sh", ".json", ".yaml", ".yml", ".md", ".toml"}:
                paths.add(candidate)
    return paths


def declared_command_paths(root: Path) -> set[str]:
    paths: set[str] = set()
    pipeline = _load_pipeline(root)
    for stage in pipeline.stages():
        for argument in stage.command:
            normalized = str(argument).replace("\\", "/")
            if normalized.startswith(tuple(prefix + "/" for prefix in RELEASE_PATH_PREFIXES)):
                paths.add(normalized)
    for pattern in ("README*.md", "reproduce_*.sh", "reproduce_*.ps1"):
        for source in root.glob(pattern):
            text = source.read_text(encoding="utf-8-sig", errors="replace")
            paths.update(match.replace("\\", "/") for match in COMMAND_PATH.findall(text))
    return paths


def audit(root: Path) -> dict[str, object]:
    root = root.resolve()
    scan_root = root
    nested = root / "supplementary" / "replication_package"
    if not (scan_root / "scripts" / "run_stored_results_pipeline.py").is_file() and (
        nested / "scripts" / "run_stored_results_pipeline.py"
    ).is_file():
        scan_root = nested
    declared = declared_command_paths(scan_root) | declared_test_paths(scan_root)
    authoring_only = sorted(path for path in declared if path in AUTHORING_ONLY)
    required = sorted(path for path in declared if path not in AUTHORING_ONLY)
    missing = [path for path in required if not (scan_root / Path(path)).is_file()]
    return {
        "schema_version": "1.0",
        "package_root": ".",
        "release_scan_root": scan_root.relative_to(root).as_posix() if scan_root != root else ".",
        "declared_path_count": len(declared),
        "required_path_count": len(required),
        "authoring_only_references": authoring_only,
        "missing_required_paths": missing,
        "status": "PASS" if not missing else "FAIL",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Check static release-file dependency closure.")
    parser.add_argument("--package-root", type=Path, default=PROJECT_ROOT)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()
    root = args.package_root.resolve()
    report = audit(root)
    report_path = args.report or root / "reports" / "package_closure_gate.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    if report["status"] != "PASS":
        print(
            "PACKAGE_CLOSURE_FAIL missing="
            + str(len(report["missing_required_paths"]))
        )
        for path in report["missing_required_paths"]:
            print(f"- {path}")
        raise SystemExit(1)
    print(
        f"PACKAGE_CLOSURE_PASS required={report['required_path_count']} "
        f"authoring_only={len(report['authoring_only_references'])}"
    )


if __name__ == "__main__":
    main()
