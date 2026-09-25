from __future__ import annotations

import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


DEFAULT_PATTERNS = [
    "paper_outputs/tables/*.csv",
    "paper_outputs/tables/*.md",
    "paper_outputs/tables/*.json",
    "paper_outputs/result_brief.md",
]


def collect_generated_outputs(project_root: str | Path) -> list[Path]:
    root = Path(project_root)
    targets: list[Path] = []
    for pattern in DEFAULT_PATTERNS:
        targets.extend(root.glob(pattern))
    return sorted({path.resolve() for path in targets if path.is_file()})


def _count_formal_metrics(project_root: Path) -> int:
    results_dir = project_root / "results"
    if not results_dir.exists():
        return 0
    return sum(1 for path in results_dir.rglob("metrics.json") if "archive_" not in path.parts)


def _count_progress_files(project_root: Path) -> int:
    results_dir = project_root / "results"
    if not results_dir.exists():
        return 0
    return sum(1 for path in results_dir.rglob("progress.json") if "archive_" not in path.parts)


def _print_empty_guidance(project_root: Path, expected_count: int) -> None:
    formal_metrics = _count_formal_metrics(project_root)
    progress_files = _count_progress_files(project_root)
    print(
        "[CLEAN INFO] paper_outputs has no generated tables/brief yet. "
        "This is normal before build_paper_tables.py has produced tables."
    )
    print(f"[STATUS] formal_metrics={formal_metrics}/{expected_count} progress_files={progress_files}")
    if formal_metrics < expected_count:
        print("[NEXT] If run_paper_experiments.py is still running, wait until metrics.json count reaches the expected count.")
        print(
            "[NEXT] Then run: python "
            "scripts\\check_result_consistency.py --experiment-name paper_main_v3_seed42 "
            f"--expected-count {expected_count} --min-epochs 30 --require-protocol-v2"
        )
    else:
        print("[NEXT] Results appear complete enough to try rebuilding tables.")
    print(
        "[NEXT] Rebuild tables with: python "
        "scripts\\build_paper_tables.py --experiment-name paper_main_v3_seed42"
    )


def clean_generated_outputs(
    project_root: str | Path = PROJECT_ROOT,
    *,
    dry_run: bool = False,
    expected_count: int = 24,
) -> dict[str, int]:
    root = Path(project_root).resolve()
    targets = collect_generated_outputs(root)

    if not targets:
        print("[CLEAN] no generated paper output files found")
        _print_empty_guidance(root, expected_count)
        return {"target_count": 0, "removed_count": 0}

    removed = 0
    for path in targets:
        rel = path.relative_to(root)
        if dry_run:
            print(f"[DRY-RUN] would remove {rel}")
        else:
            path.unlink()
            removed += 1
            print(f"[REMOVED] {rel}")
    print(f"[CLEAN] files={'would_remove' if dry_run else 'removed'} count={len(targets)}")
    print(
        "[NEXT] Rebuild tables with: python "
        "scripts\\build_paper_tables.py --experiment-name paper_main_v3_seed42"
    )
    return {"target_count": len(targets), "removed_count": 0 if dry_run else removed}


def main() -> None:
    parser = argparse.ArgumentParser(description="Remove generated paper output files before rebuilding tables.")
    parser.add_argument("--project-root", default=str(PROJECT_ROOT), help="Project root. Defaults to this repository root.")
    parser.add_argument("--dry-run", action="store_true", help="Show files that would be removed without deleting them.")
    parser.add_argument("--expected-count", type=int, default=24, help="Expected formal main-run metrics count for guidance.")
    args = parser.parse_args()
    clean_generated_outputs(args.project_root, dry_run=args.dry_run, expected_count=args.expected_count)


if __name__ == "__main__":
    main()
