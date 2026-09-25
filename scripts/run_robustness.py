from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from rul.experiments import evaluate_robustness, metrics_rows_to_csv


DEFAULT_EXPERIMENT_NAME = "paper_main_v3_seed42"
DEFAULT_SUBSETS = ["FD001", "FD004"]
DEFAULT_MODELS = ["gru", "tcn_gru", "rast_gru", "rast_gru_v2"]


def resolve_run_dirs(
    results_dir: str | Path,
    *,
    explicit_run_dirs: list[str | Path] | None = None,
    experiment_name: str | None = None,
    subsets: list[str] | None = None,
    models: list[str] | None = None,
) -> list[Path]:
    explicit = [Path(path) for path in (explicit_run_dirs or [])]
    if explicit:
        return explicit
    if not experiment_name:
        raise ValueError("Either --run-dir or --experiment-name must be provided.")
    root = Path(results_dir)
    subset_list = subsets or DEFAULT_SUBSETS
    model_list = models or DEFAULT_MODELS
    run_dirs = [root / experiment_name / subset / model for subset in subset_list for model in model_list]
    return [path for path in run_dirs if path.exists()]


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate noise and sensor-missing robustness for trained runs.")
    parser.add_argument("--run-dir", nargs="*", default=None, help="One or more explicit trained run directories.")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument(
        "--experiment-name",
        default=DEFAULT_EXPERIMENT_NAME,
        help=f"Resolve run dirs under results/<experiment>/<subset>/<model>. Defaults to {DEFAULT_EXPERIMENT_NAME}.",
    )
    parser.add_argument("--subsets", nargs="+", default=DEFAULT_SUBSETS)
    parser.add_argument("--models", nargs="+", default=DEFAULT_MODELS)
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    run_dirs = resolve_run_dirs(
        args.results_dir,
        explicit_run_dirs=args.run_dir,
        experiment_name=args.experiment_name,
        subsets=args.subsets,
        models=args.models,
    )
    if not run_dirs:
        raise FileNotFoundError(
            "No trained run directories found for robustness evaluation. "
            f"Expected results under {Path(args.results_dir) / args.experiment_name}; "
            "run scripts\\run_paper_experiments.py first or pass --run-dir."
        )

    all_rows = []
    print(
        f"[ROBUSTNESS START] experiment={args.experiment_name} "
        f"run_dirs={len(run_dirs)} subsets={','.join(args.subsets)} models={','.join(args.models)}",
        flush=True,
    )
    for run_dir in run_dirs:
        print(f"[ROBUSTNESS RUN] {run_dir}", flush=True)
        rows = evaluate_robustness(run_dir)
        for row in rows:
            row["run_dir"] = str(run_dir)
        out = Path(run_dir) / "robustness.csv"
        metrics_rows_to_csv(rows, out)
        all_rows.extend(rows)
        print(f"Saved robustness metrics to {out}")

    summary = Path(args.results_dir) / "robustness_runs.csv"
    metrics_rows_to_csv(all_rows, summary)
    print(f"Saved combined robustness metrics to {summary}")


if __name__ == "__main__":
    main()
