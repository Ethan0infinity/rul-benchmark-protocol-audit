from __future__ import annotations

import argparse
import csv
import math
import sys
from collections import defaultdict
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


MODEL_LABELS = {
    "gru": "GRU",
    "tcn_gru": "TCN-GRU",
    "rast_gru": "RAST-GRU",
    "rast_gru_v2": "OCM-MST-GRU",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze prediction-trajectory monotonicity from validation-window predictions."
    )
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--experiment-name", default="paper_main_v3_seed42")
    parser.add_argument("--subset", default="FD004")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["gru", "tcn_gru", "rast_gru_v2"],
        help="Model directories to analyze.",
    )
    parser.add_argument(
        "--output-dir",
        default="paper_outputs/tables",
        help="Directory for CSV/Markdown summary outputs.",
    )
    parser.add_argument(
        "--figure-dir",
        default="paper_outputs/tables",
        help="Directory for the trajectory figure.",
    )
    parser.add_argument(
        "--jump-tolerance",
        type=float,
        default=1.0,
        help="Predicted-RUL increase larger than this value counts as a monotonicity violation.",
    )
    parser.add_argument(
        "--example-unit",
        type=int,
        default=None,
        help="Validation engine id to plot. Defaults to the engine with the longest OCM-MST-GRU trajectory.",
    )
    return parser.parse_args()


def read_predictions(path: Path) -> list[dict[str, float]]:
    rows: list[dict[str, float]] = []
    with path.open("r", newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        for row in reader:
            rows.append(
                {
                    "unit_id": int(float(row["unit_id"])),
                    "end_cycle": int(float(row["end_cycle"])),
                    "true_rul": float(row["true_rul"]),
                    "pred_rul": float(row["pred_rul"]),
                }
            )
    return rows


def group_by_unit(rows: list[dict[str, float]]) -> dict[int, list[dict[str, float]]]:
    grouped: dict[int, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["unit_id"])].append(row)
    for unit_rows in grouped.values():
        unit_rows.sort(key=lambda item: item["end_cycle"])
    return dict(grouped)


def summarize_model(rows: list[dict[str, float]], tolerance: float) -> dict[str, float]:
    grouped = group_by_unit(rows)
    transitions = 0
    violations = 0
    positive_jumps: list[float] = []
    signed_jumps: list[float] = []
    engines_with_violation = 0
    late_transitions = 0
    late_violations = 0

    for unit_rows in grouped.values():
        unit_has_violation = False
        for prev, cur in zip(unit_rows, unit_rows[1:]):
            jump = cur["pred_rul"] - prev["pred_rul"]
            signed_jumps.append(jump)
            transitions += 1
            if prev["true_rul"] <= 30 or cur["true_rul"] <= 30:
                late_transitions += 1
            if jump > tolerance:
                violations += 1
                positive_jumps.append(jump)
                unit_has_violation = True
                if prev["true_rul"] <= 30 or cur["true_rul"] <= 30:
                    late_violations += 1
        if unit_has_violation:
            engines_with_violation += 1

    mean_abs_jump = sum(abs(value) for value in signed_jumps) / len(signed_jumps)
    mean_positive_jump = sum(positive_jumps) / len(positive_jumps) if positive_jumps else 0.0
    return {
        "unit_count": len(grouped),
        "trajectory_rows": len(rows),
        "transitions": transitions,
        "violation_rate": violations / transitions if transitions else math.nan,
        "late_zone_violation_rate": late_violations / late_transitions if late_transitions else math.nan,
        "engine_violation_rate": engines_with_violation / len(grouped) if grouped else math.nan,
        "mean_abs_step_change": mean_abs_jump,
        "mean_positive_violation": mean_positive_jump,
    }


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fieldnames = [
        "experiment_name",
        "subset",
        "model",
        "unit_count",
        "trajectory_rows",
        "transitions",
        "violation_rate",
        "late_zone_violation_rate",
        "engine_violation_rate",
        "mean_abs_step_change",
        "mean_positive_violation",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "| Model | Violation rate | Late-zone violation rate | Engine violation rate | Mean abs step change |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            "| {model} | {violation_rate:.3f} | {late_zone_violation_rate:.3f} | "
            "{engine_violation_rate:.3f} | {mean_abs_step_change:.3f} |".format(**row)
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def choose_example_unit(rows_by_model: dict[str, list[dict[str, float]]], preferred_model: str) -> int:
    grouped = group_by_unit(rows_by_model[preferred_model])
    return max(grouped.items(), key=lambda item: len(item[1]))[0]


def make_figure(
    path: Path,
    rows_by_model: dict[str, list[dict[str, float]]],
    example_unit: int,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.figure(figsize=(7.2, 4.2))
    true_plotted = False
    for model, rows in rows_by_model.items():
        unit_rows = [row for row in rows if int(row["unit_id"]) == example_unit]
        unit_rows.sort(key=lambda item: item["end_cycle"])
        if not unit_rows:
            continue
        x = [row["end_cycle"] for row in unit_rows]
        if not true_plotted:
            plt.plot(x, [row["true_rul"] for row in unit_rows], color="black", linewidth=2.0, label="True RUL")
            true_plotted = True
        plt.plot(
            x,
            [row["pred_rul"] for row in unit_rows],
            linewidth=1.8,
            label=MODEL_LABELS.get(model, model),
        )

    plt.xlabel("Cycle")
    plt.ylabel("RUL")
    plt.title(f"FD004 validation trajectory example: engine {example_unit}")
    plt.grid(alpha=0.25)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=220)
    plt.close()


def main() -> None:
    args = parse_args()
    results_root = PROJECT_ROOT / args.results_dir
    output_dir = PROJECT_ROOT / args.output_dir
    figure_dir = PROJECT_ROOT / args.figure_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    figure_dir.mkdir(parents=True, exist_ok=True)

    rows_by_model: dict[str, list[dict[str, float]]] = {}
    summary_rows: list[dict[str, object]] = []
    for model in args.models:
        pred_path = results_root / args.experiment_name / args.subset / model / "val_predictions.csv"
        if not pred_path.exists():
            raise FileNotFoundError(f"Missing validation predictions: {pred_path}")
        rows = read_predictions(pred_path)
        rows_by_model[model] = rows
        summary = summarize_model(rows, args.jump_tolerance)
        summary_rows.append(
            {
                "experiment_name": args.experiment_name,
                "subset": args.subset,
                "model": MODEL_LABELS.get(model, model),
                **summary,
            }
        )

    csv_path = output_dir / "table_trajectory_monotonicity.csv"
    md_path = output_dir / "table_trajectory_monotonicity.md"
    write_csv(csv_path, summary_rows)
    write_markdown(md_path, summary_rows)

    preferred = "rast_gru_v2" if "rast_gru_v2" in rows_by_model else args.models[-1]
    example_unit = args.example_unit or choose_example_unit(rows_by_model, preferred)
    figure_path = figure_dir / "fig_fd004_trajectory_example.png"
    make_figure(figure_path, rows_by_model, example_unit)

    print(f"TRAJECTORY_ANALYSIS_PASS csv={csv_path}")
    print(f"TRAJECTORY_ANALYSIS_PASS figure={figure_path}")
    print(f"TRAJECTORY_ANALYSIS_PASS example_unit={example_unit}")


if __name__ == "__main__":
    main()
