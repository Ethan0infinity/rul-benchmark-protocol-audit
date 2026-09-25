from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rul.experiments import _load_run
from run_formal_benchmark import FORMAL_BENCHMARK_SEEDS


ADDITIVE_GRIDS = {
    "Gaussian noise SD": (0.01, 0.03, 0.05, 0.10),
    "Linear-drift endpoint": (0.02, 0.05, 0.10),
    "Bias magnitude": (0.10, 0.25, 0.50),
    "Burst-noise SD": (0.10, 0.25, 0.50),
}


def training_references(results_dir: Path) -> dict[str, float]:
    iqr_values: list[float] = []
    local_values: list[float] = []
    for seed in FORMAL_BENCHMARK_SEEDS:
        run_dir = results_dir / f"paper_main_v3_seed{seed}" / "FD004" / "rast_gru_v2"
        _, _, prepared, _, _ = _load_run(run_dir)
        sensors = prepared.train.x[..., prepared.sensor_indices]
        quartiles = np.quantile(sensors, [0.25, 0.75], axis=(0, 1))
        iqr = quartiles[1] - quartiles[0]
        local = np.median(np.abs(np.diff(sensors, axis=1)), axis=(0, 1))
        iqr_values.extend(iqr[iqr > 1e-9].tolist())
        local_values.extend(local[local > 1e-9].tolist())
    return {
        "pooled_nonconstant_sensor_iqr_median": float(np.median(iqr_values)),
        "pooled_nonconstant_sensor_iqr_q10": float(np.quantile(iqr_values, 0.10)),
        "pooled_nonconstant_sensor_iqr_q90": float(np.quantile(iqr_values, 0.90)),
        "pooled_nonzero_local_abs_change_median": float(np.median(local_values)),
        "pooled_nonzero_local_abs_change_q10": float(np.quantile(local_values, 0.10)),
        "pooled_nonzero_local_abs_change_q90": float(np.quantile(local_values, 0.90)),
    }


def build_rows(reference: dict[str, float]) -> pd.DataFrame:
    rows = []
    iqr = reference["pooled_nonconstant_sensor_iqr_median"]
    local = reference["pooled_nonzero_local_abs_change_median"]
    for family, grid in ADDITIVE_GRIDS.items():
        for amplitude in grid:
            rows.append(
                {
                    "family": family,
                    "normalized_amplitude": amplitude,
                    "training_scale_definition": "within-condition training sensor SD",
                    "amplitude_per_median_training_iqr": amplitude / iqr,
                    "amplitude_per_median_local_abs_change": amplitude / local,
                }
            )
    return pd.DataFrame(rows)


def write_latex(frame: pd.DataFrame, reference: dict[str, float], output: Path) -> None:
    rows = []
    for row in frame.to_dict("records"):
        rows.append(
            f"{row['family']} & {row['normalized_amplitude']:.2f} & "
            f"{row['amplitude_per_median_training_iqr']:.3f} & "
            f"{row['amplitude_per_median_local_abs_change']:.3f} \\\\"
        )
    text = "\n".join(
        [
            r"\begin{table}[!t]",
            r"\centering",
            r"\scriptsize",
            r"\caption{Calibration of additive normalized-space perturbations. One amplitude unit equals one training-fitted within-condition sensor standard deviation. Ratios use pooled nonconstant selected-sensor summaries over five FD004 training cores; they describe algorithmic scale and are not mappings to physical sensor units.}",
            r"\label{tab:stress-amplitude-calibration}",
            r"\begin{tabular}{lrrr}",
            r"\toprule",
            r"Family & Amplitude & / median training IQR & / median local $|\Delta x|$ \\",
            r"\midrule",
            *rows,
            r"\bottomrule",
            r"\end{tabular}",
            (
                r"\begin{minipage}{0.96\linewidth}\footnotesize "
                f"The pooled median training IQR is {reference['pooled_nonconstant_sensor_iqr_median']:.3f} "
                f"(10th--90th percentile {reference['pooled_nonconstant_sensor_iqr_q10']:.3f}--"
                f"{reference['pooled_nonconstant_sensor_iqr_q90']:.3f}); the pooled median nonzero "
                f"within-window absolute change is {reference['pooled_nonzero_local_abs_change_median']:.3f} "
                f"({reference['pooled_nonzero_local_abs_change_q10']:.3f}--"
                f"{reference['pooled_nonzero_local_abs_change_q90']:.3f}). Missingness and stuck-at "
                r"levels are channel/time fractions rather than additive amplitudes.\end{minipage}"
            ),
            r"\end{table}",
            "",
        ]
    )
    output.write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Calibrate normalized-space stress amplitudes.")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "paper_outputs" / "analysis_working"))
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    reference = training_references(Path(args.results_dir))
    frame = build_rows(reference)
    frame.to_csv(output / "stress_amplitude_calibration.csv", index=False)
    pd.DataFrame([reference]).to_csv(output / "stress_training_scale_reference.csv", index=False)
    write_latex(frame, reference, output / "table_stress_amplitude_calibration.tex")
    print(
        "STRESS_AMPLITUDE_CALIBRATION_READY "
        f"rows={len(frame)} median_iqr={reference['pooled_nonconstant_sensor_iqr_median']:.3f} "
        f"median_local={reference['pooled_nonzero_local_abs_change_median']:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
