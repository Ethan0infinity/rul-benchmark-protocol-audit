from __future__ import annotations

import argparse
import copy
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))

from rul.experiments import _load_run
from run_formal_benchmark import FORMAL_BENCHMARK_SEEDS


MODELS = ("rast_gru", "rast_gru_v2")
DISPLAY = {"rast_gru": "RAST-GRU", "rast_gru_v2": "OCM-Asym"}


def timed_forward(
    model: torch.nn.Module,
    sample: torch.Tensor,
    device: torch.device,
    *,
    warmup: int,
    repeats: int,
) -> np.ndarray:
    model.eval()
    sample = sample.to(device)
    times = []
    with torch.no_grad():
        for _ in range(warmup):
            model(sample)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        for _ in range(repeats):
            start = time.perf_counter_ns()
            model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            times.append((time.perf_counter_ns() - start) / 1e6)
    return np.asarray(times, dtype=float)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a disclosed repeated inference-timing protocol.")
    parser.add_argument("--results-dir", default=str(PROJECT_ROOT / "results"))
    parser.add_argument("--output-dir", default=str(PROJECT_ROOT / "paper_outputs" / "analysis_working"))
    parser.add_argument("--gpu-warmup", type=int, default=50)
    parser.add_argument("--gpu-repeats", type=int, default=200)
    parser.add_argument("--cpu-warmup", type=int, default=20)
    parser.add_argument("--cpu-repeats", type=int, default=100)
    args = parser.parse_args()
    output = Path(args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for seed in FORMAL_BENCHMARK_SEEDS:
        for model_name in MODELS:
            run_dir = Path(args.results_dir) / f"paper_main_v3_seed{seed}" / "FD004" / model_name
            config, metrics, prepared, model, device = _load_run(run_dir)
            sample_values = prepared.test.x
            if bool(config["data"].get("append_missing_mask", False)):
                observed_mask = np.ones(
                    (*sample_values.shape[:-1], len(prepared.sensor_indices)),
                    dtype=np.float32,
                )
                sample_values = np.concatenate([sample_values, observed_mask], axis=-1)
            sample = torch.as_tensor(sample_values, dtype=torch.float32)
            protocols = []
            for batch_size in (1, 128):
                times = timed_forward(
                    model,
                    sample[:batch_size],
                    device,
                    warmup=args.gpu_warmup,
                    repeats=args.gpu_repeats,
                )
                protocols.append(("GPU", batch_size, args.gpu_warmup, args.gpu_repeats, times))
            cpu_model = copy.deepcopy(model).to(torch.device("cpu"))
            cpu_times = timed_forward(
                cpu_model,
                sample[:1],
                torch.device("cpu"),
                warmup=args.cpu_warmup,
                repeats=args.cpu_repeats,
            )
            protocols.append(("CPU", 1, args.cpu_warmup, args.cpu_repeats, cpu_times))
            for hardware, batch_size, warmup, repeats, times in protocols:
                rows.append(
                    {
                        "model": model_name,
                        "display_name": DISPLAY[model_name],
                        "seed": seed,
                        "hardware": hardware,
                        "batch_size": batch_size,
                        "warmup_iterations": warmup,
                        "timed_iterations": repeats,
                        "latency_ms_mean": float(times.mean()),
                        "latency_ms_median": float(np.median(times)),
                        "latency_ms_sd": float(times.std(ddof=1)),
                        "latency_ms_p10": float(np.quantile(times, 0.10)),
                        "latency_ms_p90": float(np.quantile(times, 0.90)),
                        "parameters": int(metrics["parameters"]),
                        "preprocessing_included": False,
                        "host_device_transfer_included": False,
                        "cuda_synchronized_each_iteration": hardware == "GPU",
                        "torch_version": torch.__version__,
                        "gpu_name": torch.cuda.get_device_name(device) if hardware == "GPU" else "",
                        "cpu_platform": platform.processor(),
                    }
                )
    detail = pd.DataFrame(rows)
    detail.to_csv(output / "inference_latency_protocol_seed_level.csv", index=False)
    summary = (
        detail.groupby(["display_name", "hardware", "batch_size"], as_index=False)
        .agg(
            median_of_seed_medians_ms=("latency_ms_median", "median"),
            seed_median_min_ms=("latency_ms_median", "min"),
            seed_median_max_ms=("latency_ms_median", "max"),
            median_p10_ms=("latency_ms_p10", "median"),
            median_p90_ms=("latency_ms_p90", "median"),
            parameters=("parameters", "first"),
            seeds=("seed", "size"),
        )
    )
    summary.to_csv(output / "inference_latency_protocol_summary.csv", index=False)
    table_rows = []
    for row in summary.to_dict("records"):
        table_rows.append(
            f"{row['display_name']} & {row['parameters']:,} & {row['hardware']} & "
            f"{int(row['batch_size'])} & {row['median_of_seed_medians_ms']:.3f} & "
            f"[{row['seed_median_min_ms']:.3f}, {row['seed_median_max_ms']:.3f}] & "
            f"[{row['median_p10_ms']:.3f}, {row['median_p90_ms']:.3f}] \\\\"
        )
    latex = "\n".join(
        [
            r"\begin{table}[!t]",
            r"\centering",
            r"\scriptsize",
            r"\caption{Repeated FD004 forward-pass timing. GPU measurements use 50 warm-up and 200 synchronized timed iterations; CPU measurements use 20 warm-up and 100 timed iterations. Preprocessing and host--device transfer are excluded. Values summarize five independently trained checkpoints.}",
            r"\label{tab:inference-latency-protocol}",
            r"\begin{tabular}{lrlrrrr}",
            r"\toprule",
            r"Point & Params & Device & Batch & Median ms & Seed range & Median [p10,p90] \\",
            r"\midrule",
            *table_rows,
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
            "",
        ]
    )
    (output / "table_inference_latency_protocol.tex").write_text(latex, encoding="utf-8")
    print(f"INFERENCE_LATENCY_PROTOCOL_READY rows={len(detail)} summary={len(summary)}", flush=True)


if __name__ == "__main__":
    main()
