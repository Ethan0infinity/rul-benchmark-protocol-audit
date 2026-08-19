from __future__ import annotations

import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def save_json(data: dict[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8-sig") as f:
        return json.load(f)


def get_device(name: str = "auto") -> torch.device:
    if name == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


@torch.no_grad()
def measure_inference_time(
    model: torch.nn.Module,
    sample: torch.Tensor,
    device: torch.device,
    warmup: int = 10,
    repeats: int = 50,
) -> float:
    """Return average milliseconds per batch."""
    model.eval()
    sample = sample.to(device)
    for _ in range(warmup):
        _ = model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(repeats):
        _ = model(sample)
    if device.type == "cuda":
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return elapsed * 1000.0 / max(1, repeats)


@torch.no_grad()
def profile_forward_flops(model: torch.nn.Module, sample: torch.Tensor) -> int:
    """Profile supported batch-1 forward FLOPs on CPU.

    PyTorch does not assign FLOPs to every operator, so this is an auditable
    profiler-supported count rather than a hand-computed architecture claim.
    """
    cpu_model = model.to(torch.device("cpu")).eval()
    cpu_sample = sample[:1].to(torch.device("cpu"))
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU], with_flops=True) as prof:
        _ = cpu_model(cpu_sample)
    return int(sum(int(event.flops or 0) for event in prof.key_averages()))


def timestamp() -> str:
    return time.strftime("%Y%m%d-%H%M%S")
