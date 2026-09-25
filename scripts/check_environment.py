from __future__ import annotations

import importlib
import platform
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def module_version(name: str) -> str:
    try:
        module = importlib.import_module(name)
        return getattr(module, "__version__", "installed")
    except Exception as exc:
        return f"ERROR: {exc}"


def nvidia_smi() -> str:
    try:
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"], text=True, timeout=20)
        return out.strip()
    except Exception as exc:
        return f"ERROR: {exc}"


def main() -> None:
    import torch

    lines = [
        "# Environment Report",
        "",
        f"Python executable: `{sys.executable}`",
        f"Python version: `{platform.python_version()}`",
        f"OS: `{platform.platform()}`",
        "",
        "## Packages",
        "",
    ]
    for name in ["torch", "numpy", "pandas", "sklearn", "matplotlib", "yaml", "scipy", "tqdm", "joblib"]:
        lines.append(f"- `{name}`: `{module_version(name)}`")
    lines.extend(
        [
            "",
            "## CUDA",
            "",
            f"- torch CUDA available: `{torch.cuda.is_available()}`",
            f"- torch CUDA version: `{torch.version.cuda}`",
            f"- CUDA device count: `{torch.cuda.device_count()}`",
        ]
    )
    if torch.cuda.is_available():
        lines.append(f"- CUDA device 0: `{torch.cuda.get_device_name(0)}`")
    lines.extend(["", "## nvidia-smi", "", "```text", nvidia_smi(), "```", ""])

    reports_dir = PROJECT_ROOT / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    report = reports_dir / "environment_report.md"
    report.write_text("\n".join(lines), encoding="utf-8")
    print(f"Environment report: {report}")
    print(f"torch_cuda_available={torch.cuda.is_available()}")
    print("ENVIRONMENT_CHECK_DONE")


if __name__ == "__main__":
    main()
