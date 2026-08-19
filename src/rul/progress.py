from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class RunProgressContext:
    experiment_name: str
    subset: str
    model: str
    run_dir: str | Path
    device: str
    total_epochs: int
    batch_size: int
    train_batches: int
    n_train_windows: int
    n_val_units: int
    n_test_units: int
    n_features: int
    n_sensors: int


def _seconds_text(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(int(seconds), 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m{sec:02d}s"


def _run_label(run_index: int | None, total_runs: int | None) -> str:
    if run_index is None or total_runs is None:
        return "[RUN START]"
    width = max(3, len(str(total_runs)))
    return f"[RUN {run_index:0{width}d}/{total_runs:0{width}d} START]"


def format_run_start(
    context: RunProgressContext,
    *,
    run_index: int | None = None,
    total_runs: int | None = None,
) -> str:
    lines = [
        "=" * 88,
        f"{_run_label(run_index, total_runs)} experiment={context.experiment_name}",
        f"subset={context.subset} model={context.model} device={context.device} epochs={context.total_epochs} batch_size={context.batch_size}",
        f"windows={context.n_train_windows} val_units={context.n_val_units} test_units={context.n_test_units} train_batches={context.train_batches}",
        f"features={context.n_features} sensors={context.n_sensors}",
        f"output={context.run_dir}",
        "=" * 88,
    ]
    return "\n".join(lines)


def format_epoch_status(
    *,
    epoch: int,
    total_epochs: int,
    train_loss: float,
    val_loss: float,
    best_val_loss: float,
    best_epoch: int,
    patience_wait: int,
    patience: int,
    elapsed_sec: float,
    learning_rate: float,
) -> str:
    return (
        f"[EPOCH {epoch:03d}/{total_epochs:03d}] "
        f"train_loss={train_loss:.4f} "
        f"val_loss={val_loss:.4f} "
        f"best={best_val_loss:.4f}@{best_epoch} "
        f"patience={patience_wait}/{patience} "
        f"lr={learning_rate:.3g} "
        f"elapsed={_seconds_text(elapsed_sec)}"
    )


def format_run_finish(metrics: dict[str, Any]) -> str:
    return (
        "[RUN FINISH] "
        f"experiment={metrics.get('experiment_name', 'default')} "
        f"subset={metrics.get('subset')} model={metrics.get('model')} "
        f"best_epoch={metrics.get('best_epoch')} "
        f"test_rmse={float(metrics.get('test_rmse', 0.0)):.4f} "
        f"test_mae={float(metrics.get('test_mae', 0.0)):.4f} "
        f"device={metrics.get('device')} "
        f"output={metrics.get('run_dir')}"
    )


def append_epoch_history_csv(path: str | Path, row: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists() and path.stat().st_size > 0
    fieldnames = list(row.keys())
    if exists:
        with path.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            existing_header = next(reader, None)
        if existing_header:
            fieldnames = existing_header
    with path.open("a", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def write_progress_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = dict(payload)
    data["updated_at"] = datetime.now().isoformat(timespec="seconds")
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
