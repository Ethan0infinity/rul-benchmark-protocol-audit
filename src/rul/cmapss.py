from __future__ import annotations

from pathlib import Path

import pandas as pd


SETTINGS = [f"setting_{i}" for i in range(1, 4)]
SENSORS = [f"s{i}" for i in range(1, 22)]
COLUMNS = ["unit_id", "cycle", *SETTINGS, *SENSORS]


def find_cmapss_file(data_root: str | Path, filename: str) -> Path:
    root = Path(data_root)
    direct = root / filename
    if direct.exists():
        return direct
    matches = list(root.rglob(filename)) if root.exists() else []
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Cannot find {filename} under {root.resolve()}")


def read_cmapss_table(data_root: str | Path, filename: str) -> pd.DataFrame:
    path = find_cmapss_file(data_root, filename)
    df = pd.read_csv(path, sep=r"\s+", header=None, names=COLUMNS, engine="python")
    return df.dropna(axis=1, how="all")


def read_rul_file(data_root: str | Path, subset: str) -> pd.Series:
    path = find_cmapss_file(data_root, f"RUL_{subset}.txt")
    values = pd.read_csv(path, sep=r"\s+", header=None, engine="python").iloc[:, 0]
    values.index = range(1, len(values) + 1)
    return values.astype(float)


def add_train_rul(df: pd.DataFrame, rul_cap: int | None = 125) -> pd.DataFrame:
    out = df.copy()
    max_cycle = out.groupby("unit_id")["cycle"].transform("max")
    out["rul"] = max_cycle - out["cycle"]
    if rul_cap is not None:
        out["rul"] = out["rul"].clip(upper=rul_cap)
    return out


def add_test_rul(df: pd.DataFrame, final_rul: pd.Series, rul_cap: int | None = 125) -> pd.DataFrame:
    out = df.copy()
    max_cycle = out.groupby("unit_id")["cycle"].transform("max")
    final_rul_by_row = out["unit_id"].map(final_rul)
    out["rul"] = final_rul_by_row + (max_cycle - out["cycle"])
    if rul_cap is not None:
        out["rul"] = out["rul"].clip(upper=rul_cap)
    return out


def load_subset(data_root: str | Path, subset: str, rul_cap: int | None = 125) -> tuple[pd.DataFrame, pd.DataFrame]:
    subset = subset.upper()
    train = read_cmapss_table(data_root, f"train_{subset}.txt")
    test = read_cmapss_table(data_root, f"test_{subset}.txt")
    final_rul = read_rul_file(data_root, subset)
    return add_train_rul(train, rul_cap), add_test_rul(test, final_rul, rul_cap)

