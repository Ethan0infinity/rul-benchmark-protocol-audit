from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from torch.utils.data import Dataset

from .augmentations import apply_augmentation_profile_with_mask
from .condition_norm import (
    ConditionAwareStandardScaler,
    ContinuousConditionCorrector,
    OfficialStateStandardScaler,
)


@dataclass
class WindowData:
    x: np.ndarray
    y: np.ndarray
    unit_ids: np.ndarray
    end_cycles: np.ndarray
    observed_mask: np.ndarray | None = None


def build_scaler(name: str):
    name = name.lower()
    if name == "standard":
        return StandardScaler()
    if name == "minmax":
        return MinMaxScaler()
    raise ValueError(f"Unknown scaler: {name}")


def fit_transform_frames(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    scaler_name: str = "standard",
) -> tuple[pd.DataFrame, pd.DataFrame, object]:
    scaler = build_scaler(scaler_name)
    train_out = train_df.copy()
    test_out = test_df.copy()
    scaler.fit(train_out[features])
    train_out[features] = scaler.transform(train_out[features])
    test_out[features] = scaler.transform(test_out[features])
    return train_out, test_out, scaler


def _resolve_condition_count(
    subset: str | None,
    condition_clusters: dict[str, int] | int | None,
) -> int:
    if isinstance(condition_clusters, dict):
        if subset is None:
            return 1
        return int(condition_clusters.get(subset.upper(), condition_clusters.get(subset, 1)))
    if condition_clusters is not None:
        return int(condition_clusters)
    if subset and subset.upper() in {"FD002", "FD004"}:
        return 6
    return 1


def fit_transform_train_val_test(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    features: list[str],
    scaler_name: str = "standard",
    subset: str | None = None,
    condition_clusters: dict[str, int] | int | None = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, object]:
    scaler_name = scaler_name.lower()
    train_out = train_df.copy()
    val_out = val_df.copy()
    test_out = test_df.copy()

    if scaler_name in {"condition_standard", "condition_aware_standard", "condition_aware_standardization"}:
        scaler = ConditionAwareStandardScaler(
            n_conditions=_resolve_condition_count(subset, condition_clusters),
            feature_names=features,
            random_state=random_state,
        )
        train_out = scaler.fit_transform(train_out)
        val_out = scaler.transform(val_out)
        test_out = scaler.transform(test_out)
        return train_out, val_out, test_out, scaler

    if scaler_name in {"official_state_standard", "official_state"}:
        scaler = OfficialStateStandardScaler(feature_names=features)
        train_out = scaler.fit_transform(train_out)
        val_out = scaler.transform(val_out)
        test_out = scaler.transform(test_out)
        return train_out, val_out, test_out, scaler

    if scaler_name in {"continuous_condition_standard", "continuous_condition_correction"}:
        scaler = ContinuousConditionCorrector(feature_names=features)
        train_out = scaler.fit_transform(train_out)
        val_out = scaler.transform(val_out)
        test_out = scaler.transform(test_out)
        return train_out, val_out, test_out, scaler

    scaler = build_scaler(scaler_name)
    scaler.fit(train_out[features])
    train_out[features] = scaler.transform(train_out[features])
    val_out[features] = scaler.transform(val_out[features])
    test_out[features] = scaler.transform(test_out[features])
    return train_out, val_out, test_out, scaler


def split_units(unit_ids: np.ndarray, validation_split: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    unique = np.array(sorted(np.unique(unit_ids)))
    rng.shuffle(unique)
    n_val = max(1, int(round(len(unique) * validation_split)))
    val = np.sort(unique[:n_val])
    train = np.sort(unique[n_val:])
    return train, val


def _pad_sequence(values: np.ndarray, target_len: int) -> np.ndarray:
    if values.shape[0] >= target_len:
        return values[-target_len:]
    pad_len = target_len - values.shape[0]
    pad = np.repeat(values[:1], pad_len, axis=0)
    return np.concatenate([pad, values], axis=0)


def make_windows(
    df: pd.DataFrame,
    features: list[str],
    window_size: int,
    stride: int = 1,
    units: np.ndarray | list[int] | None = None,
    last_only: bool = False,
    pad_short: bool = True,
) -> WindowData:
    x_list: list[np.ndarray] = []
    y_list: list[float] = []
    unit_list: list[int] = []
    cycle_list: list[int] = []
    selected_units = set(units) if units is not None else None

    for unit_id, group in df.groupby("unit_id"):
        if selected_units is not None and unit_id not in selected_units:
            continue
        group = group.sort_values("cycle")
        values = group[features].to_numpy(dtype=np.float32)
        labels = group["rul"].to_numpy(dtype=np.float32)
        cycles = group["cycle"].to_numpy(dtype=np.int64)

        if last_only:
            if values.shape[0] < window_size and not pad_short:
                continue
            x_list.append(_pad_sequence(values, window_size))
            y_list.append(float(labels[-1]))
            unit_list.append(int(unit_id))
            cycle_list.append(int(cycles[-1]))
            continue

        if values.shape[0] < window_size:
            if pad_short:
                x_list.append(_pad_sequence(values, window_size))
                y_list.append(float(labels[-1]))
                unit_list.append(int(unit_id))
                cycle_list.append(int(cycles[-1]))
            continue

        for end in range(window_size, values.shape[0] + 1, stride):
            start = end - window_size
            x_list.append(values[start:end])
            y_list.append(float(labels[end - 1]))
            unit_list.append(int(unit_id))
            cycle_list.append(int(cycles[end - 1]))

    if not x_list:
        raise ValueError("No windows were generated. Check window_size and input data.")

    return WindowData(
        x=np.stack(x_list).astype(np.float32),
        y=np.asarray(y_list, dtype=np.float32),
        unit_ids=np.asarray(unit_list, dtype=np.int64),
        end_cycles=np.asarray(cycle_list, dtype=np.int64),
    )


def make_simulated_last_windows(
    df: pd.DataFrame,
    features: list[str],
    window_size: int,
    units: np.ndarray | list[int] | None = None,
    seed: int = 42,
    min_rul: float = 1.0,
    max_rul: float | None = None,
    pad_short: bool = True,
) -> WindowData:
    """Build one engine-level validation window at a simulated test cutoff.

    C-MAPSS test engines stop before failure. Validation engines come from
    run-to-failure training trajectories, so their physical last cycle has
    RUL=0. This helper samples one earlier cutoff per validation engine to
    mimic the test protocol without using test labels.
    """
    rng = np.random.default_rng(seed)
    x_list: list[np.ndarray] = []
    y_list: list[float] = []
    unit_list: list[int] = []
    cycle_list: list[int] = []
    selected_units = set(units) if units is not None else None

    for unit_id, group in df.groupby("unit_id"):
        if selected_units is not None and unit_id not in selected_units:
            continue
        group = group.sort_values("cycle")
        values = group[features].to_numpy(dtype=np.float32)
        labels = group["rul"].to_numpy(dtype=np.float32)
        cycles = group["cycle"].to_numpy(dtype=np.int64)

        valid = np.arange(len(group))
        valid = valid[valid >= min(window_size - 1, len(group) - 1)]
        valid = valid[labels[valid] >= min_rul]
        if max_rul is not None:
            valid = valid[labels[valid] <= max_rul]
        if len(valid) == 0:
            fallback = len(group) - 1
            if values.shape[0] < window_size and not pad_short:
                continue
            chosen = fallback
        else:
            chosen = int(rng.choice(valid))

        end = chosen + 1
        if end < window_size and not pad_short:
            continue
        x_list.append(_pad_sequence(values[:end], window_size))
        y_list.append(float(labels[chosen]))
        unit_list.append(int(unit_id))
        cycle_list.append(int(cycles[chosen]))

    if not x_list:
        raise ValueError("No simulated last windows were generated. Check validation cutoff settings.")

    return WindowData(
        x=np.stack(x_list).astype(np.float32),
        y=np.asarray(y_list, dtype=np.float32),
        unit_ids=np.asarray(unit_list, dtype=np.int64),
        end_cycles=np.asarray(cycle_list, dtype=np.int64),
    )


def make_validation_endpoint_windows(
    df: pd.DataFrame,
    features: list[str],
    window_size: int,
    strategy: str,
    seed: int = 42,
    rul_cap: float = 125.0,
    pad_short: bool = True,
) -> WindowData:
    """Build engine-balanced multi-endpoint validation windows.

    ``fixed_multi`` uses declared target RUL values. ``critical_stratified``
    draws one endpoint from each declared RUL stratum, with three strata in
    the critical RUL<=30 region. Every engine contributes at most one window
    per target or stratum, preventing long trajectories from dominating.
    """
    strategy = strategy.lower()
    fixed_targets = (10.0, 20.0, 30.0, 60.0, 90.0, 120.0)
    strata = ((1.0, 10.0), (10.0, 20.0), (20.0, 30.0), (30.0, 60.0), (60.0, 90.0), (90.0, rul_cap))
    if strategy not in {"fixed_multi", "critical_stratified"}:
        raise ValueError(f"Unknown validation endpoint strategy: {strategy}")

    rng = np.random.default_rng(seed)
    x_list: list[np.ndarray] = []
    y_list: list[float] = []
    unit_list: list[int] = []
    cycle_list: list[int] = []
    for unit_id, group in df.groupby("unit_id", sort=True):
        group = group.sort_values("cycle")
        values = group[features].to_numpy(dtype=np.float32)
        labels = group["rul"].to_numpy(dtype=np.float32)
        cycles = group["cycle"].to_numpy(dtype=np.int64)
        eligible = np.arange(len(group))
        eligible = eligible[eligible >= min(window_size - 1, len(group) - 1)]
        eligible = eligible[(labels[eligible] >= 1.0) & (labels[eligible] <= rul_cap)]
        chosen_indices: list[int] = []
        if strategy == "fixed_multi":
            for target in fixed_targets:
                candidates = eligible[np.isclose(labels[eligible], target)]
                if len(candidates):
                    chosen_indices.append(int(candidates[-1]))
        else:
            for lower, upper in strata:
                if upper == rul_cap:
                    candidates = eligible[(labels[eligible] > lower) & (labels[eligible] <= upper)]
                else:
                    candidates = eligible[(labels[eligible] > lower) & (labels[eligible] <= upper)]
                if len(candidates):
                    chosen_indices.append(int(rng.choice(candidates)))

        for chosen in chosen_indices:
            end = chosen + 1
            if end < window_size and not pad_short:
                continue
            x_list.append(_pad_sequence(values[:end], window_size))
            y_list.append(float(labels[chosen]))
            unit_list.append(int(unit_id))
            cycle_list.append(int(cycles[chosen]))

    if not x_list:
        raise ValueError(f"No validation windows generated for strategy={strategy}.")
    return WindowData(
        x=np.stack(x_list).astype(np.float32),
        y=np.asarray(y_list, dtype=np.float32),
        unit_ids=np.asarray(unit_list, dtype=np.int64),
        end_cycles=np.asarray(cycle_list, dtype=np.int64),
    )


class RULWindowDataset(Dataset):
    def __init__(
        self,
        window_data: WindowData,
        sensor_indices: list[int] | None = None,
        sensor_dropout_prob: float = 0.0,
        sensor_dropout_fraction: float = 0.0,
        noise_std: float = 0.0,
        augmentation_profile: str = "none",
        missing_rate: float = 0.0,
        block_missing_rate: float = 0.0,
        drift_rate: float = 0.0,
        mechanism_apply_probability: float = 1.0,
        corrupted_window_probability: float = 1.0,
        append_missing_mask: bool = False,
        base_seed: int | None = None,
    ):
        self.x = window_data.x
        self.y = window_data.y
        self.observed_mask = window_data.observed_mask
        self.sensor_indices = sensor_indices or []
        self.sensor_dropout_prob = sensor_dropout_prob
        self.sensor_dropout_fraction = sensor_dropout_fraction
        self.noise_std = noise_std
        self.augmentation_profile = augmentation_profile
        self.missing_rate = missing_rate
        self.block_missing_rate = block_missing_rate
        self.drift_rate = drift_rate
        self.mechanism_apply_probability = mechanism_apply_probability
        self.corrupted_window_probability = corrupted_window_probability
        self.append_missing_mask = append_missing_mask
        self.base_seed = base_seed
        self.epoch = 0

    def __len__(self) -> int:
        return self.x.shape[0]

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __getitem__(self, idx: int):
        x = self.x[idx].copy()
        y = self.y[idx]
        seed = None if self.base_seed is None else int(self.base_seed + self.epoch * len(self) + idx)
        rng = np.random.default_rng(seed)
        if self.observed_mask is None:
            mask = np.ones((x.shape[0], len(self.sensor_indices)), dtype=np.float32)
        else:
            stored_mask = self.observed_mask[idx].copy().astype(np.float32)
            if stored_mask.shape[-1] == x.shape[-1] and self.sensor_indices:
                mask = stored_mask[:, self.sensor_indices]
            else:
                mask = stored_mask
        profile_enabled = self.augmentation_profile not in {"none", "clean"}
        if self.noise_std > 0 and self.sensor_indices and not profile_enabled:
            x[:, self.sensor_indices] += rng.normal(0.0, self.noise_std, size=x[:, self.sensor_indices].shape)
        if self.sensor_dropout_prob > 0 and self.sensor_indices and not profile_enabled and rng.random() < self.sensor_dropout_prob:
            k = max(1, int(round(len(self.sensor_indices) * self.sensor_dropout_fraction)))
            chosen_positions = rng.choice(len(self.sensor_indices), size=min(k, len(self.sensor_indices)), replace=False)
            chosen = [self.sensor_indices[int(pos)] for pos in chosen_positions]
            x[:, chosen] = 0.0
            mask[:, chosen_positions] = 0.0
        if profile_enabled:
            augmented, profile_mask = apply_augmentation_profile_with_mask(
                x[None, ...],
                profile_name=self.augmentation_profile,
                sensor_indices=self.sensor_indices,
                noise_std=self.noise_std,
                missing_rate=self.missing_rate,
                block_missing_rate=self.block_missing_rate,
                drift_rate=self.drift_rate,
                mechanism_apply_probability=self.mechanism_apply_probability,
                corrupted_window_probability=self.corrupted_window_probability,
                seed=seed,
            )
            x = augmented[0]
            mask *= profile_mask[0]
        if self.append_missing_mask:
            x = np.concatenate([x, mask], axis=-1)
        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32)
