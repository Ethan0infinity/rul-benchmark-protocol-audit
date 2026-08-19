from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

from .preprocessing import WindowData


TRAIN_UNITS = [2, 5, 10, 16, 18]
VALIDATION_UNITS = [20]
TEST_UNITS = [11, 14, 15]


def _decode_names(values: np.ndarray) -> list[str]:
    names = []
    for value in np.asarray(values).reshape(-1):
        if isinstance(value, bytes):
            names.append(value.decode("utf-8"))
        else:
            names.append(str(value))
    return names


@dataclass
class SampledSplit:
    settings: np.ndarray
    sensors: np.ndarray
    rul: np.ndarray
    units: np.ndarray
    setting_names: list[str]
    sensor_names: list[str]


@dataclass
class NCMAPSSPrepared:
    train: WindowData
    val: WindowData
    test: WindowData
    feature_names: list[str]
    sensor_indices: list[int]
    metadata: dict[str, object]


class ConditionClusterNormalizer:
    def __init__(self, n_clusters: int = 6, random_state: int = 20260710):
        self.n_clusters = int(n_clusters)
        self.random_state = int(random_state)
        self.setting_scaler = StandardScaler()
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=self.random_state, n_init=20)
        self.sensor_means: np.ndarray | None = None
        self.sensor_stds: np.ndarray | None = None

    def fit(self, settings: np.ndarray, sensors: np.ndarray) -> "ConditionClusterNormalizer":
        scaled_settings = self.setting_scaler.fit_transform(settings)
        labels = self.kmeans.fit_predict(scaled_settings)
        means = []
        stds = []
        global_mean = sensors.mean(axis=0)
        global_std = sensors.std(axis=0)
        global_std[global_std < 1e-6] = 1.0
        for cluster in range(self.n_clusters):
            selected = sensors[labels == cluster]
            if len(selected) < 2:
                means.append(global_mean)
                stds.append(global_std)
                continue
            mean = selected.mean(axis=0)
            std = selected.std(axis=0)
            std[std < 1e-6] = 1.0
            means.append(mean)
            stds.append(std)
        self.sensor_means = np.asarray(means, dtype=np.float32)
        self.sensor_stds = np.asarray(stds, dtype=np.float32)
        return self

    def transform(self, settings: np.ndarray, sensors: np.ndarray) -> np.ndarray:
        if self.sensor_means is None or self.sensor_stds is None:
            raise RuntimeError("ConditionClusterNormalizer must be fitted before transform.")
        scaled_settings = self.setting_scaler.transform(settings)
        labels = self.kmeans.predict(scaled_settings)
        scaled_sensors = (sensors - self.sensor_means[labels]) / self.sensor_stds[labels]
        return np.concatenate([scaled_settings, scaled_sensors], axis=1).astype(np.float32)

    def metadata(self) -> dict[str, object]:
        if self.sensor_means is None or self.sensor_stds is None:
            raise RuntimeError("ConditionClusterNormalizer is not fitted.")
        return {
            "condition_clusters": self.n_clusters,
            "setting_mean": self.setting_scaler.mean_.tolist(),
            "setting_scale": self.setting_scaler.scale_.tolist(),
            "condition_centroids_scaled": self.kmeans.cluster_centers_.tolist(),
            "sensor_means": self.sensor_means.tolist(),
            "sensor_stds": self.sensor_stds.tolist(),
        }


def load_sampled_split(path: str | Path, split: str, sampling: int = 100) -> SampledSplit:
    split = split.lower()
    if split not in {"dev", "test"}:
        raise ValueError("split must be 'dev' or 'test'")
    sample_slice = slice(None, None, int(sampling))
    with h5py.File(path, "r") as hdf:
        settings = np.asarray(hdf[f"W_{split}"][sample_slice], dtype=np.float32)
        measured = np.asarray(hdf[f"X_s_{split}"][sample_slice], dtype=np.float32)
        virtual = np.asarray(hdf[f"X_v_{split}"][sample_slice, :2], dtype=np.float32)
        rul = np.asarray(hdf[f"Y_{split}"][sample_slice], dtype=np.float32).reshape(-1)
        auxiliary = np.asarray(hdf[f"A_{split}"][sample_slice], dtype=np.float32)
        setting_names = _decode_names(hdf["W_var"][:])
        measured_names = _decode_names(hdf["X_s_var"][:])
        virtual_names = _decode_names(hdf["X_v_var"][:])[:2]
        auxiliary_names = _decode_names(hdf["A_var"][:])
    unit_index = auxiliary_names.index("unit")
    units = auxiliary[:, unit_index].astype(np.int64)
    sensors = np.concatenate([measured, virtual], axis=1)
    return SampledSplit(settings, sensors, rul, units, setting_names, measured_names + virtual_names)


def _select_units(split: SampledSplit, units: list[int]) -> SampledSplit:
    mask = np.isin(split.units, units)
    return SampledSplit(
        split.settings[mask],
        split.sensors[mask],
        split.rul[mask],
        split.units[mask],
        split.setting_names,
        split.sensor_names,
    )


def make_unit_windows(
    features: np.ndarray,
    labels: np.ndarray,
    units: np.ndarray,
    *,
    window_size: int,
    stride: int,
) -> WindowData:
    x_rows = []
    y_rows = []
    unit_rows = []
    end_rows = []
    for unit in sorted(np.unique(units)):
        mask = units == unit
        unit_features = np.ascontiguousarray(features[mask])
        unit_labels = labels[mask]
        if len(unit_features) < window_size:
            continue
        view = np.lib.stride_tricks.sliding_window_view(unit_features, window_size, axis=0)
        view = np.moveaxis(view, -1, 1)[::stride]
        end_indices = np.arange(window_size - 1, len(unit_features), stride)
        x_rows.append(np.asarray(view, dtype=np.float32))
        y_rows.append(unit_labels[end_indices].astype(np.float32))
        unit_rows.append(np.full(len(end_indices), int(unit), dtype=np.int64))
        end_rows.append(end_indices.astype(np.int64))
    return WindowData(
        x=np.concatenate(x_rows),
        y=np.concatenate(y_rows),
        unit_ids=np.concatenate(unit_rows),
        end_cycles=np.concatenate(end_rows),
    )


def prepare_ncmapss_ds02(
    path: str | Path,
    *,
    sampling: int = 100,
    window_size: int = 50,
    stride: int = 1,
    condition_clusters: int = 6,
    train_units: list[int] | None = None,
    validation_units: list[int] | None = None,
    test_units: list[int] | None = None,
    test_source: str = "test",
) -> NCMAPSSPrepared:
    train_units = list(TRAIN_UNITS if train_units is None else train_units)
    validation_units = list(VALIDATION_UNITS if validation_units is None else validation_units)
    test_units = list(TEST_UNITS if test_units is None else test_units)
    test_source = str(test_source).lower()
    if test_source not in {"dev", "test"}:
        raise ValueError("test_source must be 'dev' or 'test'")
    if set(train_units) & set(validation_units) or set(train_units) & set(test_units) or set(validation_units) & set(test_units):
        raise ValueError("N-CMAPSS train, validation, and test units must be disjoint.")
    dev = load_sampled_split(path, "dev", sampling=sampling)
    test_pool = dev if test_source == "dev" else load_sampled_split(path, "test", sampling=sampling)
    train_split = _select_units(dev, train_units)
    val_split = _select_units(dev, validation_units)
    test_split = _select_units(test_pool, test_units)
    observed_dev_units = sorted(int(unit) for unit in np.unique(dev.units))
    observed_test_units = sorted(int(unit) for unit in np.unique(test_split.units))
    expected_dev_units = train_units + validation_units + (test_units if test_source == "dev" else [])
    if observed_dev_units != sorted(expected_dev_units):
        raise ValueError(f"Unexpected N-CMAPSS development units: {observed_dev_units}")
    if observed_test_units != sorted(test_units):
        raise ValueError(f"Unexpected N-CMAPSS test units: {observed_test_units}")

    normalizer = ConditionClusterNormalizer(n_clusters=condition_clusters).fit(
        train_split.settings,
        train_split.sensors,
    )
    train_features = normalizer.transform(train_split.settings, train_split.sensors)
    val_features = normalizer.transform(val_split.settings, val_split.sensors)
    test_features = normalizer.transform(test_split.settings, test_split.sensors)
    feature_names = train_split.setting_names + train_split.sensor_names
    sensor_indices = list(range(len(train_split.setting_names), len(feature_names)))
    metadata = {
        "sampling": sampling,
        "window_size": window_size,
        "stride": stride,
        "train_units": train_units,
        "validation_units": validation_units,
        "test_units": test_units,
        "test_source": test_source,
        "feature_names": feature_names,
        "sensor_indices": sensor_indices,
        **normalizer.metadata(),
    }
    return NCMAPSSPrepared(
        train=make_unit_windows(
            train_features, train_split.rul, train_split.units, window_size=window_size, stride=stride
        ),
        val=make_unit_windows(val_features, val_split.rul, val_split.units, window_size=window_size, stride=stride),
        test=make_unit_windows(test_features, test_split.rul, test_split.units, window_size=window_size, stride=stride),
        feature_names=feature_names,
        sensor_indices=sensor_indices,
        metadata=metadata,
    )


def save_prepared_cache(prepared: NCMAPSSPrepared, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        train_x=prepared.train.x,
        train_y=prepared.train.y,
        train_units=prepared.train.unit_ids,
        train_end=prepared.train.end_cycles,
        val_x=prepared.val.x,
        val_y=prepared.val.y,
        val_units=prepared.val.unit_ids,
        val_end=prepared.val.end_cycles,
        test_x=prepared.test.x,
        test_y=prepared.test.y,
        test_units=prepared.test.unit_ids,
        test_end=prepared.test.end_cycles,
        metadata=np.asarray(json.dumps(prepared.metadata, separators=(",", ":"))),
    )


def load_prepared_cache(path: str | Path) -> NCMAPSSPrepared:
    with np.load(path, allow_pickle=False) as payload:
        metadata = json.loads(str(payload["metadata"].item()))
        train = WindowData(payload["train_x"], payload["train_y"], payload["train_units"], payload["train_end"])
        val = WindowData(payload["val_x"], payload["val_y"], payload["val_units"], payload["val_end"])
        test = WindowData(payload["test_x"], payload["test_y"], payload["test_units"], payload["test_end"])
    return NCMAPSSPrepared(train, val, test, metadata["feature_names"], metadata["sensor_indices"], metadata)
