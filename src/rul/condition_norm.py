from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler

from .cmapss import SENSORS, SETTINGS


@dataclass
class ConditionAwareStandardScaler:
    """Standardize sensor channels inside learned operating conditions.

    Operating-condition clusters are learned from the three C-MAPSS setting
    channels on the training engines only. Settings are scaled globally, while
    sensor channels are scaled per assigned condition.
    """

    n_conditions: int
    feature_names: list[str]
    random_state: int = 42
    condition_column: str = "condition_id"
    setting_names: list[str] = field(default_factory=lambda: list(SETTINGS))

    def __post_init__(self) -> None:
        self.n_conditions = max(1, int(self.n_conditions))
        self.setting_features = [name for name in self.feature_names if name in self.setting_names]
        self.sensor_features = [name for name in self.feature_names if name in SENSORS]
        self.other_features = [
            name for name in self.feature_names if name not in self.setting_features and name not in self.sensor_features
        ]
        self.condition_setting_scaler: StandardScaler | None = None
        self.kmeans: KMeans | None = None
        self.global_scaler: StandardScaler | None = None
        self.sensor_scalers: dict[int, StandardScaler] = {}
        self.fallback_sensor_scaler: StandardScaler | None = None
        self.effective_conditions = 1

    def fit(self, df: pd.DataFrame) -> ConditionAwareStandardScaler:
        if not self.feature_names:
            raise ValueError("feature_names must not be empty")
        missing = [name for name in self.feature_names if name not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns for condition scaler: {missing}")

        self.effective_conditions = min(self.n_conditions, max(1, len(df)))
        settings = self._settings_matrix(df)
        if self.effective_conditions > 1 and settings.shape[1] > 0:
            self.condition_setting_scaler = StandardScaler()
            settings_scaled = self.condition_setting_scaler.fit_transform(settings)
            self.kmeans = KMeans(n_clusters=self.effective_conditions, random_state=self.random_state, n_init=10)
            labels = self.kmeans.fit_predict(settings_scaled)
        else:
            labels = np.zeros(len(df), dtype=np.int64)

        global_features = self.setting_features + self.other_features
        if global_features:
            self.global_scaler = StandardScaler().fit(df[global_features])

        if self.sensor_features:
            self.fallback_sensor_scaler = StandardScaler().fit(df[self.sensor_features])
            self.sensor_scalers = {}
            for condition_id in range(self.effective_conditions):
                mask = labels == condition_id
                if np.any(mask):
                    self.sensor_scalers[condition_id] = StandardScaler().fit(df.loc[mask, self.sensor_features])
                else:
                    self.sensor_scalers[condition_id] = self.fallback_sensor_scaler
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.sensor_features and self.fallback_sensor_scaler is None:
            raise RuntimeError("ConditionAwareStandardScaler must be fitted before transform")
        out = df.copy()
        out[self.feature_names] = out[self.feature_names].astype(float)
        labels = self.predict_condition(out)
        out[self.condition_column] = labels.astype(np.int64)

        global_features = self.setting_features + self.other_features
        if global_features and self.global_scaler is not None:
            out[global_features] = self.global_scaler.transform(out[global_features])

        if self.sensor_features:
            for condition_id in np.unique(labels):
                mask = labels == condition_id
                scaler = self.sensor_scalers.get(int(condition_id), self.fallback_sensor_scaler)
                out.loc[mask, self.sensor_features] = scaler.transform(out.loc[mask, self.sensor_features])
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def predict_condition(self, df: pd.DataFrame) -> np.ndarray:
        if self.kmeans is None or self.condition_setting_scaler is None:
            return np.zeros(len(df), dtype=np.int64)
        settings = self._settings_matrix(df)
        settings_scaled = self.condition_setting_scaler.transform(settings)
        return self.kmeans.predict(settings_scaled).astype(np.int64)

    def _settings_matrix(self, df: pd.DataFrame) -> np.ndarray:
        available = [name for name in self.setting_names if name in df.columns]
        if not available:
            return np.empty((len(df), 0), dtype=np.float64)
        return df[available].to_numpy(dtype=np.float64)


@dataclass
class OfficialStateStandardScaler:
    """Scale sensors within rounded C-MAPSS operating-state tuples.

    State definitions, per-state sensor scalers, and the fallback scaler are
    fitted on the training partition only. An unseen validation/test state uses
    the training-global sensor scaler instead of fitting new statistics.
    """

    feature_names: list[str]
    condition_column: str = "condition_id"
    setting_names: list[str] = field(default_factory=lambda: list(SETTINGS))
    decimals: int = 0

    def __post_init__(self) -> None:
        self.setting_features = [name for name in self.feature_names if name in self.setting_names]
        self.sensor_features = [name for name in self.feature_names if name in SENSORS]
        self.other_features = [
            name for name in self.feature_names if name not in self.setting_features and name not in self.sensor_features
        ]
        self.state_to_id: dict[tuple[float, ...], int] = {}
        self.sensor_scalers: dict[int, StandardScaler] = {}
        self.fallback_sensor_scaler: StandardScaler | None = None
        self.global_scaler: StandardScaler | None = None

    def fit(self, df: pd.DataFrame) -> OfficialStateStandardScaler:
        self._validate_features(df)
        states = self._state_keys(df)
        unique_states = sorted(set(states))
        self.state_to_id = {state: index for index, state in enumerate(unique_states)}
        labels = np.asarray([self.state_to_id[state] for state in states], dtype=np.int64)

        global_features = self.setting_features + self.other_features
        if global_features:
            self.global_scaler = StandardScaler().fit(df[global_features])
        if self.sensor_features:
            self.fallback_sensor_scaler = StandardScaler().fit(df[self.sensor_features])
            self.sensor_scalers = {}
            for condition_id in range(len(unique_states)):
                mask = labels == condition_id
                self.sensor_scalers[condition_id] = StandardScaler().fit(df.loc[mask, self.sensor_features])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.sensor_features and self.fallback_sensor_scaler is None:
            raise RuntimeError("OfficialStateStandardScaler must be fitted before transform")
        self._validate_features(df)
        out = df.copy()
        out[self.feature_names] = out[self.feature_names].astype(float)
        labels = self.predict_condition(out)
        out[self.condition_column] = labels

        global_features = self.setting_features + self.other_features
        if global_features and self.global_scaler is not None:
            out[global_features] = self.global_scaler.transform(out[global_features])
        if self.sensor_features:
            for condition_id in np.unique(labels):
                mask = labels == condition_id
                scaler = self.sensor_scalers.get(int(condition_id), self.fallback_sensor_scaler)
                out.loc[mask, self.sensor_features] = scaler.transform(out.loc[mask, self.sensor_features])
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def predict_condition(self, df: pd.DataFrame) -> np.ndarray:
        states = self._state_keys(df)
        return np.asarray([self.state_to_id.get(state, -1) for state in states], dtype=np.int64)

    def _state_keys(self, df: pd.DataFrame) -> list[tuple[float, ...]]:
        if not self.setting_features:
            return [tuple()] * len(df)
        values = np.round(df[self.setting_features].to_numpy(dtype=np.float64), decimals=self.decimals)
        return [tuple(row.tolist()) for row in values]

    def _validate_features(self, df: pd.DataFrame) -> None:
        missing = [name for name in self.feature_names if name not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns for official-state scaler: {missing}")


@dataclass
class ContinuousConditionCorrector:
    """Remove a smooth training-fitted setting response from sensor channels.

    A degree-2 polynomial ridge regression predicts each selected sensor from
    standardized operating settings. Sensor residuals are then standardized
    with training-only statistics. This is an algorithmic control for smooth
    condition correction, not a physical engine model.
    """

    feature_names: list[str]
    ridge_alpha: float = 1.0
    polynomial_degree: int = 2
    setting_names: list[str] = field(default_factory=lambda: list(SETTINGS))

    def __post_init__(self) -> None:
        self.setting_features = [name for name in self.feature_names if name in self.setting_names]
        self.sensor_features = [name for name in self.feature_names if name in SENSORS]
        self.other_features = [
            name for name in self.feature_names if name not in self.setting_features and name not in self.sensor_features
        ]
        self.setting_scaler: StandardScaler | None = None
        self.polynomial: PolynomialFeatures | None = None
        self.regressor: Ridge | None = None
        self.residual_scaler: StandardScaler | None = None
        self.other_scaler: StandardScaler | None = None

    def fit(self, df: pd.DataFrame) -> ContinuousConditionCorrector:
        self._validate_features(df)
        if self.setting_features:
            self.setting_scaler = StandardScaler().fit(df[self.setting_features])
            settings = self.setting_scaler.transform(df[self.setting_features])
            self.polynomial = PolynomialFeatures(
                degree=self.polynomial_degree,
                include_bias=True,
            ).fit(settings)
            design = self.polynomial.transform(settings)
        else:
            design = np.ones((len(df), 1), dtype=np.float64)

        if self.sensor_features:
            self.regressor = Ridge(alpha=self.ridge_alpha, fit_intercept=False)
            sensors = df[self.sensor_features].to_numpy(dtype=np.float64)
            self.regressor.fit(design, sensors)
            predicted = np.asarray(self.regressor.predict(design), dtype=np.float64).reshape(
                len(df),
                len(self.sensor_features),
            )
            residuals = sensors - predicted
            self.residual_scaler = StandardScaler().fit(residuals)
        if self.other_features:
            self.other_scaler = StandardScaler().fit(df[self.other_features])
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.sensor_features and (self.regressor is None or self.residual_scaler is None):
            raise RuntimeError("ContinuousConditionCorrector must be fitted before transform")
        self._validate_features(df)
        out = df.copy()
        out[self.feature_names] = out[self.feature_names].astype(float)

        if self.setting_features and self.setting_scaler is not None and self.polynomial is not None:
            settings = self.setting_scaler.transform(out[self.setting_features])
            design = self.polynomial.transform(settings)
            out[self.setting_features] = settings
        else:
            design = np.ones((len(out), 1), dtype=np.float64)
        if self.sensor_features and self.regressor is not None and self.residual_scaler is not None:
            sensors = out[self.sensor_features].to_numpy(dtype=np.float64)
            predicted = np.asarray(self.regressor.predict(design), dtype=np.float64).reshape(
                len(out),
                len(self.sensor_features),
            )
            residuals = sensors - predicted
            out[self.sensor_features] = self.residual_scaler.transform(residuals)
        if self.other_features and self.other_scaler is not None:
            out[self.other_features] = self.other_scaler.transform(out[self.other_features])
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def _validate_features(self, df: pd.DataFrame) -> None:
        missing = [name for name in self.feature_names if name not in df.columns]
        if missing:
            raise ValueError(f"Missing feature columns for continuous condition correction: {missing}")
