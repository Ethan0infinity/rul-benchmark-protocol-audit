from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from .cmapss import SENSORS, SETTINGS


@dataclass(frozen=True)
class FeatureSelectionResult:
    features: list[str]
    selected_sensors: list[str]
    scores: pd.DataFrame


def score_sensor_features(train_df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for sensor in SENSORS:
        values = train_df[sensor].astype(float)
        variance = float(values.var())
        if variance <= 1e-12:
            rul_corr = 0.0
            cycle_corr = 0.0
        else:
            rul_corr = abs(float(values.corr(train_df["rul"], method="spearman") or 0.0))
            cycle_corr = abs(float(values.corr(train_df["cycle"], method="spearman") or 0.0))
        rows.append(
            {
                "feature": sensor,
                "variance": variance,
                "abs_rul_corr": rul_corr,
                "abs_cycle_corr": cycle_corr,
            }
        )
    scores = pd.DataFrame(rows)
    for col in ["variance", "abs_rul_corr", "abs_cycle_corr"]:
        min_v = scores[col].min()
        max_v = scores[col].max()
        if max_v > min_v:
            scores[f"{col}_norm"] = (scores[col] - min_v) / (max_v - min_v)
        else:
            scores[f"{col}_norm"] = 0.0
    scores["hybrid_score"] = (
        0.30 * scores["variance_norm"]
        + 0.40 * scores["abs_rul_corr_norm"]
        + 0.30 * scores["abs_cycle_corr_norm"]
    )
    scores["sensor_index"] = scores["feature"].str.removeprefix("s").astype(int)
    return scores.sort_values(
        ["hybrid_score", "sensor_index"],
        ascending=[False, True],
        kind="mergesort",
    ).reset_index(drop=True)


def select_features(
    train_df: pd.DataFrame,
    feature_mode: str = "selected",
    max_selected_sensors: int = 14,
    include_settings: bool = True,
    output_csv: str | Path | None = None,
) -> FeatureSelectionResult:
    feature_mode = feature_mode.lower()
    scores = score_sensor_features(train_df)
    if feature_mode == "all":
        selected_sensors = list(SENSORS)
    elif feature_mode == "selected":
        selected_sensors = scores.head(max_selected_sensors)["feature"].tolist()
    elif feature_mode == "manual_fd001":
        selected_sensors = ["s2", "s3", "s4", "s7", "s8", "s9", "s11", "s12", "s13", "s14", "s15", "s17", "s20", "s21"]
    else:
        raise ValueError(f"Unknown feature_mode: {feature_mode}")

    features = [*SETTINGS, *selected_sensors] if include_settings else selected_sensors
    if output_csv is not None:
        path = Path(output_csv)
        path.parent.mkdir(parents=True, exist_ok=True)
        scores.to_csv(path, index=False)
    return FeatureSelectionResult(features=features, selected_sensors=selected_sensors, scores=scores)


def sensor_indices_in_features(features: list[str]) -> list[int]:
    return [i for i, name in enumerate(features) if name in SENSORS]


def feature_summary(features: list[str]) -> dict[str, object]:
    return {
        "n_features": len(features),
        "n_settings": len([f for f in features if f in SETTINGS]),
        "n_sensors": len([f for f in features if f in SENSORS]),
        "features": features,
    }
