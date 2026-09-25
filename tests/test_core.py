from __future__ import annotations

import csv
import json
import os
import subprocess
import sys
import tempfile
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from rul.augmentations import (
    apply_block_missing,
    apply_burst_noise,
    apply_correlated_group_missing_with_mask,
    apply_gaussian_noise,
    apply_sensor_bias_shift,
    apply_sensor_missing,
    apply_stuck_at_fault,
)
from rul.cmapss import COLUMNS, add_test_rul, add_train_rul
from rul.features import select_features
from rul.metrics import nasa_score, regression_metrics
from rul.models import build_model
from rul.preprocessing import make_validation_endpoint_windows, make_windows, split_units


def synthetic_cmapss(units: int = 4, cycles: int = 40) -> pd.DataFrame:
    rows = []
    for unit in range(1, units + 1):
        for cycle in range(1, cycles + 1):
            settings = [0.1 * unit, 0.01 * cycle, 1.0]
            sensors = [cycle * (i / 100.0) + unit * 0.01 for i in range(1, 22)]
            rows.append([unit, cycle, *settings, *sensors])
    return pd.DataFrame(rows, columns=COLUMNS)


def test_train_and_test_rul_labels_are_capped():
    train = add_train_rul(synthetic_cmapss(units=1, cycles=150), rul_cap=125)
    assert train["rul"].max() == 125
    assert train.loc[train["cycle"] == 150, "rul"].iloc[0] == 0

    final_rul = pd.Series({1: 20.0})
    test = add_test_rul(synthetic_cmapss(units=1, cycles=10), final_rul, rul_cap=125)
    assert test.loc[test["cycle"] == 10, "rul"].iloc[0] == 20.0
    assert test.loc[test["cycle"] == 1, "rul"].iloc[0] == 29.0


def _load_dynamic_policy_module_for_test():
    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_dynamic_maintenance_policy.py"
    spec = importlib.util.spec_from_file_location("dynamic_policy_core_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_observed_prefix_fixed_age_does_not_infer_post_cutoff_action():
    module = _load_dynamic_policy_module_for_test()
    group = pd.DataFrame(
        {
            "end_cycle": np.arange(1, 51),
            "true_rul": np.full(50, 20.0),
            "pred_rul": np.full(50, 100.0),
        }
    )
    result = module.simulate_fixed_age_policy(group, age_trigger=100.0, lead_time=5.0, corrective_ratio=10.0)
    assert result["resolved"] == 0
    assert result["unresolved_beyond_observation"] == 1
    assert np.isnan(result["cost"])


def test_dynamic_policy_pairing_reports_resolution_discordance():
    module = _load_dynamic_policy_module_for_test()
    common = {
        "subset": "FD001",
        "split_level": 42,
        "training_stream": 31415,
        "rul_trigger_threshold": 20.0,
        "lead_time_cycles": 5.0,
        "persistence_windows": 2,
        "corrective_cost_ratio": 10.0,
    }
    rows = []
    outcomes = [(1, 1, 1, 2.0, 3.0), (2, 1, 0, 2.0, np.nan), (3, 0, 1, np.nan, 4.0), (4, 0, 0, np.nan, np.nan)]
    for unit_id, reference_resolved, comparison_resolved, reference_cost, comparison_cost in outcomes:
        for model, resolved, cost in (
            ("rast_gru", reference_resolved, reference_cost),
            ("rast_gru_v2", comparison_resolved, comparison_cost),
        ):
            rows.append(
                {
                    **common,
                    "unit_id": unit_id,
                    "model": model,
                    "policy_type": "model_threshold",
                    "resolved": resolved,
                    "unresolved_beyond_observation": 1 - resolved,
                    "cost": cost,
                }
            )
    result = module.build_paired_model_comparisons(pd.DataFrame(rows)).iloc[0]
    assert int(result["both_resolved_count"]) == 1
    assert int(result["reference_only_resolved_count"]) == 1
    assert int(result["comparison_only_resolved_count"]) == 1
    assert int(result["neither_resolved_count"]) == 1
    assert int(result["paired_cost_support"]) == 1
    assert np.isclose(result["mean_cost_delta_comparison_minus_reference_among_both_resolved"], 1.0)


def test_standard_protocol_paired_metric_definitions():
    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "analyze_standard_protocol_track.py"
    spec = importlib.util.spec_from_file_location("standard_protocol_metrics_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    true_rul = np.array([10.0, 20.0, 40.0])
    error = np.array([-1.0, 2.0, 11.0])
    assert np.isclose(module.paired_metric(error, true_rul, "rmse"), np.sqrt(126.0 / 3.0))
    assert np.isclose(module.paired_metric(error, true_rul, "lpr30"), 0.5)
    expected_nasa = np.mean(np.where(error < 0.0, np.exp(-error / 13.0) - 1.0, np.exp(error / 10.0) - 1.0))
    assert np.isclose(module.paired_metric(error, true_rul, "nasa_per_engine"), expected_nasa)

    conventional_metadata = {
        "scaler": "standard",
        "append_missing_mask": False,
        "checkpoint_selection_metric": "val_last_rmse",
    }
    conventional_config = {
        "training": {
            "loss": "huber",
            "late_over_weight": 1.0,
            "smooth_late_risk_weight": 0.0,
            "selection_metric": "val_last_rmse",
            "selection_lpr_weight": 0.0,
            "selection_severe_late_weight": 0.0,
        },
        "augmentation": {"profile": "none"},
    }
    module.validate_track_protocol(
        conventional_metadata,
        conventional_config,
        "standardized_conventional_protocol",
        Path("synthetic-run"),
    )
    invalid_metadata = dict(conventional_metadata, append_missing_mask=True)
    try:
        module.validate_track_protocol(
            invalid_metadata,
            conventional_config,
            "standardized_conventional_protocol",
            Path("synthetic-run"),
        )
    except ValueError as exc:
        assert "append_missing_mask" in str(exc)
    else:
        raise AssertionError("Track protocol validation accepted a mixed-protocol run")


def test_feature_selection_and_windows():
    df = add_train_rul(synthetic_cmapss(units=4, cycles=40), rul_cap=125)
    selected = select_features(df, feature_mode="selected", max_selected_sensors=8)
    assert len(selected.selected_sensors) == 8
    assert selected.features[:3] == ["setting_1", "setting_2", "setting_3"]

    windows = make_windows(df, selected.features, window_size=10, stride=5)
    assert windows.x.shape[1:] == (10, len(selected.features))
    assert windows.y.ndim == 1
    assert len(windows.unit_ids) == len(windows.y)


def test_validation_endpoint_designs_expand_critical_support_without_engine_imbalance():
    df = add_train_rul(synthetic_cmapss(units=4, cycles=160), rul_cap=125)
    features = ["setting_1", "s1", "s2"]
    fixed = make_validation_endpoint_windows(df, features, 30, "fixed_multi", seed=42)
    stratified = make_validation_endpoint_windows(df, features, 30, "critical_stratified", seed=42)
    for data in (fixed, stratified):
        per_engine = pd.Series(data.unit_ids).value_counts()
        assert per_engine.nunique() == 1
        assert int(np.sum(data.y <= 30.0)) >= 3 * 4
        assert len(data.y) > 4


def test_normalized_curve_mean_divides_by_scenario_intensity_range():
    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_stress_pair_extension.py"
    spec = importlib.util.spec_from_file_location("stress_curve_mean_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frame = pd.DataFrame(
        {
            "scenario": ["clean", "fault", "fault"],
            "level": [0.0, 0.2, 0.4],
            "rmse": [1.0, 2.0, 3.0],
        }
    )
    assert np.isclose(module.normalized_curve_mean(frame, "rmse"), 2.0)


def test_prepare_data_validation_windows_cover_non_terminal_rul():
    import rul.training as training

    train = add_train_rul(synthetic_cmapss(units=6, cycles=50), rul_cap=125)
    final_rul = pd.Series({unit: 20.0 + unit for unit in range(1, 5)})
    test = add_test_rul(synthetic_cmapss(units=4, cycles=25), final_rul, rul_cap=125)
    original_load_subset = training.load_subset

    def fake_load_subset(*_args, **_kwargs):
        return train, test

    config = {
        "_project_root": ".",
        "project": {"seed": 42},
        "data": {
            "root": ".",
            "rul_cap": 125,
            "validation_split": 0.5,
            "window_size": 10,
            "stride": 5,
            "scaler": "standard",
            "feature_mode": "selected",
            "max_selected_sensors": 8,
            "include_settings": True,
        },
    }

    try:
        training.load_subset = fake_load_subset
        prepared = training.prepare_data(config, "FD001")
    finally:
        training.load_subset = original_load_subset

    assert len(prepared.val.y) > 6
    assert prepared.val.y.max() > 0.0
    assert len(set(prepared.val.y.tolist())) > 1


def test_prepare_data_splits_units_before_scaler_fit_and_keeps_last_window_validation():
    import rul.training as training

    train = add_train_rul(synthetic_cmapss(units=8, cycles=35), rul_cap=125)
    train.loc[train["unit_id"].isin([1, 2]), "s1"] += 5000.0
    final_rul = pd.Series({unit: 20.0 + unit for unit in range(1, 5)})
    test = add_test_rul(synthetic_cmapss(units=4, cycles=25), final_rul, rul_cap=125)
    original_load_subset = training.load_subset

    def fake_load_subset(*_args, **_kwargs):
        return train, test

    config = {
        "_project_root": ".",
        "project": {"seed": 7},
        "data": {
            "root": ".",
            "rul_cap": 125,
            "validation_split": 0.25,
            "window_size": 10,
            "stride": 5,
            "scaler": "standard",
            "feature_mode": "all",
            "include_settings": False,
        },
    }

    try:
        training.load_subset = fake_load_subset
        prepared = training.prepare_data(config, "FD001")
    finally:
        training.load_subset = original_load_subset

    train_units, val_units = split_units(train["unit_id"].to_numpy(), 0.25, 7)
    expected_mean = train[train["unit_id"].isin(train_units)][prepared.features].mean().to_numpy()
    assert np.allclose(prepared.scaler.mean_, expected_mean)
    assert set(np.unique(prepared.val.unit_ids)) == set(val_units.tolist())
    assert set(np.unique(prepared.val_last.unit_ids)) == set(val_units.tolist())
    assert len(prepared.val_last.y) == len(val_units)
    assert prepared.val_last.y.max() > 0.0
    assert len(set(prepared.val_last.y.tolist())) > 1


def test_condition_aware_scaler_normalizes_sensors_inside_each_operating_condition():
    from rul.condition_norm import ConditionAwareStandardScaler

    rows = []
    for condition, setting_1, sensor_shift in [(0, -5.0, 100.0), (1, 5.0, 300.0)]:
        for i in range(10):
            rows.append(
                {
                    "unit_id": condition + 1,
                    "cycle": i + 1,
                    "setting_1": setting_1 + 0.01 * i,
                    "setting_2": 0.0,
                    "setting_3": 1.0,
                    "s1": sensor_shift + i,
                    "s2": sensor_shift * 2 + i,
                    "rul": 10 - i,
                }
            )
    df = pd.DataFrame(rows)
    scaler = ConditionAwareStandardScaler(n_conditions=2, feature_names=["setting_1", "setting_2", "setting_3", "s1", "s2"])
    scaled = scaler.fit_transform(df)

    assert "condition_id" in scaled.columns
    for _, group in scaled.groupby("condition_id"):
        assert abs(float(group["s1"].mean())) < 1e-6
        assert abs(float(group["s2"].mean())) < 1e-6

    test = pd.DataFrame(
        [
            {"unit_id": 99, "cycle": 1, "setting_1": 5.02, "setting_2": 0.0, "setting_3": 1.0, "s1": 304.5, "s2": 604.5, "rul": 12.0}
        ]
    )
    transformed = scaler.transform(test)
    assert int(transformed["condition_id"].iloc[0]) in {0, 1}
    assert abs(float(transformed["s1"].iloc[0])) < 2.0


def test_metrics_and_augmentations():
    y_true = np.array([10.0, 20.0, 30.0])
    y_pred = np.array([12.0, 18.0, 29.0])
    metrics = regression_metrics(y_true, y_pred)
    assert metrics["rmse"] > 0
    assert metrics["mae"] > 0
    assert nasa_score(y_true, y_pred) > 0

    x = np.ones((3, 10, 6), dtype=np.float32)
    noisy = apply_gaussian_noise(x, 0.1, [2, 3], seed=1)
    missing = apply_sensor_missing(x, 0.5, [2, 3, 4, 5], seed=1)
    block = apply_block_missing(x, 0.3, [2, 3, 4, 5], seed=1)
    assert noisy.shape == x.shape
    assert missing.shape == x.shape
    assert block.shape == x.shape
    assert np.any(missing[..., 2:] == 0.0)
    assert np.any(block[..., 2:] == 0.0)


def test_extended_fault_generators_preserve_non_sensor_features_and_masks():
    x = np.ones((6, 30, 5), dtype=np.float32)
    sensor_indices = [2, 3, 4]
    trend = x + np.arange(30, dtype=np.float32)[None, :, None]
    outputs = [
        apply_sensor_bias_shift(x, 0.25, sensor_indices, seed=42),
        apply_stuck_at_fault(trend, 0.34, sensor_indices, seed=42),
        apply_burst_noise(x, 0.5, sensor_indices, seed=42),
    ]
    assert np.array_equal(outputs[0][..., :2], x[..., :2])
    assert np.array_equal(outputs[1][..., :2], trend[..., :2])
    assert np.array_equal(outputs[2][..., :2], x[..., :2])
    correlated, mask = apply_correlated_group_missing_with_mask(x, 0.34, sensor_indices, seed=42)
    assert np.array_equal(correlated[..., :2], x[..., :2])
    assert mask.shape == (6, 30, 3)
    assert np.any(mask == 0.0)
    assert np.any(correlated[..., sensor_indices] == 0.0)


def test_stuck_at_fault_supports_multiple_channels_without_axis_broadcasting():
    trend = np.broadcast_to(np.arange(30, dtype=np.float32)[None, :, None], (4, 30, 6)).copy()
    sensor_indices = [2, 3, 4, 5]
    stuck = apply_stuck_at_fault(trend, 1.0, sensor_indices, seed=7)
    assert stuck.shape == trend.shape
    assert np.array_equal(stuck[..., :2], trend[..., :2])
    for window_idx in range(stuck.shape[0]):
        for sensor_idx in sensor_indices:
            differences = np.diff(stuck[window_idx, :, sensor_idx])
            frozen = np.flatnonzero(differences == 0.0)
            assert len(frozen) > 0
            onset = int(frozen[0] + 1)
            assert np.all(stuck[window_idx, onset:, sensor_idx] == stuck[window_idx, onset, sensor_idx])


def test_critical_zone_metrics_and_asymmetric_loss_penalize_late_predictions():
    from rul.losses import AsymmetricWeightedHuberLoss
    from rul.metrics import critical_zone_metrics

    y_true = np.array([10.0, 20.0, 40.0, 80.0])
    y_pred = np.array([15.0, 12.0, 55.0, 70.0])
    critical = critical_zone_metrics(y_true, y_pred, thresholds=(30, 50))
    assert critical["critical_30_count"] == 2
    assert critical["critical_30_late_prediction_ratio"] == 0.5
    assert critical["critical_30_severe_late_5_ratio"] == 0.0
    assert critical["critical_30_severe_late_10_ratio"] == 0.0
    assert critical["critical_30_mean_late_excess"] == 2.5
    assert critical["critical_30_late_q95"] == 5.0
    assert critical["critical_30_late_cvar95"] == 5.0
    assert critical["critical_30_decision_cost_5"] == 16.5
    assert critical["critical_50_count"] == 3
    assert critical["critical_50_late_prediction_ratio"] == 2 / 3
    assert critical["critical_50_severe_late_5_ratio"] == 1 / 3
    assert critical["critical_50_severe_late_10_ratio"] == 1 / 3
    assert critical["critical_50_over_error_mean"] > 0
    assert critical["critical_50_under_error_mean"] < 0
    assert np.isclose(critical["critical_50_mean_late_excess"], 20.0 / 3.0)
    assert critical["critical_50_late_cvar95"] == 15.0

    target = torch.tensor([10.0])
    loss_fn = AsymmetricWeightedHuberLoss(delta=1.0, late_over_weight=3.0)
    over_loss = loss_fn(torch.tensor([15.0]), target)
    under_loss = loss_fn(torch.tensor([5.0]), target)
    assert over_loss > under_loss


def test_window_dataset_can_append_explicit_missing_mask_features():
    from rul.preprocessing import RULWindowDataset, WindowData

    x = np.ones((1, 4, 5), dtype=np.float32)
    x[0, :, 2] = 0.0
    data = WindowData(
        x=x,
        y=np.array([1.0], dtype=np.float32),
        unit_ids=np.array([1], dtype=np.int64),
        end_cycles=np.array([4], dtype=np.int64),
    )
    dataset = RULWindowDataset(data, sensor_indices=[2, 3], append_missing_mask=True)
    xb, yb = dataset[0]

    assert xb.shape == (4, 7)
    assert yb.item() == 1.0
    assert torch.all(xb[:, 5] == 1.0)
    assert torch.all(xb[:, 6] == 1.0)


def test_window_dataset_missing_mask_comes_from_augmentation_not_zero_values():
    from rul.preprocessing import RULWindowDataset, WindowData

    x = np.ones((2, 6, 5), dtype=np.float32)
    data = WindowData(
        x=x,
        y=np.array([1.0, 2.0], dtype=np.float32),
        unit_ids=np.array([1, 2], dtype=np.int64),
        end_cycles=np.array([6, 6], dtype=np.int64),
    )
    kwargs = dict(
        sensor_indices=[2, 3, 4],
        augmentation_profile="missing",
        missing_rate=1 / 3,
        append_missing_mask=True,
        base_seed=123,
    )
    first = RULWindowDataset(data, **kwargs)
    second = RULWindowDataset(data, **kwargs)
    x1, _ = first[0]
    x2, _ = second[0]

    assert torch.equal(x1, x2)
    mask = x1[:, 5:]
    values = x1[:, :5]
    assert torch.any(mask == 0.0)
    missing_sensor_columns = torch.where(mask[0] == 0.0)[0] + 2
    for col in missing_sensor_columns.tolist():
        assert torch.all(values[:, col] == 0.0)


def test_window_dataset_epoch_seed_changes_augmentation_reproducibly():
    from rul.preprocessing import RULWindowDataset, WindowData

    x = np.ones((2, 6, 5), dtype=np.float32)
    data = WindowData(
        x=x,
        y=np.array([1.0, 2.0], dtype=np.float32),
        unit_ids=np.array([1, 2], dtype=np.int64),
        end_cycles=np.array([6, 6], dtype=np.int64),
    )
    kwargs = dict(
        sensor_indices=[2, 3, 4],
        augmentation_profile="missing_noise_drift",
        missing_rate=1 / 3,
        noise_std=0.05,
        append_missing_mask=True,
        base_seed=123,
    )
    first = RULWindowDataset(data, **kwargs)
    second = RULWindowDataset(data, **kwargs)

    first.set_epoch(0)
    second.set_epoch(0)
    epoch0_a, _ = first[0]
    epoch0_b, _ = second[0]
    first.set_epoch(1)
    second.set_epoch(1)
    epoch1_a, _ = first[0]
    epoch1_b, _ = second[0]

    assert torch.equal(epoch0_a, epoch0_b)
    assert torch.equal(epoch1_a, epoch1_b)
    assert not torch.equal(epoch0_a, epoch1_a)


def test_window_random_missing_uses_different_sensor_masks_per_window():
    from rul.augmentations import apply_window_random_missing_with_mask

    x = np.ones((4, 6, 5), dtype=np.float32)
    corrupted, mask = apply_window_random_missing_with_mask(x, missing_rate=1 / 3, sensor_indices=[2, 3, 4], seed=5)

    assert corrupted.shape == x.shape
    assert mask.shape == (4, 6, 3)
    per_window_patterns = {tuple(mask[i, 0].tolist()) for i in range(mask.shape[0])}
    assert len(per_window_patterns) > 1


def test_explicit_shuffle_seed_reproduces_loader_order_without_changing_dataset_seed():
    from rul.preprocessing import WindowData
    from rul.training import _make_loader

    data = WindowData(
        x=np.arange(20 * 2, dtype=np.float32).reshape(20, 2, 1),
        y=np.arange(20, dtype=np.float32),
        unit_ids=np.arange(20, dtype=np.int64),
        end_cycles=np.ones(20, dtype=np.int64),
    )

    def order(seed: int) -> list[int]:
        loader = _make_loader(
            data,
            batch_size=4,
            shuffle=True,
            num_workers=0,
            sensor_indices=[],
            shuffle_seed=seed,
        )
        return [int(value) for _, targets in loader for value in targets]

    assert order(11) == order(11)
    assert order(11) != order(12)


def test_checkpoint_score_prefers_lower_engine_level_validation_metric():
    from rul.training import checkpoint_score_from_metrics

    metrics = {"val_loss": 0.5, "val_last_rmse": 12.0, "val_last_nasa_score": 120.0}
    assert checkpoint_score_from_metrics(metrics, "val_last_rmse") == 12.0
    assert checkpoint_score_from_metrics(metrics, "val_last_nasa_score") == 120.0
    assert checkpoint_score_from_metrics(metrics, "val_loss_all_windows") == 0.5


def test_monotonicity_metrics_count_prediction_increases_within_engine():
    from rul.metrics import monotonicity_metrics
    from rul.preprocessing import WindowData

    data = WindowData(
        x=np.zeros((4, 3, 2), dtype=np.float32),
        y=np.array([30.0, 20.0, 15.0, 10.0], dtype=np.float32),
        unit_ids=np.array([1, 1, 1, 2], dtype=np.int64),
        end_cycles=np.array([3, 4, 5, 3], dtype=np.int64),
    )
    metrics = monotonicity_metrics(data, np.array([30.0, 32.0, 20.0, 10.0]))
    assert metrics["monotonic_pairs"] == 2
    assert metrics["monotonic_violation_count"] == 1
    assert metrics["monotonic_violation_rate"] == 0.5


def test_wilcoxon_signed_rank_detects_consistent_paired_improvement():
    from rul.statistics import wilcoxon_signed_rank

    baseline = np.array([5.0, 6.0, 7.0, 8.0, 9.0])
    proposed = np.array([3.0, 4.0, 5.0, 6.0, 7.0])
    result = wilcoxon_signed_rank(baseline, proposed)
    assert result["n"] == 5
    assert result["w_plus"] > result["w_minus"]
    assert 0.0 <= result["p_value"] <= 1.0
    assert result["effect_size"] > 0.0


def test_sensor_drift_and_named_profile_change_only_sensor_channels():
    from rul.augmentations import apply_sensor_drift, apply_augmentation_profile

    x = np.ones((2, 12, 6), dtype=np.float32)
    drifted = apply_sensor_drift(x, drift_rate=0.2, sensor_indices=[2, 3], seed=1)
    assert drifted.shape == x.shape
    assert np.allclose(drifted[..., :2], 1.0)
    assert not np.allclose(drifted[..., 2:4], 1.0)

    profiled = apply_augmentation_profile(
        x,
        profile_name="missing_noise_drift",
        sensor_indices=[2, 3, 4, 5],
        noise_std=0.05,
        missing_rate=0.25,
        block_missing_rate=0.25,
        drift_rate=0.1,
        seed=2,
    )
    assert profiled.shape == x.shape
    assert np.allclose(profiled[..., :2], 1.0)


def test_augmentation_probability_zero_preserves_clean_window_and_mask():
    from rul.augmentations import apply_augmentation_profile_with_mask

    x = np.ones((1, 12, 6), dtype=np.float32)
    augmented, mask = apply_augmentation_profile_with_mask(
        x,
        profile_name="missing_noise_drift",
        sensor_indices=[2, 3, 4, 5],
        noise_std=0.05,
        missing_rate=0.25,
        block_missing_rate=0.25,
        drift_rate=0.1,
        mechanism_apply_probability=0.0,
        seed=17,
    )

    assert np.array_equal(augmented, x)
    assert np.all(mask == 1.0)


def test_augmentation_probability_one_preserves_locked_protocol_semantics():
    from rul.augmentations import apply_augmentation_profile_with_mask

    x = np.ones((1, 12, 6), dtype=np.float32)
    kwargs = dict(
        profile_name="missing_noise_drift",
        sensor_indices=[2, 3, 4, 5],
        noise_std=0.05,
        missing_rate=0.25,
        block_missing_rate=0.25,
        drift_rate=0.1,
        seed=23,
    )
    locked_x, locked_mask = apply_augmentation_profile_with_mask(x, **kwargs)
    explicit_x, explicit_mask = apply_augmentation_profile_with_mask(
        x,
        mechanism_apply_probability=1.0,
        corrupted_window_probability=1.0,
        **kwargs,
    )

    assert np.array_equal(explicit_x, locked_x)
    assert np.array_equal(explicit_mask, locked_mask)


def test_clean_corrupted_mixture_is_seeded_per_window():
    from rul.preprocessing import RULWindowDataset, WindowData

    x = np.ones((20, 8, 6), dtype=np.float32)
    data = WindowData(
        x=x,
        y=np.arange(20, dtype=np.float32),
        unit_ids=np.arange(20, dtype=np.int64),
        end_cycles=np.full(20, 8, dtype=np.int64),
    )
    kwargs = dict(
        sensor_indices=[2, 3, 4, 5],
        augmentation_profile="missing_noise_drift",
        noise_std=0.05,
        missing_rate=0.25,
        block_missing_rate=0.25,
        drift_rate=0.1,
        mechanism_apply_probability=1.0,
        corrupted_window_probability=0.5,
        append_missing_mask=True,
        base_seed=101,
    )
    first = RULWindowDataset(data, **kwargs)
    second = RULWindowDataset(data, **kwargs)
    first_rows = [first[index][0] for index in range(len(first))]
    second_rows = [second[index][0] for index in range(len(second))]
    changed = [not torch.equal(row[:, :6], torch.ones_like(row[:, :6])) for row in first_rows]

    assert all(torch.equal(left, right) for left, right in zip(first_rows, second_rows))
    assert any(changed)
    assert not all(changed)


def test_augmentation_probabilities_reject_values_outside_unit_interval():
    from rul.augmentations import apply_augmentation_profile_with_mask

    x = np.ones((1, 4, 4), dtype=np.float32)
    for keyword, value in (
        ("mechanism_apply_probability", -0.1),
        ("mechanism_apply_probability", 1.1),
        ("corrupted_window_probability", -0.1),
        ("corrupted_window_probability", 1.1),
    ):
        kwargs = {keyword: value}
        try:
            apply_augmentation_profile_with_mask(
                x,
                profile_name="noise",
                sensor_indices=[2, 3],
                noise_std=0.1,
                **kwargs,
            )
        except ValueError:
            pass
        else:
            raise AssertionError(f"Expected ValueError for {keyword}={value}")


def test_all_models_forward():
    x = torch.randn(5, 30, 12)
    for name in [
        "mlp",
        "cnn1d",
        "lstm",
        "gru",
        "tcn",
        "tcn_gru",
        "cnn_lstm",
        "attention_gru",
        "bigru_attention",
        "transformer_lite",
        "dual_attention_tcn",
        "regime_dual_attention_cnn_gru",
        "sensor_graph_gru",
        "quantile_gru",
        "official_dual_mixer",
        "rs_tcn_gru",
        "rast_gru",
        "rast_gru_v2",
    ]:
        model = build_model(
            name,
            input_features=12,
            window_size=30,
            hidden_size=16,
            tcn_channels=[8, 16],
            kernel_size=3,
            dropout=0.1,
        )
        y = model(x)
        assert y.shape == (5,)


def test_official_dual_mixer_adapter_matches_declared_reference_shape_and_size():
    model = build_model(
        "official_dual_mixer",
        input_features=14,
        window_size=30,
        hidden_size=32,
        tcn_channels=[48, 64],
        kernel_size=3,
        dropout=0.0,
    )
    assert len(model.layers) == 6
    assert sum(parameter.numel() for parameter in model.parameters()) == 64123
    assert model(torch.randn(3, 30, 14)).shape == (3,)


def test_model_feature_metadata_uses_architecture_not_global_defaults():
    from rul.training import model_feature_metadata

    global_defaults = {
        "use_reliability_gate": True,
        "use_channel_gate": True,
        "use_local_trend": True,
        "use_temporal_attention": True,
        "use_uncertainty_head": True,
    }
    gru = model_feature_metadata("gru", global_defaults, append_missing_mask=True)
    assert not any(gru.values())

    attention = model_feature_metadata("attention_gru", global_defaults, append_missing_mask=True)
    assert attention["model_use_temporal_attention"] is True
    assert attention["model_use_reliability_gate"] is False
    assert attention["model_use_uncertainty_head"] is False

    proposed = model_feature_metadata("rast_gru_v2", global_defaults, append_missing_mask=True)
    assert proposed["model_use_mask_aware_reliability_gate"] is True
    assert proposed["model_use_channel_gate"] is True
    assert proposed["model_use_local_trend"] is True
    assert proposed["model_use_uncertainty_head"] is True


def test_risk_objective_quantiles_and_validation_score_are_well_formed():
    from rul.losses import build_loss
    from rul.training import training_objective, validation_risk_score

    model = build_model(
        "rast_gru_v2",
        input_features=12,
        window_size=30,
        hidden_size=16,
        tcn_channels=[8, 16],
        kernel_size=3,
        dropout=0.1,
        mask_feature_count=4,
    )
    x = torch.randn(6, 30, 12)
    x[:, :, -4:] = 1.0
    x[:3, :, -2:] = 0.0
    y = torch.tensor([5.0, 10.0, 20.0, 40.0, 60.0, 80.0])
    cfg = {
        "late_life_threshold": 30.0,
        "smooth_late_risk_weight": 0.1,
        "smooth_late_risk_temperature": 2.0,
        "reliability_supervision_weight": 0.1,
        "consistency_weight": 0.05,
        "consistency_noise_std": 0.01,
        "quantile_calibration_weight": 0.1,
        "selection_lpr_weight": 0.25,
        "selection_severe_late_weight": 0.25,
    }
    criterion = build_loss("asymmetric_weighted_huber", 1.0, 30.0, 1.5, 1.25)
    loss, components = training_objective(model, x, y, criterion, cfg, [4, 5, 6, 7])
    lower, median, upper = model.forward_quantiles(x)

    assert torch.isfinite(loss)
    assert {"base", "smooth_late_risk", "reliability_supervision", "consistency", "quantile_calibration", "total"} <= set(components)
    assert torch.all(lower <= median)
    assert torch.all(median <= upper)
    score = validation_risk_score(
        {
            "rmse": 12.5,
            "critical_30_late_prediction_ratio": 0.4,
            "critical_30_severe_late_10_ratio": 0.2,
        },
        cfg,
        125.0,
    )
    assert abs(score - 0.25) < 1e-12


def test_decision_cost_sensitivity_covers_every_model_and_cost_ratio():
    import importlib.util

    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / "scripts" / "build_advanced_evidence.py"
    spec = importlib.util.spec_from_file_location("advanced_decision_cost_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    grid = {}
    for seed in module.FORMAL_BENCHMARK_SEEDS:
        for subset in module.FORMAL_BENCHMARK_SUBSETS:
            for index, model in enumerate(module.FORMAL_BENCHMARK_MODELS):
                grid[(seed, subset, model)] = pd.DataFrame(
                    {"true_rul": [10.0, 20.0], "pred_rul": [11.0 + index * 0.01, 19.0]}
                )
    tasks, summary = module.build_decision_cost_sensitivity(grid)
    assert len(tasks) == 4 * len(module.FORMAL_BENCHMARK_SEEDS) * len(module.FORMAL_BENCHMARK_SUBSETS) * len(module.FORMAL_BENCHMARK_MODELS)
    assert len(summary) == 4 * len(module.FORMAL_BENCHMARK_MODELS)
    assert set(summary["late_to_early_cost_ratio"]) == {1.0, 2.0, 5.0, 10.0}


def test_advanced_evidence_interval_calibration_and_exact_hypervolume():
    import importlib.util

    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / "scripts" / "build_advanced_evidence.py"
    spec = importlib.util.spec_from_file_location("advanced_calibration_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    assert abs(module.dominated_hypervolume_3d(np.asarray([[0.0, 0.0, 0.0]])) - 1.1**3) < 1e-12
    assert abs(
        module.dominated_hypervolume_3d(np.asarray([[0.0, 0.0, 0.0], [0.5, 0.5, 0.5]])) - 1.1**3
    ) < 1e-12

    interval_grid = {}
    for seed in module.FORMAL_BENCHMARK_SEEDS:
        for subset in module.FORMAL_BENCHMARK_SUBSETS:
            for model in module.INTERVAL_MODELS:
                interval_grid[(seed, subset, model)] = pd.DataFrame(
                    {
                        "unit_id": [1, 2],
                        "true_rul": [10.0, 50.0],
                        "q10_rul": [8.0, 45.0],
                        "q50_rul": [10.0, 50.0],
                        "q90_rul": [12.0, 55.0],
                    }
                )
    long, summary = module.build_interval_calibration(interval_grid)
    assert len(long) == 40
    assert len(summary) == 2
    assert np.allclose(long["empirical_coverage"], 1.0)
    assert np.allclose(long["absolute_coverage_error"], 0.2)


def test_empirical_residual_quantile_uses_conservative_order_statistic():
    import importlib.util

    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / "scripts" / "calibrate_prediction_intervals.py"
    spec = importlib.util.spec_from_file_location("empirical_interval_adjustment_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    qhat, level = module.conformal_quantile(np.asarray([-1.0, 0.0, 1.0, 2.0, 3.0]), alpha=0.2)
    assert qhat == 3.0
    assert level == 1.0
    qhat_inside, _ = module.conformal_quantile(np.asarray([-3.0, -2.0, -1.0, -0.5, -0.1]), alpha=0.2)
    assert qhat_inside == 0.0


def test_rast_gru_v2_supports_strict_ablation_switches_without_changing_model_family():
    x = torch.randn(4, 30, 12)
    model = build_model(
        "rast_gru_v2",
        input_features=12,
        window_size=30,
        hidden_size=16,
        tcn_channels=[8, 16],
        kernel_size=3,
        dropout=0.1,
        use_reliability_gate=False,
        use_channel_gate=False,
        use_local_trend=False,
        use_temporal_attention=False,
    )

    assert model.__class__.__name__ == "RASTGRUV2Regressor"
    assert model(x).shape == (4,)


def test_ablation_variants_toggle_only_intended_v2_switches():
    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_ablation.py"
    spec = importlib.util.spec_from_file_location("run_ablation_for_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    make_variant = module.make_variant
    assert "no_channel_gate" in module.DEFAULT_ABLATION_VARIANTS

    base = {
        "project": {"experiment_name": "paper_ablation"},
        "data": {"scaler": "condition_standard", "condition_clusters": {"FD001": 1}, "append_missing_mask": True},
        "model": {"name": "rast_gru_v2"},
        "training": {"loss": "asymmetric_weighted_huber"},
        "augmentation": {
            "profile": "missing_noise_drift",
            "train_noise_std": 0.01,
            "train_missing_rate": 0.1,
            "train_block_missing_rate": 0.1,
            "train_drift_rate": 0.01,
            "train_sensor_dropout_prob": 0.0,
        },
    }

    no_gate, gate_model = make_variant(base, "no_reliability_gate")
    no_channel, channel_model = make_variant(base, "no_channel_gate")
    no_attention, attention_model = make_variant(base, "no_temporal_attention")
    no_trend, trend_model = make_variant(base, "no_local_trend")

    assert gate_model == "rast_gru_v2"
    assert channel_model == "rast_gru_v2"
    assert attention_model == "rast_gru_v2"
    assert trend_model == "rast_gru_v2"
    assert no_gate["model"]["use_reliability_gate"] is False
    assert no_channel["model"]["use_channel_gate"] is False
    assert no_attention["model"]["use_temporal_attention"] is False
    assert no_trend["model"]["use_local_trend"] is False


def test_ablation_weighted_huber_no_asymmetry_keeps_late_life_weighting():
    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_ablation.py"
    spec = importlib.util.spec_from_file_location("run_ablation_weighted_huber_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    base = {
        "project": {"experiment_name": "paper_ablation"},
        "data": {"scaler": "condition_standard", "condition_clusters": {"FD001": 1}, "append_missing_mask": True},
        "model": {"name": "rast_gru_v2"},
        "training": {"loss": "asymmetric_weighted_huber", "late_life_weight": 1.5, "late_over_weight": 1.25},
        "augmentation": {"profile": "missing_noise_drift"},
    }
    cfg, model_name = module.make_variant(base, "weighted_huber_no_asymmetry")

    assert model_name == "rast_gru_v2"
    assert cfg["training"]["loss"] == "asymmetric_weighted_huber"
    assert cfg["training"]["late_life_weight"] == 1.5
    assert cfg["training"]["late_over_weight"] == 1.0


def test_severe_missing_probe_config_uses_fd004_ocm_defaults():
    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_severe_missing_probe.py"
    spec = importlib.util.spec_from_file_location("run_severe_missing_probe_for_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    base = {
        "project": {"experiment_name": "paper_main_v3_seed42", "seed": 42},
        "model": {"name": "gru"},
        "augmentation": {"profile": "none", "train_missing_rate": 0.0, "train_block_missing_rate": 0.0},
        "progress": {"batch_bar": True},
    }
    cfg = module.make_probe_config(base, seed=42, missing_rate=0.2, block_missing_rate=0.1, no_batch_bar=True)

    assert cfg["project"]["experiment_name"] == "severe_missing_probe_seed42"
    assert cfg["project"]["seed"] == 42
    assert cfg["model"]["name"] == "rast_gru_v2"
    assert cfg["augmentation"]["profile"] == "missing_noise_drift"
    assert cfg["augmentation"]["train_missing_rate"] == 0.2
    assert cfg["augmentation"]["train_block_missing_rate"] == 0.1
    assert cfg["progress"]["batch_bar"] is False


def test_mask_aware_reliability_gate_uses_observed_mask_features():
    from rul.models import MaskAwareReliabilityGate

    gate = MaskAwareReliabilityGate(input_features=6, mask_feature_count=2)
    x_observed = torch.ones(2, 5, 6)
    x_missing = x_observed.clone()
    x_missing[:, :, -2:] = 0.0

    out_observed = gate(x_observed)
    out_missing = gate(x_missing)

    assert gate.value_feature_count == 4
    assert gate.mask_feature_count == 2
    assert out_observed.shape == x_observed.shape
    assert not torch.allclose(out_observed[:, :, :4], out_missing[:, :, :4])


def test_paper_table_rows_are_sorted_and_rounded():
    from rul.reporting import build_main_table

    rows = [
        {"subset": "FD002", "model": "rast_gru", "test_rmse": 12.3456, "test_mae": 8.7654, "test_r2": 0.91, "test_nasa_score": 123.456, "parameters": 1000, "batch_inference_ms": 9.876},
        {"subset": "FD001", "model": "gru", "test_rmse": 20.0, "test_mae": 15.0, "test_r2": 0.7, "test_nasa_score": 300.0, "parameters": 500, "batch_inference_ms": 3.1},
    ]
    table = build_main_table(rows)
    assert list(table["subset"]) == ["FD001", "FD002"]
    assert table.loc[1, "test_rmse"] == 12.346
    assert table.loc[1, "batch_inference_ms"] == 9.876


def test_paper_table_bundle_writes_critical_zone_table():
    from rul.reporting import write_table_bundle

    rows = [
        {
            "experiment_name": "paper_main_v3_seed42",
            "subset": "FD004",
            "model": "rast_gru_v2",
            "allowed_for_paper": True,
            "result_status": "formal",
            "test_rmse": 12.0,
            "test_mae": 8.0,
            "test_r2": 0.7,
            "test_nasa_score": 100.0,
            "test_critical_30_rmse": 4.1234,
            "test_critical_30_late_prediction_ratio": 0.2,
            "test_critical_50_rmse": 6.5678,
            "test_critical_50_late_prediction_ratio": 0.3,
        }
    ]
    with tempfile.TemporaryDirectory() as tmp:
        outputs = write_table_bundle(rows, tmp)
        critical_path = Path(outputs["critical_zone_csv"])
        with critical_path.open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            table_rows = list(reader)

    assert critical_path.name == "table_critical_zone.csv"
    assert "test_critical_30_rmse" in reader.fieldnames
    assert "test_critical_30_late_prediction_ratio" in reader.fieldnames
    assert "test_critical_50_rmse" in reader.fieldnames
    assert table_rows[0]["model_display"] == "OCM-MST-GRU"


def test_build_paper_tables_writes_ablation_table_from_summary_csv():
    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_paper_tables.py"
    spec = importlib.util.spec_from_file_location("build_paper_tables_for_ablation_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        results_dir = root / "results"
        output_dir = root / "paper_outputs" / "tables"
        results_dir.mkdir()
        (results_dir / "ablation_runs.csv").write_text(
            "\n".join(
                [
                    "experiment_name,subset,model,ablation_variant,result_status,allowed_for_paper,test_rmse,test_mae,test_nasa_score,test_critical_30_rmse,test_critical_30_late_prediction_ratio,run_dir",
                    "ablation_no_missing_mask,FD004,rast_gru_v2,no_missing_mask,supplementary,True,18.1234,12.0,300.5678,8.2468,0.4,results/ablation_no_missing_mask/FD004/rast_gru_v2",
                    "ablation_full,FD001,rast_gru_v2,full,supplementary,True,13.4567,9.0,244.1111,3.8281,0.2,results/ablation_full/FD001/rast_gru_v2",
                ]
            )
            + "\n",
            encoding="utf-8",
        )

        outputs = module.write_ablation_table(results_dir, output_dir)
        with Path(outputs["ablation_csv"]).open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

    assert outputs["ablation_csv"].endswith("table_ablation.csv")
    assert outputs["ablation_md"].endswith("table_ablation.md")
    assert reader.fieldnames[:5] == ["experiment_name", "subset", "model", "model_display", "ablation_variant"]
    assert rows[0]["subset"] == "FD001"
    assert rows[0]["model_display"] == "OCM-MST-GRU"
    assert rows[0]["test_rmse"] == "13.457"
    assert rows[1]["ablation_variant"] == "no_missing_mask"


def test_build_paper_tables_writes_multiseed_summary_from_metrics():
    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_paper_tables.py"
    spec = importlib.util.spec_from_file_location("build_paper_tables_multiseed_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    rows = []
    for seed, rmse in [(42, 10.0), (123, 12.0), (2024, 14.0)]:
        rows.append(
            {
                "experiment_name": f"paper_main_v3_seed{seed}",
                "subset": "FD001",
                "model": "rast_gru_v2",
                "seed": seed,
                "result_status": "formal",
                "allowed_for_paper": True,
                "test_rmse": rmse,
                "test_mae": rmse - 1.0,
                "test_nasa_score": rmse * 10.0,
                "test_critical_30_late_prediction_ratio": 0.1,
                "single_sample_inference_ms": 1.0,
            }
        )

    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        outputs = module.write_multiseed_summary(rows, output_dir)
        with Path(outputs["multiseed_csv"]).open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            table_rows = list(reader)

    assert outputs["multiseed_csv"].endswith("table_multiseed_summary.csv")
    assert outputs["multiseed_md"].endswith("table_multiseed_summary.md")
    assert table_rows[0]["model_display"] == "OCM-MST-GRU"
    assert table_rows[0]["seed_count"] == "3"
    assert table_rows[0]["seeds"] == "42,123,2024"
    assert table_rows[0]["test_rmse_mean"] == "12.0"
    assert float(table_rows[0]["test_rmse_std"]) > 0.0


def test_build_paper_tables_writes_multiseed_overall_summary():
    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "build_paper_tables.py"
    spec = importlib.util.spec_from_file_location("build_paper_tables_multiseed_overall_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    rows = []
    for subset in ["FD001", "FD002"]:
        for seed in [42, 123, 2024]:
            rows.append(
                {
                    "experiment_name": f"paper_main_v3_seed{seed}",
                    "subset": subset,
                    "model": "rast_gru_v2",
                    "seed": seed,
                    "result_status": "formal",
                    "allowed_for_paper": True,
                    "test_rmse": 10.0 if subset == "FD001" else 20.0,
                    "test_mae": 8.0,
                    "test_nasa_score": 100.0,
                    "test_critical_30_late_prediction_ratio": 0.2,
                }
            )

    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp)
        outputs = module.write_multiseed_summary(rows, output_dir)
        with Path(outputs["multiseed_overall_csv"]).open("r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            table_rows = list(reader)

    assert outputs["multiseed_overall_csv"].endswith("table_multiseed_overall.csv")
    assert outputs["multiseed_overall_md"].endswith("table_multiseed_overall.md")
    assert table_rows[0]["model_display"] == "OCM-MST-GRU"
    assert table_rows[0]["subset_count"] == "2"
    assert table_rows[0]["seed_count"] == "3"
    assert table_rows[0]["test_rmse_mean"] == "15.0"
    assert float(table_rows[0]["test_rmse_std"]) == 0.0


def test_paper_metric_filtering_and_table_identity():
    from rul.reporting import build_main_table, filter_metric_rows

    rows = [
        {"experiment_name": "paper_main_seed42", "subset": "FD001", "model": "rast_gru", "test_rmse": 12.0, "test_mae": 8.0, "test_r2": 0.9, "test_nasa_score": 120.0, "parameters": 1000, "batch_inference_ms": 9.0},
        {"experiment_name": "ablation_full", "subset": "FD001", "model": "rast_gru", "test_rmse": 13.0, "test_mae": 9.0, "test_r2": 0.8, "test_nasa_score": 130.0, "parameters": 1000, "batch_inference_ms": 10.0},
        {"subset": "FD002", "model": "gru", "test_rmse": 30.0, "test_mae": 20.0, "test_r2": 0.1, "test_nasa_score": 300.0, "parameters": 500, "batch_inference_ms": 3.0},
    ]

    filtered = filter_metric_rows(rows, experiment_prefix="paper_main")
    assert len(filtered) == 1
    assert filtered[0]["experiment_name"] == "paper_main_seed42"

    table = build_main_table(filtered)
    assert list(table.columns[:3]) == ["experiment_name", "subset", "model"]
    assert table.loc[0, "experiment_name"] == "paper_main_seed42"


def test_paper_tables_keep_run_traceability_fields():
    from rul.reporting import build_main_table

    rows = [
        {
            "experiment_name": "paper_main_seed42",
            "subset": "FD001",
            "model": "rast_gru_v2",
            "created_time": "2026-07-04T18:00:00+08:00",
            "run_dir": "results/paper_main_seed42/FD001/rast_gru_v2",
            "test_rmse": 11.1111,
            "test_mae": 8.2222,
            "test_r2": 0.9,
            "test_nasa_score": 100.0,
            "parameters": 1234,
            "batch_inference_ms": 3.4567,
        }
    ]
    table = build_main_table(rows)
    assert "created_time" in table.columns
    assert "run_dir" in table.columns
    assert table.loc[0, "run_dir"].endswith("rast_gru_v2")


def test_collect_metrics_ignores_archived_results_by_default():
    from rul.reporting import collect_metrics

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        active_dir = root / "paper_main_v3_seed42" / "FD001" / "gru"
        archive_dir = root / "archive_smoke_v2_20260704_230439" / "paper_main_v3_seed42" / "FD001" / "gru"
        active_dir.mkdir(parents=True)
        archive_dir.mkdir(parents=True)
        (active_dir / "metrics.json").write_text(json.dumps({"experiment_name": "paper_main_v3_seed42", "subset": "FD001", "model": "gru"}), encoding="utf-8")
        (archive_dir / "metrics.json").write_text(json.dumps({"experiment_name": "paper_main_v3_seed42", "subset": "FD001", "model": "gru"}), encoding="utf-8")

        rows = collect_metrics(root)
        archived_rows = collect_metrics(root, include_archives=True)

    assert len(rows) == 1
    assert "archive_" not in rows[0]["run_dir"]
    assert len(archived_rows) == 2


def test_progress_messages_and_files_are_pycharm_readable():
    from rul.progress import (
        RunProgressContext,
        append_epoch_history_csv,
        format_epoch_status,
        format_run_start,
        write_progress_json,
    )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        context = RunProgressContext(
            experiment_name="paper_main_seed42",
            subset="FD001",
            model="rast_gru",
            run_dir=tmp_path / "paper_main_seed42" / "FD001" / "rast_gru",
            device="cuda",
            total_epochs=80,
            batch_size=128,
            train_batches=113,
            n_train_windows=14459,
            n_val_units=20,
            n_test_units=100,
            n_features=17,
            n_sensors=14,
        )
        banner = format_run_start(context, run_index=2, total_runs=16)
        assert "[RUN 002/016 START]" in banner
        assert "experiment=paper_main_seed42" in banner
        assert "subset=FD001 model=rast_gru" in banner
        assert "epochs=80" in banner
        assert "device=cuda" in banner
        assert str(context.run_dir) in banner

        status = format_epoch_status(
            epoch=3,
            total_epochs=80,
            train_loss=12.34567,
            val_loss=10.11111,
            best_val_loss=9.87654,
            best_epoch=2,
            patience_wait=1,
            patience=12,
            elapsed_sec=4.2,
            learning_rate=0.001,
        )
        assert "[EPOCH 003/080]" in status
        assert "train_loss=12.3457" in status
        assert "val_loss=10.1111" in status
        assert "best=9.8765@2" in status
        assert "patience=1/12" in status

        csv_path = tmp_path / "epoch_history.csv"
        append_epoch_history_csv(csv_path, {"epoch": 3, "train_loss": 12.3, "val_loss": 10.1})
        append_epoch_history_csv(csv_path, {"epoch": 4, "train_loss": 11.8, "val_loss": 9.9})
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        assert [row["epoch"] for row in rows] == ["3", "4"]

        progress_path = tmp_path / "progress.json"
        write_progress_json(progress_path, {"status": "running", "current_epoch": 3, "total_epochs": 80})
        payload = json.loads(progress_path.read_text(encoding="utf-8"))
        assert payload["status"] == "running"
        assert payload["current_epoch"] == 3


def test_protocol_lock_and_result_status_metadata():
    import yaml

    from rul.protocol import classify_result_status, load_protocol_lock

    lock_path = Path(__file__).resolve().parents[1] / "configs" / "protocol_v3.lock.yaml"
    lock = load_protocol_lock(lock_path)
    assert lock["protocol_name"] == "paper_main_v3"
    assert str(lock["protocol_version"]) == "3.0"
    assert "test_rmse" in lock["required_metrics"]
    assert "protocol_version" in lock["required_metrics"]

    raw = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    assert raw["required_selection_metric"] == "val_last_risk_score"
    assert raw["required_append_missing_mask"] is True

    formal = classify_result_status("paper_main_v3_seed42", Path("results/paper_main_v3_seed42/FD001/rast_gru_v2"), lock)
    close_prior = classify_result_status("close_prior_fd002_v3_seed42", Path("results/close_prior_fd002_v3_seed42"), lock)
    design = classify_result_status("design_sensitivity_v3_seed42_window_w20_cap125", Path("results/design_sensitivity"), lock)
    crossed = classify_result_status(
        "crossed_asym_fd001_split123_stream31415",
        Path("results/round12_new_evidence/crossed_asym_fd001_split123_stream31415"),
        lock,
    )
    preprocessing = classify_result_status(
        "cmapss_preprocess_sensor_budget8_rast_fd004_composite42",
        Path("results/round12_new_evidence/cmapss_preprocess_sensor_budget8_rast_fd004_composite42"),
        lock,
    )
    smoke = classify_result_status("smoke_protocol_v2", Path("results/smoke_protocol_v2/FD001/rast_gru_v2"), lock)

    assert formal["result_status"] == "formal"
    assert formal["allowed_for_paper"] is True
    assert formal["is_smoke"] is False
    assert close_prior["result_status"] == "supplementary" and close_prior["allowed_for_paper"] is True
    assert design["result_status"] == "supplementary" and design["allowed_for_paper"] is True
    assert crossed["result_status"] == "supplementary" and crossed["allowed_for_paper"] is True
    assert preprocessing["result_status"] == "supplementary" and preprocessing["allowed_for_paper"] is True
    assert smoke["result_status"] == "smoke"
    assert smoke["allowed_for_paper"] is False
    assert smoke["is_smoke"] is True


def test_protocol_row_validation_rejects_smoke_and_missing_fields():
    from rul.protocol import load_protocol_lock, validate_metric_row_against_protocol

    lock = load_protocol_lock(Path(__file__).resolve().parents[1] / "configs" / "protocol_v3.lock.yaml")
    valid = {
        "experiment_name": "paper_main_v3_seed42",
        "run_dir": "results/paper_main_v3_seed42/FD001/rast_gru_v2",
        "protocol_name": "paper_main_v3",
        "protocol_version": "3.0",
        "result_status": "formal",
        "allowed_for_paper": True,
        "is_smoke": False,
        "is_archive": False,
        "selection_metric": "val_last_risk_score",
        "validation_last_strategy": "simulated",
        "scaler": "condition_standard",
        "append_missing_mask": True,
        "test_rmse": 10.0,
        "test_mae": 8.0,
        "test_nasa_score": 100.0,
        "test_critical_30_rmse": 4.0,
        "test_critical_30_late_prediction_ratio": 0.2,
        "test_critical_30_severe_late_10_ratio": 0.1,
        "test_critical_50_rmse": 6.0,
        "best_epoch": 30,
        "best_epoch_val_all_loss": 9.0,
        "best_checkpoint_score": 10.0,
        "best_selection_score": 10.0,
        "checkpoint_selection_metric": "val_last_risk_score",
        "completed_epochs": 80,
        "planned_epochs": 80,
        "augmentation_seed": 42,
        "deterministic_training": True,
        "profiled_flops_batch1": 1000,
        "training_elapsed_sec": 60.0,
        "peak_gpu_memory_mb": 100.0,
    }
    assert validate_metric_row_against_protocol(valid, lock) == []

    invalid = dict(valid)
    invalid["experiment_name"] = "smoke_protocol_v2"
    invalid["result_status"] = "smoke"
    invalid["allowed_for_paper"] = False
    invalid.pop("test_rmse")
    issues = validate_metric_row_against_protocol(invalid, lock)
    assert any("forbidden" in issue or "not allowed" in issue for issue in issues)
    assert any("missing required metric" in issue for issue in issues)


def test_rul_numeric_benchmark_recomputes_metric_gold_values():
    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "benchmark" / "run_benchmark.py"
    spec = importlib.util.spec_from_file_location("run_benchmark_for_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    results = module.run_all_benchmarks(include_smoke_training=False)
    statuses = {row["name"]: row["status"] for row in results}
    assert statuses["toy_rul_metric"] == "PASS"
    assert statuses["nasa_score_direction"] == "PASS"
    assert statuses["leakage_guard"] == "PASS"


def test_eval_harness_detects_mixed_protocol_and_overclaim_text():
    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "eval_harness" / "run_evals.py"
    spec = importlib.util.spec_from_file_location("run_evals_for_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "paper_outputs").mkdir()
        (root / "paper_outputs" / "table.md").write_text("archive_old_run and experiment_name=paper_main_seed42", encoding="utf-8")
        (root / "draft.md").write_text("This is state-of-the-art and outperforms all existing methods.", encoding="utf-8")
        results = module.run_evals(project_root=root, no_write=True)

    by_name = {row["scenario"]: row for row in results}
    assert by_name["no_mixed_protocol"]["status"] == "FAIL"
    assert by_name["no_overclaim_sota"]["status"] == "FAIL"


def test_vocabulary_lint_excludes_its_generated_gate_artifact():
    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "lint_evidence_vocabulary.py"
    spec = importlib.util.spec_from_file_location("lint_evidence_vocabulary_for_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        nested_report = root / "supplementary" / "replication_package" / "reports"
        nested_report.mkdir(parents=True)
        gate_artifact = nested_report / "evidence_vocabulary_gate.json"
        gate_artifact.write_text(
            json.dumps({"issues": ["obsolete token 'suggestive'"]}),
            encoding="utf-8",
        )
        (root / "README.md").write_text("Reader-visible release text.", encoding="utf-8")

        scanned = [
            path
            for path in module.iter_files(root)
            if not module.is_generated_gate_artifact(path)
        ]

    assert gate_artifact not in scanned
    assert [path.name for path in scanned] == ["README.md"]


def test_eval_harness_does_not_treat_false_is_smoke_csv_column_as_smoke_result():
    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "eval_harness" / "run_evals.py"
    spec = importlib.util.spec_from_file_location("run_evals_csv_for_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "paper_outputs").mkdir()
        (root / "paper_outputs" / "formal.csv").write_text(
            "experiment_name,is_smoke,result_status\npaper_main_v3_seed42,False,formal\n",
            encoding="utf-8",
        )
        result = module.eval_no_mixed_protocol(root)
    assert result.status == "PASS"


def test_eval_harness_excludes_explicitly_quarantined_smoke_evidence():
    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "eval_harness" / "run_evals.py"
    spec = importlib.util.spec_from_file_location("run_evals_quarantine_for_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        quarantine = root / "paper_outputs" / "release_evidence_smoke"
        quarantine.mkdir(parents=True)
        (quarantine / "smoke.csv").write_text(
            "experiment_name,is_smoke,result_status\nsmoke_trial,True,smoke\n",
            encoding="utf-8",
        )
        result = module.eval_no_mixed_protocol(root)
    assert result.status == "PASS"


def test_eval_harness_blocks_performance_and_robustness_claims_without_evidence():
    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "eval_harness" / "run_evals.py"
    spec = importlib.util.spec_from_file_location("run_evals_claims_for_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "docs").mkdir()
        (root / "paper_outputs").mkdir()
        (root / "src" / "rul").mkdir(parents=True)
        (root / "src" / "rul" / "training.py").write_text("train_core_df val_df fit_transform_train_val_test select_features(", encoding="utf-8")
        (root / "src" / "rul" / "experiments.py").write_text("global_sensor_missing window_random_missing block_missing sensor_drift", encoding="utf-8")
        (root / "configs").mkdir()
        (root / "configs" / "protocol_v3.lock.yaml").write_text(
            "protocol_name: paper_main_v3\nprotocol_version: '3.0'\nrequired_selection_metric: val_last_risk_score\n",
            encoding="utf-8",
        )
        (root / "docs" / "research_protocol_v3.md").write_text("protocol", encoding="utf-8")
        (root / "references.bib").write_text("@article{x, title={x}}\n", encoding="utf-8")
        (root / "workflow_state.json").write_text(
            json.dumps({"main_experiment_completed": 0, "paper_claim_level": "no_performance_claim_allowed"}),
            encoding="utf-8",
        )
        (root / "docs" / "paper_outline.md").write_text(
            "Our method improves RMSE and is robust under sensor degradation.",
            encoding="utf-8",
        )
        results = module.run_evals(project_root=root, no_write=True)

    by_name = {row["scenario"]: row for row in results}
    assert by_name["no_performance_claim_without_results"]["status"] == "FAIL"
    assert by_name["no_robustness_claim_without_robustness_csv"]["status"] == "FAIL"


def test_eval_harness_requires_ablation_evidence_after_claims_are_allowed():
    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "eval_harness" / "run_evals.py"
    spec = importlib.util.spec_from_file_location("run_evals_ablation_claims_for_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "docs").mkdir()
        (root / "paper_outputs").mkdir()
        (root / "src" / "rul").mkdir(parents=True)
        (root / "src" / "rul" / "training.py").write_text("train_core_df val_df fit_transform_train_val_test select_features(", encoding="utf-8")
        (root / "src" / "rul" / "experiments.py").write_text("global_sensor_missing window_random_missing block_missing sensor_drift", encoding="utf-8")
        (root / "configs").mkdir()
        (root / "configs" / "protocol_v3.lock.yaml").write_text(
            "protocol_name: paper_main_v3\nprotocol_version: '3.0'\nrequired_selection_metric: val_last_risk_score\n",
            encoding="utf-8",
        )
        (root / "docs" / "research_protocol_v3.md").write_text("protocol", encoding="utf-8")
        (root / "references.bib").write_text("@article{x, title={x}}\n", encoding="utf-8")
        (root / "workflow_state.json").write_text(
            json.dumps({"main_experiment_completed": 24, "paper_claim_level": "single_seed_claim_allowed"}),
            encoding="utf-8",
        )
        (root / "docs" / "paper_outline.md").write_text(
            "The method uses explicit missing mask and condition-aware normalization.",
            encoding="utf-8",
        )
        results = module.run_evals(project_root=root, no_write=True)

    by_name = {row["scenario"]: row for row in results}
    assert by_name["no_missing_ablation_for_mask_claim"]["status"] == "FAIL"
    assert by_name["no_condition_norm_claim_without_fd004_ablation"]["status"] == "FAIL"


def test_reporting_adds_display_names_and_evidence_level():
    from rul.reporting import build_main_table, evidence_level_for_row, model_display_name

    assert model_display_name("rast_gru_v2") == "OCM-MST-GRU"
    assert model_display_name("cnn_lstm") == "CNN-LSTM"
    assert model_display_name("attention_gru") == "Attention-GRU"
    assert model_display_name("transformer_lite") == "Transformer-lite"
    assert model_display_name("dual_attention_tcn") == "Dual-attention TCN"
    assert model_display_name("sensor_graph_gru") == "SensorGraph-GRU"
    assert model_display_name("quantile_gru") == "Quantile-GRU"
    row = {
        "experiment_name": "paper_main_v3_seed42",
        "subset": "FD004",
        "model": "rast_gru_v2",
        "protocol_version": "3.0",
        "result_status": "formal",
        "allowed_for_paper": True,
        "seed": 42,
        "test_rmse": 10.0,
        "test_mae": 8.0,
        "test_r2": 0.9,
        "test_nasa_score": 100.0,
        "parameters": 123,
        "batch_inference_ms": 1.0,
    }
    assert evidence_level_for_row(row) == "formal_single_seed"
    table = build_main_table([row])
    assert table.loc[0, "model_display"] == "OCM-MST-GRU"
    assert table.loc[0, "evidence_level"] == "formal_single_seed"


def test_robustness_relative_degradation_rates_are_added():
    from rul.experiments import add_relative_degradation

    rows = add_relative_degradation(
        [
            {"scenario": "clean", "level": 0.0, "rmse": 10.0, "nasa_score": 100.0},
            {"scenario": "sensor_drift", "level": 0.05, "rmse": 12.0, "nasa_score": 150.0},
        ]
    )

    assert rows[0]["relative_rmse_increase"] == 0.0
    assert rows[0]["relative_nasa_score_increase"] == 0.0
    assert abs(rows[1]["relative_rmse_increase"] - 0.2) < 1e-9
    assert abs(rows[1]["relative_nasa_score_increase"] - 0.5) < 1e-9


def test_statistical_tests_include_safety_metric_rows():
    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_statistical_tests.py"
    spec = importlib.util.spec_from_file_location("run_statistical_tests_for_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        for model, preds in {"gru": [40, 35, 30, 25, 20], "rast_gru_v2": [30, 25, 20, 15, 10]}.items():
            run_dir = root / "paper_main_v3_seed42" / "FD001" / model
            run_dir.mkdir(parents=True)
            rows = ["unit_id,end_cycle,true_rul,pred_rul"]
            for idx, pred in enumerate(preds, start=1):
                rows.append(f"{idx},100,{10 + idx},{pred}")
            (run_dir / "test_predictions.csv").write_text("\n".join(rows) + "\n", encoding="utf-8")

        results = module.compare_models(
            root,
            "paper_main_v3_seed42",
            ["FD001"],
            "rast_gru_v2",
            ["gru"],
        )

    metric_names = {row["metric"] for row in results}
    assert {
        "abs_error",
        "nasa_score_contribution",
        "critical_30_abs_error",
        "critical_30_late_indicator",
    }.issubset(metric_names)


def test_statistical_tests_default_to_paper_main_experiment_for_pycharm_run():
    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_statistical_tests.py"
    spec = importlib.util.spec_from_file_location("run_statistical_tests_defaults_for_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    parser = module.build_arg_parser()
    args = parser.parse_args([])
    assert args.experiment_name == "paper_main_v3_seed42"
    assert args.output.endswith("table_statistical_tests.csv")


def test_result_brief_includes_ablation_robustness_and_claim_limits():
    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "write_result_brief.py"
    spec = importlib.util.spec_from_file_location("write_result_brief_for_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    with tempfile.TemporaryDirectory() as tmp:
        tables_dir = Path(tmp)
        (tables_dir / "table_main_accuracy.csv").write_text(
            "subset,model,test_rmse,test_mae,test_nasa_score\nFD004,rast_gru_v2,20.0,15.0,500.0\n",
            encoding="utf-8",
        )
        (tables_dir / "table_complexity.csv").write_text(
            "subset,model,parameters,test_rmse\nFD004,rast_gru_v2,53897,20.0\n",
            encoding="utf-8",
        )
        (tables_dir / "table_critical_zone.csv").write_text(
            "subset,model,test_critical_30_rmse,test_critical_30_late_prediction_ratio,test_critical_50_rmse\nFD004,rast_gru_v2,6.0,0.2,8.0\n",
            encoding="utf-8",
        )
        (tables_dir / "table_ablation.csv").write_text(
            "subset,model,ablation_variant,test_rmse,test_nasa_score,test_critical_30_late_prediction_ratio\nFD004,rast_gru_v2,no_missing_mask,25.0,800.0,0.6\n",
            encoding="utf-8",
        )
        (tables_dir / "table_robustness.csv").write_text(
            "subset,model,scenario,level,rmse,relative_rmse_increase,nasa_score,relative_nasa_score_increase,critical_30_late_prediction_ratio\nFD004,rast_gru_v2,global_sensor_missing,0.3,40.0,1.0,2000.0,3.0,0.8\n",
            encoding="utf-8",
        )
        (tables_dir / "table_statistical_tests.csv").write_text(
            "subset,baseline_model,proposed_model,metric,p_value,effect_size,mean_baseline_metric,mean_proposed_metric\nFD004,tcn_gru,rast_gru_v2,abs_error,0.03,0.2,18.0,15.0\n",
            encoding="utf-8",
        )
        (tables_dir / "table_multiseed_summary.csv").write_text(
            "subset,model,model_display,seed_count,seeds,test_rmse_mean,test_rmse_std,test_nasa_score_mean,test_nasa_score_std\nFD004,rast_gru_v2,OCM-MST-GRU,3,\"42,123,2024\",15.0,0.5,1200.0,50.0\n",
            encoding="utf-8",
        )
        (tables_dir / "table_multiseed_overall.csv").write_text(
            "model,model_display,subset_count,seed_count,seeds,test_rmse_mean,test_rmse_std,test_nasa_score_mean,test_nasa_score_std\nrast_gru_v2,OCM-MST-GRU,4,3,\"42,123,2024\",14.1,0.9,679.3,420.0\n",
            encoding="utf-8",
        )
        (tables_dir / "table_severe_missing_probe.csv").write_text(
            "experiment_name,subset,model,model_display,result_status,allowed_for_paper,test_rmse,test_nasa_score,test_critical_30_late_prediction_ratio\nsevere_missing_probe_seed42,FD004,rast_gru_v2,OCM-MST-GRU,draft,False,17.112,1483.12,0.038\n",
            encoding="utf-8",
        )

        brief = module.build_result_brief(tables_dir)

    assert "## Ablation Evidence" in brief
    assert "## Robustness Stress Cases" in brief
    assert "## Statistical Tests" in brief
    assert "## Multi-Seed Evidence" in brief
    assert "## Multi-Seed Overall Evidence" in brief
    assert "## Severe Missing Draft Probe" in brief
    assert "not part of the formal main-result table" in brief
    assert "severe missing" in brief
    assert "accuracy-risk-efficiency trade-off" in brief
    assert "five-composite-seed evidence" in brief
    assert "Current formal claim level is single-seed" not in brief
    assert "Do not claim" in brief
    assert "FD004 contains 249 training engines and 248 test engines" in brief


def test_quality_gate_stage_defaults_and_workflow_claim_level():
    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "paper_quality_gate.py"
    spec = importlib.util.spec_from_file_location("paper_quality_gate_for_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    preflight = module.default_gate_options("preflight")
    final = module.default_gate_options("final")
    assert preflight["expected_count"] == 0
    assert preflight["min_epochs"] is None
    assert final["expected_count"] == 260
    assert final["min_epochs"] >= 30
    assert module.claim_level_for_state(completed_runs=0, expected_runs=24, passed=True, stage="preflight") == "no_performance_claim_allowed"
    assert module.claim_level_for_state(completed_runs=24, expected_runs=24, passed=True, stage="final") == "single_seed_claim_allowed"


def test_quality_gate_claim_level_upgrades_when_multiseed_ready():
    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "paper_quality_gate.py"
    spec = importlib.util.spec_from_file_location("paper_quality_gate_claim_for_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    single = {"paper_tables_ready": True, "multi_seed_ready": False}
    multi = {"paper_tables_ready": True, "multi_seed_ready": True}

    assert module.claim_level_for_evidence(24, 24, True, "final", single) == "single_seed_claim_allowed"
    assert module.claim_level_for_evidence(24, 24, True, "final", multi) == "multi_seed_claim_allowed"


def test_quality_gate_filters_main_results_separately_from_ablation():
    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "paper_quality_gate.py"
    spec = importlib.util.spec_from_file_location("paper_quality_gate_filters_for_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    rows = [
        {"experiment_name": "paper_main_v3_seed42", "subset": "FD001", "model": "gru"},
        {"experiment_name": "paper_main_v3_seed42", "subset": "FD001", "model": "rast_gru_v2"},
        {"experiment_name": "ablation_no_missing_mask", "subset": "FD004", "model": "rast_gru_v2"},
    ]

    assert module.resolve_gate_experiment_names("final", None, None) is None
    assert module.select_rows_for_gate(rows, stage="final", experiment_names=None, experiment_prefix=None) == rows[:2]
    assert module.select_rows_for_gate(rows, stage="preflight", experiment_names=None, experiment_prefix=None) == []


def test_quality_gate_evidence_statuses_reflect_generated_outputs():
    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "paper_quality_gate.py"
    spec = importlib.util.spec_from_file_location("paper_quality_gate_evidence_for_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        results_dir = root / "results"
        tables_dir = root / "paper_outputs" / "tables"
        results_dir.mkdir(parents=True)
        tables_dir.mkdir(parents=True)
        (results_dir / "ablation_runs.csv").write_text("subset,ablation_variant\nFD004,no_missing_mask\n", encoding="utf-8")
        for name in [
            "table_main_accuracy.csv",
            "table_complexity.csv",
            "table_critical_zone.csv",
            "table_ablation.csv",
            "table_robustness.csv",
            "table_statistical_tests.csv",
        ]:
            (tables_dir / name).write_text("subset,model\nFD004,rast_gru_v2\n", encoding="utf-8")
        for seed in module.MULTISEED_REQUIRED_SEEDS:
            for subset in module.MULTISEED_REQUIRED_SUBSETS:
                for model in module.MULTISEED_REQUIRED_MODELS:
                    run_dir = results_dir / f"paper_main_v3_seed{seed}" / subset / model
                    run_dir.mkdir(parents=True)
                    (run_dir / "metrics.json").write_text(
                        json.dumps(
                            {
                                "experiment_name": f"paper_main_v3_seed{seed}",
                                "subset": subset,
                                "model": model,
                                "seed": seed,
                                "result_status": "formal",
                                "allowed_for_paper": True,
                            }
                        ),
                        encoding="utf-8",
                    )

        statuses = module.evidence_statuses(root)

    assert statuses["ablation_status"] == "complete"
    assert statuses["robustness_status"] == "complete"
    assert statuses["statistical_test_status"] == "complete"
    assert statuses["tables_status"] == "complete"
    assert statuses["multi_seed_status"] == "complete"
    assert statuses["ablation_ready"] is True
    assert statuses["robustness_ready"] is True
    assert statuses["paper_tables_ready"] is True
    assert statuses["multi_seed_ready"] is True


def test_quality_gate_importable_without_scripts_on_sys_path():
    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / "scripts" / "paper_quality_gate.py"
    code = f"""
import importlib.util
import sys
from pathlib import Path

project_root = Path({str(project_root)!r})
script_path = Path({str(script_path)!r})
sys.path = [
    item for item in sys.path
    if item not in {{str(project_root), str(project_root / 'scripts'), str(project_root / 'src')}}
]
spec = importlib.util.spec_from_file_location('paper_quality_gate_external_import_test', script_path)
module = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = module
spec.loader.exec_module(module)
print(module.default_gate_options('final')['expected_count'])
"""
    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run([sys.executable, "-c", code], cwd=tmp, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert proc.returncode == 0, proc.stdout
    assert "260" in proc.stdout


def test_batch_robustness_resolves_existing_run_dirs():
    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_robustness.py"
    spec = importlib.util.spec_from_file_location("run_robustness_for_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run_dir = root / "paper_main_v3_seed42" / "FD001" / "rast_gru_v2"
        run_dir.mkdir(parents=True)
        resolved = module.resolve_run_dirs(
            root,
            explicit_run_dirs=[],
            experiment_name="paper_main_v3_seed42",
            subsets=["FD001"],
            models=["rast_gru_v2"],
        )
    assert resolved == [run_dir]


def test_robustness_defaults_to_paper_main_key_runs_for_pycharm_run():
    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_robustness.py"
    spec = importlib.util.spec_from_file_location("run_robustness_defaults_for_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    parser = module.build_arg_parser()
    args = parser.parse_args([])
    assert args.experiment_name == "paper_main_v3_seed42"
    assert args.subsets == ["FD001", "FD004"]
    assert args.models == ["gru", "tcn_gru", "rast_gru", "rast_gru_v2"]

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        expected = []
        for subset in args.subsets:
            for model in args.models:
                run_dir = root / "paper_main_v3_seed42" / subset / model
                run_dir.mkdir(parents=True)
                expected.append(run_dir)
        resolved = module.resolve_run_dirs(
            root,
            explicit_run_dirs=args.run_dir,
            experiment_name=args.experiment_name,
            subsets=args.subsets,
            models=args.models,
        )
    assert resolved == expected


def test_condition_cluster_plot_function_writes_png():
    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "plot_condition_clusters.py"
    spec = importlib.util.spec_from_file_location("plot_condition_clusters_for_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    df = pd.DataFrame(
        {
            "setting_1": [-1.0, -0.8, 0.8, 1.0],
            "setting_2": [-1.0, -0.9, 0.9, 1.0],
            "setting_3": [1.0, 1.0, 1.0, 1.0],
        }
    )
    with tempfile.TemporaryDirectory() as tmp:
        out = Path(tmp) / "condition_clusters_FD002.png"
        module.plot_condition_clusters(df, subset="FD002", n_clusters=2, output_path=out)
        assert out.exists()
        assert out.stat().st_size > 0


def test_condition_cluster_plot_uses_headless_backend_and_project_mpl_cache():
    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / "scripts" / "plot_condition_clusters.py"
    expected_cache = project_root / ".mpl_cache"
    code = f"""
import importlib.util
import os
import sys
from pathlib import Path

os.environ.pop('MPLCONFIGDIR', None)
script_path = Path({str(script_path)!r})
spec = importlib.util.spec_from_file_location('plot_condition_clusters_headless_test', script_path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
import matplotlib
print(matplotlib.get_backend())
expected_cache = Path({str(expected_cache)!r}).resolve()
actual_cache = Path(os.environ.get('MPLCONFIGDIR', '')).resolve()
print(f"CACHE_MATCH={{actual_cache == expected_cache}}")
"""
    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run([sys.executable, "-c", code], cwd=tmp, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    assert proc.returncode == 0, proc.stdout
    assert "agg" in proc.stdout.lower()
    assert "CACHE_MATCH=True" in proc.stdout


def test_metric_bar_plot_accepts_model_only_summary_tables():
    from rul.plotting import plot_metric_bars

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        summary = root / "table_multiseed_overall.csv"
        output = root / "fig_test_rmse_mean.png"
        summary.write_text(
            "model,model_display,test_rmse_mean\nrast_gru_v2,OCM-MST-GRU,14.1\ngru,GRU,15.1\n",
            encoding="utf-8",
        )

        plot_metric_bars(summary, output, metric="test_rmse_mean")

        assert output.exists()
        assert output.stat().st_size > 0


def test_eval_harness_requires_dataset_integrity_evidence_for_claim():
    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "eval_harness" / "run_evals.py"
    spec = importlib.util.spec_from_file_location("run_evals_dataset_integrity_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "README.md").write_text("This project uses validated data integrity checks for canonical files.", encoding="utf-8")
        (root / "references.bib").write_text("@article{x, title={x}}\n", encoding="utf-8")

        rows = module.run_evals(root, no_write=True)
        by_name = {row["scenario"]: row for row in rows}
        assert by_name["dataset_integrity_claim_check"]["status"] == "FAIL"

        reports = root / "reports"
        reports.mkdir()
        (reports / "data_validation_report.md").write_text(
            "# C-MAPSS Data Validation Report\n\nOverall status: PASS\nChecks: 88 total, 0 failed\n",
            encoding="utf-8",
        )
        (reports / "raw_file_checksums.sha256").write_text("abcd  train_FD001.txt\n", encoding="utf-8")
        (reports / "dataset_sufficiency_report.md").write_text("# Dataset Sufficiency and Quality Analysis\n", encoding="utf-8")

        rows = module.run_evals(root, no_write=True)
        by_name = {row["scenario"]: row for row in rows}
        assert by_name["dataset_integrity_claim_check"]["status"] == "PASS"


def test_eval_harness_requires_critical_zone_and_statistics_evidence_after_claims_are_allowed():
    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "eval_harness" / "run_evals.py"
    spec = importlib.util.spec_from_file_location("run_evals_submission_evidence_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "docs").mkdir()
        (root / "paper_outputs" / "tables").mkdir(parents=True)
        (root / "src" / "rul").mkdir(parents=True)
        (root / "src" / "rul" / "training.py").write_text("train_core_df val_df fit_transform_train_val_test select_features(", encoding="utf-8")
        (root / "src" / "rul" / "experiments.py").write_text("global_sensor_missing window_random_missing block_missing sensor_drift", encoding="utf-8")
        (root / "configs").mkdir()
        (root / "configs" / "protocol_v3.lock.yaml").write_text(
            "protocol_name: paper_main_v3\nprotocol_version: '3.0'\nrequired_selection_metric: val_last_risk_score\n",
            encoding="utf-8",
        )
        (root / "docs" / "research_protocol_v3.md").write_text("protocol", encoding="utf-8")
        (root / "references.bib").write_text("@article{x, title={x}}\n", encoding="utf-8")
        (root / "workflow_state.json").write_text(
            json.dumps({"main_experiment_completed": 24, "paper_claim_level": "single_seed_claim_allowed"}),
            encoding="utf-8",
        )
        (root / "docs" / "paper_outline.md").write_text(
            "We report critical-zone near-failure safety-critical analysis and statistically significant results.",
            encoding="utf-8",
        )

        rows = module.run_evals(root, no_write=True)
        by_name = {row["scenario"]: row for row in rows}
        assert by_name["no_critical_zone_claim_without_table"]["status"] == "FAIL"
        assert by_name["no_statistics_claim_without_wilcoxon"]["status"] == "FAIL"

        (root / "paper_outputs" / "tables" / "table_critical_zone.csv").write_text(
            "subset,model,test_critical_30_rmse,test_critical_30_late_prediction_ratio,test_critical_50_rmse\nFD004,rast_gru_v2,4,0.2,6\n",
            encoding="utf-8",
        )
        (root / "paper_outputs" / "tables" / "table_statistical_tests.csv").write_text(
            "subset,metric,p_value,effect_size\nFD004,abs_error,0.01,0.5\n",
            encoding="utf-8",
        )

        rows = module.run_evals(root, no_write=True)
        by_name = {row["scenario"]: row for row in rows}
        assert by_name["no_critical_zone_claim_without_table"]["status"] == "PASS"
        assert by_name["no_statistics_claim_without_wilcoxon"]["status"] == "PASS"


def test_paper_main_run_summaries_keep_latest_seed_and_all_files():
    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "run_paper_experiments.py"
    spec = importlib.util.spec_from_file_location("run_paper_experiments_summary_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    with tempfile.TemporaryDirectory() as tmp:
        results_dir = Path(tmp)
        first = [{"experiment_name": "paper_main_v3_seed42", "seed": 42, "subset": "FD001", "model": "gru", "test_rmse": 1.0}]
        second = [{"experiment_name": "paper_main_v3_seed123", "seed": 123, "subset": "FD001", "model": "gru", "test_rmse": 2.0}]
        module.write_paper_main_summaries(first, results_dir)
        outputs = module.write_paper_main_summaries(second, results_dir)

        assert (results_dir / "paper_main_runs.csv").exists()
        assert (results_dir / "paper_main_runs_latest.csv").exists()
        assert (results_dir / "paper_main_runs_seed42.csv").exists()
        assert (results_dir / "paper_main_runs_seed123.csv").exists()
        assert (results_dir / "paper_main_runs_all.csv").exists()
        assert outputs["latest"].name == "paper_main_runs_latest.csv"
        with (results_dir / "paper_main_runs_all.csv").open("r", encoding="utf-8") as f:
            all_rows = list(csv.DictReader(f))
        assert {row["seed"] for row in all_rows} == {"42", "123"}


def test_clean_paper_outputs_explains_empty_state_and_next_steps():
    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "clean_paper_outputs.py"
    spec = importlib.util.spec_from_file_location("clean_paper_outputs_for_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / "results" / "paper_main_v3_seed42" / "FD001" / "gru").mkdir(parents=True)
        (root / "results" / "paper_main_v3_seed42" / "FD001" / "gru" / "progress.json").write_text(
            json.dumps({"status": "running", "current_epoch": 31, "total_epochs": 80}),
            encoding="utf-8",
        )
        capture = StringIO()
        with redirect_stdout(capture):
            summary = module.clean_generated_outputs(root, dry_run=False)

    output = capture.getvalue()
    assert summary["removed_count"] == 0
    assert "[CLEAN] no generated paper output files found" in output
    assert "formal_metrics=0/24" in output
    assert "This is normal before build_paper_tables.py has produced tables." in output
    assert "run_paper_experiments.py" in output
    assert "check_result_consistency.py" in output


def test_replication_readme_documents_data_integrity_checks():
    root = Path(__file__).resolve().parents[1]
    candidates = [root / "README_REPLICATION.md", root / "replication_package" / "README_REPLICATION.md"]
    readme_path = next((path for path in candidates if path.exists()), candidates[-1])
    readme = readme_path.read_text(encoding="utf-8")
    assert "Data Integrity" in readme
    assert "DATA_VALIDATION_PASS" in readme
    assert "88 total, 0 failed" in readme
    assert "FD004: 249 train engines, 248 test engines" in readme


def test_replication_manifest_hashes_metrics_and_predictions():
    import importlib.util

    script_path = Path(__file__).resolve().parents[1] / "scripts" / "create_replication_package.py"
    spec = importlib.util.spec_from_file_location("create_replication_package_for_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        run_dir = root / "results" / "paper_main_v3_seed42" / "FD001" / "rast_gru_v2"
        run_dir.mkdir(parents=True)
        metrics_path = run_dir / "metrics.json"
        predictions_path = run_dir / "test_predictions.csv"
        config_path = run_dir / "run_config.yaml"
        metrics_path.write_text(
            json.dumps(
                {
                    "experiment_name": "paper_main_v3_seed42",
                    "subset": "FD001",
                    "model": "rast_gru_v2",
                    "seed": 42,
                    "protocol_version": "3.0",
                    "allowed_for_paper": True,
                }
            ),
            encoding="utf-8",
        )
        predictions_path.write_text("unit_id,end_cycle,true_rul,pred_rul\n1,10,20,19\n", encoding="utf-8")
        config_path.write_text("project:\n  seed: 42\n", encoding="utf-8")

        manifest = module.build_results_manifest(root / "results")

    assert len(manifest) == 1
    row = manifest[0]
    assert row["experiment_name"] == "paper_main_v3_seed42"
    assert row["allowed_for_paper"] is True
    assert len(row["sha256_metrics"]) == 64
    assert len(row["sha256_predictions"]) == 64


def test_ablation_training_budget_matches_formal_main_protocol():
    from rul.config import load_config

    project_root = Path(__file__).resolve().parents[1]
    main = load_config(project_root / "configs" / "paper_main.yaml")
    ablation = load_config(project_root / "configs" / "paper_ablation.yaml")
    for key in [
        "epochs",
        "batch_size",
        "learning_rate",
        "weight_decay",
        "patience",
        "selection_metric",
        "loss",
        "huber_delta",
        "late_life_threshold",
        "late_life_weight",
        "late_over_weight",
    ]:
        assert ablation["training"][key] == main["training"][key]


def test_multiseed_ablation_full_row_is_loaded_from_formal_main_result():
    import importlib.util
    import yaml

    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / "scripts" / "run_multiseed_ablation.py"
    spec = importlib.util.spec_from_file_location("run_multiseed_ablation_reference_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    with tempfile.TemporaryDirectory() as tmp:
        results_root = Path(tmp) / "results"
        run_dir = results_root / "paper_main_v3_seed42" / "FD004" / "rast_gru_v2"
        run_dir.mkdir(parents=True)
        expected = {
            "data": {"window_size": 30},
            "model": {"name": "rast_gru_v2"},
            "training": {"epochs": 80, "patience": 12, "selection_metric": "val_last_risk_score"},
            "augmentation": {"seed": 42, "deterministic": True},
        }
        (run_dir / "run_config.yaml").write_text(yaml.safe_dump(expected, sort_keys=False), encoding="utf-8")
        (run_dir / "metrics.json").write_text(
            json.dumps(
                {
                    "experiment_name": "paper_main_v3_seed42",
                    "result_status": "formal",
                    "allowed_for_paper": True,
                    "test_rmse": 15.25,
                }
            ),
            encoding="utf-8",
        )

        row = module._main_full_metrics(results_root, 42, "FD004", expected)
        assert row["experiment_name"] == "paper_main_v3_seed42"
        assert row["test_rmse"] == 15.25
        assert row["ablation_full_reference"] is True
        assert Path(row["run_dir"]) == run_dir


def test_replication_copy_keeps_checkpoint_binary_bytes_unchanged():
    import importlib.util

    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / "scripts" / "create_replication_package.py"
    spec = importlib.util.spec_from_file_location("replication_binary_copy_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        src = root / "source" / "best_model.pt"
        dst = root / "package" / "best_model.pt"
        src.parent.mkdir(parents=True)
        payload = b"PK\x03\x04\x00\xff\x80\r\n\x00binary-checkpoint"
        src.write_bytes(payload)
        module.copy_file_clean(src, dst, root / "source")
        assert dst.read_bytes() == payload


def test_replication_text_sanitizer_normalizes_known_workspace_roots():
    import importlib.util
    import json

    actual_root = Path(__file__).resolve().parents[1]
    script_path = actual_root / "scripts" / "create_replication_package.py"
    spec = importlib.util.spec_from_file_location("replication_path_sanitizer_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    project_root = Path(r"D:\ResearchWorkspace\rul-project")
    escaped_root = json.dumps(str(project_root), ensure_ascii=True)[1:-1]
    source = "\n".join(
        (
            str(project_root / "results" / "run" / "metrics.json"),
            r"R:\rs_tcn_gru_rul\results\audit\metrics.json",
            (project_root.parent / "正文" / "main_revised.tex").as_posix(),
            escaped_root + r"\\results",
        )
    )
    sanitized = module.sanitize_text(source, project_root).replace("\\", "/")

    assert "results/run/metrics.json" in sanitized
    assert "results/audit/metrics.json" in sanitized
    assert "manuscript/main_revised.tex" in sanitized
    assert "D:" not in sanitized
    assert "R:" not in sanitized


def test_checkpoint_integrity_rejects_utf8_bom_prefix():
    import importlib.util

    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / "scripts" / "check_checkpoint_integrity.py"
    spec = importlib.util.spec_from_file_location("checkpoint_integrity_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "best_model.pt"
        path.write_bytes(b"\xef\xbb\xbfPK\x03\x04")
        assert "BOM" in module.check_checkpoint(path)


def test_formal_benchmark_reuses_only_loadable_checkpoints():
    import importlib.util

    project_root = Path(__file__).resolve().parents[1]
    script_path = project_root / "scripts" / "run_formal_benchmark.py"
    spec = importlib.util.spec_from_file_location("formal_checkpoint_reuse_test", script_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp) / "run"
        run_dir.mkdir()
        (run_dir / "metrics.json").write_text(json.dumps({"result_status": "formal"}), encoding="utf-8")
        checkpoint_path = run_dir / "best_model.pt"
        torch.save({"model_state": {"weight": torch.ones(1)}}, checkpoint_path)
        assert module.existing_formal_benchmark_run_metrics(run_dir) is not None
        checkpoint_path.write_bytes(b"\xef\xbb\xbf" + checkpoint_path.read_bytes())
        assert module.existing_formal_benchmark_run_metrics(run_dir) is None


def test_ncmapss_condition_normalization_and_unit_windows_are_leakage_safe_shapes():
    from rul.ncmapss import ConditionClusterNormalizer, make_unit_windows

    rng = np.random.default_rng(42)
    train_settings = np.vstack([rng.normal(-2.0, 0.1, (40, 2)), rng.normal(2.0, 0.1, (40, 2))])
    train_sensors = np.vstack([rng.normal(10.0, 1.0, (40, 3)), rng.normal(30.0, 2.0, (40, 3))])
    normalizer = ConditionClusterNormalizer(n_clusters=2, random_state=42).fit(train_settings, train_sensors)
    transformed = normalizer.transform(train_settings, train_sensors)
    assert transformed.shape == (80, 5)
    assert np.isfinite(transformed).all()

    units = np.repeat([2, 5], 40)
    labels = np.tile(np.arange(40, 0, -1, dtype=np.float32), 2)
    windows = make_unit_windows(transformed, labels, units, window_size=10, stride=2)
    assert windows.x.shape == (32, 10, 5)
    assert windows.y.shape == (32,)
    assert set(np.unique(windows.unit_ids)) == {2, 5}
    assert np.all(windows.end_cycles[:16] == np.arange(9, 40, 2))


def test_ncmapss_cache_uses_json_metadata_without_pickle():
    from rul.ncmapss import NCMAPSSPrepared, load_prepared_cache, save_prepared_cache
    from rul.preprocessing import WindowData

    x = np.ones((3, 4, 2), dtype=np.float32)
    split = WindowData(
        x=x,
        y=np.asarray([3.0, 2.0, 1.0], dtype=np.float32),
        unit_ids=np.asarray([2, 2, 2]),
        end_cycles=np.asarray([3, 4, 5]),
    )
    prepared = NCMAPSSPrepared(
        train=split,
        val=split,
        test=split,
        feature_names=["setting", "sensor"],
        sensor_indices=[1],
        metadata={"feature_names": ["setting", "sensor"], "sensor_indices": [1], "window_size": 4},
    )
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cache.npz"
        save_prepared_cache(prepared, path)
        with np.load(path, allow_pickle=False) as payload:
            assert payload["metadata"].dtype.kind in {"U", "S"}
        loaded = load_prepared_cache(path)
        assert loaded.metadata["window_size"] == 4
        assert loaded.train.x.shape == (3, 4, 2)


def test_late_tail_metrics_distinguish_frequency_magnitude_and_cvar():
    from build_advanced_evidence import prediction_metrics

    frame = pd.DataFrame(
        {
            "unit_id": [1, 2, 3, 4, 5],
            "true_rul": [10.0, 20.0, 30.0, 40.0, 50.0],
            "pred_rul": [10.0, 25.0, 50.0, 60.0, 40.0],
        }
    )
    metrics = prediction_metrics(frame)
    assert np.isclose(metrics["lpr30"], 2 / 3)
    assert np.isclose(metrics["mle30"], 25 / 3)
    assert np.isclose(metrics["conditional_mle30"], 12.5)
    assert metrics["late_q95_30"] > metrics["late_q90_30"]
    assert np.isclose(metrics["late_cvar95_30"], 20.0)


def test_crossed_rmse_recomputes_square_mean_root_inside_each_draw():
    from analyze_normalized_perturbation_evidence import sampled_seed_metrics

    truth = np.array([0.0, 0.0])
    predictions = np.array([[0.0, 2.0], [1.0, 1.0]])
    counts = np.array([[2, 0], [0, 2]], dtype=np.int16)
    observed = sampled_seed_metrics(truth, predictions, counts, "rmse")
    expected = np.array([[0.0, 2.0], [1.0, 1.0]])
    assert np.allclose(observed, expected)
    assert not np.isclose(observed[0, 1], np.sqrt(np.mean([0.0, 4.0])))


def test_threshold_bootstrap_preserves_engine_pairing():
    from analyze_threshold_sensitivity import paired_threshold_bootstrap

    with tempfile.TemporaryDirectory() as tmp:
        results = Path(tmp)
        for seed in [42, 123]:
            baseline_dir = results / f"paper_main_v3_seed{seed}" / "FD004" / "gru"
            proposed_dir = results / f"paper_main_v3_seed{seed}" / "FD004" / "rast_gru_v2"
            baseline_dir.mkdir(parents=True)
            proposed_dir.mkdir(parents=True)
            common = {"unit_id": [1, 2, 3, 4], "true_rul": [5.0, 10.0, 20.0, 40.0]}
            pd.DataFrame({**common, "pred_rul": [15.0, 20.0, 30.0, 40.0]}).to_csv(
                baseline_dir / "test_predictions.csv", index=False
            )
            pd.DataFrame({**common, "pred_rul": [5.0, 10.0, 20.0, 40.0]}).to_csv(
                proposed_dir / "test_predictions.csv", index=False
            )
        result = paired_threshold_bootstrap(
            results,
            seeds=[42, 123],
            subsets=["FD004"],
            models=["gru", "rast_gru_v2"],
            rul_thresholds=[30.0],
            late_error_thresholds=[0.0],
            proposed_model="rast_gru_v2",
            reps=100,
        )
        assert len(result) == 1
        assert np.isclose(result.iloc[0]["paired_difference_baseline_minus_proposed"], 1.0)
        assert np.isclose(result.iloc[0]["probability_proposed_lower"], 1.0)
        assert np.isclose(result.iloc[0]["probability_tie"], 0.0)
        assert np.isclose(result.iloc[0]["probability_proposed_higher"], 0.0)


def test_threshold_bootstrap_reports_exact_ties_separately():
    from analyze_threshold_sensitivity import paired_threshold_bootstrap

    with tempfile.TemporaryDirectory() as tmp:
        results = Path(tmp)
        for model in ["gru", "rast_gru_v2"]:
            run_dir = results / "paper_main_v3_seed42" / "FD004" / model
            run_dir.mkdir(parents=True)
            pd.DataFrame(
                {
                    "unit_id": [1, 2, 3],
                    "true_rul": [5.0, 10.0, 20.0],
                    "pred_rul": [5.0, 10.0, 20.0],
                }
            ).to_csv(run_dir / "test_predictions.csv", index=False)
        result = paired_threshold_bootstrap(
            results,
            seeds=[42],
            subsets=["FD004"],
            models=["gru", "rast_gru_v2"],
            rul_thresholds=[30.0],
            late_error_thresholds=[0.0],
            proposed_model="rast_gru_v2",
            reps=50,
        )
        assert np.isclose(result.iloc[0]["probability_proposed_lower"], 0.0)
        assert np.isclose(result.iloc[0]["probability_tie"], 1.0)
        assert np.isclose(result.iloc[0]["probability_proposed_higher"], 0.0)


def test_five_seed_ablation_covers_every_declared_ocm_module_family():
    from run_multiseed_ablation import MAJOR_FD004_VARIANTS

    required = {
        "full",
        "no_condition_norm",
        "no_missing_mask",
        "no_reliability_gate",
        "no_channel_gate",
        "no_local_trend",
        "no_temporal_attention",
        "no_degradation_aug",
        "weighted_huber_no_asymmetry",
        "no_smooth_late_risk",
        "no_reliability_supervision",
        "no_consistency_regularization",
        "no_quantile_calibration",
    }
    assert set(MAJOR_FD004_VARIANTS) == required


def test_manuscript_evidence_adds_canonical_display_names_to_external_results():
    from build_manuscript_evidence import ensure_model_display

    frame = pd.DataFrame({"model": ["rast_gru_v2", "dual_attention_tcn"], "unit_macro_rmse_mean": [8.0, 7.0]})
    labeled = ensure_model_display(frame)
    assert labeled["model_display"].tolist() == ["OCM-MST-GRU", "Dual-attention TCN"]
    assert "model_display" not in frame.columns


def test_generated_result_sections_route_language_specific_tables():
    from build_manuscript_evidence import write_results_sections

    with tempfile.TemporaryDirectory() as tmp:
        output = Path(tmp)
        write_results_sections(output)
        english = (output / "results_section_en.tex").read_text(encoding="utf-8")
        chinese = (output / "results_section_zh.tex").read_text(encoding="utf-8")
        assert "_zh.tex" not in english
        assert "table_protocol_track_sensitivity.tex" not in english
        assert "table_stress_auc_direct_predecessor.tex" not in english
        assert "fixed-task" in english
        assert "estimation" in english
        assert "diagnostic" in english
        assert "five levels support only a finite-design pattern" in english
        assert "Exact sign-flip values" in english
        assert "reported in the Supplementary Information" in english
        assert r"endpoint\_tolerance\_sensitivity.csv" in english
        assert "table_stress_family_uncertainty.tex" not in english
        assert "table_core_asym_cmapss.tex" in english
        assert r"normalized\_perturbation\_contrasts.csv" in english
        assert "fig_condition_normalization_controls" in english
        assert "table_core_asym_cmapss_zh.tex" in chinese
        assert "table_protocol_track_sensitivity_zh.tex" not in chinese
        assert "table_stress_auc_direct_predecessor_zh.tex" not in chinese
        assert r"normalized\_perturbation\_contrasts.csv" in chinese
        assert "fig_condition_normalization_controls" in chinese
        assert "numbers_zh.tex" not in chinese
        assert english.count(r"\FloatBarrier") >= 3
        assert chinese.count(r"\FloatBarrier") >= 3


def test_supplement_retains_full_major_revision_tables():
    project_root = Path(__file__).resolve().parents[1]
    release_supplement = project_root / "manuscript" / "supplement_en_RESS.tex"
    if release_supplement.exists():
        english = release_supplement.read_text(encoding="utf-8")
        if "table_full_5x10_finite_design" in english:
            assert "table_dynamic_policy_paired" in english
            assert "table_battery_primary" in english
            return
    supplementary = project_root / "supplementary"
    if not (supplementary / "supplement_en.tex").exists():
        supplementary = project_root.parent / "manuscript"
    if not (supplementary / "supplement_en.tex").exists():
        return
    english = (supplementary / "supplement_en.tex").read_text(encoding="utf-8")
    chinese_path = supplementary / "supplement_zh.tex"
    chinese = chinese_path.read_text(encoding="utf-8") if chinese_path.exists() else None
    assert "table_official_dual_mixer.tex" in english
    assert "table_core_asym_decision_cost.tex" not in english
    assert "table_endpoint_epoch_agreement.tex" in english
    assert "table_protocol_track_sensitivity.tex" in english
    assert "table_track_b_absolute.tex" in english
    assert "table_primary_fixed_task_estimates.tex" in english
    assert "table_small_cluster_calibration.tex" in english
    assert "table_multiplicity_registry.tex" in english
    assert "table_normalized_perturbation_summary.tex" in english
    assert "table_stress_family_uncertainty.tex" not in english
    assert "table_stress_auc_direct_predecessor.tex" not in english
    assert "table_core_asym_paired.tex" not in english
    assert "table_endpoint_selection_confirmation.tex" not in english
    assert "table_augmentation_distribution_summary.tex" in english
    assert "table_seed_factorial_decomposition.tex" in english
    assert "table_ncmapss_dev_rotation.tex" in english
    assert "table_ocm_attribution.tex" in english
    assert "table_ncmapss.tex" in english
    assert "table_reliability_observed_fraction.tex" in english
    if chinese is None:
        return
    assert "table_official_dual_mixer_zh.tex" in chinese
    assert "table_core_asym_decision_cost_zh.tex" not in chinese
    assert "table_endpoint_epoch_agreement_zh.tex" in chinese
    assert "table_protocol_track_sensitivity_zh.tex" in chinese
    assert "table_track_b_absolute_zh.tex" in chinese
    assert "table_primary_fixed_task_estimates.tex" in chinese
    assert "table_small_cluster_calibration.tex" in chinese
    assert "table_multiplicity_registry.tex" in chinese
    assert "table_normalized_perturbation_summary.tex" in chinese
    assert "table_stress_family_uncertainty_zh.tex" not in chinese
    assert "table_stress_auc_direct_predecessor_zh.tex" not in chinese
    assert "table_core_asym_paired_zh.tex" not in chinese
    assert "table_endpoint_selection_confirmation_zh.tex" not in chinese
    assert "table_augmentation_distribution_summary_zh.tex" in chinese
    assert "table_seed_factorial_decomposition_zh.tex" in chinese
    assert "table_ncmapss_dev_rotation_zh.tex" in chinese
    assert "table_ocm_attribution_zh.tex" in chinese
    assert "table_ncmapss_zh.tex" in chinese
    assert "table_reliability_observed_fraction_zh.tex" in chinese
    assert chinese.rfind("table_fast_cudnn_repeat_audit_zh.tex") < chinese.rfind(r"\end{document}")


def test_ocm_objective_buildup_changes_only_declared_auxiliary_terms():
    from rul.config import load_config
    from run_attribution_experiments import make_buildup_config

    project_root = Path(__file__).resolve().parents[1]
    base = load_config(project_root / "configs" / "paper_main.yaml")
    expected = {
        "base_only": (0.0, 0.0, 0.0, 0.0),
        "plus_risk": (0.10, 0.0, 0.0, 0.0),
        "plus_risk_cons": (0.10, 0.0, 0.05, 0.0),
    }
    keys = (
        "smooth_late_risk_weight",
        "reliability_supervision_weight",
        "consistency_weight",
        "quantile_calibration_weight",
    )
    for variant, values in expected.items():
        configured = make_buildup_config(base, variant, seed=123)
        assert tuple(configured["training"][key] for key in keys) == values
        assert configured["model"]["name"] == "rast_gru_v2"
        assert configured["model"]["use_uncertainty_head"] is False
        assert configured["project"]["seed"] == 123
        assert configured["training"]["epochs"] == base["training"]["epochs"]


def test_result_consistency_uses_ncmapss_specific_schema_without_val_last_artifact():
    from check_result_consistency import check_rows

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)
        for filename in ["metrics.json", "run_config.yaml", "best_model.pt", "test_predictions.csv"]:
            (run_dir / filename).write_bytes(b"test")
        row = {
            "dataset": "N-CMAPSS_DS02",
            "experiment_name": "ncmapss_ds02_seed42",
            "subset": "DS02",
            "model": "gru",
            "run_dir": str(run_dir),
            "seed": 42,
            "best_epoch": 3,
            "test_rmse": 8.0,
            "test_mae": 5.0,
            "test_r2": 0.8,
            "test_nasa_score": 100.0,
            "test_critical_30_rmse": 3.0,
            "test_critical_50_rmse": 4.0,
            "checkpoint_selection_metric": "val_risk_score",
            "best_val_rmse": 7.0,
            "best_val_risk_score": 0.2,
            "completed_epochs": 10,
            "planned_epochs": 80,
            "training_complete_reason": "early_stopped",
            "parameters": 1000,
            "single_sample_inference_ms": 1.0,
            "unit_macro_rmse": 8.1,
            "unit_macro_mae": 5.1,
            "unit_macro_nasa_per_window": 0.8,
            "unit_macro_lpr30": 0.2,
        }
        assert check_rows([row], expected_count=1) == []


def test_submission_zip_validation_requires_generated_latex_dependencies():
    import zipfile

    from create_submission_package import validate_zip

    with tempfile.TemporaryDirectory() as tmp:
        archive = Path(tmp) / "submission.zip"
        with zipfile.ZipFile(archive, "w") as zf:
            for name in [
                "manuscript/main_revised.tex",
                "manuscript/references.bib",
                "manuscript/figures/fig_fd004_pareto_tradeoff.png",
                "supplementary/replication_package/README_REPLICATION.md",
                "supplementary/replication_package/reproduce_minimal.sh",
            ]:
                zf.writestr(name, "test")
        try:
            validate_zip(archive)
        except RuntimeError as exc:
            assert "manuscript/generated/numbers.tex" in str(exc)
        else:
            raise AssertionError("ZIP validation accepted a manuscript missing generated LaTeX dependencies")


def test_submission_zip_round_filter_preserves_round20_and_rejects_round3():
    import zipfile

    from create_submission_package import validate_zip

    with tempfile.TemporaryDirectory() as tmp:
        round20_archive = Path(tmp) / "round20.zip"
        with zipfile.ZipFile(round20_archive, "w") as zf:
            zf.writestr("docs/ROUND20_VALIDATION_RECORD.md", "current")
        try:
            validate_zip(round20_archive)
        except RuntimeError as exc:
            assert "revision-round paths" not in str(exc)
            assert "missing required files" in str(exc)
        else:
            raise AssertionError("Incomplete ZIP should fail required-file validation")

        round3_archive = Path(tmp) / "round3.zip"
        with zipfile.ZipFile(round3_archive, "w") as zf:
            zf.writestr("templates/results_section_en_round3.tex", "legacy")
        try:
            validate_zip(round3_archive)
        except RuntimeError as exc:
            assert "revision-round paths" in str(exc)
        else:
            raise AssertionError("Legacy round3 path should be rejected")


def test_cross_backbone_protocol_buildup_adds_only_cumulative_switches():
    from rul.config import load_config
    from run_cross_backbone_protocol_buildup import STAGES, staged_config

    root = Path(__file__).resolve().parents[1]
    main = load_config(root / "configs" / "paper_main.yaml")
    conventional = load_config(root / "configs" / "standard_protocol_fd004.yaml")
    configs = [staged_config(main, conventional, "rast_gru", stage, 42) for stage in STAGES]

    assert configs[0]["data"]["scaler"] == "standard"
    assert configs[1]["data"]["scaler"] == "condition_standard"
    assert configs[1]["data"]["append_missing_mask"] is False
    assert configs[2]["data"]["append_missing_mask"] is True
    assert configs[2]["augmentation"]["profile"] == "none"
    assert configs[3]["augmentation"]["profile"] == "missing_noise_drift"
    assert configs[3]["training"]["loss"] == "huber"
    assert configs[4]["training"]["loss"] == "asymmetric_weighted_huber"
    assert configs[4]["training"]["selection_metric"] == "val_last_rmse"
    assert configs[5]["training"]["selection_metric"] == "val_last_risk_score"
    for cfg in configs:
        assert cfg["model"]["name"] == "rast_gru"
        assert cfg["training"]["epochs"] == 80
        assert cfg["data"]["window_size"] == 30


def test_seed_factorial_config_separates_split_from_training_randomness():
    from rul.config import load_config
    from run_split_training_seed_factorial import make_config

    root = Path(__file__).resolve().parents[1]
    base = load_config(root / "configs" / "paper_main.yaml")
    cfg, model = make_config(base, "Core", split_seed=123, training_seed=2024, epochs=None)

    assert model == "rast_gru_v2"
    assert cfg["project"]["split_seed"] == 123
    assert cfg["project"]["initialization_seed"] == 2024
    assert cfg["project"]["seed"] == 2024
    assert "shuffle_seed" not in cfg["project"]
    assert cfg["augmentation"]["seed"] == 2024
    assert cfg["training"]["late_over_weight"] == 1.0


def test_seed_factorial_decomposition_distinguishes_main_effect_sources():
    import pandas as pd

    from analyze_split_training_seed_factorial import decompose

    rows = []
    for point in ("RAST", "Core", "Asym"):
        for split_index, split_seed in enumerate((42, 123, 2024)):
            for training_index, training_seed in enumerate((42, 123, 2024)):
                rows.append(
                    {
                        "point": point,
                        "split_seed": split_seed,
                        "training_seed": training_seed,
                        "rmse": 10.0 + split_index,
                        "nasa_per_engine": 2.0 + training_index,
                        "lpr30": 0.1 + 0.01 * split_index,
                        "late_cvar95_30": 4.0 + training_index,
                    }
                )
    summary = decompose(pd.DataFrame(rows)).set_index(["point", "metric"])

    assert summary.loc[("RAST", "rmse"), "split_mean_sd"] > 0.0
    assert summary.loc[("RAST", "rmse"), "training_mean_sd"] == 0.0
    assert summary.loc[("RAST", "nasa_per_engine"), "split_mean_sd"] == 0.0
    assert summary.loc[("RAST", "nasa_per_engine"), "training_mean_sd"] > 0.0
    assert summary["nonadditive_residual_rms"].max() < 1e-12


def test_ncmapss_dev_rotation_is_disjoint_and_balanced_across_units():
    from prepare_ncmapss_dev_rotation import DEV_UNITS, rotation_split

    validation_units = []
    for test_unit in DEV_UNITS:
        train, validation, test = rotation_split(test_unit)
        assert len(train) == 4
        assert len(validation) == 1
        assert test == [test_unit]
        assert set(train).isdisjoint(validation)
        assert set(train).isdisjoint(test)
        assert set(validation).isdisjoint(test)
        assert sorted(train + validation + test) == sorted(DEV_UNITS)
        validation_units.extend(validation)
    assert sorted(validation_units) == sorted(DEV_UNITS)


def test_ncmapss_checkpoint_validation_accepts_exploratory_rotation_status():
    from run_ncmapss_benchmark import checkpoint_is_valid

    with tempfile.TemporaryDirectory() as raw:
        run_dir = Path(raw)
        torch.save({"model_state": {}}, run_dir / "best_model.pt")
        metrics = {
            "result_status": "exploratory_dev_unit_rotation",
            "dataset": "N-CMAPSS_DS02",
            "planned_epochs": 80,
        }
        (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        assert checkpoint_is_valid(run_dir)

        metrics["result_status"] = "incomplete"
        (run_dir / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
        assert not checkpoint_is_valid(run_dir)


def test_ncmapss_rejects_unknown_test_source_before_reading_data():
    from rul.ncmapss import prepare_ncmapss_ds02

    try:
        prepare_ncmapss_ds02("missing.h5", test_source="unknown")
    except ValueError as exc:
        assert "test_source" in str(exc)
    else:
        raise AssertionError("Unknown N-CMAPSS test source was accepted")


def test_fd002_protocol_transfer_changes_only_declared_point_switches():
    from rul.config import load_config
    from run_fd002_protocol_transfer import POINTS, transfer_config

    root = Path(__file__).resolve().parents[1]
    base = load_config(root / "configs" / "paper_main.yaml")
    for point, (model, late_weight) in POINTS.items():
        cfg = transfer_config(base, point, seed=123)
        assert cfg["data"]["subset"] == "FD002"
        assert cfg["model"]["name"] == model
        assert cfg["training"]["late_over_weight"] == late_weight
        assert cfg["training"]["selection_metric"] == base["training"]["selection_metric"]
        assert cfg["augmentation"]["profile"] == base["augmentation"]["profile"]
        assert cfg["augmentation"]["seed"] == 123
        assert cfg["progress"]["batch_bar"] is False


def test_fd002_protocol_transfer_holm_adjustment_preserves_order_and_monotonicity():
    import numpy as np

    from analyze_fd002_protocol_transfer import holm_adjust

    raw = np.array([0.04, 0.001, 0.20, 0.02])
    adjusted = holm_adjust(raw)

    assert np.allclose(adjusted, np.array([0.08, 0.004, 0.20, 0.06]))
    ordered = adjusted[np.argsort(raw)]
    assert np.all(np.diff(ordered) >= 0)


def test_round2_review_holm_adjustment_preserves_family_order():
    from analyze_normalized_perturbation_evidence import holm_adjust

    raw = np.array([0.04, 0.001, 0.20, 0.02])
    adjusted = holm_adjust(raw)

    assert np.allclose(adjusted, np.array([0.08, 0.004, 0.20, 0.06]))
    assert np.all(np.diff(adjusted[np.argsort(raw)]) >= 0)


def test_round2_engine_count_matrix_preserves_each_matched_draw():
    from analyze_normalized_perturbation_evidence import engine_count_matrix

    draws = np.array([[0, 1, 1, 3], [2, 2, 2, 0]])
    counts = engine_count_matrix(draws, engine_count=4)

    assert counts.shape == (4, 2)
    assert np.array_equal(counts[:, 0], np.array([1, 2, 0, 1]))
    assert np.array_equal(counts[:, 1], np.array([1, 0, 3, 0]))
    assert np.array_equal(counts.sum(axis=0), np.array([4, 4]))


def test_official_state_scaler_uses_training_states_and_global_unseen_fallback():
    import pandas as pd

    from rul.condition_norm import OfficialStateStandardScaler

    train = pd.DataFrame(
        {
            "setting_1": [0.0, 0.1, 1.0, 1.1],
            "setting_2": [0.0, 0.0, 1.0, 1.0],
            "setting_3": [0.0, 0.0, 1.0, 1.0],
            "s1": [1.0, 3.0, 11.0, 13.0],
        }
    )
    unseen = pd.DataFrame(
        {
            "setting_1": [9.0],
            "setting_2": [9.0],
            "setting_3": [9.0],
            "s1": [7.0],
        }
    )
    features = ["setting_1", "setting_2", "setting_3", "s1"]
    scaler = OfficialStateStandardScaler(feature_names=features)
    transformed = scaler.fit_transform(train)
    unseen_transformed = scaler.transform(unseen)

    assert set(transformed["condition_id"]) == {0, 1}
    assert np.allclose(transformed.groupby("condition_id")["s1"].mean().to_numpy(), 0.0)
    assert int(unseen_transformed["condition_id"].iloc[0]) == -1
    assert np.isfinite(unseen_transformed[features].to_numpy()).all()


def test_continuous_condition_corrector_fits_training_residuals_only():
    import pandas as pd

    from rul.condition_norm import ContinuousConditionCorrector

    setting = np.linspace(-2.0, 2.0, 20)
    train = pd.DataFrame(
        {
            "setting_1": setting,
            "setting_2": setting**2,
            "setting_3": np.sin(setting),
            "s1": 2.0 + 3.0 * setting + 0.5 * setting**2,
        }
    )
    validation = train.iloc[:4].copy()
    validation["s1"] += 1.0
    features = ["setting_1", "setting_2", "setting_3", "s1"]
    scaler = ContinuousConditionCorrector(feature_names=features)
    transformed = scaler.fit_transform(train)
    validation_transformed = scaler.transform(validation)

    assert abs(float(transformed["s1"].mean())) < 1e-10
    assert np.isfinite(validation_transformed[features].to_numpy()).all()


def test_feature_selection_breaks_equal_scores_by_sensor_number():
    import pandas as pd

    from rul.cmapss import SENSORS
    from rul.features import score_sensor_features

    rows = 8
    frame = pd.DataFrame(
        {
            "cycle": np.arange(rows, dtype=float),
            "rul": np.arange(rows, 0, -1, dtype=float),
            **{sensor: np.ones(rows, dtype=float) for sensor in SENSORS},
        }
    )
    scores = score_sensor_features(frame)
    assert scores["feature"].tolist() == SENSORS


def test_critical_tail_marks_conditional_severity_undefined_without_late_events():
    from rul.metrics import critical_zone_metrics

    metrics = critical_zone_metrics([5.0, 10.0, 20.0], [4.0, 9.0, 19.0], thresholds=(30.0,))
    assert metrics["critical_30_late_event_count"] == 0
    assert metrics["critical_30_conditional_tail_defined"] is False
    assert metrics["critical_30_conditional_over_error_mean"] is None
    assert metrics["critical_30_conditional_late_q95"] is None
    assert metrics["critical_30_mean_late_excess"] == 0.0


def test_diagnostic_max_t_envelope_handles_inactive_zero_variance_contrast():
    from analyze_fixed_benchmark_evidence import diagnostic_max_t_envelope

    draws = np.array(
        [
            [0.8, 0.0, -1.2],
            [1.1, 0.0, -0.8],
            [0.9, 0.0, -1.0],
            [1.2, 0.0, -1.3],
            [1.0, 0.0, -0.9],
        ],
        dtype=float,
    )
    observed = np.array([1.0, 0.0, -1.0], dtype=float)
    result = diagnostic_max_t_envelope(draws, observed)

    assert result["interval_low"].shape == observed.shape
    assert result["interval_high"].shape == observed.shape
    assert result["adjusted_p"].shape == observed.shape
    assert np.array_equal(result["active_contrast"], np.array([True, False, True]))
    assert result["interval_low"][1] == 0.0
    assert result["interval_high"][1] == 0.0
    assert result["adjusted_p"][1] == 1.0
    assert np.all(result["adjusted_p"] >= 1.0 / (len(draws) + 1.0))


def test_release_multiplicity_registry_has_no_confirmatory_family():
    from analyze_fixed_benchmark_evidence import multiplicity_registry

    registry = multiplicity_registry()
    assert len(registry) == 22
    assert registry["family_id"].is_unique
    assert not registry["decision_role"].str.lower().str.contains(
        r"(?<!cannot support )confirmatory claim$",
        regex=True,
    ).any()
    assert set(registry["family_id"]) >= {
        "E-FIXED-12",
        "S-ENDPOINT-80",
        "S-PERTURB-54",
        "S-NORM-9",
        "D-STREAM10-FIXED",
        "D-PREF-REFIT-3X3",
        "D-SHARED-FACT-8",
        "D-LOFO-4",
        "D-NCMAPSS-SW9",
        "D-NCMAPSS-H3",
        "D-CROSSED-3X3",
        "D-CMAPSS-PREP5",
    }


def test_round8_no_overlap_audit_keeps_only_zero_augmentation_rows():
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "analyze_augmentation_distribution_sensitivity.py"
    spec = importlib.util.spec_from_file_location("round8_augmentation_audit_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    frame = pd.DataFrame(
        {
            "point": ["RAST", "RAST", "Core"],
            "condition": ["p0", "p1", "p0"],
            "metric": ["clean_rmse"] * 3,
        }
    )
    audited = module.no_overlap_audit(frame)
    assert audited["condition"].tolist() == ["p0", "p0"]
    assert set(audited["design_role"]) == {
        "no-overlap augmentation diagnostic; all perturbation families are held out jointly during training"
    }


def test_round8_stored_results_pipeline_declares_resumeable_evidence_stages():
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_stored_results_pipeline.py"
    spec = importlib.util.spec_from_file_location("round8_staged_pipeline_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    names = [stage.name for stage in module.stages()]
    assert "fixed_benchmark_statistics" in names
    assert "preference_selection_stability" in names
    assert "no_overlap_augmentation" in names
    assert names.index("manuscript_evidence") < names.index("manuscript_evidence_check")
    assert names.index("release_evidence") < names.index("release_evidence_gate")


def test_round12_stored_results_pipeline_normalizes_relative_log_dir():
    import importlib.util
    import pytest

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_stored_results_pipeline.py"
    spec = importlib.util.spec_from_file_location("round12_log_dir_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    expected = (module.PROJECT_ROOT / "reports" / "round12").resolve()
    assert module.normalize_log_dir("reports/round12") == expected
    with pytest.raises(ValueError):
        module.normalize_log_dir(module.PROJECT_ROOT.parent / "outside")


def test_round12_predicate_skip_satisfies_from_stage_prerequisite():
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_stored_results_pipeline.py"
    spec = importlib.util.spec_from_file_location("round12_skip_state_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    stage = module.Stage(
        "optional_cache",
        ["python", "-V"],
        predicate=lambda: False,
        skip_reason="not distributed",
    )
    assert module.prerequisite_satisfied(stage, {"status": "SKIP"})
    assert not module.prerequisite_satisfied(stage, {"status": "FAIL"})


def test_deposit_checksum_precedes_runtime_authority_archival():
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_stored_results_pipeline.py"
    spec = importlib.util.spec_from_file_location("runtime_archive_order_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    with tempfile.TemporaryDirectory() as temporary:
        log_dir = Path(temporary)
        authority = log_dir / "runtime_contract_full.json"
        authority.write_text(
            json.dumps({"pipeline_version": "release-4.9"}),
            encoding="utf-8",
        )
        module.archive_runtime_authority_after_checksum("other_stage", "PASS", log_dir)
        assert authority.exists()
        module.archive_runtime_authority_after_checksum(
            "verify_deposit_checksum",
            "PASS",
            log_dir,
        )
        assert not authority.exists()
        assert (log_dir / "historical" / "release-4.9" / authority.name).exists()


def test_finalize_revision_builds_figures_before_figure_manifest():
    script_path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "finalize_review_revision.ps1"
    )
    if not script_path.exists():
        # Authoring-only Windows helper; it is intentionally absent from the
        # portable replication package.
        return
    script = script_path.read_text(encoding="utf-8")
    revised = script.index('Invoke-CheckedPython @("scripts\\build_revised_assets.py")')
    advanced = script.index('Invoke-CheckedPython @("scripts\\build_advanced_evidence.py")')
    consistency = script.index(
        'Invoke-CheckedPython @("scripts\\check_figure_data_consistency.py")'
    )
    assert revised < advanced < consistency


def test_package_closure_gate_detects_a_missing_pipeline_script():
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "check_package_closure.py"
    spec = importlib.util.spec_from_file_location("package_closure_for_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        (root / "scripts").mkdir()
        (root / "tests").mkdir()
        (root / "scripts" / "run_stored_results_pipeline.py").write_text(
            "from dataclasses import dataclass\n"
            "@dataclass\n"
            "class Stage:\n"
            "    name: str\n"
            "    command: tuple[str, ...]\n"
            "def stages():\n"
            "    return [Stage('missing', ('python', 'scripts/missing.py'))]\n",
            encoding="utf-8",
        )
        report = module.audit(root)

    assert report["status"] == "FAIL"
    assert report["missing_required_paths"] == ["scripts/missing.py"]


def test_lightweight_packager_excludes_python_caches():
    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "create_submission_package.py"
    ).read_text(encoding="utf-8")
    assert '"__pycache__" in relative.parts' in script
    assert 'source.suffix.lower() == ".pyc"' in script
    assert 'env["PYTHONDONTWRITEBYTECODE"] = "1"' in script


def test_round12_new_experiment_grid_is_factorized_and_counted():
    import importlib.util
    import shutil
    import yaml

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_round12_new_experiments.py"
    spec = importlib.util.spec_from_file_location("round12_experiment_grid_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    original_root = module.PROJECT_ROOT
    temp_dir = tempfile.TemporaryDirectory()
    module.PROJECT_ROOT = Path(temp_dir.name)
    config_path = module.PROJECT_ROOT / "configs" / "paper_main.yaml"
    config_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(original_root / "configs" / "paper_main.yaml", config_path)
    base = module.load_config(config_path)
    config_text = yaml.safe_dump(base, sort_keys=False)
    for stream_seed in module.PREPROCESS_SEEDS:
        for subset in module.PREPROCESS_SUBSETS:
            for model_name in ("rast_gru", "rast_gru_v2"):
                path = (
                    module.PROJECT_ROOT
                    / "results"
                    / f"paper_main_v3_seed{stream_seed}"
                    / subset
                    / model_name
                    / "run_config.yaml"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(config_text, encoding="utf-8")
        for window in (20, 50):
            path = (
                module.PROJECT_ROOT
                / "results"
                / f"design_sensitivity_v3_seed{stream_seed}_window_w{window}_cap125"
                / module.PREPROCESS_SUBSETS[0]
                / "rast_gru_v2"
                / "run_config.yaml"
            )
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(config_text, encoding="utf-8")
    for split_seed in module.CROSSED_SPLITS:
        if split_seed == 42:
            continue
        for stream_seed in module.CROSSED_STREAMS:
            for point, model_name in (("rast", "rast_gru"), ("asym", "rast_gru_v2")):
                experiment = (
                    f"preference_rast_reference_split{split_seed}_stream{stream_seed}"
                    if point == "rast"
                    else f"preference_ll1p25_lo1p50_split{split_seed}_stream{stream_seed}"
                )
                path = (
                    module.PROJECT_ROOT
                    / "results"
                    / "round12_new_evidence"
                    / experiment
                    / "FD004"
                    / model_name
                    / "run_config.yaml"
                )
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(config_text, encoding="utf-8")
    independent = module.independent_jobs(base, (31415,), 2)
    preference = module.preference_jobs(base, (42,), (31415,), 2)
    factorial = module.factorial_jobs(base, (42,), 2)
    lofo = module.lofo_jobs(base, (42,), 2)
    crossed = module.crossed_split_stream_jobs(
        base,
        module.CROSSED_SPLITS,
        module.CROSSED_STREAMS,
        2,
    )
    preprocessing = module.cmapss_preprocessing_jobs(
        base,
        module.PREPROCESS_SEEDS,
        2,
    )
    assert len(independent) == 8
    assert len(preference) == 10
    assert len(factorial) == 8
    assert len(lofo) == 8
    assert len(crossed) == 72
    assert len(preprocessing) == 30
    assert {
        (job.metadata["split_seed"], job.metadata["training_stream_seed"])
        for job in crossed
    } == {
        (split_seed, stream_seed)
        for split_seed in module.CROSSED_SPLITS
        for stream_seed in module.CROSSED_STREAMS
    }
    assert sum(
        job.metadata["artifact_reuse_source"] == "new_crossed_training"
        for job in crossed
    ) == 36
    assert sum(
        job.metadata["artifact_reuse_source"] == "new_preprocessing_training"
        for job in preprocessing
    ) == 18
    assert {
        (job.metadata["audit_axis"], job.metadata["audit_level"])
        for job in preprocessing
    } == {
        ("window", 20),
        ("reference", 30),
        ("window", 50),
        ("sensor_budget", 8),
        ("sensor_budget", 21),
    }
    module.validate_lofo_exposure_matrix(lofo)
    noise_jobs = [
        job for job in lofo if job.metadata["held_out_family"] == "noise"
    ]
    assert len(noise_jobs) == 2
    for job in noise_jobs:
        exposure = module.lofo_exposure_metadata(job.config)
        assert exposure["primary_gaussian_noise_std"] == 0.0
        assert exposure["consistency_weight"] == 0.0
        assert exposure["consistency_gaussian_noise_std"] == 0.0
    assert {
        1 + (window - 1) * sampling
        for sampling, window in module.NCMAPSS_HORIZON_MATCHED
    } == {6001}
    assert {
        (
            job.metadata["backbone"],
            job.metadata["risk_factor"],
            job.metadata["consistency_factor"],
        )
        for job in factorial
    } == {
        (backbone, risk, consistency)
        for backbone in ("rast", "ocm")
        for risk in (0, 1)
        for consistency in (0, 1)
    }
    for job in preference:
        assert job.config["project"]["split_seed"] == 42
        assert job.config["project"]["initialization_seed"] == 31415
        assert job.config["project"]["shuffle_seed"] == 31415
        assert job.config["augmentation"]["seed"] == 31415
    assert sum(job.metadata["point"] == "rast_reference" for job in preference) == 1
    assert sum(job.metadata["point"] == "asym" for job in preference) == 9
    temp_dir.cleanup()
    module.PROJECT_ROOT = original_root


def test_round16_lofo_curve_summaries_distinguish_nonzero_and_zero_anchor():
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "analyze_round12_new_experiments.py"
    spec = importlib.util.spec_from_file_location("round16_lofo_curve_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    summary = module.response_curve_summaries(
        np.asarray([0.2, 0.4]),
        np.asarray([12.0, 14.0]),
        clean_value=10.0,
    )

    assert summary["nonzero_range_absolute_mean_response"] == 13.0
    assert np.isclose(summary["nonzero_range_clean_corrected_mean_response"], 3.0)
    assert np.isclose(summary["zero_anchored_absolute_auc"], 12.0)
    assert np.isclose(summary["zero_anchored_clean_corrected_degradation_auc"], 2.0)


def test_round16_level_subset_sensitivity_enumerates_all_three_of_five_choices():
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "analyze_round12_new_experiments.py"
    spec = importlib.util.spec_from_file_location("round16_level_subset_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    frame = pd.DataFrame(
        {
            "subset": ["FD001"] * 5,
            "metric": ["test_rmse"] * 5,
            "composite_seed": [1, 2, 3, 4, 5],
            "asym_minus_rast": [-2.0, -1.0, 0.0, 1.0, 2.0],
        }
    )

    detail, summary = module._enumerate_level_subsets(
        frame,
        level_column="composite_seed",
        selected_levels=(1, 2, 3),
        source_design="test",
    )

    assert len(detail) == 10
    assert int(summary.iloc[0]["combination_count"]) == 10
    assert float(summary.iloc[0]["reported_subset_mean"]) == -1.0


def test_round16_ncmapss_summary_emits_overlap_and_exposure_diagnostics():
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "analyze_round12_new_experiments.py"
    spec = importlib.util.spec_from_file_location("round16_ncmapss_summary_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        module.INPUT = root / "input"
        module.OUTPUT = root / "output"
        module.INPUT.mkdir()
        module.OUTPUT.mkdir()
        pd.DataFrame(
            [
                {
                    "sampling_interval": 100,
                    "sensitivity_window_size": 30,
                    "seed": seed,
                    "n_train_windows": 300,
                    "n_test_windows": 30,
                    "unit_macro_rmse": 10.0 + seed,
                    "unit_macro_mae": 8.0 + seed,
                    "unit_macro_nasa_per_window": 1.0 + seed,
                    "unit_macro_lpr30": 0.1 * seed,
                }
                for seed in (1, 2)
            ]
        ).to_csv(module.INPUT / "ncmapss_sampling_window_seed_metrics.csv", index=False)

        result = module.ncmapss_summary_for("ncmapss_sampling_window")
        summary = pd.read_csv(module.OUTPUT / "ncmapss_sampling_window_summary.csv")

    assert result == {"runs": 2, "cells": 1}
    assert np.isclose(summary.loc[0, "sampled_window_overlap_fraction"], 29.0 / 30.0)
    assert np.isclose(summary.loc[0, "train_nonoverlap_equivalent"], 10.0)
    assert "optimizer updates" in summary.loc[0, "optimization_exposure_note"]


def test_round8_replication_package_preserves_stage_runtime_logs():
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "create_replication_package.py"
    spec = importlib.util.spec_from_file_location("round8_runtime_log_package_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory() as temporary:
        reports = Path(temporary) / "reports"
        runtime_log = reports / "stored_results_pipeline" / "logs" / "stage.log"
        unrelated_log = reports / "other" / "logs" / "debug.log"
        runtime_log.parent.mkdir(parents=True)
        unrelated_log.parent.mkdir(parents=True)
        runtime_log.write_text("ok", encoding="utf-8")
        stale_runtime_log = runtime_log.parent / "old_stage.log"
        stale_runtime_log.write_text("old", encoding="utf-8")
        contract = reports / "stored_results_pipeline" / "runtime_contract_full.json"
        contract.write_text(
            json.dumps(
                {
                    "pipeline_version": "release-5.0",
                    "stages": [
                        {
                            "name": "stage",
                            "log": "reports/stored_results_pipeline/logs/stage.log",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        unrelated_log.write_text("debug", encoding="utf-8")
        sufficiency = reports / "dataset_sufficiency_report.md"
        sufficiency.write_text("report", encoding="utf-8")
        transient_round_report = reports / "round15_release_authority_pending.json"
        transient_round_report.write_text("{}", encoding="utf-8")
        assert not module.should_skip(runtime_log, reports)
        assert module.should_skip(stale_runtime_log, reports)
        assert not module.should_skip(sufficiency, reports)
        assert module.should_skip(unrelated_log, reports)
        assert module.should_skip(transient_round_report, reports)
        assert "research_protocol_v3.md" in module.RELEASE_DOC_ALLOWLIST
        assert "ROUND19_REVISION_RESPONSE_20260809.md" in module.RELEASE_DOC_ALLOWLIST
        assert "ROUND16_REVISION_RESPONSE_20260802.md" not in module.RELEASE_DOC_ALLOWLIST
        assert "ROUND13_REVISION_RESPONSE_20260801.md" not in module.RELEASE_DOC_ALLOWLIST
        docs = Path(temporary) / "docs"
        docs.mkdir()
        protocol = docs / "RETROSPECTIVE_GOVERNANCE_EXTENSION_PROTOCOL.md"
        protocol.write_text("protocol", encoding="utf-8")
        assert not module.should_skip(protocol, docs)


def test_replication_round_filter_does_not_drop_round20_release_docs():
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "create_replication_package.py"
    spec = importlib.util.spec_from_file_location("replication_round_filter_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory() as temporary:
        docs = Path(temporary) / "docs"
        docs.mkdir()
        round2 = docs / "ROUND2_LEGACY.md"
        embedded_round3 = docs / "results_section_en_round3.tex"
        round20 = docs / "ROUND20_VALIDATION_RECORD.md"
        round2.write_text("legacy", encoding="utf-8")
        embedded_round3.write_text("legacy", encoding="utf-8")
        round20.write_text("current", encoding="utf-8")
        assert module.should_skip(round2, docs)
        assert module.should_skip(embedded_round3, docs)
        assert not module.should_skip(round20, docs)
        assert "ROUND20_VALIDATION_RECORD.md" in module.RELEASE_DOC_ALLOWLIST


def test_round15_preprocessing_assets_do_not_overwrite_independent_zh_table():
    import importlib.util

    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "build_round12_new_evidence_assets.py"
    )
    spec = importlib.util.spec_from_file_location("round15_asset_isolation_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        module.SOURCE = root / "analysis"
        module.TABLE_OUTPUT = root / "tables"
        module.SOURCE.mkdir(parents=True)
        pd.DataFrame(
            [
                {
                    "subset": "FD004",
                    "metric": "rmse",
                    "stream_count": 10,
                    "asym_minus_rast_mean": 0.25,
                    "asym_minus_rast_sd": 0.1,
                    "negative_count": 4,
                    "positive_count": 6,
                    "zero_count": 0,
                }
            ]
        ).to_csv(module.SOURCE / "independent_stream_paired_summary.csv", index=False)
        pd.DataFrame(
            [
                {
                    "subset": "FD004",
                    "audit_axis": "reference",
                    "audit_level": 30,
                    "window_size": 30,
                    "sensor_budget": 14,
                    "metric": "rmse",
                    "asym_minus_rast_mean": 0.5,
                    "asym_minus_rast_sd": 0.2,
                    "negative_count": 1,
                    "positive_count": 2,
                    "zero_count": 0,
                }
            ]
        ).to_csv(module.SOURCE / "cmapss_preprocessing_paired_summary.csv", index=False)

        module.build_independent_table()
        independent_zh = module.TABLE_OUTPUT / "table_round12_independent_streams_zh.tex"
        before = independent_zh.read_bytes()
        assert "训练流" in before.decode("utf-8")
        module.build_preprocessing_table()
        assert independent_zh.read_bytes() == before
        assert (module.TABLE_OUTPUT / "table_round12_cmapss_preprocessing_zh.tex").exists()


def test_round9_submission_verifier_rejects_mismatched_release_identity():
    import importlib.util

    script = Path(__file__).resolve().parents[1] / "scripts" / "verify_submission_package.py"
    spec = importlib.util.spec_from_file_location("round8_release_sync_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        main = root / "manuscript" / "generated" / "release_sync.tex"
        supplement = root / "supplementary" / "manuscript" / "generated" / "release_sync.tex"
        main_source = root / "manuscript" / "main_revised.tex"
        supplement_source = (
            root / "supplementary" / "manuscript" / "supplement_en.tex"
        )
        main.parent.mkdir(parents=True)
        supplement.parent.mkdir(parents=True)
        main_source.write_text("main", encoding="utf-8")
        supplement_source.write_text("supplement", encoding="utf-8")
        payload = (
            "\\newcommand{\\EvidenceReleaseID}{r9}\n"
            "\\newcommand{\\EvidenceProtocolVersion}{4.2}\n"
        )
        main.write_text(payload, encoding="utf-8")
        supplement.write_text(payload, encoding="utf-8")
        (root / "cross_document_release.json").write_text(
            json.dumps(
                {
                    "release_id": "r9",
                    "main_source_sha256": module.sha256_file(main_source),
                    "supplement_source_sha256": module.sha256_file(
                        supplement_source
                    ),
                    "sync_file_sha256": module.sha256_file(main),
                }
            ),
            encoding="utf-8",
        )
        assert module.verify_cross_document_release(root) == []
        supplement.write_text(payload.replace("r9", "r8"), encoding="utf-8")
        assert module.verify_cross_document_release(root)


def test_lofo_tolerance_severity_is_recomputed_from_engine_errors():
    import importlib.util

    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "analyze_round12_new_experiments.py"
    )
    spec = importlib.util.spec_from_file_location("lofo_tolerance_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    rows = []
    for perturbation_seed, errors in ((1, (12.0, 7.0, -1.0)), (2, (6.0, 4.0, -1.0))):
        for unit_id, (true_rul, error) in enumerate(
            zip((10.0, 20.0, 30.0), errors, strict=True),
            start=1,
        ):
            rows.append(
                {
                    "held_out_family": "noise",
                    "point": "asym",
                    "scenario": "gaussian_noise",
                    "level": 0.1,
                    "training_stream_seed": 42,
                    "perturbation_seed": perturbation_seed,
                    "unit_id": unit_id,
                    "true_rul": true_rul,
                    "error": error,
                }
            )
    tolerance = module.build_lofo_tolerance_rows(pd.DataFrame(rows))
    first = tolerance[
        (tolerance["perturbation_seed"] == 1)
        & np.isclose(tolerance["epsilon"], 5.0)
    ].iloc[0]
    assert first["late_event_count"] == 2
    assert np.isclose(first["late_ratio"], 2.0 / 3.0)
    assert np.isclose(first["zimle"], 3.0)
    assert np.isclose(first["cmle"], 4.5)
    epsilon10 = tolerance[np.isclose(tolerance["epsilon"], 10.0)]
    assert epsilon10["late_event_count"].tolist() == [1, 0]
    assert epsilon10["cmle"].isna().sum() == 1
    summary = module.summarize_lofo_tolerance_rows(tolerance)
    summary10 = summary[np.isclose(summary["epsilon"], 10.0)].iloc[0]
    assert summary10["zero_event_evaluation_count"] == 1
    assert np.isclose(summary10["zero_event_evaluation_rate"], 0.5)
    assert summary10["undefined_cmle_evaluation_count"] == 1
    assert np.isclose(summary10["undefined_cmle_evaluation_rate"], 0.5)


def test_release_50_runtime_contract_declares_context_and_26_stages():
    import importlib.util

    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_stored_results_pipeline.py"
    )
    spec = importlib.util.spec_from_file_location("runtime_contract_50_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    assert module.PIPELINE_VERSION == "release-5.0"
    declared = module.stages()
    assert len(declared) == 26
    assert declared[-2].name == "package_closure"
    context = module.execution_context()
    assert context["kind"] in {"author-workspace", "distributed-package-workspace"}
    assert "checksum_authority_sha256" in context
    assert "ncmapss_cache_distribution" in context


def test_packaged_release_checksum_audit_detects_mutation():
    import importlib.util

    script = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "run_package_release_authority.py"
    )
    spec = importlib.util.spec_from_file_location("immutable_release_test", script)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    with tempfile.TemporaryDirectory() as temporary:
        root = Path(temporary)
        payload = root / "payload.txt"
        payload.write_text("fixed\n", encoding="utf-8")
        import hashlib

        digest = hashlib.sha256(payload.read_bytes()).hexdigest()
        (root / "checksums.sha256").write_text(
            f"{digest}  payload.txt\n", encoding="utf-8"
        )
        assert module.checksum_issues(root) == []
        payload.write_text("changed\n", encoding="utf-8")
        assert module.checksum_issues(root) == ["mismatch: payload.txt"]
