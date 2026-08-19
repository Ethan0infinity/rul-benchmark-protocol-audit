from __future__ import annotations

import numpy as np


def _next_seed(rng: np.random.Generator | None) -> int | None:
    if rng is None:
        return None
    return int(rng.integers(0, np.iinfo(np.int32).max))


def apply_gaussian_noise(
    x: np.ndarray,
    noise_std: float,
    sensor_indices: list[int] | None = None,
    seed: int | None = None,
) -> np.ndarray:
    if noise_std <= 0:
        return x.copy()
    rng = np.random.default_rng(seed)
    out = x.copy()
    indices = sensor_indices if sensor_indices else list(range(out.shape[-1]))
    noise = rng.normal(loc=0.0, scale=noise_std, size=out[..., indices].shape)
    out[..., indices] = out[..., indices] + noise
    return out


def apply_sensor_missing(
    x: np.ndarray,
    missing_rate: float,
    sensor_indices: list[int],
    seed: int | None = None,
) -> np.ndarray:
    if missing_rate <= 0 or not sensor_indices:
        return x.copy()
    rng = np.random.default_rng(seed)
    out = x.copy()
    k = max(1, int(round(len(sensor_indices) * missing_rate)))
    chosen = rng.choice(sensor_indices, size=min(k, len(sensor_indices)), replace=False)
    out[..., chosen] = 0.0
    return out


def apply_sensor_missing_with_mask(
    x: np.ndarray,
    missing_rate: float,
    sensor_indices: list[int],
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if missing_rate <= 0 or not sensor_indices:
        return x.copy(), np.ones((*x.shape[:-1], len(sensor_indices)), dtype=np.float32)
    rng = np.random.default_rng(seed)
    out = x.copy()
    mask = np.ones((*out.shape[:-1], len(sensor_indices)), dtype=np.float32)
    k = max(1, int(round(len(sensor_indices) * missing_rate)))
    chosen_positions = rng.choice(len(sensor_indices), size=min(k, len(sensor_indices)), replace=False)
    chosen_features = [sensor_indices[int(pos)] for pos in chosen_positions]
    out[..., chosen_features] = 0.0
    mask[..., chosen_positions] = 0.0
    return out, mask


def apply_window_random_missing_with_mask(
    x: np.ndarray,
    missing_rate: float,
    sensor_indices: list[int],
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply independent full-window sensor missingness per sample window."""
    if missing_rate <= 0 or not sensor_indices:
        return x.copy(), np.ones((*x.shape[:-1], len(sensor_indices)), dtype=np.float32)
    rng = np.random.default_rng(seed)
    out = x.copy()
    mask = np.ones((*out.shape[:-1], len(sensor_indices)), dtype=np.float32)
    k = max(1, int(round(len(sensor_indices) * missing_rate)))
    for window_idx in range(out.shape[0]):
        chosen_positions = rng.choice(len(sensor_indices), size=min(k, len(sensor_indices)), replace=False)
        chosen_features = [sensor_indices[int(pos)] for pos in chosen_positions]
        out[window_idx, :, chosen_features] = 0.0
        mask[window_idx, :, chosen_positions] = 0.0
    return out, mask


def apply_block_missing(
    x: np.ndarray,
    missing_rate: float,
    sensor_indices: list[int],
    seed: int | None = None,
) -> np.ndarray:
    if missing_rate <= 0 or not sensor_indices:
        return x.copy()
    rng = np.random.default_rng(seed)
    out = x.copy()
    time_len = out.shape[1]
    block_len = max(1, int(round(time_len * missing_rate)))
    start = int(rng.integers(0, max(1, time_len - block_len + 1)))
    k = max(1, int(round(len(sensor_indices) * missing_rate)))
    chosen = rng.choice(sensor_indices, size=min(k, len(sensor_indices)), replace=False)
    out[:, start : start + block_len, chosen] = 0.0
    return out


def apply_block_missing_with_mask(
    x: np.ndarray,
    missing_rate: float,
    sensor_indices: list[int],
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if missing_rate <= 0 or not sensor_indices:
        return x.copy(), np.ones((*x.shape[:-1], len(sensor_indices)), dtype=np.float32)
    rng = np.random.default_rng(seed)
    out = x.copy()
    mask = np.ones((*out.shape[:-1], len(sensor_indices)), dtype=np.float32)
    time_len = out.shape[1]
    block_len = max(1, int(round(time_len * missing_rate)))
    start = int(rng.integers(0, max(1, time_len - block_len + 1)))
    k = max(1, int(round(len(sensor_indices) * missing_rate)))
    chosen_positions = rng.choice(len(sensor_indices), size=min(k, len(sensor_indices)), replace=False)
    chosen_features = [sensor_indices[int(pos)] for pos in chosen_positions]
    out[:, start : start + block_len, chosen_features] = 0.0
    mask[:, start : start + block_len, chosen_positions] = 0.0
    return out, mask


def apply_sensor_drift(
    x: np.ndarray,
    drift_rate: float,
    sensor_indices: list[int],
    seed: int | None = None,
) -> np.ndarray:
    """Apply a smooth additive drift to selected sensor channels."""
    if drift_rate <= 0 or not sensor_indices:
        return x.copy()
    rng = np.random.default_rng(seed)
    out = x.copy()
    time_len = out.shape[1]
    ramp = np.linspace(0.0, drift_rate, time_len, dtype=np.float32).reshape(1, time_len, 1)
    signs = rng.choice([-1.0, 1.0], size=(1, 1, len(sensor_indices))).astype(np.float32)
    out[..., sensor_indices] = out[..., sensor_indices] + ramp * signs
    return out


def apply_sensor_bias_shift(
    x: np.ndarray,
    bias_amplitude: float,
    sensor_indices: list[int],
    seed: int | None = None,
) -> np.ndarray:
    """Add a signed constant bias in normalized sensor-standard-deviation units."""
    if bias_amplitude <= 0 or not sensor_indices:
        return x.copy()
    rng = np.random.default_rng(seed)
    out = x.copy()
    signs = rng.choice([-1.0, 1.0], size=(1, 1, len(sensor_indices))).astype(np.float32)
    out[..., sensor_indices] = out[..., sensor_indices] + float(bias_amplitude) * signs
    return out


def apply_stuck_at_fault(
    x: np.ndarray,
    sensor_fraction: float,
    sensor_indices: list[int],
    seed: int | None = None,
) -> np.ndarray:
    """Freeze selected channels at a random onset value within each window."""
    if sensor_fraction <= 0 or not sensor_indices:
        return x.copy()
    rng = np.random.default_rng(seed)
    out = x.copy()
    k = max(1, int(round(len(sensor_indices) * sensor_fraction)))
    chosen = rng.choice(sensor_indices, size=min(k, len(sensor_indices)), replace=False)
    low = max(1, out.shape[1] // 3)
    high = max(low + 1, (2 * out.shape[1]) // 3 + 1)
    for window_idx in range(out.shape[0]):
        onset = int(rng.integers(low, min(high, out.shape[1])))
        # NumPy moves an advanced-indexed channel axis before the time axis.
        # Assign one channel at a time so multi-sensor faults keep time semantics.
        for sensor_idx in chosen:
            out[window_idx, onset:, int(sensor_idx)] = out[window_idx, onset - 1, int(sensor_idx)]
    return out


def apply_burst_noise(
    x: np.ndarray,
    noise_std: float,
    sensor_indices: list[int],
    seed: int | None = None,
    channel_fraction: float = 0.25,
    block_fraction: float = 0.25,
) -> np.ndarray:
    """Inject Gaussian noise into a shared sensor group over a contiguous time block."""
    if noise_std <= 0 or not sensor_indices:
        return x.copy()
    rng = np.random.default_rng(seed)
    out = x.copy()
    k = max(1, int(round(len(sensor_indices) * channel_fraction)))
    chosen = rng.choice(sensor_indices, size=min(k, len(sensor_indices)), replace=False)
    block_len = max(1, int(round(out.shape[1] * block_fraction)))
    for window_idx in range(out.shape[0]):
        start = int(rng.integers(0, max(1, out.shape[1] - block_len + 1)))
        noise = rng.normal(0.0, noise_std, size=(len(chosen), block_len))
        out[window_idx, start : start + block_len, chosen] += noise
    return out


def apply_correlated_group_missing_with_mask(
    x: np.ndarray,
    sensor_fraction: float,
    sensor_indices: list[int],
    seed: int | None = None,
    block_fraction: float = 0.5,
) -> tuple[np.ndarray, np.ndarray]:
    """Remove one correlated sensor group over a shared contiguous block."""
    if sensor_fraction <= 0 or not sensor_indices:
        return x.copy(), np.ones((*x.shape[:-1], len(sensor_indices)), dtype=np.float32)
    rng = np.random.default_rng(seed)
    out = x.copy()
    mask = np.ones((*out.shape[:-1], len(sensor_indices)), dtype=np.float32)
    k = max(1, int(round(len(sensor_indices) * sensor_fraction)))
    positions = rng.choice(len(sensor_indices), size=min(k, len(sensor_indices)), replace=False)
    features = [sensor_indices[int(position)] for position in positions]
    block_len = max(1, int(round(out.shape[1] * block_fraction)))
    start = int(rng.integers(0, max(1, out.shape[1] - block_len + 1)))
    out[:, start : start + block_len, features] = 0.0
    mask[:, start : start + block_len, positions] = 0.0
    return out, mask


def apply_augmentation_profile(
    x: np.ndarray,
    profile_name: str,
    sensor_indices: list[int],
    noise_std: float = 0.0,
    missing_rate: float = 0.0,
    block_missing_rate: float = 0.0,
    drift_rate: float = 0.0,
    mechanism_apply_probability: float = 1.0,
    corrupted_window_probability: float = 1.0,
    seed: int | None = None,
) -> np.ndarray:
    """Apply a named sensor degradation profile."""
    profile = (profile_name or "none").lower()
    out = x.copy()
    if profile in {"none", "clean"}:
        return out
    if not 0.0 <= mechanism_apply_probability <= 1.0:
        raise ValueError("mechanism_apply_probability must be in [0, 1]")
    if not 0.0 <= corrupted_window_probability <= 1.0:
        raise ValueError("corrupted_window_probability must be in [0, 1]")
    rng = np.random.default_rng(seed)
    if corrupted_window_probability < 1.0 and rng.random() >= corrupted_window_probability:
        return out

    def enabled() -> bool:
        return mechanism_apply_probability >= 1.0 or rng.random() < mechanism_apply_probability

    if profile in {"noise", "noise_only", "missing_noise_drift", "full"} and enabled():
        out = apply_gaussian_noise(out, noise_std, sensor_indices, seed=seed)
    if profile in {"missing", "missing_only", "missing_noise_drift", "full"} and enabled():
        out = apply_sensor_missing(out, missing_rate, sensor_indices, seed=seed)
    if profile in {"block", "block_missing", "missing_noise_drift", "full"} and enabled():
        out = apply_block_missing(out, block_missing_rate, sensor_indices, seed=seed)
    if profile in {"drift", "drift_only", "missing_noise_drift", "full"} and enabled():
        out = apply_sensor_drift(out, drift_rate, sensor_indices, seed=seed)
    valid = {
        "none",
        "clean",
        "noise",
        "noise_only",
        "missing",
        "missing_only",
        "block",
        "block_missing",
        "drift",
        "drift_only",
        "missing_noise_drift",
        "full",
    }
    if profile not in valid:
        raise ValueError(f"Unknown augmentation profile: {profile_name}")
    return out


def apply_augmentation_profile_with_mask(
    x: np.ndarray,
    profile_name: str,
    sensor_indices: list[int],
    noise_std: float = 0.0,
    missing_rate: float = 0.0,
    block_missing_rate: float = 0.0,
    drift_rate: float = 0.0,
    mechanism_apply_probability: float = 1.0,
    corrupted_window_probability: float = 1.0,
    seed: int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply a degradation profile and return an explicit observed mask.

    The mask has shape ``x.shape[:-1] + (len(sensor_indices),)``. Values are
    1 for observed sensor values and 0 for corruption-induced missing values.
    """
    profile = (profile_name or "none").lower()
    out = x.copy()
    mask = np.ones((*out.shape[:-1], len(sensor_indices)), dtype=np.float32)
    if profile in {"none", "clean"}:
        return out, mask
    if not 0.0 <= mechanism_apply_probability <= 1.0:
        raise ValueError("mechanism_apply_probability must be in [0, 1]")
    if not 0.0 <= corrupted_window_probability <= 1.0:
        raise ValueError("corrupted_window_probability must be in [0, 1]")

    seed_rng = np.random.default_rng(seed) if seed is not None else None
    if corrupted_window_probability < 1.0:
        gate_rng = seed_rng if seed_rng is not None else np.random.default_rng()
        if gate_rng.random() >= corrupted_window_probability:
            return out, mask

    def enabled() -> bool:
        if mechanism_apply_probability >= 1.0:
            return True
        gate_rng = seed_rng if seed_rng is not None else np.random.default_rng()
        return bool(gate_rng.random() < mechanism_apply_probability)

    if profile in {"noise", "noise_only", "missing_noise_drift", "full"} and enabled():
        out = apply_gaussian_noise(out, noise_std, sensor_indices, seed=_next_seed(seed_rng))
    if profile in {"missing", "missing_only", "missing_noise_drift", "full"} and enabled():
        out, missing_mask = apply_sensor_missing_with_mask(out, missing_rate, sensor_indices, seed=_next_seed(seed_rng))
        mask *= missing_mask
    if profile in {"block", "block_missing", "missing_noise_drift", "full"} and enabled():
        out, block_mask = apply_block_missing_with_mask(out, block_missing_rate, sensor_indices, seed=_next_seed(seed_rng))
        mask *= block_mask
    if profile in {"drift", "drift_only", "missing_noise_drift", "full"} and enabled():
        out = apply_sensor_drift(out, drift_rate, sensor_indices, seed=_next_seed(seed_rng))
    valid = {
        "none",
        "clean",
        "noise",
        "noise_only",
        "missing",
        "missing_only",
        "block",
        "block_missing",
        "drift",
        "drift_only",
        "missing_noise_drift",
        "full",
    }
    if profile not in valid:
        raise ValueError(f"Unknown augmentation profile: {profile_name}")
    return out, mask
