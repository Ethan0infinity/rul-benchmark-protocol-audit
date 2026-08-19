from __future__ import annotations

import torch
from torch import nn


class WeightedHuberLoss(nn.Module):
    """Huber loss with a larger weight near end-of-life samples."""

    def __init__(self, delta: float = 1.0, late_life_threshold: float = 30.0, late_life_weight: float = 1.5):
        super().__init__()
        self.delta = float(delta)
        self.late_life_threshold = float(late_life_threshold)
        self.late_life_weight = float(late_life_weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view_as(target)
        err = pred - target
        abs_err = torch.abs(err)
        quadratic = torch.minimum(abs_err, torch.tensor(self.delta, device=pred.device, dtype=pred.dtype))
        linear = abs_err - quadratic
        loss = 0.5 * quadratic.pow(2) + self.delta * linear
        weights = torch.where(
            target <= self.late_life_threshold,
            torch.full_like(target, self.late_life_weight),
            torch.ones_like(target),
        )
        return (loss * weights).mean()


class AsymmetricWeightedHuberLoss(nn.Module):
    """Weighted Huber with stronger penalty for late RUL over-estimation."""

    def __init__(
        self,
        delta: float = 1.0,
        late_life_threshold: float = 30.0,
        late_life_weight: float = 1.5,
        late_over_weight: float = 1.25,
    ):
        super().__init__()
        self.delta = float(delta)
        self.late_life_threshold = float(late_life_threshold)
        self.late_life_weight = float(late_life_weight)
        self.late_over_weight = float(late_over_weight)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view_as(target)
        err = pred - target
        abs_err = torch.abs(err)
        quadratic = torch.minimum(abs_err, torch.tensor(self.delta, device=pred.device, dtype=pred.dtype))
        linear = abs_err - quadratic
        loss = 0.5 * quadratic.pow(2) + self.delta * linear
        late_life_weights = torch.where(
            target <= self.late_life_threshold,
            torch.full_like(target, self.late_life_weight),
            torch.ones_like(target),
        )
        asymmetric_weights = torch.where(
            err > 0,
            torch.full_like(target, self.late_over_weight),
            torch.ones_like(target),
        )
        return (loss * late_life_weights * asymmetric_weights).mean()


class SmoothLateRiskLoss(nn.Module):
    """Differentiable near-failure overestimation risk."""

    def __init__(self, threshold: float = 30.0, temperature: float = 2.0):
        super().__init__()
        self.threshold = float(threshold)
        self.temperature = float(temperature)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = pred.view_as(target)
        temperature = max(self.temperature, 1e-6)
        near_failure_weight = torch.sigmoid((self.threshold - target) / temperature)
        smooth_overestimate = torch.nn.functional.softplus((pred - target) / temperature) * temperature
        return (near_failure_weight * smooth_overestimate).sum() / near_failure_weight.sum().clamp_min(1.0)


def pinball_loss(prediction: torch.Tensor, target: torch.Tensor, quantile: float) -> torch.Tensor:
    prediction = prediction.view_as(target)
    error = target - prediction
    q = float(quantile)
    return torch.maximum(q * error, (q - 1.0) * error).mean()


def build_loss(
    name: str,
    delta: float,
    late_life_threshold: float,
    late_life_weight: float,
    late_over_weight: float = 1.0,
) -> nn.Module:
    name = name.lower()
    if name == "weighted_huber":
        return WeightedHuberLoss(delta, late_life_threshold, late_life_weight)
    if name in {"asymmetric_weighted_huber", "weighted_asymmetric_huber", "asymmetric_huber"}:
        return AsymmetricWeightedHuberLoss(delta, late_life_threshold, late_life_weight, late_over_weight)
    if name == "huber":
        return nn.HuberLoss(delta=delta)
    if name == "mse":
        return nn.MSELoss()
    if name == "mae":
        return nn.L1Loss()
    raise ValueError(f"Unknown loss: {name}")
