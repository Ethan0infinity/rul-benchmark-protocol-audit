from __future__ import annotations

import torch
from torch import nn


class MLPRegressor(nn.Module):
    def __init__(self, input_features: int, window_size: int, hidden_size: int = 128, dropout: float = 0.1):
        super().__init__()
        in_dim = input_features * window_size
        self.net = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_dim, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x).squeeze(-1)


class CNN1DRegressor(nn.Module):
    def __init__(self, input_features: int, hidden_size: int = 64, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.net = nn.Sequential(
            nn.Conv1d(input_features, hidden_size, kernel_size, padding=padding),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_size),
            nn.Conv1d(hidden_size, hidden_size, kernel_size, padding=padding),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        return self.net(x).squeeze(-1)


class RNNRegressor(nn.Module):
    def __init__(
        self,
        input_features: int,
        hidden_size: int = 64,
        dropout: float = 0.1,
        rnn_type: str = "gru",
        bidirectional: bool = False,
    ):
        super().__init__()
        rnn_cls = nn.GRU if rnn_type.lower() == "gru" else nn.LSTM
        self.rnn = rnn_cls(
            input_size=input_features,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(out_dim, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        output, _ = self.rnn(x)
        return self.head(output[:, -1]).squeeze(-1)


class Chomp1d(nn.Module):
    def __init__(self, chomp_size: int):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.chomp_size == 0:
            return x
        return x[:, :, : -self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, dropout: float):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.net = nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding, dilation=dilation),
            Chomp1d(padding),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.relu(self.net(x) + self.downsample(x))


class TemporalConvNet(nn.Module):
    def __init__(self, input_features: int, channels: list[int], kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        layers = []
        in_channels = input_features
        for i, out_channels in enumerate(channels):
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size, dilation=2**i, dropout=dropout))
            in_channels = out_channels
        self.network = nn.Sequential(*layers)
        self.out_channels = channels[-1]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        return self.network(x).transpose(1, 2)


class TCNRegressor(nn.Module):
    def __init__(self, input_features: int, channels: list[int], kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        self.tcn = TemporalConvNet(input_features, channels, kernel_size, dropout)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(self.tcn.out_channels, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.tcn(x)
        return self.head(z[:, -1]).squeeze(-1)


class TCNGRURegressor(nn.Module):
    def __init__(
        self,
        input_features: int,
        channels: list[int],
        hidden_size: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.tcn = TemporalConvNet(input_features, channels, kernel_size, dropout)
        self.gru = nn.GRU(self.tcn.out_channels, hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.tcn(x)
        output, _ = self.gru(z)
        return self.head(output[:, -1]).squeeze(-1)


class CNNLSTMRegressor(nn.Module):
    """Compact CNN-LSTM baseline for local trend extraction plus recurrent memory."""

    def __init__(
        self,
        input_features: int,
        hidden_size: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        padding = kernel_size // 2
        conv_channels = max(16, hidden_size)
        self.conv = nn.Sequential(
            nn.Conv1d(input_features, conv_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_channels, conv_channels, kernel_size, padding=padding),
            nn.BatchNorm1d(conv_channels),
            nn.ReLU(),
        )
        self.lstm = nn.LSTM(conv_channels, hidden_size, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(hidden_size, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.conv(x.transpose(1, 2)).transpose(1, 2)
        output, _ = self.lstm(z)
        return self.head(output[:, -1]).squeeze(-1)


class ChannelGate(nn.Module):
    """Small channel-wise gate for selected sensor/setting inputs."""

    def __init__(self, input_features: int, reduction: int = 4):
        super().__init__()
        hidden = max(4, input_features // reduction)
        self.net = nn.Sequential(
            nn.Linear(input_features, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_features),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.net(x.mean(dim=1)).unsqueeze(1)
        return x * weights


class ReliabilityAwareSensorGate(nn.Module):
    """Estimate per-channel reliability from window statistics.

    The gate is intentionally small: it uses mean, standard deviation, zero
    ratio, and average temporal change for each channel, then predicts a
    reliability weight in [0, 1].
    """

    def __init__(self, input_features: int, hidden_size: int | None = None):
        super().__init__()
        hidden = hidden_size or max(8, input_features * 2)
        self.net = nn.Sequential(
            nn.Linear(input_features * 4, hidden),
            nn.ReLU(),
            nn.Linear(hidden, input_features),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=1)
        std = x.std(dim=1, unbiased=False)
        zero_ratio = (torch.abs(x) < 1e-6).float().mean(dim=1)
        if x.shape[1] > 1:
            diff = torch.abs(x[:, 1:, :] - x[:, :-1, :]).mean(dim=1)
        else:
            diff = torch.zeros_like(mean)
        stats = torch.cat([mean, std, zero_ratio, diff], dim=-1)
        weights = self.net(stats).unsqueeze(1)
        return x * weights


class MaskAwareReliabilityGate(nn.Module):
    """Reliability gate that explicitly uses appended observed-mask channels."""

    def __init__(self, input_features: int, mask_feature_count: int = 0, hidden_size: int | None = None):
        super().__init__()
        self.input_features = int(input_features)
        self.mask_feature_count = max(0, int(mask_feature_count))
        self.mask_feature_count = min(self.mask_feature_count, self.input_features)
        self.value_feature_count = self.input_features - self.mask_feature_count
        if self.mask_feature_count:
            stat_dim = self.value_feature_count * 4 + self.mask_feature_count * 3
        else:
            stat_dim = self.input_features * 4
        hidden = hidden_size or max(8, self.input_features * 2)
        self.net = nn.Sequential(
            nn.Linear(stat_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, self.input_features),
            nn.Sigmoid(),
        )

    def weights(self, x: torch.Tensor) -> torch.Tensor:
        if self.mask_feature_count <= 0:
            mean = x.mean(dim=1)
            std = x.std(dim=1, unbiased=False)
            zero_ratio = (torch.abs(x) < 1e-6).float().mean(dim=1)
            diff = torch.abs(x[:, 1:, :] - x[:, :-1, :]).mean(dim=1) if x.shape[1] > 1 else torch.zeros_like(mean)
            stats = torch.cat([mean, std, zero_ratio, diff], dim=-1)
        else:
            values = x[:, :, : self.value_feature_count]
            masks = x[:, :, self.value_feature_count :]
            value_mean = values.mean(dim=1)
            value_std = values.std(dim=1, unbiased=False)
            value_zero_ratio = (torch.abs(values) < 1e-6).float().mean(dim=1)
            value_diff = (
                torch.abs(values[:, 1:, :] - values[:, :-1, :]).mean(dim=1)
                if values.shape[1] > 1
                else torch.zeros_like(value_mean)
            )
            mask_observed_ratio = masks.mean(dim=1)
            mask_missing_ratio = 1.0 - mask_observed_ratio
            if masks.shape[1] > 1:
                consecutive_missing = ((masks[:, 1:, :] < 0.5) & (masks[:, :-1, :] < 0.5)).float().mean(dim=1)
            else:
                consecutive_missing = mask_missing_ratio
            stats = torch.cat(
                [
                    value_mean,
                    value_std,
                    value_zero_ratio,
                    value_diff,
                    mask_observed_ratio,
                    mask_missing_ratio,
                    consecutive_missing,
                ],
                dim=-1,
            )
        return self.net(stats)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.weights(x).unsqueeze(1)


class DepthwiseSeparableTemporalConv(nn.Module):
    """Lightweight local temporal trend extractor."""

    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3, dropout: float = 0.1):
        super().__init__()
        padding = kernel_size // 2
        self.depthwise = nn.Conv1d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            padding=padding,
            groups=in_channels,
        )
        self.pointwise = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1) if in_channels != out_channels else nn.Identity()
        self.norm = nn.LayerNorm(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = self.residual(x.transpose(1, 2)).transpose(1, 2)
        z = x.transpose(1, 2)
        z = self.pointwise(self.depthwise(z)).transpose(1, 2)
        z = self.norm(z)
        z = self.activation(z)
        z = self.dropout(z)
        return z + residual


class TemporalAttentionPooling(nn.Module):
    """Learn a weighted summary over the time dimension."""

    def __init__(self, hidden_size: int):
        super().__init__()
        self.score = nn.Sequential(
            nn.Linear(hidden_size, max(8, hidden_size // 2)),
            nn.Tanh(),
            nn.Linear(max(8, hidden_size // 2), 1),
        )

    def weights(self, sequence: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.score(sequence), dim=1)

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        weights = self.weights(sequence)
        return torch.sum(sequence * weights, dim=1)


class AttentionGRURegressor(nn.Module):
    """GRU/BiGRU baseline with temporal attention pooling."""

    def __init__(
        self,
        input_features: int,
        hidden_size: int = 64,
        dropout: float = 0.1,
        bidirectional: bool = False,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_features,
            hidden_size,
            batch_first=True,
            bidirectional=bidirectional,
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.attention = TemporalAttentionPooling(out_dim)
        self.head = nn.Sequential(
            nn.LayerNorm(out_dim),
            nn.Dropout(dropout),
            nn.Linear(out_dim, max(8, out_dim // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(8, out_dim // 2), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        sequence, _ = self.gru(x)
        pooled = self.attention(sequence)
        return self.head(pooled).squeeze(-1)


class TransformerLiteRegressor(nn.Module):
    """Small Transformer encoder baseline kept lightweight for local/cloud runs."""

    def __init__(
        self,
        input_features: int,
        window_size: int,
        hidden_size: int = 64,
        dropout: float = 0.1,
    ):
        super().__init__()
        nhead = 4 if hidden_size % 4 == 0 else 2 if hidden_size % 2 == 0 else 1
        self.input_proj = nn.Linear(input_features, hidden_size)
        self.position = nn.Parameter(torch.zeros(1, window_size, hidden_size))
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_size,
            nhead=nhead,
            dim_feedforward=hidden_size * 2,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=2,
            enable_nested_tensor=False,
        )
        self.attention = TemporalAttentionPooling(hidden_size)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, max(8, hidden_size // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(8, hidden_size // 2), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.input_proj(x)
        z = z + self.position[:, : z.shape[1], :]
        sequence = self.encoder(z)
        pooled = self.attention(sequence)
        return self.head(pooled).squeeze(-1)


class DualAttentionTCNRegressor(nn.Module):
    """TCN baseline with explicit channel and temporal attention."""

    def __init__(
        self,
        input_features: int,
        channels: list[int],
        hidden_size: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.channel_attention = nn.Sequential(
            nn.Linear(input_features * 2, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, input_features),
            nn.Sigmoid(),
        )
        self.tcn = TemporalConvNet(input_features, channels, kernel_size, dropout)
        self.temporal_attention = TemporalAttentionPooling(self.tcn.out_channels)
        self.head = nn.Sequential(
            nn.LayerNorm(self.tcn.out_channels),
            nn.Dropout(dropout),
            nn.Linear(self.tcn.out_channels, max(8, self.tcn.out_channels // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(8, self.tcn.out_channels // 2), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        statistics = torch.cat([x.mean(dim=1), x.std(dim=1, unbiased=False)], dim=-1)
        channel_weights = self.channel_attention(statistics).unsqueeze(1)
        sequence = self.tcn(x * channel_weights)
        pooled = self.temporal_attention(sequence)
        return self.head(pooled).squeeze(-1)


class ChannelAttention1D(nn.Module):
    """Squeeze-excitation channel attention for temporal convolution features."""

    def __init__(self, channels: int, reduction: int = 4):
        super().__init__()
        hidden = max(8, channels // reduction)
        self.net = nn.Sequential(
            nn.Linear(channels, hidden),
            nn.ReLU(),
            nn.Linear(hidden, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        weights = self.net(x.mean(dim=-1)).unsqueeze(-1)
        return x * weights


class RegimeDualAttentionCNNGRURegressor(nn.Module):
    """Protocol-adapted close-prior baseline: dilated CNN, channel attention, GRU, and temporal attention.

    The condition-aware scaler is supplied by the shared preprocessing protocol rather
    than embedded in the network. This model is an auditable architecture adaptation,
    not a claim of bitwise reproduction of a third-party implementation.
    """

    supports_risk_regularization = True

    def __init__(
        self,
        input_features: int,
        hidden_size: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        channels = max(32, hidden_size)
        self.block1 = TemporalBlock(input_features, channels, kernel_size, dilation=1, dropout=dropout)
        self.attention1 = ChannelAttention1D(channels)
        self.block2 = TemporalBlock(channels, channels, kernel_size, dilation=2, dropout=dropout)
        self.attention2 = ChannelAttention1D(channels)
        self.gru = nn.GRU(channels, hidden_size, batch_first=True)
        self.temporal_attention = TemporalAttentionPooling(hidden_size)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, max(8, hidden_size // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(8, hidden_size // 2), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = x.transpose(1, 2)
        z = self.attention1(self.block1(z))
        z = self.attention2(self.block2(z)).transpose(1, 2)
        sequence, _ = self.gru(z)
        pooled = self.temporal_attention(sequence)
        return self.head(pooled).squeeze(-1)


class SensorGraphGRURegressor(nn.Module):
    """Lightweight learned sensor-graph mixing followed by a GRU encoder."""

    def __init__(self, input_features: int, hidden_size: int = 64, dropout: float = 0.1):
        super().__init__()
        self.adjacency_logits = nn.Parameter(torch.eye(input_features) * 2.0)
        self.graph_norm = nn.LayerNorm(input_features)
        self.graph_projection = nn.Linear(input_features, input_features)
        self.gru = nn.GRU(input_features, hidden_size, batch_first=True)
        self.temporal_attention = TemporalAttentionPooling(hidden_size)
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, max(8, hidden_size // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(8, hidden_size // 2), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        adjacency = torch.softmax(self.adjacency_logits, dim=-1)
        mixed = torch.matmul(x, adjacency)
        mixed = self.graph_norm(x + self.graph_projection(mixed))
        sequence, _ = self.gru(mixed)
        pooled = self.temporal_attention(sequence)
        return self.head(pooled).squeeze(-1)


class QuantileGRURegressor(nn.Module):
    """Probabilistic GRU baseline with 0.1/0.5/0.9 quantile outputs."""

    def __init__(self, input_features: int, hidden_size: int = 64, dropout: float = 0.1):
        super().__init__()
        self.gru = nn.GRU(input_features, hidden_size, batch_first=True)
        self.temporal_attention = TemporalAttentionPooling(hidden_size)
        self.backbone = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, max(8, hidden_size // 2)),
            nn.GELU(),
        )
        self.quantile_head = nn.Linear(max(8, hidden_size // 2), 3)

    def forward_quantiles(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        sequence, _ = self.gru(x)
        features = self.backbone(self.temporal_attention(sequence))
        raw = self.quantile_head(features)
        lower = raw[:, 0]
        median = lower + torch.nn.functional.softplus(raw[:, 1])
        upper = median + torch.nn.functional.softplus(raw[:, 2])
        return lower, median, upper

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, median, _ = self.forward_quantiles(x)
        return median


class RSTCNGRURegressor(nn.Module):
    """Robust sensor-aware TCN-GRU with input channel gate and feature fusion."""

    def __init__(
        self,
        input_features: int,
        channels: list[int],
        hidden_size: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.channel_gate = ChannelGate(input_features)
        self.tcn = TemporalConvNet(input_features, channels, kernel_size, dropout)
        self.gru = nn.GRU(self.tcn.out_channels, hidden_size, batch_first=True)
        self.tcn_proj = nn.Linear(self.tcn.out_channels, hidden_size)
        self.fusion_gate = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.Sigmoid())
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.channel_gate(x)
        z = self.tcn(x)
        gru_out, _ = self.gru(z)
        tcn_last = self.tcn_proj(z[:, -1])
        gru_last = gru_out[:, -1]
        gate = self.fusion_gate(torch.cat([tcn_last, gru_last], dim=-1))
        fused = gate * tcn_last + (1.0 - gate) * gru_last
        return self.head(fused).squeeze(-1)


class RASTGRURegressor(nn.Module):
    """Reliability-aware sensor-gated TCN-GRU for paper experiments."""

    def __init__(
        self,
        input_features: int,
        channels: list[int],
        hidden_size: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.reliability_gate = ReliabilityAwareSensorGate(input_features)
        self.channel_gate = ChannelGate(input_features)
        self.tcn = TemporalConvNet(input_features, channels, kernel_size, dropout)
        self.gru = nn.GRU(self.tcn.out_channels, hidden_size, batch_first=True)
        self.tcn_proj = nn.Linear(self.tcn.out_channels, hidden_size)
        self.fusion_gate = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.Sigmoid())
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, max(8, hidden_size // 2)),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(max(8, hidden_size // 2), 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reliability_gate(x)
        x = self.channel_gate(x)
        z = self.tcn(x)
        gru_out, _ = self.gru(z)
        tcn_last = self.tcn_proj(z[:, -1])
        gru_last = gru_out[:, -1]
        gate = self.fusion_gate(torch.cat([tcn_last, gru_last], dim=-1))
        fused = gate * tcn_last + (1.0 - gate) * gru_last
        return self.head(fused).squeeze(-1)


class RASTGRUV2Regressor(nn.Module):
    """Operating-condition and mask-friendly lightweight RAST-GRU++ model."""

    def __init__(
        self,
        input_features: int,
        channels: list[int],
        hidden_size: int = 64,
        kernel_size: int = 3,
        dropout: float = 0.1,
        use_reliability_gate: bool = True,
        use_channel_gate: bool = True,
        use_local_trend: bool = True,
        use_temporal_attention: bool = True,
        use_uncertainty_head: bool = True,
        mask_feature_count: int = 0,
    ):
        super().__init__()
        local_channels = channels[0] if channels else hidden_size
        local_channels = max(8, int(local_channels))
        self.use_local_trend = bool(use_local_trend)
        self.use_temporal_attention = bool(use_temporal_attention)
        self.use_uncertainty_head = bool(use_uncertainty_head)
        self.supports_risk_regularization = True
        self.reliability_gate = (
            MaskAwareReliabilityGate(input_features, mask_feature_count=mask_feature_count)
            if use_reliability_gate
            else nn.Identity()
        )
        self.channel_gate = ChannelGate(input_features) if use_channel_gate else nn.Identity()
        if self.use_local_trend:
            self.local_trend = nn.Sequential(
                DepthwiseSeparableTemporalConv(input_features, local_channels, kernel_size, dropout),
                DepthwiseSeparableTemporalConv(local_channels, local_channels, kernel_size, dropout),
            )
            gru_input = local_channels
        else:
            self.local_trend = nn.Identity()
            gru_input = input_features
        self.gru = nn.GRU(gru_input, hidden_size, batch_first=True)
        self.attention_pool = TemporalAttentionPooling(hidden_size) if self.use_temporal_attention else nn.Identity()
        self.last_proj = nn.Linear(hidden_size, hidden_size)
        self.fusion_gate = nn.Sequential(nn.Linear(hidden_size * 2, hidden_size), nn.Sigmoid()) if self.use_temporal_attention else nn.Identity()
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, max(8, hidden_size // 2)),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(max(8, hidden_size // 2), 1),
        )
        self.quantile_spread_head = nn.Sequential(
            nn.LayerNorm(hidden_size),
            nn.Linear(hidden_size, 2),
        ) if self.use_uncertainty_head else None

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        x = self.reliability_gate(x)
        x = self.channel_gate(x)
        z = self.local_trend(x)
        sequence, _ = self.gru(z)
        last = self.last_proj(sequence[:, -1])
        if self.use_temporal_attention:
            pooled = self.attention_pool(sequence)
            gate = self.fusion_gate(torch.cat([pooled, last], dim=-1))
            fused = gate * pooled + (1.0 - gate) * last
        else:
            fused = last
        return fused

    def forward_quantiles(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        fused = self.encode(x)
        median = self.head(fused).squeeze(-1)
        if self.quantile_spread_head is None:
            return median, median, median
        spread = torch.nn.functional.softplus(self.quantile_spread_head(fused))
        return median - spread[:, 0], median, median + spread[:, 1]

    def reliability_supervision_loss(self, x: torch.Tensor) -> torch.Tensor:
        if not isinstance(self.reliability_gate, MaskAwareReliabilityGate):
            return x.new_zeros(())
        gate = self.reliability_gate
        if gate.mask_feature_count <= 0:
            return x.new_zeros(())
        masks = x[:, :, gate.value_feature_count :]
        target = masks.mean(dim=1)
        weights = gate.weights(x)
        start = gate.value_feature_count - gate.mask_feature_count
        predicted = weights[:, start : gate.value_feature_count].clamp(1e-6, 1.0 - 1e-6)
        return torch.nn.functional.binary_cross_entropy(predicted, target)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, median, _ = self.forward_quantiles(x)
        return median


class DualMixerLayer(nn.Module):
    """Dual-path mixer layer adapted from Fu et al. (2024), MIT license."""

    def __init__(self, window_size: int, hidden_size: int, dropout: float) -> None:
        super().__init__()
        self.time_mixer = nn.Sequential(
            nn.Linear(window_size, window_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(window_size * 2, window_size),
            nn.Dropout(dropout),
        )
        self.feature_mixer = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Dropout(dropout),
        )
        self.time_gate = nn.Sequential(nn.Linear(window_size, window_size), nn.Sigmoid())
        self.feature_gate = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Sigmoid())
        self.time_norm_1 = nn.LayerNorm(window_size)
        self.feature_norm_1 = nn.LayerNorm(hidden_size)
        self.time_norm_2 = nn.LayerNorm(window_size)
        self.feature_norm_2 = nn.LayerNorm(hidden_size)

    def forward(self, time_path: torch.Tensor, feature_path: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        mixed_time = self.time_norm_1(self.time_mixer(time_path.transpose(1, 2)) + time_path.transpose(1, 2))
        mixed_feature = self.feature_norm_1(self.feature_mixer(feature_path) + feature_path)
        mixed_time = self.time_norm_2(mixed_time + self.feature_gate(mixed_feature).transpose(1, 2))
        mixed_feature = self.feature_norm_2(mixed_feature + self.time_gate(mixed_time).transpose(1, 2))
        return mixed_time.transpose(1, 2), mixed_feature


class OfficialDualMixerRegressor(nn.Module):
    """Official-code-derived Dual-Mixer architecture under the local locked split.

    The layer equations follow the authors' public MIT-licensed implementation
    (repository commit recorded in THIRD_PARTY_NOTICES.md). FSGRI contrastive
    sampling is intentionally excluded, matching the repository's default
    `contra_training=False` Dual-Mixer experiment.
    """

    def __init__(
        self,
        input_features: int,
        window_size: int,
        hidden_size: int = 32,
        num_layers: int = 6,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.input_embedding = nn.Linear(input_features, hidden_size)
        self.layers = nn.ModuleList(
            DualMixerLayer(window_size, hidden_size, dropout) for _ in range(num_layers)
        )
        self.output_time_gate = nn.Sequential(nn.Linear(window_size, window_size), nn.Sigmoid())
        self.output_feature_gate = nn.Sequential(nn.Linear(hidden_size, hidden_size), nn.Sigmoid())
        self.head = nn.Linear(window_size * hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        time_path = self.input_embedding(x)
        feature_path = time_path
        for layer in self.layers:
            time_path, feature_path = layer(time_path, feature_path)
        time_weighted = self.output_time_gate(time_path.transpose(1, 2)).transpose(1, 2)
        feature_weighted = self.output_feature_gate(feature_path)
        fused = torch.flatten(time_weighted + feature_weighted, start_dim=1)
        return self.head(fused).squeeze(-1)


def build_model(
    name: str,
    input_features: int,
    window_size: int,
    hidden_size: int,
    tcn_channels: list[int],
    kernel_size: int,
    dropout: float,
    use_reliability_gate: bool = True,
    use_channel_gate: bool = True,
    use_local_trend: bool = True,
    use_temporal_attention: bool = True,
    use_uncertainty_head: bool = True,
    mask_feature_count: int = 0,
) -> nn.Module:
    name = name.lower()
    if name == "mlp":
        return MLPRegressor(input_features, window_size, hidden_size, dropout)
    if name in {"cnn", "cnn1d", "1d_cnn"}:
        return CNN1DRegressor(input_features, hidden_size, kernel_size, dropout)
    if name == "lstm":
        return RNNRegressor(input_features, hidden_size, dropout, rnn_type="lstm")
    if name == "gru":
        return RNNRegressor(input_features, hidden_size, dropout, rnn_type="gru")
    if name == "bigru":
        return RNNRegressor(input_features, hidden_size, dropout, rnn_type="gru", bidirectional=True)
    if name == "tcn":
        return TCNRegressor(input_features, tcn_channels, kernel_size, dropout)
    if name == "tcn_gru":
        return TCNGRURegressor(input_features, tcn_channels, hidden_size, kernel_size, dropout)
    if name in {"cnn_lstm", "cnnlstm"}:
        return CNNLSTMRegressor(input_features, hidden_size, kernel_size, dropout)
    if name in {"attention_gru", "att_gru"}:
        return AttentionGRURegressor(input_features, hidden_size, dropout, bidirectional=False)
    if name in {"bigru_attention", "attention_bigru", "bi_gru_attention"}:
        return AttentionGRURegressor(input_features, hidden_size, dropout, bidirectional=True)
    if name in {"transformer_lite", "lite_transformer"}:
        return TransformerLiteRegressor(input_features, window_size, hidden_size, dropout)
    if name in {"dual_attention_tcn", "da_tcn"}:
        return DualAttentionTCNRegressor(input_features, tcn_channels, hidden_size, kernel_size, dropout)
    if name in {"regime_dual_attention_cnn_gru", "regime_da_cnn_gru"}:
        return RegimeDualAttentionCNNGRURegressor(input_features, hidden_size, kernel_size, dropout)
    if name in {"sensor_graph_gru", "sg_gru"}:
        return SensorGraphGRURegressor(input_features, hidden_size, dropout)
    if name in {"quantile_gru", "probabilistic_gru"}:
        return QuantileGRURegressor(input_features, hidden_size, dropout)
    if name in {"official_dual_mixer", "dual_mixer_official", "dual_mixer"}:
        return OfficialDualMixerRegressor(
            input_features=input_features,
            window_size=window_size,
            hidden_size=hidden_size,
            num_layers=6,
            dropout=dropout,
        )
    if name == "rs_tcn_gru":
        return RSTCNGRURegressor(input_features, tcn_channels, hidden_size, kernel_size, dropout)
    if name in {"rast_gru", "ra_tcn_gru", "rast_tcn_gru"}:
        return RASTGRURegressor(input_features, tcn_channels, hidden_size, kernel_size, dropout)
    if name in {"rast_gru_v2", "rast_gru_pp", "ocm_rast_gru"}:
        return RASTGRUV2Regressor(
            input_features,
            tcn_channels,
            hidden_size,
            kernel_size,
            dropout,
            use_reliability_gate=use_reliability_gate,
            use_channel_gate=use_channel_gate,
            use_local_trend=use_local_trend,
            use_temporal_attention=use_temporal_attention,
            use_uncertainty_head=use_uncertainty_head,
            mask_feature_count=mask_feature_count,
        )
    raise ValueError(f"Unknown model name: {name}")
