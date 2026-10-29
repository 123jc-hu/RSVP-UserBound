"""PRISM, the reference decoder of the paper.

Forward path:

    x [B, 1, C, T]
      -> sagittal L/M/R projection to 3M virtual channels
      -> concat raw and virtual channels [B, C+3M, T]
      -> pre-W_s per-channel 3-scale temporal filters, concat over scales
      -> pre-W_s spatial dropout shared across the three scale branches
      -> W_s spatial projection from 3(C+3M) features to D components
      -> component-wise 3-scale temporal processing
      -> global component mixer
      -> max-norm linear classifier

Configuration keys (erp_net_*) also cover the ablations and the random-grouping control.
"""

from __future__ import annotations

import random
import re
from typing import Sequence

import torch
import torch.nn.functional as F
from torch import nn


def _cfg(config: dict, key: str, default):
    value = config.get(key, default)
    return default if value is None else value


def _odd_kernel(length: int, *, minimum: int = 3) -> int:
    length = max(int(length), int(minimum))
    return length if length % 2 == 1 else length + 1


def _kernel_family(fs: int, mode: str = "multiscale") -> tuple[int, ...]:
    mode = str(mode).strip().lower()
    if mode == "single_half":
        return (_odd_kernel(int(fs) // 2, minimum=3),)
    if mode != "multiscale":
        raise ValueError(f"Unsupported PRISM kernel mode: {mode}")
    return (
        _odd_kernel(int(fs) // 2, minimum=9),
        _odd_kernel(int(fs) // 4, minimum=9),
        _odd_kernel(int(fs) // 8, minimum=9),
    )


def _make_activation(name: str) -> nn.Module:
    name = str(name).strip().lower()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU(inplace=False)
    if name == "elu":
        return nn.ELU()
    raise ValueError(f"Unsupported PRISM activation: {name}")


def _make_depthwise_temporal_block(
    channels: int,
    kernel_size: int,
    *,
    activation: str,
    bias: bool = False,
) -> nn.Sequential:
    pad = int(kernel_size) // 2
    return nn.Sequential(
        nn.Conv1d(
            int(channels),
            int(channels),
            kernel_size=int(kernel_size),
            padding=pad,
            groups=int(channels),
            bias=bool(bias),
        ),
        nn.BatchNorm1d(int(channels)),
        _make_activation(activation),
    )


_LMR_LEFT_INDICES: tuple[int, ...] = (
    0, 3, 5, 6, 7, 8,
    14, 15, 16, 17,
    23, 24, 25, 26,
    32, 33, 34, 35,
    41, 42, 43, 44,
    50, 51, 52,
    57, 58,
)
_LMR_MIDLINE_INDICES: tuple[int, ...] = (1, 9, 18, 27, 36, 45, 53, 59)
_LMR_RIGHT_INDICES: tuple[int, ...] = (
    2, 4, 10, 11, 12, 13,
    19, 20, 21, 22,
    28, 29, 30, 31,
    37, 38, 39, 40,
    46, 47, 48, 49,
    54, 55, 56,
    60, 61,
)

_THINGS_CHANNELS_63: tuple[str, ...] = (
    "Fp1", "Fz", "F3", "F7", "FT9", "FC5", "FC1", "C3", "T7", "TP9",
    "CP5", "CP1", "Pz", "P3", "P7", "O1", "Oz", "O2", "P4", "P8",
    "TP10", "CP6", "CP2", "C4", "T8", "FT10", "FC6", "FC2", "F4", "F8",
    "Fp2", "AF7", "AF3", "AFz", "F1", "F5", "FT7", "FC3", "C1", "C5",
    "TP7", "CP3", "P1", "P5", "PO7", "PO3", "POz", "PO4", "PO8", "P6",
    "P2", "CPz", "CP4", "TP8", "C6", "C2", "FC4", "FT8", "F6", "AF8",
    "AF4", "F2", "FCz",
)

def _validate_region_groups(groups: Sequence[Sequence[int]], n_channels: int) -> tuple[tuple[int, ...], ...]:
    cleaned: list[tuple[int, ...]] = []
    seen: list[int] = []
    for group in groups:
        idx = tuple(int(i) for i in group)
        if not idx:
            raise ValueError("PRISM regional input projector received an empty region.")
        cleaned.append(idx)
        seen.extend(idx)
    if sorted(seen) != list(range(int(n_channels))):
        raise ValueError(
            "PRISM regional input projector groups must cover each raw channel exactly once. "
            f"Got sorted indices={sorted(seen)} for n_channels={n_channels}."
        )
    return tuple(cleaned)


def _sagittal_groups_from_names(channel_names: Sequence[str]) -> tuple[tuple[int, ...], ...]:
    """Derive left/midline/right groups from standard EEG channel names."""
    left: list[int] = []
    midline: list[int] = []
    right: list[int] = []
    for index, name in enumerate(channel_names):
        normalized = str(name).strip()
        if normalized.lower().endswith("z"):
            midline.append(index)
            continue
        match = re.search(r"(\d+)$", normalized)
        if match is None:
            raise ValueError(
                f"Cannot infer sagittal side for EEG channel {normalized!r}."
            )
        (left if int(match.group(1)) % 2 else right).append(index)
    return tuple(left), tuple(midline), tuple(right)


def _regional_groups(
    mode: str,
    n_channels: int,
    group_seed: int = 2026,
) -> tuple[tuple[int, ...], ...]:
    mode = str(mode).strip().lower()
    n_channels = int(n_channels)
    if n_channels not in {62, 63}:
        raise ValueError(
            "PRISM regional input projector currently supports the 62-channel THU/CAS "
            "layout and the 63-channel THINGS layout only. "
            f"Got n_channels={n_channels}."
        )
    lmr_groups = (
        _LMR_LEFT_INDICES,
        _LMR_MIDLINE_INDICES,
        _LMR_RIGHT_INDICES,
    )
    if n_channels == 63:
        # THINGS has a different channel order from THU/CAS. Derive its groups
        # from the actual montage names rather than reusing 62-channel indices.
        lmr_groups = _sagittal_groups_from_names(_THINGS_CHANNELS_63)
    if mode == "sagittal_lmr":
        return _validate_region_groups(lmr_groups, n_channels)
    if mode == "random_lmr":
        # Control: groups of the same sizes with randomly assigned channels.
        group_sizes = tuple(len(group) for group in lmr_groups)
        shuffled = list(range(n_channels))
        random.Random(int(group_seed)).shuffle(shuffled)
        groups = []
        offset = 0
        for size in group_sizes:
            groups.append(tuple(shuffled[offset:offset + size]))
            offset += size
        return _validate_region_groups(groups, n_channels)
    raise ValueError(f"Unsupported PRISM regional input projector mode: {mode}")


class RegionalChannelProjector(nn.Module):
    """Project each fixed channel group into equal-size virtual channels."""

    def __init__(
        self,
        n_channels: int,
        mode: str,
        virtual_channels_per_region: int,
        group_seed: int = 2026,
    ):
        super().__init__()
        self.n_channels = int(n_channels)
        self.mode = str(mode).strip().lower()
        self.group_seed = int(group_seed)
        self.virtual_channels_per_region = int(virtual_channels_per_region)
        if self.virtual_channels_per_region <= 0:
            raise ValueError("virtual_channels_per_region must be positive.")

        self.region_groups = _regional_groups(
            self.mode,
            self.n_channels,
            group_seed=self.group_seed,
        )
        self.projectors = nn.ParameterList(
            [
                nn.Parameter(torch.empty(self.virtual_channels_per_region, len(group)))
                for group in self.region_groups
            ]
        )
        for weight in self.projectors:
            nn.init.xavier_uniform_(weight)

    @property
    def out_channels(self) -> int:
        return len(self.region_groups) * self.virtual_channels_per_region

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        region_outputs = []
        for group, weight in zip(self.region_groups, self.projectors):
            region_x = x[:, list(group), :]
            region_outputs.append(torch.einsum("bct,mc->bmt", region_x, weight))
        return torch.cat(region_outputs, dim=1)


class MaxNormLinear(nn.Linear):
    """Linear classifier with EEG-style max-norm weight constraint."""

    def __init__(self, *args, max_norm: float = 1.0, **kwargs):
        super().__init__(*args, **kwargs)
        self.max_norm = float(max_norm)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.max_norm > 0.0:
            with torch.no_grad():
                self.weight.renorm_(p=2, dim=0, maxnorm=self.max_norm)
        return super().forward(x)


class PreWsTemporalFilter(nn.Module):
    """Per-channel multi-scale temporal filtering before W_s.

    Each branch is depthwise Conv1d(C -> C) + BN + activation. Branch outputs
    are concatenated on the channel axis, giving [B, 3C, T].
    """

    def __init__(
        self,
        n_channels: int,
        fs: int,
        dropout: float,
        activation: str,
        kernel_mode: str = "multiscale",
    ):
        super().__init__()
        self.n_channels = int(n_channels)
        self.kernel_mode = str(kernel_mode).strip().lower()
        self.kernels = _kernel_family(int(fs), mode=self.kernel_mode)
        self.branches = nn.ModuleList(
            [
                _make_depthwise_temporal_block(
                    self.n_channels,
                    kernel,
                    activation=activation,
                    bias=False,
                )
                for kernel in self.kernels
            ]
        )
        self.dropout = nn.Dropout(float(dropout))

    @property
    def n_scales(self) -> int:
        return len(self.branches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        branch_outputs = [branch(x) for branch in self.branches]
        return self.dropout(torch.cat(branch_outputs, dim=1))


class SpatialProjector(nn.Module):
    """Learned W_s projection from 3C scale-channel features to D components."""

    def __init__(self, in_channels: int, n_components: int):
        super().__init__()
        self.in_channels = int(in_channels)
        self.n_components = int(n_components)
        self.weight = nn.Parameter(torch.empty(self.n_components, self.in_channels))
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.einsum("bct,dc->bdt", x, self.weight)


class ComponentWiseMultiScaleTemporal(nn.Module):
    """Per-component multi-scale temporal processing after W_s."""

    def __init__(
        self,
        n_components: int,
        fs: int,
        token_dim: int,
        dropout: float,
        activation: str,
        kernel_mode: str = "multiscale",
    ):
        super().__init__()
        self.n_components = int(n_components)
        self.token_dim = int(token_dim)
        self.kernel_mode = str(kernel_mode).strip().lower()
        self.kernels = _kernel_family(int(fs), mode=self.kernel_mode)
        self.branches = nn.ModuleList(
            [
                nn.Sequential(
                    _make_depthwise_temporal_block(
                        self.n_components,
                        kernel,
                        activation=activation,
                        bias=False,
                    ),
                    nn.AvgPool1d(kernel_size=2 * self.token_dim, stride=2 * self.token_dim),
                    nn.Dropout(float(dropout)),
                )
                for kernel in self.kernels
            ]
        )

    @property
    def feature_dim(self) -> int:
        return self.token_dim * len(self.branches)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, D, T] -> three [B, D, token_dim] -> [B, D, 3*token_dim]
        return torch.cat([branch(x) for branch in self.branches], dim=-1)


class GlobalComponentMixer(nn.Module):
    """Mix D spatial components for each temporal token position."""

    def __init__(
        self,
        n_components: int,
        feature_dim: int,
        dropout: float,
        activation: str,
    ):
        super().__init__()
        self.mixer = nn.Sequential(
            nn.Conv1d(
                int(feature_dim),
                int(feature_dim),
                kernel_size=int(n_components),
                groups=int(feature_dim),
                bias=False,
            ),
            nn.BatchNorm1d(int(feature_dim)),
            _make_activation(activation),
            nn.Conv1d(int(feature_dim), int(feature_dim), kernel_size=1, bias=False),
            nn.Dropout(float(dropout)),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [B, D, F] -> [B, F, D] -> [B, F, 1] -> [B, F]
        return self.mixer(x.transpose(1, 2)).squeeze(-1)


class PRISM(nn.Module):
    """Final PRISM backbone used in the manuscript."""

    def __init__(self, config: dict):
        super().__init__()
        self.raw_input_channels = int(config["n_channels"])
        self.input_projector_mode = str(
            _cfg(config, "erp_net_input_region_projector", "none")
        ).strip().lower()
        self.input_projector_fusion = str(
            _cfg(config, "erp_net_input_region_projector_fusion", "replace")
        ).strip().lower()
        self.input_projector_virtual_channels = int(
            _cfg(config, "erp_net_input_region_projector_dim", 8)
        )
        self.input_projector_group_seed = int(
            _cfg(config, "erp_net_input_region_projector_group_seed", 2026)
        )
        self.input_projector_scale = float(
            _cfg(config, "erp_net_input_region_projector_scale", 1.0)
        )
        if self.input_projector_mode in {"", "none", "off", "false"}:
            self.input_projector = nn.Identity()
            self.n_channels = self.raw_input_channels
        else:
            self.input_projector = RegionalChannelProjector(
                n_channels=self.raw_input_channels,
                mode=self.input_projector_mode,
                virtual_channels_per_region=self.input_projector_virtual_channels,
                group_seed=self.input_projector_group_seed,
            )
            self.regional_input_channels = int(self.input_projector.out_channels)
            self.n_channels = self.regional_input_channels
            if self.input_projector_fusion == "concat":
                self.n_channels += self.raw_input_channels
        self.fs = int(config["fs"])
        self.n_class = int(config["n_class"])

        self.n_components = int(
            _cfg(config, "erp_net_n_components", max(1, self.n_channels // 2))
        )
        self.token_dim = int(_cfg(config, "erp_net_token_dim", 8))
        self.dropout_p = float(_cfg(config, "erp_net_dropout", 0.3))
        self.activation = str(_cfg(config, "erp_net_activation", "gelu")).strip().lower()
        self.prews_spatial_dropout_prob = float(
            _cfg(config, "erp_net_prews_spatial_dropout_prob", 0.0)
        )
        self.max_norm = float(_cfg(config, "erp_net_max_norm", 1.0))
        self.bypass_spatial_projector = bool(
            _cfg(config, "erp_net_bypass_spatial_projector", False)
        )


        self.pre_ws_filter = PreWsTemporalFilter(
            n_channels=self.n_channels,
            fs=self.fs,
            dropout=self.dropout_p,
            activation=self.activation,
            kernel_mode=str(_cfg(config, "erp_net_prews_kernel_mode", "multiscale")),
        )
        self.prews_branch_count = self.pre_ws_filter.n_scales
        self.pre_ws_out_channels = self.prews_branch_count * self.n_channels

        if self.bypass_spatial_projector:
            self.spatial_projector = nn.Identity()
            self.projected_n_components = int(self.pre_ws_out_channels)
        else:
            self.spatial_projector = SpatialProjector(
                in_channels=self.pre_ws_out_channels,
                n_components=self.n_components,
            )
            self.projected_n_components = int(self.n_components)
        self.component_temporal = ComponentWiseMultiScaleTemporal(
            n_components=self.projected_n_components,
            fs=self.fs,
            token_dim=self.token_dim,
            dropout=self.dropout_p,
            activation=self.activation,
            kernel_mode=str(
                _cfg(config, "erp_net_component_temporal_kernel_mode", "multiscale")
            ),
        )
        self.token_feature_dim = int(self.component_temporal.feature_dim)
        self.component_mixer = GlobalComponentMixer(
            n_components=self.projected_n_components,
            feature_dim=self.token_feature_dim,
            dropout=self.dropout_p,
            activation=self.activation,
        )
        self.feature_dim = int(self.token_feature_dim)
        self.classifier = MaxNormLinear(
            self.feature_dim,
            self.n_class,
            max_norm=self.max_norm,
        )

    def _apply_prews_spatial_dropout(
        self,
        x: torch.Tensor,
        *,
        branch_count: int | None = None,
        n_channels: int | None = None,
    ) -> torch.Tensor:
        if (not self.training) or self.prews_spatial_dropout_prob <= 0.0:
            return x

        branch_count = int(self.prews_branch_count if branch_count is None else branch_count)
        n_channels = int(self.n_channels if n_channels is None else n_channels)
        expected_channels = branch_count * n_channels
        if int(x.shape[1]) != expected_channels:
            raise RuntimeError(
                f"Expected [B,{expected_channels},T] before W_s, got {tuple(x.shape)}"
            )

        keep_prob = 1.0 - float(self.prews_spatial_dropout_prob)

        # Use the same physical-channel mask across all three scale groups.
        x_view = x.view(x.shape[0], branch_count, n_channels, x.shape[2])
        mask = torch.empty(
            int(x.shape[0]),
            1,
            n_channels,
            1,
            device=x.device,
            dtype=x.dtype,
        ).bernoulli_(keep_prob)
        x_view = x_view * (mask / keep_prob)
        return x_view.reshape(x.shape[0], expected_channels, x.shape[2])

    def forward(self, x: torch.Tensor, return_features: bool = False):
        if x.ndim == 4:
            x = x.squeeze(1)
        if x.ndim != 3:
            raise ValueError(f"PRISM expects [B,1,C,T] or [B,C,T], got {tuple(x.shape)}")
        if int(x.shape[1]) != self.raw_input_channels:
            raise ValueError(
                f"PRISM expected {self.raw_input_channels} channels, got {int(x.shape[1])}"
            )

        if isinstance(self.input_projector, RegionalChannelProjector):
            regional_x = self.input_projector(x)
            if self.input_projector_fusion == "concat":
                x = torch.cat([x, regional_x * self.input_projector_scale], dim=1)
            x = self.pre_ws_filter(x)           # [B, 3C, T]
            x = self._apply_prews_spatial_dropout(x)
            x = self.spatial_projector(x)       # [B, D, T] or [B, 3C, T] when no W_s
        else:
            x = self.input_projector(x)
            x = self.pre_ws_filter(x)           # [B, 3C, T]
            x = self._apply_prews_spatial_dropout(x)
            x = self.spatial_projector(x)       # [B, D, T] or [B, 3C, T] when no W_s
        component_series = x
        x = self.component_temporal(x)          # [B, D, F]
        features = self.component_mixer(x)      # [B, F]
        logits = self.classifier(features)

        if return_features:
            return logits, {
                "features": F.normalize(features, p=2, dim=-1),
                "component_series": component_series,
            }
        return logits


def count_trainable_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def module_parameter_summary(model: PRISM) -> dict[str, int]:
    modules: list[tuple[str, nn.Module]] = [
        ("input_projector", model.input_projector),
        ("pre_ws_filter", model.pre_ws_filter),
        ("spatial_projector", model.spatial_projector),
        ("component_temporal", model.component_temporal),
        ("component_mixer", model.component_mixer),
        ("classifier", model.classifier),
    ]
    return {name: count_trainable_parameters(module) for name, module in modules}


