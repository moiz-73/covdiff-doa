"""A small U-Net operating on ``(B, 2, M, M)`` covariance tensors.

Channel 0 is the real part and channel 1 the imaginary part of the covariance
(see :func:`covdiff.utils.to_model_input`). The network is deliberately tiny
(well under 5M parameters at the default configuration) because the spatial
resolution is at most :math:`16 \\times 16` and the dataset is small.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

__all__ = ["SinusoidalTimeEmbedding", "ResidualBlock", "CovUNet"]


def _groups(channels: int) -> int:
    """GroupNorm group count: ``min(8, channels)`` with a divisibility guard."""
    g = min(8, int(channels))
    while g > 1 and int(channels) % g != 0:
        g -= 1
    return max(g, 1)


class SinusoidalTimeEmbedding(nn.Module):
    """Standard sinusoidal embedding of integer diffusion timesteps.

    Parameters
    ----------
    dim:
        Embedding width (must be even).
    max_period:
        Longest wavelength of the geometric frequency ladder.
    """

    def __init__(self, dim: int, max_period: float = 10000.0):
        super().__init__()
        dim = int(dim)
        if dim < 2 or dim % 2 != 0:
            raise ValueError("time embedding dim must be even and >= 2")
        self.dim = dim
        half = dim // 2
        freqs = torch.exp(
            -math.log(float(max_period))
            * torch.arange(half, dtype=torch.float32)
            / float(half)
        )
        self.register_buffer("freqs", freqs, persistent=False)

    def forward(self, t: Tensor) -> Tensor:
        """Embed ``(B,)`` (or ``(B, 1)``) timesteps into ``(B, dim)``."""
        t = t.reshape(-1).to(dtype=torch.float32, device=self.freqs.device)
        args = t[:, None] * self.freqs[None, :]
        return torch.cat([torch.sin(args), torch.cos(args)], dim=-1)


class ResidualBlock(nn.Module):
    """``Conv -> GroupNorm -> SiLU`` residual block with additive time context."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_emb_dim: int,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.norm1 = nn.GroupNorm(_groups(in_channels), in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.time_proj = nn.Linear(time_emb_dim, out_channels)
        self.norm2 = nn.GroupNorm(_groups(out_channels), out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.dropout = nn.Dropout(float(dropout)) if dropout > 0 else nn.Identity()
        self.skip = (
            nn.Conv2d(in_channels, out_channels, 1)
            if in_channels != out_channels
            else nn.Identity()
        )

    def forward(self, x: Tensor, t_emb: Tensor) -> Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.time_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class CovUNet(nn.Module):
    """Denoising U-Net for covariance tensors.

    Parameters
    ----------
    in_channels:
        Number of input channels (2: real and imaginary parts).
    base_channels:
        Width of the first level; later levels are ``base_channels * mult``.
    channel_mults:
        Per-level channel multipliers. The encoder downsamples
        ``len(channel_mults) - 1`` times, so ``M`` should be divisible by
        ``2 ** (len(channel_mults) - 1)`` (8 and 16 both work with the default).
    time_emb_dim:
        Width of the sinusoidal timestep embedding.
    dropout:
        Dropout probability inside the residual blocks (0 disables it).

    Notes
    -----
    The final convolution is zero-initialised, so a freshly constructed network
    predicts exactly zero noise. Spatial sizes are matched against the skip
    tensors with :func:`torch.nn.functional.interpolate`, which keeps the module
    usable for odd ``M`` as well.
    """

    def __init__(
        self,
        in_channels: int = 2,
        base_channels: int = 32,
        channel_mults: tuple[int, ...] = (1, 2, 4),
        time_emb_dim: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        channel_mults = tuple(int(m) for m in channel_mults)
        if not channel_mults:
            raise ValueError("channel_mults must not be empty")
        if any(m < 1 for m in channel_mults):
            raise ValueError("channel_mults entries must be >= 1")

        self.in_channels = int(in_channels)
        self.out_channels = int(in_channels)
        self.base_channels = int(base_channels)
        self.channel_mults = channel_mults
        self.time_emb_dim = int(time_emb_dim)
        self.n_levels = len(channel_mults)

        channels = [self.base_channels * m for m in channel_mults]
        self.channels = tuple(channels)

        self.time_embed = SinusoidalTimeEmbedding(self.time_emb_dim)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.time_emb_dim, 4 * self.time_emb_dim),
            nn.SiLU(),
            nn.Linear(4 * self.time_emb_dim, self.time_emb_dim),
        )

        self.init_conv = nn.Conv2d(self.in_channels, channels[0], 3, padding=1)

        self.downs = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        prev = channels[0]
        for level, ch in enumerate(channels):
            blocks = nn.ModuleList()
            for _ in range(2):
                blocks.append(ResidualBlock(prev, ch, self.time_emb_dim, dropout))
                prev = ch
            self.downs.append(blocks)
            if level < self.n_levels - 1:
                self.downsamples.append(nn.Conv2d(ch, ch, 3, stride=2, padding=1))
            else:
                self.downsamples.append(nn.Identity())

        self.mid = nn.ModuleList(
            [
                ResidualBlock(prev, prev, self.time_emb_dim, dropout),
                ResidualBlock(prev, prev, self.time_emb_dim, dropout),
            ]
        )

        self.up_convs = nn.ModuleList()
        self.ups = nn.ModuleList()
        for level in reversed(range(self.n_levels)):
            ch = channels[level]
            if level < self.n_levels - 1:
                self.up_convs.append(nn.Conv2d(prev, ch, 3, padding=1))
            else:
                self.up_convs.append(nn.Identity())
            blocks = nn.ModuleList()
            blocks.append(ResidualBlock(2 * ch, ch, self.time_emb_dim, dropout))
            blocks.append(ResidualBlock(ch, ch, self.time_emb_dim, dropout))
            prev = ch
            self.ups.append(blocks)

        self.out_norm = nn.GroupNorm(_groups(channels[0]), channels[0])
        self.out_conv = nn.Conv2d(channels[0], self.out_channels, 3, padding=1)
        nn.init.zeros_(self.out_conv.weight)
        nn.init.zeros_(self.out_conv.bias)

    # ------------------------------------------------------------------ forward
    def forward(self, x: Tensor, t: Tensor) -> Tensor:
        """Denoise ``x`` at timesteps ``t``.

        Parameters
        ----------
        x:
            ``(B, in_channels, M, M)`` real tensor.
        t:
            Integer timesteps, ``(B,)`` or a scalar.

        Returns
        -------
        torch.Tensor
            Same shape as ``x``.
        """
        if x.dim() != 4:
            raise ValueError("CovUNet expects a (B, C, M, M) tensor")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"expected {self.in_channels} input channels, got {x.shape[1]}"
            )

        t_emb = self.time_mlp(self.time_embed(t))

        h = self.init_conv(x)
        skips: list[Tensor] = []
        for blocks, down in zip(self.downs, self.downsamples):
            for block in blocks:
                h = block(h, t_emb)
            skips.append(h)
            h = down(h)

        for block in self.mid:
            h = block(h, t_emb)

        for i, level in enumerate(reversed(range(self.n_levels))):
            skip = skips[level]
            if h.shape[-2:] != skip.shape[-2:]:
                h = F.interpolate(h, size=skip.shape[-2:], mode="nearest")
            up = self.up_convs[i]
            if not isinstance(up, nn.Identity):
                h = up(h)
            h = torch.cat([h, skip], dim=1)
            for block in self.ups[i]:
                h = block(h, t_emb)

        h = self.out_conv(F.silu(self.out_norm(h)))
        return h

    # ------------------------------------------------------------------ helpers
    def n_parameters(self) -> int:
        """Total number of trainable scalar parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
