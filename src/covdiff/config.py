"""Configuration objects shared across :mod:`covdiff`.

Every experiment in the repository is driven by the frozen dataclasses defined
here so that configuration can be serialised next to results and diffed
between runs.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from typing import Any, Optional, Tuple

import numpy as np

#: The nine SNR levels used throughout the source paper.
DEFAULT_SNR_GRID: Tuple[float, ...] = (-20.0, -15.0, -10.0, -5.0, 0.0, 5.0, 10.0, 15.0, 20.0)


@dataclass(frozen=True)
class ArrayConfig:
    """Uniform linear array geometry.

    The paper uses ``n_sensors=16``, half-wavelength spacing, a 1 degree grid
    over ``[-60, 60]`` degrees.
    """

    n_sensors: int = 16
    spacing_ratio: float = 0.5  # element spacing d / wavelength
    angle_min: float = -60.0
    angle_max: float = 60.0
    grid_step_deg: float = 1.0

    def __post_init__(self) -> None:
        if self.n_sensors < 2:
            raise ValueError("n_sensors must be >= 2")
        if not (0.0 < self.spacing_ratio <= 1.0):
            raise ValueError("spacing_ratio must be in (0, 1]")
        if self.angle_max <= self.angle_min:
            raise ValueError("angle_max must be greater than angle_min")
        if self.grid_step_deg <= 0:
            raise ValueError("grid_step_deg must be positive")

    @property
    def n_grid(self) -> int:
        """Number of angle-grid points (inclusive of both endpoints)."""
        return int(round((self.angle_max - self.angle_min) / self.grid_step_deg)) + 1

    def angles(self) -> np.ndarray:
        """Return the search grid in degrees."""
        return np.linspace(self.angle_min, self.angle_max, self.n_grid)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SignalConfig:
    """One simulated snapshot regime.

    Parameters
    ----------
    n_sources:
        Number of far-field narrowband sources.
    snr_db:
        Per-source SNR in dB, with unit noise variance and unit average power.
    n_snapshots:
        Number of snapshots ``T`` used to form the sample covariance.
    noise_var:
        Nominal noise variance (the reference value when non-uniform noise is
        requested).
    coherent_rho:
        If not ``None``, sources are fully coherent with correlation
        coefficient ``rho``; the signal covariance becomes rank deficient.
    nonuniform_noise_db:
        Optional per-sensor noise powers in dB. Must have length
        ``n_sensors`` and is validated in :mod:`covdiff.sim`.
    seed:
        Seed for the local RNG; ``None`` means "use the ambient RNG state".
    """

    n_sources: int = 2
    snr_db: float = 0.0
    n_snapshots: int = 1000
    noise_var: float = 1.0
    coherent_rho: Optional[float] = None
    nonuniform_noise_db: Optional[Tuple[float, ...]] = None
    seed: Optional[int] = None

    def __post_init__(self) -> None:
        if self.n_sources < 1:
            raise ValueError("n_sources must be >= 1")
        if self.n_snapshots < 1:
            raise ValueError("n_snapshots must be >= 1")
        if self.noise_var <= 0:
            raise ValueError("noise_var must be positive")
        if self.coherent_rho is not None and not (-1.0 <= self.coherent_rho <= 1.0):
            raise ValueError("coherent_rho must be in [-1, 1]")

    def with_snr(self, snr_db: float) -> "SignalConfig":
        return replace(self, snr_db=float(snr_db))

    def with_snapshots(self, n_snapshots: int) -> "SignalConfig":
        return replace(self, n_snapshots=int(n_snapshots))

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class DiffusionConfig:
    """Forward/reverse process hyper-parameters.

    ``schedule='exponential'`` reproduces Eq. (14) of the source paper,
    ``beta_t = beta_min + (beta_max - beta_min) * (1 - exp(-upsilon * (t/H)**gamma))``.
    """

    n_steps: int = 1000
    schedule: str = "exponential"  # "linear" | "cosine" | "exponential"
    beta_min: float = 1e-4
    beta_max: float = 0.02
    upsilon: float = 10.0
    gamma: float = 1.5
    cosine_s: float = 0.008
    drop_diagonal: bool = True
    predict: str = "epsilon"  # "epsilon" | "v"

    def __post_init__(self) -> None:
        if self.schedule not in {"linear", "cosine", "exponential"}:
            raise ValueError(f"unknown schedule {self.schedule!r}")
        if self.predict not in {"epsilon", "v"}:
            raise ValueError(f"unknown prediction target {self.predict!r}")
        if self.n_steps < 2:
            raise ValueError("n_steps must be >= 2")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ModelConfig:
    """Architecture of the covariance U-Net."""

    base_channels: int = 32
    channel_mults: Tuple[int, ...] = (1, 2, 4)
    time_emb_dim: int = 128
    dropout: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class TrainConfig:
    """Optimisation schedule for the diffusion prior."""

    epochs: int = 400
    batch_size: int = 32
    lr: float = 1e-3
    weight_decay: float = 0.0
    grad_clip: float = 1.0
    n_train_angle_pairs: int = 7260
    n_snapshots: int = 1000
    snr_grid: Tuple[float, ...] = DEFAULT_SNR_GRID
    val_fraction: float = 0.1
    n_val_samples: int = 512
    seed: int = 0
    device: str = "auto"
    num_workers: int = 0

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ExperimentConfig:
    """Everything needed to reproduce one evaluation table."""

    array: ArrayConfig = ArrayConfig()
    signal: SignalConfig = SignalConfig()
    diffusion: DiffusionConfig = DiffusionConfig()
    model: ModelConfig = ModelConfig()
    train: TrainConfig = TrainConfig()
    n_monte_carlo: int = 50
    seed: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "array": self.array.to_dict(),
            "signal": self.signal.to_dict(),
            "diffusion": self.diffusion.to_dict(),
            "model": self.model.to_dict(),
            "train": self.train.to_dict(),
            "n_monte_carlo": self.n_monte_carlo,
            "seed": self.seed,
        }


def angle_pair_grid(
    array: ArrayConfig, n_sources: int = 2, min_separation_deg: float = 1.0
) -> np.ndarray:
    """Enumerate unordered angle tuples on the array's angle grid.

    For ``n_sources=2`` this reproduces the paper's 7260 training pairs:
    ``C(121, 2) = 7260`` pairs over the 1 degree grid on ``[-60, 60]``.
    """
    from itertools import combinations

    grid = array.angles()
    combos = np.asarray(list(combinations(grid, n_sources)), dtype=float)
    if min_separation_deg <= 0 or n_sources == 1:
        return combos
    gaps = np.diff(combos, axis=1)
    keep = gaps.min(axis=1) >= min_separation_deg - 1e-9
    return combos[keep]
