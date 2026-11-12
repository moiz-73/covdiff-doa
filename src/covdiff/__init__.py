"""covdiff — diffusion models at the covariance-matrix level.

The package is deliberately layered so that the *reproducible* parts work
without PyTorch:

``covdiff.sim``        simulation of ULA covariance matrices
``covdiff.baselines``  classical estimators and subspace DOA methods
``covdiff.metrics``    covariance / DOA metrics
``covdiff.audit``      structural validity of a recovered covariance
``covdiff.eval``       experiment protocols and runners
``covdiff.diffusion``  DDPM prior (requires the ``torch`` extra)
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import (  # noqa: F401
    DEFAULT_SNR_GRID,
    ArrayConfig,
    DiffusionConfig,
    ExperimentConfig,
    ModelConfig,
    SignalConfig,
    TrainConfig,
    angle_pair_grid,
)

__all__ = [
    "__version__",
    "DEFAULT_SNR_GRID",
    "ArrayConfig",
    "SignalConfig",
    "DiffusionConfig",
    "ModelConfig",
    "TrainConfig",
    "ExperimentConfig",
    "angle_pair_grid",
]
