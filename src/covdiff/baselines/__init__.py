"""Classical baselines: subspace DOA estimators and covariance denoisers.

Two groups are exported.

* :mod:`covdiff.baselines.subspace` — MUSIC and ESPRIT, both
  ``(R, n_sources, array) -> angles``.
* :mod:`covdiff.baselines.shrinkage` — sample-covariance denoisers, all
  ``(R_hat, n_snapshots, **kwargs) -> Hermitian matrix``.

``COVARIANCE_ESTIMATORS`` is the registry consumed by
:mod:`covdiff.eval.protocol`; the ``"oracle"`` estimator is intentionally *not*
registered because it needs the ground truth instead of a sample covariance.
"""

from __future__ import annotations

from .shrinkage import (
    diagonal_loading,
    eigen_clip,
    identity,
    ledoit_wolf,
    oas,
    oracle,
    toeplitz_projection,
)
from .subspace import esprit_estimate, music_estimate, music_spectrum

#: Name -> estimator ``f(R_hat, n_snapshots, **kwargs)``.
COVARIANCE_ESTIMATORS = {
    "raw": identity,
    "lw": ledoit_wolf,
    "oas": oas,
    "diagload": diagonal_loading,
    "toeplitz": toeplitz_projection,
    "eigclip": eigen_clip,
}

__all__ = [
    "COVARIANCE_ESTIMATORS",
    "music_spectrum",
    "music_estimate",
    "esprit_estimate",
    "identity",
    "ledoit_wolf",
    "oas",
    "diagonal_loading",
    "toeplitz_projection",
    "eigen_clip",
    "oracle",
]
