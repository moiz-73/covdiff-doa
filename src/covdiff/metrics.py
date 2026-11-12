"""Evaluation metrics for DOA estimates and recovered covariance matrices.

Every function is pure and returns a Python ``float``. Degenerate inputs
(empty arrays, non-finite entries, zero variance, singular matrices) return
``nan`` rather than raising, so that a Monte Carlo sweep can record a failed
draw and continue. The one exception is a shape mismatch between two
covariance matrices, which is a programming error and raises ``ValueError``.
"""

from __future__ import annotations

from typing import Dict

import numpy as np
from scipy.linalg import logm

from .utils import flatten_offdiag, hermitian, spd_project

__all__ = [
    "rmse_angles",
    "nmse_covariance",
    "log_euclidean_distance",
    "subspace_angle_deg",
    "pearson_offdiag",
    "covariance_metrics",
]

_NAN = float("nan")


def _square(matrix: np.ndarray, name: str = "matrix") -> np.ndarray:
    """Return ``matrix`` as a square ``complex128`` array, validating the shape."""
    out = np.asarray(matrix, dtype=np.complex128)
    if out.ndim != 2 or out.shape[0] != out.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {out.shape}")
    return out


def rmse_angles(estimated: np.ndarray, true: np.ndarray) -> float:
    """Root-mean-square error between two sets of angle estimates.

    Both sequences are sorted ascending and truncated to the shorter length
    before the error is formed, so the metric is invariant to the permutation
    of the estimates (DOA estimators have no intrinsic ordering).

    Parameters
    ----------
    estimated, true:
        One-dimensional sequences of angles in degrees. Non-finite entries are
        discarded.

    Returns
    -------
    float
        ``sqrt(mean((est - true) ** 2))`` in degrees, or ``nan`` when either
        input has no finite entry.
    """
    est = np.asarray(estimated, dtype=float).reshape(-1)
    tru = np.asarray(true, dtype=float).reshape(-1)
    est = est[np.isfinite(est)]
    tru = tru[np.isfinite(tru)]
    if est.size == 0 or tru.size == 0:
        return _NAN
    n = int(min(est.size, tru.size))
    est = np.sort(est)[:n]
    tru = np.sort(tru)[:n]
    return float(np.sqrt(np.mean((est - tru) ** 2)))


def nmse_covariance(R_est: np.ndarray, R_true: np.ndarray) -> float:
    """Normalised mean-square error ``||R_est - R_true||_F^2 / ||R_true||_F^2``.

    Returns ``nan`` when the reference has zero (or non-finite) energy or when
    the estimate contains non-finite entries.
    """
    est = _square(R_est, "R_est")
    tru = _square(R_true, "R_true")
    if est.shape != tru.shape:
        raise ValueError(f"shape mismatch: {est.shape} vs {tru.shape}")
    diff = float(np.linalg.norm(est - tru, "fro"))
    denom = float(np.linalg.norm(tru, "fro"))
    if not (np.isfinite(diff) and np.isfinite(denom)) or denom <= 0.0:
        return _NAN
    return float(diff**2 / denom**2)


def log_euclidean_distance(R1: np.ndarray, R2: np.ndarray) -> float:
    """Affine-invariant log-Euclidean distance between two covariances.

    Both matrices are first projected onto the SPD cone, then
    ``||log(R1) - log(R2)||_F`` is returned. The metric is symmetric in its
    arguments and vanishes for identical SPD matrices.
    """
    a = _square(R1, "R1")
    b = _square(R2, "R2")
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch: {a.shape} vs {b.shape}")
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        return _NAN
    a = spd_project(a)
    b = spd_project(b)
    log_a = logm(a)
    log_b = logm(b)
    # SciPy < 1.16 returned ``(logm, errest)`` when ``disp=True``; newer versions
    # return the matrix directly. Accept both shapes.
    if isinstance(log_a, tuple):
        log_a = log_a[0]
    if isinstance(log_b, tuple):
        log_b = log_b[0]
    distance = float(np.linalg.norm(np.asarray(log_a) - np.asarray(log_b), "fro"))
    return distance if np.isfinite(distance) else _NAN


def _top_subspace(matrix: np.ndarray, n_sources: int) -> np.ndarray:
    """Orthonormal basis (``M, L``) of the dominant ``L`` eigenvectors."""
    h = hermitian(matrix)
    vals, vecs = np.linalg.eigh(h)
    order = np.argsort(vals)[::-1][:n_sources]
    return vecs[:, order]


def subspace_angle_deg(R_est: np.ndarray, R_true: np.ndarray, n_sources: int) -> float:
    """Mean principal angle (degrees) between the two signal subspaces.

    The signal subspace is the span of the ``n_sources`` dominant eigenvectors
    of each matrix. Principal angles are obtained from the singular values of
    ``Q_est^H Q_true`` as ``arccos(sigma)``; the **mean** angle over all pairs is
    returned (the maximum angle is the stricter alternative and is not used
    here, to match the paper's averaged reporting).

    Returns ``nan`` when ``n_sources`` is non-positive or either matrix is
    non-finite.
    """
    est = _square(R_est, "R_est")
    tru = _square(R_true, "R_true")
    if est.shape != tru.shape:
        raise ValueError(f"shape mismatch: {est.shape} vs {tru.shape}")
    m = est.shape[0]
    if not np.isfinite(est).all() or not np.isfinite(tru).all():
        return _NAN
    k = int(min(max(int(n_sources), 0), m))
    if k <= 0:
        return _NAN
    q_est = _top_subspace(est, k)
    q_true = _top_subspace(tru, k)
    cosines = np.linalg.svd(q_est.conj().T @ q_true, compute_uv=False)
    cosines = np.clip(np.real(cosines), 0.0, 1.0)
    angles = np.degrees(np.arccos(cosines))
    mean_angle = float(np.mean(angles))
    return mean_angle if np.isfinite(mean_angle) else _NAN


def pearson_offdiag(R_est: np.ndarray, R_true: np.ndarray, drop_diagonal: bool = True) -> float:
    """Pearson correlation of the off-diagonal entries of two matrices.

    The complex matrices are flattened with :func:`covdiff.utils.flatten_offdiag`
    (real parts followed by imaginary parts). A proportional estimate
    ``R_est = c * R_true`` yields ``1.0``.

    Returns ``nan`` if either vector is constant (zero variance), shorter than
    two samples, or contains non-finite values.
    """
    est = np.asarray(R_est, dtype=np.complex128)
    tru = np.asarray(R_true, dtype=np.complex128)
    if est.shape != tru.shape or est.ndim != 2 or est.shape[0] != est.shape[1]:
        raise ValueError(f"expected square matrices of equal shape, got {est.shape} and {tru.shape}")
    v_est = flatten_offdiag(est, drop_diagonal=drop_diagonal).astype(float)
    v_true = flatten_offdiag(tru, drop_diagonal=drop_diagonal).astype(float)
    mask = np.isfinite(v_est) & np.isfinite(v_true)
    v_est = v_est[mask]
    v_true = v_true[mask]
    if v_est.size < 2:
        return _NAN
    if np.std(v_est) <= 1e-15 or np.std(v_true) <= 1e-15:
        return _NAN
    corr = float(np.corrcoef(v_est, v_true)[0, 1])
    return corr if np.isfinite(corr) else _NAN


def covariance_metrics(R_est: np.ndarray, R_true: np.ndarray, n_sources: int) -> Dict[str, float]:
    """Bundle the covariance-recovery metrics used by the evaluation tables.

    Returns
    -------
    dict
        Keys ``nmse``, ``log_euclidean``, ``subspace_angle_deg``,
        ``pearson_offdiag`` (see the individual functions) plus ``min_eig`` and
        ``cond`` of the estimate's Hermitian part. ``cond`` is ``inf`` for a
        singular matrix.
    """
    est = _square(R_est, "R_est")
    tru = _square(R_true, "R_true")
    if est.shape != tru.shape:
        raise ValueError(f"shape mismatch: {est.shape} vs {tru.shape}")

    if np.isfinite(est).all():
        eigvals = np.linalg.eigvalsh(hermitian(est))
        min_eig = float(np.min(eigvals)) if eigvals.size else _NAN
        abs_vals = np.abs(eigvals)
        smallest = float(np.min(abs_vals)) if abs_vals.size else 0.0
        cond = float("inf") if smallest == 0.0 else float(np.max(abs_vals) / smallest)
    else:
        min_eig = _NAN
        cond = _NAN

    return {
        "nmse": nmse_covariance(est, tru),
        "log_euclidean": log_euclidean_distance(est, tru),
        "subspace_angle_deg": subspace_angle_deg(est, tru, n_sources),
        "pearson_offdiag": pearson_offdiag(est, tru),
        "min_eig": min_eig,
        "cond": cond,
    }
