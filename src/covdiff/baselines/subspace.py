"""Subspace-based direction-of-arrival estimation (MUSIC and ESPRIT).

Both methods operate on a Hermitian covariance matrix ``R`` of shape ``(M, M)``
and a uniform-linear-array geometry from :class:`covdiff.config.ArrayConfig`.
Following the signal model of :mod:`covdiff.sim`,

.. math:: a(\\theta) = [e^{j 2\\pi (d/\\lambda)(m-1)\\sin\\theta}]_{m=1..M},

so the MUSIC null spectrum is evaluated on the same steering convention and the
ESPRIT rotational invariance uses the phase step
``2*pi*spacing_ratio*sin(theta)``.
"""

from __future__ import annotations

import numpy as np

from covdiff.config import ArrayConfig
from covdiff.utils import hermitian

__all__ = ["music_spectrum", "music_estimate", "esprit_estimate"]

_EPS = np.finfo(np.float64).tiny


def _steering_matrix(angles_deg, array: ArrayConfig) -> np.ndarray:
    """Steering matrix ``(n_angles, M)`` for angles in degrees."""
    angles = np.asarray(angles_deg, dtype=float).ravel()
    m = np.arange(array.n_sensors, dtype=float)[None, :]
    phase = 2.0 * np.pi * array.spacing_ratio * m * np.sin(np.deg2rad(angles))[:, None]
    return np.exp(1j * phase)


def _as_covariance(R: np.ndarray, n_sources: int) -> np.ndarray:
    """Validate and return ``R`` as a ``(M, M)`` complex128 Hermitian matrix."""
    matrix = np.asarray(R, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"expected a square covariance matrix, got shape {matrix.shape}")
    m = matrix.shape[0]
    n_sources = int(n_sources)
    if not 1 <= n_sources <= m - 1:
        raise ValueError(f"n_sources must satisfy 1 <= K <= M-1 (M={m}), got {n_sources}")
    matrix = hermitian(matrix)
    if not np.all(np.isfinite(matrix)):
        raise ValueError("covariance matrix contains non-finite entries")
    return matrix


def music_spectrum(
    R: np.ndarray,
    n_sources: int,
    array: ArrayConfig,
    angles_deg: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate the MUSIC pseudo-spectrum.

    Parameters
    ----------
    R:
        Hermitian covariance matrix, shape ``(M, M)``.
    n_sources:
        Assumed number of sources ``K`` (must satisfy ``1 <= K <= M - 1``).
    array:
        Array geometry; supplies ``M``, ``spacing_ratio`` and the default grid.
    angles_deg:
        Optional search grid in degrees. Defaults to ``array.angles()``.

    Returns
    -------
    angles_deg, spectrum:
        The grid (``(G,)``) and the pseudo-spectrum in dB,
        ``10*log10(1 / (a^H En En^H a))``.

    Notes
    -----
    The noise subspace ``En`` spans the ``M - K`` eigenvectors of the smallest
    eigenvalues. ``eigh`` returns eigenvalues in ascending order, so the noise
    block is a leading slice of the eigenvector matrix.
    """
    matrix = _as_covariance(R, n_sources)
    m = matrix.shape[0]
    k = int(n_sources)
    grid = array.angles() if angles_deg is None else np.asarray(angles_deg, dtype=float).ravel()
    if grid.size == 0:
        raise ValueError("angle grid must not be empty")

    _, eigvecs = np.linalg.eigh(matrix)
    en = eigvecs[:, : m - k]
    steering = _steering_matrix(grid, array)
    projected = steering.conj() @ en
    denom = np.sum(projected.real**2 + projected.imag**2, axis=1)
    denom = np.maximum(denom, _EPS)
    spectrum = 10.0 * np.log10(1.0 / denom)
    return grid, spectrum


def _pick_peaks(
    spectrum: np.ndarray,
    angles: np.ndarray,
    k: int,
    min_sep_deg: float = 1.0,
) -> np.ndarray:
    """Greedy top-``k`` peak picking with a minimum angular separation.

    Candidates are visited in decreasing spectral order; a candidate is kept
    when it is at least ``min_sep_deg`` away from every already accepted angle.
    If that pass cannot fill ``k`` slots (a coarse grid), the separation is
    halved repeatedly and, as a last resort, the remaining best candidates are
    repeated so that exactly ``k`` angles are always returned.

    Returns
    -------
    np.ndarray
        ``(k,)`` angles in degrees, sorted ascending.
    """
    spectrum = np.asarray(spectrum, dtype=float).ravel()
    angles = np.asarray(angles, dtype=float).ravel()
    if spectrum.shape != angles.shape:
        raise ValueError("spectrum and angles must have the same length")
    k = int(k)
    if k <= 0:
        return np.empty(0, dtype=float)
    if not np.any(np.isfinite(spectrum)):
        raise ValueError("spectrum contains no finite values")

    order = np.argsort(-np.nan_to_num(spectrum, nan=-np.inf), kind="stable")
    chosen: list[int] = []
    sep0 = max(float(min_sep_deg), 0.0)
    for sep in (sep0, sep0 / 2.0, sep0 / 4.0, sep0 / 8.0, 0.0):
        for idx in order:
            if len(chosen) >= k:
                break
            if idx in chosen:
                continue
            if all(abs(angles[idx] - angles[j]) >= sep - 1e-12 for j in chosen):
                chosen.append(int(idx))
        if len(chosen) >= k:
            break

    selected = np.sort(angles[np.asarray(chosen, dtype=int)]) if chosen else np.empty(0)
    if selected.size < k:
        # Grid too coarse to honour the separation: repeat the best angles.
        selected = np.resize(selected if selected.size else angles[order[:1]], k)
    return np.sort(np.asarray(selected[:k], dtype=float))


def music_estimate(R: np.ndarray, n_sources: int, array: ArrayConfig) -> np.ndarray:
    """MUSIC angle estimates on the array grid.

    Returns ``(n_sources,)`` angles in degrees, sorted ascending. The estimate
    is always exactly ``n_sources`` long.
    """
    grid, spectrum = music_spectrum(R, n_sources, array)
    return _pick_peaks(spectrum, grid, int(n_sources), min_sep_deg=array.grid_step_deg)


def esprit_estimate(R: np.ndarray, n_sources: int, array: ArrayConfig) -> np.ndarray:
    """ESPRIT angle estimates.

    The signal subspace ``Es`` holds the ``K`` dominant eigenvectors. With the
    shift-invariance ``E2 ~ E1 Psi`` solved by least squares,

    .. math:: \\Psi = (E_1^H E_1)^{-1} E_1^H E_2,

    the eigenvalues of ``Psi`` are ``exp(j 2*pi*spacing_ratio*sin(theta))``, so
    ``theta = arcsin(angle(lambda) / (2*pi*spacing_ratio))``. Non-physical
    arguments (``|sin| > 1``) are clipped to the valid range.

    Degenerate inputs (rank-deficient shift matrix, non-finite eigenvalues)
    fall back to :func:`music_estimate` so that a ``(K,)`` array is always
    returned; angles are sorted ascending.
    """
    try:
        matrix = _as_covariance(R, n_sources)
    except ValueError:
        return music_estimate(R, n_sources, array)
    m = matrix.shape[0]
    k = int(n_sources)

    _, eigvecs = np.linalg.eigh(matrix)
    es = eigvecs[:, m - k :]
    e1 = es[:-1, :]
    e2 = es[1:, :]
    if e1.shape[0] < k or np.linalg.matrix_rank(e1) < k:
        return music_estimate(R, n_sources, array)

    psi, *_ = np.linalg.lstsq(e1, e2, rcond=None)
    if not np.all(np.isfinite(psi)):
        return music_estimate(R, n_sources, array)
    with np.errstate(invalid="ignore"):
        roots = np.linalg.eigvals(psi)
        phase = np.angle(roots)
        sin_theta = phase / (2.0 * np.pi * array.spacing_ratio)
        if not np.all(np.isfinite(sin_theta)):
            return music_estimate(R, n_sources, array)
        sin_theta = np.clip(sin_theta.real, -1.0, 1.0)
        angles = np.degrees(np.arcsin(sin_theta))

    angles = np.sort(np.asarray(angles, dtype=float).ravel())
    if angles.size != k:
        return music_estimate(R, n_sources, array)
    return angles
