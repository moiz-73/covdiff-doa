"""Classical covariance estimators used as baselines and as denoisers.

Every estimator has the signature ``f(R_hat, n_snapshots, **kwargs)`` and
returns a Hermitian ``(M, M)`` matrix of the same shape. They are registered in
:data:`covdiff.baselines.COVARIANCE_ESTIMATORS` and consumed by the evaluation
protocols as ``denoiser(R_hat, signal)`` adapters.

The sample covariance of ``T`` snapshots ``y_t ~ CN(0, Sigma)`` is
``R_hat = (1/T) sum_t y_t y_t^H``. It is unbiased but full rank even when
``Sigma`` is not, which is precisely the failure mode the structure audit in
:mod:`covdiff.audit` measures.
"""

from __future__ import annotations

import numpy as np

from covdiff.utils import diag_load, hermitian, spd_project, toeplitz_project

__all__ = [
    "identity",
    "ledoit_wolf",
    "oas",
    "diagonal_loading",
    "toeplitz_projection",
    "eigen_clip",
    "oracle",
]

_FLOAT_EPS = np.finfo(np.float64).eps
_SPD_EPS = 1e-10


def _as_covariance(R_hat: np.ndarray) -> np.ndarray:
    """Cast to complex128 and return the Hermitian part."""
    matrix = np.asarray(R_hat, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"expected a square covariance matrix, got shape {matrix.shape}")
    return hermitian(matrix)


def _mean_diagonal(matrix: np.ndarray) -> float:
    """``mu = real(trace(R)) / M``, the scale of the ``mu * I`` target."""
    return float(np.real(np.trace(matrix)) / matrix.shape[0])


def identity(R_hat: np.ndarray, n_snapshots: int, **kwargs) -> np.ndarray:
    """Return the sample covariance unchanged (only symmetrised).

    This is the ``"raw"`` entry of the registry: no shrinkage, no projection.
    """
    return _as_covariance(R_hat)


def ledoit_wolf(R_hat: np.ndarray, n_snapshots: int, **kwargs) -> np.ndarray:
    """Ledoit-Wolf shrinkage towards ``mu * I`` for complex circular Gaussian data.

    Parameters
    ----------
    R_hat:
        Sample covariance ``(M, M)``.
    n_snapshots:
        Number of snapshots ``T`` used to form ``R_hat`` (``T >= 1``).

    Returns
    -------
    np.ndarray
        ``(1 - lam) * R_hat + lam * mu * I``, Hermitian.

    Notes
    -----
    Derivation. Write ``S = R_hat`` and ``mu = tr(Sigma) / M``. The shrinkage
    family is ``Sigma_hat = (1 - lam) S + lam mu I`` and the oracle intensity
    minimising ``E||Sigma_hat - Sigma||_F^2`` is

    .. math:: \\lambda^\\star = \\frac{E\\|S - \\Sigma\\|_F^2}
                                    {\\|\\Sigma - \\mu I\\|_F^2}

    because ``S`` is unbiased. For ``y ~ CN(0, Sigma)`` the fourth moment is
    ``E[(y^H y)^2] = (tr Sigma)^2 + tr(Sigma^2)``, hence
    ``E||y y^H - Sigma||_F^2 = (tr Sigma)^2``. Averaging the ``T`` independent
    snapshots scales the error by ``1/T``:

    .. math:: E\\|S - \\Sigma\\|_F^2 = \\frac{(tr \\Sigma)^2}{T}.

    Replacing ``tr Sigma`` by ``tr S`` and dividing the *per-entry* mean squared
    error by ``M`` gives the practical estimates used here::

        b2 = real(trace(R_hat))**2 / (M * T)        # numerator (noise term)
        d2 = ||R_hat - mu I||_F^2 / M               # denominator (signal term)
        lam = clip(b2 / d2, 0, 1)

    so ``lam -> 0`` for large ``T`` (trust the sample covariance) and
    ``lam -> 1`` when the sample covariance is dominated by estimation noise.
    If ``R_hat`` is already isotropic (``d2 = 0``) the data carry no direction,
    and ``lam = 0`` is returned because the target equals the input.
    """
    matrix = _as_covariance(R_hat)
    m = matrix.shape[0]
    t = max(int(n_snapshots), 1)
    mu = _mean_diagonal(matrix)
    trace = float(np.real(np.trace(matrix)))

    b2 = trace**2 / (m * t)
    d2 = float(np.linalg.norm(matrix - mu * np.eye(m), "fro") ** 2) / m
    if d2 <= _FLOAT_EPS * max(abs(mu) ** 2, 1.0):
        return matrix
    lam = float(np.clip(b2 / d2, 0.0, 1.0))
    shrunk = (1.0 - lam) * matrix + lam * mu * np.eye(m, dtype=np.complex128)
    return hermitian(shrunk)


def _oas_intensity(matrix: np.ndarray, n_snapshots: int) -> float:
    """OAS intensity with ``mu * I`` target, in ``[0, 1]``.

    Chen et al.'s Oracle Approximating Shrinkage replaces the biased
    denominator of Ledoit-Wolf by an unbiased estimate,

    .. math::
        \\lambda = \\frac{(1 - 2/M)\\, tr(R^2) + tr(R)^2}
                        {(T + 1 - 2/M)\\, (tr(R^2) - tr(R)^2 / M)},

    where ``tr(R^2) = ||R||_F^2`` for Hermitian ``R`` and the right factor is
    ``||R - mu I||_F^2``. The complex circular case enters through the same
    ``mu`` target as :func:`ledoit_wolf`; the result is clipped to ``[0, 1]``.
    """
    m = matrix.shape[0]
    t = max(int(n_snapshots), 1)
    trace = float(np.real(np.trace(matrix)))
    frobenius_sq = float(np.linalg.norm(matrix, "fro") ** 2)
    numerator = (1.0 - 2.0 / m) * frobenius_sq + trace**2
    denominator = (t + 1.0 - 2.0 / m) * (frobenius_sq - trace**2 / m)
    floor = _FLOAT_EPS * max(frobenius_sq, 1.0)
    if denominator <= floor:
        return 0.0
    return float(np.clip(numerator / denominator, 0.0, 1.0))


def oas(R_hat: np.ndarray, n_snapshots: int, **kwargs) -> np.ndarray:
    """Oracle Approximating Shrinkage towards ``mu * I``.

    Uses :func:`_oas_intensity` and returns ``(1 - lam) * R_hat + lam * mu * I``
    with ``lam`` clipped to ``[0, 1]``.
    """
    matrix = _as_covariance(R_hat)
    m = matrix.shape[0]
    mu = _mean_diagonal(matrix)
    lam = _oas_intensity(matrix, n_snapshots)
    shrunk = (1.0 - lam) * matrix + lam * mu * np.eye(m, dtype=np.complex128)
    return hermitian(shrunk)


def diagonal_loading(
    R_hat: np.ndarray, n_snapshots: int, loading: float = 0.1, **kwargs
) -> np.ndarray:
    """Classic diagonal loading: ``R + loading * mean(diag(R)) * I``.

    The loading is relative to the mean diagonal (the mean eigenvalue proxy),
    so an increase of ``loading * real(trace(R_hat)) / M`` is added to *every*
    diagonal entry; the off-diagonal structure is untouched. With
    ``loading >= 0`` the result cannot lose positive definiteness and the
    spatial white-noise gain is raised by ``10*log10(1 + loading)`` dB.
    """
    matrix = _as_covariance(R_hat)
    return hermitian(diag_load(matrix, float(loading)))


def toeplitz_projection(R_hat: np.ndarray, n_snapshots: int, **kwargs) -> np.ndarray:
    """Project onto Hermitian Toeplitz matrices, then onto the SPD cone.

    For a uniform linear array with i.i.d. noise the true covariance is
    Toeplitz, so averaging each diagonal removes the finite-sample Toeplitz
    defect. ``spd_project`` then repairs any negative eigenvalue introduced by
    the averaging (with ``eps = 1e-10``, so the result is strictly positive
    definite rather than merely semi-definite).

    Notes
    -----
    The eigenvalue clipping inside :func:`covdiff.utils.spd_project` is not
    itself a Toeplitz operation, so the repaired matrix is re-projected onto
    the Toeplitz subspace and finally made strictly positive definite by adding
    ``c * I``. Adding a multiple of the identity preserves Toeplitz structure,
    hence the returned matrix is simultaneously Hermitian, Toeplitz and SPD as
    the estimator contract requires (the residual is at floating-point level
    rather than the ~1e-3 level left by a bare SPD projection).
    """
    matrix = _as_covariance(R_hat)
    m = matrix.shape[0]
    repaired = toeplitz_project(spd_project(toeplitz_project(matrix)))
    min_eig = float(np.linalg.eigvalsh(repaired).min())
    if min_eig < _SPD_EPS:
        repaired = repaired + (_SPD_EPS - min_eig) * np.eye(m, dtype=np.complex128)
    return hermitian(repaired)


def eigen_clip(R_hat: np.ndarray, n_snapshots: int, noise_var: float = 1.0, **kwargs) -> np.ndarray:
    """Marchenko-Pastur eigenvalue clipping.

    The sample eigenvalues of pure noise spread over
    ``[lam_-, lam_+]`` with the upper edge

    .. math:: \\lambda_+ = \\sigma^2 (1 + \\sqrt{M/T})^2 .

    Eigenvalues below ``lam_+`` are attributed to the noise floor and replaced
    by their common mean (a rank-``K`` "signal plus white noise" model); the
    eigenvalues above the edge are kept unchanged. This is the classical
    low-rank denoiser and is expected to be far too aggressive at low SNR,
    which is exactly what the audit in :mod:`covdiff.audit` is meant to expose.

    Parameters
    ----------
    R_hat:
        Sample covariance ``(M, M)``.
    n_snapshots:
        Number of snapshots ``T``.
    noise_var:
        Assumed noise variance ``sigma^2`` (``1.0`` for unit-variance noise).
    """
    matrix = _as_covariance(R_hat)
    m = matrix.shape[0]
    t = max(int(n_snapshots), 1)
    edge = float(noise_var) * (1.0 + np.sqrt(m / t)) ** 2

    eigvals, eigvecs = np.linalg.eigh(matrix)
    below = eigvals < edge
    if np.any(below):
        eigvals = eigvals.copy()
        eigvals[below] = float(np.mean(eigvals[below]))
    clipped = (eigvecs * eigvals) @ eigvecs.conj().T
    return hermitian(clipped)


def oracle(R_true: np.ndarray, **kwargs) -> np.ndarray:
    """Return the ground-truth covariance (the ``oracle`` upper bound)."""
    matrix = np.asarray(R_true, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"expected a square covariance matrix, got shape {matrix.shape}")
    return hermitian(matrix.copy())
