"""Simulation of uniform-linear-array (ULA) covariance matrices.

The module implements the signal model of the source paper (Sec. 2.1): ``K``
far-field narrowband sources impinge on an ``M``-element ULA with element
spacing ``d / lambda = array.spacing_ratio``.  Snapshot vectors are

``y_t = A s_t + n_t``, with ``s_t ~ CN(0, P)`` and ``n_t ~ CN(0, Q)``,

so the true covariance is ``R = A P A^H + Q`` and the sample covariance is
``R_hat = (1/T) sum_t y_t y_t^H``.  Because ``R_hat`` is built from the
explicit snapshots rather than from a Wishart draw, it is exactly the
quantity a receiver would form.
"""

from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

from covdiff.config import (
    DEFAULT_SNR_GRID,
    ArrayConfig,
    SignalConfig,
    angle_pair_grid,
)
from covdiff.utils import hermitian, random_cn

__all__ = [
    "steering_vector",
    "steering_matrix",
    "signal_covariance",
    "noise_covariance",
    "array_covariance",
    "sample_covariance",
    "simulate",
    "generate_dataset",
    "make_training_set",
]

#: Diagonal loading used whenever a Cholesky factorisation is attempted.
_JITTER = 1e-12
#: Upper bound on the number of complex entries in one batched snapshot array.
_CHUNK_BUDGET = 2_000_000


# ---------------------------------------------------------------------------
# validation helpers
# ---------------------------------------------------------------------------
def _as_angles(angles_deg: object, name: str = "angles_deg") -> np.ndarray:
    """Return a validated 1-D float array of angles in degrees."""
    angles = np.asarray(angles_deg, dtype=float)
    if angles.ndim != 1:
        raise ValueError(
            f"{name} must be a 1-D array of angles in degrees, got shape {angles.shape}"
        )
    if not np.all(np.isfinite(angles)):
        raise ValueError(f"{name} contains non-finite values")
    return angles


def _snr_powers(snr_db: object, n_sources: int) -> np.ndarray:
    """Convert SNR values in dB to linear per-source powers, shape ``(K,)``."""
    values = np.asarray(snr_db, dtype=float)
    if values.ndim == 0:
        if not np.isfinite(values):
            raise ValueError("snr_db must be finite")
        return np.full(n_sources, 10.0 ** (float(values) / 10.0))
    if values.ndim != 1 or values.size != n_sources:
        raise ValueError(
            f"snr_db must be a scalar or a length-{n_sources} sequence, got shape {values.shape}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("snr_db contains non-finite values")
    return 10.0 ** (values / 10.0)


def _chol_sqrt(matrix: np.ndarray) -> np.ndarray:
    """Return ``L`` with ``L L^H = matrix``, falling back to an eigen square root."""
    m = matrix.shape[0]
    eye = np.eye(m, dtype=matrix.dtype)
    for scale in (1.0, 1e2, 1e4):
        try:
            return np.linalg.cholesky(matrix + scale * _JITTER * eye)
        except np.linalg.LinAlgError:
            continue
    eigvals, eigvecs = np.linalg.eigh(hermitian(matrix))
    return eigvecs * np.sqrt(np.clip(eigvals, 0.0, None))


def _batch_chol(matrix: np.ndarray) -> np.ndarray:
    """Batched :func:`_chol_sqrt` for a stack of Hermitian PSD matrices."""
    eye = np.eye(matrix.shape[-1], dtype=matrix.dtype)
    try:
        return np.linalg.cholesky(matrix + _JITTER * eye)
    except np.linalg.LinAlgError:  # pragma: no cover - numerical safety net
        return np.stack([_chol_sqrt(item) for item in matrix])


def _hermitian_batch(matrices: np.ndarray) -> np.ndarray:
    """Hermitian part of a stack of square matrices."""
    return 0.5 * (matrices + np.swapaxes(matrices.conj(), -1, -2))


def _batch_steering(angles_deg: np.ndarray, array: ArrayConfig) -> np.ndarray:
    """Steering matrices for a ``(C, K)`` block of angle tuples, shape ``(C, M, K)``."""
    omega = 2.0 * np.pi * array.spacing_ratio * np.sin(np.deg2rad(angles_deg))
    sensor = np.arange(array.n_sensors, dtype=float)
    return np.exp(1j * omega[:, None, :] * sensor[None, :, None])


def _progress(iterable: Iterable, total: int, enabled: bool, desc: str):
    """Wrap ``iterable`` in a tqdm bar when requested and available."""
    if not enabled:
        return iterable
    try:  # pragma: no cover - tqdm is an optional nicety at import time
        from tqdm import tqdm
    except ImportError:  # pragma: no cover
        return iterable
    return tqdm(iterable, total=total, desc=desc, leave=False)


# ---------------------------------------------------------------------------
# array geometry
# ---------------------------------------------------------------------------
def steering_vector(angle_deg: float, array: ArrayConfig) -> np.ndarray:
    """Return the ULA steering vector of a single source.

    Parameters
    ----------
    angle_deg:
        Direction of arrival in degrees, measured from broadside.
    array:
        Array geometry; only ``n_sensors`` and ``spacing_ratio`` are used.

    Returns
    -------
    numpy.ndarray
        ``(M,)`` ``complex128`` vector with unit-modulus entries.

    Notes
    -----
    With ``u = sin(theta)`` the vector is

    ``a(theta)[m] = exp(j * 2 * pi * (d / lambda) * (m - 1) * u)``,

    i.e. a linear phase ramp whose increment between adjacent sensors is
    ``2 * pi * (d / lambda) * sin(theta)``.
    """
    if np.ndim(angle_deg) != 0:
        raise ValueError("angle_deg must be a scalar angle in degrees")
    angle = float(angle_deg)
    if not np.isfinite(angle):
        raise ValueError("angle_deg must be finite")
    sensor = np.arange(array.n_sensors, dtype=float)
    phase = 2.0 * np.pi * array.spacing_ratio * sensor * np.sin(np.deg2rad(angle))
    return np.exp(1j * phase).astype(np.complex128)


def steering_matrix(angles_deg: Sequence[float] | np.ndarray, array: ArrayConfig) -> np.ndarray:
    """Stack the steering vectors of ``K`` sources into one matrix.

    Parameters
    ----------
    angles_deg:
        Length-``K`` sequence of directions of arrival in degrees.
    array:
        Array geometry.

    Returns
    -------
    numpy.ndarray
        ``(M, K)`` ``complex128`` matrix whose columns are
        :func:`steering_vector` evaluated at ``angles_deg``.

    Notes
    -----
    ``A = [a(theta_1), ..., a(theta_K)]`` with
    ``A[m, k] = exp(j * 2 * pi * (d / lambda) * (m - 1) * sin(theta_k))``.
    """
    angles = _as_angles(angles_deg)
    omega = 2.0 * np.pi * array.spacing_ratio * np.sin(np.deg2rad(angles))
    sensor = np.arange(array.n_sensors, dtype=float)
    return np.exp(1j * omega[None, :] * sensor[:, None]).astype(np.complex128)


# ---------------------------------------------------------------------------
# second-order statistics
# ---------------------------------------------------------------------------
def signal_covariance(
    n_sources: int,
    snr_db: float,
    coherent_rho: float | None = None,
) -> np.ndarray:
    """Return the source covariance matrix ``P``, shape ``(K, K)``.

    Parameters
    ----------
    n_sources:
        Number of sources ``K``.
    snr_db:
        Per-source SNR in dB with unit noise variance.  Either a scalar
        (equal power for every source) or a length-``K`` sequence.
    coherent_rho:
        If given, sources are mutually coherent with correlation
        coefficient ``rho``; the resulting ``P`` is rank deficient.

    Returns
    -------
    numpy.ndarray
        ``(K, K)`` Hermitian PSD ``complex128`` matrix.

    Notes
    -----
    Uncorrelated sources: ``P = diag(10 ** (snr_db / 10))``.

    Coherent sources: ``P = 10 ** (snr_db / 10) * Toeplitz(rho ** |i - j|)``,
    i.e. ``P[i, j] = p * rho ** |i - j|``.  For ``|rho| = 1`` this is rank
    one, which is the fully coherent (multipath) case of the paper.  A single
    common power is required because the coherence coefficient alone sets the
    relative scaling.
    """
    if int(n_sources) != n_sources or n_sources < 1:
        raise ValueError(f"n_sources must be a positive integer, got {n_sources!r}")
    k = int(n_sources)
    powers = _snr_powers(snr_db, k)

    if coherent_rho is None:
        return np.diag(powers).astype(np.complex128)

    rho = float(coherent_rho)
    if not np.isfinite(rho) or not (-1.0 <= rho <= 1.0):
        raise ValueError(f"coherent_rho must be in [-1, 1], got {coherent_rho!r}")
    if not np.allclose(powers, powers[0]):
        raise ValueError("coherent_rho requires a single common snr_db for all sources")
    lags = np.abs(np.subtract.outer(np.arange(k), np.arange(k)))
    return (powers[0] * rho**lags).astype(np.complex128)


def noise_covariance(
    array: ArrayConfig,
    noise_var: float = 1.0,
    nonuniform_noise_db: tuple[float, ...] | None = None,
) -> np.ndarray:
    """Return the noise covariance matrix ``Q``, shape ``(M, M)``.

    Parameters
    ----------
    array:
        Array geometry.
    noise_var:
        Nominal noise variance, used when ``nonuniform_noise_db`` is ``None``.
    nonuniform_noise_db:
        Optional per-sensor noise powers in dB.  Must contain exactly
        ``array.n_sensors`` entries.

    Returns
    -------
    numpy.ndarray
        ``(M, M)`` diagonal ``complex128`` matrix.

    Notes
    -----
    ``Q = noise_var * I`` for spatially white noise, or
    ``Q = diag(10 ** (nu_db / 10))`` for the non-uniform case.  The noise is
    always assumed spatially uncorrelated, so ``Q`` stays diagonal.
    """
    m = array.n_sensors
    if nonuniform_noise_db is not None:
        values = np.asarray(nonuniform_noise_db, dtype=float).reshape(-1)
        if values.size != m:
            raise ValueError(
                "nonuniform_noise_db must have length n_sensors="
                f"{m}, got {values.size} entries"
            )
        if not np.all(np.isfinite(values)):
            raise ValueError("nonuniform_noise_db contains non-finite values")
        return np.diag(10.0 ** (values / 10.0)).astype(np.complex128)

    variance = float(noise_var)
    if not np.isfinite(variance) or variance <= 0.0:
        raise ValueError(f"noise_var must be positive and finite, got {noise_var!r}")
    return (variance * np.eye(m)).astype(np.complex128)


def array_covariance(
    angles_deg: Sequence[float] | np.ndarray,
    array: ArrayConfig,
    signal: SignalConfig,
) -> np.ndarray:
    """Return the exact array covariance ``R = A P A^H + Q``.

    Parameters
    ----------
    angles_deg:
        Length-``K`` directions of arrival in degrees; must match
        ``signal.n_sources``.
    array:
        Array geometry.
    signal:
        Source / noise configuration (SNR, coherence, noise powers).

    Returns
    -------
    numpy.ndarray
        ``(M, M)`` Hermitian PSD ``complex128`` matrix.

    Notes
    -----
    ``R = sum_{k, l} sqrt(p_k p_l) rho^{|k - l|} a(theta_k) a(theta_l)^H + Q``
    with ``rho = 0`` in the uncorrelated case.  The result is explicitly
    Hermitianised so that downstream eigen-decompositions are well posed.
    """
    angles = _as_angles(angles_deg)
    if angles.size != signal.n_sources:
        raise ValueError(
            "angles_deg must contain signal.n_sources="
            f"{signal.n_sources} angles, got {angles.size}"
        )
    A = steering_matrix(angles, array)
    P = signal_covariance(signal.n_sources, signal.snr_db, signal.coherent_rho)
    Q = noise_covariance(array, signal.noise_var, signal.nonuniform_noise_db)
    return hermitian(A @ P @ A.conj().T + Q)


def sample_covariance(
    R: np.ndarray,
    n_snapshots: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Draw ``T`` snapshots from ``CN(0, R)`` and return their sample covariance.

    Parameters
    ----------
    R:
        True ``(M, M)`` Hermitian PSD covariance matrix.
    n_snapshots:
        Number of snapshots ``T`` (must be a positive integer).
    rng:
        Random generator used for the complex Gaussian draws.

    Returns
    -------
    numpy.ndarray
        ``(M, M)`` Hermitian PSD ``complex128`` matrix ``R_hat``.

    Notes
    -----
    With ``R = L L^H`` (Cholesky, with a small jitter) and
    ``Z ~ CN(0, I)`` of shape ``(M, T)``, the snapshots are ``Y = L Z`` and

    ``R_hat = (1 / T) Y Y^H``.

    Conditional on ``R``, ``T * R_hat`` is complex Wishart distributed, so
    ``E[R_hat] = R`` -- the property exercised by the convergence test.
    """
    matrix = np.asarray(R)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError(f"R must be a square 2-D matrix, got shape {matrix.shape}")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("R contains non-finite values")
    if int(n_snapshots) != n_snapshots or n_snapshots < 1:
        raise ValueError(f"n_snapshots must be a positive integer, got {n_snapshots!r}")

    m = matrix.shape[0]
    t = int(n_snapshots)
    L = _chol_sqrt(hermitian(matrix.astype(np.complex128)))
    snapshots = L @ random_cn((m, t), rng)
    return hermitian((snapshots @ snapshots.conj().T) / t)


def simulate(
    angles_deg: Sequence[float] | np.ndarray,
    array: ArrayConfig,
    signal: SignalConfig,
    rng: np.random.Generator | None = None,
) -> dict:
    """Simulate one true / sample covariance pair.

    Parameters
    ----------
    angles_deg:
        Length-``K`` directions of arrival in degrees.
    array:
        Array geometry.
    signal:
        Source / noise / snapshot configuration.  ``signal.seed`` seeds a
        fresh generator when ``rng`` is not supplied.
    rng:
        Optional random generator.

    Returns
    -------
    dict
        ``"R_true"`` ``(M, M)`` ``complex128``, ``"R_hat"`` ``(M, M)``
        ``complex128``, ``"angles_deg"`` ``(K,)`` ``float64``,
        ``"snr_db"`` ``float``, ``"n_snapshots"`` ``int`` and
        ``"noise_cov"`` ``(M, M)``.

    Notes
    -----
    ``R_true = A P A^H + Q`` and ``R_hat`` is the sample covariance of
    ``signal.n_snapshots`` i.i.d. circular complex Gaussian snapshots drawn
    from ``CN(0, R_true)``.
    """
    angles = _as_angles(angles_deg)
    if rng is None:
        rng = np.random.default_rng(signal.seed)

    snr = np.asarray(signal.snr_db, dtype=float)
    if snr.ndim != 0:
        raise ValueError("simulate requires a scalar signal.snr_db")

    R_true = array_covariance(angles, array, signal)
    R_hat = sample_covariance(R_true, signal.n_snapshots, rng)
    return {
        "R_true": R_true,
        "R_hat": R_hat,
        "angles_deg": angles.copy(),
        "snr_db": float(snr),
        "n_snapshots": int(signal.n_snapshots),
        "noise_cov": noise_covariance(array, signal.noise_var, signal.nonuniform_noise_db),
    }


# ---------------------------------------------------------------------------
# dataset generation
# ---------------------------------------------------------------------------
def _validate_complex_dtype(dtype) -> np.dtype:
    """Return ``dtype`` if it is a complex floating type, else raise."""
    try:
        resolved = np.dtype(dtype)
    except TypeError as exc:  # pragma: no cover - defensive
        raise ValueError(f"dtype must be a numpy complex dtype, got {dtype!r}") from exc
    if not np.issubdtype(resolved, np.complexfloating):
        raise ValueError(
            f"dtype must be a complex floating type (complex64/complex128), got {resolved}"
        )
    return resolved


def _labels_for_pairs(angle_pairs: np.ndarray, array: ArrayConfig) -> np.ndarray:
    """Multi-hot encode angle tuples on the array's 1-degree search grid."""
    n = angle_pairs.shape[0]
    grid_index = np.rint(
        (angle_pairs - array.angle_min) / array.grid_step_deg
    ).astype(np.int64)
    np.clip(grid_index, 0, array.n_grid - 1, out=grid_index)
    labels = np.zeros((n, array.n_grid), dtype=np.float32)
    labels[np.arange(n)[:, None], grid_index] = 1.0
    return labels


def generate_dataset(
    array: ArrayConfig,
    signal: SignalConfig,
    angle_pairs: np.ndarray,
    *,
    dtype=np.complex64,
    rng: np.random.Generator | None = None,
    progress: bool = True,
) -> dict:
    """Generate a batch of ``(R_true, R_hat)`` pairs for many angle tuples.

    Parameters
    ----------
    array:
        Array geometry.
    signal:
        Source / noise / snapshot configuration shared by every pair.
    angle_pairs:
        ``(N, K)`` array of direction-of-arrival tuples in degrees; ``K`` must
        equal ``signal.n_sources``.
    dtype:
        Complex dtype of the returned matrix arrays (default ``complex64``).
    rng:
        Optional random generator; defaults to one seeded by ``signal.seed``.
    progress:
        Show a tqdm progress bar over the pair batches.

    Returns
    -------
    dict
        ``"R_true"`` and ``"R_hat"`` with shape ``(N, M, M)`` and dtype
        ``dtype``; ``"angles_deg"`` ``(N, K)`` ``float32``; ``"labels"``
        ``(N, n_grid)`` ``float32`` multi-hot; ``"snr_db"`` ``(N,)``
        ``float32``; ``"n_snapshots"`` ``int``.

    Notes
    -----
    Each pair contributes ``R = A P A^H + Q`` and the empirical
    ``R_hat = (1/T) sum_t y_t y_t^H``.  Snapshot generation is batched over
    pairs: for a block of ``C`` tuples the code draws ``Z ~ CN(0, I)`` of
    shape ``(C, K, T)``, forms ``S = L_P Z`` and ``Y = A S + L_Q Z'``, and
    computes ``R_hat`` with one batched matrix product.  The block size is
    chosen from a fixed memory budget, so no Python loop over the ``N`` pairs
    is ever executed.
    """
    resolved_dtype = _validate_complex_dtype(dtype)
    pairs = np.asarray(angle_pairs, dtype=float)
    if pairs.ndim != 2:
        raise ValueError(
            f"angle_pairs must be a 2-D (N, n_sources) array, got shape {pairs.shape}"
        )
    if pairs.shape[1] != signal.n_sources:
        raise ValueError(
            "angle_pairs must have signal.n_sources="
            f"{signal.n_sources} columns, got {pairs.shape[1]}"
        )
    if pairs.shape[0] < 1:
        raise ValueError("angle_pairs must contain at least one tuple")
    if not np.all(np.isfinite(pairs)):
        raise ValueError("angle_pairs contains non-finite values")

    rng = np.random.default_rng(signal.seed) if rng is None else rng
    n_pairs, k = pairs.shape
    m = array.n_sensors
    t = int(signal.n_snapshots)

    P = signal_covariance(k, signal.snr_db, signal.coherent_rho)
    Q = noise_covariance(array, signal.noise_var, signal.nonuniform_noise_db)
    q_diag = np.real(np.diag(Q)).astype(float)
    sqrt_q = np.sqrt(q_diag)

    R_true_out = np.empty((n_pairs, m, m), dtype=resolved_dtype)
    R_hat_out = np.empty((n_pairs, m, m), dtype=resolved_dtype)

    chunk = max(1, min(256, _CHUNK_BUDGET // max(1, m * t)))
    starts = range(0, n_pairs, chunk)
    n_chunks = (n_pairs + chunk - 1) // chunk
    for start in _progress(starts, n_chunks, progress, "simulate"):
        stop = min(start + chunk, n_pairs)
        block = pairs[start:stop]
        c = stop - start

        A = _batch_steering(block, array)  # (C, M, K)
        P_block = np.broadcast_to(P, (c, k, k))
        R_true = np.einsum("cmk,ckl,cnl->cmn", A, P_block, A.conj())
        R_true += np.diag(q_diag)[None, :, :]

        # Snapshots: s = L_P z with P = L_P L_P^H, then y = A s + sqrt(Q) n.
        L_P = _batch_chol(np.ascontiguousarray(P_block))
        Z = random_cn((c, k, t), rng)
        sources = np.einsum("ckl,clt->ckt", L_P, Z)
        y = np.einsum("cmk,ckt->cmt", A, sources)
        y += sqrt_q[None, :, None] * random_cn((c, m, t), rng)
        R_hat = np.einsum("cmt,cnt->cmn", y, np.conj(y)) / t

        R_true_out[start:stop] = _hermitian_batch(R_true).astype(resolved_dtype)
        R_hat_out[start:stop] = _hermitian_batch(R_hat).astype(resolved_dtype)

    snr_value = np.asarray(signal.snr_db, dtype=float)
    if snr_value.ndim != 0:
        raise ValueError("generate_dataset requires a scalar signal.snr_db")

    return {
        "R_true": R_true_out,
        "R_hat": R_hat_out,
        "angles_deg": pairs.astype(np.float32),
        "labels": _labels_for_pairs(pairs, array),
        "snr_db": np.full(n_pairs, float(snr_value), dtype=np.float32),
        "n_snapshots": t,
    }


def make_training_set(
    array: ArrayConfig,
    signal: SignalConfig,
    *,
    n_pairs: int = 7260,
    snr_grid: Iterable[float] = DEFAULT_SNR_GRID,
    n_snapshots: int = 1000,
    dtype=np.complex64,
    rng: np.random.Generator | None = None,
    progress: bool = True,
) -> dict:
    """Build the paper's training set over an angle grid and an SNR grid.

    Parameters
    ----------
    array:
        Array geometry (default gives the 121-point grid on ``[-60, 60]``).
    signal:
        Base configuration; ``snr_db`` and ``n_snapshots`` are overridden here.
    n_pairs:
        Number of angle tuples to draw from :func:`covdiff.config.angle_pair_grid`.
        The default ``7260 = C(121, 2)`` reproduces the paper.
    snr_grid:
        SNR levels in dB; the paper uses nine values from -20 to 20 dB.
    n_snapshots:
        Snapshots per sample ``T`` (paper: 1000).
    dtype:
        Complex dtype of ``"R_true"`` / ``"R_hat"`` (default ``complex64``).
    rng:
        Optional random generator shared by pair selection and sampling.
    progress:
        Show progress bars.

    Returns
    -------
    dict
        Same keys as :func:`generate_dataset`, with ``N = n_pairs * len(snr_grid)``
        rows.  ``"snr_db"`` records the SNR level of each row.

    Notes
    -----
    Every angle tuple is simulated once per SNR level, so each sample is
    ``(angles_deg, snr_db) -> (R_true, R_hat)`` and the multi-hot label
    depends only on the tuple.  With ``7260`` pairs and nine SNRs this yields
    ``65,340`` matrices; at ``M = 16`` the two ``complex64`` arrays occupy
    ``2 * 65340 * 16 * 16 * 8`` bytes ~= 268 MB, comfortably below the 1 GB
    budget.  The destination arrays are pre-allocated and filled one SNR
    slice at a time, so no extra full-size copy is produced.
    """
    resolved_dtype = _validate_complex_dtype(dtype)
    snr_values = [float(value) for value in snr_grid]
    if not snr_values:
        raise ValueError("snr_grid must contain at least one SNR level")
    if not np.all(np.isfinite(snr_values)):
        raise ValueError("snr_grid contains non-finite values")
    if int(n_pairs) != n_pairs or n_pairs < 1:
        raise ValueError(f"n_pairs must be a positive integer, got {n_pairs!r}")
    if int(n_snapshots) != n_snapshots or n_snapshots < 1:
        raise ValueError(f"n_snapshots must be a positive integer, got {n_snapshots!r}")

    rng = np.random.default_rng(signal.seed) if rng is None else rng
    available = angle_pair_grid(array, signal.n_sources)
    n_pairs = int(n_pairs)
    if n_pairs > available.shape[0]:
        raise ValueError(
            f"n_pairs={n_pairs} exceeds the {available.shape[0]} available angle "
            f"pairs for n_sources={signal.n_sources}"
        )
    if n_pairs < available.shape[0]:
        selection = rng.permutation(available.shape[0])[:n_pairs]
        pairs = available[selection]
    else:
        pairs = available

    n_snr = len(snr_values)
    total = n_pairs * n_snr
    m = array.n_sensors
    R_true_out = np.empty((total, m, m), dtype=resolved_dtype)
    R_hat_out = np.empty((total, m, m), dtype=resolved_dtype)
    angles_out = np.empty((total, signal.n_sources), dtype=np.float32)
    labels_out = np.empty((total, array.n_grid), dtype=np.float32)
    snr_out = np.empty(total, dtype=np.float32)

    for index in _progress(range(n_snr), n_snr, progress, "snr"):
        snr_db = snr_values[index]
        block_signal = signal.with_snr(snr_db).with_snapshots(int(n_snapshots))
        block = generate_dataset(
            array,
            block_signal,
            pairs,
            dtype=resolved_dtype,
            rng=rng,
            progress=False,
        )
        start = index * n_pairs
        stop = start + n_pairs
        R_true_out[start:stop] = block["R_true"]
        R_hat_out[start:stop] = block["R_hat"]
        angles_out[start:stop] = block["angles_deg"]
        labels_out[start:stop] = block["labels"]
        snr_out[start:stop] = block["snr_db"]

    return {
        "R_true": R_true_out,
        "R_hat": R_hat_out,
        "angles_deg": angles_out,
        "labels": labels_out,
        "snr_db": snr_out,
        "n_snapshots": int(n_snapshots),
    }
