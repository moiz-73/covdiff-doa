"""Linear-algebra helpers shared by the simulation, baselines and model code."""

from __future__ import annotations

from typing import Tuple

import numpy as np

__all__ = [
    "random_cn",
    "hermitian",
    "toeplitz_project",
    "spd_project",
    "diag_load",
    "to_model_input",
    "from_model_output",
    "flatten_offdiag",
    "unflatten_offdiag",
    "normalize_covariance",
    "denormalize_covariance",
    "offdiag_mask",
    "fro_norm",
]


def random_cn(shape: Tuple[int, ...], rng: np.random.Generator) -> np.ndarray:
    """Draw circularly-symmetric complex Gaussian samples of unit variance."""
    return (rng.standard_normal(shape) + 1j * rng.standard_normal(shape)) / np.sqrt(2.0)


def hermitian(matrix: np.ndarray) -> np.ndarray:
    """Return the Hermitian part ``(X + X^H) / 2``."""
    return 0.5 * (matrix + matrix.conj().T)


def toeplitz_project(matrix: np.ndarray) -> np.ndarray:
    """Orthogonal projection of a square matrix onto Hermitian Toeplitz matrices.

    Each element on a given diagonal is replaced by the mean of that diagonal.
    """
    m = matrix.shape[0]
    out = np.zeros_like(matrix, dtype=np.complex128)
    for k in range(m):
        idx = np.arange(m - k)
        mean = matrix[idx, idx + k].mean()
        out[idx, idx + k] = mean
        if k > 0:
            out[idx + k, idx] = np.conj(mean)
    return out


def spd_project(matrix: np.ndarray, eps: float = 1e-10) -> np.ndarray:
    """Project a Hermitian matrix onto the cone of positive-definite matrices."""
    h = hermitian(matrix)
    eigvals, eigvecs = np.linalg.eigh(h)
    eigvals = np.clip(eigvals, eps, None)
    return (eigvecs * eigvals) @ eigvecs.conj().T


def diag_load(matrix: np.ndarray, loading: float) -> np.ndarray:
    """Add ``loading * mean(diag)`` to the diagonal (classic diagonal loading)."""
    m = matrix.shape[0]
    scale = float(np.real(np.trace(matrix)) / m)
    return matrix + loading * scale * np.eye(m, dtype=matrix.dtype)


def offdiag_mask(n_sensors: int) -> np.ndarray:
    """Boolean mask selecting off-diagonal entries (``True`` = keep)."""
    mask = ~np.eye(n_sensors, dtype=bool)
    return mask


def flatten_offdiag(matrix: np.ndarray, drop_diagonal: bool = True) -> np.ndarray:
    """Flatten a complex matrix into the real vector used by the source paper.

    The paper concatenates the real parts and imaginary parts of the
    off-diagonal entries, yielding a vector of length ``2 * M * (M - 1)``
    when ``drop_diagonal=True`` and ``2 * M * M`` otherwise.
    """
    m = matrix.shape[0]
    mask = offdiag_mask(m) if drop_diagonal else np.ones((m, m), dtype=bool)
    return np.concatenate([matrix.real[mask], matrix.imag[mask]])


def unflatten_offdiag(vector: np.ndarray, n_sensors: int, drop_diagonal: bool = True) -> np.ndarray:
    """Inverse of :func:`flatten_offdiag`, returning a Hermitian matrix.

    Because both ``(i, j)`` and ``(j, i)`` are carried in the vector, the two
    estimates are averaged as ``(v_ij + conj(v_ji)) / 2``; the result is
    therefore Hermitian by construction. The imaginary diagonal is discarded.
    """
    m = n_sensors
    mask = offdiag_mask(m) if drop_diagonal else np.ones((m, m), dtype=bool)
    n = mask.sum()
    vector = np.asarray(vector).reshape(-1)
    real = vector[:n].reshape(m, m)
    imag = vector[n : 2 * n].reshape(m, m)
    out = np.zeros((m, m), dtype=np.complex128)
    out[mask] = real[mask] + 1j * imag[mask]
    out = 0.5 * (out + out.conj().T)
    if drop_diagonal:
        np.fill_diagonal(out, 0.0)
    return out


def to_model_input(matrix: np.ndarray, drop_diagonal: bool = True) -> np.ndarray:
    """Convert a complex covariance into a ``(2, M, M)`` real tensor.

    Channel 0 is the real part, channel 1 the imaginary part. When
    ``drop_diagonal`` is true the model never sees the (noise-dominated)
    diagonal, exactly as in the source paper.
    """
    out = np.stack([matrix.real, matrix.imag], axis=0).astype(np.float32)
    if drop_diagonal:
        for c in range(out.shape[0]):
            np.fill_diagonal(out[c], 0.0)
    return out


def from_model_output(tensor: np.ndarray, sym: bool = True) -> np.ndarray:
    """Convert a ``(2, M, M)`` real tensor back into a complex matrix."""
    tensor = np.asarray(tensor)
    if tensor.ndim != 3 or tensor.shape[0] != 2:
        raise ValueError("expected a (2, M, M) tensor")
    matrix = tensor[0] + 1j * tensor[1]
    return hermitian(matrix) if sym else matrix


def normalize_covariance(matrix: np.ndarray) -> Tuple[np.ndarray, float]:
    """Scale a covariance so that its mean diagonal equals one.

    Returns the scaled matrix and the positive scale factor, so that
    ``denormalize_covariance(scaled, scale)`` recovers the input.
    """
    scale = float(np.real(np.trace(matrix)) / matrix.shape[0])
    if scale <= 0:
        scale = 1.0
    return matrix / scale, scale


def denormalize_covariance(matrix: np.ndarray, scale: float) -> np.ndarray:
    """Inverse of :func:`normalize_covariance`."""
    return matrix * scale


def fro_norm(matrix: np.ndarray) -> float:
    """Frobenius norm as a Python float."""
    return float(np.linalg.norm(matrix, "fro"))
