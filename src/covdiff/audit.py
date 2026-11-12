"""Structural audit of recovered covariance matrices.

The scientific question behind this repository is not only *how accurate* a
recovery method is, but whether the matrix it returns is a legal covariance
matrix at all. :func:`audit_covariance` reports the two structural defects that
matter for a uniform linear array (Hermitian symmetry and Toeplitz structure)
together with the spectral properties that determine whether the matrix is a
valid PSD covariance (and how badly conditioned it is).
"""

from __future__ import annotations

import math
from typing import Dict, Iterable, Mapping

import numpy as np

from .utils import hermitian, toeplitz_project

__all__ = [
    "hermitian_defect",
    "toeplitz_defect",
    "psd_report",
    "audit_covariance",
    "audit_batch",
    "format_audit",
]

#: Relative tolerance below which an eigenvalue is treated as numerical zero.
_EIG_TOL = 1e-12

_READABLE = {
    "hermitian_defect": "Hermitian defect",
    "toeplitz_defect": "Toeplitz defect",
    "min_eig": "Min eigenvalue",
    "max_eig": "Max eigenvalue",
    "neg_eig_fraction": "Negative eigenvalue fraction",
    "neg_energy_ratio": "Negative energy ratio",
    "condition_number": "Condition number",
    "n_matrices": "Matrices audited",
}


def _as_square(matrix: np.ndarray, name: str = "matrix") -> np.ndarray:
    out = np.asarray(matrix, dtype=np.complex128)
    if out.ndim != 2 or out.shape[0] != out.shape[1]:
        raise ValueError(f"{name} must be a square matrix, got shape {out.shape}")
    return out


def _safe_ratio(numerator: float, denominator: float) -> float:
    if not (np.isfinite(numerator) and np.isfinite(denominator)):
        return float("nan")
    if denominator <= 0.0:
        return 0.0 if numerator == 0.0 else float("nan")
    return float(numerator / denominator)


def hermitian_defect(R: np.ndarray) -> float:
    """Relative asymmetry ``||R - R^H||_F / ||R||_F``.

    Zero for a Hermitian matrix, ``O(1)`` for a matrix whose skew-Hermitian
    part is comparable to its norm. Returns ``0.0`` for the zero matrix.
    """
    a = _as_square(R)
    if not np.isfinite(a).all():
        return float("nan")
    return _safe_ratio(
        float(np.linalg.norm(a - a.conj().T, "fro")), float(np.linalg.norm(a, "fro"))
    )


def toeplitz_defect(R: np.ndarray) -> float:
    """Relative distance to the nearest Hermitian Toeplitz matrix.

    ``||R - toeplitz_project(R)||_F / ||R||_F``; zero for an exactly Toeplitz
    matrix.
    """
    a = _as_square(R)
    if not np.isfinite(a).all():
        return float("nan")
    residual = a - toeplitz_project(a)
    return _safe_ratio(
        float(np.linalg.norm(residual, "fro")), float(np.linalg.norm(a, "fro"))
    )


def psd_report(R: np.ndarray) -> Dict[str, float]:
    """Spectral validity report of the Hermitian part of ``R``.

    Returns
    -------
    dict
        ``min_eig``, ``max_eig`` -- extreme eigenvalues of ``hermitian(R)``.
        ``neg_eig_fraction`` -- fraction of eigenvalues below ``-tol``.
        ``neg_energy_ratio`` -- ``||sum_{lambda<0} lambda v v^H||_F / ||R||_F``,
        i.e. how much of the matrix's Frobenius energy lives in the negative
        eigenspace.
        ``condition_number`` -- ``|lambda|_max / |lambda|_min`` (``inf`` when
        singular).

    Notes
    -----
    A non-Hermitian input is symmetrised first, so the spectral numbers are
    always those of a Hermitian matrix; use :func:`hermitian_defect` to see how
    much information that symmetrisation destroyed. Eigenvalues are compared
    against ``-tol`` with ``tol = 1e-12 * max(1, |lambda|_max)`` so that
    round-off noise in an otherwise PSD matrix does not produce a spurious
    negative-eigenvalue fraction.
    """
    a = _as_square(R)
    if not np.isfinite(a).all():
        return {
            "min_eig": float("nan"),
            "max_eig": float("nan"),
            "neg_eig_fraction": float("nan"),
            "neg_energy_ratio": float("nan"),
            "condition_number": float("nan"),
        }

    h = hermitian(a)
    eigvals, eigvecs = np.linalg.eigh(h)
    eigvals = np.real(eigvals)
    norm = float(np.linalg.norm(h, "fro"))

    scale = float(np.max(np.abs(eigvals))) if eigvals.size else 0.0
    tol = _EIG_TOL * max(1.0, scale)
    negative = eigvals < -tol

    if negative.any():
        negative_part = (eigvecs[:, negative] * eigvals[negative]) @ eigvecs[:, negative].conj().T
        neg_energy = float(np.linalg.norm(negative_part, "fro"))
    else:
        neg_energy = 0.0

    abs_vals = np.abs(eigvals)
    smallest = float(np.min(abs_vals)) if abs_vals.size else float("nan")
    condition_number = (
        float("inf") if smallest == 0.0 else float(np.max(abs_vals) / smallest)
    )

    return {
        "min_eig": float(np.min(eigvals)) if eigvals.size else float("nan"),
        "max_eig": float(np.max(eigvals)) if eigvals.size else float("nan"),
        "neg_eig_fraction": float(np.mean(negative)) if eigvals.size else float("nan"),
        "neg_energy_ratio": _safe_ratio(neg_energy, norm),
        "condition_number": condition_number,
    }


def audit_covariance(R: np.ndarray) -> Dict[str, float]:
    """Full structural report for one matrix.

    Combines :func:`hermitian_defect`, :func:`toeplitz_defect` and
    :func:`psd_report`. The spectral part is computed on ``hermitian(R)``, so a
    badly non-Hermitian estimate is still described without raising.
    """
    a = _as_square(R)
    report = {
        "hermitian_defect": hermitian_defect(a),
        "toeplitz_defect": toeplitz_defect(a),
    }
    report.update(psd_report(a))
    return report


def audit_batch(matrices: Iterable[np.ndarray]) -> Dict[str, float]:
    """Audit a stack of matrices and average every metric over the stack.

    ``matrices`` may be a single ``(M, M)`` matrix or a stack of shape
    ``(N, M, M)``. The number of audited matrices is reported as
    ``n_matrices``; all other entries are means of the per-matrix reports
    (non-finite values are ignored).
    """
    arr = np.asarray(matrices, dtype=np.complex128)
    if arr.ndim == 2:
        batch = arr[None, ...]
    elif arr.ndim == 3:
        batch = arr
    else:
        raise ValueError(f"expected (M, M) or (N, M, M), got shape {arr.shape}")

    reports = [audit_covariance(matrix) for matrix in batch]
    keys = list(reports[0]) if reports else []
    summary: Dict[str, float] = {}
    for key in keys:
        values = np.array([rep.get(key, np.nan) for rep in reports], dtype=float)
        finite = values[np.isfinite(values)]
        summary[key] = float(np.mean(finite)) if finite.size else float("nan")
    summary["n_matrices"] = float(len(reports))
    return summary


def _readable(key: str) -> str:
    if key in _READABLE:
        return _READABLE[key]
    return key.replace("_", " ").strip().capitalize()


def _format_value(value: float) -> str:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(v):
        return "nan"
    if math.isinf(v):
        return "inf" if v > 0 else "-inf"
    if v == 0.0:
        return "0"
    magnitude = abs(v)
    if magnitude >= 1e4 or magnitude < 1e-4:
        return f"{v:.4e}"
    return f"{v:.6g}"


def format_audit(report: Mapping[str, float]) -> str:
    """Render an audit report as a two-column markdown table.

    Parameters
    ----------
    report:
        Mapping as returned by :func:`audit_covariance` or
        :func:`audit_batch`. Unknown keys are humanised.

    Returns
    -------
    str
        A markdown table with columns ``Metric`` and ``Value``.
    """
    lines = ["| Metric | Value |", "| --- | --- |"]
    for key, value in report.items():
        lines.append(f"| {_readable(str(key))} | {_format_value(value)} |")
    return "\n".join(lines)
