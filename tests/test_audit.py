"""Unit tests for :mod:`covdiff.audit`."""

from __future__ import annotations

import numpy as np
import pytest
from scipy.linalg import toeplitz

from covdiff.audit import (
    audit_batch,
    audit_covariance,
    format_audit,
    hermitian_defect,
    psd_report,
    toeplitz_defect,
)
from covdiff.utils import hermitian


def _spd(m: int = 8, seed: int = 0, scale: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((m, m)) + 1j * rng.standard_normal((m, m))
    return scale * (a @ a.conj().T + m * np.eye(m)) / m


def _hermitian_toeplitz(m: int = 6) -> np.ndarray:
    first = np.array([3.0, 0.8 + 0.2j, 0.3 - 0.1j, 0.1 + 0.0j, 0.05, 0.01], dtype=np.complex128)
    return toeplitz(first[:m], np.conj(first[:m]))


def test_hermitian_defect_flags_a_non_hermitian_matrix() -> None:
    good = _spd(6, seed=1)
    skew = np.triu(0.5 * np.ones((6, 6), dtype=np.complex128), k=1)
    bad = good + skew

    assert hermitian_defect(good) < 1e-12
    assert hermitian_defect(bad) > 0.1
    assert hermitian_defect(np.zeros((3, 3))) == 0.0


def test_toeplitz_defect_separates_toeplitz_from_general_covariances() -> None:
    R = _hermitian_toeplitz()
    assert toeplitz_defect(R) < 1e-12

    bad = R.copy()
    bad[0, 0] += 1.0
    bad[1, 1] += 2.0
    assert toeplitz_defect(bad) > 0.01
    assert toeplitz_defect(_spd(6, seed=2)) > 0.01


def test_psd_report_measures_negative_eigenvalues() -> None:
    R = np.diag([-1.0, 1.0, 2.0, 4.0]).astype(np.complex128)
    report = psd_report(R)

    assert report["min_eig"] == pytest.approx(-1.0)
    assert report["max_eig"] == pytest.approx(4.0)
    assert report["neg_eig_fraction"] == pytest.approx(0.25)
    # ||lambda_min * v v^H||_F = 1, ||R||_F = sqrt(22).
    assert report["neg_energy_ratio"] == pytest.approx(1.0 / np.sqrt(22.0))
    assert report["condition_number"] == pytest.approx(4.0)


def test_audit_catches_non_hermitian_and_indefinite_input() -> None:
    bad = np.array([[1.0, 1.0 + 2.0j], [0.0, -1.0]], dtype=np.complex128)
    report = audit_covariance(bad)

    assert report["hermitian_defect"] > 0.1
    assert report["min_eig"] < 0.0
    assert report["neg_eig_fraction"] > 0.0
    assert report["neg_energy_ratio"] > 0.0
    # Spectral quantities are those of the symmetrised matrix, not of `bad`.
    symmetrised = hermitian(bad)
    assert report["min_eig"] == pytest.approx(np.linalg.eigvalsh(symmetrised).min())


def test_psd_matrix_reports_no_negative_energy() -> None:
    R = _spd(8, seed=3)
    report = audit_covariance(R)

    assert report["hermitian_defect"] < 1e-12
    assert report["neg_eig_fraction"] == 0.0
    assert report["neg_energy_ratio"] == 0.0
    assert report["min_eig"] > 0.0
    assert report["condition_number"] >= 1.0


def test_audit_batch_averages_a_stack() -> None:
    R = _spd(5, seed=4)
    stack = np.stack([R, 2.0 * R, 3.0 * R])
    report = audit_batch(stack)

    expected_min = np.mean([np.linalg.eigvalsh(hermitian(k * R)).min() for k in (1.0, 2.0, 3.0)])
    assert report["n_matrices"] == 3.0
    assert report["min_eig"] == pytest.approx(expected_min)
    assert report["neg_energy_ratio"] == 0.0
    assert audit_batch(R)["n_matrices"] == 1.0


def test_format_audit_returns_a_markdown_table() -> None:
    report = audit_covariance(_spd(5, seed=5))
    markdown = format_audit(report)
    lines = markdown.strip().splitlines()

    assert lines[0] == "| Metric | Value |"
    assert set(lines[1].replace("|", "").split()) == {"---"}
    assert len(lines) == 2 + len(report)
    assert all(line.startswith("|") and line.endswith("|") for line in lines)
    assert "Hermitian defect" in markdown
    assert "Toeplitz defect" in markdown
    assert "Negative energy ratio" in markdown
    assert "Condition number" in markdown

    # Readable names for a literal report, including the degenerate formats.
    literal = format_audit(
        {"hermitian_defect": 0.0, "neg_energy_ratio": float("nan"), "condition_number": np.inf}
    )
    assert "Hermitian defect" in literal
    assert "| nan |" in literal
    assert "| inf |" in literal
    assert "Condition number" in literal
