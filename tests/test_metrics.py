"""Unit tests for :mod:`covdiff.metrics`."""

from __future__ import annotations

import numpy as np
import pytest

from covdiff.metrics import (
    covariance_metrics,
    log_euclidean_distance,
    nmse_covariance,
    pearson_offdiag,
    rmse_angles,
    subspace_angle_deg,
)


def _spd(m: int = 6, seed: int = 0, scale: float = 1.0) -> np.ndarray:
    """A well-conditioned complex Hermitian positive-definite matrix."""
    rng = np.random.default_rng(seed)
    a = rng.standard_normal((m, m)) + 1j * rng.standard_normal((m, m))
    return scale * (a @ a.conj().T + m * np.eye(m)) / m


def test_rmse_angles_is_permutation_invariant() -> None:
    est = np.array([3.0, 1.0, 2.0])
    true = np.array([2.5, 1.5, 3.5])
    reference = rmse_angles(est, true)
    assert reference == pytest.approx(np.sqrt(np.mean((np.sort(est) - np.sort(true)) ** 2)))
    assert rmse_angles(est[[2, 0, 1]], true[[1, 2, 0]]) == pytest.approx(reference)
    assert rmse_angles(np.sort(est)[::-1], true) == pytest.approx(reference)


def test_rmse_angles_guards_empty_and_non_finite() -> None:
    assert np.isnan(rmse_angles([], [1.0]))
    assert np.isnan(rmse_angles([np.nan, np.inf], [1.0, 2.0]))
    assert rmse_angles([1.0], [1.0, 2.0]) == pytest.approx(0.0)  # truncated to length 1


def test_nmse_of_a_matrix_with_itself_is_zero() -> None:
    R = _spd(seed=1)
    assert nmse_covariance(R, R) == pytest.approx(0.0, abs=1e-14)
    assert nmse_covariance(2.0 * R, R) == pytest.approx(1.0, abs=1e-12)
    assert np.isnan(nmse_covariance(R, np.zeros_like(R)))


def test_log_euclidean_is_symmetric_and_zero_for_identical_spd() -> None:
    a = _spd(5, seed=2)
    b = _spd(5, seed=3, scale=3.0)
    assert log_euclidean_distance(a, a) == pytest.approx(0.0, abs=1e-9)
    assert log_euclidean_distance(a, b) == pytest.approx(log_euclidean_distance(b, a), rel=1e-9)
    assert log_euclidean_distance(a, b) > 0.0
    assert np.isnan(log_euclidean_distance(a, np.full_like(a, np.nan)))


def test_subspace_angle_identical_is_zero_and_orthogonal_is_ninety() -> None:
    R = _spd(6, seed=4)
    assert subspace_angle_deg(R, R, 2) == pytest.approx(0.0, abs=1e-4)

    # Top-2 eigenspaces are span{e1, e2} and span{e3, e4}: orthogonal.
    r1 = np.diag([3.0, 3.0, 2.0, 1.0]).astype(np.complex128)
    r2 = np.diag([2.0, 1.0, 3.0, 3.0]).astype(np.complex128)
    assert subspace_angle_deg(r1, r2, 2) == pytest.approx(90.0, abs=1e-4)
    # Overlapping subspaces give intermediate angles.
    r3 = np.diag([3.0, 2.0, 3.0, 1.0]).astype(np.complex128)
    angle = subspace_angle_deg(r1, r3, 2)
    assert 0.0 < angle < 90.0


def test_subspace_angle_degenerate_inputs() -> None:
    R = _spd(5, seed=5)
    assert np.isnan(subspace_angle_deg(R, R, 0))
    # n_sources larger than M is clamped to the full space, so the angle is 0.
    assert subspace_angle_deg(R, R, 99) == pytest.approx(0.0, abs=1e-4)


def test_pearson_offdiag_is_one_for_proportional_estimates() -> None:
    R = _spd(6, seed=6)
    assert pearson_offdiag(2.5 * R, R) == pytest.approx(1.0, abs=1e-9)
    assert pearson_offdiag(-3.0 * R, R) == pytest.approx(-1.0, abs=1e-9)


def test_pearson_offdiag_guards_zero_variance() -> None:
    identity = np.eye(4, dtype=np.complex128)
    assert np.isnan(pearson_offdiag(identity, identity))  # all off-diagonals are zero
    R = _spd(4, seed=7)
    assert np.isnan(pearson_offdiag(np.full_like(R, np.nan), R))


def test_covariance_metrics_returns_the_documented_keys() -> None:
    R = _spd(8, seed=8)
    est = R + 0.05 * _spd(8, seed=9)
    metrics = covariance_metrics(est, R, n_sources=2)
    assert set(metrics) == {
        "nmse",
        "log_euclidean",
        "subspace_angle_deg",
        "pearson_offdiag",
        "min_eig",
        "cond",
    }
    assert all(np.isfinite(value) for value in metrics.values())
    assert metrics["min_eig"] > 0.0
    assert metrics["cond"] >= 1.0
    assert metrics["nmse"] > 0.0


def test_covariance_metrics_singular_condition_number_is_infinite() -> None:
    metrics = covariance_metrics(np.zeros((4, 4), dtype=np.complex128), np.eye(4), 2)
    assert metrics["cond"] == np.inf
    assert metrics["min_eig"] == pytest.approx(0.0)
