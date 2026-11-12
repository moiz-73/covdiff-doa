"""Tests for :mod:`covdiff.baselines` (MUSIC, ESPRIT and covariance denoisers).

The fixtures build the ULA covariance directly from the steering model of the
SPEC (``a(theta) = exp(j*2*pi*spacing*(m-1)*sin(theta))``) so that the baselines
are tested without depending on :mod:`covdiff.sim`.
"""

from __future__ import annotations

import numpy as np
import pytest

from covdiff.baselines import (
    COVARIANCE_ESTIMATORS,
    diagonal_loading,
    eigen_clip,
    esprit_estimate,
    ledoit_wolf,
    music_estimate,
    music_spectrum,
    oas,
    oracle,
    toeplitz_projection,
)
from covdiff.baselines.subspace import _pick_peaks
from covdiff.config import ArrayConfig
from covdiff.utils import hermitian, toeplitz_project

ARRAY = ArrayConfig()  # 16 sensors, half-wavelength spacing, 1 degree grid
TRUE_TWO = np.array([-10.0, 10.0])


def _steering(angles_deg, array=ARRAY):
    """``(M, K)`` steering matrix for angles in degrees."""
    angles = np.atleast_1d(np.asarray(angles_deg, dtype=float))
    m = np.arange(array.n_sensors)[:, None]
    return np.exp(1j * 2 * np.pi * array.spacing_ratio * m * np.sin(np.deg2rad(angles))[None, :])


def _simulate(angles_deg, snr_db, n_snapshots, seed=0, noise_var=1.0, array=ARRAY):
    """Return ``(R_true, R_hat)`` for uncorrelated sources at ``snr_db``."""
    a = _steering(angles_deg, array)
    power = np.diag(np.full(a.shape[1], 10.0 ** (snr_db / 10.0)))
    r_true = a @ power @ a.conj().T + noise_var * np.eye(array.n_sensors)
    chol = np.linalg.cholesky(r_true + 1e-12 * np.eye(array.n_sensors))
    rng = np.random.default_rng(seed)
    noise = (rng.standard_normal((array.n_sensors, n_snapshots))
             + 1j * rng.standard_normal((array.n_sensors, n_snapshots))) / np.sqrt(2.0)
    snapshots = chol @ noise
    return r_true, hermitian(snapshots @ snapshots.conj().T / n_snapshots)


def _toeplitz_defect(matrix):
    return float(np.linalg.norm(matrix - toeplitz_project(matrix)) / np.linalg.norm(matrix))


def _angle_rmse(estimated, true):
    return float(np.sqrt(np.mean((np.sort(estimated) - np.sort(true)) ** 2)))


# --------------------------------------------------------------------------- MUSIC


def test_music_resolves_two_sources_at_high_snr():
    errors = []
    for seed in range(5):
        _, r_hat = _simulate(TRUE_TWO, snr_db=20.0, n_snapshots=1000, seed=seed)
        estimated = music_estimate(r_hat, 2, ARRAY)
        assert estimated.shape == (2,)
        errors.append(np.abs(np.sort(estimated) - TRUE_TWO).max())
    assert max(errors) <= 1.0, f"MUSIC error above 1 degree: {errors}"


def test_music_spectrum_is_in_db_on_the_array_grid():
    _, r_hat = _simulate(TRUE_TWO, snr_db=20.0, n_snapshots=1000, seed=1)
    grid, spectrum = music_spectrum(r_hat, 2, ARRAY)
    assert grid.shape == spectrum.shape == (ARRAY.n_grid,)
    assert np.all(np.isfinite(spectrum))
    assert np.allclose(grid, ARRAY.angles())
    assert abs(grid[np.argmax(spectrum)] - TRUE_TWO[1]) <= 1.0


def test_music_degrades_at_low_snr_but_returns_k_angles():
    high, low = [], []
    for seed in range(5):
        _, r_hat = _simulate(TRUE_TWO, snr_db=10.0, n_snapshots=1000, seed=seed)
        high.append(_angle_rmse(music_estimate(r_hat, 2, ARRAY), TRUE_TWO))
        _, r_hat = _simulate(TRUE_TWO, snr_db=-15.0, n_snapshots=200, seed=seed)
        estimate = music_estimate(r_hat, 2, ARRAY)
        assert estimate.shape == (2,)
        assert np.all(np.diff(estimate) > 0), "estimates must be sorted ascending"
        assert np.all(np.isfinite(estimate))
        assert np.all((estimate >= ARRAY.angle_min) & (estimate <= ARRAY.angle_max))
        low.append(_angle_rmse(estimate, TRUE_TWO))
    assert np.mean(low) > np.mean(high)


# -------------------------------------------------------------------------- ESPRIT


def test_esprit_matches_music_at_high_snr():
    for seed in range(5):
        _, r_hat = _simulate(TRUE_TWO, snr_db=20.0, n_snapshots=1000, seed=seed)
        music = music_estimate(r_hat, 2, ARRAY)
        esprit = esprit_estimate(r_hat, 2, ARRAY)
        assert esprit.shape == (2,)
        assert _angle_rmse(esprit, TRUE_TWO) <= 1.0
        assert np.max(np.abs(esprit - music)) <= 1.0


def test_esprit_falls_back_to_music_on_degenerate_input():
    for matrix in (np.zeros((ARRAY.n_sensors, ARRAY.n_sensors), complex), np.eye(ARRAY.n_sensors)):
        estimate = esprit_estimate(matrix, 2, ARRAY)
        assert estimate.shape == (2,)
        assert np.all(np.isfinite(estimate))
        assert np.all((estimate >= ARRAY.angle_min) & (estimate <= ARRAY.angle_max))


# ---------------------------------------------------------------------- peak picking


def test_pick_peaks_enforces_separation_and_always_returns_k():
    angles = np.arange(0.0, 10.0)
    spectrum = np.array([0.0, 5.0, 4.0, 3.0, 2.0, 1.0, 0.5, 0.4, 0.3, 0.2])
    peaks = _pick_peaks(spectrum, angles, 2, min_sep_deg=1.0)
    assert peaks.shape == (2,)
    assert np.min(np.abs(np.diff(np.sort(peaks)))) >= 1.0 - 1e-9
    coarse = _pick_peaks(np.array([1.0, 2.0, 3.0, 4.0]), np.array([0.0, 0.5, 1.0, 1.5]), 6)
    assert coarse.shape == (6,)  # padded, never short
    assert _pick_peaks(spectrum, angles, 0).shape == (0,)


# ----------------------------------------------------------------------- shrinkage


@pytest.mark.parametrize("estimator", [ledoit_wolf, oas], ids=["lw", "oas"])
def test_shrinkage_intensity_in_unit_interval_and_psd_preserved(estimator):
    for n_snapshots in (5, 50, 1000):
        _, r_hat = _simulate(TRUE_TWO, snr_db=5.0, n_snapshots=n_snapshots, seed=2)
        shrunk = estimator(r_hat, n_snapshots)
        mu = np.real(np.trace(r_hat)) / r_hat.shape[0]
        direction = mu * np.eye(r_hat.shape[0]) - r_hat
        numerator = np.real(np.vdot(direction, shrunk - r_hat))
        lam = float(numerator / np.real(np.vdot(direction, direction)))
        assert 0.0 <= lam <= 1.0 + 1e-12
        assert np.allclose(shrunk, r_hat + lam * direction, atol=1e-10)
        assert np.allclose(shrunk, shrunk.conj().T, atol=1e-10)
        assert np.linalg.eigvalsh(shrunk).min() > -1e-10


def test_ledoit_wolf_matches_the_documented_formula():
    _, r_hat = _simulate(TRUE_TWO, snr_db=5.0, n_snapshots=20, seed=3)
    m = r_hat.shape[0]
    mu = float(np.real(np.trace(r_hat)) / m)
    b2 = float(np.real(np.trace(r_hat)) ** 2 / (m * 20))
    d2 = float(np.linalg.norm(r_hat - mu * np.eye(m)) ** 2 / m)
    lam = float(np.clip(b2 / d2, 0.0, 1.0))
    expected = (1.0 - lam) * r_hat + lam * mu * np.eye(m)
    assert np.allclose(ledoit_wolf(r_hat, 20), expected, atol=1e-12)


def test_diagonal_loading_adds_exactly_the_expected_increment():
    _, r_hat = _simulate(TRUE_TWO, snr_db=0.0, n_snapshots=100, seed=4)
    loaded = diagonal_loading(r_hat, 100, loading=0.25)
    expected = 0.25 * np.real(np.trace(r_hat)) / r_hat.shape[0]
    assert np.allclose(np.diag(loaded) - np.diag(r_hat), expected, rtol=1e-12, atol=1e-12)
    off = ~np.eye(r_hat.shape[0], dtype=bool)
    assert np.allclose(loaded[off], r_hat[off], atol=1e-12)
    assert np.linalg.eigvalsh(loaded).min() > np.linalg.eigvalsh(r_hat).min()


def test_toeplitz_projection_is_toeplitz_hermitian_and_spd():
    _, r_hat = _simulate([-25.0, 10.0], snr_db=0.0, n_snapshots=50, seed=5)
    projected = toeplitz_projection(r_hat, 50)
    assert np.allclose(projected, projected.conj().T, atol=1e-12)
    assert _toeplitz_defect(projected) < 1e-10
    assert np.linalg.eigvalsh(projected).min() > 0.0


def test_eigen_clip_does_not_increase_frobenius_distance():
    rng = np.random.default_rng(6)
    raw, clipped = [], []
    for _ in range(10):
        r_true, r_hat = _simulate([-20.0, 20.0], snr_db=0.0, n_snapshots=30,
                                  seed=int(rng.integers(1 << 30)))
        denoised = eigen_clip(r_hat, 30)
        r_raw = np.linalg.norm(r_hat - r_true)
        r_clip = np.linalg.norm(denoised - r_true)
        assert r_clip <= r_raw * (1.0 + 1e-3) + 1e-12  # loose: per-realisation slack
        raw.append(r_raw)
        clipped.append(r_clip)
    assert np.mean(clipped) < np.mean(raw)


# ------------------------------------------------------------------------ registry


def test_registry_has_exactly_the_specified_keys():
    assert set(COVARIANCE_ESTIMATORS) == {"raw", "lw", "oas", "diagload", "toeplitz", "eigclip"}
    assert all(callable(f) for f in COVARIANCE_ESTIMATORS.values())


@pytest.mark.parametrize("name", sorted(COVARIANCE_ESTIMATORS))
def test_every_estimator_returns_the_input_shape_and_hermitian(name):
    _, r_hat = _simulate(TRUE_TWO, snr_db=10.0, n_snapshots=200, seed=7)
    estimate = COVARIANCE_ESTIMATORS[name](r_hat.astype(np.complex64), 200)
    assert estimate.shape == r_hat.shape
    assert np.allclose(estimate, estimate.conj().T, atol=1e-10)
    assert np.all(np.isfinite(estimate))


def test_oracle_returns_the_ground_truth():
    r_true, _ = _simulate(TRUE_TWO, snr_db=10.0, n_snapshots=200, seed=8)
    assert np.allclose(oracle(r_true), r_true)
