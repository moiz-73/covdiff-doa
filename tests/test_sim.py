"""Tests for :mod:`covdiff.sim`."""

import numpy as np
import pytest

from covdiff.config import DEFAULT_SNR_GRID, ArrayConfig, SignalConfig, angle_pair_grid
from covdiff.sim import (
    array_covariance,
    generate_dataset,
    make_training_set,
    noise_covariance,
    sample_covariance,
    signal_covariance,
    simulate,
    steering_matrix,
    steering_vector,
)
from covdiff.utils import toeplitz_project

ARRAY = ArrayConfig(n_sensors=4)
FULL = ArrayConfig()
SIG = SignalConfig(n_sources=2, snr_db=0.0, n_snapshots=50)


def test_steering_vector_phase_progression():
    a = steering_vector(30.0, FULL)
    assert a.shape == (16,) and a.dtype == np.complex128
    assert np.allclose(np.abs(a), 1.0)
    # Increment 2*pi*(d/lambda)*sin(theta) = 2*pi*0.5*0.5 = pi/2.
    assert np.isclose(np.angle(a[1] / a[0]), np.pi / 2)
    assert np.allclose(a, np.exp(1j * np.pi / 2 * np.arange(16)))
    assert np.allclose(steering_vector(0.0, FULL), 1.0)
    with pytest.raises(ValueError):
        steering_vector(np.array([1.0, 2.0]), FULL)


def test_steering_matrix_matches_steering_vector():
    angles = np.array([-20.0, 0.0, 35.0])
    A = steering_matrix(angles, ARRAY)
    assert A.shape == (4, 3)
    assert np.allclose(A, np.stack([steering_vector(t, ARRAY) for t in angles], axis=1))


def test_signal_covariance():
    assert np.allclose(signal_covariance(3, 10.0), 10.0 * np.eye(3))
    C = signal_covariance(3, 10.0, coherent_rho=1.0)
    assert np.allclose(C, 10.0 * np.ones((3, 3)))
    assert np.linalg.matrix_rank(C, tol=1e-9) == 1
    lags = np.abs(np.subtract.outer(np.arange(3), np.arange(3)))
    assert np.allclose(signal_covariance(3, 0.0, coherent_rho=0.5), 0.5**lags)
    with pytest.raises(ValueError):
        signal_covariance(2, 0.0, coherent_rho=1.5)


def test_noise_covariance_nonuniform_diagonal():
    nu = (-3.0, 0.0, 4.0, 6.0)
    Q = noise_covariance(ARRAY, nonuniform_noise_db=nu)
    assert np.allclose(Q, np.diag(10.0 ** (np.array(nu) / 10.0)))
    assert np.allclose(Q, np.diag(np.diag(Q)))
    assert np.allclose(noise_covariance(ARRAY, noise_var=2.5), 2.5 * np.eye(4))
    with pytest.raises(ValueError, match="nonuniform_noise_db"):
        noise_covariance(ARRAY, nonuniform_noise_db=(1.0, 2.0))


def test_array_covariance_hermitian_psd():
    R = array_covariance(np.array([-10.0, 12.0]), FULL, SignalConfig())
    assert R.shape == (16, 16) and R.dtype == np.complex128
    assert np.allclose(R, R.conj().T, atol=1e-12)
    assert np.linalg.eigvalsh(R).min() > 0.0
    assert np.linalg.eigvalsh(R - noise_covariance(FULL)).min() > -1e-9
    with pytest.raises(ValueError, match="angles_deg"):
        array_covariance(np.array([1.0]), ARRAY, SIG)


def test_toeplitz_for_uncorrelated_equal_power_sources():
    angles = np.array([-20.0, -5.0, 15.0])
    R = array_covariance(angles, FULL, SignalConfig(n_sources=3, snr_db=5.0))
    assert np.allclose(R, toeplitz_project(R), atol=1e-10)
    Rc = array_covariance(
        angles, FULL, SignalConfig(n_sources=3, snr_db=5.0, coherent_rho=0.9)
    )
    assert not np.allclose(Rc, toeplitz_project(Rc), atol=1e-2)


def test_coherent_rho_one_rank_deficiency():
    R = array_covariance(
        np.array([-15.0, 20.0]),
        FULL,
        SignalConfig(n_sources=2, snr_db=10.0, coherent_rho=1.0),
    )
    eig_signal = np.sort(np.linalg.eigvalsh(R - noise_covariance(FULL)))
    assert eig_signal[-1] > 10.0
    assert np.all(np.abs(eig_signal[:-1]) < 1e-8)
    assert np.allclose(np.sort(np.linalg.eigvalsh(R))[::-1][1:], 1.0, atol=1e-8)


def test_sample_covariance_converges_to_R():
    R = array_covariance(np.array([-5.0, 10.0]), ARRAY, SIG)
    R_hat = sample_covariance(R, 250_000, np.random.default_rng(0))
    assert R_hat.dtype == np.complex128 and np.allclose(R_hat, R_hat.conj().T)
    assert np.linalg.norm(R_hat - R) / np.linalg.norm(R) < 0.05
    single = sample_covariance(R, 1, np.random.default_rng(1))
    assert np.linalg.matrix_rank(single, tol=1e-8) == 1
    with pytest.raises(ValueError, match="n_snapshots"):
        sample_covariance(R, 0, np.random.default_rng(1))


def test_simulate_keys_and_validation():
    out = simulate(
        np.array([-3.0, 4.0]),
        ARRAY,
        SignalConfig(n_snapshots=500, snr_db=3.0),
        rng=np.random.default_rng(2),
    )
    assert set(out) == {
        "R_true", "R_hat", "angles_deg", "snr_db", "n_snapshots", "noise_cov"
    }
    assert out["R_true"].shape == (4, 4) and out["angles_deg"].dtype == np.float64
    assert out["n_snapshots"] == 500 and np.isclose(out["snr_db"], 3.0)
    assert np.linalg.norm(out["R_hat"] - out["R_true"]) < np.linalg.norm(out["R_true"])
    with pytest.raises(ValueError, match="nonuniform_noise_db"):
        simulate(np.array([-3.0, 4.0]), ARRAY, SignalConfig(nonuniform_noise_db=(0.0, 1.0)))


def test_generate_dataset_small_array():
    pairs = np.array([
        [-20.0, 10.0], [0.0, 25.0], [-40.0, 40.0], [12.0, -12.0],
        [5.0, 55.0], [-60.0, 60.0], [7.0, 8.0],
    ])
    signal = SignalConfig(n_sources=2, snr_db=0.0, n_snapshots=40)
    ds = generate_dataset(ARRAY, signal, pairs, rng=np.random.default_rng(3), progress=False)
    n, m = pairs.shape[0], ARRAY.n_sensors
    assert ds["R_true"].shape == (n, m, m) and ds["R_true"].dtype == np.complex64
    assert ds["R_hat"].shape == (n, m, m) and ds["R_hat"].dtype == np.complex64
    assert ds["angles_deg"].shape == (n, 2) and ds["angles_deg"].dtype == np.float32
    assert ds["snr_db"].shape == (n,) and ds["snr_db"].dtype == np.float32
    assert ds["labels"].shape == (n, ARRAY.n_grid) and ds["labels"].dtype == np.float32
    assert ds["n_snapshots"] == 40
    assert np.allclose(ds["labels"].sum(axis=1), 2.0)
    expected = np.rint((pairs - ARRAY.angle_min) / ARRAY.grid_step_deg).astype(int)
    for i in range(n):
        assert np.array_equal(np.flatnonzero(ds["labels"][i]), np.sort(expected[i]))
    reference = array_covariance(pairs[0], ARRAY, signal)
    assert np.allclose(ds["R_true"][0], reference, atol=1e-5)
    ds128 = generate_dataset(
        ARRAY, signal, pairs[:3], dtype=np.complex128,
        rng=np.random.default_rng(4), progress=False,
    )
    assert ds128["R_true"].dtype == np.complex128
    with pytest.raises(ValueError, match="angle_pairs"):
        generate_dataset(ARRAY, SIG, np.zeros((3, 3)), progress=False)
    with pytest.raises(ValueError, match="complex"):
        generate_dataset(ARRAY, SIG, np.zeros((3, 2)), dtype=np.float32, progress=False)


def test_make_training_set_and_angle_pair_grid():
    assert angle_pair_grid(ArrayConfig()).shape == (7260, 2)
    ds = make_training_set(
        ARRAY, SignalConfig(n_sources=2), n_pairs=5, snr_grid=(0.0, 10.0),
        n_snapshots=30, rng=np.random.default_rng(5), progress=False,
    )
    assert ds["R_true"].shape == (10, 4, 4) and ds["R_true"].dtype == np.complex64
    assert ds["angles_deg"].shape == (10, 2)
    assert ds["labels"].shape == (10, ARRAY.n_grid)
    assert np.allclose(ds["snr_db"][:5], 0.0) and np.allclose(ds["snr_db"][5:], 10.0)
    assert ds["n_snapshots"] == 30 and np.allclose(ds["labels"].sum(axis=1), 2.0)
    assert np.allclose(ds["angles_deg"][:5], ds["angles_deg"][5:])
    default = make_training_set(
        ARRAY, SignalConfig(), n_pairs=2, n_snapshots=5,
        rng=np.random.default_rng(6), progress=False,
    )
    assert default["R_true"].shape[0] == 2 * len(DEFAULT_SNR_GRID)
    with pytest.raises(ValueError, match="n_pairs"):
        make_training_set(ARRAY, SignalConfig(), n_pairs=10_000, progress=False)
