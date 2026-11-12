"""Tests for :mod:`covdiff.diffusion.schedules` (pure NumPy, no torch needed)."""

from __future__ import annotations

import numpy as np
import pytest

from covdiff.config import DiffusionConfig
from covdiff.diffusion import (
    GaussianSchedule,
    beta_schedule,
    cosine_beta_schedule,
    exponential_beta_schedule,
    linear_beta_schedule,
    timestep_for_snr_db,
)

N_STEPS = 1000


def _schedules() -> dict[str, np.ndarray]:
    return {
        "linear": linear_beta_schedule(N_STEPS),
        "cosine": cosine_beta_schedule(N_STEPS),
        "exponential": exponential_beta_schedule(N_STEPS),
    }


@pytest.mark.parametrize("name", ["linear", "cosine", "exponential"])
def test_schedule_bounds_and_monotonicity(name: str) -> None:
    betas = _schedules()[name]
    assert betas.shape == (N_STEPS,)
    assert np.all(np.isfinite(betas))
    assert np.all(betas > 0.0), f"{name}: betas must be strictly positive"
    assert np.all(betas < 1.0), f"{name}: betas must be below one"
    assert np.all(np.diff(betas) >= -1e-12), f"{name}: betas must be non-decreasing"


@pytest.mark.parametrize("name", ["linear", "cosine", "exponential"])
def test_alpha_bar_decreases_and_vanishes(name: str) -> None:
    schedule = GaussianSchedule(_schedules()[name])
    abar = schedule.alphas_cumprod
    assert abar[0] < 1.0
    assert np.all(np.diff(abar) < 0.0), f"{name}: alpha_bar must decrease strictly"
    assert abar[-1] < 0.01, f"{name}: terminal alpha_bar={abar[-1]:.3e} is too large"
    # Derived quantities agree with their definitions.
    np.testing.assert_allclose(schedule.sqrt_alphas_cumprod, np.sqrt(abar), rtol=1e-12)
    np.testing.assert_allclose(
        schedule.sqrt_one_minus_alphas_cumprod, np.sqrt(1.0 - abar), rtol=1e-12
    )
    np.testing.assert_allclose(
        schedule.snr, abar / (1.0 - abar), rtol=1e-9
    )
    assert np.all(schedule.posterior_variance >= 0.0)


def test_exponential_matches_closed_form() -> None:
    beta_min, beta_max, upsilon, gamma = 1e-4, 0.02, 10.0, 1.5
    betas = exponential_beta_schedule(
        N_STEPS, beta_min=beta_min, beta_max=beta_max, upsilon=upsilon, gamma=gamma
    )
    t = np.arange(1, N_STEPS + 1, dtype=float)
    expected = beta_min + (beta_max - beta_min) * (
        1.0 - np.exp(-upsilon * (t / N_STEPS) ** gamma)
    )
    np.testing.assert_allclose(betas, expected, rtol=1e-12, atol=0.0)

    # Endpoints of Eq. (14).
    np.testing.assert_allclose(
        betas[-1],
        beta_min + (beta_max - beta_min) * (1.0 - np.exp(-upsilon)),
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        betas[0],
        beta_min + (beta_max - beta_min) * (1.0 - np.exp(-upsilon * (1.0 / N_STEPS) ** gamma)),
        rtol=1e-12,
    )
    assert betas[-1] == pytest.approx(beta_max, rel=1e-4)


@pytest.mark.parametrize("cfg_name", ["linear", "cosine", "exponential"])
def test_beta_schedule_from_config(cfg_name: str) -> None:
    cfg = DiffusionConfig(n_steps=N_STEPS, schedule=cfg_name)
    betas = beta_schedule(cfg)
    assert betas.shape == (N_STEPS,)
    assert np.all(betas > 0.0) and np.all(betas < 1.0)
    assert np.all(np.diff(betas) >= -1e-12)
    assert GaussianSchedule(betas).alphas_cumprod[-1] < 0.01


def test_beta_schedule_rejects_unknown_name() -> None:
    cfg = DiffusionConfig(schedule="linear")
    object.__setattr__(cfg, "schedule", "quadratic")
    with pytest.raises(ValueError):
        beta_schedule(cfg)


def test_gaussian_schedule_q_sample_moments() -> None:
    schedule = GaussianSchedule(linear_beta_schedule(200))
    rng = np.random.default_rng(0)
    m = 6
    x0 = np.ones((4000, m, m))
    noise = rng.standard_normal(x0.shape)

    for t in (0, 50, 120, 199):
        x_t = schedule.q_sample(x0, t, noise)
        mean = float(x_t.mean())
        var = float(x_t.var())
        np.testing.assert_allclose(mean, schedule.sqrt_alphas_cumprod[t], atol=0.02)
        np.testing.assert_allclose(
            var, schedule.sqrt_one_minus_alphas_cumprod[t] ** 2, atol=0.02
        )


def test_gaussian_schedule_validates_betas() -> None:
    with pytest.raises(ValueError):
        GaussianSchedule(np.array([0.0, 0.1]))
    with pytest.raises(ValueError):
        GaussianSchedule(np.array([0.1, 1.2]))
    with pytest.raises(ValueError):
        GaussianSchedule(np.array([0.2, 0.1]))


def test_snr_db_is_decreasing_in_time() -> None:
    schedule = GaussianSchedule(exponential_beta_schedule(N_STEPS))
    ts = np.arange(N_STEPS)
    snr_db = schedule.snr_db(ts)
    assert np.all(np.diff(snr_db) < 0.0)
    assert isinstance(schedule.snr_db(0), float)


def test_timestep_for_snr_db_monotone_and_roundtrip() -> None:
    schedule = GaussianSchedule(exponential_beta_schedule(N_STEPS))
    # Higher SNR means less noise, i.e. an earlier (smaller) timestep.
    targets = np.linspace(-60.0, 30.0, 25)
    timesteps = np.array(
        [timestep_for_snr_db(db, schedule.alphas_cumprod) for db in targets]
    )
    assert np.all(np.diff(timesteps) <= 0), "timesteps must be non-increasing in SNR"
    assert timesteps[0] >= timesteps[-1]
    assert set(np.unique(timesteps)).issubset(set(range(N_STEPS)))

    # Round trip: t -> SNR_t -> t within two steps.
    for t in (0, 5, 50, 125, 250, 400, 500, 700, 850, 950, N_STEPS - 1):
        db = schedule.snr_db(t)
        recovered = schedule.timestep_for_snr_db(db)
        assert abs(int(recovered) - int(t)) <= 2, (t, recovered)

    # A request above the maximum SNR clamps to the first timestep, and one
    # below the minimum to the last.
    assert timestep_for_snr_db(1e6, schedule.alphas_cumprod) == 0
    assert timestep_for_snr_db(-1e6, schedule.alphas_cumprod) == N_STEPS - 1
