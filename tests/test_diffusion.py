"""Tests for the torch-backed diffusion package (skipped when torch is absent)."""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from covdiff.config import DEFAULT_SNR_GRID  # noqa: E402
from covdiff.diffusion import (  # noqa: E402
    CovUNet,
    GaussianDiffusion,
    GaussianSchedule,
    SNRClassifier,
    denoise_covariance,
    exponential_beta_schedule,
    predict_snr_db,
    predict_start_timestep,
    snr_class_index,
    train_snr_classifier,
)


def _schedule(n_steps: int = 200) -> GaussianSchedule:
    return GaussianSchedule(exponential_beta_schedule(n_steps))


def _tiny_unet(**kwargs) -> CovUNet:
    params = dict(in_channels=2, base_channels=8, channel_mults=(1, 2), time_emb_dim=32)
    params.update(kwargs)
    return CovUNet(**params)


def test_unet_shapes_and_parameter_budget() -> None:
    model = CovUNet()
    n_params = sum(p.numel() for p in model.parameters())
    assert n_params < 5_000_000, f"default CovUNet has {n_params} parameters"

    for m in (16, 8):
        x = torch.randn(3, 2, m, m)
        t = torch.arange(3)
        out = model(x, t)
        assert out.shape == (3, 2, m, m)
        assert torch.isfinite(out).all()

    # Time embedding width and the per-sample timestep interface.
    emb = model.time_embed(torch.tensor([0, 1, 2]))
    assert emb.shape == (3, model.time_emb_dim)


def test_unet_rejects_bad_input() -> None:
    model = _tiny_unet()
    with pytest.raises(ValueError):
        model(torch.randn(2, 3, 8, 8), torch.zeros(2, dtype=torch.long))
    with pytest.raises(ValueError):
        model(torch.randn(2, 2, 8), torch.zeros(2, dtype=torch.long))


def test_q_sample_mean_and_variance() -> None:
    diffusion = GaussianDiffusion(_schedule(1000))
    schedule = diffusion.schedule
    torch.manual_seed(0)
    x0 = torch.ones(2000, 2, 8, 8)
    for t in (10, 250, 600, 900):
        noise = torch.randn(x0.shape)
        x_t = diffusion.q_sample(x0, t, noise)
        expected_mean = float(schedule.sqrt_alphas_cumprod[t])
        expected_var = float(1.0 - schedule.alphas_cumprod[t])
        assert abs(float(x_t.mean()) - expected_mean) < 0.02
        assert abs(float(x_t.var()) - expected_var) < 0.02


def test_prediction_components_roundtrip() -> None:
    schedule = _schedule(200)
    x0 = torch.randn(4, 2, 8, 8)
    noise = torch.randn(4, 2, 8, 8)
    t = torch.full((4,), 120, dtype=torch.long)

    eps_diffusion = GaussianDiffusion(schedule, predict="epsilon")
    x_t = eps_diffusion.q_sample(x0, t, noise)
    x0_hat, eps_hat = eps_diffusion._predict_components(noise, x_t, t)
    torch.testing.assert_close(x0_hat, x0, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(eps_hat, noise, atol=1e-6, rtol=1e-6)

    v_diffusion = GaussianDiffusion(schedule, predict="v")
    view = (4, 1, 1, 1)
    sqrt_ab = torch.as_tensor(schedule.sqrt_alphas_cumprod[t.numpy()]).view(view).float()
    sqrt_1mab = (
        torch.as_tensor(schedule.sqrt_one_minus_alphas_cumprod[t.numpy()])
        .view(view)
        .float()
    )
    v = sqrt_ab * noise - sqrt_1mab * x0
    x0_hat, eps_hat = v_diffusion._predict_components(v, x_t, t)
    torch.testing.assert_close(x0_hat, x0, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(eps_hat, noise, atol=1e-5, rtol=1e-5)


def test_training_loss_decreases() -> None:
    torch.manual_seed(0)
    diffusion = GaussianDiffusion(_schedule(200))
    model = _tiny_unet()

    x0 = torch.randn(32, 2, 8, 8)
    t = torch.full((32,), 150, dtype=torch.long)
    generator = torch.Generator().manual_seed(0)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-2)

    losses: list[float] = []
    for _ in range(30):
        loss = diffusion.training_loss(model, x0, t=t, generator=generator)
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        optimiser.step()
        losses.append(float(loss.item()))

    first = float(np.mean(losses[:5]))
    last = float(np.mean(losses[-5:]))
    assert np.isfinite(losses).all()
    assert last < first, f"loss did not decrease: {first:.4f} -> {last:.4f}"

    # Random timesteps and the v-parameterisation are also well defined.
    random_t_loss = diffusion.training_loss(model, x0, generator=generator)
    v_loss = GaussianDiffusion(diffusion.schedule, predict="v").training_loss(
        model, x0, generator=generator
    )
    assert torch.isfinite(random_t_loss) and torch.isfinite(v_loss)


def test_ddim_sample_is_finite_and_deterministic() -> None:
    schedule = _schedule(50)
    diffusion = GaussianDiffusion(schedule)
    model = _tiny_unet()

    torch.manual_seed(0)
    a = diffusion.ddim_sample(model, (2, 2, 8, 8), n_steps=7, eta=0.0)
    torch.manual_seed(0)
    c = diffusion.ddim_sample(model, (2, 2, 8, 8), n_steps=7, eta=0.0)
    assert a.shape == (2, 2, 8, 8)
    assert torch.isfinite(a).all()
    # eta=0 makes the update deterministic given the same initial noise.
    torch.testing.assert_close(a, c)

    # A stochastic (eta=1) run and a late-start run also stay finite.
    torch.manual_seed(0)
    b = diffusion.ddim_sample(model, (2, 2, 8, 8), n_steps=10, eta=1.0, start_t=30)
    assert torch.isfinite(b).all()


def test_snr_class_index_grid() -> None:
    assert snr_class_index(-20.0, DEFAULT_SNR_GRID) == 0
    assert snr_class_index(0.0, DEFAULT_SNR_GRID) == 4
    assert snr_class_index(100.0, DEFAULT_SNR_GRID) == len(DEFAULT_SNR_GRID) - 1


def test_ancestral_sample_starts_late() -> None:
    schedule = _schedule(20)
    diffusion = GaussianDiffusion(schedule)
    model = _tiny_unet()
    torch.manual_seed(0)
    out = diffusion.sample(model, (2, 2, 8, 8), start_t=5)
    assert out.shape == (2, 2, 8, 8)
    assert torch.isfinite(out).all()


def test_denoise_covariance_is_hermitian() -> None:
    rng = np.random.default_rng(0)
    m = 16
    a = rng.standard_normal((m, m)) + 1j * rng.standard_normal((m, m))
    R_hat = (a @ a.conj().T) / m + 0.5 * np.eye(m)

    schedule = _schedule(40)
    diffusion = GaussianDiffusion(schedule)
    model = _tiny_unet()

    for sampler, kwargs in (("ddim", {"n_steps": 8}), ("ddpm", {"start_t": 6})):
        out = denoise_covariance(model, diffusion, R_hat, sampler=sampler, **kwargs)
        assert out.shape == (m, m)
        assert out.dtype == np.complex128
        assert np.isfinite(out).all()
        np.testing.assert_allclose(out, out.conj().T, atol=1e-8)

    # Normalisation can be disabled, and a conditional (start_t) run works.
    raw = denoise_covariance(
        model, diffusion, R_hat, normalize=False, sampler="ddim", n_steps=5
    )
    conditional = denoise_covariance(
        model, diffusion, R_hat, sampler="ddim", n_steps=5, start_t=20
    )
    assert raw.shape == conditional.shape == (m, m)
    assert np.isfinite(raw).all() and np.isfinite(conditional).all()

    # A seeded generator makes the conditional path reproducible.
    first = denoise_covariance(
        model,
        diffusion,
        R_hat,
        sampler="ddim",
        n_steps=5,
        start_t=20,
        generator=torch.Generator().manual_seed(0),
    )
    second = denoise_covariance(
        model,
        diffusion,
        R_hat,
        sampler="ddim",
        n_steps=5,
        start_t=20,
        generator=torch.Generator().manual_seed(0),
    )
    np.testing.assert_allclose(first, second, atol=1e-6)

    with pytest.raises(ValueError):
        denoise_covariance(model, diffusion, R_hat, sampler="nope")


def test_snr_classifier_overfits_tiny_set() -> None:
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    m = 8
    grid = (-10.0, 0.0, 10.0)
    n_samples = 256

    labels = rng.integers(0, len(grid), size=n_samples)
    base = np.linspace(1.0, 0.2, m)
    matrices = []
    for c in labels:
        profile = base ** (1.0 + 0.8 * c)
        matrices.append(np.diag(profile).astype(np.complex128))
    snr_db = np.asarray(grid)[labels]

    model = SNRClassifier(n_sensors=m, n_classes=len(grid), hidden=(64, 128, 64))
    train_snr_classifier(
        model, matrices, snr_db, grid, epochs=60, lr=1e-2, batch_size=32, seed=0
    )

    feats = torch.as_tensor(
        np.stack([np.linalg.eigvalsh(R)[::-1].real for R in matrices]),
        dtype=torch.float32,
    )
    with torch.no_grad():
        preds = model(feats).argmax(dim=1).numpy()
    accuracy = float((preds == labels).mean())
    assert accuracy > 0.8, f"classifier accuracy {accuracy:.3f}"

    sample = matrices[0]
    estimate = predict_snr_db(model, sample, grid)
    assert grid[0] - 1.0 <= estimate <= grid[-1] + 1.0
    start_t = predict_start_timestep(model, sample, _schedule(50), grid)
    assert isinstance(start_t, int)
    assert 0 <= start_t < 50
