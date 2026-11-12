"""Cross-module tests: the pieces have to work together, not just alone."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from covdiff.audit import audit_covariance
from covdiff.baselines import COVARIANCE_ESTIMATORS
from covdiff.baselines.subspace import music_estimate
from covdiff.config import ArrayConfig, SignalConfig
from covdiff.eval import protocol
from covdiff.metrics import rmse_angles
from covdiff.sim import simulate

ROOT = Path(__file__).resolve().parents[1]


def test_import_does_not_pull_torch():
    code = "import sys, covdiff; print('torch' in sys.modules)"
    out = subprocess.check_output([sys.executable, "-c", code], cwd=ROOT, text=True).strip()
    assert out == "False"


def test_simulate_denoise_estimate_audit():
    array = ArrayConfig(n_sensors=8, grid_step_deg=0.5)
    signal = SignalConfig(n_sources=2, snr_db=0.0, n_snapshots=200)
    truth = np.array([-10.0, 10.0])

    sim = simulate(truth, array, signal, rng=np.random.default_rng(0))
    R_hat = sim["R_hat"]
    R_true = sim["R_true"]

    report = audit_covariance(R_hat)
    assert report["hermitian_defect"] < 1e-12
    assert report["neg_eig_fraction"] == 0.0

    cleaned = COVARIANCE_ESTIMATORS["toeplitz"](R_hat, 200)
    assert cleaned.shape == R_hat.shape
    assert np.allclose(cleaned, cleaned.conj().T, atol=1e-10)

    estimate = np.sort(music_estimate(cleaned, 2, array))
    assert np.all(np.isfinite(estimate))
    assert rmse_angles(estimate, truth) < 5.0


def test_protocol_runs_with_two_methods():
    array = ArrayConfig(n_sensors=8, grid_step_deg=1.0)
    signal = SignalConfig(n_sources=2, snr_db=-5.0, n_snapshots=400)
    estimators = protocol.make_angle_estimators(["music", "esprit", "music+lw"])

    frame = protocol.run_rmse_vs_snr(
        array, signal, estimators, snr_grid=(-5.0, 5.0), n_mc=2, seed=1, progress=False
    )

    assert set(frame["method"]) == {"music", "esprit", "music+lw"}
    assert list(frame.columns[:3]) == ["method", "snr_db", "rmse_mean"]
    assert np.all(np.isfinite(frame["rmse_mean"]))


def test_structure_audit_reacts_to_a_broken_matrix():
    rng = np.random.default_rng(0)
    a = (rng.standard_normal((6, 6)) + 1j * rng.standard_normal((6, 6))) / np.sqrt(2)
    broken = a @ a.conj().T
    broken[0, 1] += 0.5  # non-Hermitian perturbation
    report = audit_covariance(broken)
    assert report["hermitian_defect"] > 0
    eigvals = np.linalg.eigvalsh(0.5 * (broken + broken.conj().T))
    assert report["min_eig"] <= eigvals.max()


def test_checkpoint_roundtrip(tmp_path):
    torch = pytest.importorskip("torch")

    from covdiff.config import DiffusionConfig, ModelConfig
    from covdiff.diffusion.io import load_checkpoint, make_ddpm_denoiser, save_checkpoint
    from covdiff.diffusion.model import GaussianDiffusion
    from covdiff.diffusion.schedules import GaussianSchedule, beta_schedule
    from covdiff.diffusion.unet import CovUNet

    cfg = DiffusionConfig(n_steps=50)
    model_cfg = ModelConfig(base_channels=8, channel_mults=(1, 2), time_emb_dim=32)
    model = CovUNet(in_channels=2, base_channels=8, channel_mults=(1, 2), time_emb_dim=32)
    diffusion = GaussianDiffusion(GaussianSchedule(beta_schedule(cfg)))

    path = save_checkpoint(
        tmp_path / "tiny.pt",
        model=model,
        diffusion=diffusion,
        model_config=model_cfg,
        diffusion_config=cfg,
    )
    ckpt = load_checkpoint(path)

    assert ckpt.diffusion_config.n_steps == 50
    denoiser = make_ddpm_denoiser(ckpt.model, ckpt.diffusion, n_steps=2, start_t=10)
    R = np.eye(8, dtype=np.complex128) + 0.1 * np.ones((8, 8))
    out = denoiser(R, 100)
    assert out.shape == R.shape
    assert np.allclose(out, out.conj().T, atol=1e-6)
    assert np.all(np.isfinite(out))


def test_ddpm_denoiser_uses_snr_classifier():
    """Regression: the SNR-adaptive start must call the classifier correctly."""
    pytest.importorskip("torch")

    from covdiff.config import DiffusionConfig, ModelConfig
    from covdiff.diffusion.io import make_ddpm_denoiser
    from covdiff.diffusion.model import GaussianDiffusion
    from covdiff.diffusion.schedules import GaussianSchedule, beta_schedule
    from covdiff.diffusion.snr import SNRClassifier
    from covdiff.diffusion.unet import CovUNet

    cfg = DiffusionConfig(n_steps=50)
    model = CovUNet(in_channels=2, base_channels=8, channel_mults=(1, 2), time_emb_dim=32)
    schedule = GaussianSchedule(beta_schedule(cfg))
    diffusion = GaussianDiffusion(schedule)
    snr_model = SNRClassifier(n_sensors=8, n_classes=3)

    denoiser = make_ddpm_denoiser(
        model,
        diffusion,
        n_steps=3,
        snr_model=snr_model,
        schedule=schedule,
        snr_grid=(-10.0, 0.0, 10.0),
    )
    R = np.eye(8, dtype=np.complex128) + 0.05 * np.ones((8, 8))
    out = denoiser(R, 100)
    assert out.shape == R.shape
    assert np.allclose(out, out.conj().T, atol=1e-6)

    with pytest.raises(ValueError):
        make_ddpm_denoiser(model, diffusion)
