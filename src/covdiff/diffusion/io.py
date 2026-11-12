"""Checkpointing utilities for the covariance diffusion prior.

Keeps the on-disk format in one place so that ``scripts/train_diffusion.py``
and ``scripts/run_experiment.py`` (and any downstream notebook) agree on how a
trained prior is stored and reloaded.
"""

from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Optional, Sequence

import numpy as np
import torch

from ..config import DEFAULT_SNR_GRID, DiffusionConfig, ModelConfig, TrainConfig
from .model import GaussianDiffusion, denoise_covariance
from .schedules import GaussianSchedule, beta_schedule
from .snr import SNRClassifier, predict_start_timestep
from .unet import CovUNet

__all__ = ["save_checkpoint", "load_checkpoint", "make_ddpm_denoiser", "Checkpoint"]

FORMAT_VERSION = 1


class Checkpoint(dict):
    """Lightweight container returned by :func:`load_checkpoint`.

    Behaves like a dict (``ckpt["model"]``) and exposes the common fields as
    attributes (``ckpt.model``) for convenience.
    """

    def __getattr__(self, item: str) -> Any:  # pragma: no cover - trivial
        try:
            return self[item]
        except KeyError as exc:  # pragma: no cover
            raise AttributeError(item) from exc


def save_checkpoint(
    path: str | Path,
    *,
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    model_config: ModelConfig,
    diffusion_config: DiffusionConfig,
    train_config: Optional[TrainConfig] = None,
    snr_model: Optional[torch.nn.Module] = None,
    snr_grid: Sequence[float] = DEFAULT_SNR_GRID,
    history: Optional[dict[str, list[float]]] = None,
    extra: Optional[dict[str, Any]] = None,
) -> str:
    """Serialise a trained prior (and optional SNR classifier) to ``path``.

    The file is a plain ``torch.save`` payload with a ``format_version`` field so
    that future format changes can be detected instead of silently mis-loaded.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format_version": FORMAT_VERSION,
        "model": model.state_dict(),
        "model_config": asdict(model_config),
        "diffusion_config": asdict(diffusion_config),
        "train_config": asdict(train_config) if train_config is not None else None,
        "snr_model": snr_model.state_dict() if snr_model is not None else None,
        "snr_grid": list(map(float, snr_grid)),
        "history": history or {},
        "extra": extra or {},
    }
    torch.save(payload, path)
    return str(path)


def _build_model(model_config: dict[str, Any]) -> CovUNet:
    return CovUNet(
        in_channels=2,
        base_channels=int(model_config.get("base_channels", 32)),
        channel_mults=tuple(model_config.get("channel_mults", (1, 2, 4))),
        time_emb_dim=int(model_config.get("time_emb_dim", 128)),
        dropout=float(model_config.get("dropout", 0.0)),
    )


def load_checkpoint(path: str | Path, device: str = "cpu") -> Checkpoint:
    """Load a checkpoint written by :func:`save_checkpoint`.

    Returns a :class:`Checkpoint` with keys ``model``, ``diffusion``,
    ``schedule``, ``model_config``, ``diffusion_config``, ``train_config``,
    ``snr_model``, ``snr_grid`` and ``history``.
    """
    payload = torch.load(Path(path), map_location=device, weights_only=False)
    version = int(payload.get("format_version", 0))
    if version != FORMAT_VERSION:
        raise ValueError(
            f"unsupported checkpoint format {version}; expected {FORMAT_VERSION}"
        )

    model_config = ModelConfig(**payload["model_config"])
    diffusion_config = DiffusionConfig(**payload["diffusion_config"])
    train_config = (
        TrainConfig(**payload["train_config"]) if payload.get("train_config") else None
    )

    schedule = GaussianSchedule(beta_schedule(diffusion_config))
    model = _build_model(payload["model_config"])
    model.load_state_dict(payload["model"])
    model.to(device).eval()

    snr_model = None
    snr_state = payload.get("snr_model")
    if snr_state is not None:
        snr_model = SNRClassifier(n_sensors=_infer_n_sensors(snr_state))
        snr_model.load_state_dict(snr_state)
        snr_model.to(device).eval()

    diffusion = GaussianDiffusion(
        schedule,
        predict=diffusion_config.predict,
        drop_diagonal=diffusion_config.drop_diagonal,
    )
    return Checkpoint(
        model=model,
        diffusion=diffusion,
        schedule=schedule,
        model_config=model_config,
        diffusion_config=diffusion_config,
        train_config=train_config,
        snr_model=snr_model,
        snr_grid=payload.get("snr_grid") or list(DEFAULT_SNR_GRID),
        history=payload.get("history", {}),
        extra=payload.get("extra", {}),
    )


def _infer_n_sensors(snr_state: dict[str, torch.Tensor]) -> int:  # pragma: no cover
    """Infer the input width of the SNR classifier from its first linear layer."""
    for key, value in snr_state.items():
        if key.endswith("weight") and value.ndim == 2:
            return int(value.shape[1])
    return 16


def make_ddpm_denoiser(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    *,
    n_steps: int = 50,
    sampler: str = "ddim",
    start_t: Optional[int] = None,
    snr_model: Optional[torch.nn.Module] = None,
    schedule: Optional[GaussianSchedule] = None,
    snr_grid: Sequence[float] = DEFAULT_SNR_GRID,
    device: Any = "cpu",
) -> Callable[[np.ndarray, Any], np.ndarray]:
    """Wrap a trained prior as ``(R_hat, n_snapshots) -> R_denoised``.

    The returned callable matches the classical-estimator protocol used by
    :func:`covdiff.eval.protocol.apply_denoiser`, so ``"music+ddpm"`` and
    ``"ddpm"`` can be mixed with Ledoit-Wolf, OAS, etc. in the same sweep.

    When ``snr_model`` is supplied the reverse chain starts at the timestep that
    matches the predicted SNR (the source paper's acceleration trick). If you
    want a fixed starting point instead, pass ``start_t``; supplying neither is
    an error because the reverse chain would then start from pure noise
    (unconditional generation) and the output would not be a denoising of
    ``R_hat`` at all.
    """
    if snr_model is None and start_t is None:
        raise ValueError(
            "make_ddpm_denoiser needs either snr_model= (SNR-adaptive start) or "
            "start_t= (fixed start); without one the sampler runs the full chain "
            "from pure noise and returns an unconditional sample"
        )

    def denoiser(R_hat: np.ndarray, n_snapshots: Any = None) -> np.ndarray:
        t0 = start_t
        if snr_model is not None:
            if schedule is None:
                raise ValueError("schedule= is required when snr_model= is provided")
            t0 = int(
                predict_start_timestep(
                    snr_model, np.asarray(R_hat), schedule, snr_grid, device=device
                )
            )
        return denoise_covariance(
            model,
            diffusion,
            np.asarray(R_hat, dtype=np.complex128),
            sampler=sampler,
            n_steps=n_steps,
            start_t=t0,
            device=device,
        )

    denoiser.__name__ = "ddpm_denoiser"
    return denoiser
