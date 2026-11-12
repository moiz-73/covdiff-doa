"""Gaussian diffusion process and covariance denoising wrapper.

The forward process uses the (exponential) schedule of :mod:`covdiff.diffusion.schedules`
and the reverse process is either the ancestral DDPM sampler or a deterministic
DDIM sampler. Both operate on the real ``(2, M, M)`` tensor representation of a
complex covariance matrix produced by :func:`covdiff.utils.to_model_input`.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import torch
from torch import Tensor

from covdiff.diffusion.schedules import GaussianSchedule
from covdiff.utils import (
    denormalize_covariance,
    from_model_output,
    hermitian,
    normalize_covariance,
    to_model_input,
)

__all__ = ["GaussianDiffusion", "denoise_covariance"]


def _as_timesteps(t: Any, batch: int, device: torch.device) -> Tensor:
    """Normalise ``t`` (int, sequence or tensor) to a ``(batch,)`` long tensor."""
    if isinstance(t, Tensor):
        out = t.reshape(-1).to(device=device, dtype=torch.long)
        if out.numel() == 1 and batch != 1:
            out = out.expand(batch)
        return out
    if isinstance(t, (int, np.integer)):
        return torch.full((batch,), int(t), device=device, dtype=torch.long)
    arr = torch.as_tensor(np.asarray(t, dtype=np.int64).reshape(-1))
    arr = arr.to(device=device, dtype=torch.long)
    if arr.numel() == 1 and batch != 1:
        arr = arr.expand(batch)
    return arr


class GaussianDiffusion:
    """DDPM forward/reverse process for covariance tensors.

    Parameters
    ----------
    schedule:
        Pre-computed :class:`~covdiff.diffusion.schedules.GaussianSchedule`.
    predict:
        ``"epsilon"`` to predict the injected noise or ``"v"`` to predict the
        velocity ``v = sqrt(alpha_bar) * eps - sqrt(1 - alpha_bar) * x0``.
    drop_diagonal:
        When ``True`` the (noise-dominated) diagonal entries are excluded from
        the training loss, matching the paper's ``to_model_input`` convention.
    """

    def __init__(
        self,
        schedule: GaussianSchedule,
        predict: str = "epsilon",
        drop_diagonal: bool = True,
    ):
        if predict not in {"epsilon", "v"}:
            raise ValueError(f"unknown prediction target {predict!r}")
        self.schedule = schedule
        self.predict = predict
        self.drop_diagonal = bool(drop_diagonal)
        self.n_steps = int(schedule.n_steps)
        #: Value used by ``clip_denoised`` in the samplers (paper-scale covariances).
        self.clip_value = 1.0
        self._coef_cache: dict[tuple[str, str, str], Tensor] = {}

    # ---------------------------------------------------------------- utilities
    def _coef(self, name: str, device: torch.device, dtype: torch.dtype) -> Tensor:
        """Cached torch view of one array of the schedule."""
        key = (name, str(device), str(dtype))
        tensor = self._coef_cache.get(key)
        if tensor is None:
            values = np.asarray(getattr(self.schedule, name), dtype=np.float64)
            tensor = torch.as_tensor(values, device=device, dtype=dtype)
            self._coef_cache[key] = tensor
        return tensor

    @staticmethod
    def _view(x: Tensor) -> tuple[int, ...]:
        return (-1,) + (1,) * (x.dim() - 1)

    def _loss_mask(self, x: Tensor) -> Tensor:
        """Mask that zeroes both diagonals when ``drop_diagonal`` is set."""
        if not self.drop_diagonal:
            return torch.ones_like(x)
        m = x.shape[-1]
        mask = torch.ones((1, x.shape[1], m, m), dtype=x.dtype, device=x.device)
        idx = torch.arange(m, device=x.device)
        mask[:, :, idx, idx] = 0.0
        return mask

    def _predict_components(self, pred: Tensor, x_t: Tensor, t: Tensor) -> tuple[Tensor, Tensor]:
        """Split a network prediction into ``(x0, epsilon)`` estimates."""
        view = self._view(x_t)
        sqrt_ab = self._coef("sqrt_alphas_cumprod", x_t.device, x_t.dtype)[t].view(view)
        sqrt_1mab = self._coef(
            "sqrt_one_minus_alphas_cumprod", x_t.device, x_t.dtype
        )[t].view(view)
        if self.predict == "epsilon":
            eps = pred
            x0 = (x_t - sqrt_1mab * eps) / sqrt_ab.clamp(min=1e-8)
        else:
            x0 = sqrt_ab * x_t - sqrt_1mab * pred
            eps = sqrt_ab * pred + sqrt_1mab * x_t
        return x0, eps

    # ------------------------------------------------------------------ forward
    def q_sample(
        self,
        x0: Tensor,
        t: Any,
        noise: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Sample ``x_t ~ q(x_t | x_0)``.

        ``x_t = sqrt(alpha_bar_t) * x0 + sqrt(1 - alpha_bar_t) * noise``.
        """
        t = _as_timesteps(t, int(x0.shape[0]), x0.device)
        if noise is None:
            noise = torch.randn(
                x0.shape, generator=generator, device=x0.device, dtype=x0.dtype
            )
        view = self._view(x0)
        sqrt_ab = self._coef("sqrt_alphas_cumprod", x0.device, x0.dtype)[t].view(view)
        sqrt_1mab = self._coef(
            "sqrt_one_minus_alphas_cumprod", x0.device, x0.dtype
        )[t].view(view)
        return sqrt_ab * x0 + sqrt_1mab * noise

    # ------------------------------------------------------------------- losses
    def training_loss(
        self,
        model: torch.nn.Module,
        x0: Tensor,
        t: Optional[Any] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """Simple denoising MSE loss.

        Parameters
        ----------
        model:
            Noise-prediction network, called as ``model(x_t, t)``.
        x0:
            Clean training batch, ``(B, 2, M, M)``.
        t:
            Optional timesteps; sampled uniformly when omitted.
        generator:
            Optional ``torch.Generator`` controlling timesteps and noise.

        Returns
        -------
        torch.Tensor
            Scalar loss (mean over all non-masked entries).
        """
        batch = int(x0.shape[0])
        if t is None:
            t = torch.randint(
                0, self.n_steps, (batch,), device=x0.device, generator=generator
            )
        t = _as_timesteps(t, batch, x0.device)
        noise = torch.randn(
            x0.shape, generator=generator, device=x0.device, dtype=x0.dtype
        )
        x_t = self.q_sample(x0, t, noise=noise)
        pred = model(x_t, t)

        if self.predict == "epsilon":
            target = noise
        else:
            view = self._view(x0)
            sqrt_ab = self._coef("sqrt_alphas_cumprod", x0.device, x0.dtype)[t].view(view)
            sqrt_1mab = self._coef(
                "sqrt_one_minus_alphas_cumprod", x0.device, x0.dtype
            )[t].view(view)
            target = sqrt_ab * noise - sqrt_1mab * x0

        loss = (pred - target) ** 2
        mask = self._loss_mask(x0)
        if self.drop_diagonal:
            denom = mask.sum() * batch
            return (loss * mask).sum() / denom
        return loss.mean()

    # ----------------------------------------------------------------- samplers
    @torch.no_grad()
    def sample(
        self,
        model: torch.nn.Module,
        shape: tuple[int, ...],
        device: Any = "cpu",
        start_t: Optional[int] = None,
        generator: Optional[torch.Generator] = None,
        clip_denoised: bool = True,
        x_init: Optional[Tensor] = None,
    ) -> Tensor:
        """Ancestral (DDPM) sampling.

        Parameters
        ----------
        model:
            Noise-prediction network.
        shape:
            Output shape, e.g. ``(n, 2, M, M)``.
        device:
            Torch device.
        start_t:
            Start the reverse chain at this timestep instead of ``H - 1`` (the
            paper's SNR-prediction trick: begin where the forward noise level
            matches the estimated SNR).
        generator:
            Optional RNG for reproducibility.
        clip_denoised:
            Clip the ``x0`` estimate to ``[-clip_value, clip_value]`` each step.
        x_init:
            Optional initial state; when omitted the chain starts from
            ``N(0, I)``. Pass a partially noised ``x0`` to condition the sample.

        Returns
        -------
        torch.Tensor
            ``shape``-shaped tensor.
        """
        dev = torch.device(device)
        t_start = self.n_steps - 1 if start_t is None else int(start_t)
        t_start = max(0, min(t_start, self.n_steps - 1))

        if x_init is None:
            x = torch.randn(
                shape, generator=generator, device=dev, dtype=torch.float32
            )
        else:
            x = x_init.to(device=dev)
            if tuple(x.shape) != tuple(shape):
                raise ValueError(f"x_init shape {tuple(x.shape)} != {tuple(shape)}")

        batch = int(shape[0])
        view = self._view(x)
        for step in range(t_start, -1, -1):
            t = torch.full((batch,), step, device=dev, dtype=torch.long)
            pred = model(x, t)
            x0, _ = self._predict_components(pred, x, t)
            if clip_denoised:
                x0 = x0.clamp(-self.clip_value, self.clip_value)

            coef1 = self._coef(
                "posterior_mean_coef1", dev, x.dtype
            )[t].view(view)
            coef2 = self._coef(
                "posterior_mean_coef2", dev, x.dtype
            )[t].view(view)
            mean = coef1 * x0 + coef2 * x

            if step > 0:
                var = self._coef("posterior_variance", dev, x.dtype)[t].view(view)
                noise = torch.randn(
                    shape, generator=generator, device=dev, dtype=x.dtype
                )
                x = mean + torch.sqrt(var.clamp(min=0.0)) * noise
            else:
                x = mean
        return x

    @torch.no_grad()
    def ddim_sample(
        self,
        model: torch.nn.Module,
        shape: tuple[int, ...],
        device: Any = "cpu",
        n_steps: int = 50,
        eta: float = 0.0,
        start_t: Optional[int] = None,
        x_init: Optional[Tensor] = None,
        generator: Optional[torch.Generator] = None,
    ) -> Tensor:
        """DDIM sampling over an evenly spaced subset of the trained chain.

        Parameters
        ----------
        model:
            Noise-prediction network.
        shape:
            Output shape, e.g. ``(n, 2, M, M)``.
        device:
            Torch device.
        n_steps:
            Number of reverse updates; the timesteps are
            ``round(linspace(0, start_t, n_steps))`` in descending order.
        eta:
            Stochasticity in ``[0, 1]``. ``eta=0`` is the deterministic DDIM
            update; ``eta=1`` reproduces the ancestral sampler.
        start_t:
            First timestep of the chain (default ``H - 1``).
        x_init:
            Optional initial state; defaults to ``N(0, I)``.
        generator:
            Optional RNG for the initial state and the ``eta > 0`` noise.

        Returns
        -------
        torch.Tensor
            ``shape``-shaped tensor.
        """
        dev = torch.device(device)
        t_start = self.n_steps - 1 if start_t is None else int(start_t)
        t_start = max(0, min(t_start, self.n_steps - 1))
        n_steps = max(1, min(int(n_steps), t_start + 1))

        seq = np.unique(np.round(np.linspace(0, t_start, n_steps)).astype(np.int64))[::-1]
        batch = int(shape[0])

        if x_init is None:
            x = torch.randn(
                shape, generator=generator, device=dev, dtype=torch.float32
            )
        else:
            x = x_init.to(device=dev)
            if tuple(x.shape) != tuple(shape):
                raise ValueError(f"x_init shape {tuple(x.shape)} != {tuple(shape)}")

        for i, step in enumerate(seq):
            prev = int(seq[i + 1]) if i + 1 < len(seq) else -1
            t = torch.full((batch,), int(step), device=dev, dtype=torch.long)
            pred = model(x, t)
            x0, eps = self._predict_components(pred, x, t)
            x0 = x0.clamp(-self.clip_value, self.clip_value)

            abar_t = float(self.schedule.alphas_cumprod[int(step)])
            abar_prev = (
                1.0 if prev < 0 else float(self.schedule.alphas_cumprod[prev])
            )
            sigma = float(eta) * np.sqrt(
                max(0.0, (1.0 - abar_prev) / max(1.0 - abar_t, 1e-30))
                * max(0.0, 1.0 - abar_t / max(abar_prev, 1e-30))
            )
            c = np.sqrt(max(0.0, 1.0 - abar_prev - sigma**2))
            x = float(np.sqrt(abar_prev)) * x0 + float(c) * eps
            if sigma > 0.0:
                x = x + sigma * torch.randn(
                    shape, generator=generator, device=dev, dtype=x.dtype
                )
        return x


@torch.no_grad()
def denoise_covariance(
    model: torch.nn.Module,
    diffusion: GaussianDiffusion,
    R_hat: np.ndarray,
    *,
    normalize: bool = True,
    sampler: str = "ddim",
    n_steps: int = 50,
    start_t: Optional[int] = None,
    device: Any = "cpu",
    generator: Optional[torch.Generator] = None,
) -> np.ndarray:
    """Denoise a sample covariance matrix with a trained diffusion prior.

    The complex matrix is converted to the real ``(2, M, M)`` representation,
    optionally rescaled so that its mean diagonal is one, passed through the
    reverse process and mapped back to a Hermitian ``complex128`` matrix.

    Parameters
    ----------
    model:
        Trained noise-prediction network.
    diffusion:
        The matching :class:`GaussianDiffusion` instance.
    R_hat:
        Noisy sample covariance, ``(M, M)`` complex (or real).
    normalize:
        Apply :func:`covdiff.utils.normalize_covariance` before sampling and
        undo the scaling afterwards.
    sampler:
        ``"ddim"`` (deterministic, fast) or ``"ddpm"`` (ancestral).
    n_steps:
        Number of DDIM updates (ignored by the ancestral sampler).
    start_t:
        Timestep at which the reverse chain starts. When given, the chain is
        initialised from the forward-noised input (conditional denoising), so
        pass ``snr.predict_start_timestep(...)`` to implement the paper's
        SNR-adaptive schedule. When ``None`` the full chain is run from pure
        noise, i.e. unconditional prior sampling.
    device:
        Torch device.
    generator:
        Optional RNG. Passing a seeded generator makes the call reproducible,
        including the noise injected when ``start_t`` initialises the chain
        from the measurement.

    Returns
    -------
    numpy.ndarray
        ``(M, M)`` Hermitian ``complex128`` matrix.
    """
    matrix = np.asarray(R_hat, dtype=np.complex128)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("R_hat must be a square (M, M) matrix")

    scale = 1.0
    if normalize:
        matrix, scale = normalize_covariance(matrix)

    inputs = to_model_input(matrix, drop_diagonal=diffusion.drop_diagonal)
    x0 = torch.as_tensor(inputs, dtype=torch.float32, device=device).unsqueeze(0)
    shape = tuple(x0.shape)

    x_init = None
    if start_t is not None:
        t_start = max(0, min(int(start_t), diffusion.n_steps - 1))
        t = torch.full((1,), t_start, device=x0.device, dtype=torch.long)
        x_init = diffusion.q_sample(x0, t, generator=generator)
        start_t = t_start

    if sampler == "ddim":
        out = diffusion.ddim_sample(
            model,
            shape,
            device=device,
            n_steps=n_steps,
            eta=0.0,
            start_t=start_t,
            x_init=x_init,
            generator=generator,
        )
    elif sampler in {"ddpm", "ancestral"}:
        out = diffusion.sample(
            model,
            shape,
            device=device,
            start_t=start_t,
            x_init=x_init,
            generator=generator,
        )
    else:
        raise ValueError(f"unknown sampler {sampler!r}; use 'ddim' or 'ddpm'")

    tensor = out.detach().cpu().numpy()[0]
    denoised = from_model_output(tensor)
    if diffusion.drop_diagonal:
        # The network only ever sees the off-diagonal structure (the paper's
        # preprocessing). A covariance without a diagonal is not a covariance:
        # restore the per-sensor powers from the measurement, which are the
        # best-estimated entries of R_hat.
        np.fill_diagonal(denoised, np.diag(matrix))
    if normalize:
        denoised = denormalize_covariance(denoised, scale)
    return np.asarray(hermitian(denoised), dtype=np.complex128)
