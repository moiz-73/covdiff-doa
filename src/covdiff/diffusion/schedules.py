"""Noise schedules for the covariance-level diffusion prior.

The source paper fixes the forward process with an *exponential* beta schedule,
Eq. (14):

.. math::

    \\beta_t = \\beta_{\\min} + (\\beta_{\\max} - \\beta_{\\min})
              \\left(1 - e^{-\\upsilon (t/H)^{\\gamma}}\\right),
    \\qquad t = 1, \\dots, H .

This module is pure NumPy: it never imports :mod:`torch`, so it can be used by
the data-preparation and analysis code without the optional extra installed.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from covdiff.config import DiffusionConfig

__all__ = [
    "linear_beta_schedule",
    "cosine_beta_schedule",
    "exponential_beta_schedule",
    "beta_schedule",
    "GaussianSchedule",
    "timestep_for_snr_db",
]

#: Smallest allowed ``1 - alpha_bar`` value, avoids division by zero in the SNR.
_TINY = 1e-30


def _validate_n_steps(n_steps: int) -> int:
    n = int(n_steps)
    if n < 2:
        raise ValueError("n_steps must be >= 2")
    return n


def _validate_betas(betas: np.ndarray, *, atol: float = 1e-12) -> np.ndarray:
    """Return ``betas`` as a flat float64 array, checking ``0 < beta < 1``."""
    b = np.asarray(betas, dtype=np.float64).reshape(-1)
    if b.size < 2:
        raise ValueError("a beta schedule needs at least two steps")
    if not np.all(np.isfinite(b)):
        raise ValueError("beta schedule contains non-finite values")
    if np.any(b <= 0.0) or np.any(b >= 1.0):
        raise ValueError(
            f"betas must lie strictly inside (0, 1); got [{b.min()!r}, {b.max()!r}]"
        )
    if b.size > 1 and np.any(np.diff(b) < -atol):
        raise ValueError("beta schedule must be non-decreasing")
    return b


def linear_beta_schedule(
    n_steps: int, beta_min: float = 1e-4, beta_max: float = 0.02
) -> np.ndarray:
    """Classic linear (Ho et al., 2020) schedule.

    Parameters
    ----------
    n_steps:
        Number of diffusion steps ``H``.
    beta_min, beta_max:
        Endpoints of the linear ramp; must satisfy ``0 < beta_min < beta_max < 1``.

    Returns
    -------
    numpy.ndarray
        ``(n_steps,)`` float64 array of betas.
    """
    n = _validate_n_steps(n_steps)
    return np.linspace(float(beta_min), float(beta_max), n, dtype=np.float64)


def cosine_beta_schedule(
    n_steps: int, s: float = 0.008, beta_max: float = 0.999
) -> np.ndarray:
    """Cosine schedule of Nichol & Dhariwal (2021).

    ``alpha_bar_t = f(t) / f(0)`` with ``f(t) = cos((t/H + s) / (1 + s) * pi/2)**2``;
    the betas are recovered as ``1 - alpha_bar_t / alpha_bar_{t-1}`` and clipped
    to ``beta_max`` (the analytic schedule reaches exactly ``1`` on the last step).

    Parameters
    ----------
    n_steps:
        Number of diffusion steps ``H``.
    s:
        Small offset that keeps ``beta_0`` from collapsing to zero.
    beta_max:
        Upper clip applied to the recovered betas.

    Returns
    -------
    numpy.ndarray
        ``(n_steps,)`` float64 array of betas.
    """
    n = _validate_n_steps(n_steps)
    steps = np.arange(n + 1, dtype=np.float64)
    f = np.cos((steps / n + float(s)) / (1.0 + float(s)) * np.pi / 2.0) ** 2
    alphas_cumprod = f / f[0]
    betas = 1.0 - alphas_cumprod[1:] / alphas_cumprod[:-1]
    return np.clip(betas, 1e-8, float(beta_max))


def exponential_beta_schedule(
    n_steps: int,
    beta_min: float = 1e-4,
    beta_max: float = 0.02,
    upsilon: float = 10.0,
    gamma: float = 1.5,
) -> np.ndarray:
    """Exponential schedule, Eq. (14) of the source paper.

    ``beta_t = beta_min + (beta_max - beta_min) * (1 - exp(-upsilon * (t/H)**gamma))``
    for ``t = 1 .. H``, so that ``beta_1`` is close to ``beta_min`` and
    ``beta_H = beta_min + (beta_max - beta_min) * (1 - exp(-upsilon))``.

    Parameters
    ----------
    n_steps:
        Number of diffusion steps ``H``.
    beta_min, beta_max:
        Envelope of the ramp.
    upsilon:
        Growth rate of the exponential (``10`` in the paper).
    gamma:
        Curvature exponent (``1.5`` in the paper).

    Returns
    -------
    numpy.ndarray
        ``(n_steps,)`` float64 array of betas, indexed by ``t - 1``.
    """
    n = _validate_n_steps(n_steps)
    t = np.arange(1, n + 1, dtype=np.float64)
    frac = (t / n) ** float(gamma)
    betas = float(beta_min) + (float(beta_max) - float(beta_min)) * (
        1.0 - np.exp(-float(upsilon) * frac)
    )
    return betas.astype(np.float64)


def beta_schedule(cfg: DiffusionConfig) -> np.ndarray:
    """Build the beta schedule selected by ``cfg.schedule``.

    Notes
    -----
    ``cfg.beta_min`` / ``cfg.beta_max`` delimit the linear and exponential
    envelopes. The cosine schedule has its own analytic cap, so it uses
    ``beta_max=0.999`` instead of the (much smaller) linear envelope.
    """
    if cfg.schedule == "linear":
        return linear_beta_schedule(cfg.n_steps, cfg.beta_min, cfg.beta_max)
    if cfg.schedule == "cosine":
        return cosine_beta_schedule(cfg.n_steps, s=cfg.cosine_s, beta_max=0.999)
    if cfg.schedule == "exponential":
        return exponential_beta_schedule(
            cfg.n_steps,
            beta_min=cfg.beta_min,
            beta_max=cfg.beta_max,
            upsilon=cfg.upsilon,
            gamma=cfg.gamma,
        )
    raise ValueError(f"unknown schedule {cfg.schedule!r}")  # pragma: no cover


def timestep_for_snr_db(snr_db: float, alphas_cumprod: np.ndarray) -> int:
    """Timestep whose forward SNR is closest to ``snr_db``.

    ``SNR_t = alpha_bar_t / (1 - alpha_bar_t)`` in dB is ``10 log10(SNR_t)``;
    the returned index minimises ``|10 log10(SNR_t) - snr_db|``.

    Parameters
    ----------
    snr_db:
        Target signal-to-noise ratio in dB.
    alphas_cumprod:
        ``(H,)`` cumulative product of ``1 - beta``.

    Returns
    -------
    int
        Index in ``[0, H - 1]``. Because ``SNR_t`` decreases with ``t``, a
        *higher* target SNR maps to an *earlier* (smaller) timestep.
    """
    ab = np.asarray(alphas_cumprod, dtype=np.float64).reshape(-1)
    if ab.size == 0:
        raise ValueError("alphas_cumprod must be non-empty")
    snr = ab / np.maximum(1.0 - ab, _TINY)
    with np.errstate(divide="ignore", invalid="ignore"):
        snr_db_curve = 10.0 * np.log10(np.maximum(snr, _TINY))
    return int(np.argmin(np.abs(snr_db_curve - float(snr_db))))


def _is_torch_like(value: Any) -> bool:
    """True when ``value`` looks like a :class:`torch.Tensor` (no import needed)."""
    module = type(value).__module__
    return module == "torch" or module.startswith("torch.")


def _broadcast(values: np.ndarray, like: Any) -> Any:
    """Reshape per-sample coefficients so they broadcast against ``like``."""
    if _is_torch_like(like):
        import torch  # local, keeps this module importable without torch

        out = torch.as_tensor(values, dtype=like.dtype, device=like.device)
        while out.ndim < like.ndim:
            out = out.unsqueeze(-1)
        return out
    out = np.asarray(values, dtype=np.float64)
    while out.ndim < np.ndim(like):
        out = out[..., None]
    return out


class GaussianSchedule:
    """Pre-computed coefficients of a Gaussian forward process.

    Parameters
    ----------
    betas:
        ``(H,)`` array of per-step noise variances, each strictly inside ``(0, 1)``.

    Attributes
    ----------
    betas, alphas:
        ``(H,)`` forward variances and their complements.
    alphas_cumprod:
        ``(H,)`` cumulative product ``prod_{s<=t} alpha_s``; also called
        ``alpha_bar_t``.
    alphas_cumprod_prev:
        Same array shifted by one, with ``1.0`` in front.
    sqrt_alphas, sqrt_alphas_cumprod, sqrt_one_minus_alphas_cumprod:
        Square roots used by the samplers.
    posterior_variance:
        ``beta_t (1 - alpha_bar_{t-1}) / (1 - alpha_bar_t)``.
    posterior_mean_coef1, posterior_mean_coef2:
        Coefficients of the ancestral posterior mean as a function of ``x0`` and
        ``x_t``.
    snr:
        ``alpha_bar_t / (1 - alpha_bar_t)``.
    """

    def __init__(self, betas: np.ndarray):
        self.betas = _validate_betas(betas)
        self.n_steps = int(self.betas.size)
        self.alphas = 1.0 - self.betas
        self.alphas_cumprod = np.cumprod(self.alphas).astype(np.float64)
        self.alphas_cumprod_prev = np.concatenate(
            [np.ones(1, dtype=np.float64), self.alphas_cumprod[:-1]]
        )
        self.sqrt_alphas = np.sqrt(self.alphas)
        self.sqrt_alphas_cumprod = np.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = np.sqrt(
            np.maximum(1.0 - self.alphas_cumprod, 0.0)
        )
        self.posterior_variance = (
            self.betas
            * np.maximum(1.0 - self.alphas_cumprod_prev, _TINY)
            / np.maximum(1.0 - self.alphas_cumprod, _TINY)
        )
        sqrt_alphas_cumprod_prev = np.sqrt(self.alphas_cumprod_prev)
        self.posterior_mean_coef1 = (
            sqrt_alphas_cumprod_prev
            * self.betas
            / np.maximum(1.0 - self.alphas_cumprod, _TINY)
        )
        self.posterior_mean_coef2 = (
            self.sqrt_alphas
            * np.maximum(1.0 - self.alphas_cumprod_prev, 0.0)
            / np.maximum(1.0 - self.alphas_cumprod, _TINY)
        )
        self.snr = self.alphas_cumprod / np.maximum(
            1.0 - self.alphas_cumprod, _TINY
        )

    # ------------------------------------------------------------------ helpers
    @property
    def sqrt_alphas_cumprod_prev(self) -> np.ndarray:
        """``sqrt(alpha_bar_{t-1})`` with ``alpha_bar_{-1} = 1``."""
        return np.sqrt(self.alphas_cumprod_prev)

    def _tensor_view(self, like: Any) -> tuple[int, ...]:
        return (-1,) + (1,) * (np.ndim(like) - 1)

    # ------------------------------------------------------------------ forward
    def q_sample(self, x0: Any, t: Any, noise: Any) -> Any:
        """Sample ``x_t`` from ``q(x_t | x_0)``.

        Parameters
        ----------
        x0:
            Clean sample, ``(B, ...)``; a NumPy array or a ``torch.Tensor``.
        t:
            Timestep(s), scalar or ``(B,)``.
        noise:
            Standard normal noise broadcastable to ``x0``.

        Returns
        -------
        Same type and shape as ``x0``.
        """
        idx = np.asarray(t, dtype=np.int64)
        if idx.ndim > 0:
            idx = idx.reshape(-1)
        sqrt_ab = _broadcast(self.sqrt_alphas_cumprod[idx], x0)
        sqrt_1mab = _broadcast(self.sqrt_one_minus_alphas_cumprod[idx], x0)
        return sqrt_ab * x0 + sqrt_1mab * noise

    def snr_db(self, t: Any) -> Any:
        """Forward SNR in dB at timestep(s) ``t`` (float for scalar input)."""
        idx = np.asarray(t, dtype=np.int64)
        snr = self.snr[idx]
        with np.errstate(divide="ignore", invalid="ignore"):
            out = 10.0 * np.log10(np.maximum(snr, _TINY))
        if out.ndim == 0:
            return float(out)
        return out

    def timestep_for_snr_db(self, snr_db: float) -> int:
        """Timestep of this schedule whose forward SNR matches ``snr_db``."""
        return timestep_for_snr_db(snr_db, self.alphas_cumprod)

    def __len__(self) -> int:
        return self.n_steps

    def __repr__(self) -> str:  # pragma: no cover - debug helper
        return (
            f"GaussianSchedule(n_steps={self.n_steps}, "
            f"beta=[{self.betas[0]:.3e}, {self.betas[-1]:.3e}], "
            f"alpha_bar[-1]={self.alphas_cumprod[-1]:.3e})"
        )
