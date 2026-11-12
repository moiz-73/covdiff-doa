"""Diffusion prior for covariance matrices (requires the ``torch`` extra).

Importing :mod:`covdiff` must never import PyTorch, so every public name here is
resolved lazily through a module-level ``__getattr__``. The schedule helpers live
in :mod:`covdiff.diffusion.schedules` and are pure NumPy; ``unet``, ``model`` and
``snr`` need ``torch`` and are only imported when one of their names is used.

Examples
--------
>>> from covdiff.diffusion import beta_schedule          # NumPy only
>>> from covdiff.diffusion import CovUNet, GaussianDiffusion   # imports torch
"""

from __future__ import annotations

from importlib import import_module

__all__ = [
    # schedules (NumPy)
    "linear_beta_schedule",
    "cosine_beta_schedule",
    "exponential_beta_schedule",
    "beta_schedule",
    "GaussianSchedule",
    "timestep_for_snr_db",
    # unet (torch)
    "SinusoidalTimeEmbedding",
    "ResidualBlock",
    "CovUNet",
    # model (torch)
    "GaussianDiffusion",
    "denoise_covariance",
    # snr (torch)
    "SNRClassifier",
    "snr_class_index",
    "eigenvalues",
    "train_snr_classifier",
    "predict_snr_db",
    "predict_start_timestep",
]

#: Public name -> submodule that defines it.
_LAZY_ATTRS: dict[str, str] = {
    "linear_beta_schedule": "schedules",
    "cosine_beta_schedule": "schedules",
    "exponential_beta_schedule": "schedules",
    "beta_schedule": "schedules",
    "GaussianSchedule": "schedules",
    "timestep_for_snr_db": "schedules",
    "SinusoidalTimeEmbedding": "unet",
    "ResidualBlock": "unet",
    "CovUNet": "unet",
    "GaussianDiffusion": "model",
    "denoise_covariance": "model",
    "SNRClassifier": "snr",
    "snr_class_index": "snr",
    "eigenvalues": "snr",
    "train_snr_classifier": "snr",
    "predict_snr_db": "snr",
    "predict_start_timestep": "snr",
}


def __getattr__(name: str):
    """Resolve a public name by importing only the submodule that defines it."""
    module_name = _LAZY_ATTRS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(globals()) | set(_LAZY_ATTRS))
