"""Evaluation layer: metrics protocols, experiments, runners and plots.

Importing this package stays cheap and side-effect free. The runner and the
plotting helpers are exposed lazily through :pep:`562` ``__getattr__``: the
plot helpers because importing ``covdiff.eval.plots`` switches matplotlib to
the ``Agg`` backend, and the runner so that ``python -m covdiff.eval.runner``
does not execute the module twice.
"""

from __future__ import annotations

from .protocol import (
    THETA1,
    THETA2,
    apply_denoiser,
    call_estimator,
    ground_truth_angles,
    make_angle_estimators,
    make_covariance_denoisers,
    run_coherence,
    run_covariance_recovery,
    run_nonuniform_noise,
    run_rmse_vs_separation,
    run_rmse_vs_snapshots,
    run_rmse_vs_snr,
    run_structure_audit,
)

__all__ = [
    "THETA1",
    "THETA2",
    "ground_truth_angles",
    "make_angle_estimators",
    "make_covariance_denoisers",
    "call_estimator",
    "apply_denoiser",
    "run_rmse_vs_snr",
    "run_rmse_vs_snapshots",
    "run_rmse_vs_separation",
    "run_coherence",
    "run_nonuniform_noise",
    "run_covariance_recovery",
    "run_structure_audit",
    "EXPERIMENTS",
    "ANGLE_EXPERIMENTS",
    "COVARIANCE_EXPERIMENTS",
    "run_experiments",
    "save_results",
    "main",
    "plot_metric",
    "plot_audit",
]

#: name -> module that defines it (imported on first access)
_LAZY = {
    "EXPERIMENTS": "runner",
    "ANGLE_EXPERIMENTS": "runner",
    "COVARIANCE_EXPERIMENTS": "runner",
    "run_experiments": "runner",
    "save_results": "runner",
    "main": "runner",
    "plot_metric": "plots",
    "plot_audit": "plots",
}


def __getattr__(name: str):
    """Import the runner / plotting helpers only when first used."""
    module_name = _LAZY.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    module = import_module(f".{module_name}", __name__)
    value = getattr(module, name)
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    return sorted(set(__all__) | set(globals()))
