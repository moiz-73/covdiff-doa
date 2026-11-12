"""Monte Carlo experiment protocols for the paper's evaluation tables.

Two callable protocols are used throughout:

``estimator(R_hat, n_sources, array) -> angles``
    A DOA estimator. Estimators that need the snapshot count (the
    ``<subspace>+<denoiser>`` combinations) additionally accept an optional
    ``n_snapshots`` argument; :func:`call_estimator` passes it when present and
    falls back to the three-argument form otherwise.

``denoiser(R_hat, signal) -> matrix``
    A covariance recovery method. The classical estimators in
    :mod:`covdiff.baselines` instead take ``(R_hat, n_snapshots)``; that form is
    recognised automatically by :func:`apply_denoiser`.

Ground truth follows the source paper: the two sources sit at
``theta1 = 5.99`` and ``theta2 = 7.01`` degrees for the SNR and snapshot sweeps,
and at ``theta1`` and ``theta1 + separation`` for the resolution sweep.

Every Monte Carlo draw uses its own generator, ``np.random.default_rng(seed +
offset)`` with a unique offset per (condition, draw), so the whole protocol is
reproducible from the seed alone. ``covdiff.sim`` is imported lazily inside the
functions so that this module stays importable while the simulator is being
written.
"""

from __future__ import annotations

import inspect
from dataclasses import replace
from typing import Callable, Dict, Mapping, Sequence

import numpy as np
import pandas as pd

from ..audit import audit_covariance
from ..config import DEFAULT_SNR_GRID, ArrayConfig, SignalConfig
from ..metrics import covariance_metrics, rmse_angles
from ..utils import hermitian

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
]

#: First source angle used in the paper's two-source experiments (degrees).
THETA1: float = 5.99
#: Second source angle used in the paper's two-source experiments (degrees).
THETA2: float = 7.01

_ANGLE_BACKENDS = ("music", "esprit")


# ---------------------------------------------------------------------------
# Ground truth
# ---------------------------------------------------------------------------
def ground_truth_angles(
    n_sources: int = 2, separation: float | None = None, theta1: float = THETA1
) -> np.ndarray:
    """Return the fixed true angles for a condition.

    Parameters
    ----------
    n_sources:
        Number of sources. With two sources the paper's ``(5.99, 7.01)`` pair
        is reproduced exactly.
    separation:
        If given, the sources are ``theta1`` and ``theta1 + separation``
        (resolution experiments). For ``n_sources > 2`` the remaining sources
        continue at the same spacing.
    theta1:
        First source angle in degrees.

    Returns
    -------
    ndarray
        Sorted angles in degrees, shape ``(n_sources,)``.
    """
    n = int(n_sources)
    if n < 1:
        raise ValueError("n_sources must be >= 1")
    if separation is not None:
        step = float(separation)
    else:
        step = (THETA2 - THETA1) / max(n - 1, 1)
    return np.asarray([theta1 + i * step for i in range(n)], dtype=float)


# ---------------------------------------------------------------------------
# Protocol adapters
# ---------------------------------------------------------------------------
def _positional_names(fn: Callable) -> list[str]:
    try:
        params = inspect.signature(fn).parameters.values()
    except (TypeError, ValueError):  # builtins / C callables
        return []
    return [
        p.name
        for p in params
        if p.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD)
    ]


def call_estimator(
    estimator: Callable,
    R_hat: np.ndarray,
    n_sources: int,
    array: ArrayConfig,
    n_snapshots: int,
) -> np.ndarray:
    """Call a DOA estimator, supplying ``n_snapshots`` only if it accepts it."""
    names = _positional_names(estimator)
    if "n_snapshots" in names:
        return np.asarray(estimator(R_hat, n_sources, array, n_snapshots=int(n_snapshots)))
    if len(names) >= 4:
        return np.asarray(estimator(R_hat, n_sources, array, int(n_snapshots)))
    return np.asarray(estimator(R_hat, n_sources, array))


def apply_denoiser(denoiser: Callable, R_hat: np.ndarray, signal: SignalConfig) -> np.ndarray:
    """Call a covariance denoiser with either protocol signature.

    ``denoiser(R_hat, signal)`` is preferred; a classical estimator declared as
    ``(R_hat, n_snapshots, **kwargs)`` is called with ``signal.n_snapshots``; a
    callable that only wants the matrix (e.g. an oracle) receives ``R_hat``.
    """
    names = _positional_names(denoiser)
    if "n_snapshots" in names:
        matrix = denoiser(R_hat, int(signal.n_snapshots))
    elif "signal" in names:
        matrix = denoiser(R_hat, signal)
    elif len(names) >= 2:
        matrix = denoiser(R_hat, int(signal.n_snapshots))
    else:
        matrix = denoiser(R_hat)
    return hermitian(np.asarray(matrix, dtype=np.complex128))


class _DenoiseThenEstimate:
    """Apply a covariance denoiser, then run a subspace estimator on the result."""

    def __init__(self, backend: Callable, denoiser: Callable, name: str) -> None:
        self.backend = backend
        self.denoiser = denoiser
        self.name = name

    def __call__(
        self, R_hat: np.ndarray, n_sources: int, array: ArrayConfig, n_snapshots: int = 1000
    ) -> np.ndarray:
        cleaned = self.denoiser(hermitian(np.asarray(R_hat, dtype=np.complex128)), int(n_snapshots))
        return np.asarray(self.backend(cleaned, n_sources, array))

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return f"_DenoiseThenEstimate({self.name!r})"


def make_angle_estimators(names: Sequence[str]) -> Dict[str, Callable]:
    """Build DOA estimators from names such as ``"music"`` or ``"music+lw"``.

    A plain name selects ``music``/``esprit`` from
    :mod:`covdiff.baselines.subspace`. A ``+`` suffix selects a covariance
    denoiser from ``covdiff.baselines.COVARIANCE_ESTIMATORS`` that is applied to
    the sample covariance first, e.g. ``"esprit+oas"``. Baselines are imported
    lazily so that this module works before they exist.
    """
    from ..baselines import COVARIANCE_ESTIMATORS
    from ..baselines.subspace import esprit_estimate, music_estimate

    backends = {"music": music_estimate, "esprit": esprit_estimate}
    estimators: Dict[str, Callable] = {}
    for name in names:
        key = str(name).strip()
        if key in backends:
            estimators[key] = backends[key]
            continue
        if "+" not in key:
            raise ValueError(
                f"unknown estimator {key!r}; use one of {sorted(backends)} "
                f"or '<method>+<denoiser>' with denoisers {sorted(COVARIANCE_ESTIMATORS)}"
            )
        backend_name, denoiser_name = (part.strip() for part in key.split("+", 1))
        backend_name = backend_name.lower()
        if backend_name not in backends:
            raise ValueError(
                f"unknown subspace method {backend_name!r}; available: {sorted(backends)}"
            )
        if denoiser_name not in COVARIANCE_ESTIMATORS:
            raise ValueError(
                f"unknown denoiser {denoiser_name!r}; available: {sorted(COVARIANCE_ESTIMATORS)}"
            )
        estimators[key] = _DenoiseThenEstimate(
            backends[backend_name], COVARIANCE_ESTIMATORS[denoiser_name], key
        )
    return estimators


def make_covariance_denoisers(names: Sequence[str]) -> Dict[str, Callable]:
    """Map estimator names to the classical covariance estimators.

    These are passed to :func:`run_covariance_recovery` and
    :func:`run_structure_audit`; :func:`apply_denoiser` adapts their
    ``(R_hat, n_snapshots)`` signature to the protocol.
    """
    from ..baselines import COVARIANCE_ESTIMATORS

    denoisers: Dict[str, Callable] = {}
    for name in names:
        key = str(name).strip()
        if key not in COVARIANCE_ESTIMATORS:
            raise ValueError(
                f"unknown covariance estimator {key!r}; available: {sorted(COVARIANCE_ESTIMATORS)}"
            )
        denoisers[key] = COVARIANCE_ESTIMATORS[key]
    return denoisers


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------
def _progress(iterable, enabled: bool, desc: str):
    if not enabled:
        return iterable
    try:
        from tqdm import tqdm
    except ImportError:  # pragma: no cover - tqdm is a hard dependency
        return iterable
    return tqdm(iterable, desc=desc, leave=False, ncols=88)


def _mean_std(values: Sequence[float], n_mc: int, *, rmse: bool = False) -> tuple[float, float]:
    """Mean (or pooled RMSE) and standard deviation over Monte Carlo draws."""
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan"), float("nan")
    centre = float(np.sqrt(np.mean(finite**2))) if rmse else float(np.mean(finite))
    if int(n_mc) <= 1 or finite.size < 2:
        return centre, float("nan")
    return centre, float(np.std(finite, ddof=0))


def _seed(seed: int, condition: int, draw: int, n_mc: int) -> int:
    return int(seed) + int(condition) * max(int(n_mc), 1) + int(draw)


def _simulate(
    angles: np.ndarray, array: ArrayConfig, signal: SignalConfig, rng: np.random.Generator
) -> Mapping[str, np.ndarray]:
    from ..sim import simulate  # lazy: the simulator is written by another module

    return simulate(np.asarray(angles, dtype=float), array, signal, rng=rng)


def _aggregate_reports(
    reports: Sequence[Mapping[str, float] | None], n_mc: int
) -> Dict[str, float]:
    keys: list[str] = []
    for report in reports:
        if report is not None:
            keys = list(report)
            break
    summary: Dict[str, float] = {}
    for key in keys:
        values = [float("nan") if rep is None else rep.get(key, float("nan")) for rep in reports]
        mean, std = _mean_std(values, n_mc)
        summary[key] = mean
        summary[f"{key}_std"] = std
    return summary


def _angle_sweep(
    array: ArrayConfig,
    signal: SignalConfig,
    estimators: Mapping[str, Callable],
    sweep_name: str,
    sweep_values: Sequence[float],
    n_mc: int,
    seed: int,
    progress: bool,
    configure: Callable[[SignalConfig, float], SignalConfig],
    truth_for: Callable[[float], np.ndarray],
    desc: str,
) -> pd.DataFrame:
    """Shared Monte Carlo loop for the four DOA sweeps."""
    rows: list[dict[str, float | str]] = []
    for cond_idx, value in enumerate(_progress(sweep_values, progress, desc)):
        sig = configure(signal, float(value))
        truth = np.asarray(truth_for(float(value)), dtype=float)
        errors: Dict[str, list[float]] = {name: [] for name in estimators}
        for draw in range(int(n_mc)):
            rng = np.random.default_rng(_seed(seed, cond_idx, draw, n_mc))
            sim = _simulate(truth, array, sig, rng)
            R_hat = np.asarray(sim["R_hat"], dtype=np.complex128)
            for name, estimator in estimators.items():
                try:
                    angles = call_estimator(
                        estimator, R_hat, sig.n_sources, array, sig.n_snapshots
                    )
                    errors[name].append(rmse_angles(angles, truth))
                except Exception:  # a failed draw is a data point, never an abort
                    errors[name].append(float("nan"))
        for name, values in errors.items():
            mean, std = _mean_std(values, n_mc, rmse=True)
            rows.append(
                {
                    "method": name,
                    sweep_name: float(value),
                    "rmse_mean": mean,
                    "rmse_std": std,
                    "n_mc": int(n_mc),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# E1 / E2 / E3 / E7 -- DOA estimation sweeps
# ---------------------------------------------------------------------------
def run_rmse_vs_snr(
    array: ArrayConfig,
    signal: SignalConfig,
    estimators: Mapping[str, Callable],
    snr_grid: Sequence[float] = DEFAULT_SNR_GRID,
    n_mc: int = 50,
    seed: int = 0,
    progress: bool = True,
) -> pd.DataFrame:
    """RMSE of angle estimates versus per-source SNR (paper Fig. 4a).

    Ground truth is fixed at ``(5.99, 7.01)`` degrees for two sources; every
    condition is averaged over ``n_mc`` independent snapshots draws.
    """
    return _angle_sweep(
        array,
        signal,
        estimators,
        "snr_db",
        [float(s) for s in snr_grid],
        n_mc,
        seed,
        progress,
        configure=lambda sig, value: sig.with_snr(value),
        truth_for=lambda _value: ground_truth_angles(signal.n_sources),
        desc="RMSE vs SNR",
    )


def run_rmse_vs_snapshots(
    array: ArrayConfig,
    signal: SignalConfig,
    estimators: Mapping[str, Callable],
    snapshots: Sequence[int] = (100, 200, 300, 400, 500, 750, 1000, 5000),
    n_mc: int = 50,
    seed: int = 0,
    progress: bool = True,
) -> pd.DataFrame:
    """RMSE of angle estimates versus the number of snapshots ``T``."""
    frame = _angle_sweep(
        array,
        signal,
        estimators,
        "n_snapshots",
        [float(t) for t in snapshots],
        n_mc,
        seed,
        progress,
        configure=lambda sig, value: sig.with_snapshots(int(value)),
        truth_for=lambda _value: ground_truth_angles(signal.n_sources),
        desc="RMSE vs snapshots",
    )
    # Keep the swept variable integral in the output table.
    frame["n_snapshots"] = frame["n_snapshots"].astype(int)
    return frame


def run_rmse_vs_separation(
    array: ArrayConfig,
    signal: SignalConfig,
    estimators: Mapping[str, Callable],
    separations: Sequence[float] = (1, 3, 6, 9, 12),
    n_mc: int = 50,
    seed: int = 0,
    progress: bool = True,
) -> pd.DataFrame:
    """RMSE versus angular separation, with ``theta1 = 5.99`` fixed.

    The second source is placed at ``theta1 + separation`` degrees.
    """
    return _angle_sweep(
        array,
        signal,
        estimators,
        "separation_deg",
        [float(s) for s in separations],
        n_mc,
        seed,
        progress,
        configure=lambda sig, _value: sig,
        truth_for=lambda sep: ground_truth_angles(signal.n_sources, separation=sep),
        desc="RMSE vs separation",
    )


def run_coherence(
    array: ArrayConfig,
    signal: SignalConfig,
    estimators: Mapping[str, Callable],
    rhos: Sequence[float] = np.arange(0.1, 1.01, 0.1),
    n_mc: int = 50,
    seed: int = 0,
    progress: bool = True,
) -> pd.DataFrame:
    """RMSE versus the source coherence coefficient ``rho`` (E7).

    ``coherent_rho`` makes the signal covariance Toeplitz and rank deficient,
    which is the regime in which subspace methods are expected to fail.
    """
    return _angle_sweep(
        array,
        signal,
        estimators,
        "rho",
        [round(float(r), 10) for r in rhos],
        n_mc,
        seed,
        progress,
        configure=lambda sig, value: replace(sig, coherent_rho=value),
        truth_for=lambda _value: ground_truth_angles(signal.n_sources),
        desc="RMSE vs coherence",
    )


def run_nonuniform_noise(
    array: ArrayConfig,
    signal: SignalConfig,
    estimators: Mapping[str, Callable],
    snr_grid: Sequence[float] = (-10, -5, 0, 5, 10),
    n_mc: int = 50,
    seed: int = 0,
    noise_db_range: tuple[float, float] = (-3, 3),
    progress: bool = True,
) -> pd.DataFrame:
    """RMSE under per-sensor (non-uniform) noise powers (E6).

    For every draw, each sensor receives a noise power drawn uniformly from
    ``noise_db_range`` in dB; the resulting tuple is passed to
    :class:`covdiff.config.SignalConfig` through ``nonuniform_noise_db``.
    """
    rows: list[dict[str, float | str]] = []
    snr_values = [float(s) for s in snr_grid]
    for cond_idx, value in enumerate(_progress(snr_values, progress, "RMSE vs non-uniform noise")):
        truth = ground_truth_angles(signal.n_sources)
        errors: Dict[str, list[float]] = {name: [] for name in estimators}
        for draw in range(int(n_mc)):
            rng = np.random.default_rng(_seed(seed, cond_idx, draw, n_mc))
            noise_db = rng.uniform(float(noise_db_range[0]), float(noise_db_range[1]), array.n_sensors)
            sig = replace(
                signal.with_snr(value),
                nonuniform_noise_db=tuple(float(x) for x in noise_db),
            )
            try:
                sim = _simulate(truth, array, sig, rng)
                R_hat = np.asarray(sim["R_hat"], dtype=np.complex128)
            except Exception:
                for name in estimators:
                    errors[name].append(float("nan"))
                continue
            for name, estimator in estimators.items():
                try:
                    angles = call_estimator(
                        estimator, R_hat, sig.n_sources, array, sig.n_snapshots
                    )
                    errors[name].append(rmse_angles(angles, truth))
                except Exception:
                    errors[name].append(float("nan"))
        for name, values in errors.items():
            mean, std = _mean_std(values, n_mc, rmse=True)
            rows.append(
                {
                    "method": name,
                    "snr_db": value,
                    "rmse_mean": mean,
                    "rmse_std": std,
                    "n_mc": int(n_mc),
                }
            )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# E4 / E5 -- covariance recovery and structural audit
# ---------------------------------------------------------------------------
def _recovery_sweep(
    array: ArrayConfig,
    signal: SignalConfig,
    denoisers: Mapping[str, Callable],
    snr_grid: Sequence[float],
    n_mc: int,
    seed: int,
    progress: bool,
    metric_fn: Callable[[np.ndarray, np.ndarray, int], Mapping[str, float]],
    desc: str,
) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    snr_values = [float(s) for s in snr_grid]
    truth = ground_truth_angles(signal.n_sources)
    for cond_idx, snr in enumerate(_progress(snr_values, progress, desc)):
        sig = signal.with_snr(snr)
        collected: Dict[str, list[Mapping[str, float] | None]] = {
            name: [] for name in denoisers
        }
        for draw in range(int(n_mc)):
            rng = np.random.default_rng(_seed(seed, cond_idx, draw, n_mc))
            try:
                sim = _simulate(truth, array, sig, rng)
            except Exception:
                for name in denoisers:
                    collected[name].append(None)
                continue
            R_hat = np.asarray(sim["R_hat"], dtype=np.complex128)
            R_true = np.asarray(sim["R_true"], dtype=np.complex128)
            for name, denoiser in denoisers.items():
                try:
                    R_est = apply_denoiser(denoiser, R_hat, sig)
                    collected[name].append(metric_fn(R_est, R_true, sig.n_sources))
                except Exception:
                    collected[name].append(None)
        for name, reports in collected.items():
            row: dict[str, float | str] = {"method": name, "snr_db": snr}
            row.update(_aggregate_reports(reports, n_mc))
            row["n_mc"] = int(n_mc)
            rows.append(row)
    return pd.DataFrame(rows)


def run_covariance_recovery(
    array: ArrayConfig,
    signal: SignalConfig,
    denoisers: Mapping[str, Callable],
    snr_grid: Sequence[float] = (-20, -15, -10, -5),
    n_mc: int = 50,
    seed: int = 0,
    progress: bool = True,
) -> pd.DataFrame:
    """Covariance-recovery quality versus SNR (E4).

    Carries the metric columns of :func:`covdiff.metrics.covariance_metrics`
    (``nmse``, ``log_euclidean``, ``subspace_angle_deg``, ``pearson_offdiag``,
    ``min_eig``, ``cond``) averaged over draws, each with a ``_std`` companion.
    """
    return _recovery_sweep(
        array,
        signal,
        denoisers,
        snr_grid,
        n_mc,
        seed,
        progress,
        metric_fn=covariance_metrics,
        desc="Covariance recovery",
    )


def run_structure_audit(
    array: ArrayConfig,
    signal: SignalConfig,
    denoisers: Mapping[str, Callable],
    snr_grid: Sequence[float] = (-20, -10, 0, 10),
    n_mc: int = 50,
    seed: int = 0,
    progress: bool = True,
) -> pd.DataFrame:
    """Structural validity of recovered covariances versus SNR (E5).

    Carries the columns of :func:`covdiff.audit.audit_covariance` averaged over
    draws: Hermitian and Toeplitz defects, extreme eigenvalues, negative
    eigenvalue fraction and energy ratio, and the condition number.
    """
    return _recovery_sweep(
        array,
        signal,
        denoisers,
        snr_grid,
        n_mc,
        seed,
        progress,
        metric_fn=lambda R_est, R_true, _n: audit_covariance(R_est),
        desc="Structure audit",
    )
