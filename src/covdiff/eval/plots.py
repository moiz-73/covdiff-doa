"""Matplotlib helpers for the evaluation DataFrames.

The plotting backend is forced to ``Agg`` *before* ``pyplot`` is imported so the
module is safe on headless machines and inside CI. Every function writes a
DPI-150 PNG, closes the figure and returns the path as a string.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402  (must follow matplotlib.use)
import numpy as np
import pandas as pd

__all__ = ["plot_metric", "plot_audit"]

_X_CANDIDATES = ("snr_db", "n_snapshots", "separation_deg", "rho")


def _target(path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    return out


def _finite_error(values: np.ndarray) -> np.ndarray:
    err = np.asarray(values, dtype=float)
    return np.where(np.isfinite(err), err, 0.0)


def plot_metric(
    df: pd.DataFrame,
    x: str,
    y: str = "rmse_mean",
    yerr: Optional[str] = "rmse_std",
    *,
    xlabel: str,
    ylabel: str,
    title: str,
    path: str | Path,
) -> str:
    """Line plot with error bars, one line per ``method``.

    Parameters
    ----------
    df:
        Long-form frame with a ``method`` column and the ``x``/``y`` columns.
    x, y:
        Column names for the abscissa and the metric.
    yerr:
        Column of standard deviations; non-finite entries are drawn as zero
        error. Pass ``None`` to disable error bars.
    xlabel, ylabel, title:
        Axis and figure labels.
    path:
        Output PNG path (parent directories are created).

    Returns
    -------
    str
        The written path.
    """
    if "method" not in df.columns:
        raise ValueError("DataFrame must have a 'method' column")

    out = _target(path)
    methods: Sequence[str] = list(pd.unique(df["method"]))
    fig, ax = plt.subplots(figsize=(6.8, 4.2))

    for method in methods:
        subset = df[df["method"] == method].sort_values(x)
        xs = subset[x].to_numpy(dtype=float)
        ys = subset[y].to_numpy(dtype=float)
        err = None
        if yerr is not None and yerr in subset.columns:
            err = _finite_error(subset[yerr].to_numpy(dtype=float))
        ax.errorbar(
            xs,
            ys,
            yerr=err,
            marker="o",
            markersize=3.5,
            linewidth=1.4,
            capsize=2.5,
            elinewidth=0.9,
            label=str(method),
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.grid(True, which="major", alpha=0.3, linestyle=":")

    if len(methods) > 4:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8)
        fig.subplots_adjust(right=0.72)
    else:
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out)


def plot_audit(
    df: pd.DataFrame,
    path: str | Path,
    metric: str = "neg_energy_ratio",
    x: Optional[str] = None,
) -> str:
    """Grouped bar chart of an audit metric per method and swept condition.

    The abscissa is inferred from the frame (``snr_db`` by default) unless
    ``x`` is given. Bars are grouped by method, which makes the comparison of
    structural validity across estimators direct.
    """
    if "method" not in df.columns:
        raise ValueError("DataFrame must have a 'method' column")
    if metric not in df.columns:
        raise ValueError(f"missing metric column {metric!r}")

    x_col = x
    if x_col is None:
        x_col = next((c for c in _X_CANDIDATES if c in df.columns), None)
    if x_col is None:
        raise ValueError(f"could not infer the swept variable among {_X_CANDIDATES}")

    out = _target(path)
    methods: Sequence[str] = list(pd.unique(df["method"]))
    conditions = sorted(pd.unique(df[x_col]))
    positions = np.arange(len(conditions), dtype=float)
    width = 0.8 / max(len(methods), 1)

    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    for index, method in enumerate(methods):
        subset = df[df["method"] == method].set_index(x_col)
        heights = np.array(
            [float(subset[metric].get(condition, np.nan)) for condition in conditions]
        )
        heights = np.where(np.isfinite(heights), heights, 0.0)
        ax.bar(
            positions - 0.4 + width * (index + 0.5),
            heights,
            width=width,
            label=str(method),
        )

    ax.set_xticks(positions)
    ax.set_xticklabels([f"{c:g}" for c in conditions])
    ax.set_xlabel(x_col.replace("_", " "))
    ax.set_ylabel(metric.replace("_", " "))
    ax.set_title(f"{metric.replace('_', ' ')} by method")
    ax.grid(True, axis="y", alpha=0.3, linestyle=":")

    if len(methods) > 4:
        ax.legend(loc="upper left", bbox_to_anchor=(1.02, 1.0), frameon=False, fontsize=8)
        fig.subplots_adjust(right=0.72)
    else:
        ax.legend(frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(out)
