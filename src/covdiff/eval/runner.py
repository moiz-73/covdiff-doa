"""Experiment runner: the seven paper experiments behind one small CLI.

``python -m covdiff.eval.runner --experiments snr snapshots --n-mc 20``

Nothing heavy (torch, the simulator, matplotlib) is imported at module import:
the protocols import ``covdiff.sim`` and ``covdiff.baselines`` lazily, so the
CLI's ``--help`` and argument validation work before those modules exist.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Mapping, Sequence

import pandas as pd

from ..config import ArrayConfig, SignalConfig
from .protocol import (
    make_angle_estimators,
    make_covariance_denoisers,
    run_covariance_recovery,
    run_coherence,
    run_nonuniform_noise,
    run_rmse_vs_separation,
    run_rmse_vs_snapshots,
    run_rmse_vs_snr,
    run_structure_audit,
)

__all__ = ["EXPERIMENTS", "ANGLE_EXPERIMENTS", "COVARIANCE_EXPERIMENTS", "run_experiments", "save_results", "main"]

#: Registry used by the CLI and by :func:`run_experiments`.
EXPERIMENTS = {
    "snr": run_rmse_vs_snr,
    "snapshots": run_rmse_vs_snapshots,
    "separation": run_rmse_vs_separation,
    "coherence": run_coherence,
    "nonuniform": run_nonuniform_noise,
    "recovery": run_covariance_recovery,
    "audit": run_structure_audit,
}

#: Experiments whose `method` column contains DOA estimators.
ANGLE_EXPERIMENTS = frozenset({"snr", "snapshots", "separation", "coherence", "nonuniform"})
#: Experiments whose `method` column contains covariance denoisers.
COVARIANCE_EXPERIMENTS = frozenset({"recovery", "audit"})


def run_experiments(
    names: Sequence[str],
    *,
    array: ArrayConfig,
    signal: SignalConfig,
    method_names: Sequence[str],
    n_mc: int = 50,
    seed: int = 0,
    progress: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Run the named experiments and return one DataFrame per experiment.

    Parameters
    ----------
    names:
        Any subset of :data:`EXPERIMENTS`.
    array, signal:
        Base configuration; each protocol applies its own sweep on top.
    method_names:
        For DOA experiments, names accepted by
        :func:`covdiff.eval.protocol.make_angle_estimators` (``music``,
        ``esprit``, ``music+lw``, ...). For ``recovery``/``audit``, names from
        ``covdiff.baselines.COVARIANCE_ESTIMATORS``.
    n_mc:
        Monte Carlo draws per condition.
    seed:
        Base seed; draw ``k`` of condition ``c`` uses
        ``default_rng(seed + c * n_mc + k)``.
    progress:
        Show a tqdm bar per condition.

    Returns
    -------
    dict
        ``{name: DataFrame}``; each frame carries the run configuration in
        ``DataFrame.attrs["config"]`` so :func:`save_results` can snapshot it.
    """
    results: Dict[str, pd.DataFrame] = {}
    unknown = [name for name in names if name not in EXPERIMENTS]
    if unknown:
        raise ValueError(f"unknown experiments {unknown}; choose from {sorted(EXPERIMENTS)}")

    for name in names:
        if name in ANGLE_EXPERIMENTS:
            estimators = make_angle_estimators(method_names)
            frame = EXPERIMENTS[name](
                array, signal, estimators, n_mc=n_mc, seed=seed, progress=progress
            )
        else:
            denoisers = make_covariance_denoisers(method_names)
            frame = EXPERIMENTS[name](
                array, signal, denoisers, n_mc=n_mc, seed=seed, progress=progress
            )
        frame.attrs["config"] = {
            "experiment": name,
            "array": array.to_dict(),
            "signal": signal.to_dict(),
            "methods": [str(m) for m in method_names],
            "n_mc": int(n_mc),
            "seed": int(seed),
        }
        results[name] = frame
    return results


def _json_default(value):
    if isinstance(value, (Path,)):
        return str(value)
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def save_results(results: Mapping[str, pd.DataFrame], out_dir: str | Path = "results") -> Dict[str, str]:
    """Write one CSV per experiment plus a ``summary.json`` config snapshot.

    Returns
    -------
    dict
        Mapping ``experiment name -> path`` of the written CSV files (the
        summary is reported under the key ``"summary"``).
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    paths: Dict[str, str] = {}
    summary: Dict[str, object] = {
        "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "experiments": {},
    }
    for name, frame in results.items():
        path = out / f"{name}.csv"
        frame.to_csv(path, index=False)
        paths[str(name)] = str(path)
        summary["experiments"][str(name)] = {  # type: ignore[index]
            "csv": str(path),
            "rows": int(len(frame)),
            "columns": [str(c) for c in frame.columns],
            "config": frame.attrs.get("config"),
        }

    summary_path = out / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=_json_default)
    paths["summary"] = str(summary_path)
    return paths


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m covdiff.eval.runner",
        description="Run the covdiff evaluation experiments and write CSV/JSON results.",
    )
    parser.add_argument(
        "--experiments",
        nargs="+",
        default=list(EXPERIMENTS),
        choices=sorted(EXPERIMENTS),
        help="experiments to run (default: all seven)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["music", "esprit", "music+lw", "esprit+oas"],
        help="DOA estimators for angle sweeps, covariance estimators for recovery/audit",
    )
    parser.add_argument("--n-mc", type=int, default=50, help="Monte Carlo draws per condition")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-sensors", type=int, default=16)
    parser.add_argument("--snapshots", type=int, default=1000)
    parser.add_argument("--snr-db", type=float, default=0.0)
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns a process exit code."""
    args = _build_parser().parse_args(argv)

    array = ArrayConfig(n_sensors=args.n_sensors)
    signal = SignalConfig(snr_db=args.snr_db, n_snapshots=args.snapshots)

    results = run_experiments(
        args.experiments,
        array=array,
        signal=signal,
        method_names=args.methods,
        n_mc=args.n_mc,
        seed=args.seed,
        progress=not args.no_progress,
    )
    paths = save_results(results, args.out_dir)
    for name, path in paths.items():
        print(f"{name}: {path}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
