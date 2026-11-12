#!/usr/bin/env python
"""Run the covdiff evaluation suite and write CSV / JSON / PNG artefacts.

Two modes:

* classical — no checkpoint required; MUSIC/ESPRIT combined with Ledoit-Wolf,
  OAS, diagonal loading, Toeplitz projection and Marchenko-Pastur eigen
  clipping. This is the comparison table the source paper is missing.
* diffusion — add ``--checkpoint results/ddpm.pt`` to include the trained
  covariance prior as ``ddpm``, ``music+ddpm`` and ``esprit+ddpm``.

Examples
--------
::

    python scripts/run_experiment.py --experiments snr recovery audit --n-mc 20
    python scripts/run_experiment.py --experiments all --n-mc 20 --checkpoint results/ddpm.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from covdiff.config import DEFAULT_SNR_GRID, ArrayConfig, SignalConfig  # noqa: E402
from covdiff.eval import protocol  # noqa: E402
from covdiff.eval.plots import plot_audit, plot_metric  # noqa: E402
from covdiff.eval.runner import save_results  # noqa: E402

ANGLE_EXPERIMENTS = {
    "snr": protocol.run_rmse_vs_snr,
    "snapshots": protocol.run_rmse_vs_snapshots,
    "separation": protocol.run_rmse_vs_separation,
    "coherence": protocol.run_coherence,
    "nonuniform": protocol.run_nonuniform_noise,
}
COVARIANCE_EXPERIMENTS = {
    "recovery": protocol.run_covariance_recovery,
    "audit": protocol.run_structure_audit,
}
CLASSICAL_DENOISERS = ("raw", "lw", "oas", "toeplitz", "eigclip", "diagload")


def _is_angle_method(name: str) -> bool:
    return name.split("+", 1)[0] in {"music", "esprit"}


def _make_estimator(name: str, denoisers: dict) -> object:
    from covdiff.baselines.subspace import esprit_estimate, music_estimate

    if "+" not in name:
        return {"music": music_estimate, "esprit": esprit_estimate}[name]
    backend_name, denoiser_name = name.split("+", 1)
    backend = {"music": music_estimate, "esprit": esprit_estimate}[backend_name]
    denoiser = denoisers[denoiser_name]

    def estimator(R_hat, n_sources, array, n_snapshots=1000):
        cleaned = denoiser(np.asarray(R_hat, dtype=np.complex128), int(n_snapshots))
        return np.asarray(backend(cleaned, n_sources, array))

    estimator.__name__ = f"{backend_name}_plus_{denoiser_name}"
    return estimator


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--experiments", nargs="+", default=["snr", "recovery", "audit"])
    parser.add_argument("--methods", nargs="+", default=None,
                        help="method names; defaults depend on whether a checkpoint is given")
    parser.add_argument("--n-mc", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-sensors", type=int, default=16)
    parser.add_argument("--snapshots", type=int, default=1000)
    parser.add_argument("--snr-db", type=float, default=-10.0)
    parser.add_argument("--n-sources", type=int, default=2)
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument("--ddim-steps", type=int, default=50, help="reverse-chain length for the diffusion prior")
    parser.add_argument(
        "--start-t",
        type=int,
        default=None,
        help="fixed reverse-chain start for the diffusion prior; when omitted the "
             "SNR classifier from the checkpoint picks it (the paper's Eq. 18-20)",
    )
    parser.add_argument("--sampler", choices=("ddim", "ddpm"), default="ddim")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out-dir", type=Path, default=Path("results"))
    parser.add_argument("--no-plots", action="store_true")
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if "all" in args.experiments:
        args.experiments = list(ANGLE_EXPERIMENTS) + list(COVARIANCE_EXPERIMENTS)
    unknown = [e for e in args.experiments if e not in ANGLE_EXPERIMENTS and e not in COVARIANCE_EXPERIMENTS]
    if unknown:
        raise SystemExit(f"unknown experiments: {unknown}")

    array = ArrayConfig(n_sensors=args.n_sensors)
    signal = SignalConfig(n_sources=args.n_sources, snr_db=args.snr_db, n_snapshots=args.snapshots)

    # -- covariance denoisers -------------------------------------------------
    registry = protocol.make_covariance_denoisers(list(CLASSICAL_DENOISERS))
    if args.checkpoint is not None:
        from covdiff.diffusion.io import load_checkpoint, make_ddpm_denoiser

        ckpt = load_checkpoint(args.checkpoint, device=args.device)
        registry["ddpm"] = make_ddpm_denoiser(
            ckpt.model,
            ckpt.diffusion,
            n_steps=args.ddim_steps,
            sampler=args.sampler,
            start_t=args.start_t,
            snr_model=None if args.start_t is not None else ckpt.snr_model,
            schedule=ckpt.schedule,
            snr_grid=ckpt.snr_grid,
            device=args.device,
        )
        print(f"loaded diffusion prior from {args.checkpoint} "
              f"(start_t={args.start_t if args.start_t is not None else 'snr-predicted'})")

    if args.methods is None:
        methods = ["music", "esprit", "music+lw", "esprit+lw", "music+toeplitz", "esprit+eigclip"]
        if "ddpm" in registry:
            methods += ["music+ddpm", "esprit+ddpm"]
    else:
        methods = args.methods

    angle_methods = {m: _make_estimator(m, registry) for m in methods if _is_angle_method(m)}
    cov_methods = {m: registry[m] for m in methods if not _is_angle_method(m)}
    if not angle_methods:
        angle_methods = {"music": _make_estimator("music", registry)}
    if not cov_methods:
        cov_methods = {k: registry[k] for k in ("lw", "toeplitz")}

    print(f"angle methods      : {sorted(angle_methods)}")
    print(f"covariance methods : {sorted(cov_methods)}")

    results = {}
    for name in args.experiments:
        if name in ANGLE_EXPERIMENTS:
            frame = ANGLE_EXPERIMENTS[name](
                array, signal, angle_methods,
                n_mc=args.n_mc, seed=args.seed, progress=not args.no_progress,
            )
        else:
            frame = COVARIANCE_EXPERIMENTS[name](
                array, signal, cov_methods,
                n_mc=args.n_mc, seed=args.seed, progress=not args.no_progress,
            )
        frame.attrs["config"] = {
            "experiment": name,
            "array": array.to_dict(),
            "signal": signal.to_dict(),
            "methods": sorted(angle_methods if name in ANGLE_EXPERIMENTS else cov_methods),
            "n_mc": args.n_mc,
            "seed": args.seed,
            "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        }
        results[name] = frame
        print(f"  {name:12s} {len(frame):3d} rows")

    paths = save_results(results, args.out_dir)
    print(f"\nwrote {len(paths) - 1} CSV files + summary.json to {args.out_dir}")

    if not args.no_plots:
        fig_dir = Path(args.out_dir) / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)
        for name, frame in results.items():
            try:
                if name == "audit":
                    out = plot_audit(frame, fig_dir / "audit.png")
                elif name == "recovery":
                    out = plot_metric(
                        frame, x="snr_db", y="nmse", yerr="nmse_std",
                        xlabel="SNR (dB)", ylabel="normalised covariance MSE",
                        title="Covariance recovery", path=fig_dir / "recovery_nmse.png",
                    )
                else:
                    x = {"snr": "snr_db", "snapshots": "n_snapshots",
                         "separation": "separation_deg", "coherence": "rho",
                         "nonuniform": "snr_db"}[name]
                    out = plot_metric(
                        frame, x=x, y="rmse_mean", yerr="rmse_std",
                        xlabel=x.replace("_", " ").replace("db", "(dB)"),
                        ylabel="RMSE (degrees)",
                        title=f"DOA RMSE vs {x}", path=fig_dir / f"{name}.png",
                    )
                print(f"  figure: {out}")
            except Exception as exc:  # plots must never kill a run
                print(f"  [warn] could not plot {name}: {exc}")

    (Path(args.out_dir) / "run_manifest.json").write_text(
        json.dumps(
            {
                "argv": sys.argv[1:],
                "experiments": list(results),
                "methods": methods,
                "n_mc": args.n_mc,
                "seed": args.seed,
                "snr_grid": list(DEFAULT_SNR_GRID),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
