#!/usr/bin/env python
"""One-command demo: simulate, denoise, estimate, audit — no PyTorch needed.

Runs a small Monte Carlo sweep on a 12-element ULA, prints two tables and
writes a figure under ``results/quickstart/``. Takes ~20-40 seconds on a
laptop CPU.

::

    python scripts/quickstart.py
    python scripts/quickstart.py --n-mc 50 --n-sensors 16
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from covdiff.audit import audit_covariance, format_audit  # noqa: E402
from covdiff.baselines import COVARIANCE_ESTIMATORS  # noqa: E402
from covdiff.baselines.subspace import esprit_estimate, music_estimate  # noqa: E402
from covdiff.config import ArrayConfig, SignalConfig  # noqa: E402
from covdiff.metrics import rmse_angles  # noqa: E402
from covdiff.sim import simulate  # noqa: E402

TRUTH = np.array([0.0, 12.0])


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-sensors", type=int, default=12)
    parser.add_argument("--snapshots", type=int, default=100)
    parser.add_argument("--snr-db", type=float, default=0.0)
    parser.add_argument("--n-mc", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--grid-step", type=float, default=0.1, help="MUSIC search grid (degrees)")
    parser.add_argument("--out-dir", type=Path, default=Path("results/quickstart"))
    parser.add_argument("--no-plot", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    array = ArrayConfig(n_sensors=args.n_sensors, grid_step_deg=args.grid_step)
    signal = SignalConfig(n_sources=2, snr_db=args.snr_db, n_snapshots=args.snapshots)

    denoisers = ["lw", "toeplitz", "eigclip"]
    methods = {
        "MUSIC (raw)": (None, music_estimate),
        "ESPRIT (raw)": (None, esprit_estimate),
    }
    for key in denoisers:
        methods[f"MUSIC + {key}"] = (key, music_estimate)
        methods[f"ESPRIT + {key}"] = (key, esprit_estimate)

    errors = {name: [] for name in methods}
    audits: dict[str, list[dict]] = {k: [] for k in ["raw", *denoisers]}

    print(f"ULA M={args.n_sensors}, T={args.snapshots}, SNR={args.snr_db:+.1f} dB, "
          f"{args.n_mc} Monte Carlo draws\n")

    for draw in range(args.n_mc):
        rng = np.random.default_rng(args.seed + draw)
        sim = simulate(TRUTH, array, signal, rng=rng)
        R_hat = sim["R_hat"]
        audits["raw"].append(audit_covariance(R_hat))

        cleaned = {"raw": R_hat}
        for key in denoisers:
            cleaned[key] = COVARIANCE_ESTIMATORS[key](R_hat, args.snapshots)
            audits[key].append(audit_covariance(cleaned[key]))

        for name, (key, backend) in methods.items():
            source = R_hat if key is None else cleaned[key]
            errors[name].append(rmse_angles(backend(source, 2, array), TRUTH))

    rows = sorted(((k, float(np.mean(v)), float(np.std(v))) for k, v in errors.items()), key=lambda r: r[1])
    print(f"{'method':24s} {'RMSE (deg)':>12s} {'std':>8s}")
    print("-" * 46)
    for name, mean, std in rows:
        print(f"{name:24s} {mean:12.3f} {std:8.3f}")

    print("\nstructure audit (mean over draws)")
    print(f"{'method':24s} {'herm':>9s} {'toeplitz':>9s} {'min eig':>10s} {'neg eig %':>10s} {'neg energy':>11s}")
    print("-" * 78)
    for key in ["raw", *denoisers]:
        report = {
            metric: float(np.mean([a[metric] for a in audits[key]]))
            for metric in audits[key][0]
        }
        print(
            f"{key:24s} {report['hermitian_defect']:9.2e} {report['toeplitz_defect']:9.2e} "
            f"{report['min_eig']:10.3f} {100 * report['neg_eig_fraction']:9.2f}% "
            f"{report['neg_energy_ratio']:11.2e}"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.out_dir / "quickstart_rows.npy", np.array(rows, dtype=object), allow_pickle=True)

    if not args.no_plot:
        try:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.pyplot as plt

            names = [r[0] for r in rows]
            means = [r[1] for r in rows]
            fig, ax = plt.subplots(figsize=(8, 0.45 * len(names) + 1.4))
            ax.barh(names[::-1], means[::-1], color="#3b6ea5")
            ax.set_xlabel("RMSE (degrees)")
            ax.set_title(f"DOA recovery after covariance denoising\nM={args.n_sensors}, T={args.snapshots}, "
                         f"SNR={args.snr_db:+.0f} dB")
            fig.tight_layout()
            path = args.out_dir / "quickstart.png"
            fig.savefig(path, dpi=150)
            plt.close(fig)
            print(f"\nfigure: {path}")
        except Exception as exc:  # pragma: no cover
            print(f"[warn] plotting skipped: {exc}")

    print("\naudit table (markdown, raw sample covariance):")
    print(format_audit({k: float(np.mean([a[k] for a in audits["raw"]])) for k in audits["raw"][0]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
