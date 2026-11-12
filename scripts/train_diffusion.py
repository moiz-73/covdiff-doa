#!/usr/bin/env python
"""Train the covariance diffusion prior (DDPM) used for matrix-level denoising.

The script reproduces the training recipe of

    K. Liu, H. Cui, H. Wu, "Diffusion model-based covariance matrix recovery
    method for DOA estimation", Signal Processing 246 (2026) 110606.

at a scale that is friendly to a laptop CPU (a few hundred angle pairs, a few
epochs) while keeping every knob of the full recipe exposed. Pass
``--pairs 7260 --epochs 400`` to run the paper's configuration.

Examples
--------
Smoke run (~2 minutes on CPU)::

    python scripts/train_diffusion.py --pairs 300 --epochs 20

Full-ish run::

    python scripts/train_diffusion.py --pairs 7260 --epochs 400 --out results/ddpm_full.pt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from covdiff.config import (  # noqa: E402
    DEFAULT_SNR_GRID,
    ArrayConfig,
    DiffusionConfig,
    ModelConfig,
    SignalConfig,
    TrainConfig,
)
from covdiff.sim import make_training_set  # noqa: E402
from covdiff.utils import normalize_covariance, to_model_input  # noqa: E402


def _pick_device(requested: str) -> str:
    import torch

    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--pairs", type=int, default=300, help="angle pairs on the 1-degree grid")
    parser.add_argument("--snapshots", type=int, default=1000, help="snapshots per sample")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--n-sensors", type=int, default=16)
    parser.add_argument("--schedule", default="exponential", choices=("linear", "cosine", "exponential"))
    parser.add_argument("--upsilon", type=float, default=10.0, help="slope knob of Eq. (14)")
    parser.add_argument("--gamma", type=float, default=1.5, help="curvature knob of Eq. (14)")
    parser.add_argument("--diffusion-steps", type=int, default=1000)
    parser.add_argument("--base-channels", type=int, default=32)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=Path("results/ddpm.pt"))
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument(
        "--train-snr-classifier",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="also fit the eigenvalue -> SNR classifier used to start the reverse chain",
    )
    parser.add_argument("--snr-epochs", type=int, default=20)
    parser.add_argument("--no-progress", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    import torch

    from covdiff.diffusion.io import save_checkpoint
    from covdiff.diffusion.model import GaussianDiffusion, denoise_covariance
    from covdiff.diffusion.schedules import GaussianSchedule, beta_schedule
    from covdiff.diffusion.snr import (
        SNRClassifier,
        predict_snr_db,
        train_snr_classifier,
    )
    from covdiff.diffusion.unet import CovUNet

    device = _pick_device(args.device)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    array = ArrayConfig(n_sensors=args.n_sensors)
    signal = SignalConfig(n_sources=2, snr_db=0.0, n_snapshots=args.snapshots)
    diffusion_cfg = DiffusionConfig(
        n_steps=args.diffusion_steps,
        schedule=args.schedule,
        upsilon=args.upsilon,
        gamma=args.gamma,
    )
    model_cfg = ModelConfig(base_channels=args.base_channels)
    train_cfg = TrainConfig(
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_train_angle_pairs=args.pairs,
        n_snapshots=args.snapshots,
        seed=args.seed,
        device=device,
    )

    started = time.time()
    print(f"[1/4] simulating {args.pairs} angle pairs x {len(DEFAULT_SNR_GRID)} SNRs "
          f"(M={args.n_sensors}, T={args.snapshots}) ...", flush=True)
    data = make_training_set(
        array,
        signal,
        n_pairs=args.pairs,
        snr_grid=DEFAULT_SNR_GRID,
        n_snapshots=args.snapshots,
        progress=not args.no_progress,
    )
    r_true = data["R_true"].astype(np.complex128)
    r_hat = data["R_hat"].astype(np.complex128)
    n = r_true.shape[0]
    print(f"      dataset: {n} samples in {time.time() - started:.1f}s", flush=True)

    # Targets: each covariance is normalised by its own mean diagonal so that
    # the network sees O(1) entries regardless of the SNR.
    x0 = np.empty((n, 2, args.n_sensors, args.n_sensors), dtype=np.float32)
    for i in range(n):
        scaled, _ = normalize_covariance(r_true[i])
        x0[i] = to_model_input(scaled, drop_diagonal=diffusion_cfg.drop_diagonal)

    perm = np.random.default_rng(args.seed).permutation(n)
    n_val = max(1, int(round(args.val_fraction * n)))
    val_idx, train_idx = perm[:n_val], perm[n_val:]

    x_train = torch.from_numpy(x0[train_idx])
    x_val = torch.from_numpy(x0[val_idx])

    schedule = GaussianSchedule(beta_schedule(diffusion_cfg))
    diffusion = GaussianDiffusion(
        schedule, predict=diffusion_cfg.predict, drop_diagonal=diffusion_cfg.drop_diagonal
    )
    model = CovUNet(
        in_channels=2,
        base_channels=model_cfg.base_channels,
        channel_mults=model_cfg.channel_mults,
        time_emb_dim=model_cfg.time_emb_dim,
        dropout=model_cfg.dropout,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[2/4] model: CovUNet, {n_params / 1e6:.2f}M parameters on {device}", flush=True)

    optimiser = torch.optim.AdamW(model.parameters(), lr=args.lr)
    loader = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(x_train),
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
    )
    history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    best_state = None

    print(f"[3/4] training for {args.epochs} epochs ({len(loader)} steps/epoch)", flush=True)
    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            loss = diffusion.training_loss(model, batch)
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            if train_cfg.grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), train_cfg.grad_clip)
            optimiser.step()
            running += float(loss.detach())
        train_loss = running / max(1, len(loader))

        model.eval()
        with torch.no_grad():
            val_loss = float(
                np.mean(
                    [
                        float(diffusion.training_loss(model, x_val[i : i + 256].to(device)))
                        for i in range(0, len(x_val), 256)
                    ]
                )
            )
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        if epoch == 1 or epoch % max(1, args.epochs // 10) == 0 or epoch == args.epochs:
            print(
                f"      epoch {epoch:3d}/{args.epochs}  train {train_loss:.5f}  val {val_loss:.5f}",
                flush=True,
            )

    if best_state is not None:
        model.load_state_dict(best_state)

    snr_model = None
    snr_grid = list(DEFAULT_SNR_GRID)
    if args.train_snr_classifier:
        print(f"[4/4] fitting the SNR classifier ({args.snr_epochs} epochs) ...", flush=True)
        snr_model = SNRClassifier(n_sensors=args.n_sensors, n_classes=len(snr_grid))
        snr_model = train_snr_classifier(
            snr_model,
            [r_hat[i] for i in train_idx],
            [float(s) for s in data["snr_db"][train_idx]],
            snr_grid,
            epochs=args.snr_epochs,
            lr=args.lr,
            batch_size=64,
            device=device,
            seed=args.seed,
        )

        # Report the predicted-SNR error on the held-out split.
        errs = []
        for i in val_idx[:64]:
            pred = predict_snr_db(snr_model, r_hat[i], snr_grid, device=device)
            errs.append(abs(float(pred) - float(data["snr_db"][i])))
        print(f"      held-out |SNR error|: {np.mean(errs):.2f} dB", flush=True)

    # Sanity check: a single conditional denoising step, end to end.
    probe = r_hat[val_idx[0]]
    denoised = denoise_covariance(
        model,
        diffusion,
        probe,
        sampler="ddim",
        n_steps=20,
        start_t=int(schedule.timestep_for_snr_db(float(data["snr_db"][val_idx[0]]))),
        device=device,
    )
    print(
        f"      probe: ||R_hat - R_denoised||_F / ||R_hat||_F = "
        f"{np.linalg.norm(probe - denoised) / np.linalg.norm(probe):.4f}",
        flush=True,
    )

    path = save_checkpoint(
        args.out,
        model=model,
        diffusion=diffusion,
        model_config=model_cfg,
        diffusion_config=diffusion_cfg,
        train_config=train_cfg,
        snr_model=snr_model,
        snr_grid=snr_grid,
        history=history,
        extra={
            "n_params": n_params,
            "n_train": int(len(train_idx)),
            "n_val": int(len(val_idx)),
            "n_sensors": args.n_sensors,
            "n_pairs": args.pairs,
            "best_val_loss": best_val,
            "wall_seconds": round(time.time() - started, 1),
        },
    )
    metrics_path = Path(args.out).with_suffix(".json")
    metrics_path.write_text(json.dumps(history, indent=2), encoding="utf-8")
    print(f"\ncheckpoint: {path}")
    print(f"history   : {metrics_path}")
    print(f"wall time : {time.time() - started:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
