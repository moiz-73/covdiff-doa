# Reproducing the results

Everything below runs on a laptop CPU. Timings are for a 12-core Windows box;
expect 2-4x slower on a small VM.

## 0. Install

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
source .venv/bin/activate
pip install -e ".[dev]"          # numpy/scipy/sklearn/pandas/matplotlib/pytest
pip install -e ".[torch]"        # only if you want the diffusion prior
```

The core package (simulation, baselines, metrics, audit, evaluation) does **not**
import PyTorch. `import covdiff` stays torch-free; the diffusion subpackage is
loaded lazily.

## 1. The 30-second sanity check

```bash
python scripts/quickstart.py
```

Expected output shape (numbers move in the third decimal with the seed):

```
ULA M=12, T=100, SNR=+0.0 dB, 20 Monte Carlo draws

method                     RMSE (deg)      std
----------------------------------------------
ESPRIT + toeplitz               0.110    0.078
ESPRIT (raw)                    0.163    0.082
MUSIC + toeplitz                0.945    2.515
MUSIC (raw)                     8.026    1.810
```

Writes `results/quickstart/quickstart.png`.

## 2. Classical comparison table (no checkpoint)

```bash
python scripts/run_experiment.py \
    --experiments snr snapshots separation recovery audit \
    --n-mc 20 --out-dir results/classical
```

Every sweep writes `<experiment>.csv`, `summary.json` and `figures/*.png`.
`--n-mc 1` is enough to check the plumbing; `--n-mc 50` matches the paper's
Monte Carlo budget.

## 3. Train the diffusion prior

Smoke scale (~4 minutes, 6% of the paper's budget):

```bash
python scripts/train_diffusion.py --pairs 300 --epochs 15 --out results/ddpm.pt
```

Paper scale (expect ~1-2 hours on CPU, or use `--device cuda`):

```bash
python scripts/train_diffusion.py --pairs 7260 --epochs 400 --out results/ddpm_full.pt
```

The script prints the training/validation loss, the held-out SNR-classifier
error, and a probe denoising residual, then writes a checkpoint plus a
`<name>.json` loss history.

## 4. Diffusion comparison table

```bash
python scripts/run_experiment.py \
    --experiments snr recovery audit \
    --n-mc 20 --checkpoint results/ddpm.pt --ddim-steps 50 \
    --out-dir results/diffusion
```

Adds `ddpm`, `music+ddpm` and `esprit+ddpm` to the tables. Use `--sampler ddpm`
for the ancestral sampler and `--ddim-steps 10` for the fast path.

## 5. Tests

```bash
pytest -q            # 78 tests, ~15 s
pytest -q -k "not torch"
```

## What "reproduced" means here

The repository reproduces the **protocol**, not the headline numbers:

* the signal model, the sampling of `R_hat`, and the 65,340-sample recipe are
  implemented exactly as described (16 sensors, 2 sources, 1 degree grid,
  9 SNRs, `T=1000`);
* the noise schedules (linear, cosine, exponential Eq. 14), the DDPM forward
  and reverse processes, the U-Net and the SNR-predictor head are implemented
  from the paper's equations;
* the paper's covariance classifier head is **not** reproduced — this
  repository evaluates denoising quality directly (NMSE, log-Euclidean,
  subspace angle, Pearson) and downstream DOA through MUSIC/ESPRIT, which
  isolates the contribution being tested.

If you want the paper's exact classifier, add a 121-way MLP head on top of
`covdiff.diffusion.unet.CovUNet` and train it with BCE on the multi-hot labels
already produced by `covdiff.sim.make_training_set`.
