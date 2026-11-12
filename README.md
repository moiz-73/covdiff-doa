<div align="center">

# covdiff

**Diffusion models at the covariance-matrix level — with the audit the paper skips.**

Simulation · classical baselines · structural audit · DDPM prior · reproducible protocols

</div>

---

A diffusion model that "denoises a covariance matrix" can quietly return
something that is **not a covariance matrix**: not Hermitian, not positive
semidefinite, not Toeplitz. This repository exists to make that measurable, and
to give the idea a fair comparison against the estimators a signal-processing
reviewer will actually ask about.

It is built around one paper:

> K. Liu, H. Cui, H. Wu, *Diffusion model-based covariance matrix recovery method
> for DOA estimation*, **Signal Processing** 246 (2026) 110606.
> [doi:10.1016/j.sigpro.2026.110606](https://doi.org/10.1016/j.sigpro.2026.110606)

and it ships the things that paper does not: code, a structure audit, and
Ledoit–Wolf / OAS / Toeplitz / Marchenko–Pastur baselines.

## What is in the box

| Module | What it does |
| --- | --- |
| `covdiff.sim` | ULA signal model, coherent and non-uniform-noise regimes, batched dataset generation (65,340 samples in ~3 min) |
| `covdiff.baselines` | MUSIC, ESPRIT, Ledoit–Wolf, OAS, diagonal loading, Toeplitz projection, MP eigen clipping |
| `covdiff.metrics` | DOA RMSE, NMSE, log-Euclidean distance, subspace angle, off-diagonal Pearson |
| `covdiff.audit` | Hermitian defect, Toeplitz defect, negative eigenvalue fraction and energy, condition number |
| `covdiff.diffusion` | linear / cosine / exponential (Eq. 14) schedules, covariance U-Net, DDPM + DDIM samplers, SNR classifier, checkpoint I/O |
| `covdiff.eval` | the paper's seven experiment protocols, Monte Carlo runner, CSV/JSON/PNG output |

The core is **torch-free**; the diffusion prior is an optional extra.

## Install

```bash
git clone https://github.com/moiz-73/covdiff-doa
cd covdiff-doa
python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"        # core
pip install -e ".[torch]"      # + diffusion prior
```

## 30-second demo

```bash
python scripts/quickstart.py
```

```
ULA M=12, T=100, SNR=+0.0 dB, 20 Monte Carlo draws

method                     RMSE (deg)      std
----------------------------------------------
ESPRIT + toeplitz               0.110    0.078
ESPRIT (raw)                    0.163    0.082
MUSIC + toeplitz                0.945    2.515
MUSIC (raw)                     8.026    1.810
```

Same covariance, same estimator, one structural projection: **8.0° → 0.9°**.
That is the whole thesis of the repository in one table.

## The three gaps this repo makes measurable

### 1. The i.i.d. assumption is wrong except in the noise-only case

The paper's feasibility argument starts from

```
vec(R_hat - R) ~ AsN(0, R^T ⊗ R / T)
```

and then states that every entry has the same variance and is i.i.d. Those two
statements cannot both be true: `R^T ⊗ R / T` is isotropic only when
`R = c I`. For any array with signal present, the sampling error is **coloured
and heteroscedastic**, with

```
Cov(Δ_ij, Δ_kl) = R_ik R_jl / T,     Δ_ji = conj(Δ_ij).
```

The consequence is not cosmetic: the DDPM forward process injects *isotropic*
Gaussian noise into a quantity whose real corruption is correlated. The model
is trained on the wrong noise. See [`docs/METHOD.md`](docs/METHOD.md) §8(a) for
the derivation.

### 2. Nothing keeps the output on the covariance manifold

The preprocessing flattens the off-diagonal entries into a real vector, the
network predicts noise for that vector, and the result is fed to a classifier.
At no point is the output constrained to be Hermitian, PSD, or Toeplitz.
Adding independent noise to `r_ij` and `r_ji` breaks conjugacy by construction.

`covdiff.audit` measures exactly this. On the sample covariance (which is a
valid covariance by construction) it reports a Toeplitz defect of ~0.15 at
`T=100`, while the structure-preserving baselines drive it to ~1e-16 and raise
the smallest eigenvalue. Any diffusion result should be held to the same test.

### 3. The baselines are missing

The paper compares against MUSIC, ESPRIT and four neural estimators. It does
not compare against the estimators that were built for exactly this regime:
Ledoit–Wolf (2004), OAS, analytical nonlinear shrinkage (2020),
Gavish–Donoho optimal shrinkage (2017), the graphical lasso, or Toeplitz /
Burg structured estimation. `covdiff.baselines` ships the ones that matter for
DOA, and the result is interesting:

* **Shrinkage to `μI` is useless for DOA.** It shrinks the signal subspace and
  the noise subspace by the same factor, so the MUSIC spectrum — which is
  scale-invariant in the eigenvector basis — barely moves. LW/OAS are
  numerically identical to the raw covariance to three decimals in every sweep
  we ran.
* **Structure projection is not useless.** Enforcing Hermitian Toeplitz
  structure removes the sampling-noise component that violates the array
  manifold, and improves MUSIC by an order of magnitude at low snapshots.

That distinction is invisible in the original paper's comparison table.

## Results

All numbers below are from `M=16`, `T=1000`, 10 Monte Carlo draws, on the
1-degree angle grid used by the paper. The diffusion prior is a **smoke-scale**
model: 2,700 training samples and 15 epochs, i.e. **4% of the paper's 65,340
samples and 4% of its 400 epochs**. Read the ranking as "what the pipeline does
when you turn it on", not as a reproduction of the paper's headline numbers.
The scripts to train at full scale are in the repository.

### Covariance recovery (normalised MSE, lower is better)

| SNR (dB) | raw `R_hat` | Ledoit–Wolf | MP eigen-clip | **Toeplitz** | DDPM |
| --- | --- | --- | --- | --- | --- |
| −20 | 0.0160 | 0.0042 | 0.0030 | **0.0019** | 0.0050 |
| −15 | 0.0147 | 0.0121 | 0.0035 | **0.0022** | 0.0060 |
| −10 | 0.0116 | 0.0109 | 0.0047 | **0.0016** | 0.0052 |
| −5  | 0.0046 | 0.0046 | 0.0032 | **0.0008** | 0.0031 |

Three things to notice:

1. **The DDPM works.** It improves on the raw sample covariance at every SNR,
   by 3x at −20 dB.
2. **A one-line structural projection works better.** Enforcing Hermitian
   Toeplitz structure is 2-4x better than the diffusion prior at every SNR, and
   it costs a few microseconds.
3. **Ledoit–Wolf and OAS do nothing for DOA.** Their intensity is ~0.02 at this
   aspect ratio, so the estimate is numerically the raw covariance. Shrinkage
   towards `μI` shrinks the signal and the noise subspace together.

On the paper's own metric (Pearson correlation of the off-diagonal entries at
−20 dB) the ordering is the same: raw 0.534, DDPM 0.723, Toeplitz 0.880.

### The audit: what the diffusion prior does to the manifold

Minimum eigenvalue of the recovered matrix at 10 dB:

| method | min eigenvalue | negative eigenvalues | Toeplitz defect |
| --- | --- | --- | --- |
| raw `R_hat` | 0.806 | 0% | 0.010 |
| Ledoit–Wolf | 0.831 | 0% | 0.011 |
| MP eigen-clip | 0.994 | 0% | 0.010 |
| Toeplitz projection | 0.769 | 0% | **0.000** |
| DDPM | **−1.620** | **26.2%** | 0.020 |

The diffusion output is Hermitian (the repository symmetrises it) but **not
positive semidefinite**: a quarter of its eigenvalues are negative. A MUSIC
run on that matrix is being handed noise-subspace eigenvectors that are not
noise. Every classical baseline stays on the cone, because every classical
baseline is a projection.

This is not an artefact of under-training in the usual sense: the network has no
mechanism that could enforce PSD, so a better loss cannot guarantee it. It is an
architectural gap. Fixing it means diffusing in a parameterisation that is PSD
by construction — Cholesky, matrix logarithm, or the SPD manifold.

### A note on the SNR → timestep mapping (Eq. 20)

Taking Eq. (20) literally and initialising the reverse chain with
`q_sample(R_hat, t)` at the predicted SNR starts the chain at `t ≈ 410` for
−20 dB. At that point `alpha_bar ≈ 0.01`, so the measurement has been re-noised
to almost pure noise and the "denoising" becomes unconditional generation:
NMSE rises to **1.31** (worse than predicting the zero matrix). The table above
therefore uses a fixed light start (`t=5`), which keeps the measurement intact.

The likely cause is the same i.i.d. error as in §1: the array SNR is not the
SNR of the covariance measurement. At `T=1000` the estimation error carries
about 1% of the energy, which corresponds to `t ≈ 5-30`, not `t ≈ 400`. If the
authors instead feed `R_hat` directly as `x_t` (the "cold diffusion" reading),
that is a different algorithm and should be stated explicitly. Either way, the
mapping deserves a sensitivity plot that the paper does not provide.

## Run the paper's protocols

```bash
# classical sweep: RMSE vs SNR, snapshots, separation, coherence, non-uniform noise
python scripts/run_experiment.py --experiments all --n-mc 50 --out-dir results/classical

# train the prior (smoke scale, ~4 min on CPU)
python scripts/train_diffusion.py --pairs 300 --epochs 15 --out results/ddpm.pt

# add the diffusion prior to the comparison
python scripts/run_experiment.py --experiments snr recovery audit \
    --n-mc 20 --checkpoint results/ddpm.pt --out-dir results/diffusion
```

Full details, timings and expected output: [`docs/REPRODUCING.md`](docs/REPRODUCING.md).

## Repository layout

```
src/covdiff/        simulation, baselines, metrics, audit, diffusion, eval
scripts/            quickstart, train_diffusion, run_experiment
tests/              79 tests, torch-optional
docs/               METHOD.md (paper + critique), LITERATURE.md (reading map), REPRODUCING.md
```

## Where the research can go next

These are the openings the harness is designed to test, in ascending order of
effort:

1. **Diffuse in the Toeplitz generator space.** A Hermitian Toeplitz `R` is
   determined by its first column: `2M-1 = 31` real numbers for `M=16`, versus
   the paper's `2M(M-1) = 480`. Structure comes free and the dimension drops 15x.
2. **Match the forward process to the sampling noise.** Whiten by
   `(R^T ⊗ R / T)^{-1/2}`, or replace Gaussian corruption with a Wishart
   likelihood. Removes gap 1 instead of inheriting it.
3. **Posterior sampling instead of blind denoising.** `vec(R_hat) = vec(R) + ε`
   with known covariance is a linear inverse problem; DPS/EM gives a posterior
   mean and error bars.
4. **SPD / Riemannian diffusion.** See `SPD-DDPM` (AAAI 2024) and
   `Riemannian Score-Based Generative Modelling` (NeurIPS 2022) in
   [`docs/LITERATURE.md`](docs/LITERATURE.md).
5. **Fast sampling.** DDIM, consistency and shortcut models collapse the
   1000-step chain to 4-8 steps and remove the paper's inference-time weakness.
6. **Train without clean targets.** In practice `R` is never observed, only
   `R_hat`; SURE/Noise2Self losses or the EM diffusion-prior objective make
   real-data training possible.

## Tests

```bash
pytest -q                       # 79 passed
pytest -q -k "not torch"        # core only
```

CI runs the torch-free suite on Python 3.10/3.11/3.12 plus a 3-draw quickstart.

## Citation

If this harness saves you time, cite the repository (see `CITATION.cff`) and the
paper it audits.

## License

MIT — see [`LICENSE`](LICENSE).
