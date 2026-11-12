# SPEC — module contracts for the covdiff build

This file is the single source of truth for the parallel build. Every agent owns a
disjoint set of files. **Do not edit files outside your assignment.**

Repo root: `D:\covdiff-doa`
Package root: `D:\covdiff-doa\src\covdiff`
Python: 3.10+ (runtime is 3.12). Use `from __future__ import annotations`.
Style: numpy docstrings, type hints, no external I/O in library code, no network.
Dependencies available: numpy, scipy, scikit-learn, pandas, matplotlib, tqdm, pytest, torch.

## 0. Shared conventions

Complex covariance matrices are `np.complex128` (datasets may use `complex64` to save
memory). Hermitian by construction. Everything is single-precision only at the model
boundary.

Constants and dataclasses live in `covdiff.config`; linear algebra helpers in
`covdiff.utils` (both already written — import, do not modify).

Existing helpers you MUST reuse:

```python
from covdiff.config import ArrayConfig, SignalConfig, DiffusionConfig, ModelConfig, TrainConfig
from covdiff.config import DEFAULT_SNR_GRID, angle_pair_grid
from covdiff.utils import (random_cn, hermitian, toeplitz_project, spd_project, diag_load,
                           to_model_input, from_model_output, flatten_offdiag,
                           unflatten_offdiag, normalize_covariance, denormalize_covariance,
                           offdiag_mask, fro_norm)
```

## 1. `src/covdiff/sim.py` — OWNER: SIM agent

Signal model (paper Sec. 2.1). ULA, `M = array.n_sensors`, spacing `d/lambda =
array.spacing_ratio`.

```
a(theta) = [exp(j*2*pi*spacing_ratio*(m-1)*sin(theta)) for m in 1..M]   # (M,), complex
A        = [a(theta_1), ..., a(theta_K)]                                # (M, K)
P        = diag(10**(snr_db/10))                     # uncorrelated sources
P        = 10**(snr_db/10) * Toeplitz(rho**|i-j|)    # if coherent_rho is not None
Q        = noise_var * I  (or diag(10**(nu_db/10)) when nonuniform_noise_db given)
R        = A @ P @ A.conj().T + Q
y_t      = A @ s_t + n_t,  s_t ~ CN(0, P), n_t ~ CN(0, Q)
R_hat    = (1/T) * sum_t y_t y_t^H
```

Required public API:

```python
def steering_vector(angle_deg: float, array: ArrayConfig) -> np.ndarray: ...      # (M,) complex128
def steering_matrix(angles_deg, array: ArrayConfig) -> np.ndarray: ...           # (M, K)
def signal_covariance(n_sources: int, snr_db: float,
                      coherent_rho: float | None = None) -> np.ndarray: ...      # (K, K)
def noise_covariance(array: ArrayConfig, noise_var: float = 1.0,
                     nonuniform_noise_db: tuple[float, ...] | None = None) -> np.ndarray:
def array_covariance(angles_deg, array: ArrayConfig, signal: SignalConfig) -> np.ndarray:
def sample_covariance(R: np.ndarray, n_snapshots: int,
                      rng: np.random.Generator) -> np.ndarray:
def simulate(angles_deg, array: ArrayConfig, signal: SignalConfig,
             rng: np.random.Generator | None = None) -> dict:
    # keys: "R_true", "R_hat", "angles_deg" (np.ndarray float), "snr_db",
    #       "n_snapshots", "noise_cov"
def generate_dataset(array: ArrayConfig, signal: SignalConfig,
                     angle_pairs: np.ndarray, *, dtype=np.complex64,
                     rng: np.random.Generator | None = None,
                     progress: bool = True) -> dict:
    # angle_pairs: (N, n_sources). returns
    #   "R_true": (N, M, M) dtype, "R_hat": (N, M, M) dtype,
    #   "angles_deg": (N, n_sources) float32, "labels": (N, n_grid) float32
    #   "snr_db": (N,) float32, "n_snapshots": int
    # labels are multi-hot on the 1-degree grid (paper uses BCE with 121 outputs).
def make_training_set(array, signal, *, n_pairs=7260, snr_grid=DEFAULT_SNR_GRID,
                      n_snapshots=1000, dtype=np.complex64, rng=None,
                      progress=True) -> dict: ...
```

Rules:
- `simulate` must validate `len(nonuniform_noise_db) == n_sensors`.
- Sampling uses `random_cn` (unit variance per complex entry), so scale by `sqrt(P)` and
  `sqrt(Q)` through Cholesky: `L = cholesky(P + 1e-12 I)`, `S = L @ random_cn((K, T))`.
- `sample_covariance` may build snapshots explicitly (simplest and exact) or use a
  complex Wishart draw; correctness matters, speed second.
- `generate_dataset` must not materialise more than ~1 GB. Use `complex64`.
- Vectorise over pairs (chunk if needed), no Python loop over N.
- `make_training_set` reproduces the paper: 7260 pairs x 9 SNRs = 65,340 samples with
  `T = 1000`.

## 2. `src/covdiff/baselines/` — OWNER: BASE agent

Files: `__init__.py`, `subspace.py`, `shrinkage.py`.

### `subspace.py`

```python
def music_spectrum(R: np.ndarray, n_sources: int, array: ArrayConfig,
                   angles_deg: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    # returns (angles_deg, spectrum), spectrum in dB
def music_estimate(R, n_sources, array) -> np.ndarray: ...     # (n_sources,) sorted asc
def esprit_estimate(R, n_sources, array) -> np.ndarray: ...
def _pick_peaks(spectrum, angles, k, min_sep_deg=1.0) -> np.ndarray: ...
```

- MUSIC: `eigh`, noise subspace = eigenvectors of the `M-K` smallest eigenvalues,
  `P(theta) = 1 / (a^H En En^H a)`, grid from `array.angles()`.
- ESPRIT: signal subspace `Es` = top-`K` eigenvectors; `E1 = Es[:-1]`, `E2 = Es[1:]`;
  `Psi = lstsq(E1, E2)`; `angles = arcsin(angle(eig(Psi)) / (2*pi*spacing_ratio))` in
  degrees. Guard against non-physical values (|sin| > 1) by clipping, and against
  degenerate matrices by falling back to MUSIC.
- Peak picking: greedy top-k on the spectrum with a minimum separation in degrees;
  always return exactly `k` angles (pad by rescanning the grid if needed).

### `shrinkage.py`

Classical covariance estimators. Each takes `(R_hat, n_snapshots, ...)` and returns a
Hermitian matrix of the same shape.

```python
def identity(R_hat, n_snapshots, **kwargs) -> np.ndarray          # no-op
def ledoit_wolf(R_hat, n_snapshots, **kwargs) -> np.ndarray
def oas(R_hat, n_snapshots, **kwargs) -> np.ndarray
def diagonal_loading(R_hat, n_snapshots, loading: float = 0.1, **kwargs) -> np.ndarray
def toeplitz_projection(R_hat, n_snapshots, **kwargs) -> np.ndarray
def eigen_clip(R_hat, n_snapshots, noise_var: float = 1.0, **kwargs) -> np.ndarray
def oracle(R_true, **kwargs) -> np.ndarray                        # returns R_true
```

- `ledoit_wolf`: shrink towards `mu * I`, `mu = real(trace(R_hat))/M`. For complex
  circular Gaussian the exact expected error is `E||S - Sigma||_F^2 = (tr Sigma)^2 / T`,
  so use `b2 = real(trace(R_hat))**2 / (M * T)` and
  `d2 = ||R_hat - mu I||_F^2 / M`, `lam = clip(b2 / d2, 0, 1)`,
  `R = (1 - lam) * R_hat + lam * mu * I`. Document the derivation in the docstring.
- `oas`: use the standard OAS intensity formula with the same `mu` target; clip to [0, 1].
- `eigen_clip`: Marchenko–Pastur edge `lam_plus = noise_var * (1 + sqrt(M/T))**2`;
  replace eigenvalues below `lam_plus` by their mean, keep the rest; return
  `V diag(clipped) V^H`.
- `toeplitz_projection`: `covdiff.utils.toeplitz_project` then `spd_project`.
- All returns must be Hermitian; apply `hermitian()` at the end.

### `__init__.py`

Export every public function and a registry:

```python
COVARIANCE_ESTIMATORS = {
    "raw": identity,
    "lw": ledoit_wolf,
    "oas": oas,
    "diagload": diagonal_loading,
    "toeplitz": toeplitz_projection,
    "eigclip": eigen_clip,
}
```

## 3. `src/covdiff/metrics.py`, `audit.py`, `eval/` — OWNER: EVAL agent

### `metrics.py`

```python
def rmse_angles(estimated, true) -> float
def nmse_covariance(R_est, R_true) -> float
def log_euclidean_distance(R1, R2) -> float
def subspace_angle_deg(R_est, R_true, n_sources: int) -> float
def pearson_offdiag(R_est, R_true, drop_diagonal: bool = True) -> float
def covariance_metrics(R_est, R_true, n_sources: int) -> dict[str, float]
```

- `rmse_angles`: sort both ascending, truncate to the shorter length,
  `sqrt(mean((est - true)**2))`; returns `nan` if either is empty.
- `nmse_covariance`: `||R_est - R_true||_F^2 / ||R_true||_F^2`.
- `log_euclidean_distance`: `spd_project` both then `scipy.linalg.logm`; return the
  Frobenius norm of the difference of the logs.
- `subspace_angle_deg`: principal angles between the top-`L` eigenvector subspaces;
  return the mean angle in degrees (also acceptable to return max — document it).
- `pearson_offdiag`: `flatten_offdiag` both matrices, `np.corrcoef`, guard zero variance.
- `covariance_metrics` keys: `nmse`, `log_euclidean`, `subspace_angle_deg`,
  `pearson_offdiag`, `min_eig`, `cond`.

### `audit.py`

The empirical core of the repository: does a recovery method return a *valid*
covariance matrix?

```python
def hermitian_defect(R) -> float        # ||R - R^H||_F / ||R||_F
def toeplitz_defect(R) -> float         # ||R - toeplitz_project(R)||_F / ||R||_F
def psd_report(R) -> dict[str, float]   # min_eig, max_eig, neg_eig_fraction,
                                        # neg_energy_ratio, condition_number
def audit_covariance(R) -> dict[str, float]   # union of the above
def audit_batch(matrices) -> dict[str, float] # mean over a stack
def format_audit(report: dict[str, float]) -> str   # markdown table string
```

- `neg_energy_ratio = ||sum_{lambda<0} lambda v v^H||_F / ||R||_F`.
- `audit_covariance` must be robust to non-Hermitian input (compute on `hermitian(R)`
  for the spectral part, report the defect separately).
- `format_audit` returns a two-column markdown table with readable names.

### `eval/protocol.py`

Estimator protocol: `estimator(R_hat, n_sources, array) -> np.ndarray` of angles.
Denoiser protocol: `denoiser(R_hat, signal) -> np.ndarray` of a matrix.

```python
def make_angle_estimators(names: Sequence[str]) -> dict[str, Callable]
    # "music", "esprit", plus "music+<denoiser>" and "esprit+<denoiser>" combos
    # built from covdiff.baselines.COVARIANCE_ESTIMATORS
def run_rmse_vs_snr(array, signal, estimators, snr_grid=DEFAULT_SNR_GRID,
                    n_mc=50, seed=0, progress=True) -> pd.DataFrame
def run_rmse_vs_snapshots(array, signal, estimators, snapshots=(100,200,300,400,500,750,1000,5000),
                          n_mc=50, seed=0, progress=True) -> pd.DataFrame
def run_rmse_vs_separation(array, signal, estimators, separations=(1,3,6,9,12),
                           n_mc=50, seed=0, progress=True) -> pd.DataFrame
def run_coherence(array, signal, estimators, rhos=np.arange(0.1,1.01,0.1),
                  n_mc=50, seed=0, progress=True) -> pd.DataFrame
def run_nonuniform_noise(array, signal, estimators, snr_grid=(-10,-5,0,5,10),
                         n_mc=50, seed=0, noise_db_range=(-3,3), progress=True) -> pd.DataFrame
def run_covariance_recovery(array, signal, denoisers, snr_grid=(-20,-15,-10,-5),
                            n_mc=50, seed=0, progress=True) -> pd.DataFrame
def run_structure_audit(array, signal, denoisers, snr_grid=(-20,-10,0,10),
                        n_mc=50, seed=0, progress=True) -> pd.DataFrame
```

Every returned DataFrame has at least: `method`, and the swept variable
(`snr_db`, `n_snapshots`, `separation_deg`, `rho`), plus `rmse_mean`, `rmse_std`
(`std = nan` when `n_mc == 1`). Recovery/audit frames carry the metric columns from
`covariance_metrics` / `audit_covariance` instead of RMSE.

Monte Carlo: for each condition draw `n_mc` independent realisations with
`rng = np.random.default_rng(seed + offset)`; ground truth angles fixed per condition
(e.g. `theta1 = 5.99`, `theta2 = 7.01` for E1/E2 as in the paper).

### `eval/runner.py`

```python
EXPERIMENTS = {"snr": run_rmse_vs_snr, "snapshots": ..., "separation": ...,
               "coherence": ..., "nonuniform": ..., "recovery": ..., "audit": ...}
def run_experiments(names: Sequence[str], *, array, signal, method_names,
                    n_mc=50, seed=0) -> dict[str, pd.DataFrame]
def save_results(results: dict[str, pd.DataFrame], out_dir: str | Path) -> dict[str, str]
    # writes <name>.csv per frame and a summary.json with the config snapshot
def main(argv=None) -> int
```

`save_results` creates `out_dir` (default `results/`), returns the written paths.

### `eval/plots.py`

```python
def plot_metric(df, x, y="rmse_mean", yerr="rmse_std", *, xlabel, ylabel, title,
                path) -> str          # error-bar line plot, one line per method
def plot_audit(df, path) -> str      # grouped bars of neg_energy_ratio per method
```

Use the `Agg` backend, `matplotlib.use("Agg")` before pyplot import, tight_layout,
DPI 150, and return the file path.

## 4. `src/covdiff/diffusion/` — OWNER: DIFF agent

Files: `__init__.py`, `schedules.py`, `unet.py`, `model.py`, `snr.py`.

### `schedules.py`

```python
def linear_beta_schedule(n_steps, beta_min=1e-4, beta_max=0.02) -> np.ndarray
def cosine_beta_schedule(n_steps, s=0.008, beta_max=0.999) -> np.ndarray
def exponential_beta_schedule(n_steps, beta_min=1e-4, beta_max=0.02,
                              upsilon=10.0, gamma=1.5) -> np.ndarray
def beta_schedule(cfg: DiffusionConfig) -> np.ndarray
class GaussianSchedule:
    def __init__(self, betas: np.ndarray)
    # attributes: betas, alphas, alphas_cumprod, sqrt_alphas_cumprod,
    #             sqrt_one_minus_alphas_cumprod, posterior_variance,
    #             snr (alphas_cumprod / (1 - alphas_cumprod))
    def q_sample(self, x0, t, noise): ...
    def snr_db(self, t): ...
    def timestep_for_snr_db(self, snr_db: float) -> int: ...
def timestep_for_snr_db(snr_db: float, alphas_cumprod: np.ndarray) -> int
```

`exponential_beta_schedule` implements Eq. (14): with `t = 1..H`,
`beta_t = beta_min + (beta_max - beta_min) * (1 - exp(-upsilon * (t/H)**gamma))`.
Assert monotone non-decreasing, `0 < beta < 1`, `alphas_cumprod[-1] < 0.01`.
`timestep_for_snr_db` picks `argmin |10*log10(alpha_bar/(1-alpha_bar)) - snr_db|`.

### `unet.py`

```python
class SinusoidalTimeEmbedding(nn.Module): ...
class CovUNet(nn.Module):
    def __init__(self, in_channels: int = 2, base_channels: int = 32,
                 channel_mults: tuple[int, ...] = (1, 2, 4),
                 time_emb_dim: int = 128, dropout: float = 0.0): ...
    def forward(self, x: Tensor, t: Tensor) -> Tensor
```

- Input `(B, 2, M, M)`, output same shape. Must accept any `M` that is divisible by
  `2 ** (len(channel_mults) - 1)` (16 works).
- Blocks: `Conv2d -> GroupNorm -> SiLU`, residual, time embedding injected by adding a
  linear projection to each block; downsample with strided conv, upsample with
  `F.interpolate` + conv, skip connections.
- Keep it < 5M parameters with the default config. Use `nn.GroupNorm(min(8, c), c)`.

### `model.py`

```python
class GaussianDiffusion:
    def __init__(self, schedule: GaussianSchedule, predict: str = "epsilon",
                 drop_diagonal: bool = True): ...
    def q_sample(self, x0, t, noise=None, generator=None): ...
    def training_loss(self, model, x0, t=None, generator=None) -> Tensor
    def sample(self, model, shape, device="cpu", start_t: int | None = None,
               generator=None, clip_denoised: bool = True) -> Tensor
    def ddim_sample(self, model, shape, device="cpu", n_steps: int = 50,
                    eta: float = 0.0, start_t: int | None = None) -> Tensor
@torch.no_grad()
def denoise_covariance(model, diffusion, R_hat, *, normalize: bool = True,
                       sampler: str = "ddim", n_steps: int = 50,
                       start_t: int | None = None, device="cpu") -> np.ndarray
```

- `training_loss` samples `t` uniformly (if not given), noise from the generator,
  supports `predict="epsilon"` and `predict="v"`.
- `sample` is the ancestral sampler; `start_t` lets the caller start the chain late
  (the SNR-prediction trick from the paper).
- `denoise_covariance` accepts a complex `R_hat` (M,M), builds `to_model_input`,
  normalises with `normalize_covariance`, runs the sampler, denormalises, returns a
  Hermitian complex matrix using `from_model_output`.
- All torch imports must be lazy-safe: `import torch` at module top is fine (torch is an
  optional extra), but `covdiff/__init__.py` must NOT import the diffusion subpackage.

### `snr.py`

```python
class SNRClassifier(nn.Module):
    def __init__(self, n_sensors: int, n_classes: int = 9, hidden=(256, 512, 256)): ...
    def forward(self, eigvals: Tensor) -> Tensor          # logits (B, n_classes)
def snr_class_index(snr_db: float, snr_grid) -> int
def eigenvalues(R: np.ndarray) -> np.ndarray               # descending real
def train_snr_classifier(model, R_hats, snr_db, snr_grid, *, epochs=30, lr=1e-3,
                         batch_size=64, device="cpu", seed=0) -> model
@torch.no_grad()
def predict_snr_db(model, R: np.ndarray, snr_grid, device="cpu") -> float
@torch.no_grad()
def predict_start_timestep(model, R, schedule, snr_grid, device="cpu") -> int
```

The classifier input is the descending eigenvalue vector scaled by its mean (guard
zero). Output is a softmax over the SNR grid; `predict_snr_db` returns the
expectation.

## 5. `docs/` — OWNER: DOCS agent

- `docs/METHOD.md`: what the source paper does, its four contributions, and a precise
  critique with the three holes (i.i.d. claim vs `R^T ⊗ R / T`; no PSD/Hermitian/Toeplitz
  constraint; missing classical baselines). Include equations in LaTeX.
- `docs/LITERATURE.md`: the annotated reading map, grouped in six sections
  (SPD/manifold diffusion; array-processing diffusion; covariance-level diffusion in
  other domains; inverse-problem reframing; classical covariance estimation;
  training without clean targets) with DOIs/arXiv ids and a one-line "why it matters".
- `docs/REPRODUCING.md`: step-by-step commands, expected runtime, expected outputs.

## 6. Integration (main agent)

`scripts/`, `tests/` for cross-module behaviour, `README.md`, `CITATION.cff`, git and
GitHub are handled by the main agent.
