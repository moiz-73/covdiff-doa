# METHOD — what the paper does, and where it does not hold

**Source.** K. Liu, H. Cui, H. Wu, *Diffusion model-based covariance matrix recovery method for DOA estimation*,
Signal Processing **246** (2026) 110606. `doi:10.1016/j.sigpro.2026.110606`. Received 7 Jul 2025, accepted 18 Mar
2026.

Sections 1–6 restate the method with its equation numbers, Section 7 isolates what is genuinely new, Section 8 is
the critique (three defects, the first of which invalidates the paper's stated reason for using a diffusion model at
all), Section 9 says what this repository adds, Section 10 records the fidelity boundary. Notation: $M$ sensors, $K$
sources, $T$ snapshots, $H$ diffusion steps, $D$ samples; the paper writes $L$ for the number of sources, we use $K$
to free $L$ for Monte Carlo trials.

## 1. Signal model and the two covariances

A ULA with half-wavelength spacing receives $K$ uncorrelated narrowband far-field sources,
$\mathbf{y}(k)=\mathbf{A}\mathbf{s}(k)+\mathbf{n}(k)$, $k=1,\dots,T$ (Eq. 1), with
$\mathbf{A}=[\mathbf{a}(\theta_1),\dots,\mathbf{a}(\theta_K)]$ and (Eq. 2)

$$\mathbf{a}(\theta_k)=\big[\,1,\;e^{j2\pi\frac{d}{\lambda}\sin\theta_k},\;\dots,\;e^{j2\pi\frac{d}{\lambda}(M-1)\sin\theta_k}\,\big]^{T}.$$

The **expectation covariance** (Eq. 3) and the **sample covariance** (Eq. 4) are

$$\mathbf{R}=\mathbb{E}\{\mathbf{y}\mathbf{y}^{H}\}=\mathbf{A}\mathbf{P}\mathbf{A}^{H}+\mathbf{Q},\qquad
\hat{\mathbf{R}}=\frac{1}{T}\sum_{k=1}^{T}\mathbf{y}(k)\mathbf{y}^{H}(k),$$

with $\mathbf{P}=\mathrm{diag}(\mathbf{p})$ and $\mathbf{Q}=\mathrm{diag}(\boldsymbol{\sigma})$. $\hat{\mathbf{R}}$
is the only object an estimator ever sees: Hermitian PSD by construction, but a *random matrix* — not $\mathbf{R}$
perturbed by small white noise. That distinction is Section 8(a). Under the paper's own assumptions
($\mathbf{Q}=\sigma^2\mathbf{I}$, uncorrelated sources, exact steering vectors) $\mathbf{R}$ is additionally
**Toeplitz with constant diagonal**, $R_{ij}=r_{|i-j|}$, $R_{ii}=\sum_k p_k+\sigma^2$ — used in Sec. 3.1 to motivate
the method and never used again.

**Why the two differ.** For $\mathbf{y}\sim\mathcal{CN}(0,\mathbf{R})$ the sample covariance is a complex Wishart
draw, $T\hat{\mathbf{R}}\sim\mathcal{CW}_M(T,\mathbf{R})$: unbiased, but with $O(1/T)$ variance that is
*structured*. Eq. 11 states the asymptotic law

$$\Delta=\mathrm{vec}(\hat{\mathbf{R}})-\mathrm{vec}(\mathbf{R})\;\sim\;
\mathcal{AN}\!\big(\mathbf{0}_{M^2},\;\tfrac{1}{T}\mathbf{R}^{T}\otimes\mathbf{R}\big),$$

the standard covariance-matching result it cites (Ottersten–Stoica–Roy, ref. [29]). The equation is correct. The
sentence immediately after it is not.

## 2. The DDPM machinery, as used

The forward chain is textbook (Eq. 5–7):
$\mathbf{R}_t=\sqrt{\alpha_t}\mathbf{R}_{t-1}+\sqrt{1-\alpha_t}\boldsymbol{\varepsilon}_t$ with
$\boldsymbol{\varepsilon}_t\sim\mathcal{N}(\mathbf{0},\mathbf{I})$, whose closed form (Eq. 6, restated as Eq. 12) is

$$\mathbf{R}_t=\sqrt{\bar\alpha_t}\mathbf{R}_0+\sqrt{1-\bar\alpha_t}\boldsymbol{\varepsilon},\qquad
q(\mathbf{R}_t\mid\mathbf{R}_0)=\mathcal{N}\!\big(\sqrt{\bar\alpha_t}\mathbf{R}_0,(1-\bar\alpha_t)\mathbf{I}\big),\qquad
\bar\alpha_t=\prod_{l\le t}\alpha_l .$$

Note what that identity covariance means: **every entry of $\mathbf{R}_t$, real and imaginary, receives the same
independent noise variance $1-\bar\alpha_t$** — the choice the paper never revisits and its own Eq. 11 contradicts.
The reverse chain (Eq. 8–10, restated as Eq. 13) is
$p_\theta(\mathbf{R}_{t-1}\mid\mathbf{R}_t,\mathbf{R}_0)=\mathcal{N}(\mathbf{R}_{t-1};\boldsymbol{\mu}_\theta,\sigma^2_\theta\mathbf{I})$
with

$$\boldsymbol{\mu}_\theta=\frac{1}{\sqrt{\alpha_t}}\Big(\mathbf{R}_t-\frac{1-\alpha_t}{\sqrt{1-\bar\alpha_t}}\boldsymbol{\varepsilon}_\theta(\mathbf{R}_t,t)\Big),\qquad
\sigma^2_\theta=\frac{(1-\bar\alpha_{t-1})(1-\alpha_t)}{1-\bar\alpha_t},$$

and training minimises noise prediction (Eq. 23)

$$\mathcal{L}(\theta)=\mathbb{E}_{t,\mathbf{R}_0,\boldsymbol{\varepsilon}}\big\|\boldsymbol{\varepsilon}-\boldsymbol{\varepsilon}_\theta\big(\sqrt{\bar\alpha_t}\mathbf{R}_0+\sqrt{1-\bar\alpha_t}\boldsymbol{\varepsilon},\,t\big)\big\|_2^2 .$$

Nothing in Eq. 5–10 or 23 is modified. But $\mathbf{R}_t$ is "the corresponding sampling covariance matrix at the
$t$th time step" (Sec. 3.1) — a *synthetic* interpolation, not a realisable sequence of covariances: the forward
process degrades $\mathbf{R}$ at a chosen rate, it does not model how $\hat{\mathbf{R}}$ arises.

## 3. The exponential noise schedule (Eq. 14)

$$\beta_t=\beta_{\min}+(\beta_{\max}-\beta_{\min})\big(1-e^{-\upsilon (t/H)^{\gamma}}\big),\qquad
\alpha_t=1-\beta_t,\qquad \beta_{\min}=10^{-4},\;\beta_{\max}=0.02,$$

with two tune parameters, selected as $\upsilon=10$, $\gamma=1.5$ by coordinate search ($\upsilon\in\{5,\dots,15\}$
at fixed $\gamma=1.5$, then $\gamma\in\{0.5,1.0,1.5,2.0,2.5\}$ at fixed $\upsilon=10$). Rationale: a linear schedule
"adds noise slowly when $t$ is small", so the model sees insufficiently corrupted covariances early and under-learns
the low-SNR regime. Two remarks matter later.

1. $1-e^{-\upsilon u^{\gamma}}$ saturates fast, so $\beta_t\approx\beta_{\max}$ over most of $t$: with
   $\upsilon=10,\gamma=1.5$, $\sum_t\beta_t\approx16$ and $\bar\alpha_H\approx e^{-16}\approx10^{-7}$ — the
   terminal state is comfortably standard normal, as required.
2. A monotone $\beta_t$ is a **reparameterisation of the same reverse path along $t$**. With $t\sim U$ in
   training, changing the schedule changes *how many training examples land in each SNR decade*, not which
   noise levels are representable: the described mechanism is an importance weighting over noise levels in
   disguise — a legitimate heuristic, not a new generative process, predicting exactly the small margin
   reported ($0.82°\to0.79°$).

The schedule is tuned on the Pearson correlation of off-diagonal entries (Eq. 16) over 500 matrices per SNR on the
9-point grid; Section 8(c) explains why that metric cannot carry the decision.

## 4. SNR-predictor timestep trick (Eq. 18–21)

The forward process monotonically lowers SNR, the reverse process raises it, so the chain need not start at $t=H$:
start at the $t$ whose SNR matches the data. Under the unit-power convention of Eq. 6, $\mathbf{R}_t$ carries signal
power $\bar\alpha_t$ and noise power $1-\bar\alpha_t$, so (Eq. 20)

$$\frac{\bar\alpha_t}{1-\bar\alpha_t}=\frac{p_s}{p_n}=10^{\mathrm{SNR}/10}
\quad\Longrightarrow\quad
t^\star=\bar\alpha^{-1}\!\Big(\frac{10^{\mathrm{SNR}/10}}{1+10^{\mathrm{SNR}/10}}\Big).$$

$t^\star$ needs two steps. **(i) SNR classifier.** Input: descending eigenvalue vector $\boldsymbol{\lambda}'$ of
$\hat{\mathbf{R}}$; FC(256)–FC(512)–FC(256)–FC(9), ReLU between layers, Sigmoid over $\{-20,-15,\dots,20\}$ dB.
"Sigmoid" is inconsistent with a 9-way categorical target, so we treat the head as 9 logits with cross-entropy and
report the softmax expectation — the only reading under which the paper's own training procedure is well defined.
**(ii) Lookup.** Invert $\bar\alpha_t$ and take the nearest $t$ (`timestep_for_snr_db`), i.e.
$\arg\min_t\big|10\log_{10}(\bar\alpha_t/(1-\bar\alpha_t))-\widehat{\mathrm{SNR}}\big|$.

The claimed benefit is compute: Table 1 gives EDDPM at $H=1000$ dropping from ~1.15 s/sample at $-20$ dB to ~0.16
s/sample at $+20$ dB against a flat ~0.08 s for the CNN. Real benefit; the derivation is not — see 8(a), where the
covariance-level distortion is $(\sum_i\lambda_i)^2/(T\sum_i\lambda_i^2)$, not $10^{-\mathrm{SNR}/10}$.

## 5. The three networks

**Input representation** (Sec. 3.3.1, Fig. 5). Diagonal dropped, both strict triangles kept, real and imaginary
parts split and concatenated into $\tilde{\mathbf{R}}\in\mathbb{R}^{1\times 2M(M-1)}$, reshaped to a $2\times
M\times M$ two-channel image. Stated reason: diagonal entries are "insufficiently robust against noise".

**DDPM module** (Sec. 3.3.3, Fig. 6a). U-Net, 18 stages: $f_1,f_3,\dots,f_{17}$ DoubleConv (two convolutions,
normalisation, ReLU); $f_2,f_4,f_6,f_8$ $2\times2$ max-pooling downsamplers (text says "step size of 1", almost
certainly stride 2); $f_{10},f_{12},f_{14},f_{16}$ transposed-convolution upsamplers; $f_{17}$ a pre-output
convolution, so Eq. 22 writes the stack as $f(\hat{\mathbf{R}})=f_{18}(f_{17}(\cdots
f_1(\hat{\mathbf{R}})))=\hat{\mathbf{U}}$. The timestep goes through FC(64)+SiLU and FC to 128 dimensions, then is
**concatenated** with the input rather than added (non-standard; the usual form is an additive projection per
block). Channel widths appear only in Fig. 6(a).

**DOA classifier** (Sec. 3.3.4, Fig. 6c). Two Conv2d ($8$ then $16$ channels, $3\times3$, stride 1, no padding) each
with normalisation and ReLU, then five FC layers (512, 256, 256, 256, 121), ReLU, Sigmoid. Loss is BCE against a
multi-hot label on the 1° grid over $[-60°,60°]$ (Eq. 24):

$$\mathcal{L}=-\frac{1}{n}\sum_{i=1}^{n}\big[y_i\log p_i+(1-y_i)\log(1-p_i)\big],$$

i.e. 121 independent Bernoulli decisions, not a distribution over angle sets: no "exactly $K$ sources" constraint.

## 6. Training recipe (Sec. 3.3.5)

| Item | Value |
|---|---|
| Array / sources | ULA, $M=16$, $d=\lambda/2$; $K=2$ on the 1° grid over $[-60°,60°]$ (121 cells) |
| Angle pairs | all ordered pairs, distinct grid angles: $\binom{121}{2}=7260$ |
| SNR grid / snapshots | $\{-20,-15,-10,-5,0,5,10,15,20\}$ dB (9 levels); $T=1000$; $H=1000$ |
| Dataset size | $D=7260\times9=65{,}340$ |
| Diffusion training | Adam, lr $10^{-3}$, batch 32, 400 epochs |
| Classifier / SNR head | Adam, lr $10^{-3}$, batch 32 / 64, 200 / 30 epochs; classifier lr halved every 10 batches |

Diffusion targets are *simulated* $\mathbf{R}$ (Eq. 3), never measured; the network predicts the Gaussian noise
added to $\tilde{\mathbf{R}}$, not the sampling error $\hat{\mathbf{R}}-\mathbf{R}$. The classifier is then trained
on covariances recovered from $\hat{\mathbf{R}}$, 90/10 split — so the diffusion model never sees a real
$\hat{\mathbf{R}}$ in training and the classifier never sees a real $\mathbf{R}$. That is why the method is a prior
over a synthetic data manifold, not an estimator with sampling-theoretic guarantees. **Naming:** LDDPM = this
pipeline with the linear schedule; EDDPM = this schedule; all other methods use $\hat{\mathbf{R}}$ directly.

## 7. What is actually new

Four claimed contributions (Sec. 1); two are framing/application, two are method.

| Component | Off-the-shelf | In this paper | New? |
|---|---|---|---|
| Forward/reverse chains, $\varepsilon$-loss, $\mathcal{N}(0,\mathbf{I})$ prior | Ho et al. 2020 (ref. [28]), Eq. 5–10, 23 | identical, symbol $x\to\mathbf{R}$ | No |
| U-Net with timestep embedding | standard DDPM U-Net | 18-stage encoder/decoder, re-skinned | No |
| Drop-diagonal, split real/imag input | Zheng [11], Papageorgiou [16], Merkofer [19] | same, $2M(M-1)$ real vector | No |
| Noise schedule $\beta_t$ | linear (Ho), cosine (Nichol–Dhariwal [30]) | Eq. 14, exponential, $(\upsilon,\gamma)=(10,1.5)$ | **Yes** (c2) |
| Reverse-chain entry point | always $t=H$ | $t^\star$ from a 9-class eigenvalue classifier, Eq. 18–21 | **Yes** (c3) |
| Diffusion for covariance DOA | GAN/CNN covariance reconstruction ([24],[25],[26]) | first diffusion-based covariance recovery for DOA | application (c1, c4) |
| "Feasibility" argument, Eq. 11 | Ottersten et al. [29] | reused, then misread | No — 8(a) |

Honest summary: **one new schedule and one new initialisation rule for the reverse chain, wrapped around an
unmodified DDPM and a new classifier.** Contribution (1), the feasibility analysis that motivates the whole
approach, is not a contribution — it misreads a 1998 result, and it is the weakest part of the paper.

## 8. Critique

### 8(a) The i.i.d. claim is false, and the forward process depends on it

Sec. 3.1 reasons: (1) Eq. 11 holds; (2) $\mathbf{R}$ is Toeplitz with constant diagonal; (3) therefore every entry
of $\Delta$ has mean 0 and **identical variance**; (4) therefore the entries are **independent and identically
distributed**; (5) therefore DDPM, which injects i.i.d. Gaussian noise, is the right denoiser. Step 3 is true. Step
4 is a non sequitur; step 5 is false.

**Step 3, precisely.** The marginal variance is the matching diagonal of the Kronecker covariance,

$$\mathrm{Var}(\Delta_{ij})=\frac{1}{T}\big[\mathbf{R}^{T}\otimes\mathbf{R}\big]_{(ij),(ij)}
=\frac{R_{ii}R_{jj}}{T}=\frac{\big(\sum_k p_k+\sigma^2\big)^2}{T},$$

independent of $(i,j)$ precisely *because* the ULA diagonal is constant. The premise does real work — but only for
the variances.

**Step 4 fails on two independent grounds.**

*Cross-entry correlation.* Off-diagonal blocks are not zero:

$$\mathrm{Cov}\big(\Delta_{ij},\Delta_{kl}\big)=\frac{R_{ik}R_{jl}}{T}\ \neq\ 0 .$$

For a ULA, $R_{ik}=\sum_k p_k e^{j\pi(i-k)\sin\theta_k}+\sigma^2\delta_{ik}$ is nonzero for essentially every pair
with small $|i-k|$, so the $M^2$-dimensional error vector is **strongly coloured**. Toeplitz structure makes this
worse, not better: it is the maximally structured case, not the isotropic one.

*Conjugate symmetry.* $\Delta_{ji}=\overline{\Delta_{ij}}$ **exactly**, hence
$\mathrm{Cov}(\Delta_{ij},\Delta_{ji})=\mathrm{Var}(\Delta_{ij})$ and correlation $+1$: real parts identical,
imaginary parts sign-flipped. Two slots, one random variable — not independent draws.

**When is it isotropic?** Isotropy needs $\mathbf{R}^{T}\otimes\mathbf{R}=c\,\mathbf{I}_{M^2}$. Fix $i\neq j$ and
$l\neq i$; the $(ij),(il)$ entry is $R_{ii}R_{jl}/T$, so isotropy forces $R_{jl}=0$ for all $j\neq l$: $\mathbf{R}$
must be diagonal, and with constant diagonal $\mathbf{R}=c'\mathbf{I}$. That is the **noise-only case** — no
sources, or $\mathrm{SNR}\to-\infty$, the one regime where there is nothing to recover. "Each element is i.i.d."
holds exactly where the method is useless and fails everywhere it is used.

**Why it matters operationally.** Three consequences, in increasing severity.

*The forward process models the wrong noise.* Eq. 12 adds $\mathcal{N}(0,(1-\bar\alpha_t)\mathbf{I})$ to a matrix
whose real sampling error has covariance $\mathbf{R}^{T}\otimes\mathbf{R}/T$. The chain removes white noise from
something never corrupted by white noise, so $\boldsymbol{\varepsilon}_\theta$ must absorb the mismatch — which it
can only do by memorising $\hat{\mathbf{R}}\mapsto\mathbf{R}$ over the training distribution. That is regression in
disguise, and it explains why gains concentrate at low SNR and small separation (Fig. 10), where the prior works.

*The SNR-to-timestep map conflates two different SNRs.* Eq. 19's SNR is $p_s/p_n$ of the *snapshots*; the relevant
quantity for the covariance is the relative distortion

$$\frac{\mathbb{E}\|\Delta\|_F^2}{\|\mathbf{R}\|_F^2}
=\frac{(\mathrm{tr}\,\mathbf{R})^2}{T\|\mathbf{R}\|_F^2}
=\frac{1}{T}\cdot\frac{\big(\sum_i\lambda_i\big)^2}{\sum_i\lambda_i^2},$$

using $\mathbb{E}\|\Delta\|_F^2=\mathrm{tr}(\mathbf{R}^{T}\otimes\mathbf{R})/T=(\mathrm{tr}\,\mathbf{R})^2/T$. For
$M$ equal eigenvalues this is $M/T$, for one dominant eigenvalue $1/T$; with $M=16$, $T=1000$ the covariance NMSE
sits in $[10^{-3},1.6\times10^{-2}]$ whether the array SNR is $-20$ or $+20$ dB. Eq. 20's identification
$\bar\alpha_t/(1-\bar\alpha_t)=10^{\mathrm{SNR}/10}$ is therefore statistically wrong: it hands the covariance chain
an SNR belonging to the snapshot model. The classifier partially rescues it — the eigenvalue profile *is*
informative about $p_s/p_n$ — but the justification does not survive, and $t^\star$ comes out systematically too
large (too noisy a start), buying accuracy at the cost of the compute the trick was invented to save.

*Diffusion is under-motivated, not refuted.* A conditional generative prior over $\mathbf{R}$ can beat
$\hat{\mathbf{R}}$ at low SNR even with a mismatched forward process; what fails is the paper's *reason* for
choosing diffusion. The correct framing is the inverse-problem one (Section D of `LITERATURE.md`):
$\hat{\mathbf{R}}$ is a coloured, heteroscedastic, structured observation of $\mathbf{R}$, so the right tools are
noise-matched forward operators or a diffusion on the SPD manifold (Section A) — not an isotropic DDPM bolted onto a
wishful-i.i.d. argument.

### 8(b) Nothing enforces the structure of a covariance matrix

$\hat{\mathbf{R}}$ is Hermitian PSD by construction and $\mathbf{R}$ is additionally Toeplitz. The method discards
the diagonal, treats the $2M(M-1)$ real slots of both strict triangles as an unstructured image, denoises them and
reassembles a matrix. Nothing enforces:

| Property | Enforced by | Consequence of violation |
|---|---|---|
| $R=R^{H}$ | nothing | complex eigenvalues; no spectral estimator applies |
| $\lambda_{\min}\ge0$ | nothing | MUSIC's noise subspace is not a subspace; $\mathbf{a}^H\mathbf{E}_n\mathbf{E}_n^H\mathbf{a}$ can go negative |
| $R_{ij}=r_{\lvert i-j\rvert}$ | nothing | ULA Vandermonde structure lost; ESPRIT shift invariance invalid |
| $R_{ii}=\sum_k p_k+\sigma^2$ | diagonal dropped | per-sensor power and noise floor unidentifiable |
| low-SNR rank structure | nothing | signal subspace dimension not preserved |

**Hermitian symmetry is broken by the forward process itself.** The chain adds independent $\varepsilon$ to slot
$(i,j)$ and to slot $(j,i)$. In the data $\Delta_{ji}=\overline{\Delta_{ij}}$; in the model
$\varepsilon_{ji}\neq\overline{\varepsilon_{ij}}$, so the recovered matrix satisfies $R_{ji}\neq\overline{R_{ij}}$
almost surely. Even a perfect denoiser leaves a Hermitian defect of the order of the residual off-diagonal noise,
and the input representation doubles the damage by feeding both triangles as independent measurements when they are
one measurement with a sign flip on the imaginary part. A correct input is the $M(M-1)$ strict-upper entries; a
correct head parameterises one triangle and reconstructs the other by conjugation.

**PSD failure is what bites downstream.** MUSIC partitions the spectrum at $M-K$: $\mathbf{E}_n$ = eigenvectors of
the $M-K$ smallest eigenvalues, $P(\theta)=1/(\mathbf{a}^H\mathbf{E}_n\mathbf{E}_n^H\mathbf{a})$ — which presupposes
real, orderable eigenvalues and a genuine noise subspace. With negative eigenvalues the ordering is meaningless,
$\mathbf{E}_n$ can contain a signal direction, and the denominator (an energy, hence nonnegative for a true PSD
matrix) can go negative, making the dB spectrum undefined or producing spurious peaks at the very angles the
classifier is meant to resolve. The paper never reports $\lambda_{\min}$, never shows whether recovery improves or
degrades conditioning, and never runs MUSIC/ESPRIT *on the recovered matrix*: it only feeds a classifier retrained
on recovered matrices. That closes the loop — the classifier adapts to whatever distortion the DDPM introduces, so
the experiment cannot detect it, and the central claim is never tested by the estimators that would expose a bad
covariance.

**Toeplitz is invoked and abandoned.** Sec. 3.1 uses Toeplitz structure to argue the i.i.d. claim; Sec. 3.3.1 throws
the structure away. Projecting onto Toeplitz then PSD is one line, provided here as `toeplitz_project` /
`spd_project` and as the `toeplitz` estimator. A method that does not match that projection's structural guarantees
has to justify the omission.

### 8(c) The evaluation cannot support the conclusions

**No classical covariance baselines.** The comparison set is MUSIC, ESPRIT, CNN, DCNN, DCT-ViT, a classifier-only
variant, ESPRIT-CNN, LDDPM, EDDPM. Not one shrinkage or spectral estimator appears: no Ledoit–Wolf, OAS, analytical
nonlinear shrinkage, Gavish–Donoho singular-value shrinkage, graphical lasso, diagonal loading, Marchenko–Pastur
clipping, or plain Toeplitz+PSD projection. At $M=16$, $T=1000$ the aspect ratio $M/T=0.016$ is benign, and
Ledoit–Wolf/OAS are closed-form, training-free, structurally safe. If a 400-epoch diffusion model trained on 65,340
samples cannot beat a one-line shrinkage estimator the thesis collapses — and the experiment that settles it is
absent. This repository runs exactly that experiment.

**The recovery metric is weak.** Eq. 16 correlates vectorised off-diagonal entries of recovered and true matrices.
Pearson correlation is invariant to affine rescaling, so a method that recovers the *shape* while getting scale,
eigenvalue spread or noise floor wrong scores near 1. It is dominated by the few largest entries (the near-diagonal
lags, also the least noisy), blind to Hermitian/PSD/Toeplitz violations, and silent about the subspace angle that
determines MUSIC's error. It is also the metric used to select $\upsilon,\gamma$ (Fig. 3), so the schedule is tuned
on the same statistic later used to declare the schedule better. NMSE, log-Euclidean distance, subspace angle and
conditioning are each one call away and none is reported.

**No CRLB, no oracle, no attribution.** No comparison to the stochastic Cramér–Rao bound, and — worse for a paper
about denoising — no **oracle** row feeding the true $\mathbf{R}$ to the same classifier. Without it the reported
gap cannot be split into (i) the value of denoising and (ii) the value of the new classifier; LDDPM and EDDPM use a
bespoke 121-output head while the "classifier" baseline is a different network, so the headline comparison confounds
architecture with the diffusion stage. The one number that isolates the effect — same weights, same seeds,
classifier on $\hat{\mathbf{R}}$ vs on recovered $\mathbf{R}$ — is missing.

**Resolution is at the grid limit.** All angles live on a 1° grid and labels are multi-hot on it. Experiment 1
places sources at $5.99°$ and $7.01°$, about one cell apart. The reported RMSEs of 1.12 (classifier), 0.82 (LDDPM),
0.79 (EDDPM) are all on the order of the quantisation floor, and the EDDPM-vs-classifier margin ($0.33°$) is a third
of a cell; Experiment 4 repeats the ordering with the same magnitudes. A 121-cell classifier cannot resolve $0.33°$
— the margin measures how often the peak lands in the neighbouring cell, not sub-degree accuracy. Comparing a grid
classifier against MUSIC/ESPRIT (continuous, super-resolution) on one RMSE axis is not like-for-like, and the paper
does not say how the continuous estimates were quantised, if at all.

**Distribution shift is untested and the test set is the training grid.** Training uses all 7,260 ordered pairs on
the 1° grid; Experiment 4 then sweeps "random grid-on angles" over that same grid with $\Delta\theta=2°$ — a subset
of the training support. No held-out angular region, no $K\neq2$, no $M\neq16$, no non-ULA geometry, no calibrated
or mutually coupled array, no non-Gaussian sources, no real measurements (the data-availability statement says only
that the generation method is described). The non-uniform-noise and coherence experiments vary noise and source
correlation only, both from the same simulator the model was trained on.

**No uncertainty reporting.** 500 Monte Carlo trials per point, one curve per method, no error bars, no confidence
intervals, no paired test, no seed or variance disclosure; differences of $0.03°$ between LDDPM and EDDPM are
presented as a finding with no evidence that they exceed run-to-run noise.

## 9. What this repository adds

**1. The audit (`covdiff.audit`).** Does a "recovery" method return a *valid* covariance matrix? For any matrix:
Hermitian defect $\|\mathbf{R}-\mathbf{R}^H\|_F/\|\mathbf{R}\|_F$, Toeplitz defect
$\|\mathbf{R}-\mathcal{T}(\mathbf{R})\|_F/\|\mathbf{R}\|_F$, $\lambda_{\min}$, $\lambda_{\max}$, fraction of
negative eigenvalues, negative-energy ratio
$\|\sum_{\lambda_i<0}\lambda_i\mathbf{v}_i\mathbf{v}_i^H\|_F/\|\mathbf{R}\|_F$, and condition number.
`run_structure_audit` sweeps this over the SNR grid for every denoiser — the table 8(b) says the paper should have
printed. Spectral metrics use the Hermitian part, with the defect reported separately, so a method cannot hide
asymmetry inside a spectral score.

**2. The missing baselines (`covdiff.baselines`).** Training-free estimators taking $(\hat{\mathbf{R}},T)$: raw;
Ledoit–Wolf shrinkage to $\mu\mathbf{I}$ using the complex Wishart error
$\mathbb{E}\|\hat{\mathbf{R}}-\mathbf{R}\|_F^2=(\mathrm{tr}\,\mathbf{R})^2/T$; OAS; diagonal loading; Toeplitz then
PSD projection; Marchenko–Pastur clipping at $\lambda_+=\sigma^2(1+\sqrt{M/T})^2$; plus an `oracle` returning
$\mathbf{R}$, bounding what any denoiser could achieve. MUSIC and ESPRIT stack on top of each, so the omitted
comparison ("does diffusion beat shrinkage at $-20$ dB?") becomes one call to `make_angle_estimators`.

**3. A protocol that can falsify the claims (`covdiff.eval`).** Seven sweeps — SNR, snapshots, separation,
coherence, non-uniform noise, covariance recovery, structure audit — with fixed ground-truth angles, explicit seeds,
$n_{mc}$ realisations, mean *and* standard deviation, CSV plus a `summary.json` snapshot in `results/`. Recovery is
scored with NMSE, log-Euclidean distance, subspace angle, Pearson-on-off-diagonals (so the paper's own metric is
reproduced, not caricatured), $\lambda_{\min}$ and condition number. The diffusion reimplementation keeps both of
the paper's ideas — exponential schedule (`exponential_beta_schedule`, Eq. 14) and SNR-predicted start
(`SNRClassifier`, `predict_start_timestep`) — while making the failure modes observable instead of absorbing them
into a retrained classifier.

None of this requires the paper to be wrong. It requires it to be *checkable*.

## 10. Fidelity boundary

| Boundary | Content |
|---|---|
| **Exact** | Signal model (Eq. 1–4); the $7{,}260\times9=65{,}340$ training set; $M=16$, $K=2$, 1° grid, $T=1000$, $H=1000$; schedule formula and $(\upsilon,\gamma)=(10,1.5)$; SNR-classifier input and 9-class output; Eq. 23 loss; classifier BCE on 121 multi-hot cells; experiment axes and Monte Carlo counts |
| **Substituted** | U-Net channel widths and pooling stride are not in the text (Fig. 6a only): we use base width 32, multipliers $(1,2,4)$, additive timestep conditioning, GroupNorm/SiLU, under 5M parameters. The Sigmoid-over-9-logits inconsistency is resolved as cross-entropy. Sampler is DDIM (50 steps) for reproducibility, ancestral sampling for fidelity checks; the paper names no sampler |
| **Not checkable** | Absolute RMSE agreement with Figs. 8–10 (plots, no numeric axis data); inference times (hardware unspecified); recovery images of Figs. 12–13, 15, 17 (100-run averages, seeds unstated). We compare *orderings and trends* with the paper's claims and report our own numbers with variances instead of asserting numerical agreement |
