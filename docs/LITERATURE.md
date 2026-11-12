# LITERATURE — an annotated reading map

Twenty-four papers, grouped by the job they do for this project. Every entry was resolved against Crossref or the
arXiv API when this file was written; no reference here is inferred from memory alone.

How to use the map: Section E is the competition (what the paper under study fails to compare against). Section A
is where the mathematics of the object it claims to model — a covariance matrix — is actually taken seriously.
Section D is the framing that makes the problem well posed. Sections B, C and F are context and tooling.

| Section | What it supplies | Where it lands in this repo |
|---|---|---|
| A | geometry of $\mathbf{R}$ (SPD cone, Wishart noise) | redesign direction; `spd_project` edge cases |
| B | prior art and training-free competitors | comparison rows in the `eval` sweeps |
| C | covariance-level diffusion done elsewhere | protocol design for `metrics` / `audit` |
| D | correct statement of $\hat{\mathbf{R}}$ as an observation | `diffusion.model` `start_t`, guidance hooks |
| E | the baselines the paper omits | `covdiff.baselines` |
| F | learning without clean targets | training path for `covdiff.diffusion` |

---

## A. Diffusion on SPD / matrix manifolds

The paper under study treats a covariance matrix as a two-channel image and adds entrywise Gaussian noise. These
seven do the opposite: they respect that $\mathbf{R}$ lives on the cone of Hermitian positive-definite matrices,
where the natural metric is not Euclidean and the natural noise is not isotropic. If you want to fix the method
rather than audit it, start here.

- **Li, Yu, He, Shen, Li, Sun & Lin** — *SPD-DDPM: Denoising Diffusion Probabilistic Models in the Symmetric
  Positive Definite Space*, AAAI 2024. `doi:10.1609/aaai.v38i12.29276`
  Why it matters: the direct drop-in replacement for the paper's DDPM — a forward process defined on SPD matrices
  that stays inside the cone, so every sample is a valid covariance by construction. The paper under study should
  have cited this and did not.
- **De Bortoli, Mathieu, Hutchinson, Thornton, Teh & Doucet** — *Riemannian Score-Based Generative Modelling*,
  NeurIPS 2022. `arXiv:2202.02763`
  Why it matters: the general recipe (forward process from the manifold's Brownian motion, score in the tangent
  space) that makes manifold diffusion more than a metaphor; read §2–3 for the construction, not the proofs.
- **Jo & Hwang** — *Generative Modeling on Manifolds Through Mixture of Riemannian Diffusion Processes*, 2023.
  `arXiv:2310.07216`
  Why it matters: one diffusion per chart, mixed by a learned gate — the practical answer when the manifold has
  regions with different curvature, which is what happens to the SPD cone at low SNR (nearly singular matrices).
- **Thornton, Hutchinson, Mathieu, De Bortoli, Teh & Doucet** — *Riemannian Diffusion Schrödinger Bridge*, 2022.
  `arXiv:2207.03024`
  Why it matters: bridges let you move between two *given* distributions instead of from noise — exactly the shape
  of the problem here (sample covariance → expectation covariance) if you are willing to give up the DDPM scaffold.
- **Brooks, Schwander, Barbaresco, Schneider & Cord** — *Riemannian batch normalization for SPD neural networks*,
  NeurIPS 2019. `arXiv:1909.02414`
  Why it matters: the unglamorous layer that makes SPD networks train; if you build an SPD denoiser you will need
  its Fréchet-mean normalisation, and this is the reference implementation everyone copies.
- **Ju, Kobler, Collas, Kawanabe, Guan & Thirion** — *SPD Matrix Learning for Neuroimaging Analysis:
  Perspectives, Methods, and Challenges*, IEEE TPAMI 2026. `doi:10.1109/TPAMI.2026.3726269`
  Why it matters: a survey of what actually breaks when you learn on covariance matrices (ill-conditioning,
  rank deficiency, small-sample bias) — the failure list in Section 8(b) of `METHOD.md`, in another domain.
- **Mayerhofer** — *Wishart Processes and Wishart Distributions: An Affine Processes Point of View*, 2012.
  `arXiv:1201.6634`
  Why it matters: the honest generative model for a *sample* covariance is Wishart, not Gaussian. This gives the
  matrix-valued process whose stationary law is Wishart — the noise model the paper's Eq. 11 implies and Eq. 12
  ignores.

## B. Array processing / DOA with diffusion

The immediate neighbourhood of the paper. Four entries: two are diffusion-for-DOA, one is a generative emulation
approach, one is the Riemannian take on covariance fitting. Together they show that "first diffusion model for
DOA", claimed in the introduction, is at best a 2025-vintage claim.

- **Qian, Zhou & Shi** — *Denoising Diffusion Model for DOA Estimation*, ICASSP 2026.
  `doi:10.1109/ICASSP55912.2026.11462557`
  Why it matters: the closest contemporaneous work — diffusion applied directly to the DOA problem. Compare its
  noise model and evaluation protocol against the paper under study before accepting any novelty claim.
- **Umra, Ahmed & Sezgin** — *Generative Large-Array Emulation for DOA Estimation in MIMO Radar via Conditional
  Diffusion*, 2026. `arXiv:2609.30562`
  Why it matters: conditional diffusion used to *synthesise* array data for a large aperture — a different use of
  the same machinery, and a useful contrast: generation of data versus recovery of a moment.
- **He, Zhang, Wang, Liu, Wang, Tang & Yu** — *Diff-AgentDOA: Diffusion-agent model for synthetic data generation in
  shallow-water target direction of arrival estimation*, Information Fusion 2026. `doi:10.1016/j.inffus.2026.104371`
  Why it matters: shows the field's current pattern — diffusion as a data amplifier for a downstream estimator,
  which sidesteps the "is the recovered matrix a valid covariance?" question entirely.
- **Picard, Bar & Talmon** — *Riemannian Covariance Fitting for Direction-of-Arrival Estimation*, 2024.
  `arXiv:2404.03401`
  Why it matters: fits the covariance on the manifold instead of denoising its entries; a training-free baseline
  that is philosophically the opposite of this paper and belongs in any honest comparison table.

## C. Covariance-level diffusion in other domains

Two papers that already did what this paper claims to do first — diffusion over spatial/channel covariance
matrices — in EEG and MIMO. Read them for protocol design as much as for method.

- **Ju, Kobler & Guan** — *Score-Based Data Generation for EEG Spatial Covariance Matrices: Towards Boosting BCI
  Performance*, EMBC 2023. `doi:10.1109/EMBC40787.2023.10340899`
  Why it matters: score-based generation of covariance matrices with a downstream classifier as the metric — nearly
  the same architecture of argument as the paper under study, one domain over, and it reports the eigen-spectrum.
- **Chen, Shin & Nallanathan** — *Generative Diffusion Model-Based Variational Inference for MIMO Channel
  Estimation*, IEEE TCOM 2025. `doi:10.1109/TCOMM.2025.3556753`
  Why it matters: treats the covariance as a latent to be inferred rather than an image to be denoised, with a
  variational objective — the statistically literate version of the same idea.

## D. Reframing as an inverse problem

The cleanest way to state what the paper is doing: $\hat{\mathbf{R}}$ is a noisy, coloured observation of
$\mathbf{R}$ under a known-ish measurement operator. Once written that way, the isotropic-DDPM assumption is
visibly the wrong tool, and the literature below supplies the right ones.

- **Chung, Kim, McCann, Klasky & Ye** — *Diffusion Posterior Sampling for General Noisy Inverse Problems*, 2022.
  `arXiv:2209.14687`
  Why it matters: the reference formulation of "diffusion prior + measurement operator", including the
  $\nabla\log p(\mathbf{y}\mid\mathbf{x}_t)$ guidance term that a covariance-aware forward model would supply.
- **Boys, Girolami, Pidstrigach, Reich, Mosca & Akyildiz** — *Tweedie Moment Projected Diffusions for Inverse
  Problems*, 2023. `arXiv:2310.06721`
  Why it matters: replaces DPS's gradient guidance with exact moment matching under a Gaussian likelihood — which
  is precisely the structure of the sampling error in Eq. 11 of the paper under study.
- **Rozet, Andry, Lanusse & Louppe** — *Learning Diffusion Priors from Observations by Expectation Maximization*,
  2024. `arXiv:2405.13712`
  Why it matters: you can learn the prior *from corrupted data alone*; directly relevant, since a real deployment
  has no clean $\mathbf{R}$ to train on, only $\hat{\mathbf{R}}$'s.
- **Bansal, Borgnia, Chu, Li, Kazemi, Huang, Goldblum, Geiping & Goldstein** — *Cold Diffusion: Inverting Arbitrary
  Image Transforms Without Noise*, NeurIPS 2022. `arXiv:2208.09392`
  Why it matters: proves the "noise" in a diffusion model need not be Gaussian — any degrading operator with a
  usable inverse works. This is the licence to use a Wishart/coloured forward process instead of white noise.

## E. Classical covariance estimation (the baselines the paper omits)

If you read only one section of this map, read this one. Every entry is training-free, closed-form or convex, and
most are one line of NumPy. Any of them beating a 400-epoch diffusion model at $M=16$, $T=1000$ would end the
paper's argument; that experiment is the reason this repository exists.

- **Ledoit & Wolf** — *A well-conditioned estimator for large-dimensional covariance matrices*, Journal of
  Multivariate Analysis 2004. `doi:10.1016/S0047-259X(03)00096-4`
  Why it matters: the canonical shrinkage estimator, and the one implemented here (`covdiff.baselines.ledoit_wolf`)
  with the complex-Wishart error term $(\mathrm{tr}\,\mathbf{R})^2/T$ that the paper's own Eq. 11 implies.
- **Ledoit & Wolf** — *Analytical nonlinear shrinkage of large-dimensional covariance matrices*, Annals of
  Statistics 2020. `doi:10.1214/19-AOS1921`
  Why it matters: shrinks each eigenvalue by a different amount, which is what a *coloured* sampling error demands;
  the natural competitor to a learned denoiser in the eigenbasis.
- **Gavish & Donoho** — *Optimal Shrinkage of Singular Values*, IEEE Transactions on Information Theory 2017.
  `doi:10.1109/TIT.2017.2653801`
  Why it matters: the optimal hard threshold for singular values under unknown noise level — the principled version
  of the paper's "drop the small eigenvalues" instinct, and a direct check on whether diffusion adds anything.
- **Friedman, Hastie & Tibshirani** — *Sparse inverse covariance estimation with the graphical lasso*,
  Biostatistics 2008. `doi:10.1093/biostatistics/kxm045`
  Why it matters: enforces sparsity *and* positive definiteness by construction — the structural guarantees
  Section 8(b) of `METHOD.md` shows the diffusion pipeline lacks.
- **Bun, Bouchaud & Potters** — *Cleaning large correlation matrices: tools from random matrix theory*, Physics
  Reports 2017. `doi:10.1016/j.physrep.2016.10.005`
  Why it matters: the theory of exactly how far a sample covariance sits from the truth at finite $T/M$, including
  the Marchenko–Pastur edge used by `eigen_clip`. Read §1–3 to understand why the paper's i.i.d. premise is not a
  technicality.

## F. Training without clean targets

Two methodological escape hatches, both relevant because this repository cannot obtain measured $\mathbf{R}$ either.

- **Batson & Royer** — *Noise2Self: Blind Denoising by Self-Supervision*, ICML 2019. `arXiv:1901.11365`
  Why it matters: gives a way to train a denoiser with no ground truth, using only a partition of the observation —
  the honest alternative to the paper's simulate-clean-then-corrupt protocol.
- **Tashiro, Song, Song & Ermon** — *CSDI: Conditional Score-based Diffusion Models for Probabilistic Time Series
  Imputation*, NeurIPS 2021. `arXiv:2107.03502`
  Why it matters: the reference design for conditional diffusion where part of the input is *observed and must be
  preserved* — the pattern to copy if the diagonal and the Hermitian structure are to be conditioned on rather than
  discarded.

---

## Reading order for a newcomer

Six steps, roughly one working day, ending with the tools to judge this paper independently.

1. **Ledoit & Wolf 2004** (Section E) — 45 min. Start with the baseline. You cannot evaluate a covariance-recovery
   method until you know what a one-line shrinkage estimator achieves at your $M/T$.
2. **Bun, Bouchaud & Potters 2017**, §1–3 (Section E) — 1 h 15 min. Why $\hat{\mathbf{R}}-\mathbf{R}$ is not white
   noise, quantitatively. This is the fact the paper under study misstates.
3. **Ju, Kobler & Guan 2023** (Section C) — 30 min. The closest prior art: diffusion over covariance matrices with
   a downstream task metric. Note what they report that this paper does not.
4. **Li et al., SPD-DDPM 2024** (Section A) — 45 min. What it looks like when the diffusion respects the SPD
   geometry; keep it as the redesign target.
5. **Chung et al., DPS 2022** (Section D) — 1 h. The inverse-problem framing: prior + measurement operator +
   guidance. After this, "denoise an image-like matrix" stops looking like the natural formulation.
6. **Rozet et al. 2024** (Section D) — 45 min. How to learn the prior when only corrupted observations exist — the
   situation any real deployment is in, and the one the paper's training recipe assumes away.

Total ≈ 5 hours of reading. Steps 1–2 alone are enough to reproduce the critique in `METHOD.md` §8(a) and §8(c);
steps 3–6 are what you need to design something better.
