# The math behind `acoustic-gap`

This document explains the theory the toolkit rests on: **what "acoustic domain
gap" means formally, why we measure it in learned embedding spaces, and how each
distance (FAD, MMD, Wasserstein) is defined, estimated, and interpreted.** Every
section points at the code that implements it, and all referenced papers are
linked in [§8](#8-references).

> **Notation.** We compare a *real* dataset and a *simulated* dataset. After
> preprocessing and embedding, each dataset is a cloud of vectors in
> $\mathbb{R}^d$: $X=\{x_1,\dots,x_m\}$ (real) drawn i.i.d. from a distribution
> $P$, and $Y=\{y_1,\dots,y_n\}$ (sim) drawn from $Q$. Every metric below is an
> estimate of a **distance between the distributions $P$ and $Q$**, and every
> metric is **lower-is-better** (0 ⇔ indistinguishable).

---

## 1. Problem framing: domain gap as a distribution distance

A "domain gap" is the statement that real and simulated audio are drawn from
*different distributions*. If we had a feature map $\phi:\text{audio}\to\mathbb{R}^d$
that captured the acoustic properties we care about, the gap is any statistical
distance $D\big(\phi_\# P,\ \phi_\# Q\big)$ between the pushforward distributions
of embeddings.

Two design choices follow:

1. **We measure in embedding space, not on raw waveforms.** Raw audio lives in a
   space where perceptually irrelevant differences (phase, exact sample offset)
   dominate Euclidean geometry. A pretrained encoder maps audio to a space where
   distance tracks *acoustic* similarity. This is the same rationale as FID for
   images (Heusel et al. 2017) and FAD for audio (Kilgour et al. 2019).
2. **We use several distances, not one.** FAD, MMD and Wasserstein make different
   assumptions (Gaussianity, RKHS mean-embeddings, optimal transport). Agreement
   across them is evidence the gap is real; disagreement is diagnostic.

The theoretical justification for "distribution distance ⇒ transfer difficulty"
is the domain-adaptation bound of Ben-David et al. (2010): the target-domain error
of a model trained on the source domain is bounded by the source error **plus a
divergence between the two domains' distributions**. Shrinking that divergence
(the sim-to-real gap) is exactly what improving a simulator should do.

---

## 2. Embedding spaces (the feature map $\phi$)

The choice of $\phi$ decides *what kind of gap you can see*. The toolkit runs two
complementary backbones (`acoustic_gap/features/`), by design:

### 2.1 Content-invariant view — PANN / VGGish

**VGGish** (Hershey et al. 2017) and **PANNs** (Kong et al. 2020) are CNNs trained
for audio *tagging* on **AudioSet** (Gemmeke et al. 2017). Trained to recognize
what sound is present, they learn a representation of general spectro-temporal
*texture* that is largely invariant to fine linguistic content. This is the
backbone FAD was originally defined on, and it answers: **"does this simulated
clip sound like a plausible real-world sound?"** Implemented in
`features/panns.py` via the `frechet_audio_distance` package (PANN Cnn14).

### 2.2 Channel-sensitive view — WavLM x-vector

**WavLM** (Chen et al. 2022) is a self-supervised speech model; the
`*-sv` variant adds an **x-vector** speaker-verification head (Snyder et al.
2018). Because it is trained to tell *recordings* apart, its embeddings are highly
sensitive to the **recording chain**: microphone frequency response, room
reverberation, and noise floor — precisely the factors that separate real
recordings from a simulator. It answers: **"does the sim reproduce the real
recording conditions?"** Implemented in `features/wavlm.py` (loaded fully offline
from a local snapshot).

### 2.3 Why both

The content-invariant and channel-sensitive views *disentangle* the gap by
construction. A large PANN/VGGish gap with a small WavLM gap suggests the *content
distribution* differs; a small PANN gap with a large WavLM gap suggests the
content is fine but the *channel* (mic/reverb/noise) is wrong. The per-condition
breakdown (§6) then localizes which channel factor dominates.

A backbone is a `FeatureExtractor` (`features/base.py`), so any encoder with a
`embed_batch` method is swappable without touching the rest of the pipeline.

---

## 3. Pooling: from frames to a fixed-length utterance vector

Backbones emit one vector per short segment. To compare *utterances* we aggregate
the segment vectors of each clip (`acoustic_gap/pooling.py`).

- **Mean pooling (default).** The utterance embedding is
  $\bar{\phi}=\frac{1}{T}\sum_{t=1}^{T}\phi_t$. This is the Monte-Carlo estimate of
  the expected segment embedding $\mathbb{E}[\phi]$; it is unbiased, permutation-
  invariant, and stable. For most acoustic-channel statistics (which are roughly
  stationary within a clip) the mean is a sufficient summary.
- **RNN summarizer (optional).** An **untrained, seeded** (bi)GRU reads the ordered
  segment sequence and returns its final hidden state. Because we have no labels to
  train it, it is used purely as a fixed, reproducible non-linear projection (a
  "reservoir") that can capture *temporal* structure the mean discards. Assumption
  stated plainly: this is an alternative fixed transform, not a learned pooler.

---

## 4. Content disentanglement (optional residualization)

The channel-sensitive backbone still leaks *some* content information. To isolate
the channel/domain signal we optionally **regress the content out** of the
embeddings (`pooling.disentangle_content`).

Given a one-hot content label matrix $C\in\{0,1\}^{N\times k}$ and embeddings
$E\in\mathbb{R}^{N\times d}$, we fit a **ridge linear probe** and keep the residual:

$$
W^\star=\arg\min_W \lVert E - CW\rVert_F^2 + \alpha\lVert W\rVert_F^2
      =(C^\top C+\alpha I)^{-1}C^\top E,
\qquad
\tilde{E}=E-CW^\star .
$$

The residual $\tilde{E}$ is, by the normal equations, the component of the
embedding **orthogonal to everything the content label can linearly predict** —
this is the geometric content of the Frisch–Waugh–Lovell "partialling-out"
theorem. Whatever gap remains in $\tilde{E}$ is not explainable by content class,
so it is attributable to the acoustic channel.

**Tradeoff (stated):** a *linear* probe removes only linearly-decodable content.
It is closed-form, deterministic, and dependency-light, and it is the standard
first-order correction (cf. iterative nullspace projection / INLP, Ravfogel et al.
2020, for the non-linear generalization we deliberately do **not** take here).

---

## 5. Distribution distances

All three metrics are implemented in `acoustic_gap/distances.py` and orchestrated
by `compute_distances`. Each is lower-is-better and each estimates a different
functional of $(P,Q)$.

### 5.1 FAD — Fréchet distance between Gaussians

FAD (Kilgour et al. 2019), like FID (Heusel et al. 2017), fits a single Gaussian
to each embedding cloud and takes the **2-Wasserstein distance between those
Gaussians**, which has a closed form (Dowson & Landau 1982):

$$
d_{\mathrm{F}}^2
= \lVert \mu_r-\mu_s\rVert_2^2
+ \operatorname{tr}\!\Big(\Sigma_r+\Sigma_s-2\big(\Sigma_r\Sigma_s\big)^{1/2}\Big),
$$

where $(\mu_r,\Sigma_r)$ and $(\mu_s,\Sigma_s)$ are the empirical mean/covariance
of the real and sim embeddings. The matrix square root is computed with
`scipy.linalg.sqrtm`, with an $\varepsilon I$ regularizer for the rank-deficient
case (`frechet_distance`).

- **Strength:** cheap, captures first- and second-order (mean + covariance)
  differences; directly comparable to the wider FAD/FID literature.
- **Caveat:** it assumes a Gaussian in embedding space and is therefore *blind to
  higher moments*; the estimator is also known to be **biased upward at small
  sample sizes** (fewer clips ⇒ inflated FAD), so compare like-sized sets. The
  `frechet_audio_distance` directory number (`fad_via_frechet_audio_distance`) is provided as the
  "canonical" reference computation.

### 5.2 MMD — kernel two-sample distance

Maximum Mean Discrepancy (Gretton et al. 2012) embeds each distribution as its
**mean element in a reproducing-kernel Hilbert space** $\mathcal{H}$ and measures
the distance between those means:

$$
\mathrm{MMD}(P,Q)=\big\lVert \mathbb{E}_{x\sim P}[\,k(x,\cdot)\,]-\mathbb{E}_{y\sim Q}[\,k(y,\cdot)\,]\big\rVert_{\mathcal H}.
$$

For a **characteristic** kernel (the Gaussian RBF is one), $\mathrm{MMD}(P,Q)=0
\iff P=Q$ — so MMD sees *all* moments, not just the first two. We use the
**unbiased** estimator of $\mathrm{MMD}^2$ (`mmd_rbf`):

$$
\widehat{\mathrm{MMD}}^2_u
=\frac{1}{m(m\!-\!1)}\!\sum_{i\ne j}\! k(x_i,x_j)
+\frac{1}{n(n\!-\!1)}\!\sum_{i\ne j}\! k(y_i,y_j)
-\frac{2}{mn}\sum_{i,j} k(x_i,y_j),
$$

with the RBF kernel $k(x,y)=\exp\!\big(-\lVert x-y\rVert^2/2\sigma^2\big)$.

**Bandwidth — the median heuristic (default).** We set $\sigma$ from the pooled
sample so the kernel operates at the data's natural scale:
$\sigma=\sqrt{\operatorname{median}\{\lVert z_i-z_j\rVert^2\}/2}$ over pairs of
points $z$. This is the standard parameter-free choice (discussed in Gretton et
al. 2012; analyzed by Garreau et al. 2017); a fixed $\sigma$ is configurable when
you want a specific length scale.

- **Strength:** distribution-free, sensitive to all moments, no Gaussian
  assumption.
- **Caveat:** the *value* depends on the kernel/bandwidth, so MMD numbers are
  comparable across conditions **only under a fixed bandwidth policy** (which is
  why the policy is explicit in config). Naïve estimation is $O((m+n)^2)$, so
  inputs are subsampled to `max_samples`.

### 5.3 Wasserstein — optimal transport

The $p$-Wasserstein distance is the minimum "cost to morph" one distribution into
the other (Villani; Peyré & Cuturi 2019):

$$
W_p(P,Q)=\Big(\inf_{\pi\in\Pi(P,Q)} \int \lVert x-y\rVert^p\, d\pi(x,y)\Big)^{1/p},
$$

where $\Pi(P,Q)$ is the set of couplings with the given marginals. Unlike FAD/MMD
it is a true metric with an interpretable transport meaning. We report two forms:

**(a) Per-dimension 1-D Wasserstein (`per_dimension_wasserstein`).** In 1-D,
$W_1$ has a closed form via the CDFs,
$W_1(P,Q)=\int_{\mathbb R}\lvert F_P(t)-F_Q(t)\rvert\,dt$, computed exactly by
`scipy.stats.wasserstein_distance`. We average it over embedding dimensions. This
is a fast, robust **marginal** gap that ignores cross-dimension coupling.

**(b) Multivariate Wasserstein.** Two backends capture the joint geometry:

- **Sliced-Wasserstein (default, `sliced_wasserstein`).** Average the 1-D $W_1$ of
  random 1-D projections (Rabin et al. 2011; Bonneel et al. 2015; Kolouri et al.
  2019):
  $\mathrm{SW}_1(P,Q)=\mathbb{E}_{\theta\sim\mathcal U(S^{d-1})}\big[W_1(\theta_\#P,\theta_\#Q)\big]$,
  estimated with `sliced_n_projections` directions. Cost $O(L\,n\log n)$ — it
  **scales to large embedding sets**, which is why it is the default.
- **Exact EMD via POT (`wasserstein_pot`).** Solves the discrete
  optimal-transport LP (the Earth Mover's Distance; Rubner et al. 2000) with
  `ot.emd2` (Flamary et al. 2021) on uniform marginals and squared-Euclidean cost,
  returning $W_2$. Exact but $O(n^3\log n)$, so inputs are subsampled to
  `max_samples`. (Entropic/Sinkhorn OT, Cuturi 2013, is the faster approximate
  alternative if you swap the solver.)

---

## 6. Per-condition decomposition & the "largest contributor"

A single top-level score cannot tell you *why* the sim is off. The manifest
carries a **condition label per segment** (inferred from sub-directory names —
`clean/`, `reverb-only/`, `noise-only/`, `mic-*/`; see `preprocessing.py`), so the
pipeline computes every distance **once overall and once per condition**
(`pipeline._run_backbone`).

The reporter then flags the condition with the **largest** headline distance
(`reporting._largest_contributor`), by default WavLM × Wasserstein — the
channel-sensitive backbone under the transport metric. Because conditions are
matched between real and sim, the condition with the biggest sim-to-real distance
is the acoustic factor your simulator most needs to fix. This turns an aggregate
score into an **actionable diagnosis**.

---

## 7. Reading the numbers (caveats in one place)

- **Lower is better, always.** 0 ⇔ the sim distribution is indistinguishable from
  real *in that embedding space*.
- **Scale is not absolute.** Values depend on the backbone (and, for MMD, the
  bandwidth). Compare *relatively* — across conditions, or before/after a
  simulator change — not against an external threshold.
- **Match sample sizes.** FAD especially is biased upward for small $N$; compare
  real/sim sets of similar size, and prefer the sliced/MMD metrics when $N$ is
  small.
- **Cross-metric agreement is signal.** If FAD, MMD, and Wasserstein all rank the
  same condition worst, that ranking is robust to any single metric's assumptions.

---

## 8. References

**Metric foundations**
- Heusel, Ramsauer, Unterthiner, Nessler, Hochreiter. *GANs Trained by a Two
  Time-Scale Update Rule Converge to a Local Nash Equilibrium* (FID). NeurIPS 2017.
  https://arxiv.org/abs/1706.08500
- Kilgour, Zuluaga, Roblek, Sharifi. *Fréchet Audio Distance: A Reference-Free
  Metric for Evaluating Music Enhancement Algorithms.* Interspeech 2019.
  https://arxiv.org/abs/1812.08466
- Gui, Gamper, Braun, Emmanouilidou. *Adapting Frechet Audio Distance for
  Generative Music Evaluation* (the `fadtk` toolkit). ICASSP 2024.
  https://arxiv.org/abs/2311.01616
- Dowson & Landau. *The Fréchet distance between multivariate normal
  distributions.* J. Multivariate Analysis, 1982.
  https://doi.org/10.1016/0047-259X(82)90077-X
- Gretton, Borgwardt, Rasch, Schölkopf, Smola. *A Kernel Two-Sample Test.* JMLR
  2012. https://jmlr.org/papers/v13/gretton12a.html
- Garreau, Jitkrittum, Kanagawa. *Large sample analysis of the median heuristic.*
  2017. https://arxiv.org/abs/1707.07269

**Optimal transport**
- Peyré & Cuturi. *Computational Optimal Transport.* 2019.
  https://arxiv.org/abs/1803.00567
- Rubner, Tomasi, Guibas. *The Earth Mover's Distance as a Metric for Image
  Retrieval.* IJCV 2000. https://doi.org/10.1023/A:1026543900054
- Bonneel, Rabin, Peyré, Pfister. *Sliced and Radon Wasserstein Barycenters of
  Measures.* J. Math. Imaging Vis. 2015.
  https://doi.org/10.1007/s10851-014-0506-3
- Kolouri, Nadjahi, Simsekli, Badeau, Rohde. *Generalized Sliced Wasserstein
  Distances.* NeurIPS 2019. https://arxiv.org/abs/1902.00434
- Cuturi. *Sinkhorn Distances: Lightspeed Computation of Optimal Transport.*
  NeurIPS 2013. https://arxiv.org/abs/1306.0895
- Flamary et al. *POT: Python Optimal Transport.* JMLR 2021.
  https://jmlr.org/papers/v22/20-451.html

**Audio / speech embeddings**
- Hershey et al. *CNN Architectures for Large-Scale Audio Classification*
  (VGGish). ICASSP 2017. https://arxiv.org/abs/1609.09430
- Gemmeke et al. *AudioSet: An ontology and human-labeled dataset for audio
  events.* ICASSP 2017. https://doi.org/10.1109/ICASSP.2017.7952261
- Kong, Cao, Iqbal, Wang, Wang, Plumbley. *PANNs: Large-Scale Pretrained Audio
  Neural Networks for Audio Pattern Recognition.* IEEE/ACM TASLP 2020.
  https://arxiv.org/abs/1912.10211
- Chen et al. *WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack
  Speech Processing.* IEEE JSTSP 2022. https://arxiv.org/abs/2110.13900
- Snyder, Garcia-Romero, Sell, Povey, Khudanpur. *X-vectors: Robust DNN Embeddings
  for Speaker Recognition.* ICASSP 2018.
  https://doi.org/10.1109/ICASSP.2018.8461375

**Domain gap / feature disentanglement**
- Ben-David, Blitzer, Crammer, Kulesza, Pereira, Vaughan. *A theory of learning
  from different domains.* Machine Learning 2010.
  https://doi.org/10.1007/s10994-009-5152-4
- Ravfogel, Elazar, Gonen, Twiton, Goldberg. *Null It Out: Guarding Protected
  Attributes by Iterative Nullspace Projection* (INLP). ACL 2020.
  https://arxiv.org/abs/2004.07667
