# Method summary (pseudocode)

A one-page, paper-style description of what `acoustic-gap` computes. Full
derivations and references are in [THEORY.md](THEORY.md); the code map is in
[ARCHITECTURE.md](ARCHITECTURE.md).

### Notation

| symbol | meaning |
|---|---|
| $\mathcal{D}_r,\ \mathcal{D}_s$ | real and simulated audio datasets (directories) |
| $c \in \mathcal{C}$ | acoustic condition (e.g. `clean`, `reverb-only`, `noise-only`, `mic-*`) |
| $\phi_b(\cdot)$ | embedding backbone $b$ (PANN/VGGish = content-invariant; WavLM x-vector = channel-sensitive) |
| $X_b^{c},\ Y_b^{c}\subset\mathbb{R}^d$ | utterance embeddings for real / sim, backbone $b$, condition $c$ |
| $D\in\{\text{FAD},\text{MMD},W\}$ | a distribution distance (lower = smaller gap) |

---

## Algorithm 1 — Acoustic domain-gap estimation

```text
Require: real dir D_r, sim dir D_s, config (backbones B, metrics M,
         window w, hop h, sample rate sr, silence thr τ)
Ensure:  per-(backbone, condition, metric) gaps + largest contributor

 1: # ---- Preprocess: build one segment manifest for both datasets ----
 2: S ← ∅
 3: for (dataset, root) in {(real, D_r), (sim, D_s)} do
 4:     for file f in root do
 5:         x ← resample(to_mono(load(f)), sr)          # common rate, mono
 6:         c ← condition_of(f)                          # from sub-directory name
 7:         for each window (w, hop) segment g of x do
 8:             if rms_dBFS(g) < τ: continue              # drop silence
 9:             S ← S ∪ {(dataset, c, source=f, audio=peaknorm(g))}
10:
11: # ---- For every backbone: embed, pool, measure ----
12: results ← ∅
13: for backbone b in B do
14:     E ← φ_b(S.audio)                                 # segment embeddings (N×d)
15:     (U, meta) ← Pool(E, S)                            # → utterance embeddings
16:     if config.disentangle: U ← Residualize(U, meta)   # strip content (Alg. 2)
17:
18:     for c in {OVERALL} ∪ conditions(meta) do
19:         X ← U[ meta.dataset=real ∧ meta.cond∈c ]      # real cloud
20:         Y ← U[ meta.dataset=sim  ∧ meta.cond∈c ]      # sim  cloud
21:         if |X|=0 or |Y|=0: continue                   # need both sides
22:         for metric D in M do
23:             results[b,c,D] ← D(X, Y)                   # Alg. 3–5
24:
25: # ---- Diagnose ----
26: (b*, D*) ← headline backbone/metric (default WavLM, Wasserstein)
27: c* ← argmax_{c ≠ OVERALL} results[b*, c, D*]           # largest contributor
28: return results, c*
```

**Pooling (line 15).** Mean pooling is the default; it estimates the expected
segment embedding of an utterance with $T$ segments,

$$
\bar\phi \;=\; \frac{1}{T}\sum_{t=1}^{T}\phi_t \;\approx\; \mathbb{E}[\phi].
$$

(An untrained, seeded GRU summary is the optional alternative.)

---

## Algorithm 2 — Residualize (optional content disentanglement)

Regress the content label out of the (channel-sensitive) embeddings and keep the
residual — the component **not** linearly predictable from content:

$$
W^\star=(C^\top C+\alpha I)^{-1}C^\top E,
\qquad
\tilde E = E - C\,W^\star,
$$

where $C$ is the one-hot content-label matrix and $E$ the embeddings. Return $\tilde E$.

---

## Distances $D(X,Y)$ — real cloud $X$ vs sim cloud $Y$

All three are **lower-is-better** and estimate a distance between the underlying
distributions $P$ (real) and $Q$ (sim).

### Algorithm 3 — FAD (Fréchet distance of Gaussians)

Fit a Gaussian to each cloud and take the closed-form 2-Wasserstein distance
between them:

$$
\mathrm{FAD}(X,Y)=\lVert\mu_X-\mu_Y\rVert_2^2
+\operatorname{tr}\!\Big(\Sigma_X+\Sigma_Y-2(\Sigma_X\Sigma_Y)^{1/2}\Big),
$$

with $(\mu_X,\Sigma_X),(\mu_Y,\Sigma_Y)$ the empirical mean/covariance.

### Algorithm 4 — MMD (RBF kernel, unbiased)

$$
\widehat{\mathrm{MMD}}^2
=\frac{1}{m(m\!-\!1)}\!\sum_{i\ne j}\!k(x_i,x_j)
+\frac{1}{n(n\!-\!1)}\!\sum_{i\ne j}\!k(y_i,y_j)
-\frac{2}{mn}\sum_{i,j}k(x_i,y_j),
$$

$$
k(x,y)=\exp\!\Big(\!-\tfrac{\lVert x-y\rVert^2}{2\sigma^2}\Big),
\qquad
\sigma=\sqrt{\tfrac12\,\operatorname{median}\{\lVert z_i-z_j\rVert^2\}}\ \text{ (median heuristic).}
$$

### Algorithm 5 — Wasserstein

Per-dimension 1-D $W_1$ (exact, via CDFs) **and** a multivariate estimate:

$$
W_1^{\text{dim}}(X,Y)=\frac1d\sum_{j=1}^{d}\int_{\mathbb R}\big|F_{X_j}(t)-F_{Y_j}(t)\big|\,dt,
$$

$$
\underbrace{\mathrm{SW}_1(X,Y)=\mathbb{E}_{\theta\sim\mathcal U(S^{d-1})}\!\big[W_1(\theta^\top X,\theta^\top Y)\big]}_{\text{sliced (default, scalable)}}
\quad\text{or}\quad
\underbrace{W_2(X,Y)=\Big(\min_{\pi\in\Pi}\textstyle\sum_{ij}\pi_{ij}\lVert x_i-y_j\rVert^2\Big)^{1/2}}_{\text{exact EMD via POT}} .
$$

---

**Reading the output.** For each backbone $b$ and metric $D$: `results[b, OVERALL, D]`
is the headline gap, `results[b, c, D]` is the per-condition breakdown, and $c^\*$
(line 27) is the acoustic condition your simulator most needs to fix.
