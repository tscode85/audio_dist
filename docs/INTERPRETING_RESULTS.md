# Interpreting the metrics

This guide explains how to read the toolkit's output — what each number means,
what scale it lives on, what you may and may **not** compare, and how to turn the
numbers into a decision. It is written around a real example run:

```
Top-level scores (lower = smaller gap):
  panns    fad              27.6234
  panns    mmd              0.3303
  panns    wasserstein_1d   0.0385
  panns    wasserstein_mv   0.0786
  panns    wasserstein      0.0786
  wavlm    fad              25.5355
  wavlm    mmd              0.3894
  wavlm    wasserstein_1d   0.1445
  wavlm    wasserstein_mv   0.1702
  wavlm    wasserstein      0.1702

>>> Largest domain-gap contributor: 'eval_val_seed1337' (wavlm/wasserstein = 0.1702)
```

The maths behind each metric is in [THEORY.md](THEORY.md); this document is about
**reading** them.

---

## 0. Three golden rules (read these first)

1. **Every metric is lower-is-better and unbounded-above** (except MMD, see §2).
   `0` means "indistinguishable"; there is **no fixed "good" threshold**. A number
   is only meaningful *relative to a reference*.
2. **Never compare magnitudes across backbones.** `panns fad 27.6` vs
   `wavlm fad 25.5` does **not** mean the PANN gap is bigger — the two live in
   different embedding spaces (PANN is 2048-dim, WavLM x-vector is 512-dim, with
   totally different coordinate scales). Compare *within* a backbone, or use the
   one partially-normalised metric (MMD) with care.
3. **Calibrate with a real-vs-real baseline** (§6). Without it you cannot tell
   whether `fad = 27` is "large" or "just finite-sample noise". This is the single
   most important thing missing from the run above.

---

## 1. What the rows are

Each `(backbone, metric)` row is a distance between the **real** and **simulated**
utterance-embedding distributions in that backbone's space.

| backbone | what it "sees" | dim |
|---|---|---|
| `panns` | **content / texture** — is it a plausible real-world sound? | 2048 |
| `wavlm` | **recording channel** — mic response, reverb, noise floor | 512 |

The three metric families (per backbone):

| row | metric | what it captures |
|---|---|---|
| `fad` | Fréchet Audio Distance | difference in **mean + covariance** (Gaussian approx.) |
| `mmd` | kernel Maximum Mean Discrepancy (RBF, MMD²) | difference in **all moments** (distribution shape) |
| `wasserstein_1d` | mean per-dimension 1-D Wasserstein-1 | **marginal** difference, averaged over dims |
| `wasserstein_mv` | sliced multivariate Wasserstein-1 | **joint** difference incl. cross-dimension coupling |
| `wasserstein` | *alias of* `wasserstein_mv` | the headline Wasserstein number |

> Note the three "wasserstein" rows: `wasserstein` is simply a copy of
> `wasserstein_mv`. So you really have **two** Wasserstein views — marginal
> (`_1d`) and joint (`_mv`).

---

## 2. Metric-by-metric

### FAD — Fréchet Audio Distance  (`panns 27.6`, `wavlm 25.5`)

- **Definition:** `‖μ_r−μ_s‖² + tr(Σ_r+Σ_s−2(Σ_rΣ_s)^½)` — treats each cloud as
  one Gaussian and measures the distance between them.
- **Scale:** unbounded; grows with embedding dimensionality and variance (the
  `tr(Σ_r+Σ_s)` term is a sum over dimensions). This is exactly why **FAD is not
  comparable across backbones**. A 2048-dim space will tend to produce larger FAD
  than a 512-dim space for the *same* relative gap.
- **Reading your run:** `27.6` (PANN) and `25.5` (WavLM) are in the normal range
  for these encoders, but on their own they say only "the means/covariances
  differ." Whether that is a lot depends on the real-vs-real baseline (§6).
- **Sensitive to:** first- and second-order shifts. **Blind to:** higher moments,
  multi-modal structure. **Bias:** inflated at small sample counts — with your
  ~96k utterances this bias is negligible (good).

### MMD — kernel two-sample distance  (`panns 0.330`, `wavlm 0.389`)

- **Definition:** squared RKHS distance between the distributions' kernel mean
  embeddings, RBF kernel, **median-heuristic bandwidth** (the bandwidth adapts to
  each space's scale).
- **Scale:** bounded, roughly `[0, ~2]`. `0` ⇔ identical; the value rises as the
  two clouds separate. As a rule of thumb (bandwidth-dependent): `<~0.02` ≈
  indistinguishable, `~0.1–0.5` ≈ **clearly different but strongly overlapping**,
  `→1+` ≈ nearly disjoint.
- **Why it matters here:** because the median heuristic **self-normalises** to each
  space, MMD is the **least unfair metric to compare across backbones**. Your
  `wavlm 0.389 > panns 0.330` is therefore a (cautious) signal that the
  **channel** gap is *somewhat larger* than the **content** gap — the classic
  sim-to-real pattern: the simulator gets the content roughly right but the
  recording chain slightly wrong. Both values are moderate — a two-sample test
  would confidently reject "sim = real," but the distributions still overlap a lot.
- **Sensitive to:** all moments (characteristic kernel). **Depends on:** the
  bandwidth policy — only compare MMD numbers computed the same way.

### Wasserstein — optimal transport  (`_1d` and `_mv`)

- **`wasserstein_1d`** (`panns 0.0385`, `wavlm 0.1445`): average, over embedding
  dimensions, of the 1-D Wasserstein-1 distance between the marginals. It ignores
  correlations between dimensions.
- **`wasserstein_mv`** (`panns 0.0786`, `wavlm 0.1702`): sliced-Wasserstein-1 over
  random projections — captures **joint** structure the marginals miss.
- **Scale:** in the embedding's own coordinate units → **relative only, and not
  comparable across backbones.**
- **Reading the `_mv / _1d` ratio** (a within-backbone diagnostic):
  - PANN: `0.0786 / 0.0385 ≈ 2.0` → the joint gap is about double the marginal
    average, i.e. a meaningful chunk of the content gap lives in **correlations
    between dimensions**, not in any single marginal.
  - WavLM: `0.1702 / 0.1445 ≈ 1.18` → the channel gap is already visible in the
    **marginals**; projections add little. Loosely, the channel discrepancy is
    "spread evenly" across coordinates (consistent with a broadband effect like a
    mic EQ tilt or a different noise floor), whereas the content discrepancy is
    more "relational."

---

## 3. Do the metrics agree?

Within each backbone, cross-metric agreement is the robustness check — if FAD, MMD
and Wasserstein all point the same way, the ranking is not an artefact of one
metric's assumptions.

- **WavLM:** MMD (0.389) and Wasserstein (0.170) are both clearly non-zero and both
  larger than PANN's shape-metrics → **consistent**: a real, moderate channel gap.
- **PANN:** MMD (0.330) clearly non-zero, Wasserstein smaller → also a real content
  gap, a touch smaller on the transport metric.
- **The one apparent disagreement** — PANN FAD (27.6) > WavLM FAD (25.5) while
  MMD/Wasserstein say the opposite — is **not** a real disagreement: cross-backbone
  FAD magnitudes are not comparable (Rule 2). Ignore it.

**Net read of this run:** the simulator differs from real audio by a *moderate,
clearly-detectable* amount on **both** axes, with the **recording-channel**
(WavLM) gap modestly larger than the **content** (PANN) gap. That points you at
mic/reverb/noise realism first — *but* see §4 and §6 before acting.

---

## 4. The "largest contributor" line — why yours is not yet useful

```
>>> Largest domain-gap contributor: 'eval_val_seed1337' (wavlm/wasserstein = 0.1702)
```

This flag is designed to tell you **which acoustic condition** (noise-only,
reverb-only, clean, a specific mic) drives the gap. In your run there is only
**one** condition, `eval_val_seed1337`, so "largest contributor" is just the
overall number relabelled — **no localisation is happening.**

The condition label comes from the immediate sub-directory under each dataset
root. To get the diagnosis the toolkit is built for, organise **both** datasets
with matching condition sub-folders, e.g.:

```
real/clean/…        sim/clean/…
real/noise_only/…   sim/noise_only/…
real/reverb_only/…  sim/reverb_only/…
real/mic_A/…        sim/mic_A/…
```

Then each condition gets its own FAD/MMD/Wasserstein, and the flag will name the
condition with the biggest sim-to-real distance — the thing to fix first.

---

## 5. `wasserstein_1d` vs `wasserstein_mv` — which to trust

Use **`wasserstein_mv`** (the headline `wasserstein`) as the primary transport
number — it accounts for cross-dimension structure. Use `wasserstein_1d` as a
cheap cross-check and for the ratio diagnostic in §2. If the two diverge a lot
(as in PANN here), the gap is concentrated in *correlations*, which the marginal
view understates.

---

## 6. Calibrate: the real-vs-real baseline (do this next)

Absolute values mean little without a floor. Two baselines make every number
above interpretable:

1. **Real-vs-real (the noise floor).** Split your real set into two disjoint halves
   and run the pipeline with `--real halfA --sim halfB`. Whatever distance you get
   is the *irreducible* value from finite samples + within-domain variability. If
   real-vs-real FAD ≈ 3 and real-vs-sim FAD ≈ 27, the gap is large and real. If
   real-vs-real FAD ≈ 24, then 27 is essentially noise.
2. **Sim-vs-sim (optional).** Same, on the simulated set — tells you the
   simulator's own internal variability for reference.

Interpretation becomes a ratio/margin: **gap = (real-vs-sim) − (real-vs-real)**,
and "how many baselines above the floor" is far more informative than the raw
number. Report all three (real-vs-sim, real-vs-real, sim-vs-sim) whenever you can.

---

## 7. Sample size, subsampling & reproducibility

- **FAD** used all your utterances (bias negligible at ~96k).
- **MMD** and **exact-EMD Wasserstein** subsample to `distances.max_samples`
  (default 2000) for tractability, so those numbers carry sampling variance. To
  tighten them: raise `max_samples`, and/or run a few `distances.random_seed`
  values and average. Sliced-Wasserstein variance also falls as you raise
  `distances.sliced_n_projections`.
- Because subsampling and the sliced projections are seeded, a fixed
  `random_seed` gives byte-identical numbers run-to-run (good for A/B comparisons
  of simulator changes).

---

## 8. Decision checklist

- [ ] Did you compute a **real-vs-real baseline**? If not, do it — the numbers
      above are uncalibrated (§6).
- [ ] Are your data organised into **matching condition sub-folders**? If not, the
      "largest contributor" is meaningless (§4).
- [ ] Compare **within** a backbone, never across (Rule 2); use **MMD** for the one
      cautious cross-backbone read.
- [ ] Do FAD/MMD/Wasserstein **agree within** each backbone? They do here → trust
      the ranking (§3).
- [ ] For A/B'ing simulator changes: hold `max_samples`, `sliced_n_projections`
      and `random_seed` fixed, and track the **drop** in each metric.

**Bottom line for this run:** a genuine, moderate sim-to-real gap on both content
and channel, with the **channel (WavLM) somewhat worse** — but you must (a) add a
real-vs-real baseline to size it and (b) split into acoustic conditions to
localise it.
