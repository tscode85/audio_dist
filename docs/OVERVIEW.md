# acoustic-gap — Complete Project Overview

**One document describing everything built so far**: what the toolkit does, how it
is architected, the algorithm and its mathematics, the two embedding backbones,
the distance metrics and how to read them, the fully-offline / air-gapped GPU
deployment, how it scales to hundred-hour corpora, the test suite, the key design
decisions (and the course-corrections made along the way), and the known
limitations.

It is self-contained. Deeper single-topic write-ups are cross-referenced:
[ALGORITHM](ALGORITHM.md) · [THEORY](THEORY.md) · [ARCHITECTURE](ARCHITECTURE.md) ·
[INTERPRETING_RESULTS](INTERPRETING_RESULTS.md) · [README](../README.md).

---

## 1. What it is

`acoustic-gap` is a production-grade, **fully offline / air-gapped** Python toolkit
that **quantifies the acoustic domain gap between a real audio dataset and a
simulated (synthetic) one**, and **diagnoses which acoustic factor** — noise
profile, room/reverb, or microphone response — drives the largest sim-to-real gap.

It does not emit a single opaque score. It reports:

- a **top-level score per metric per backbone**,
- a **per-condition breakdown** (so the gap is localised, not averaged away),
- a flagged **single largest contributor** to the gap,
- as structured **JSON + CSV + a bar chart**.

Everything runs on-premises with **no network access at runtime**: all model
weights are vendored to a local cache and loaded from disk.

### Why this is hard / interesting

"How different is simulated audio from real audio?" has no single right answer
because *different* can mean different **content** (what sounds are present) or
different **channel** (how it was recorded — mic, room, noise). The toolkit
separates these by measuring the gap in **two complementary embedding spaces** and
with **three complementary distances**, then slicing by acoustic condition.

---

## 2. The core idea

A domain gap is the statement that real audio ~ distribution `P` and simulated
audio ~ distribution `Q` differ. With an encoder `φ: audio → ℝ^d`, the gap is a
statistical distance `D(φ#P, φ#Q)` between the two clouds of embeddings.

Two deliberate choices:

1. **Measure in embedding space, not on waveforms.** Raw audio geometry is
   dominated by perceptually irrelevant differences (phase, sample offset). A
   pretrained encoder maps audio to a space where distance tracks *acoustic*
   similarity — the same rationale as FID for images and FAD for audio.
2. **Use two encoders and three distances.** Different encoders expose different
   *kinds* of gap; different distances make different statistical assumptions.
   Agreement across them is evidence the gap is real; disagreement is diagnostic.

Theoretical grounding: the **domain-adaptation bound (Ben-David et al. 2010)** —
target-domain error is bounded by source error **plus** a divergence between the
domains. Shrinking the sim-to-real divergence is exactly what a better simulator
should do.

---

## 3. Pipeline at a glance

```
config
  └─▶ preprocessing ─▶ segment manifest (metadata only; per-segment condition label)
        └─▶ STREAM utterances (one file at a time):
              load ─▶ embed with EVERY backbone ─▶ pool to one vector ─▶ discard audio
                └─▶ (optional) content disentanglement
                      └─▶ distances: FAD / MMD / Wasserstein, overall + per condition
                            └─▶ report: JSON + CSV + bar chart + largest-contributor flag
```

The **segment manifest** (a pandas DataFrame keyed by `segment_id`, carrying
`dataset`, `source_path`, `condition`, sample offsets, loudness) is the single
data contract every stage reads from. Conditions are inferred from the immediate
sub-directory under each dataset root (`real/noise_only/…` → condition
`noise_only`), so results are always sliceable by condition.

---

## 4. Module map

```
acoustic_gap/
├── config.py          # AppConfig dataclasses; YAML/JSON load + validate
├── logging_utils.py   # stdlib logging setup (never print, except the CLI summary)
├── preprocessing.py   # audio I/O, resample, mono, segment (overlap), silence filter,
│                       #   manifest build (metadata-only option), streaming iterator
├── features/
│   ├── base.py        # FeatureExtractor ABC — the backbone contract + ensure_ready()
│   ├── panns.py       # PANN (Cnn14) / VGGish via frechet_audio_distance
│   ├── wavlm.py       # WavLM x-vector via transformers (local, offline)
│   ├── dummy.py       # dependency-free descriptor backbone (tests / dry-runs)
│   └── __init__.py    # build_extractors(cfg, sample_rate) factory
├── pooling.py         # mean / RNN pooling (pool_one, pool_embeddings) + disentangle
├── distances.py       # frechet_distance, mmd_rbf, *_wasserstein, compute_distances
├── reporting.py       # GapResult, build_report, save_report, bar-chart plotting
├── pipeline.py        # AcousticGapPipeline — the streaming orchestrator
└── cli.py             # argparse entry point: `run`, `init-config`
setup/download_models.py    # one-time online vendoring of WavLM + PANN checkpoints
config/{default,offline_gpu}.yaml
docker/{Dockerfile,entrypoint.sh,docker-compose.yml,build_offline_image.sh}
tests/                      # 33 pytest tests over synthetic fixtures (fully offline)
docs/                       # THEORY, ALGORITHM, ARCHITECTURE, INTERPRETING_RESULTS, this file
```

Each module has one responsibility and depends only on `config` + the shared data
contracts, so any stage is replaceable without touching the others.

---

## 5. Stage 1 — Preprocessing

`acoustic_gap/preprocessing.py`

- **Resample** both datasets to a common rate (default 16 kHz, configurable).
- **Mono** downmix.
- **Segment** into fixed-length windows (default 3 s) with configurable overlap
  (`window_seconds`, `hop_seconds`). Handles edge cases: clips shorter than a
  window are zero-padded (or dropped), a trailing partial window is captured,
  anything below `min_clip_seconds` is dropped.
- **Silence filtering** by segment RMS in dBFS (`silence_rms_dbfs`, `drop_silence`).
- **Peak normalisation** (optional) before feature extraction.
- **Manifest**: one row per surviving segment, with a **condition label** inferred
  from the sub-directory, so downstream results slice by condition.

Two build modes:

- `keep_audio=True` — holds waveforms in RAM (fine for tests / small data).
- `keep_audio=False` — **metadata only**; audio is reloaded on demand during
  streaming. This is what the pipeline uses (see §11, scaling).

---

## 6. Stage 2 — Feature backbones

`acoustic_gap/features/` — all implement the `FeatureExtractor` ABC
(`embed_batch`, `embedding_dim`, `ensure_ready`), so backbones are swappable and
any subset runs together.

| backbone | package | what it captures | dim |
|---|---|---|---|
| **PANN (Cnn14)** | `frechet_audio_distance` (`model_name="pann"`) | **content / texture** — is it a plausible real-world sound? | 2048 |
| **WavLM x-vector** | `transformers` `WavLMForXVector` | **recording channel** — mic response, reverb, noise floor | 512 |
| **dummy** | none (hand-crafted spectral descriptors) | wiring/tests only | 32 |

- **PANN** (Kong et al. 2020) is a CNN trained on AudioSet tagging → content-
  invariant texture. We use the 16 kHz `Cnn14_16k` variant so it matches the
  pipeline's 16 kHz rate. Loaded offline from a vendored checkpoint directory; we
  call the model directly to get the 2048-d `embedding` per segment (the package's
  own `get_embeddings` flattens PANN vectors, so we bypass it).
- **WavLM x-vector** (Chen et al. 2022 + x-vector head, Snyder et al. 2018) is
  trained to tell *recordings* apart → highly sensitive to the recording chain.
  Loaded from a local HF snapshot with `HF_HUB_OFFLINE=1` / `local_files_only=True`;
  falls back CUDA→CPU gracefully.

**Why two:** a large PANN gap with a small WavLM gap ⇒ content differs; a small
PANN gap with a large WavLM gap ⇒ content is fine but the channel (mic/reverb/
noise) is wrong. The per-condition breakdown then says *which* channel factor.

`ensure_ready()` eagerly loads each backbone **before** preprocessing, so a missing
checkpoint or bad model name fails in seconds rather than after a long manifest
build.

---

## 7. Stage 3 — Pooling & disentanglement

`acoustic_gap/pooling.py`

Backbones emit one vector per segment; we aggregate to one vector per utterance:

- **Mean pooling (default)** — `φ̄ = (1/T) Σ φ_t`, the Monte-Carlo estimate of
  `E[φ]`; unbiased, permutation-invariant, stable.
- **RNN summariser (optional)** — an **untrained, seeded** (bi)GRU as a fixed,
  reproducible non-linear "reservoir" that can capture temporal structure the mean
  discards. (No labels to train it → used purely as a fixed transform.)

**Optional content disentanglement** — regress a content label out of the
channel-sensitive embeddings with a **ridge linear probe** and keep the residual:

```
W* = (Cᵀ C + α I)⁻¹ Cᵀ E ,   Ẽ = E − C W*
```

`Ẽ` is the component orthogonal to everything the content label can linearly
predict (Frisch–Waugh–Lovell partialling-out), so the remaining gap is attributed
to the acoustic channel. Tradeoff: linear removes only linearly-decodable content
(closed-form, deterministic, dependency-light); INLP is the non-linear
generalisation we deliberately do not take.

---

## 8. Stage 4 — Distance metrics

`acoustic_gap/distances.py` — all **lower-is-better**; `compute_distances`
orchestrates them. Between real cloud `X` and sim cloud `Y`:

### FAD — Fréchet Audio Distance
```
FAD = ‖μ_X − μ_Y‖² + tr(Σ_X + Σ_Y − 2 (Σ_X Σ_Y)^½)
```
Fits one Gaussian per cloud; the 2-Wasserstein distance between the Gaussians
(Dowson–Landau closed form). Captures mean + covariance; blind to higher moments;
biased upward at small N (negligible at ~10⁴+ samples). Matrix sqrt via
`scipy.linalg.sqrtm` with an `εI` fallback.

### MMD — kernel Maximum Mean Discrepancy (unbiased MMD²)
```
MMD²  = mean_{i≠j} k(x_i,x_j) + mean_{i≠j} k(y_i,y_j) − 2 mean_{i,j} k(x_i,y_j)
k(a,b) = exp(−‖a−b‖² / 2σ²),   σ = √( median{‖z_i−z_j‖²} / 2 )   (median heuristic)
```
Distance between RKHS mean embeddings; with the characteristic RBF kernel,
`MMD=0 ⇔ P=Q`, so it sees **all moments**. The median-heuristic bandwidth
self-normalises to each space — this makes MMD the least-unfair metric to compare
across backbones.

### Wasserstein — optimal transport
- **`wasserstein_1d`** — mean over dimensions of the exact 1-D `W₁`
  (`scipy.stats.wasserstein_distance`): marginal gap, ignores cross-dim coupling.
- **`wasserstein_mv` (= `wasserstein`)** — sliced-Wasserstein-1 (default; averages
  1-D `W₁` over random projections, scales linearly) **or** exact EMD via POT
  (`wasserstein_pot`, O(n³), subsampled). Captures the joint geometry.

Tractability guards: MMD and exact-EMD subsample to `distances.max_samples`
(default 2000); sliced-Wasserstein scales linearly and needs no cap. All seeded
via `distances.random_seed` for byte-identical reruns.

---

## 9. Stage 5 — Reporting & interpretation

`acoustic_gap/reporting.py`

Emits, to `report.output_dir`:

- `report.json` — `top_level_scores`, `per_condition_breakdown`, and a `headline`
  block naming the **largest contributor** (default WavLM × Wasserstein), with an
  explicit lower-is-better annotation. Degenerate/empty runs produce a valid,
  explanatory report rather than crashing.
- `results.csv` — tidy per-`(backbone, condition, metric)` table.
- `per_condition_gap.png` — bar chart comparing per-condition distances.

**Reading the numbers (full guide in [INTERPRETING_RESULTS.md](INTERPRETING_RESULTS.md)):**

- Lower is better; unbounded above (except MMD). **No fixed "good" threshold** — a
  number is meaningful only relative to a reference.
- **Never compare magnitudes across backbones** (different spaces/scales/dims);
  compare within a backbone, or use MMD for a cautious cross-backbone read.
- **Calibrate with a real-vs-real baseline** (split real in half, run
  `--real halfA --sim halfB`) to size the gap against the finite-sample floor.
- Within a backbone, cross-metric agreement (FAD/MMD/Wasserstein pointing the same
  way) is the robustness check.
- The `wasserstein_mv / wasserstein_1d` ratio is a diagnostic: ≈1 means the gap is
  in the marginals (broadband, e.g. mic EQ / noise floor); ≫1 means it lives in
  cross-dimension correlations.

---

## 10. Fully offline / air-gapped deployment

Runtime is **network-free**; everything is vendored at build time on a connected
machine, then carried over.

### Pinned stack (`requirements.txt`)

```
numpy 1.26.4 · scipy 1.11.4 · pandas 2.1.4 · soundfile 0.12.1 · librosa 0.10.1
torch 2.1.2 · torchvision 0.16.2 · torchaudio 2.1.2 (CUDA 12.1 / cu121 wheels)
transformers 4.36.2
frechet_audio_distance 0.3.4  (+ laion-clap 1.1.7, encodec 0.1.1, torchlibrosa 0.1.0)
POT 0.9.1 · matplotlib 3.8.2 · PyYAML 6.0.1 · pytest 7.4.4
```

### Vendoring weights — `setup/download_models.py`

Run once WITH internet:
```bash
python setup/download_models.py --cache-dir ./model_cache \
    --wavlm microsoft/wavlm-base-plus-sv --fad-model pann --fad-sr 16000
```
Snapshots WavLM (feature extractor + weights) to `model_cache/wavlm-base-plus-sv`
and downloads the PANN Cnn14 Zenodo checkpoint into `model_cache/fad_ckpt`. Point
`features.wavlm_local_dir` and `features.panns_checkpoint` at those. At runtime,
`HF_HUB_OFFLINE=1` / `local_files_only=True` guarantee no network.

### Docker (GPU, Ubuntu 24.04 + NVIDIA driver 580 + Container Toolkit)

`docker/Dockerfile` builds a self-contained CUDA image: installs the cu121 torch
wheels (self-contained — they bundle their own CUDA runtime; the driver-580 host
supplies only `libcuda`), the pinned requirements, then bakes the WavLM + PANN
weights into `/opt/model_cache` and sets `HF_HUB_OFFLINE=1`/`TRANSFORMERS_OFFLINE=1`.

```bash
# on a connected machine:
./docker/build_offline_image.sh          # builds + `docker save | gzip` → tarball

# on the air-gapped host:
docker load -i acoustic-gap-offline.tar.gz
docker run --rm --gpus all \
    -v /data/real:/data/real:ro -v /data/sim:/data/sim:ro -v /out:/data/report \
    acoustic-gap:offline run --config /app/config/offline_gpu.yaml \
    --real /data/real --sim /data/sim --output-dir /data/report
```
`docker/docker-compose.yml` wraps this; `entrypoint.sh` also exposes `test` and a
shell. `DOWNLOAD_MODELS=0` builds a weightless dev image.

---

## 11. Scaling to hundred-hour corpora (streaming)

The original design loaded **every segment's waveform into RAM at once** and
materialised all segment embeddings before pooling — ~90 GB for 100 h + 100 h,
which OOM'd. The pipeline now **streams**:

1. Build a **metadata-only** manifest (`keep_audio=False`).
2. Iterate **one source file at a time** (`iter_utterances`): load → embed with
   every backbone → pool to a single utterance vector → discard the audio.
3. Accumulate only the compact utterance embeddings, then compute distances.

Peak memory is now **one file's segments + the utterance embeddings**, independent
of corpus length. Validated: a run streaming from disk gives byte-identical
results to an in-memory-manifest run. Caveat: granularity is **per file** — a
single multi-hour file is still decoded whole, so split very long recordings.

---

## 12. Robustness & failure handling

- **Config validation** (`AppConfig.validate`) rejects bad settings (sample rate,
  hop > window, unknown backbone/metric, bad OT backend) before any work.
- **Fail-fast backbone loading** — `ensure_ready()` runs before preprocessing, so a
  missing checkpoint or unknown model name fails in seconds, not after a long
  manifest build (with a message listing the cause).
- **Empty / mismatched datasets** — if either side yields 0 usable segments the run
  aborts with an actionable message (wrong path, unsupported extension, clips below
  `min_clip_seconds`, over-aggressive silence filtering); if real/sim share no
  condition labels it warns and reports the overall gap only. The reporter never
  crashes on empty results.
- **Determinism** — a single `random_seed` flows into subsampling, the RNN
  summariser, and the sliced projections.
- **Logging, not printing** — everything via stdlib `logging`; only the CLI prints
  the final human summary; progress logged every 500 utterances.

---

## 13. Testing

`tests/` — **33 pytest tests, fully offline, no weights**. Tiny synthetic WAVs are
built on the fly (`conftest.py`) and every stage is exercised through the `dummy`
backbone:

| file | covers |
|---|---|
| `test_config.py` | config round-trip + validation guards |
| `test_preprocessing.py` | segmentation, overlap, short/silent handling, manifest |
| `test_features.py` | `FeatureExtractor` contract (shape, determinism, batching) |
| `test_pooling.py` | mean pooling math + content-disentanglement removal |
| `test_distances.py` | metric sanity: ~0 for identical, larger under a shift |
| `test_pipeline_and_reporting.py` | end-to-end run; correct largest-contributor; empty/mismatched-dataset handling |
| `test_streaming.py` | `keep_audio=False` holds no audio; reload==in-memory; streaming report == in-memory report |

Green suite proves the wiring, math, streaming and reporting; the heavy PANN/WavLM
backbones are exercised only in a real run with vendored weights.

---

## 14. Design decisions & course-corrections

The honest development history — decisions and the mistakes fixed along the way:

1. **Config over pydantic** — plain dataclasses avoid an extra vendored runtime
   dependency; validation is explicit.
2. **MMD kernel** — RBF + median-heuristic bandwidth (parameter-free default);
   fixed σ configurable.
3. **Multivariate Wasserstein** — sliced by default (linear-time, scales); exact
   POT-EMD opt-in when exactness matters.
4. **Pooling** — mean by default; the RNN is an untrained fixed transform (no
   labels to train it), stated as such.
5. **Disentanglement** — linear ridge probe (closed-form, deterministic); not INLP.
6. **The FAD-backend saga (three iterations, now settled):**
   - First pinned `fadtk==0.1.6` → **did not exist** on PyPI; corrected to `1.0.0`.
   - `fadtk 1.0.0` ships **VGGish, not PANN**; on request for PANN, tried
     `fadtk 1.1.0` (which forced torch 2.7 + torchvision).
   - Discovered by reading the source that **the PyPI `fadtk` implements no PANN in
     any version.** Real PANN (Cnn14) lives in **`frechet_audio_distance`**
     (`model_name="pann"`), which has loose torch pins — so we **dropped fadtk**,
     reverted to the proven **torch 2.1.2 / cu121** stack, and get true PANN there.
7. **OOM at scale** — redesigned from load-everything to **streaming** (§11).
8. **Silent-exit bug** — a run that "counted samples then exited with no report"
   was one dataset yielding 0 segments → now fails fast with guidance, and the
   reporter no longer crashes on empty results.

Full per-metric tradeoff notes live in [THEORY.md](THEORY.md).

---

## 15. Known limitations & caveats

- **Cross-backbone magnitudes are not comparable** (different spaces); the toolkit
  reports them side-by-side but interpretation must stay within a backbone (or use
  MMD cautiously).
- **No built-in real-vs-real baseline** yet — you must run it manually to calibrate
  (§9). A future convenience flag could automate it.
- **Per-file streaming granularity** — a single enormous file still loads whole;
  split long recordings.
- **Linear disentanglement** removes only linearly-decodable content leakage.
- **Heavy backbones unverified in this environment** — the PANN/WavLM code is
  written from each package's verified source and the light test suite passes, but
  the full GPU stack (and the Zenodo PANN checkpoint download) is first exercised on
  the user's connected build machine.
- **Condition localisation requires matching sub-folders** in both datasets; with a
  single condition the "largest contributor" flag is just the overall number.

---

## 16. Quick start (recap)

```bash
# 1. (connected machine) install + vendor weights
pip install -r requirements.txt
python setup/download_models.py --cache-dir ./model_cache \
    --wavlm microsoft/wavlm-base-plus-sv --fad-model pann --fad-sr 16000

# 2. run a comparison
python -m acoustic_gap.cli run --config config/default.yaml \
    --real /data/real --sim /data/sim --output-dir ./report

# offline dry-run without weights:
python -m acoustic_gap.cli run --real real/ --sim sim/ --backbones dummy

# tests:
pytest -q
```

Organise both datasets into **matching condition sub-folders**
(`real/noise_only`, `sim/noise_only`, `real/reverb_only`, …) to get the
per-condition diagnosis, and run a **real-vs-real baseline** to calibrate.

---

## 17. Key references

FAD (Kilgour 2019) · FID (Heusel 2017) · Fréchet-Gaussian (Dowson–Landau 1982) ·
MMD (Gretton 2012) · median heuristic (Garreau 2017) · Computational OT
(Peyré–Cuturi 2019) · EMD (Rubner 2000) · Sliced-Wasserstein (Bonneel 2015;
Kolouri 2019) · POT (Flamary 2021) · VGGish (Hershey 2017) · AudioSet (Gemmeke
2017) · PANNs (Kong 2020) · WavLM (Chen 2022) · x-vectors (Snyder 2018) ·
domain-adaptation bound (Ben-David 2010) · INLP (Ravfogel 2020). Full citations
with links in [THEORY.md §8](THEORY.md#8-references).
