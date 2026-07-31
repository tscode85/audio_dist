# acoustic-gap

A production-grade, **fully offline / air-gapped** Python toolkit to quantify the
**acoustic domain gap** between a real-world audio dataset and a simulated
(synthetic) one — and to diagnose **which acoustic factor** (noise profile,
room/reverb, microphone response) drives the largest sim-to-real gap.

It does not just emit one number: it reports a top-level score *and* a
per-condition breakdown, and flags the single largest contributor to the gap.

---

## Why two backbones?

| Backbone | Role | What it captures |
|----------|------|------------------|
| **PANN / VGGish** (via `fadtk`/`frechet_audio_distance`) | content-invariant | general spectro-temporal texture; "does this sound real?" |
| **WavLM x-vector** (`transformers` `WavLMForXVector`) | channel-sensitive | mic colouration, reverb, noise floor — the recording chain |

Running both side-by-side lets you separate *content* differences from
*channel/recording-condition* differences. The channel-sensitive WavLM view drives
the headline "largest contributor" flag by default.

## Metrics (all lower-is-better)

- **FAD** — Fréchet distance between Gaussians fit to the two embedding sets
  (closed form here; also computable directly via `fadtk` directory-vs-directory).
- **MMD** — kernel two-sample distance (RBF kernel, median-heuristic bandwidth).
- **Wasserstein** — per-dimension 1-D W1 (`scipy`) **and** a multivariate estimate
  (exact EMD via **POT**, or a scalable sliced-Wasserstein approximation).

---

## Architecture

```
acoustic_gap/
  config.py          # dataclass config + YAML/JSON load/validate
  preprocessing.py   # resample, mono, segment (overlap), silence filter, manifest
  features/
    base.py          # FeatureExtractor abstract interface (swappable backbones)
    panns.py         # PANN/VGGish via fadtk / frechet_audio_distance
    wavlm.py         # WavLM x-vector via transformers (local, offline)
    dummy.py         # dependency-free backbone for tests / dry-runs
  pooling.py         # mean pool (default) or RNN summariser; content disentangle
  distances.py       # FAD, MMD, Wasserstein (scipy / POT / sliced)
  reporting.py       # aggregate -> JSON + CSV + bar chart + largest-contributor flag
  pipeline.py        # config-driven runner tying it all together
  cli.py             # argparse CLI (`run`, `init-config`)
setup/download_models.py   # one-time online pre-download of all checkpoints
config/default.yaml        # starter config
tests/                     # pytest suite with synthetic audio fixtures
```

**Data flow:** `config -> preprocessing (segment manifest with per-segment
condition metadata) -> per-backbone segment embeddings -> pooling (+ optional
content disentanglement) -> distances (overall + per condition) -> report`.
Every module keys off the segment **manifest**, so results can always be sliced
by condition (`clean/`, `reverb-only/`, `noise-only/`, `mic-*/`, ...).

Conditions are inferred from the **immediate sub-directory** under each dataset
root, e.g. `real/noise-only/clip.wav` -> condition `noise-only`. Organise both
`real/` and `sim/` with matching sub-folders to get a per-condition breakdown.

---

## Offline / air-gapped setup

### 1. On a machine WITH internet — vendor dependencies and weights

```bash
pip install -r requirements.txt          # pinned versions
python setup/download_models.py \
    --cache-dir ./model_cache \
    --wavlm microsoft/wavlm-base-plus-sv \
    --fadtk-model panns-wavegram-logmel
```

This snapshots the WavLM x-vector model (feature extractor + weights) to
`./model_cache/wavlm-base-plus-sv` and warms the `fadtk` PANN/VGGish cache.

### 2. Ship to the air-gapped host

Copy the installed environment (or wheels) **and** the `./model_cache` directory
to the offline machine.

### 3. On the air-gapped host — force offline mode

```bash
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export HF_HOME=/path/to/model_cache/hf
```

Point the config at the local snapshot (already the default):

```yaml
features:
  wavlm_local_dir: /path/to/model_cache/wavlm-base-plus-sv
```

All HuggingFace calls use `local_files_only=True`, so nothing touches the network
at runtime.

---

## Run an end-to-end comparison on two directories

```bash
# Optional: emit a starter config you can edit
python -m acoustic_gap.cli init-config --output config/default.yaml

# Compare real/ vs sim/
python -m acoustic_gap.cli run \
    --config config/default.yaml \
    --real /data/real_audio \
    --sim  /data/sim_audio \
    --output-dir ./report
```

Outputs under `./report/`:

- `report.json` — top-level scores, per-condition breakdown, largest-contributor flag.
- `results.csv` — tidy per-(backbone, condition, metric) table.
- `per_condition_gap.png` — bar chart comparing per-condition distances.

The CLI also prints a summary and the flagged largest contributor, e.g.:

```
>>> Largest domain-gap contributor: 'noise-only' (wavlm/wasserstein = 0.8123)
```

### Quick dry-run without any weights

Use the dependency-free `dummy` backbone to validate wiring offline:

```bash
python -m acoustic_gap.cli run --real real/ --sim sim/ --backbones dummy
```

---

## Testing (no network, no weights)

```bash
pip install pytest soundfile PyYAML
pytest -q
```

The suite builds tiny synthetic WAV datasets on the fly and exercises every
module (preprocessing, features, pooling, distances, reporting, and an
end-to-end pipeline run) using the `dummy` backbone.

---

## Design decisions & tradeoffs

- **MMD kernel:** RBF with the **median-heuristic** bandwidth — parameter-free,
  robust, and the standard default. A fixed float sigma is configurable.
- **Multivariate Wasserstein:** defaults to **sliced-Wasserstein** (linear in
  sample count) so it scales to large embedding sets; exact **POT EMD** (O(n³),
  subsampled) is available via `wasserstein_backend: pot` when exactness matters.
- **Pooling:** **mean** by default (stable, interpretable). An **untrained,
  seeded (bi)GRU** summariser is offered as a non-linear alternative — since no
  labels are available to train it, it is used purely as a deterministic fixed
  transform.
- **Disentanglement:** a **linear ridge probe** regresses out content-predictable
  variance from channel-sensitive embeddings. Linear keeps it closed-form,
  deterministic and dependency-light; it removes first-order (linearly decodable)
  content leakage only.
- **FAD:** computed in closed form on our own embeddings for parity across
  backbones/metrics, with a direct `fadtk` directory-vs-directory reference
  available in `distances.fad_via_fadtk`.
- **Config over pydantic:** plain dataclasses avoid an extra vendored runtime
  dependency; validation is explicit in `AppConfig.validate`.
```
