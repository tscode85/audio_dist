# file: acoustic_gap/features/dummy.py
"""Deterministic, dependency-free feature backbone for tests and dry-runs.

Produces embeddings from cheap hand-crafted acoustic descriptors (log-energy,
spectral centroid/bandwidth/rolloff, zero-crossing rate, and a coarse log-mel-ish
spectrum). It has no heavy dependencies, so the full pipeline and its unit tests
run in an air-gapped CI without PANN/WavLM weights. It is *not* meant for
production measurement — only for validating wiring and math.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from .base import FeatureExtractor


class DummyEmbedding(FeatureExtractor):
    """Hand-crafted spectral descriptors as a stand-in embedding."""

    name = "dummy"

    def __init__(self, dim: int = 32, sample_rate: int = 16000, n_bands: int = 24):
        self._dim = dim
        self._sample_rate = sample_rate
        self.n_bands = n_bands

    @property
    def sample_rate(self) -> int:  # type: ignore[override]
        return self._sample_rate

    @property
    def embedding_dim(self) -> int:
        return self._dim

    def _features_one(self, wav: np.ndarray) -> np.ndarray:
        wav = np.asarray(wav, dtype=np.float64)
        n = len(wav)
        if n == 0:
            return np.zeros(self._dim, dtype=np.float32)
        # Magnitude spectrum.
        window = np.hanning(n) if n > 1 else np.ones(n)
        spec = np.abs(np.fft.rfft(wav * window)) + 1e-10
        freqs = np.fft.rfftfreq(n, d=1.0 / self._sample_rate)
        power = spec ** 2
        total = power.sum()
        centroid = float((freqs * power).sum() / total)
        bandwidth = float(np.sqrt(((freqs - centroid) ** 2 * power).sum() / total))
        cumpow = np.cumsum(power)
        rolloff = float(freqs[np.searchsorted(cumpow, 0.85 * total)]) if total > 0 else 0.0
        log_energy = float(np.log(np.mean(wav ** 2) + 1e-10))
        zcr = float(np.mean(np.abs(np.diff(np.sign(wav))) > 0)) if n > 1 else 0.0

        # Coarse log-band spectrum.
        band_edges = np.linspace(0, len(spec), self.n_bands + 1).astype(int)
        bands = np.array(
            [np.log(power[band_edges[i] : band_edges[i + 1]].mean() + 1e-10)
             for i in range(self.n_bands)]
        )
        scalars = np.array([log_energy, centroid / self._sample_rate,
                            bandwidth / self._sample_rate,
                            rolloff / self._sample_rate, zcr])
        feat = np.concatenate([scalars, bands])
        # Fit/truncate to the requested dimension.
        if len(feat) < self._dim:
            feat = np.pad(feat, (0, self._dim - len(feat)))
        else:
            feat = feat[: self._dim]
        return feat.astype(np.float32)

    def embed_batch(self, waveforms: Sequence[np.ndarray]) -> np.ndarray:
        return np.stack([self._features_one(w) for w in waveforms], axis=0)
