# file: tests/test_features.py
"""Tests for the feature-extractor interface using the dummy backbone."""
from __future__ import annotations

import numpy as np

from acoustic_gap.config import FeatureConfig
from acoustic_gap.features import DummyEmbedding, build_extractors


def test_dummy_embedding_shape_and_determinism():
    ex = DummyEmbedding(dim=32)
    wav = np.sin(np.linspace(0, 10, 16000)).astype(np.float32)
    a = ex.embed_batch([wav, wav])
    b = ex.embed_batch([wav, wav])
    assert a.shape == (2, 32)
    assert np.allclose(a, b)               # deterministic
    assert np.allclose(a[0], a[1])         # same input -> same output


def test_embed_all_batches_align():
    ex = DummyEmbedding(dim=16)
    wavs = [np.random.default_rng(i).standard_normal(8000).astype(np.float32)
            for i in range(10)]
    out = ex.embed_all(wavs, batch_size=3)
    assert out.shape == (10, 16)


def test_dummy_distinguishes_noise_level():
    ex = DummyEmbedding(dim=32)
    rng = np.random.default_rng(0)
    clean = 0.5 * np.sin(np.linspace(0, 50, 16000)).astype(np.float32)
    noisy = clean + 0.5 * rng.standard_normal(16000).astype(np.float32)
    e_clean = ex.embed_batch([clean])[0]
    e_noisy = ex.embed_batch([noisy])[0]
    assert not np.allclose(e_clean, e_noisy)


def test_build_extractors_factory_dummy():
    cfg = FeatureConfig(backbones=["dummy"])
    ex = build_extractors(cfg)
    assert "dummy" in ex
    assert ex["dummy"].embedding_dim == 32
