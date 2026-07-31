# file: tests/test_pooling.py
"""Tests for pooling and content disentanglement."""
from __future__ import annotations

import numpy as np
import pandas as pd

from acoustic_gap.config import PoolingConfig
from acoustic_gap.pooling import disentangle_content, pool_embeddings


def _manifest(n_utts=4, segs_per=3, dim=5):
    rows = []
    embs = []
    rng = np.random.default_rng(0)
    for u in range(n_utts):
        for s in range(segs_per):
            rows.append({
                "source_path": f"/x/u{u}.wav",
                "segment_index": s,
                "dataset": "real" if u % 2 == 0 else "sim",
                "condition": "clean",
                "content_label": f"c{u % 2}",
            })
            embs.append(rng.standard_normal(dim) + u)
    return np.array(embs, dtype=np.float32), pd.DataFrame(rows)


def test_mean_pool_shapes_and_values():
    emb, df = _manifest()
    cfg = PoolingConfig(strategy="mean")
    pooled, utt_df = pool_embeddings(emb, df, cfg)
    assert pooled.shape[0] == 4              # one row per utterance
    assert pooled.shape[1] == emb.shape[1]
    assert len(utt_df) == 4
    assert set(utt_df["dataset"]) == {"real", "sim"}
    # mean of segments for utt 0 (all centered near value 0)
    expected = emb[:3].mean(axis=0)
    assert np.allclose(pooled[0], expected, atol=1e-5)


def test_disentangle_removes_content_signal():
    # Build embeddings whose first dim is pure content label -> should vanish.
    rng = np.random.default_rng(1)
    rows, embs = [], []
    for u in range(8):
        label = u % 2
        for s in range(2):
            rows.append({"source_path": f"/u{u}.wav", "segment_index": s,
                         "dataset": "real", "condition": "clean",
                         "content_label": f"c{label}"})
            vec = rng.standard_normal(4)
            vec[0] = 10.0 * label            # strong content component
            embs.append(vec)
    df = pd.DataFrame(rows)
    cfg = PoolingConfig(strategy="mean", disentangle=True)
    pooled, utt_df = pool_embeddings(np.array(embs, dtype=np.float32), df, cfg)
    residual = disentangle_content(pooled, utt_df, cfg)
    # content-driven variance in dim 0 should be largely removed
    var_before = pooled[:, 0].var()
    var_after = residual[:, 0].var()
    assert var_after < 0.1 * var_before


def test_disentangle_noop_without_column():
    emb, df = _manifest()
    cfg = PoolingConfig(disentangle=True, content_label_column="missing")
    pooled, utt_df = pool_embeddings(emb, df, cfg)
    out = disentangle_content(pooled, utt_df, cfg)
    assert np.allclose(out, pooled)
