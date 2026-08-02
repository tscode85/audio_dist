# file: tests/test_streaming.py
"""Tests for the memory-bounded streaming path (large-corpus / OOM fix)."""
from __future__ import annotations

import numpy as np

from acoustic_gap.config import AppConfig, PreprocessConfig
from acoustic_gap.pipeline import AcousticGapPipeline
from acoustic_gap.preprocessing import build_manifest, iter_utterances


def test_keep_audio_false_holds_no_waveforms(synthetic_datasets):
    pre = PreprocessConfig(window_seconds=1.0, hop_seconds=1.0)
    man = build_manifest(
        synthetic_datasets["real"], synthetic_datasets["sim"], pre, keep_audio=False
    )
    assert len(man.df) > 0
    assert len(man.audio) == 0            # metadata only — no audio in RAM


def test_iter_utterances_reload_matches_in_memory(synthetic_datasets):
    pre = PreprocessConfig(window_seconds=1.0, hop_seconds=1.0)
    m_mem = build_manifest(synthetic_datasets["real"], None, pre, keep_audio=True)
    m_lazy = build_manifest(synthetic_datasets["real"], None, pre, keep_audio=False)

    mem = {p: list(au) for p, _, au in iter_utterances(m_mem, pre)}
    lazy = {p: list(au) for p, _, au in iter_utterances(m_lazy, pre)}
    assert mem.keys() == lazy.keys()
    for p in mem:
        assert len(mem[p]) == len(lazy[p])
        for a, b in zip(mem[p], lazy[p]):
            assert np.allclose(a, b, atol=1e-6)


def test_streaming_pipeline_matches_in_memory_manifest(synthetic_datasets, tmp_path):
    # Same numbers whether run() streams from disk or is handed an in-memory manifest.
    def make_cfg(out):
        return AppConfig.from_dict({
            "real_dir": str(synthetic_datasets["real"]),
            "sim_dir": str(synthetic_datasets["sim"]),
            "preprocess": {"window_seconds": 1.0, "hop_seconds": 1.0},
            "features": {"backbones": ["dummy"]},
            "distances": {"metrics": ["mmd", "wasserstein"],
                          "wasserstein_backend": "sliced", "sliced_n_projections": 32},
            "report": {"output_dir": str(out), "make_plots": False,
                       "headline_backbone": "dummy"},
        })

    # streaming (default: keep_audio=False, reload from disk)
    rep_stream = AcousticGapPipeline(make_cfg(tmp_path / "s")).run()

    # in-memory manifest handed in explicitly
    pre = PreprocessConfig(window_seconds=1.0, hop_seconds=1.0)
    man = build_manifest(synthetic_datasets["real"], synthetic_datasets["sim"],
                         pre, keep_audio=True)
    rep_mem = AcousticGapPipeline(make_cfg(tmp_path / "m")).run(manifest=man)

    a = rep_stream["top_level_scores"]["dummy"]
    b = rep_mem["top_level_scores"]["dummy"]
    for k in a:
        assert np.isclose(a[k], b[k], rtol=1e-5, atol=1e-6), (k, a[k], b[k])
