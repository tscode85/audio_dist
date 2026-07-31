# file: tests/test_preprocessing.py
"""Tests for preprocessing: segmentation, silence/short handling, manifest."""
from __future__ import annotations

import numpy as np

from acoustic_gap.config import PreprocessConfig
from acoustic_gap.preprocessing import (
    build_manifest,
    peak_normalize,
    rms_dbfs,
    segment_signal,
    to_mono,
)


def test_to_mono():
    stereo = np.stack([np.ones(10), np.full(10, 3.0)])
    assert np.allclose(to_mono(stereo), 2.0)
    assert to_mono(np.ones(5)).shape == (5,)


def test_rms_dbfs_silence():
    assert rms_dbfs(np.zeros(100)) == -np.inf
    assert rms_dbfs(np.ones(100)) == 0.0  # full-scale


def test_peak_normalize():
    x = np.array([0.0, 0.25, -0.5], dtype=np.float32)
    out = peak_normalize(x)
    assert np.isclose(np.max(np.abs(out)), 1.0)


def test_segment_overlap_and_count():
    sr = 16000
    cfg = PreprocessConfig(window_seconds=1.0, hop_seconds=0.5, min_clip_seconds=0.1)
    wav = np.ones(int(3.0 * sr), dtype=np.float32)
    segs = segment_signal(wav, sr, cfg)
    # 3s with 1s window / 0.5s hop -> starts at 0,0.5,1.0,1.5,2.0 (+ maybe tail)
    assert len(segs) >= 5
    # all full windows have equal length
    lens = {len(a) for _, _, a in segs}
    assert sr in lens


def test_short_clip_padding():
    sr = 16000
    cfg = PreprocessConfig(window_seconds=2.0, hop_seconds=1.0, pad_short_clips=True,
                           min_clip_seconds=0.1)
    wav = np.ones(int(0.5 * sr), dtype=np.float32)
    segs = segment_signal(wav, sr, cfg)
    assert len(segs) == 1
    assert len(segs[0][2]) == int(2.0 * sr)  # padded to window

    cfg_nopad = PreprocessConfig(window_seconds=2.0, hop_seconds=1.0,
                                 pad_short_clips=False, min_clip_seconds=0.1)
    assert segment_signal(wav, sr, cfg_nopad) == []


def test_below_min_clip_dropped():
    sr = 16000
    cfg = PreprocessConfig(window_seconds=2.0, min_clip_seconds=1.0)
    wav = np.ones(int(0.3 * sr), dtype=np.float32)
    assert segment_signal(wav, sr, cfg) == []


def test_build_manifest_conditions_and_silence(synthetic_datasets):
    cfg = PreprocessConfig(window_seconds=1.0, hop_seconds=1.0, drop_silence=True)
    man = build_manifest(synthetic_datasets["real"], synthetic_datasets["sim"], cfg)
    assert len(man) > 0
    conds = set(man.df["condition"])
    assert {"clean", "reverb", "noise"} <= conds
    assert set(man.df["dataset"]) == {"real", "sim"}
    # audio arrays align with rows
    assert len(man.audio) == len(man.df)


def test_manifest_drops_silence(short_and_silent):
    cfg = PreprocessConfig(window_seconds=1.0, hop_seconds=1.0, drop_silence=True,
                           silence_rms_dbfs=-50.0, pad_short_clips=True,
                           min_clip_seconds=0.1)
    man = build_manifest(short_and_silent, None, cfg)
    paths = set(man.df["source_path"].apply(lambda p: p.split("/")[-1]))
    assert "silent.wav" not in paths     # silence filtered
    assert "normal.wav" in paths
