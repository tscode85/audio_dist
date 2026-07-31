# file: tests/conftest.py
"""Shared pytest fixtures: tiny synthetic audio datasets (no network, no weights)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

try:
    import soundfile as sf
    HAVE_SF = True
except Exception:  # pragma: no cover
    HAVE_SF = False

SR = 16000


def _tone(freq: float, seconds: float, sr: int = SR, noise: float = 0.0,
          seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    t = np.linspace(0, seconds, int(seconds * sr), endpoint=False)
    sig = 0.5 * np.sin(2 * np.pi * freq * t)
    if noise:
        sig = sig + noise * rng.standard_normal(len(t))
    return sig.astype(np.float32)


def _write(path: Path, wav: np.ndarray, sr: int = SR) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(path), wav, sr)


@pytest.fixture
def synthetic_datasets(tmp_path):
    """Create real/ and sim/ dirs with conditioned sub-folders.

    'real' clips are clean tones; 'sim' clips add noise/offset so a measurable
    gap exists, with the largest gap deliberately in the 'noise' condition.
    """
    if not HAVE_SF:
        pytest.skip("soundfile not available")
    real = tmp_path / "real"
    sim = tmp_path / "sim"
    conditions = ["clean", "reverb", "noise"]
    for i in range(6):
        freq = 220 + 40 * i
        for cond in conditions:
            _write(real / cond / f"r{i}.wav", _tone(freq, 2.0, seed=i))
        # sim: clean nearly matches; noise strongly perturbed.
        _write(sim / "clean" / f"s{i}.wav", _tone(freq, 2.0, noise=0.01, seed=100 + i))
        _write(sim / "reverb" / f"s{i}.wav", _tone(freq * 1.01, 2.0, noise=0.05, seed=200 + i))
        _write(sim / "noise" / f"s{i}.wav", _tone(freq, 2.0, noise=0.4, seed=300 + i))
    return {"real": real, "sim": sim}


@pytest.fixture
def short_and_silent(tmp_path):
    """A dir with a too-short clip, a silent clip, and a normal clip."""
    if not HAVE_SF:
        pytest.skip("soundfile not available")
    d = tmp_path / "edge"
    _write(d / "short.wav", _tone(300, 0.2))          # shorter than window
    _write(d / "silent.wav", np.zeros(int(2.0 * SR), dtype=np.float32))
    _write(d / "normal.wav", _tone(300, 3.0))
    return d
