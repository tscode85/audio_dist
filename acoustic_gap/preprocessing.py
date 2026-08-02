# file: acoustic_gap/preprocessing.py
"""Audio preprocessing: resampling, mono conversion, segmentation, silence
filtering, and manifest construction.

The output of this module is a :class:`SegmentManifest` — a pandas DataFrame with
one row per audio *segment* plus per-segment condition metadata. Every downstream
module (features, pooling, distances, reporting) keys off this manifest so that
results can be sliced by condition later.

Audio I/O uses ``soundfile`` (libsndfile) and resampling uses ``librosa`` /
``scipy`` — all available offline.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import PreprocessConfig

logger = logging.getLogger(__name__)

AUDIO_EXTS = {".wav", ".flac", ".ogg", ".mp3", ".m4a", ".aif", ".aiff"}


# ---------------------------------------------------------------------------
# Low-level audio helpers
# ---------------------------------------------------------------------------
def _load_soundfile(path: Path) -> Tuple[np.ndarray, int]:
    import soundfile as sf  # local import keeps module import cheap/offline-safe

    data, sr = sf.read(str(path), always_2d=True, dtype="float32")
    # soundfile returns (frames, channels); transpose to (channels, frames).
    return data.T, int(sr)


def to_mono(wav: np.ndarray) -> np.ndarray:
    """Average multi-channel audio to a single channel.

    Parameters
    ----------
    wav:
        Array shaped ``(channels, frames)`` or ``(frames,)``.
    """
    if wav.ndim == 1:
        return wav.astype(np.float32)
    return wav.mean(axis=0).astype(np.float32)


def resample(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Resample a mono signal to ``target_sr``."""
    if orig_sr == target_sr:
        return wav.astype(np.float32)
    import librosa

    return librosa.resample(
        wav.astype(np.float32), orig_sr=orig_sr, target_sr=target_sr
    ).astype(np.float32)


def rms_dbfs(wav: np.ndarray) -> float:
    """Return the RMS level of ``wav`` in dBFS (full-scale = 1.0)."""
    if wav.size == 0:
        return -np.inf
    rms = float(np.sqrt(np.mean(np.square(wav, dtype=np.float64))))
    if rms <= 1e-12:
        return -np.inf
    return 20.0 * np.log10(rms)


def peak_normalize(wav: np.ndarray, eps: float = 1e-9) -> np.ndarray:
    peak = float(np.max(np.abs(wav))) if wav.size else 0.0
    if peak < eps:
        return wav
    return (wav / peak).astype(np.float32)


def load_audio_mono(path: Path, target_sr: int, mono: bool = True) -> np.ndarray:
    """Load, (optionally) downmix to mono and resample an audio file."""
    wav, sr = _load_soundfile(path)
    if mono:
        wav = to_mono(wav)
    else:  # keep first channel for deterministic feature extraction
        wav = wav[0] if wav.ndim == 2 else wav
    return resample(wav, sr, target_sr)


# ---------------------------------------------------------------------------
# Segmentation
# ---------------------------------------------------------------------------
@dataclass
class Segment:
    """A single fixed-length window of audio plus provenance."""

    source_path: str
    segment_index: int
    start_sample: int
    end_sample: int
    sample_rate: int
    audio: np.ndarray  # 1-D float32
    rms_dbfs: float
    condition: str
    dataset: str  # "real" or "sim"
    extra: Dict[str, object]


def segment_signal(
    wav: np.ndarray,
    sample_rate: int,
    cfg: PreprocessConfig,
) -> List[Tuple[int, int, np.ndarray]]:
    """Split a mono signal into overlapping fixed-length windows.

    Returns a list of ``(start_sample, end_sample, audio)`` tuples. Handles clips
    shorter than one window by padding (when ``cfg.pad_short_clips``) or dropping.
    """
    win = int(round(cfg.window_seconds * sample_rate))
    hop = int(round(cfg.hop_seconds * sample_rate))
    win = max(win, 1)
    hop = max(hop, 1)
    n = len(wav)

    min_len = int(round(cfg.min_clip_seconds * sample_rate))
    if n < min_len:
        return []

    if n < win:
        if not cfg.pad_short_clips:
            return []
        padded = np.zeros(win, dtype=np.float32)
        padded[:n] = wav
        return [(0, n, padded)]

    out: List[Tuple[int, int, np.ndarray]] = []
    start = 0
    while start + win <= n:
        out.append((start, start + win, wav[start : start + win].copy()))
        start += hop
    # Capture a trailing partial window so the clip end is not silently dropped.
    if start < n and (n - start) >= min_len:
        tail = np.zeros(win, dtype=np.float32)
        chunk = wav[start:n]
        tail[: len(chunk)] = chunk
        out.append((start, n, tail))
    return out


# ---------------------------------------------------------------------------
# File discovery + condition inference
# ---------------------------------------------------------------------------
def discover_audio_files(root: Path) -> List[Path]:
    """Recursively find audio files under ``root`` (sorted for determinism)."""
    root = Path(root)
    files = [p for p in root.rglob("*") if p.suffix.lower() in AUDIO_EXTS]
    return sorted(files)


def infer_condition(path: Path, root: Path) -> str:
    """Infer a condition label from the immediate sub-directory under ``root``.

    Convention: ``<root>/<condition>/.../file.wav`` -> ``condition``. Files placed
    directly in ``root`` get the condition ``"all"``. This lets users organise
    ``noise-only/``, ``reverb-only/``, ``clean/`` etc. without extra metadata.
    """
    try:
        rel = path.relative_to(root)
    except ValueError:
        return "all"
    parts = rel.parts
    return parts[0] if len(parts) > 1 else "all"


# ---------------------------------------------------------------------------
# Manifest
# ---------------------------------------------------------------------------
MANIFEST_COLUMNS = [
    "dataset",
    "source_path",
    "segment_index",
    "start_sample",
    "end_sample",
    "sample_rate",
    "rms_dbfs",
    "condition",
]


class SegmentManifest:
    """Wraps a DataFrame of segments and holds the audio arrays in memory.

    The audio arrays are stored separately (``self.audio``) keyed by a stable
    ``segment_id`` so the DataFrame stays lightweight and serialisable.
    """

    def __init__(self, df: pd.DataFrame, audio: Dict[int, np.ndarray]):
        self.df = df.reset_index(drop=True)
        self.audio = audio

    def __len__(self) -> int:
        return len(self.df)

    @property
    def segment_ids(self) -> List[int]:
        return list(self.df["segment_id"])

    def audio_batch(self, segment_ids: Sequence[int]) -> List[np.ndarray]:
        return [self.audio[i] for i in segment_ids]

    def subset(self, mask: pd.Series) -> "SegmentManifest":
        sub_df = self.df[mask].reset_index(drop=True)
        sub_audio = {int(i): self.audio[int(i)] for i in sub_df["segment_id"]}
        return SegmentManifest(sub_df, sub_audio)

    def save_csv(self, path: str | Path) -> None:
        """Persist the metadata table (audio arrays are not serialised)."""
        self.df.to_csv(path, index=False)


def build_manifest(
    real_dir: Optional[str | Path],
    sim_dir: Optional[str | Path],
    cfg: PreprocessConfig,
    condition_map: Optional[Dict[str, str]] = None,
    keep_audio: bool = True,
) -> SegmentManifest:
    """Preprocess both datasets into a single :class:`SegmentManifest`.

    Parameters
    ----------
    real_dir, sim_dir:
        Directories of audio files. Sub-directory names are used as condition
        labels (see :func:`infer_condition`).
    cfg:
        Preprocessing configuration.
    condition_map:
        Optional override mapping ``source_path -> condition`` for datasets whose
        conditions cannot be inferred from directory structure.
    keep_audio:
        When ``True`` (default) the segment waveforms are held in memory — fine
        for small datasets and unit tests. When ``False`` only per-segment
        *metadata* is retained (the audio is reloaded on demand during feature
        extraction via :func:`iter_utterances`), which keeps peak memory bounded
        to a single file regardless of total dataset size. The pipeline uses
        ``keep_audio=False`` for large (many-hour) corpora.
    """
    rows: List[dict] = []
    audio: Dict[int, np.ndarray] = {}
    seg_id = 0

    for dataset, root in (("real", real_dir), ("sim", sim_dir)):
        if root is None:
            continue
        root = Path(root)
        files = discover_audio_files(root)
        logger.info("Discovered %d audio files in %s dataset (%s)", len(files), dataset, root)
        for path in files:
            try:
                wav = load_audio_mono(path, cfg.target_sample_rate, cfg.mono)
            except Exception as exc:  # pragma: no cover - depends on codec
                logger.warning("Failed to load %s: %s", path, exc)
                continue
            condition = (
                condition_map.get(str(path))
                if condition_map and str(path) in condition_map
                else infer_condition(path, root)
            )
            segs = segment_signal(wav, cfg.target_sample_rate, cfg)
            for idx, (s0, s1, seg_wav) in enumerate(segs):
                level = rms_dbfs(seg_wav)
                if cfg.drop_silence and level < cfg.silence_rms_dbfs:
                    continue
                rows.append(
                    {
                        "segment_id": seg_id,
                        "dataset": dataset,
                        "source_path": str(path),
                        "segment_index": idx,
                        "start_sample": s0,
                        "end_sample": s1,
                        "sample_rate": cfg.target_sample_rate,
                        "rms_dbfs": level,
                        "condition": condition,
                    }
                )
                if keep_audio:
                    if cfg.peak_normalize:
                        seg_wav = peak_normalize(seg_wav)
                    audio[seg_id] = seg_wav.astype(np.float32)
                seg_id += 1
            del wav  # release the file's samples before moving on

    df = pd.DataFrame(rows, columns=["segment_id"] + MANIFEST_COLUMNS)
    logger.info(
        "Built manifest with %d segments (real=%d, sim=%d)",
        len(df),
        int((df["dataset"] == "real").sum()) if len(df) else 0,
        int((df["dataset"] == "sim").sum()) if len(df) else 0,
    )
    return SegmentManifest(df, audio)


def iter_utterances(
    manifest: SegmentManifest,
    cfg: PreprocessConfig,
) -> "Iterable[Tuple[str, pd.DataFrame, List[np.ndarray]]]":
    """Yield ``(source_path, rows, segment_waveforms)`` one utterance at a time.

    This is the memory-bounded access path used by the streaming pipeline. For
    each source file it yields only that file's segments, so at most one file's
    audio is resident at any moment — total dataset size no longer drives peak
    memory.

    If the manifest was built with ``keep_audio=True`` the in-memory arrays are
    reused; otherwise each file is reloaded from disk and re-segmented
    deterministically (segmentation is a pure function of the config), and the
    exact segments recorded in the manifest are selected by ``segment_index``.
    """
    df = manifest.df
    if len(df) == 0:
        return
    for path, sub in df.groupby("source_path", sort=False):
        sub = sub.sort_values("segment_index")
        if manifest.audio:
            audios = [manifest.audio[int(sid)] for sid in sub["segment_id"]]
            yield str(path), sub, audios
            continue
        try:
            wav = load_audio_mono(Path(path), cfg.target_sample_rate, cfg.mono)
        except Exception as exc:  # pragma: no cover - depends on codec
            logger.warning("Failed to reload %s during streaming: %s", path, exc)
            continue
        segs = segment_signal(wav, cfg.target_sample_rate, cfg)
        audios = []
        for idx in sub["segment_index"]:
            if idx < len(segs):
                a = segs[idx][2]
                if cfg.peak_normalize:
                    a = peak_normalize(a)
                audios.append(a.astype(np.float32))
        del wav
        yield str(path), sub, audios
