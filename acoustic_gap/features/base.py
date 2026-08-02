# file: acoustic_gap/features/base.py
"""Abstract feature-extractor interface.

Every backbone (PANN/VGGish, WavLM x-vector, or a test double) implements
:class:`FeatureExtractor`. The pipeline treats all backbones identically, so they
are freely swappable and any subset can be run together for comparison.
"""
from __future__ import annotations

import abc
import logging
from typing import List, Sequence

import numpy as np

logger = logging.getLogger(__name__)


class FeatureExtractor(abc.ABC):
    """Common interface for frame/segment embedding backbones.

    Implementations take mono waveforms (1-D float32 numpy arrays at
    ``self.sample_rate``) and return a 2-D embedding matrix ``(n_segments, dim)``.
    """

    #: Human-readable backbone name, e.g. "panns" or "wavlm".
    name: str = "base"

    #: The sample rate the backbone expects. Preprocessing must match this.
    sample_rate: int = 16000

    @property
    @abc.abstractmethod
    def embedding_dim(self) -> int:
        """Dimensionality of the produced embeddings."""

    @abc.abstractmethod
    def embed_batch(self, waveforms: Sequence[np.ndarray]) -> np.ndarray:
        """Embed a batch of mono waveforms.

        Returns
        -------
        np.ndarray
            Array shaped ``(len(waveforms), embedding_dim)``.
        """

    def ensure_ready(self) -> None:
        """Eagerly load/validate the model so failures surface immediately.

        The pipeline calls this *before* the (potentially long) preprocessing
        pass, so a missing offline checkpoint or an unknown model name fails in
        seconds instead of after building a large manifest. Backbones with lazy
        loading override this to trigger the load; the default is a no-op.
        """
        return None

    def embed_all(
        self, waveforms: Sequence[np.ndarray], batch_size: int = 8
    ) -> np.ndarray:
        """Embed an arbitrary number of waveforms in mini-batches."""
        if len(waveforms) == 0:
            return np.zeros((0, self.embedding_dim), dtype=np.float32)
        out: List[np.ndarray] = []
        for i in range(0, len(waveforms), batch_size):
            chunk = waveforms[i : i + batch_size]
            emb = self.embed_batch(chunk)
            out.append(np.asarray(emb, dtype=np.float32))
        result = np.concatenate(out, axis=0)
        if result.shape[0] != len(waveforms):
            raise RuntimeError(
                f"{self.name}: produced {result.shape[0]} embeddings for "
                f"{len(waveforms)} inputs"
            )
        return result
