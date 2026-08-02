# file: acoustic_gap/features/wavlm.py
"""WavLM x-vector embedding backbone via HuggingFace ``transformers``.

WavLM-*-sv is a speaker-verification head (x-vector) on top of WavLM. Because it
is trained to separate speakers / recording conditions, its embeddings are highly
*channel-sensitive*: they respond to microphone colouration, room reverberation
and noise floor — exactly the recording-chain factors that differentiate real
from simulated audio. This complements the content-invariant PANN/VGGish view.

Everything loads from a local snapshot directory with ``HF_HUB_OFFLINE=1`` so no
network access is required at runtime.
"""
from __future__ import annotations

import logging
import os
from typing import Optional, Sequence

import numpy as np

from .base import FeatureExtractor

logger = logging.getLogger(__name__)


class WavLMXVectorEmbedding(FeatureExtractor):
    """WavLM x-vector embeddings loaded from a local HF snapshot.

    Parameters
    ----------
    local_dir:
        Path to a local HF snapshot containing ``config.json``, the model weights
        and the feature-extractor config (produced by setup/download_models.py).
        When ``None``, falls back to ``model_name`` resolved from the local HF
        cache (still offline when ``HF_HUB_OFFLINE=1``).
    model_name:
        Model id used only for cache resolution / logging.
    """

    name = "wavlm"

    def __init__(
        self,
        local_dir: Optional[str] = None,
        model_name: str = "microsoft/wavlm-base-plus-sv",
        sample_rate: int = 16000,
        device: str = "cpu",
    ):
        self.local_dir = local_dir
        self.model_name = model_name
        self._sample_rate = sample_rate
        self.device = device
        self._model = None
        self._extractor = None
        self._dim: Optional[int] = None

    @property
    def sample_rate(self) -> int:  # type: ignore[override]
        return self._sample_rate

    def _source(self) -> str:
        return self.local_dir if self.local_dir else self.model_name

    def _lazy_load(self):
        if self._model is not None:
            return
        # Enforce offline mode; loading must not touch the network.
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        try:
            import torch  # noqa: F401
            from transformers import Wav2Vec2FeatureExtractor, WavLMForXVector
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "transformers + torch are required for the WavLM backbone."
            ) from exc

        # Resolve device; fall back to CPU if CUDA was requested but is absent.
        import torch as _torch

        if self.device.startswith("cuda") and not _torch.cuda.is_available():
            logger.warning("CUDA requested but unavailable; using CPU for WavLM.")
            self.device = "cpu"

        source = self._source()
        logger.info("Loading WavLM x-vector model from %s on %s (offline)", source, self.device)
        self._extractor = Wav2Vec2FeatureExtractor.from_pretrained(
            source, local_files_only=True
        )
        model = WavLMForXVector.from_pretrained(source, local_files_only=True)
        model.eval()
        model.to(self.device)
        self._model = model

    def ensure_ready(self) -> None:
        """Load the WavLM model now so a missing offline snapshot fails fast."""
        self._lazy_load()

    @property
    def embedding_dim(self) -> int:
        if self._dim is None:
            self._lazy_load()
            # WavLMForXVector exposes the x-vector size via config.xvector_output_dim.
            self._dim = int(getattr(self._model.config, "xvector_output_dim", 512))
        return self._dim

    def embed_batch(self, waveforms: Sequence[np.ndarray]) -> np.ndarray:
        import torch

        self._lazy_load()
        inputs = self._extractor(
            [np.asarray(w, dtype=np.float32) for w in waveforms],
            sampling_rate=self._sample_rate,
            return_tensors="pt",
            padding=True,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        with torch.no_grad():
            out = self._model(**inputs)
        emb = out.embeddings  # (batch, xvector_dim)
        return emb.detach().cpu().numpy().astype(np.float32)
