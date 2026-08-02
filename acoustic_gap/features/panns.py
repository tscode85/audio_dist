# file: acoustic_gap/features/panns.py
"""PANN / VGGish embedding backbones via ``fadtk`` or ``frechet_audio_distance``.

These backbones provide a *content-invariant* acoustic representation: they were
trained on large-scale audio tagging (AudioSet) and capture the general
spectro-temporal texture of a clip rather than its linguistic content. That makes
them a good backbone for measuring how "real" simulated audio sounds.

Both libraries are optional and imported lazily so the rest of the toolkit works
(and its tests run) without them. All model weights are loaded from local paths;
nothing hits the network.
"""
from __future__ import annotations

import logging
from typing import Optional, Sequence

import numpy as np

from .base import FeatureExtractor

logger = logging.getLogger(__name__)


class FadtkEmbedding(FeatureExtractor):
    """Wraps a ``fadtk`` model as a segment embedder.

    ``fadtk`` exposes a family of pretrained models (PANN, VGGish, CLAP, ...). We
    use its low-level ``get_embedding`` API on in-memory waveforms so the same
    interface works for the manual MMD/Wasserstein metrics, while FAD proper is
    computed directly by fadtk in :mod:`acoustic_gap.distances`.

    Parameters
    ----------
    model_name:
        A fadtk model identifier, e.g. ``"panns-wavegram-logmel"`` or ``"vggish"``.
    checkpoint_dir:
        Local directory holding the pre-downloaded weights. fadtk reads from its
        own cache; set the ``FADTK_CACHE``/model-specific env var accordingly in
        the offline setup (see setup/download_models.py).
    """

    def __init__(
        self,
        model_name: str = "panns-wavegram-logmel",
        checkpoint_dir: Optional[str] = None,
        sample_rate: int = 16000,
        device: str = "cpu",
    ):
        self.name = "vggish" if "vggish" in model_name.lower() else "panns"
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        self._sample_rate = sample_rate
        self.device = device
        self._model = None
        self._dim: Optional[int] = None

    def _lazy_model(self):
        if self._model is not None:
            return self._model
        try:
            from fadtk.model_loader import get_all_models  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "fadtk is required for the PANN/VGGish backbone. Install the "
                "pinned version and pre-download weights (see setup/)."
            ) from exc
        models = {m.name: m for m in get_all_models()}
        if self.model_name not in models:
            raise ValueError(
                f"fadtk model '{self.model_name}' is not provided by the installed "
                f"fadtk version. Available models: {sorted(models)}. "
                f"Set features.fadtk_model to an available name. PANN "
                f"('panns-wavegram-logmel') requires fadtk>=1.1.0; 'vggish' is in "
                f"every fadtk build."
            )
        model = models[self.model_name]
        model.load_model()  # loads weights from local cache
        self._model = model
        self._sample_rate = getattr(model, "sr", self._sample_rate)
        return model

    def ensure_ready(self) -> None:
        """Load the fadtk model now so a bad name / missing weights fails fast."""
        self._lazy_model()

    @property
    def sample_rate(self) -> int:  # type: ignore[override]
        return self._sample_rate

    @sample_rate.setter
    def sample_rate(self, value: int) -> None:
        self._sample_rate = value

    @property
    def embedding_dim(self) -> int:
        if self._dim is not None:
            return self._dim
        model = self._lazy_model()
        self._dim = int(getattr(model, "num_features", 0)) or self._probe_dim(model)
        return self._dim

    def _probe_dim(self, model) -> int:
        probe = np.zeros(self._sample_rate, dtype=np.float32)
        emb = model.get_embedding(probe)
        return int(np.asarray(emb).reshape(-1, np.asarray(emb).shape[-1]).shape[-1])

    def embed_batch(self, waveforms: Sequence[np.ndarray]) -> np.ndarray:
        model = self._lazy_model()
        vecs = []
        for wav in waveforms:
            emb = np.asarray(model.get_embedding(np.asarray(wav, dtype=np.float32)))
            # fadtk returns per-frame embeddings (frames, dim); mean-pool to a
            # single vector here so the backbone emits one vector per segment.
            if emb.ndim == 2:
                emb = emb.mean(axis=0)
            vecs.append(emb.astype(np.float32))
        return np.stack(vecs, axis=0)


class FrechetAudioDistanceEmbedding(FeatureExtractor):
    """PANN / VGGish embeddings via the ``frechet_audio_distance`` package.

    This is the real PANN path: ``model_name="pann"`` loads Kong et al.'s Cnn14
    (the 16 kHz ``Cnn14_16k`` variant at 16 kHz, matching this toolkit's default
    pipeline) and yields 2048-dim content-invariant embeddings.

    Offline use: the underlying package looks for its checkpoint under
    ``ckpt_dir`` and only downloads it (from Zenodo) if absent. Vendor the weight
    once with ``setup/download_models.py`` and point ``checkpoint_dir`` at that
    folder — nothing hits the network at runtime.

    Notes
    -----
    We call the loaded model directly (not the package's ``get_embeddings``,
    which concatenates PANN's per-clip vectors into a flat 1-D array). The
    package moves the model to CUDA automatically when available.
    """

    def __init__(
        self,
        model_name: str = "pann",
        checkpoint_dir: Optional[str] = None,
        sample_rate: int = 16000,
        device: str = "cpu",
    ):
        self.name = "vggish" if model_name.lower() == "vggish" else "panns"
        self.model_name = model_name
        self.checkpoint_dir = checkpoint_dir
        self._sample_rate = sample_rate
        self.device = device
        self._fad = None
        self._dim: Optional[int] = None

    @property
    def sample_rate(self) -> int:  # type: ignore[override]
        return self._sample_rate

    def ensure_ready(self) -> None:
        self._lazy_model()

    def _lazy_model(self):
        if self._fad is not None:
            return self._fad
        try:
            from frechet_audio_distance import FrechetAudioDistance  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise ImportError(
                "frechet_audio_distance is required for the PANN/VGGish backbone. "
                "Install it (see requirements.txt) and vendor the checkpoint with "
                "setup/download_models.py."
            ) from exc
        if self.model_name == "pann" and not self.checkpoint_dir:
            raise ValueError(
                "PANN requires features.panns_checkpoint to point at the folder "
                "holding the pre-downloaded Cnn14 weight (offline). Run "
                "setup/download_models.py to vendor it."
            )
        if self.checkpoint_dir:
            import os

            os.makedirs(self.checkpoint_dir, exist_ok=True)
        self._fad = FrechetAudioDistance(
            ckpt_dir=self.checkpoint_dir,
            model_name=self.model_name,
            sample_rate=self._sample_rate,
            use_pca=False,
            use_activation=False,
            verbose=False,
        )
        return self._fad

    @property
    def embedding_dim(self) -> int:
        if self._dim is None:
            probe = np.zeros(self._sample_rate, dtype=np.float32)
            self._dim = int(self.embed_batch([probe]).shape[-1])
        return self._dim

    def embed_batch(self, waveforms: Sequence[np.ndarray]) -> np.ndarray:
        import torch

        fad = self._lazy_model()
        device = getattr(fad, "device", None)
        vecs = []
        for wav in waveforms:
            arr = np.asarray(wav, dtype=np.float32)
            if self.model_name == "pann":
                with torch.no_grad():
                    t = torch.tensor(arr).float().unsqueeze(0)
                    if device is not None:
                        t = t.to(device)
                    out = fad.model(t, None)
                    emb = out["embedding"].data[0].cpu().numpy()  # (2048,)
            elif self.model_name == "vggish":
                emb = np.asarray(fad.model.forward(arr, self._sample_rate))
                if emb.ndim == 2:  # (frames, 128) -> mean-pool to one vector
                    emb = emb.mean(axis=0)
            else:  # pragma: no cover - other FAD models not wired here
                raise ValueError(f"unsupported FAD model_name '{self.model_name}'")
            vecs.append(np.asarray(emb, dtype=np.float32).reshape(-1))
        return np.stack(vecs, axis=0)
