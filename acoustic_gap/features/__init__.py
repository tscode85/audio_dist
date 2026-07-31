# file: acoustic_gap/features/__init__.py
"""Feature-extraction backbones and a config-driven factory."""
from __future__ import annotations

import logging
from typing import Dict, List

from ..config import FeatureConfig
from .base import FeatureExtractor
from .dummy import DummyEmbedding

logger = logging.getLogger(__name__)

__all__ = ["FeatureExtractor", "DummyEmbedding", "build_extractors"]


def build_extractors(cfg: FeatureConfig) -> Dict[str, FeatureExtractor]:
    """Instantiate the configured backbones.

    Returns an ordered mapping ``name -> FeatureExtractor``. Unknown or
    unavailable heavy backbones raise at *use* time (lazy import), not here, so a
    dry-run with the dummy backbone never pulls in torch/fadtk.
    """
    extractors: Dict[str, FeatureExtractor] = {}
    for name in cfg.backbones:
        if name == "panns":
            from .panns import FadtkEmbedding

            extractors[name] = FadtkEmbedding(
                model_name=cfg.fadtk_model,
                checkpoint_dir=cfg.panns_checkpoint,
                device=cfg.device,
            )
        elif name == "vggish":
            from .panns import FadtkEmbedding

            extractors[name] = FadtkEmbedding(
                model_name="vggish",
                checkpoint_dir=cfg.vggish_checkpoint,
                device=cfg.device,
            )
        elif name == "wavlm":
            from .wavlm import WavLMXVectorEmbedding

            extractors[name] = WavLMXVectorEmbedding(
                local_dir=cfg.wavlm_local_dir,
                model_name=cfg.wavlm_model_name,
                device=cfg.device,
            )
        elif name == "dummy":  # available for tests / offline dry-runs
            extractors[name] = DummyEmbedding()
        else:  # pragma: no cover - guarded by config.validate
            raise ValueError(f"unknown backbone: {name}")
    logger.info("Built feature extractors: %s", list(extractors))
    return extractors
