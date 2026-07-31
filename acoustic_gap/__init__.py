# file: acoustic_gap/__init__.py
"""acoustic_gap — an offline toolkit to quantify the acoustic domain gap between
real and simulated audio datasets, and localise which channel factor (noise,
reverb, microphone) drives the largest sim-to-real gap.

See README.md (repository root) for offline setup and end-to-end usage.
"""
from __future__ import annotations

from .config import AppConfig
from .pipeline import AcousticGapPipeline, run_pipeline

__all__ = ["AppConfig", "AcousticGapPipeline", "run_pipeline", "__version__"]

__version__ = "0.1.0"
