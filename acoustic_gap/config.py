# file: acoustic_gap/config.py
"""Configuration schema and loading for the acoustic-gap toolkit.

All pipeline behaviour is driven by a single :class:`AppConfig` object that is
normally materialised from a YAML/JSON file (see ``config/default.yaml``). The
config is intentionally declarative so an air-gapped deployment can be tuned
without touching code.

Design note: we use plain ``dataclasses`` rather than pydantic to avoid adding a
runtime dependency that would need to be vendored offline. Validation is done
explicitly in :meth:`AppConfig.validate`.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

try:  # PyYAML is optional; JSON configs work without it.
    import yaml  # type: ignore
except Exception:  # pragma: no cover - exercised only when PyYAML missing
    yaml = None  # type: ignore


@dataclass
class PreprocessConfig:
    """Controls resampling, segmentation and silence filtering."""

    target_sample_rate: int = 16000
    window_seconds: float = 3.0
    hop_seconds: float = 1.5  # overlap = window - hop
    mono: bool = True
    # Clips shorter than the window are kept and zero-padded when True,
    # otherwise dropped.
    pad_short_clips: bool = True
    min_clip_seconds: float = 0.5  # drop anything shorter than this outright
    # Silence filtering (RMS in dBFS). Segments below the threshold are dropped.
    drop_silence: bool = True
    silence_rms_dbfs: float = -50.0
    # Optional peak normalisation before feature extraction.
    peak_normalize: bool = True


@dataclass
class FeatureConfig:
    """Selects and configures embedding backbones."""

    # Any subset of {"panns", "vggish", "wavlm"}. Both may run for comparison.
    # NOTE: the pinned fadtk==1.0.0 ships VGGish (not PANN) as its audio-tagging
    # backbone, so "vggish" is the default content-invariant view. A true PANN
    # model ("panns-wavegram-logmel") requires fadtk>=1.1.0 (torch>=2.3); if you
    # bump that stack, set backbones=["panns", ...] and fadtk_model accordingly.
    backbones: List[str] = field(default_factory=lambda: ["vggish", "wavlm"])
    # Local checkpoint locations (populated by setup/download_models.py).
    panns_checkpoint: Optional[str] = None
    vggish_checkpoint: Optional[str] = None
    # Local HF snapshot directory for the WavLM x-vector model.
    wavlm_local_dir: Optional[str] = None
    wavlm_model_name: str = "microsoft/wavlm-base-plus-sv"
    device: str = "cpu"  # "cpu" or "cuda"
    batch_size: int = 8
    # fadtk model name for the audio-tagging backbone. Must be one that the
    # installed fadtk version provides (fadtk==1.0.0 -> "vggish").
    fadtk_model: str = "vggish"


@dataclass
class PoolingConfig:
    """Pooling and optional content-disentanglement settings."""

    # "mean" or "rnn".
    strategy: str = "mean"
    rnn_hidden_size: int = 128
    rnn_num_layers: int = 1
    rnn_bidirectional: bool = True
    # Content disentanglement: strip residual content correlation from the
    # channel-sensitive embeddings using a linear probe against content labels.
    disentangle: bool = False
    # Column in the manifest holding a content/class label to regress out.
    content_label_column: str = "content_label"
    disentangle_ridge_alpha: float = 1.0


@dataclass
class DistanceConfig:
    """Which distances to compute and their hyper-parameters."""

    metrics: List[str] = field(default_factory=lambda: ["fad", "mmd", "wasserstein"])
    # MMD kernel: "rbf" (Gaussian) is the default. bandwidth "median" uses the
    # median-heuristic; a float sets the RBF sigma directly.
    mmd_kernel: str = "rbf"
    mmd_bandwidth: Any = "median"
    # Multivariate Wasserstein backend: "pot" (exact EMD via POT) or
    # "sliced" (sliced-Wasserstein approximation, scales to large N).
    wasserstein_backend: str = "sliced"
    sliced_n_projections: int = 128
    # Cap the number of samples fed to O(n^2)/O(n^3) estimators for tractability.
    max_samples: int = 2000
    random_seed: int = 0


@dataclass
class ReportConfig:
    """Reporting and plotting options."""

    output_dir: str = "acoustic_gap_report"
    # Manifest column used to slice results into condition subsets.
    condition_column: str = "condition"
    make_plots: bool = True
    # Which (backbone, metric) pair drives the headline "largest contributor"
    # flag. Defaults to the channel-sensitive backbone under Wasserstein.
    headline_backbone: str = "wavlm"
    headline_metric: str = "wasserstein"


@dataclass
class AppConfig:
    """Top-level configuration container."""

    real_dir: Optional[str] = None
    sim_dir: Optional[str] = None
    real_manifest: Optional[str] = None
    sim_manifest: Optional[str] = None
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    pooling: PoolingConfig = field(default_factory=PoolingConfig)
    distances: DistanceConfig = field(default_factory=DistanceConfig)
    report: ReportConfig = field(default_factory=ReportConfig)
    log_level: str = "INFO"

    # ---- construction helpers -------------------------------------------------
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "AppConfig":
        """Build an :class:`AppConfig` from a nested dictionary."""
        data = dict(data or {})
        sub = {
            "preprocess": PreprocessConfig,
            "features": FeatureConfig,
            "pooling": PoolingConfig,
            "distances": DistanceConfig,
            "report": ReportConfig,
        }
        kwargs: Dict[str, Any] = {}
        for key, klass in sub.items():
            kwargs[key] = klass(**(data.pop(key, {}) or {}))
        kwargs.update(data)
        return cls(**kwargs)

    @classmethod
    def load(cls, path: str | Path) -> "AppConfig":
        """Load config from a YAML or JSON file."""
        path = Path(path)
        text = path.read_text()
        if path.suffix.lower() in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError(
                    "PyYAML is required to read YAML configs; install it or use JSON."
                )
            data = yaml.safe_load(text)
        else:
            data = json.loads(text)
        cfg = cls.from_dict(data)
        cfg.validate()
        return cfg

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        data = self.to_dict()
        if path.suffix.lower() in {".yaml", ".yml"}:
            if yaml is None:
                raise RuntimeError("PyYAML is required to write YAML configs.")
            path.write_text(yaml.safe_dump(data, sort_keys=False))
        else:
            path.write_text(json.dumps(data, indent=2))

    # ---- validation -----------------------------------------------------------
    def validate(self) -> None:
        """Raise ``ValueError`` on inconsistent configuration."""
        p = self.preprocess
        if p.target_sample_rate <= 0:
            raise ValueError("target_sample_rate must be positive")
        if p.window_seconds <= 0:
            raise ValueError("window_seconds must be positive")
        if not (0 < p.hop_seconds <= p.window_seconds):
            raise ValueError("hop_seconds must be in (0, window_seconds]")
        valid_backbones = {"panns", "vggish", "wavlm", "dummy"}
        unknown = set(self.features.backbones) - valid_backbones
        if unknown:
            raise ValueError(f"unknown backbones: {sorted(unknown)}")
        if not self.features.backbones:
            raise ValueError("at least one feature backbone must be configured")
        if self.pooling.strategy not in {"mean", "rnn"}:
            raise ValueError("pooling.strategy must be 'mean' or 'rnn'")
        valid_metrics = {"fad", "mmd", "wasserstein"}
        unknown_m = set(self.distances.metrics) - valid_metrics
        if unknown_m:
            raise ValueError(f"unknown metrics: {sorted(unknown_m)}")
        if self.distances.wasserstein_backend not in {"pot", "sliced"}:
            raise ValueError("wasserstein_backend must be 'pot' or 'sliced'")
