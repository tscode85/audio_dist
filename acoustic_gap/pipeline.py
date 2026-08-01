# file: acoustic_gap/pipeline.py
"""End-to-end pipeline runner tying all modules together.

Flow::

    config -> preprocess (manifest) -> per-backbone embeddings
           -> pool (+ optional disentangle) -> distances (overall + per condition)
           -> report (JSON/CSV/plot + largest-contributor flag)

The runner is deliberately backbone- and metric-agnostic: it iterates whatever
the config selects, so PANN/VGGish and WavLM are measured side by side.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import AppConfig
from .distances import compute_distances
from .features import build_extractors
from .pooling import disentangle_content, pool_embeddings
from .preprocessing import SegmentManifest, build_manifest
from .reporting import GapResult, save_report

logger = logging.getLogger(__name__)

OVERALL = "__overall__"


class AcousticGapPipeline:
    """Config-driven orchestrator for a real-vs-sim comparison."""

    def __init__(self, config: AppConfig):
        config.validate()
        self.config = config

    # ------------------------------------------------------------------
    def run(
        self, manifest: Optional[SegmentManifest] = None
    ) -> Dict:
        """Execute the full pipeline and return the structured report dict."""
        cfg = self.config
        if manifest is None:
            manifest = build_manifest(
                cfg.real_dir, cfg.sim_dir, cfg.preprocess
            )
        self._check_manifest(manifest)

        extractors = build_extractors(cfg.features)
        results: List[GapResult] = []

        for name, extractor in extractors.items():
            logger.info("=== Backbone: %s ===", name)
            results.extend(self._run_backbone(name, extractor, manifest))

        report = save_report(results, cfg.report)
        return report

    # ------------------------------------------------------------------
    def _check_manifest(self, manifest: SegmentManifest) -> None:
        """Fail fast (with actionable guidance) when a comparison is impossible.

        Catches the common "counted samples, then produced nothing" situation:
        one dataset yields no usable segments, or the real/sim condition labels
        do not overlap, so there is no pair to compare.
        """
        from .preprocessing import AUDIO_EXTS

        cfg = self.config
        if len(manifest) == 0:
            raise RuntimeError(
                "No audio segments were produced from EITHER dataset. Check that "
                f"--real / --sim point at directories containing audio files "
                f"({sorted(AUDIO_EXTS)})."
            )
        df = manifest.df
        n_real = int((df["dataset"] == "real").sum())
        n_sim = int((df["dataset"] == "sim").sum())
        if n_real == 0 or n_sim == 0:
            empty = "sim" if n_sim == 0 else "real"
            path = cfg.sim_dir if n_sim == 0 else cfg.real_dir
            raise RuntimeError(
                f"The '{empty}' dataset produced 0 usable segments "
                f"(real={n_real}, sim={n_sim}); a real-vs-sim comparison needs "
                f"both sides. Likely causes for '{path}':\n"
                f"  * wrong path, or audio not one of {sorted(AUDIO_EXTS)} "
                f"(matching is recursive and case-insensitive);\n"
                f"  * every clip is shorter than preprocess.min_clip_seconds "
                f"({cfg.preprocess.min_clip_seconds}s);\n"
                f"  * everything was filtered as silence — try lowering "
                f"preprocess.silence_rms_dbfs ({cfg.preprocess.silence_rms_dbfs} "
                f"dBFS) or set preprocess.drop_silence=false."
            )

        # Warn (do not fail) when conditions do not overlap: the OVERALL score
        # still works, but there will be no per-condition breakdown.
        real_conds = set(df[df["dataset"] == "real"]["condition"])
        sim_conds = set(df[df["dataset"] == "sim"]["condition"])
        logger.info("Real conditions: %s", sorted(real_conds))
        logger.info("Sim conditions:  %s", sorted(sim_conds))
        shared = real_conds & sim_conds
        if not shared:
            logger.warning(
                "Real and sim share NO condition labels (real=%s, sim=%s). Only the "
                "OVERALL gap will be reported — there is no matched condition to "
                "break down. Organise both dataset roots with the SAME "
                "sub-directory names (e.g. real/clean, sim/clean) to get a "
                "per-condition diagnosis.",
                sorted(real_conds), sorted(sim_conds),
            )

    # ------------------------------------------------------------------
    def _run_backbone(
        self, name: str, extractor, manifest: SegmentManifest
    ) -> List[GapResult]:
        cfg = self.config
        df = manifest.df
        waveforms = manifest.audio_batch(df["segment_id"].tolist())

        seg_emb = extractor.embed_all(waveforms, batch_size=cfg.features.batch_size)
        logger.info("%s: segment embeddings %s", name, seg_emb.shape)

        # Pool to utterance level.
        utt_emb, utt_df = pool_embeddings(seg_emb, df, cfg.pooling, seed=cfg.distances.random_seed)

        # Optional content disentanglement (channel-sensitive backbones benefit).
        if cfg.pooling.disentangle:
            utt_emb = disentangle_content(utt_emb, utt_df, cfg.pooling)

        results: List[GapResult] = []
        # Overall gap.
        results.extend(self._distances_for(name, OVERALL, utt_emb, utt_df))

        # Per-condition gap: only conditions present in BOTH datasets are
        # meaningful; others are reported when at least real or sim has data.
        cond_col = cfg.report.condition_column
        if cond_col in utt_df.columns:
            for condition in sorted(utt_df[cond_col].unique()):
                mask = utt_df[cond_col] == condition
                results.extend(
                    self._distances_for(name, str(condition), utt_emb[mask.to_numpy()], utt_df[mask])
                )
        return results

    # ------------------------------------------------------------------
    def _distances_for(
        self, backbone: str, condition: str, emb: np.ndarray, meta: pd.DataFrame
    ) -> List[GapResult]:
        cfg = self.config
        real_mask = (meta["dataset"] == "real").to_numpy()
        sim_mask = (meta["dataset"] == "sim").to_numpy()
        real_emb, sim_emb = emb[real_mask], emb[sim_mask]

        if real_emb.shape[0] == 0 or sim_emb.shape[0] == 0:
            logger.info(
                "Condition '%s' [%s]: skipping (real=%d sim=%d)",
                condition, backbone, real_emb.shape[0], sim_emb.shape[0],
            )
            return []

        dists = compute_distances(real_emb, sim_emb, cfg.distances)
        results = []
        for metric, value in dists.items():
            # Skip internal aliases already covered by canonical names.
            results.append(
                GapResult(
                    backbone=backbone,
                    condition=condition,
                    metric=metric,
                    value=float(value),
                    n_real=int(real_emb.shape[0]),
                    n_sim=int(sim_emb.shape[0]),
                )
            )
        return results


def run_pipeline(config: AppConfig) -> Dict:
    """Convenience wrapper."""
    return AcousticGapPipeline(config).run()
