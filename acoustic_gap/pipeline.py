# file: acoustic_gap/pipeline.py
"""End-to-end pipeline runner tying all modules together.

Flow::

    config -> preprocess (lightweight manifest, metadata only)
           -> STREAM utterances: load one file, embed with every backbone,
              pool to a single utterance vector, discard the audio
           -> distances (overall + per condition) -> report

Streaming keeps peak memory bounded to a single file plus the (compact)
utterance embeddings, so multi-hour corpora do not OOM. The runner is
backbone- and metric-agnostic: it measures whatever the config selects, so
PANN/VGGish and WavLM are computed side by side in the same pass over the data.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .config import AppConfig
from .distances import compute_distances
from .features import build_extractors
from .pooling import disentangle_content, pool_one
from .preprocessing import SegmentManifest, build_manifest, iter_utterances
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

        # Build and eagerly validate backbones FIRST: an unknown fadtk model or a
        # missing offline checkpoint should fail in seconds, not after a long
        # preprocessing pass over the whole corpus.
        extractors = build_extractors(cfg.features)
        for name, extractor in extractors.items():
            logger.info("Loading backbone '%s' ...", name)
            extractor.ensure_ready()

        if manifest is None:
            # Metadata-only manifest: bounded memory regardless of corpus size.
            manifest = build_manifest(
                cfg.real_dir, cfg.sim_dir, cfg.preprocess, keep_audio=False
            )
        self._check_manifest(manifest)

        utt_emb, utt_df = self._stream_embed_pool(extractors, manifest)

        results: List[GapResult] = []
        for name in extractors:
            U = utt_emb[name]
            if cfg.pooling.disentangle:
                U = disentangle_content(U, utt_df, cfg.pooling)
            results.extend(self._distances_for(name, OVERALL, U, utt_df))
            cond_col = cfg.report.condition_column
            if cond_col in utt_df.columns:
                for condition in sorted(utt_df[cond_col].unique()):
                    mask = (utt_df[cond_col] == condition).to_numpy()
                    results.extend(
                        self._distances_for(name, str(condition), U[mask], utt_df[mask])
                    )

        report = save_report(results, cfg.report)
        return report

    # ------------------------------------------------------------------
    def _stream_embed_pool(self, extractors, manifest: SegmentManifest):
        """Single streaming pass: per utterance, embed with every backbone and
        pool to one vector. Bounds peak memory to one file's segments plus the
        accumulated (small) utterance embeddings.
        """
        cfg = self.config
        names = list(extractors)
        pooled: Dict[str, List[np.ndarray]] = {n: [] for n in names}
        meta_rows: List[dict] = []

        n_files = manifest.df["source_path"].nunique()
        logger.info("Streaming %d utterances through backbones %s", n_files, names)

        for i, (path, rows, audios) in enumerate(iter_utterances(manifest, cfg.preprocess)):
            if not audios:  # all of this file's segments failed to reload
                continue
            first = rows.iloc[0]
            row = {
                "source_path": path,
                "dataset": first["dataset"],
                "condition": first["condition"],
                "n_segments": len(audios),
            }
            if cfg.pooling.content_label_column in rows.columns:
                row[cfg.pooling.content_label_column] = first[cfg.pooling.content_label_column]
            meta_rows.append(row)

            for name in names:
                seg_emb = extractors[name].embed_all(
                    audios, batch_size=cfg.features.batch_size
                )
                pooled[name].append(
                    pool_one(seg_emb, cfg.pooling, seed=cfg.distances.random_seed)
                )
            if (i + 1) % 500 == 0:
                logger.info("  ... embedded %d/%d utterances", i + 1, n_files)

        utt_df = pd.DataFrame(meta_rows)
        utt_emb = {
            n: (np.stack(v, axis=0).astype(np.float32) if v
                else np.zeros((0, 0), dtype=np.float32))
            for n, v in pooled.items()
        }
        for n in names:
            logger.info("%s: utterance embeddings %s", n, utt_emb[n].shape)
        return utt_emb, utt_df

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
