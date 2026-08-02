# file: acoustic_gap/pooling.py
"""Pooling and content-disentanglement.

Backbones in this toolkit already emit one vector per segment (they mean-pool
frames internally), so "pooling" here operates at the *utterance* level: it can
aggregate the segment embeddings belonging to the same source clip into a single
fixed-length utterance embedding.

Two strategies:

* ``mean`` (default) — average the segment embeddings of each utterance.
* ``rnn`` — feed the ordered segment embeddings through a (bi)GRU and take the
  final hidden state as a learned fixed-length summary. The GRU is *untrained*
  (random but fixed weights) — used as a deterministic non-linear reservoir
  summariser. Assumption: without labelled data we cannot train it, so we use it
  purely as an alternative fixed transform; the seed makes it reproducible.

Optionally, :func:`disentangle_content` strips residual content correlation from
channel-sensitive embeddings via a ridge linear probe against a content label,
isolating the acoustic-domain (channel) signal.
"""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .config import PoolingConfig

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utterance grouping
# ---------------------------------------------------------------------------
def _utterance_groups(df: pd.DataFrame) -> List[Tuple[str, np.ndarray]]:
    """Return ``(source_path, ordered row-index array)`` per utterance."""
    groups: List[Tuple[str, np.ndarray]] = []
    for path, sub in df.groupby("source_path", sort=False):
        order = sub.sort_values("segment_index").index.to_numpy()
        groups.append((str(path), order))
    return groups


# ---------------------------------------------------------------------------
# RNN summariser (untrained, deterministic)
# ---------------------------------------------------------------------------
def _rnn_summarize(
    sequences: Sequence[np.ndarray], cfg: PoolingConfig, seed: int = 0
) -> np.ndarray:
    """Summarise variable-length embedding sequences with a fixed random GRU."""
    try:
        import torch
        import torch.nn as nn
    except Exception as exc:  # pragma: no cover
        raise ImportError("torch is required for the RNN pooling strategy.") from exc

    dim = sequences[0].shape[-1]
    torch.manual_seed(seed)
    gru = nn.GRU(
        input_size=dim,
        hidden_size=cfg.rnn_hidden_size,
        num_layers=cfg.rnn_num_layers,
        batch_first=True,
        bidirectional=cfg.rnn_bidirectional,
    )
    gru.eval()
    outs = []
    with torch.no_grad():
        for seq in sequences:
            t = torch.from_numpy(np.asarray(seq, dtype=np.float32)).unsqueeze(0)
            _, h = gru(t)  # h: (num_layers*dirs, 1, hidden)
            outs.append(h[-(1 + int(cfg.rnn_bidirectional)):].reshape(-1).numpy())
    return np.stack(outs, axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Public pooling API
# ---------------------------------------------------------------------------
def pool_one(
    seg_emb: np.ndarray, cfg: PoolingConfig, seed: int = 0
) -> np.ndarray:
    """Pool a single utterance's segment embeddings into one vector.

    Used by the streaming pipeline so segment embeddings never accumulate across
    utterances. ``mean`` averages the segments; ``rnn`` runs the fixed GRU
    summariser on the ordered sequence.
    """
    if seg_emb.shape[0] == 0:
        raise ValueError("cannot pool an utterance with zero segments")
    if cfg.strategy == "rnn":
        return _rnn_summarize([seg_emb], cfg, seed=seed)[0].astype(np.float32)
    return seg_emb.mean(axis=0).astype(np.float32)


def pool_embeddings(
    embeddings: np.ndarray,
    df: pd.DataFrame,
    cfg: PoolingConfig,
    seed: int = 0,
) -> Tuple[np.ndarray, pd.DataFrame]:
    """Pool segment embeddings to utterance embeddings.

    Parameters
    ----------
    embeddings:
        Segment embedding matrix ``(n_segments, dim)`` aligned row-for-row with
        ``df``.
    df:
        Segment manifest slice (must contain ``source_path``, ``segment_index``,
        ``dataset``, ``condition``).
    cfg:
        Pooling configuration.

    Returns
    -------
    (utt_embeddings, utt_df)
        ``utt_embeddings`` shaped ``(n_utterances, out_dim)`` and a per-utterance
        metadata frame carrying ``dataset`` and ``condition``.
    """
    if len(df) != embeddings.shape[0]:
        raise ValueError("embeddings and manifest rows are misaligned")
    df = df.reset_index(drop=True)
    groups = _utterance_groups(df)

    rows = []
    if cfg.strategy == "rnn":
        seqs = [embeddings[idx] for _, idx in groups]
        pooled = _rnn_summarize(seqs, cfg, seed=seed)
    else:  # mean
        pooled = np.stack(
            [embeddings[idx].mean(axis=0) for _, idx in groups], axis=0
        ).astype(np.float32)

    for path, idx in groups:
        first = df.loc[idx[0]]
        row = {
            "source_path": path,
            "dataset": first["dataset"],
            "condition": first["condition"],
            "n_segments": len(idx),
        }
        if cfg.content_label_column in df.columns:
            row[cfg.content_label_column] = first[cfg.content_label_column]
        rows.append(row)
    utt_df = pd.DataFrame(rows)
    logger.info(
        "Pooled %d segments -> %d utterances via '%s'",
        len(df), len(utt_df), cfg.strategy,
    )
    return pooled, utt_df


# ---------------------------------------------------------------------------
# Content disentanglement
# ---------------------------------------------------------------------------
def disentangle_content(
    embeddings: np.ndarray,
    utt_df: pd.DataFrame,
    cfg: PoolingConfig,
) -> np.ndarray:
    """Remove the content-predictable component from channel-sensitive embeddings.

    Fits a ridge linear probe ``content_onehot -> embedding`` and returns the
    residuals ``embedding - prediction``. Intuition: whatever a linear map of the
    content label can explain is content-driven variance; the residual retains the
    channel/domain signal we care about. Requires ``cfg.content_label_column`` in
    ``utt_df`` — otherwise the embeddings are returned unchanged with a warning.

    Assumption/tradeoff: a *linear* probe is used (closed-form, no training loop,
    deterministic, cheap offline). It removes only linearly-decodable content and
    may leave non-linear content leakage; this is the standard, robust first-order
    correction and keeps the toolkit dependency-light.
    """
    col = cfg.content_label_column
    if col not in utt_df.columns:
        logger.warning(
            "disentangle requested but column '%s' missing; skipping.", col
        )
        return embeddings
    labels = utt_df[col].astype("category")
    if labels.nunique() < 2:
        logger.warning("disentangle: <2 content classes; skipping.")
        return embeddings

    onehot = pd.get_dummies(labels).to_numpy(dtype=np.float64)
    # Center embeddings; ridge-regress each dimension on the one-hot content.
    X = onehot
    Y = embeddings.astype(np.float64)
    alpha = float(cfg.disentangle_ridge_alpha)
    # W = (X^T X + alpha I)^-1 X^T Y
    XtX = X.T @ X + alpha * np.eye(X.shape[1])
    W = np.linalg.solve(XtX, X.T @ Y)
    pred = X @ W
    residual = (Y - pred).astype(np.float32)
    logger.info(
        "Disentangled content on %d classes; residual variance ratio=%.3f",
        labels.nunique(),
        float(residual.var() / (embeddings.var() + 1e-12)),
    )
    return residual
