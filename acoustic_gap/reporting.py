# file: acoustic_gap/reporting.py
"""Aggregation, reporting and plotting of domain-gap results.

Consumes per-(backbone, condition, metric) distances and produces:

* a tidy results DataFrame,
* a structured JSON report with a top-level score and a per-condition breakdown,
* a flag identifying the single largest contributor to the sim-to-real gap,
* an optional bar chart comparing per-condition distances.

All metrics are lower-is-better; this is annotated in every output.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .config import ReportConfig

logger = logging.getLogger(__name__)

LOWER_IS_BETTER_NOTE = (
    "All distances are LOWER-IS-BETTER: 0 means the simulated distribution is "
    "indistinguishable from the real one; larger values mean a larger domain gap."
)


@dataclass
class GapResult:
    """One measured distance for a (backbone, condition) cell."""

    backbone: str
    condition: str
    metric: str
    value: float
    n_real: int
    n_sim: int


def results_to_frame(results: List[GapResult]) -> pd.DataFrame:
    return pd.DataFrame([r.__dict__ for r in results])


def _resolve_headline(
    df: pd.DataFrame, backbone: str, metric: str
) -> tuple[Optional[str], Optional[str]]:
    """Resolve the headline (backbone, metric), falling back if unavailable.

    Keeps the "always flag a contributor" guarantee even when the configured
    headline backbone/metric wasn't actually run.
    """
    backbones = list(df["backbone"].unique())
    resolved_bb = backbone if backbone in backbones else (backbones[0] if backbones else None)
    if resolved_bb is None:
        return None, None
    metrics = list(df[df["backbone"] == resolved_bb]["metric"].unique())
    resolved_m = metric if metric in metrics else (metrics[0] if metrics else None)
    return resolved_bb, resolved_m


def _largest_contributor(
    df: pd.DataFrame, backbone: str, metric: str
) -> Optional[Dict[str, Any]]:
    """Find the worst per-condition cell for the headline (backbone, metric)."""
    backbone, metric = _resolve_headline(df, backbone, metric)
    if backbone is None or metric is None:
        return None
    sub = df[
        (df["backbone"] == backbone)
        & (df["metric"] == metric)
        & (df["condition"] != "__overall__")
    ].dropna(subset=["value"])
    if sub.empty:
        return None
    row = sub.loc[sub["value"].idxmax()]
    return {
        "condition": str(row["condition"]),
        "backbone": backbone,
        "metric": metric,
        "value": float(row["value"]),
    }


def build_report(
    results: List[GapResult],
    cfg: ReportConfig,
) -> Dict[str, Any]:
    """Assemble the structured report dict from raw results."""
    df = results_to_frame(results)

    # Top-level scores: the __overall__ condition per (backbone, metric).
    overall = df[df["condition"] == "__overall__"]
    top_level: Dict[str, Dict[str, float]] = {}
    for _, row in overall.iterrows():
        top_level.setdefault(row["backbone"], {})[row["metric"]] = (
            None if pd.isna(row["value"]) else float(row["value"])
        )

    # Per-condition breakdown.
    breakdown: Dict[str, Dict[str, Dict[str, float]]] = {}
    for (backbone, condition), grp in df[df["condition"] != "__overall__"].groupby(
        ["backbone", "condition"]
    ):
        d = {
            r["metric"]: (None if pd.isna(r["value"]) else float(r["value"]))
            for _, r in grp.iterrows()
        }
        breakdown.setdefault(backbone, {})[condition] = d

    resolved_bb, resolved_m = _resolve_headline(
        df, cfg.headline_backbone, cfg.headline_metric
    )
    contributor = _largest_contributor(df, cfg.headline_backbone, cfg.headline_metric)

    report = {
        "interpretation": LOWER_IS_BETTER_NOTE,
        "headline": {
            "backbone": resolved_bb if resolved_bb is not None else cfg.headline_backbone,
            "metric": resolved_m if resolved_m is not None else cfg.headline_metric,
            "largest_contributor": contributor,
        },
        "top_level_scores": top_level,
        "per_condition_breakdown": breakdown,
    }
    return report


def save_report(
    results: List[GapResult],
    cfg: ReportConfig,
) -> Dict[str, Any]:
    """Write JSON + CSV (+ optional plots) to ``cfg.output_dir`` and return the report."""
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = results_to_frame(results)
    df.to_csv(out_dir / "results.csv", index=False)

    report = build_report(results, cfg)
    (out_dir / "report.json").write_text(json.dumps(report, indent=2))

    contributor = report["headline"]["largest_contributor"]
    if contributor:
        logger.info(
            "Largest domain-gap contributor: condition='%s' (%s/%s = %.4f)",
            contributor["condition"], contributor["backbone"],
            contributor["metric"], contributor["value"],
        )

    if cfg.make_plots:
        try:
            _plot_per_condition(df, cfg, out_dir)
        except Exception as exc:  # pragma: no cover - plotting optional
            logger.warning("Plotting failed: %s", exc)

    logger.info("Report written to %s", out_dir)
    return report


def _plot_per_condition(df: pd.DataFrame, cfg: ReportConfig, out_dir: Path) -> None:
    """Bar chart of per-condition distances for the headline metric."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    metric = cfg.headline_metric
    sub = df[(df["metric"] == metric) & (df["condition"] != "__overall__")]
    if sub.empty:
        return
    backbones = sorted(sub["backbone"].unique())
    conditions = sorted(sub["condition"].unique())
    x = np.arange(len(conditions))
    width = 0.8 / max(len(backbones), 1)

    fig, ax = plt.subplots(figsize=(max(6, len(conditions) * 1.5), 4))
    for i, bb in enumerate(backbones):
        vals = [
            sub[(sub["backbone"] == bb) & (sub["condition"] == c)]["value"].mean()
            for c in conditions
        ]
        ax.bar(x + i * width, vals, width, label=bb)
    ax.set_xticks(x + width * (len(backbones) - 1) / 2)
    ax.set_xticklabels(conditions, rotation=30, ha="right")
    ax.set_ylabel(f"{metric} (lower = smaller gap)")
    ax.set_title("Sim-to-real acoustic domain gap by condition")
    ax.legend(title="backbone")
    fig.tight_layout()
    fig.savefig(out_dir / "per_condition_gap.png", dpi=120)
    plt.close(fig)


def summary_table(report: Dict[str, Any]) -> pd.DataFrame:
    """Flatten the report into a compact human-readable summary DataFrame."""
    rows = []
    for backbone, metrics in report["top_level_scores"].items():
        for metric, value in metrics.items():
            rows.append(
                {"scope": "overall", "backbone": backbone, "condition": "__overall__",
                 "metric": metric, "value": value}
            )
    for backbone, conds in report["per_condition_breakdown"].items():
        for condition, metrics in conds.items():
            for metric, value in metrics.items():
                rows.append(
                    {"scope": "condition", "backbone": backbone,
                     "condition": condition, "metric": metric, "value": value}
                )
    return pd.DataFrame(rows)
