# file: tests/test_pipeline_and_reporting.py
"""End-to-end pipeline + reporting tests using the dummy backbone (fully offline)."""
from __future__ import annotations

import json
from pathlib import Path

from acoustic_gap.config import AppConfig
from acoustic_gap.pipeline import AcousticGapPipeline
from acoustic_gap.reporting import GapResult, build_report, save_report, summary_table


def _config(datasets, out_dir) -> AppConfig:
    return AppConfig.from_dict({
        "real_dir": str(datasets["real"]),
        "sim_dir": str(datasets["sim"]),
        "preprocess": {"window_seconds": 1.0, "hop_seconds": 1.0},
        "features": {"backbones": ["dummy"]},
        "distances": {"metrics": ["fad", "mmd", "wasserstein"],
                      "wasserstein_backend": "sliced", "sliced_n_projections": 32},
        "report": {"output_dir": str(out_dir), "make_plots": False,
                   "headline_backbone": "dummy", "headline_metric": "wasserstein"},
    })


def test_end_to_end_pipeline(synthetic_datasets, tmp_path):
    out = tmp_path / "report"
    cfg = _config(synthetic_datasets, out)
    report = AcousticGapPipeline(cfg).run()

    assert "top_level_scores" in report
    assert "dummy" in report["top_level_scores"]
    assert "per_condition_breakdown" in report
    # report files written
    assert (out / "report.json").exists()
    assert (out / "results.csv").exists()
    loaded = json.loads((out / "report.json").read_text())
    assert loaded["headline"]["largest_contributor"] is not None


def test_largest_contributor_is_noise(synthetic_datasets, tmp_path):
    # The synthetic 'noise' condition has the biggest sim perturbation, so it
    # should be flagged as the largest contributor.
    out = tmp_path / "report2"
    cfg = _config(synthetic_datasets, out)
    report = AcousticGapPipeline(cfg).run()
    contributor = report["headline"]["largest_contributor"]
    assert contributor["condition"] == "noise"


def test_build_report_and_summary_table():
    results = [
        GapResult("wavlm", "__overall__", "wasserstein", 1.0, 10, 10),
        GapResult("wavlm", "clean", "wasserstein", 0.2, 5, 5),
        GapResult("wavlm", "noise", "wasserstein", 0.9, 5, 5),
    ]
    from acoustic_gap.config import ReportConfig
    rep = build_report(results, ReportConfig(headline_backbone="wavlm",
                                             headline_metric="wasserstein"))
    assert rep["headline"]["largest_contributor"]["condition"] == "noise"
    tbl = summary_table(rep)
    assert len(tbl) == 3
    assert "value" in tbl.columns
