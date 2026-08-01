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


def test_empty_side_raises_clear_error(synthetic_datasets, tmp_path):
    # Point sim at an empty directory -> fail fast with a helpful message.
    import pytest

    empty = tmp_path / "empty_sim"
    empty.mkdir()
    cfg = _config({"real": synthetic_datasets["real"], "sim": empty}, tmp_path / "r")
    with pytest.raises(RuntimeError, match="produced 0 usable segments"):
        AcousticGapPipeline(cfg).run()


def test_mismatched_conditions_reports_overall_only(tmp_path):
    # Real and sim have no shared condition labels: overall works, no breakdown,
    # and it must NOT crash / must still write a report.
    import numpy as np
    import soundfile as sf

    sr = 16000
    def tone(f):
        t = np.linspace(0, 2.0, int(2.0 * sr), endpoint=False)
        return (0.5 * np.sin(2 * np.pi * f * t)).astype(np.float32)

    d = tmp_path / "data"
    for i in range(4):
        rp = d / "real" / "clean" / f"r{i}.wav"
        rp.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(rp), tone(200 + 20 * i), sr)
        sp = d / "sim" / "synthetic" / f"s{i}.wav"
        sp.parent.mkdir(parents=True, exist_ok=True)
        sf.write(str(sp), tone(260 + 20 * i), sr)
    out = tmp_path / "rep"
    cfg = _config({"real": d / "real", "sim": d / "sim"}, out)
    report = AcousticGapPipeline(cfg).run()
    assert (out / "report.json").exists()
    assert report["top_level_scores"]           # overall computed
    assert report["per_condition_breakdown"] == {}  # nothing matched


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
