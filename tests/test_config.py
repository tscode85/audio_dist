# file: tests/test_config.py
"""Tests for config loading/validation."""
from __future__ import annotations

import json

import pytest

from acoustic_gap.config import AppConfig


def test_roundtrip_json(tmp_path):
    cfg = AppConfig()
    p = tmp_path / "c.json"
    cfg.save(p)
    loaded = AppConfig.load(p)
    assert loaded.preprocess.target_sample_rate == cfg.preprocess.target_sample_rate
    assert loaded.features.backbones == cfg.features.backbones


def test_from_dict_nested():
    cfg = AppConfig.from_dict({
        "real_dir": "/a", "sim_dir": "/b",
        "features": {"backbones": ["wavlm"]},
        "distances": {"metrics": ["mmd"]},
    })
    assert cfg.real_dir == "/a"
    assert cfg.features.backbones == ["wavlm"]
    assert cfg.distances.metrics == ["mmd"]


def test_validate_rejects_bad_values():
    with pytest.raises(ValueError):
        AppConfig.from_dict({"features": {"backbones": ["bogus"]}}).validate()
    with pytest.raises(ValueError):
        AppConfig.from_dict({"preprocess": {"hop_seconds": 5, "window_seconds": 1}}).validate()
    with pytest.raises(ValueError):
        AppConfig.from_dict({"distances": {"metrics": ["nope"]}}).validate()
