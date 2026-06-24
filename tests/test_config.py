"""Tests for configuration management."""

from pystrain2.config import Config, deep_merge


def test_deep_merge_simple():
    base = {"a": 1, "b": {"c": 2, "d": 3}}
    override = {"b": {"d": 4}}
    merged = deep_merge(base, override)
    assert merged["a"] == 1
    assert merged["b"]["c"] == 2
    assert merged["b"]["d"] == 4


def test_config_defaults():
    cfg = Config()
    assert cfg["data"]["format"] == "auto"
    assert cfg["outlier_detection"]["enable"] is True
    assert cfg["algorithms"]["grid"]["min_sites"] == 6


def test_config_from_overrides():
    cfg = Config(overrides={"data": {"vel_file": "/tmp/vel.gmtvec"}})
    assert cfg["data"]["vel_file"] == "/tmp/vel.gmtvec"
    assert cfg["data"]["format"] == "auto"


def test_config_to_dict():
    cfg = Config()
    d = cfg.to_dict()
    assert "data" in d
    assert d["data"]["format"] == "auto"
