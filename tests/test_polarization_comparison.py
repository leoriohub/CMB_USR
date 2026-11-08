"""Tests for the EE/TE polarization comparison (CAMB-free)."""
import json
import os

import pytest

from scripts import polarization_comparison as pc


def test_dataset_registry_is_consistent():
    names = [d[0] for d in pc.DATASETS]
    assert len(names) == len(set(names))
    kinds = {d[2] for d in pc.DATASETS}
    assert kinds == {"tt", "ee", "te"}
    # every kind has a panel label
    for kind in kinds:
        assert kind in pc.PANEL_TEXT


def test_load_config_rejects_missing_configs_list(tmp_path):
    path = tmp_path / "bad.json"
    path.write_text(json.dumps({"tag": "x"}))
    with pytest.raises(ValueError, match="configs"):
        pc.load_config(str(path))


def test_load_config_rejects_empty_list(tmp_path):
    path = tmp_path / "empty.json"
    path.write_text(json.dumps({"configs": []}))
    with pytest.raises(ValueError, match="configs"):
        pc.load_config(str(path))


def test_load_config_rejects_entry_missing_keys(tmp_path):
    path = tmp_path / "entry.json"
    path.write_text(json.dumps({"configs": [{"label": "a", "phi0": 6.4}]}))
    with pytest.raises(ValueError, match="missing 'y0'"):
        pc.load_config(str(path))


def test_load_config_accepts_bundled_tau_config():
    cfg = pc.load_config(os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "tau_profile", "higgs_usr_good.json",
    ))
    assert len(cfg["configs"]) == 4
    assert cfg["_path"].endswith("higgs_usr_good.json")


def test_summary_row_arithmetic():
    res = {
        "label": "x", "phi0": 6.4, "y0": -0.475, "n_star": 59.0,
        "p_s_file": "ps_x.json",
        "chi2": [
            {"dataset": "tt_commander", "kind": "tt", "n": 28,
             "chi2_model": 20.0, "chi2_lcdm": 21.0, "delta": -1.0},
            {"dataset": "tt_binned", "kind": "tt", "n": 83,
             "chi2_model": 80.0, "chi2_lcdm": 85.0, "delta": -5.0},
            {"dataset": "ee_low", "kind": "ee", "n": 28,
             "chi2_model": 30.0, "chi2_lcdm": 30.5, "delta": -0.5},
            {"dataset": "ee_binned", "kind": "ee", "n": 66,
             "chi2_model": 70.0, "chi2_lcdm": 71.0, "delta": -1.0},
            {"dataset": "te_binned", "kind": "te", "n": 66,
             "chi2_model": 90.0, "chi2_lcdm": 88.0, "delta": 2.0},
        ],
    }
    s = pc.summary_row(res)
    assert s["chi2_tt"] == pytest.approx(100.0)
    assert s["chi2_tt_lcdm"] == pytest.approx(106.0)
    assert s["dchi2_tt"] == pytest.approx(-6.0)
    assert s["chi2_ee"] == pytest.approx(100.0)
    assert s["dchi2_ee"] == pytest.approx(-1.5)
    assert s["chi2_te"] == pytest.approx(90.0)
    assert s["dchi2_te"] == pytest.approx(2.0)
    assert s["p_s_file"] == "ps_x.json"
