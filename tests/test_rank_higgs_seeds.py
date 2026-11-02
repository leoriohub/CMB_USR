"""Synthetic coverage for full-chi2 seed selection; no solver evaluations."""
import hashlib
import json

import pytest

from scripts import rank_higgs_seeds as ranker


def record(**updates):
    result = {"_type": "data", "status": "ok", "phi0": 6.6167,
              "y0": -0.675, "N_star": 59.0, "k_dip": 0.00015941590374559997,
              "D_ell_full": [ranker.D2_LCDM * 0.75, 100, 999999],
              "d2": 999999, "chi2_binned": 82.0, "chi2_lcdm_binned": 81.0,
              "chi2_unbinned": 2572.0, "chi2_lcdm_unbinned": 2573.0}
    result.update(updates)
    return result


def write_log(path, records):
    path.write_text("\n".join(json.dumps(row) for row in records) + "\n")
    return path


def test_ranking_filters_precision_and_provenance(tmp_path):
    path = write_log(tmp_path / "scan.jsonl", [
        {"_type": "header"}, record(status="pipe_error"),
        record(chi2_unbinned=2575), record(),
        record(D_ell_full=[ranker.D2_LCDM]),
        record(chi2_binned=92), record(chi2_unbinned=2589),
        record(D_ell_full=[]), record(chi2_unbinned=float("nan")),
        record(chi2_binned=None), record(phi0=True),
    ])
    result = ranker.rank_seeds([path])
    assert result["counts"] == {"records": 11, "headers": 1, "failed": 1,
                                "invalid": 4, "filtered": 3, "selected": 2}
    assert [seed["dchi2_total"] for seed in result["seeds"]] == [-1, 2]
    first = result["seeds"][0]
    assert first["phi0"] == 6.6167
    assert first["k_dip"] == 0.00015941590374559997
    assert first["d2_suppression"] == 25
    assert first["d2"] == ranker.D2_LCDM * 0.75
    assert first["source_line"] == 4
    assert result["inputs"][0]["sha256"] == hashlib.sha256(path.read_bytes()).hexdigest()
    assert result["ranker_sha256"] == ranker.sha256_file(ranker.__file__)
    assert result["filters"] == {"min_d2_suppression": 15.,
                                  "max_dchi2_high": 10., "max_dchi2_total": 15.}


def test_inclusive_thresholds_and_nondefault_filters(tmp_path):
    path = write_log(tmp_path / "scan.jsonl", [record(
        D_ell_full=[ranker.D2_LCDM * 0.75], chi2_binned=91, chi2_unbinned=2588)])
    assert len(ranker.rank_seeds([path], min_d2_suppression=25)["seeds"]) == 1
    assert not ranker.rank_seeds([path], min_d2_suppression=25.01)["seeds"]
    assert not ranker.rank_seeds([path], max_dchi2_high=9.99)["seeds"]
    assert not ranker.rank_seeds([path], max_dchi2_total=14.99)["seeds"]
    with pytest.raises(ValueError, match="finite"):
        ranker.rank_seeds([path], max_dchi2_total=float("inf"))


def test_globs_order_dedup_inputs_and_ties(tmp_path):
    a = write_log(tmp_path / "a.jsonl", [record(phi0=6.7), record(phi0=6.6)])
    b = write_log(tmp_path / "b.jsonl", [record(phi0=6.6)])
    one = ranker.rank_seeds([b, a])
    two = ranker.rank_seeds([str(tmp_path / "*.jsonl"), a])
    assert one == two
    assert len(one["inputs"]) == 2
    assert len(one["seeds"]) == 3  # Records retained, even repeated initial conditions.
    assert [r["phi0"] for r in one["seeds"]] == [6.6, 6.6, 6.7]


def test_invalid_inputs_fail_clearly(tmp_path):
    with pytest.raises(ValueError, match="No logs match"):
        ranker.rank_seeds([tmp_path / "missing*.jsonl"])
    with pytest.raises(ValueError, match="At least one"):
        ranker.rank_seeds([])
    malformed = tmp_path / "bad.jsonl"
    malformed.write_text('{"truncated":')
    with pytest.raises(ValueError, match=r"bad.jsonl:1"):
        ranker.rank_seeds([malformed])


def test_low_ell_only_records_never_substitute_for_full_metrics(tmp_path):
    low = record()
    del low["chi2_unbinned"]
    low.update(chi2=1, chi2_lcdm=10)
    path = write_log(tmp_path / "low.jsonl", [low])
    result = ranker.rank_seeds([path])
    assert result["counts"]["invalid"] == 1
    assert result["seeds"] == []


def test_cli_writes_deterministic_json_and_table(tmp_path, capsys):
    path = write_log(tmp_path / "scan.jsonl", [record()])
    out = tmp_path / "ranking.json"
    args = ["--logs", str(path), "--out", str(out)]
    assert ranker.main(args) == 0
    first = out.read_bytes()
    assert ranker.main(args) == 0
    assert out.read_bytes() == first
    assert "6.6167 | -0.675 | 59.0" in capsys.readouterr().out
    assert json.loads(first)["seeds"][0]["dchi2_total"] == -1
    with pytest.raises(SystemExit):
        ranker.main(["--logs", str(path), "--out", str(path)])
    assert path.read_text().startswith('{"_type": "data"')


def test_concurrent_input_mutation_rejected(tmp_path, monkeypatch):
    path = write_log(tmp_path / "scan.jsonl", [record()])
    hashes = iter(["before", "after"])
    monkeypatch.setattr(ranker, "sha256_file", lambda path: next(hashes))
    with pytest.raises(ValueError, match="Input changed"):
        ranker.rank_seeds([path])


def test_legacy_input_requires_opt_in(tmp_path, monkeypatch):
    digest = "deadbeef"
    monkeypatch.setitem(ranker.LEGACY_INPUTS, digest, "legacy end rule")
    monkeypatch.setattr(ranker, "sha256_file", lambda path: digest)
    path = tmp_path / "legacy.jsonl"
    path.write_text(json.dumps({"_type": "header"}) + "\n")
    with pytest.raises(ValueError, match="LEGACY"):
        ranker.rank_seeds([path])
    result = ranker.rank_seeds([path], allow_legacy=True)
    assert result["data_status"] == "legacy"
    assert result["legacy_inputs"][0]["sha256"] == digest
