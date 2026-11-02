"""Fast inference contract tests: no cosmological solver execution."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest

from scripts import mcmc_higgs as m


@pytest.fixture
def config():
    return json.loads((Path(__file__).resolve().parents[1] /
                       "configs/mcmc/higgs_full.json").read_text())


def gaussian_evaluate(theta, config):
    center = np.array([config["init"][p] for p in m.PARAMETERS])
    return {"status": "ok", "chi2": float(np.sum(((theta-center)/.05)**2))}


def mock_paths(tmp_path):
    return {"archive": tmp_path / "chain.npz", "checkpoint": tmp_path / "chain.h5",
            "log": tmp_path / "chain.jsonl"}


def test_config_and_smoke(config):
    assert m.validate_config(config) is config
    smoke = m.effective_config(config, True)
    assert (smoke["ell_max"], smoke["full_chi2"], smoke["nwalkers"], smoke["nsteps"]) == (30, False, 8, 20)
    assert config["ell_max"] == 2508
    assert m.build_evaluation_args(config).workers == 1
    grid = m.k_grid_from_config(config)
    assert len(grid) == 80 and grid[0] == config["k_min"] and grid[-1] == 1


@pytest.mark.parametrize("field,value", [("ell_max", 2500), ("nwalkers", 7),
    ("seed", -1), ("workers", 0), ("tag", "../escape"), ("k_min", float("nan")),
    ("checkpoint_interval", 1.5), ("num_k", 9)])
def test_bad_config(config, field, value):
    config[field] = value
    with pytest.raises(ValueError):
        m.validate_config(config)


def test_prior_short_circuits(config, monkeypatch):
    monkeypatch.setattr(m, "evaluate_point", lambda *_: pytest.fail("prior must reject before solver"))
    for theta in ([6.3, -.675, 59], [np.nan, -.675, 59], [6.6, -.675], [6.6, -1, 59]):
        assert m.log_probability(theta, config) == -np.inf


@pytest.mark.parametrize("result,expected", [({"status": "ok", "chi2": 20.46}, -10.23),
    ({"status": "pipe_error"}, -np.inf), ({"status": "ok", "chi2": np.nan}, -np.inf)])
def test_probability(config, monkeypatch, result, expected):
    monkeypatch.setattr(m, "evaluate_point", lambda *_: result)
    assert m.log_probability([6.6167, -.675, 59], config) == expected


def test_actual_api_delegation(config, monkeypatch):
    calls = []
    def fake(*theta, **kwargs):
        calls.append((theta, kwargs))
        return {"status": "ok", "chi2": 42}
    monkeypatch.setitem(sys.modules, "scripts.camb_scan", types.SimpleNamespace(evaluate_config=fake))
    assert m.evaluate_point([6.6, -.675, 59], config)["chi2"] == 42
    assert calls[0][0][3].ell_max == 2508
    assert calls[0][0][3].workers == 1
    np.testing.assert_array_equal(calls[0][1]["k_phys_grid"], m.k_grid_from_config(config))


def test_initialization(config):
    a, b = m.initial_positions(config), m.initial_positions(config)
    np.testing.assert_array_equal(a, b)
    assert a.shape == (32, 3)
    assert all(np.isfinite(m.log_prior(p, config)) for p in a)


def test_archive_diagnostics(config, tmp_path, monkeypatch):
    chain = np.tile(m.initial_positions(config)[None, :, :], (4, 1, 1))
    logs = np.array([[m.log_prior(theta, config) - .5 * gaussian_evaluate(theta, config)["chi2"]
                      for theta in step] for step in chain])
    metadata = {"effective_config": config, "config_hash": m.config_hash(config)}
    first, second = tmp_path / "a.npz", tmp_path / "b.npz"
    for path in (first, second):
        m.save_archive(path, chain, logs, np.ones(32), config["seed"], metadata)
    loaded = m.load_archive(first)
    assert loaded["chain"].shape == (4, 32, 3)
    assert m.compare_archives(first, second)["equal"]
    summary = m.summarize_archive(first, 1, 2)
    assert summary["n_samples"] == 64
    assert len(summary["quantiles"]["phi0"]) == 3
    assert summary["mean_acceptance_fraction"] == .25
    assert summary["nonfinite_log_prob_count"] == 0
    monkeypatch.setattr(m, "evaluate_point", gaussian_evaluate)
    validated = m.validate_archive(first, 1, 2)
    assert validated["equal"] and len(validated["records"]) == 20
    logs[1, 0] += 1
    m.save_archive(second, chain, logs, np.ones(32), config["seed"], metadata)
    assert not m.compare_archives(first, second)["equal"]
    assert not m.validate_archive(second, 1, 2)["equal"]
    with pytest.raises(ValueError):
        m.posterior_samples(chain, 4, 1)
    with pytest.raises(ValueError):
        m.posterior_samples(chain, 0, 0)


def test_mocked_dry_run(config, tmp_path, monkeypatch, capsys):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    monkeypatch.setattr(m, "evaluate_point", gaussian_evaluate)
    assert m.main(["--config", str(path), "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["result"]["chi2"] == 0


def test_audit_init_fixed_coordinate_pure_json(config, tmp_path, monkeypatch, capsys):
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    calls = []
    def evaluate(theta, recipe):
        print("solver progress, not JSON")
        calls.append((list(theta), deepcopy(recipe)))
        return dict(status="ok", chi2=recipe["num_k"] * .1, d2=800., k_dip=1e-4,
                    N_total=61.5, chi2_low=20., D_ell=[1, 2, 3], P_S=np.ones(3))
    monkeypatch.setattr(m, "evaluate_point", evaluate)
    assert m.main(["--config", str(path), "--audit-init"]) == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert captured.err.count("solver progress") == 3
    assert result["success"]
    assert [recipe["num_k"] for _, recipe in calls] == [80, 160, 320]
    assert all(theta == [6.6167, -.675, 59.] for theta, _ in calls)
    assert config["num_k"] == 80
    for case in result["cases"]:
        assert "D_ell" not in case["result"] and "P_S" not in case["result"]
        assert case["result"]["N_total"] == 61.5
        metadata = case["metadata"]
        assert metadata["config_hash"] == m.config_hash(metadata["effective_config"])
        assert "scripts/mcmc_higgs.py" in metadata["source_hashes"]
    assert result["cases"][2]["difference_from_base"]["chi2"] == 24.


def test_audit_failed_evaluation(config, monkeypatch):
    monkeypatch.setattr(m, "evaluate_point", lambda *_: {"status": "pipe_error", "error": "mock"})
    result = m.audit_init(config)
    assert not result["success"] and len(result["cases"]) == 3
    assert all(case["result"]["error"] == "mock" for case in result["cases"])


def test_archived_spectrum_factorial_mocked(config, tmp_path, monkeypatch, capsys):
    from scripts.constants import As, k_pivot_phys
    header_grid = np.geomspace(1e-5, 1., 10)
    ps = np.linspace(1., 2., 11)
    theta = {p: config["init"][p] for p in m.PARAMETERS}
    archive, reference = tmp_path / "archived.jsonl", tmp_path / "reference.jsonl"
    header = dict(_type="header", k_phys=header_grid.tolist(), k_pivot_phys=k_pivot_phys, As=As)
    record = dict(_type="data", status="ok", P_S=ps.tolist(), D_ell=[1., 1., 1.],
                  N_total=62.4, **theta)
    archive.write_text(json.dumps(header) + "\n" + json.dumps(record) + "\n")
    reference.write_text(json.dumps(dict(status="ok", D_ell_full=[1.] * 2507,
                                         chi2_unbinned=2571.7, **theta)) + "\n")
    calls = []
    def curves(data, ell_max):
        print("CAMB mocked progress")
        calls.append((data, ell_max))
        ells = np.arange(2, ell_max + 1)
        dell = np.full(len(ells), 1.)
        return 1., ells, dell, dell, dell, dell
    monkeypatch.setitem(sys.modules, "pspectrum_pipeline", types.SimpleNamespace(
        ensure_k_pivot=lambda grid, pivot: (np.sort(np.append(grid, pivot)), int(np.searchsorted(grid, pivot)))))
    monkeypatch.setitem(sys.modules, "scripts.camb_scan", types.SimpleNamespace(
        compute_camb_curves=curves, lcdm_baseline=lambda ell: (np.arange(2, ell+1), np.ones(ell-1)),
        chi2_vs_planck=lambda *_: 20.))
    monkeypatch.setitem(sys.modules, "scripts.planck_data", types.SimpleNamespace(C_ell_to_d_ell=lambda _, cl: cl))
    monkeypatch.setitem(sys.modules, "scripts.chi2_analysis", types.SimpleNamespace(
        chi2_unbinned=lambda *_: (2571.7, 2573.04, 2507),
        chi2_binned=lambda *_: (82.47, 81.84, 83)))
    monkeypatch.setitem(sys.modules, "scripts.optimizer_utils", types.SimpleNamespace(find_k_dip=lambda *_: 1e-4))
    monkeypatch.setattr(m, "evaluate_point", lambda *_: pytest.fail("archive audit must not run MS solver"))
    cfg = tmp_path / "config.json"
    cfg.write_text(json.dumps(config))
    assert m.main(["--config", str(cfg), "--audit-archive", "--archive-log", str(archive),
                   "--reference-log", str(reference)]) == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert captured.err.count("CAMB mocked progress") == 4
    assert result["constants_match"] and result["success"]
    assert [ell for _, ell in calls] == [2500, 2508, 2500, 2508]
    assert [len(data["k_phys"]) for data, _ in calls] == [11, 11, 10, 10]
    np.testing.assert_array_equal(calls[0][0]["P_S"], ps)
    np.testing.assert_array_equal(calls[2][0]["P_S"], ps[:10])
    assert result["metadata"]["archive_inputs"]["archive"]["physical_line"] == 2
    assert len(result["metadata"]["archive_inputs"]["reference"]["sha256"]) == 64
    assert result["cases"][0]["reference_spectrum_comparison"]["n_compared"] == 2499
    assert not result["cases"][0]["reference_spectrum_comparison"]["full_reference_coverage"]
    assert result["cases"][1]["reference_spectrum_comparison"]["exactly_equal"]
    assert result["cases"][1]["reference_chi2_difference"] == 0
    assert result["cases"][2]["pairing"] == "legacy_truncation_emulation"
    replay_path = tmp_path / "replay.json"
    replay_path.write_text(json.dumps(dict(_type="result", format_version=2,
        metadata=dict(x0=theta["phi0"], y0=theta["y0"], N_star=theta["N_star"], N_total=62.3747),
        spectrum=dict(k_phys=np.sort(np.append(header_grid, k_pivot_phys)).tolist(),
                      P_S=(ps * 1.001).tolist()))))
    replay_result = m.audit_archive(config, archive, reference, replay_path)
    assert len(replay_result["cases"]) == 6
    assert replay_result["cases"][-1]["pairing"] == "replay_correct"
    assert replay_result["replay_comparison"]["k_phys"]["exactly_equal"]
    assert replay_result["replay_comparison"]["P_S"]["max_relative_difference"] == pytest.approx(.001)
    assert "sha256" in replay_result["metadata"]["archive_inputs"]["replay"]
    # Ambiguous records must never be silently selected.
    archive.write_text(archive.read_text() + json.dumps(record) + "\n")
    with pytest.raises(ValueError, match="expected one exact init match"):
        m.audit_archive(config, archive, reference)


def test_background_fixed_factorial_mocked(config, tmp_path, monkeypatch, capsys):
    calls = []
    class FakeHiggs:
        def __init__(self, lam, xi):
            self.lam, self.xi_val = lam, xi
    def evaluate(*args, **kwargs):
        print("mock background progress")
        model = kwargs["model"]
        calls.append((args, kwargs))
        return dict(status="ok", chi2=2500. + model.bg_steps/1000,
                    N_total=62.4 if model.T_max == 500 else 61.5,
                    D_ell=[800., 900.], P_S=[2e-9] * 321)
    monkeypatch.setitem(sys.modules, "models", types.SimpleNamespace(HiggsModel=FakeHiggs))
    monkeypatch.setitem(sys.modules, "scripts.camb_scan", types.SimpleNamespace(evaluate_config=evaluate))
    monkeypatch.setitem(sys.modules, "pspectrum_pipeline", types.SimpleNamespace(
        ensure_k_pivot=lambda grid, pivot: (np.sort(np.append(grid, pivot)), int(np.searchsorted(grid, pivot)))))
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    assert m.main(["--config", str(path), "--audit-background"]) == 0
    captured = capsys.readouterr()
    result = json.loads(captured.out)
    assert captured.err.count("mock background progress") == 4
    assert result["success"] and result["num_k"] == 320
    assert [(kwargs["model"].T_max, kwargs["model"].bg_steps) for _, kwargs in calls] == [
        (500., 1000), (500., 50000), (2000., 1000), (2000., 50000)]
    assert len({id(kwargs["model"]) for _, kwargs in calls}) == 4
    for case, (args, kwargs) in zip(result["cases"], calls):
        assert args[:3] == (6.6167, -.675, 59.)
        assert args[3].num_k == 320 and args[3].workers == 1
        assert len(kwargs["k_phys_grid"]) == 320
        assert len(case["k_phys"]) == 321
        assert case["actual_model_attributes"]["x0"] == 6.6167
        assert case["actual_model_attributes"]["bg_steps"] == kwargs["model"].bg_steps
        assert case["result"]["D_ell"] == [800., 900.]
        assert len(case["result"]["P_S"]) == 321
        metadata = case["metadata"]
        assert metadata["config_hash"] == m.config_hash(metadata["effective_config"])
        assert "scripts/mcmc_higgs.py" in metadata["source_hashes"]
    assert config["num_k"] == 80


def test_audit_end_rule_isolation_mocked(config, tmp_path, monkeypatch, capsys):
    """One trajectory; legacy last-crossing must be later than current rule."""
    n = 1200
    N = np.linspace(0., 65., n)
    eps = np.full(n, 1e-3)
    eps[400] = 1.5          # current: first permanent crossing
    eps[900] = 2.0          # legacy: last crossing in reheating
    eps[901:] = 2.0

    class FakeHiggs:
        def __init__(self, lam, xi):
            self.lam, self.xi_val = lam, xi
            self.T_max, self.bg_steps = 2000., 50000

    calls = {}
    def run_background(model, T_span):
        calls["T_max"] = model.T_max
        calls["bg_steps"] = model.bg_steps
        calls["n"] = len(T_span)
        return np.zeros((4, n))

    monkeypatch.setitem(sys.modules, "models", types.SimpleNamespace(HiggsModel=FakeHiggs))
    monkeypatch.setitem(sys.modules, "inf_dyn_background", types.SimpleNamespace(
        run_background_simulation=run_background,
        get_derived_quantities=lambda sol, model: {"epsH": eps, "N": N}))
    monkeypatch.setitem(sys.modules, "pspectrum_pipeline", types.SimpleNamespace(
        find_end_of_inflation=lambda e: 400,
        get_k_pivot_code=lambda sol, der, idx, nstar, N_total=None: (
            1e-5, int(idx), N_total)))
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config))
    assert m.main(["--config", str(path), "--audit-end",
                   "--audit-tmax", "500", "--audit-bg-steps", "1000"]) == 0
    result = json.loads(capsys.readouterr().out)
    assert calls == {"T_max": 500.0, "bg_steps": 1000, "n": 1000}
    assert result["current_rule"]["end_index"] == 400
    assert result["legacy_rule"]["end_index"] == 1199
    assert result["legacy_minus_current_N_total"] > 0
    assert [c for c in result["upward_crossing_indices"]] == [400, 900]


def test_saved_spectrum_audit_and_comparison_mocked(config, tmp_path, monkeypatch, capsys):
    from scripts.constants import As, k_pivot_phys
    grid, ps = np.geomspace(1e-5, 1., 12), np.ones(12) * 2e-9
    paths = [tmp_path / "fortran.json", tmp_path / "numba.json"]
    for index, path in enumerate(paths):
        path.write_text(json.dumps(dict(metadata=dict(x0=6.6167, y0=-.675, N_star=59.,
            k_pivot_phys=k_pivot_phys, As_target=As, N_total=61.5, N_pivot=2.5, scale_factor=10.),
            spectrum=dict(k_phys=grid.tolist(), P_S=(ps*(index+1)).tolist(),
                          P_S_raw=(ps*(index+1)/10).tolist()))))
    calls = []
    def camb(grid_arg, ps_arg, ell_max):
        print("mock saved CAMB")
        calls.append((grid_arg.copy(), ps_arg.copy(), ell_max))
        return dict(ell=[2, 3], D_ell=[800., 900.], chi2_unbinned=3000.+len(calls),
                    chi2_low=20., chi2_binned=300., ell_max=ell_max)
    monkeypatch.setattr(m, "_camb_spectrum_diagnostic", camb)
    monkeypatch.setattr(m, "evaluate_point", lambda *_: pytest.fail("saved audit must not run MS"))
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config))
    assert m.main(["--config", str(config_path), "--audit-spectrum", *map(str, paths),
                   "--spectrum-label", "external A/B command logs"]) == 0
    captured = capsys.readouterr()
    output = json.loads(captured.out)
    assert captured.err.count("mock saved CAMB") == 2
    assert len(calls) == 2 and calls[0][2] == 2508
    np.testing.assert_array_equal(calls[1][1], ps*2)
    assert output["cases"][0]["spectrum_passed_unchanged"]
    assert not output["cases"][0]["declared_label_verified"]
    assert output["cases"][0]["comparison_to_requested"]["As_equal"]
    assert output["comparisons"][0]["P_S"]["max_relative_difference"] == 1.
    assert output["comparisons"][0]["P_S_raw"]["max_relative_difference"] == 1.
    assert output["comparisons"][0]["chi2_difference"] == 1.
    assert output["comparisons"][0]["background_mapping"]["N_total"]["reference"] == 61.5
    assert output["comparisons"][0]["pivot_mapping_comparison"]["derived_N_star_to_match_reference_N_pivot"] == 59.
    third = tmp_path / "different_grid.json"
    different = json.loads(paths[0].read_text())
    different["spectrum"] = dict(k_phys=grid[:10].tolist(), P_S=ps[:10].tolist())
    third.write_text(json.dumps(different))
    all_pairs = m.audit_saved_spectra(config, [third, *paths])
    assert [(x["reference_case"], x["case"]) for x in all_pairs["comparisons"]] == [(0, 1), (0, 2), (1, 2)]
    assert "P_S" not in all_pairs["comparisons"][0]
    assert all_pairs["comparisons"][2]["P_S"]["max_relative_difference"] == 1.
    paths[0].write_text(json.dumps(dict(metadata={}, spectrum=dict(k_phys=[1]*12, P_S=ps.tolist()))))
    with pytest.raises(ValueError, match="sorted positive finite"):
        m.audit_saved_spectra(config, paths[:1])


def test_explore_production_scan_mocked(config, tmp_path, monkeypatch):
    """Explore must use the production evaluator and write one record per point."""
    import concurrent.futures

    def fake_evaluate(theta, cfg):
        if theta[2] > 58:
            return {"status": "pipe_error", "error": "mock"}
        d2 = 900.0 + 10 * (theta[0] - 6.0)
        return {"status": "ok", "chi2": 2575.0 + d2 / 100, "chi2_lcdm": 2572.98,
                "dchi2": 2.0, "d2": d2, "d2_lcdm": 1028.7,
                "k_dip": 1.5e-4, "suppression_pct": 50.0, "N_total": 60.0}

    class SerialExecutor:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def map(self, func, iterable, chunksize=None):
            return map(func, iterable)

    monkeypatch.setattr(m, "evaluate_point", fake_evaluate)
    monkeypatch.setattr(m, "runtime_metadata", lambda *a, **k: {"source_hashes": {}})
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", SerialExecutor)
    out = tmp_path / "scan.jsonl"
    args = types.SimpleNamespace(
        phi0_range=(6.0, 6.2), y0_range=(-0.5, -0.4), nstar_range=(57.0, 59.0),
        n_random=0, n_phi0=2, n_y0=2, n_nstar=3, seed=1, workers=1, out=out)
    result = m.explore_configs(config, args)
    assert result["n_points"] == 12 and result["n_ok"] == 8
    assert result["best_chi2"]["chi2"] == 2584.0
    assert result["best_chi2_with_d2_below_lcdm"]["d2"] == 900.0
    lines = [json.loads(x) for x in out.read_text().splitlines()]
    assert lines[0]["_type"] == "header" and lines[0]["config_hash"] == m.config_hash(config)
    assert len([x for x in lines if x["_type"] == "data"]) == 12
    assert len({tuple(x["theta"]) for x in lines if x["_type"] == "data"}) == 12


def test_postrun_requires_explicit_slicing():
    with pytest.raises(SystemExit):
        m.main(["--summary", "unused.npz"])


def test_mocked_smoke_reproducible_and_resume(config, tmp_path, monkeypatch):
    pytest.importorskip("emcee")
    pytest.importorskip("h5py")
    cfg = m.effective_config(config, True)
    cfg["workers"] = 1
    monkeypatch.setattr(m, "evaluate_point", gaussian_evaluate)
    one, two = tmp_path / "one", tmp_path / "two"
    one.mkdir()
    two.mkdir()
    monkeypatch.setattr(m, "output_paths", lambda _: mock_paths(one))
    paths1 = m.run_sampler(cfg)
    with pytest.raises(FileExistsError):
        m.run_sampler(cfg)
    monkeypatch.setattr(m, "output_paths", lambda _: mock_paths(two))
    # Simulate interruption after checkpoint at step 10, then resume to target 20.
    append = m.append_event
    def interrupt(path, event):
        append(path, event)
        if event.get("event") == "progress" and event.get("steps") == 10:
            raise RuntimeError("simulated interruption")
    monkeypatch.setattr(m, "append_event", interrupt)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        m.run_sampler(cfg)
    monkeypatch.setattr(m, "append_event", append)
    paths2 = m.run_sampler(cfg, resume=True)
    assert m.compare_archives(paths1["archive"], paths2["archive"])["equal"]
    archive = m.load_archive(paths2["archive"])
    assert archive["chain"].shape == (20, 8, 3)
    assert archive["accepted"].shape == (8,)
    assert archive["metadata"]["thread_settings"]["OMP_NUM_THREADS"] == "1"
    assert m.main(["--compare", str(paths1["archive"]), str(paths2["archive"])]) == 0
    assert archive["metadata"]["completed_posterior"]
    assert "scripts/mcmc_higgs.py" in archive["metadata"]["source_hashes"]
    assert archive["metadata"]["resume_invocations"]
    cfg["seed"] += 1
    with pytest.raises(ValueError, match="hash mismatch"):
        m.run_sampler(cfg, resume=True)
    cfg["seed"] -= 1
    monkeypatch.setattr(m, "source_manifest", lambda: {"changed": "source"})
    with pytest.raises(ValueError, match="source identity mismatch"):
        m.run_sampler(cfg, resume=True)


def test_failed_nonfinite_archive_preserved(config, tmp_path, monkeypatch):
    pytest.importorskip("emcee")
    pytest.importorskip("h5py")
    cfg = m.effective_config(config, True)
    cfg.update(workers=1, nsteps=2)
    paths = mock_paths(tmp_path)
    monkeypatch.setattr(m, "output_paths", lambda _: paths)
    monkeypatch.setattr(m, "evaluate_point", lambda *_: {"status": "pipe_error"})
    with pytest.warns(RuntimeWarning):
        with pytest.raises(RuntimeError, match="archive preserved"):
            m.run_sampler(cfg)
    archive = m.load_archive(paths["archive"])
    assert not archive["metadata"]["completed_posterior"]
    assert archive["metadata"]["diagnostics"]["nonfinite_log_prob_count"] == 16


def test_explore_replay_and_num_k_override(config, tmp_path, monkeypatch):
    import concurrent.futures

    seen = {}

    def fake_evaluate(theta, cfg):
        seen["num_k"] = cfg["num_k"]
        return {"status": "ok", "chi2": 2580.0, "chi2_lcdm": 2572.98,
                "dchi2": 7.0, "d2": 900.0, "d2_lcdm": 1028.7, "k_dip": 2e-4,
                "suppression_pct": 40.0, "N_total": 61.0}

    class SerialExecutor:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def map(self, func, iterable, chunksize=None):
            return map(func, iterable)

    monkeypatch.setattr(m, "evaluate_point", fake_evaluate)
    monkeypatch.setattr(m, "runtime_metadata", lambda *a, **k: {"source_hashes": {}})
    monkeypatch.setattr(concurrent.futures, "ProcessPoolExecutor", SerialExecutor)
    source = tmp_path / "prev.jsonl"
    source.write_text("\n".join([
        json.dumps({"_type": "header"}),
        json.dumps({"_type": "data", "theta": [5.7, -0.12, 60.5], "ok": True}),
        json.dumps({"_type": "data", "theta": [5.7, -0.12, 60.5], "ok": True}),
        json.dumps({"_type": "data", "theta": [5.8, -0.13, 55.0], "ok": True}),
    ]) + "\n")
    out = tmp_path / "replay.jsonl"
    args = types.SimpleNamespace(
        phi0_range=(5.0, 7.0), y0_range=(-1.0, 0.0), nstar_range=(50.0, 61.0),
        n_random=0, n_phi0=2, n_y0=2, n_nstar=2, seed=1, workers=1, out=out,
        points_from=source, explore_num_k=160)
    result = m.explore_configs(config, args)
    assert result["n_points"] == 2
    assert seen["num_k"] == 160
    assert config["num_k"] == 80


def test_pareto_from_explore_mocked(tmp_path):
    rows = [
        dict(theta=[6.0, -0.27, 56.6], chi2=2572.5, d2=735.0, k_dip=1.9e-4, suppression_pct=65.0),
        dict(theta=[5.96, -0.31, 51.8], chi2=2583.8, d2=641.0, k_dip=3.1e-4, suppression_pct=60.0),
        dict(theta=[5.89, -0.26, 52.7], chi2=2610.0, d2=540.0, k_dip=3.7e-4, suppression_pct=55.0),
    ]
    path = tmp_path / "explore.jsonl"
    payload = [json.dumps({"_type": "header", "metadata": {"effective_config": {"num_k": 160}}})]
    payload += [json.dumps({"_type": "data", "ok": True, **r}) for r in rows]
    payload.append(json.dumps({"_type": "data", "ok": False, "theta": [5.5, -0.1, 55.0]}))
    path.write_text("\n".join(payload) + "\n")
    result = m.pareto_from_explore(path, chi2_caps=(0.0, 20.0, 40.0))
    assert result["n_records"] == 3 and result["chi2_min"] == 2572.5
    assert result["ladder"][0]["d2"] == 735.0
    assert result["ladder"][1]["d2"] == 641.0
    assert result["ladder"][2]["d2"] == 540.0
    assert result["input_sha256"] == m.hashlib.sha256(path.read_bytes()).hexdigest()


def test_pareto_verification_reports_shift(tmp_path, monkeypatch, config):
    path = tmp_path / "explore.jsonl"
    path.write_text("\n".join([
        json.dumps({"_type": "header", "metadata": {}}),
        json.dumps({"_type": "data", "ok": True, "theta": [6.0, -0.27, 56.6],
                    "chi2": 2572.5, "d2": 735.0, "k_dip": 1.9e-4, "suppression_pct": 65.0}),
    ]) + "\n")
    monkeypatch.setattr(m, "evaluate_point", lambda theta, cfg: {
        "status": "ok", "chi2": 2989.0, "d2": 639.0, "k_dip": 3.2e-4,
        "suppression_pct": 70.0})
    result = m.pareto_from_explore(path, chi2_caps=(0.0,), verify_num_k=640, config=config)
    entry = result["verification"]["entries"][0]
    assert entry["verified_chi2"] == 2989.0
    assert entry["chi2_shift"] == 2989.0 - 2572.5
    assert result["verification"]["num_k"] == 640


def test_weighted_k_grid_option(config, monkeypatch):
    import sys as _sys, types as _types
    build = lambda kmin, kmax, pivot, dense_min, dense_max, n_dense, n_outer: (
        __import__("numpy").geomspace(kmin, kmax, n_dense + n_outer))
    monkeypatch.setitem(_sys.modules, "pspectrum_pipeline",
                        _types.SimpleNamespace(build_weighted_kgrid=build))
    weighted = dict(config, k_grid={"mode": "weighted", "n_dense": 20, "n_outer": 5})
    m.validate_config(weighted)
    assert len(m.k_grid_from_config(weighted)) == 25
    assert len(m.k_grid_from_config(config)) == config["num_k"]
    bad = dict(config, k_grid={"mode": "nope"})
    with pytest.raises(ValueError):
        m.validate_config(bad)
