"""Tests for the τ-profile scan and the binned-polarization χ².

These tests are deliberately CAMB-free: they exercise the pure logic
(χ² conventions, grid construction, config validation) with synthetic
arrays so they run in any environment.
"""
import json
import os

import numpy as np
import pytest

from scripts.chi2_analysis import (
    chi2_binned,
    chi2_commander,
    chi2_ee_binned,
    chi2_te_binned,
    chi2_tt_combined,
    chi2_ee_low,
)
from scripts.planck_data import (
    load_planck_ee_binned,
    load_planck_ee_low,
    load_planck_te_binned,
)
from scripts import tau_profile as tp


# ── χ² conventions ─────────────────────────────────────────────────────────


@pytest.fixture()
def synthetic_binned(monkeypatch):
    """Replace the EE/TE loaders with a 3-bin synthetic dataset."""

    def _make():
        import scripts.chi2_analysis as ca

        ells = np.array([50.0, 150.0, 400.0])
        D = np.array([1.0, -2.0, 0.5])
        lo = np.array([0.1, 0.2, 0.05])
        hi = np.array([0.3, 0.4, 0.15])
        monkeypatch.setattr(
            ca, "load_planck_ee_binned", lambda: (ells, D, lo, hi)
        )
        monkeypatch.setattr(
            ca, "load_planck_te_binned", lambda: (ells, D, lo, hi)
        )
        return ells, D, lo, hi

    return _make


def test_chi2_ee_zero_for_identical_spectrum(synthetic_binned):
    ells, D, _, _ = synthetic_binned()
    chi2_m, chi2_l, n = chi2_ee_binned(D, ells, D, ells)
    assert n == len(ells)
    assert chi2_m == pytest.approx(0.0)
    assert chi2_l == pytest.approx(0.0)


def test_chi2_ee_selects_error_by_residual_sign(synthetic_binned):
    """Model above data → upper error; below → lower error (same as TT)."""
    ells, D, lo, hi = synthetic_binned()
    # Perturb bin 0 upwards: residual > 0 → sigma = hi[0]
    model_hi = D.copy()
    model_hi[0] += 1.0
    chi2_m, _, _ = chi2_ee_binned(model_hi, ells, D, ells)
    assert chi2_m == pytest.approx((1.0 / hi[0]) ** 2)

    # Perturb bin 0 downwards: residual < 0 → sigma = lo[0]
    model_lo = D.copy()
    model_lo[0] -= 1.0
    chi2_m2, _, _ = chi2_ee_binned(model_lo, ells, D, ells)
    assert chi2_m2 == pytest.approx((1.0 / lo[0]) ** 2)


def test_chi2_te_handles_negative_d_ell(synthetic_binned):
    """TE is signed: a negative residual must use the lower error bar."""
    ells, D, lo, hi = synthetic_binned()
    assert D[1] == -2.0  # synthetic TE bin is negative
    model = D.copy()
    model[1] = -1.5  # residual +0.5 (model > data)
    chi2_m, _, _ = chi2_te_binned(model, ells, D, ells)
    assert chi2_m == pytest.approx((0.5 / hi[1]) ** 2)

    model[1] = -2.5  # residual −0.5
    chi2_m2, _, _ = chi2_te_binned(model, ells, D, ells)
    assert chi2_m2 == pytest.approx((0.5 / lo[1]) ** 2)


def test_chi2_tt_combined_is_commander_plus_binned():
    """The combined TT χ² is additive and covers non-overlapping ℓ ranges."""
    ells = np.arange(2, 2509, dtype=float)
    rng = np.random.default_rng(0)
    lcdm = 1000.0 + 500.0 * np.sin(ells / 300.0)
    model = lcdm * (1.0 + 0.02 * rng.standard_normal(ells.size))

    cm, cl, n = chi2_tt_combined(model, ells, lcdm, ells)
    cm_c, cl_c, n_c = chi2_commander(model, ells, lcdm, ells)
    cm_b, cl_b, n_b = chi2_binned(model, ells, lcdm, ells)

    assert n == n_c + n_b
    assert cm == pytest.approx(cm_c + cm_b)
    assert cl == pytest.approx(cl_c + cl_b)


def test_low_ell_ee_is_the_tau_lever_arm():
    """Low-ℓ EE must cover the reionization bump (ℓ=2-29), unlike binned EE."""
    ell, D, lo, hi = load_planck_ee_low()
    assert len(ell) == 28
    assert ell.min() == 2 and ell.max() == 29
    assert np.all(lo > 0) and np.all(hi > 0)
    # Binned EE deliberately starts above the bump.
    ell_b, *_ = load_planck_ee_binned()
    assert ell_b.min() > 40


def test_real_data_loaders_have_expected_shape():
    ell_e, D_e, lo_e, hi_e = load_planck_ee_binned()
    ell_t, D_t, lo_t, hi_t = load_planck_te_binned()

    assert len(ell_e) == 66 and len(ell_t) == 66
    # Binned multipoles are non-integer bin means.
    assert not np.allclose(ell_e, np.round(ell_e)), "EE ℓ should not be integers"
    assert ell_e.min() > 40 and ell_e.max() <= 2510
    assert np.allclose(ell_e, ell_t), "EE and TE must share the bin grid"
    assert np.all(lo_e > 0) and np.all(hi_e > 0)
    assert np.all(lo_t > 0) and np.all(hi_t > 0)
    # TE is signed. EE is physically positive, but a binned estimate can
    # dip below zero at low ℓ (this file does), which is exactly why the
    # χ² picks its error bar from the residual sign rather than assuming σ.
    assert D_t.min() < 0.0
    assert np.mean(D_e) > 0.0


# ── τ grid and BAO prior ───────────────────────────────────────────────────


def test_build_tau_grid_inclusive_endpoints():
    grid = tp.build_tau_grid({"min": 0.04, "max": 0.12, "n": 17})
    assert len(grid) == 17
    assert grid[0] == pytest.approx(0.04)
    assert grid[-1] == pytest.approx(0.12)
    assert np.all(np.diff(grid) > 0)


@pytest.mark.parametrize(
    "grid,msg",
    [
        ({"min": 0.1, "max": 0.1, "n": 5}, "min < max"),
        ({"min": 0.2, "max": 0.1, "n": 5}, "min < max"),
        ({"min": 0.0, "max": 0.1, "n": 1}, "n >= 2"),
    ],
)
def test_build_tau_grid_rejects_bad_input(grid, msg):
    with pytest.raises(ValueError, match=msg):
        tp.build_tau_grid(grid)


def test_bao_penalty_values():
    prior = {"tau": 0.09, "sigma": 0.012}
    assert tp.bao_penalty(0.09, prior) == pytest.approx(0.0)
    assert tp.bao_penalty(0.09 + 0.012, prior) == pytest.approx(1.0)
    assert tp.bao_penalty(0.09 - 0.024, prior) == pytest.approx(4.0)
    assert tp.bao_penalty(0.09, None) == 0.0


# ── Config validation ──────────────────────────────────────────────────────


def _valid_config():
    return {
        "tag": "unit",
        "tau_grid": {"min": 0.04, "max": 0.12, "n": 5},
        "tt_components": ["commander", "binned"],
        "bao_prior": {"tau": 0.09, "sigma": 0.012},
        "configs": [
            {"label": "a", "phi0": 6.4, "y0": -0.475, "n_star": 59.0},
        ],
    }


def test_validate_config_accepts_valid():
    tp.validate_config(_valid_config())


@pytest.mark.parametrize(
    "mutate,match",
    [
        (lambda c: c.pop("tag"), "missing required key 'tag'"),
        (lambda c: c.__setitem__("tag", ""), "non-empty string"),
        (lambda c: c["tau_grid"].pop("n"), "missing 'n'"),
        (lambda c: c["tau_grid"].__setitem__("n", 1), ">= 2"),
        (lambda c: c["tau_grid"].__setitem__("min", 0.2), "must be <"),
        (lambda c: c.__setitem__("configs", []), "non-empty list"),
        (lambda c: c["configs"][0].pop("phi0"), "missing 'phi0'"),
        (
            lambda c: c["configs"].append(dict(c["configs"][0])),
            "duplicate config label",
        ),
        (
            lambda c: c.__setitem__("tt_components", ["nonsense"]),
            "unknown TT component",
        ),
        (
            lambda c: c.__setitem__("tt_components", ["unbinned", "commander"]),
            "double-counts",
        ),
        (
            lambda c: c.__setitem__("bao_prior", {"tau": 0.09, "sigma": 0}),
            "sigma' must be > 0",
        ),
    ],
)
def test_validate_config_rejects(mutate, match):
    cfg = _valid_config()
    mutate(cfg)
    with pytest.raises(ValueError, match=match):
        tp.validate_config(cfg)


def test_bundled_config_is_valid():
    path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "tau_profile", "higgs_usr_good.json",
    )
    cfg = tp.load_config(path)
    assert cfg["tag"] == "higgs_usr_good"
    assert len(cfg["configs"]) == 4
    labels = [e["label"] for e in cfg["configs"]]
    assert labels == ["best_chi2", "best_balance",
                      "best_quadrupole", "deep_suppression"]
    assert cfg["_sha256"] and len(cfg["_sha256"]) == 64


def test_summarise_reports_profiled_minima():
    """τ* is the argmin of the amplitude-profiled χ², on the τ grid."""
    grid = np.array([0.05, 0.06, 0.07])
    rows = []
    for i, tau in enumerate(grid):
        # Model prefers the middle τ; ΛCDM prefers the first.
        rows.append({
            "tau": float(tau),
            "raw": {c: {"model": 10.0 + i, "lcdm": 10.0 + i, "delta": 0.0,
                        "n": 94} for c in tp.COMBOS},
            "profiled": {c: {"model": float((i - 1) ** 2) + 100.0,
                             "lcdm": float(i * 3.0) + 100.0,
                             "delta": 0.0, "n": 94,
                             "a_model": 1.0 + 0.01 * i}
                         for c in tp.COMBOS},
        })
    cfg = {"tag": "t", "bao_prior": {"tau": 0.07, "sigma": 0.012}}
    metadata = {"tau_grid": grid.tolist()}
    summary = tp.summarise({"a": rows}, cfg, metadata)
    for combo in tp.COMBOS:
        entry = summary["entries"]["a"]["combos"][combo]
        assert entry["tau_star_model"] == pytest.approx(0.06, abs=1e-9)
        assert entry["chi2_model_min"] == pytest.approx(100.0)
        assert entry["tau_star_lcdm"] == pytest.approx(0.05, abs=1e-9)
        assert entry["tau_star_minus_bao"] == pytest.approx(0.06 - 0.07)
        # χ² at the BAO τ (0.07) minus the model minimum = 1 - 0 = 1.
        assert entry["bao_cost_model"] == pytest.approx(1.0)


def test_parabolic_min_recovers_quadratic_vertex():
    x = np.array([0.0, 1.0, 2.0, 3.0])
    y = (x - 1.7) ** 2
    tau, chi = tp._parabolic_min(x, y, 2)
    assert tau == pytest.approx(1.7, abs=1e-9)
    assert chi == pytest.approx(0.0, abs=1e-12)


# ── CAMB-dependent smoke test (skipped without camb) ────────────────────────


def test_tau_changes_ee_amplitude():
    """Raising τ must suppress the EE amplitude by ~e^{-2Δτ}."""
    pytest.importorskip("camb")
    from scripts.camb_wrapper import compute_cl_camb_powerlaw

    ells, Ctt_l, Cte_l, Cee_l = compute_cl_camb_powerlaw(ell_max=400, tau=0.05)
    _, Ctt_h, Cte_h, Cee_h = compute_cl_camb_powerlaw(ell_max=400, tau=0.07)
    mask = (ells >= 150) & (ells <= 300)
    ratio = np.mean(Cee_h[mask] / Cee_l[mask])
    assert ratio == pytest.approx(np.exp(-2 * 0.02), rel=0.02)
