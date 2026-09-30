"""Tests for the NN emulator pipeline."""
import numpy as np
import pytest
from scripts.emulator.grid import FIXED_K_GRID
from scripts.emulator.postprocess import (
    extract_ns, find_peak, classify_usr, classify_mass,
    mass_from_k, compute_f_total, extract_usr_info,
)


def test_fixed_k_grid():
    assert len(FIXED_K_GRID) == 200
    assert FIXED_K_GRID[0] >= 1.0
    assert FIXED_K_GRID[-1] <= 1e18
    assert np.all(np.diff(FIXED_K_GRID) > 0)


def test_extract_ns_flat():
    k = np.logspace(-2, 2, 100)
    ps = 2.1e-9 * np.ones_like(k)
    n_s, A_s = extract_ns(k, ps)
    assert n_s is not None
    assert abs(n_s - 1.0) < 0.01
    assert abs(A_s - 2.1e-9) < 1e-11


def test_extract_ns_tilted():
    k = np.logspace(-2, 2, 100)
    n_s_true = 0.965
    ps = 2.1e-9 * k ** (n_s_true - 1)
    n_s, A_s = extract_ns(k, ps)
    assert n_s is not None
    assert abs(n_s - n_s_true) < 0.01


def test_find_peak():
    k = np.logspace(0, 6, 200)
    ps = 1e-9 * np.ones_like(k)
    ps[100:150] = 1e-5
    peak = find_peak(k, ps)
    assert peak is not None
    assert peak["P_S_peak"] == 1e-5


def test_find_peak_no_peak():
    k = np.logspace(0, 6, 200)
    ps = 1e-9 * np.ones_like(k)
    peak = find_peak(k, ps)
    assert peak is not None
    assert peak["P_S_peak"] == 1e-9


def test_classify_usr():
    assert classify_usr(200) == "strong"
    assert classify_usr(150) == "transitional"
    assert classify_usr(100) == "weak"


def test_classify_mass():
    assert classify_mass(1e-16) == "asteroid_gap"
    assert classify_mass(1e-10) == "intermediate"
    assert classify_mass(1e-4) == "sub_solar_gap"
    assert classify_mass(10) == "stellar_ligo_ruled_out"


def test_mass_from_k():
    """Must use the canonical scripts.constants conversion, not local values.

    Regression guard: K_EQ/M_EQ were once hardcoded to 1.0 here, making this
    helper disagree with full_pbh_pipeline.mass_from_k by 3.2e13 and
    misclassifying stellar-mass PBHs as "intermediate".
    """
    from scripts.constants import ACCRETION, gamma_default, k_eq_default, M_eq_default
    from scripts.full_pbh_pipeline import mass_from_k as canonical_mass_from_k

    for k in (1e6, 1e10, 5.75e9):
        expected = gamma_default * M_eq_default * (k_eq_default / k) ** 2 * ACCRETION
        assert mass_from_k(k) == pytest.approx(expected, rel=1e-12)
        # and must agree with the other shipped implementation
        assert mass_from_k(k) == pytest.approx(canonical_mass_from_k(k) * ACCRETION, rel=1e-12)


def test_compute_f_total():
    k = np.logspace(6, 12, 100)
    ps = 1e-9 * np.ones_like(k)
    ps[30:70] = 1e-4
    ft = compute_f_total(k, ps)
    assert np.isfinite(ft)


def test_extract_usr_info_basic():
    k = np.logspace(-2, 10, 300)
    ps = 2.1e-9 * k ** (0.965 - 1)
    info = extract_usr_info(k, ps, N_total=170)
    assert info["n_s"] is not None
    assert info["usr_type"] == "strong"
    assert info["mass_bin"] is not None
