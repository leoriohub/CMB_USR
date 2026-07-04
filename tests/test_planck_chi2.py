"""Tests for Planck data conversions and chi² computation."""
import pytest
import numpy as np
from scripts.planck_data import C_ell_to_d_ell, d_ell_to_C_ell
from scripts.chi2_analysis import chi2_model_lcdm


@pytest.mark.fast
def test_ceell_dell_roundtrip():
    """C_ell → D_ell → C_ell must recover the original within float64 precision."""
    ells = np.arange(2, 30)
    rng = np.random.default_rng(42)
    C_ell_orig = rng.uniform(1e-10, 1e-8, size=len(ells))

    D_ell = C_ell_to_d_ell(ells, C_ell_orig)
    C_ell_round = d_ell_to_C_ell(ells, D_ell)

    assert np.allclose(C_ell_orig, C_ell_round, rtol=1e-15)


@pytest.mark.fast
def test_chi2_exact_match_zero():
    """If model == data exactly, χ² must be zero for all ell."""
    ells_model = np.arange(2, 30)
    rng = np.random.default_rng(42)
    D_model = rng.uniform(500, 6000, size=len(ells_model))

    chi2_m, chi2_l = chi2_model_lcdm(
        D_model, ells_model,
        planck_ells=ells_model, D_planck=D_model,
        D_err_lower=np.ones(len(ells_model)),
        D_err_upper=np.ones(len(ells_model)),
        ell_max=29,
    )

    assert chi2_m == pytest.approx(0.0, abs=1e-10)
    assert chi2_l == pytest.approx(0.0, abs=1e-10)


@pytest.mark.fast
def test_chi2_asymmetric_error_selection():
    """Residual sign must determine which error bar is used for each ell."""
    ells = np.arange(2, 6, dtype=float)
    D_data = np.full_like(ells, 1000.0)
    D_err_lo = np.full_like(ells, 100.0)
    D_err_hi = np.full_like(ells, 200.0)

    # Model above data → residual > 0 → use D_err_upper
    D_model_above = D_data + 50.0
    chi2_above, _ = chi2_model_lcdm(
        D_model_above, ells,
        planck_ells=ells, D_planck=D_data,
        D_err_lower=D_err_lo, D_err_upper=D_err_hi,
        ell_max=5,
    )
    expected_above = ((50.0 / 200.0) ** 2) * len(ells)
    assert chi2_above == pytest.approx(expected_above, rel=1e-12)

    # Model below data → residual < 0 → use D_err_lower
    D_model_below = D_data - 50.0
    chi2_below, _ = chi2_model_lcdm(
        D_model_below, ells,
        planck_ells=ells, D_planck=D_data,
        D_err_lower=D_err_lo, D_err_upper=D_err_hi,
        ell_max=5,
    )
    expected_below = ((50.0 / 100.0) ** 2) * len(ells)
    assert chi2_below == pytest.approx(expected_below, rel=1e-12)

    # Prove asymmetric selection is active: should NOT be equal
    assert chi2_above != pytest.approx(chi2_below, rel=1e-12)
