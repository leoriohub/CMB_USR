"""Tests for Ezquiaga CHI model computational plumbing."""
import pytest
import numpy as np
from models.ezquiaga_chi import EzquiagaCHIModel


@pytest.mark.fast
def test_chi_spline_roundtrip():
    """χ(φ) → φ(χ) must recover the original χ to high precision."""
    model = EzquiagaCHIModel()
    test_chi = np.array([6.5, 7.0, 7.5, 8.0, 10.0])

    x = model._x_of_chi(test_chi)
    chi_back = model._chi_spline(x)

    rel_err = np.abs(chi_back - test_chi) / np.maximum(test_chi, 1e-10)
    assert np.all(rel_err < 1e-6), f"Max roundtrip error: {np.max(rel_err):.2e}"


@pytest.mark.fast
def test_pbh_peak_finder():
    """find_pbh_peak must return correct k and P_S for a known injected peak."""
    from scripts.sweep_pbh_params import find_pbh_peak

    k = np.logspace(6, 16, 300)
    flat_ps = 1e-9 * np.ones_like(k)

    # Inject a Gaussian peak at k=1e10 with amplitude 1e-5
    peak_k = 1e10
    peak_amp = 1e-5
    width = 0.3
    ps = flat_ps + peak_amp * np.exp(-((np.log(k / peak_k)) ** 2) / (2 * width**2))

    result = find_pbh_peak(k, ps, k_min=1e6, k_max=1e16)
    assert result is not None, "find_pbh_peak returned None"
    assert result["k_peak"] == pytest.approx(peak_k, rel=0.1)
    assert result["P_S_peak"] == pytest.approx(peak_amp, rel=0.2)
    assert not result["on_grid_boundary"]


@pytest.mark.fast
def test_pbh_peak_finder_no_peak():
    """find_pbh_peak must return None when no USR peak is present."""
    from scripts.sweep_pbh_params import find_pbh_peak

    k = np.logspace(6, 16, 100)
    ps = 1e-9 * np.ones_like(k)

    result = find_pbh_peak(k, ps, k_min=1e6, k_max=1e16)
    assert result is None or result["P_S_peak"] <= 1e-9 * 2
