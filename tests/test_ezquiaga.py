"""Tests for Ezquiaga CHI model inflection and frame mapping."""
import pytest
import numpy as np
from models.ezquiaga_chi import inflection_parameters, EzquiagaCHIModel


@pytest.mark.fast
def test_inflection_condition():
    """With β=0, V'(x_c)=0 and V''(x_c)=0 to numerical precision."""
    x_c, c = 0.784, 0.77
    a, b = inflection_parameters(x_c, c, beta=0.0)

    model = EzquiagaCHIModel(c=c)
    model.a = a
    model.b = b

    dVdx = model._dVdx(x_c)
    d2Vdx2 = model._d2Vdx2(x_c)

    assert dVdx == pytest.approx(0.0, abs=1e-10), f"dV/dx(x_c) = {dVdx:.2e}"
    assert d2Vdx2 == pytest.approx(0.0, abs=1e-10), f"d2V/dx2(x_c) = {d2Vdx2:.2e}"


@pytest.mark.fast
def test_inflection_parameters_reproducible():
    """inflection_parameters returns known reference values."""
    a, b = inflection_parameters(0.784, 0.77, 1e-5)

    assert a == pytest.approx(5.33530386, rel=1e-8)
    assert b == pytest.approx(1.51932465, rel=1e-8)


@pytest.mark.fast
def test_chi_spline_roundtrip():
    """χ(φ) → φ(χ) must recover the original χ to high precision."""
    model = EzquiagaCHIModel()
    test_chi = np.array([6.5, 7.0, 7.5, 8.0, 10.0])

    x = model._x_of_chi(test_chi)
    chi_back = model._chi_spline(x)

    rel_err = np.abs(chi_back - test_chi) / np.maximum(test_chi, 1e-10)
    assert np.all(rel_err < 1e-6), f"Max roundtrip error: {np.max(rel_err):.2e}"
