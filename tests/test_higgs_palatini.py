"""Physical regression checks for the canonical Palatini Higgs model."""

import numpy as np
import pytest

from models.higgs_palatini import PalatiniHiggsModel


@pytest.mark.fast
def test_potential_derivatives_match_finite_differences():
    model = PalatiniHiggsModel(lam=0.13, xi=3.7)
    chi = np.array([0.17, 0.42, 0.83])
    step = 1e-4
    f_plus = model.f(chi + step)
    f = model.f(chi)
    f_minus = model.f(chi - step)

    first_difference = (f_plus - f_minus) / (2 * step)
    second_difference = (f_plus - 2 * f + f_minus) / step**2

    np.testing.assert_allclose(model.dfdx(chi), first_difference, rtol=2e-6)
    np.testing.assert_allclose(
        model.d2fdx2(chi), second_difference, rtol=2e-5, atol=2e-8
    )


@pytest.mark.fast
def test_initial_field_uses_seventy_efold_slow_roll_anchor():
    model = PalatiniHiggsModel(lam=0.13, xi=1.5e9)
    u0 = np.sqrt(model.xi_val) * model.x0
    u_end = 0.5 * np.arcsinh(np.sqrt(32 * model.xi_val))
    n_before_end = (np.sinh(u0)**2 - np.sinh(u_end)**2) / (8 * model.xi_val)

    assert n_before_end == pytest.approx(70.0, rel=1e-12)
