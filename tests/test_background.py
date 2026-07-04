"""Tests for Higgs model potentials and background ODE invariants."""
import pytest
import numpy as np
import inf_dyn_background as bg_solver


@pytest.mark.fast
def test_higgs_potential_asymptote():
    """f(x) → 1.0 and dfdx(x) → 0.0 as x → ∞."""
    from models.higgs import HiggsModel
    model = HiggsModel()
    x_large = 100.0
    f_val = model.f(x_large)
    dfdx_val = model.dfdx(x_large)
    assert f_val == pytest.approx(1.0, rel=1e-6)
    assert dfdx_val == pytest.approx(0.0, abs=1e-6)


@pytest.mark.fast
def test_higgs_potential_zero():
    """f(0) == 0 and dfdx(0) == 0."""
    from models.higgs import HiggsModel
    model = HiggsModel()
    f_val = model.f(0.0)
    dfdx_val = model.dfdx(0.0)
    assert f_val == pytest.approx(0.0, abs=1e-12)
    assert dfdx_val == pytest.approx(0.0, abs=1e-12)


@pytest.mark.fast
def test_initial_conditions_Friedmann():
    """zi from get_initial_conditions must satisfy the Friedmann constraint."""
    from models.higgs import HiggsModel
    from models.ezquiaga_chi import EzquiagaCHIModel

    configs = [
        (HiggsModel, 5.70, -0.10),
        (HiggsModel, 6.40, -0.475),
        (EzquiagaCHIModel, 8.0, -1e-4),
    ]
    for model_cls, x0, y0 in configs:
        model = model_cls()
        model.x0 = x0
        model.y0 = y0
        x0_v, y0_v, zi, Ni = model.get_initial_conditions()
        expected_zi = np.sqrt(
            y0**2 / 6 + model.v0 * model.f(x0) / (3 * model.S**2)
        )
        assert zi == pytest.approx(expected_zi, rel=1e-12), f"Failed for {model.name}"


@pytest.mark.fast
def test_einstein_frame_energy_conservation():
    """Friedmann constraint z² = y²/6 + V/3 must hold at every ODE step."""
    from models.higgs import HiggsModel
    model = HiggsModel()
    model.x0 = 5.70
    model.y0 = -0.10
    T_span = np.linspace(0.0, 500.0, 200)
    bg_sol = bg_solver.run_background_simulation(model, T_span)

    x, y, z, n = bg_sol
    V = model.v0 * model.f(x) / model.S**2

    lhs = z**2
    rhs = y**2 / 6 + V / 3

    max_rel_error = float(np.max(np.abs(lhs - rhs) / np.maximum(rhs, 1e-30)))
    assert max_rel_error < 1e-6, f"Friedmann constraint violated: {max_rel_error:.2e}"
