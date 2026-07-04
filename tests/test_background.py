"""Tests for background ODE integration and pipeline reproducibility."""
import pytest
import numpy as np
import inf_dyn_background as bg_solver
from scipy.interpolate import interp1d


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


@pytest.mark.fast
def test_background_deterministic():
    """Background solver must be deterministic: same inputs → identical outputs."""
    from models.higgs import HiggsModel
    model = HiggsModel()
    model.x0 = 5.70
    model.y0 = -0.10
    T_span = np.linspace(0.0, 500.0, 200)

    bg_1 = bg_solver.run_background_simulation(model, T_span)
    bg_2 = bg_solver.run_background_simulation(model, T_span)

    assert np.allclose(bg_1, bg_2, rtol=1e-14)


@pytest.mark.fast
def test_pipeline_reproducibility():
    """Full pipeline with same config must produce identical P_S(k)."""
    from pspectrum_pipeline import run_pspectrum_pipeline
    from models.higgs import HiggsModel

    model = HiggsModel()
    kwargs = dict(
        model=model, phi0=5.70, y0=-0.10, N_star=52,
        num_k=10, k_min=1e-4, k_max=0.1,
        ms_steps=500, bg_steps=500,
        save_outputs=False, use_numba=True,
    )
    res_1 = run_pspectrum_pipeline(**kwargs)
    res_2 = run_pspectrum_pipeline(**kwargs)

    assert np.allclose(res_1["k_phys"], res_2["k_phys"], rtol=1e-14)
    assert np.allclose(res_1["P_S"], res_2["P_S"], rtol=1e-12)


@pytest.mark.fast
def test_ms_solver_convergence():
    """P_S(k) must converge as k_start_factor increases."""
    from pspectrum_pipeline import run_pspectrum_pipeline
    from models.higgs import HiggsModel

    base_kw = dict(
        model=HiggsModel(), phi0=5.70, y0=-0.10, N_star=52,
        num_k=10, k_min=1e-4, k_max=0.1,
        ms_steps=500, bg_steps=500,
        save_outputs=False, use_numba=True,
    )

    res_10 = run_pspectrum_pipeline(k_start_factor=10, **base_kw)
    res_100 = run_pspectrum_pipeline(k_start_factor=100, **base_kw)
    res_1000 = run_pspectrum_pipeline(k_start_factor=1000, **base_kw)

    # Interpolate onto shared k grid
    k_shared = res_10["k_phys"]
    ps_100 = interp1d(res_100["k_phys"], res_100["P_S"],
                      kind='cubic', bounds_error=False, fill_value='extrapolate')(k_shared)
    ps_1000 = interp1d(res_1000["k_phys"], res_1000["P_S"],
                       kind='cubic', bounds_error=False, fill_value='extrapolate')(k_shared)

    # Convergence: P_S from k_start_factor=100 and 1000 must agree within 1%
    diff_100_1000 = float(np.nanmean(np.abs(ps_1000 - ps_100) / np.maximum(ps_100, 1e-30)))

    assert diff_100_1000 < 0.01, (
        f"MS not converged: 100→1000 mean P_S diff = {diff_100_1000:.4e}"
    )
