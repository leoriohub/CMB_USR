"""Integrated physical result tests for Starobinsky R² inflation.

Only validates physical outputs — no potential shape or derivative tests.
"""
import pytest
import numpy as np
import inf_dyn_background as bg_solver
from scripts.constants import As


@pytest.fixture
def model():
    from models.starobinsky import StarobinskyModel
    m = StarobinskyModel()
    m.x0 = 5.5
    m.y0 = -1e-4
    return m


# ── Fast tests (no ODE solver) ────────────────────────────────────────────

@pytest.mark.fast
def test_friedmann_constraint(model):
    x0_v, y0_v, zi, Ni = model.get_initial_conditions()
    expected_zi = np.sqrt(
        model.y0**2 / 6 + model.v0 * model.f(model.x0) / (3 * model.S**2)
    )
    assert zi == pytest.approx(expected_zi, rel=1e-12)


@pytest.mark.fast
def test_background_deterministic(model):
    T_span = np.linspace(0.0, 500.0, 200)
    bg_1 = bg_solver.run_background_simulation(model, T_span)
    bg_2 = bg_solver.run_background_simulation(model, T_span)
    assert np.allclose(bg_1, bg_2, rtol=1e-14), "Background not deterministic"


# ── Integration tests: background evolution ───────────────────────────────

@pytest.mark.integration
def test_N_total_for_phi0():
    from models.starobinsky import StarobinskyModel
    model = StarobinskyModel()
    model.x0 = 5.5
    model.y0 = -1e-4
    T_span = np.linspace(0.0, model.T_max, model.bg_steps)
    _, _, _, n = bg_solver.run_background_simulation(model, T_span)
    N_total = n[-1] - n[0]
    assert 55 < N_total < 75, f"N_total={N_total:.2f} outside (55, 75)"


@pytest.mark.integration
def test_n_total_analytic_consistency():
    from models.starobinsky import StarobinskyModel
    model = StarobinskyModel()
    model.y0 = -1e-4
    T_span = np.linspace(0.0, model.T_max, model.bg_steps)

    model.x0 = 5.0
    _, _, _, n = bg_solver.run_background_simulation(model, T_span)
    N5 = n[-1] - n[0]

    model.x0 = 6.0
    _, _, _, n = bg_solver.run_background_simulation(model, T_span)
    N6 = n[-1] - n[0]

    assert N6 > N5, f"N(6)={N6:.2f} <= N(5)={N5:.2f}"
    ratio = N6 / N5
    assert ratio < 4, f"N(6)/N(5)={ratio:.2f} exceeds factor-4 bound"


# ── Slow tests: MS pipeline ──────────────────────────────────────────────

@pytest.mark.slow
class TestMSPipeline:

    def test_n_s_vs_sr(self):
        from pspectrum_pipeline import run_pspectrum_pipeline
        from models.starobinsky import StarobinskyModel
        from scripts.observables import extract_ns

        model = StarobinskyModel()
        result = run_pspectrum_pipeline(
            model=model, phi0=5.5, y0=-1e-4, N_star=55,
            num_k=60, k_min=1e-5, k_max=0.5,
            k_pivot_phys=0.002,
            bg_steps=model.bg_steps, ms_steps=5000,
            save_outputs=False, backend='fortran',
        )
        n_s, meta = extract_ns(
            result["k_phys"], result["P_S"],
            k_pivot=0.002,
        )
        expected_n_s = 1 - 2 / 55
        assert n_s == pytest.approx(expected_n_s, abs=0.005), f"n_s={n_s}"

    def test_r_vs_sr(self):
        from pspectrum_pipeline import run_pspectrum_pipeline
        from models.starobinsky import StarobinskyModel

        model = StarobinskyModel()
        result = run_pspectrum_pipeline(
            model=model, phi0=5.5, y0=-1e-4, N_star=55,
            num_k=60, k_min=1e-5, k_max=0.5,
            k_pivot_phys=0.002,
            bg_steps=model.bg_steps, ms_steps=5000,
            save_outputs=False, backend='fortran',
        )
        k, PS, PT = result["k_phys"], result["P_S"], result["P_T"]
        pivot_idx = np.argmin(np.abs(k - 0.002))
        r = PT[pivot_idx] / PS[pivot_idx]
        expected_r = 12 / 55**2
        assert r == pytest.approx(expected_r, rel=0.2), f"r={r:.6f}"

    def test_ps_normalized(self):
        from pspectrum_pipeline import run_pspectrum_pipeline
        from models.starobinsky import StarobinskyModel

        model = StarobinskyModel()
        result = run_pspectrum_pipeline(
            model=model, phi0=5.5, y0=-1e-4, N_star=55,
            num_k=60, k_min=1e-5, k_max=0.5,
            k_pivot_phys=0.002,
            bg_steps=model.bg_steps, ms_steps=5000,
            save_outputs=False, backend='fortran',
        )
        k, PS = result["k_phys"], result["P_S"]
        pivot_idx = np.argmin(np.abs(k - 0.002))
        assert PS[pivot_idx] == pytest.approx(As, rel=0.05), (
            f"P_S(pivot)={PS[pivot_idx]:.3e} vs As={As:.3e}"
        )
