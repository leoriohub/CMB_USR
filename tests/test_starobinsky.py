"""Tests for Starobinsky R² inflation model."""
import pytest
import numpy as np
import inf_dyn_background as bg_solver


@pytest.fixture
def model():
    from models.starobinsky import StarobinskyModel
    m = StarobinskyModel()
    m.x0 = 5.5
    m.y0 = -1e-4
    return m


# ── Fast tests: potential shape (no ODE solver) ──────────────────────────

class TestPotentialShape:

    @pytest.mark.fast
    def test_potential_at_zero(self, model):
        assert model.f(0.0) == 0.0

    @pytest.mark.fast
    def test_potential_at_infinity(self, model):
        assert model.f(10.0) == pytest.approx(1.0, rel=1e-3)
        assert model.f(20.0) == pytest.approx(1.0, rel=1e-6)

    @pytest.mark.fast
    def test_potential_derivative_at_zero(self, model):
        assert model.dfdx(0.0) == 0.0

    @pytest.mark.fast
    def test_potential_derivative_positive(self, model):
        for x in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            assert model.dfdx(x) > 0, f"dfdx({x}) <= 0"

    @pytest.mark.fast
    def test_potential_second_derivative_at_zero(self, model):
        assert model.d2fdx2(0.0) == pytest.approx(4 / 3, rel=1e-12)

    @pytest.mark.fast
    def test_potential_second_derivative_at_infinity(self, model):
        assert model.d2fdx2(10.0) == pytest.approx(0.0, abs=1e-3)
        assert model.d2fdx2(30.0) == pytest.approx(0.0, abs=1e-8)

    @pytest.mark.fast
    def test_f_positive(self, model):
        for x in [0.1, 0.5, 1.0, 2.0, 5.0, 10.0]:
            assert model.f(x) > 0, f"f({x}) <= 0"

    @pytest.mark.fast
    @pytest.mark.parametrize("x,eps_expected", [
        (4.5, 9.03e-4),
        (5.0, 3.92e-4),
        (5.5, 1.71e-4),
    ])
    def test_epsilon_V(self, model, x, eps_expected):
        eps = 0.5 * (model.dfdx(x) / model.f(x))**2
        assert eps == pytest.approx(eps_expected, rel=0.15)

    @pytest.mark.fast
    @pytest.mark.parametrize("x,eta_expected", [
        (4.5, -0.0347),
        (5.0, -0.0229),
        (5.5, -0.0151),
    ])
    def test_eta_V(self, model, x, eta_expected):
        eta = model.d2fdx2(x) / model.f(x) - 0.5 * (model.dfdx(x) / model.f(x))**2
        assert eta == pytest.approx(eta_expected, rel=0.15)

    @pytest.mark.fast
    def test_sr_conditions(self, model):
        for x in np.linspace(4, 10, 20):
            eps = 0.5 * (model.dfdx(x) / model.f(x))**2
            eta = model.d2fdx2(x) / model.f(x) - 0.5 * (model.dfdx(x) / model.f(x))**2
            assert abs(eps) < 0.01, f"|ε_V|={eps:.4e} >= 0.01 at x={x:.2f}"
            assert abs(eta) < 0.1, f"|η_V|={eta:.4f} >= 0.1 at x={x:.2f}"


# ── Fast tests: solver integrity ─────────────────────────────────────────

class TestSolverIntegrity:

    @pytest.mark.fast
    def test_initial_conditions_Friedmann(self, model):
        x0_v, y0_v, zi, Ni = model.get_initial_conditions()
        expected_zi = np.sqrt(
            model.y0**2 / 6 + model.v0 * model.f(model.x0) / (3 * model.S**2)
        )
        assert zi == pytest.approx(expected_zi, rel=1e-12)

    @pytest.mark.fast
    def test_einstein_frame_energy_conservation(self, model):
        T_span = np.linspace(0.0, 500.0, 200)
        x, y, z, n = bg_solver.run_background_simulation(model, T_span)
        V = model.v0 * model.f(x) / model.S**2
        lhs, rhs = z**2, y**2 / 6 + V / 3
        max_rel = float(np.max(np.abs(lhs - rhs) / np.maximum(rhs, 1e-30)))
        assert max_rel < 1e-6, f"Friedmann constraint violated: {max_rel:.2e}"

    @pytest.mark.fast
    def test_background_deterministic(self, model):
        T_span = np.linspace(0.0, 500.0, 200)
        bg_1 = bg_solver.run_background_simulation(model, T_span)
        bg_2 = bg_solver.run_background_simulation(model, T_span)
        assert np.allclose(bg_1, bg_2, rtol=1e-14), "Background not deterministic"


# ── Integration tests: background evolution ──────────────────────────────

@pytest.mark.integration
class TestBackgroundIntegration:

    def test_n_total_range(self):
        from models.starobinsky import StarobinskyModel
        model = StarobinskyModel()
        model.x0 = 5.5
        model.y0 = -1e-4
        T_span = np.linspace(0.0, model.T_max, model.bg_steps)
        _, _, _, n = bg_solver.run_background_simulation(model, T_span)
        N_total = n[-1] - n[0]
        assert 55 < N_total < 75, f"N_total={N_total:.2f} outside (55, 75)"

    def test_larger_phi0_more_efolds(self):
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
        assert ratio < 4, f"N(6)/N(5)={ratio:.2f} exceeds factor-2 bound"


# ── Slow tests: MS pipeline ──────────────────────────────────────────────

@pytest.mark.slow
class TestMSPipeline:

    def test_n_s(self):
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

    def test_r(self):
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
