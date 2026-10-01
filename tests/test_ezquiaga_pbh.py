"""Regression tests for Ezquiaga CHI PBH pipeline.

Locks known-good behavior of solver + pipeline for the subsolar PBH config.
If these fail after a solver change, the change broke known physics.
"""
import pytest
import numpy as np
import inf_dyn_background as bg_solver
from models.ezquiaga_chi import EzquiagaCHIModel, inflection_parameters
from pspectrum_pipeline import find_end_of_inflation, run_pspectrum_pipeline


SUBSOLAR = dict(beta=2e-5, xc=0.784, c=0.77, chi0=8.0, y0=-1e-4, N_star=66)


def _build_subsolar_model():
    m = EzquiagaCHIModel(c=SUBSOLAR["c"])
    a, b = inflection_parameters(SUBSOLAR["xc"], SUBSOLAR["c"], beta=SUBSOLAR["beta"])
    m.a = a
    m.b = b
    m.v0 = m._V0 * m.a / (m.b * m.c) ** 2
    m.x0 = SUBSOLAR["chi0"]
    m.y0 = SUBSOLAR["y0"]
    return m


@pytest.mark.fast
def test_subsolar_config_roundtrip():
    """Building the subsolar model produces finite, positive potentials."""
    m = _build_subsolar_model()

    f_x0 = m.f(m.x0)
    dfdx_x0 = m.dfdx(m.x0)
    assert np.isfinite(f_x0)
    assert np.isfinite(dfdx_x0)
    assert f_x0 > 0


@pytest.mark.fast
def test_subsolar_N_total():
    """Background N_total regression lock for subsolar config."""
    m = _build_subsolar_model()
    T_span = np.linspace(0.0, m.T_max, m.bg_steps)
    bg_sol = bg_solver.run_background_simulation(m, T_span)
    derived = bg_solver.get_derived_quantities(bg_sol, m)
    end_idx = find_end_of_inflation(derived["epsH"])
    N_total = float(derived["N"][end_idx])

    assert N_total == pytest.approx(86.7, rel=0.01), (
        f"N_total={N_total:.2f} — solver change may have shifted e-fold count"
    )


@pytest.mark.fast
def test_subsolar_pipeline_output():
    """Full pipeline with subsolar config produces finite P_S with USR amplification."""
    m = _build_subsolar_model()
    res = run_pspectrum_pipeline(
        model=m,
        phi0=None, y0=None,
        N_star=SUBSOLAR["N_star"],
        k_pivot_phys=0.05,
        num_k=25, k_min=1e-6, k_max=1e16,
        ms_steps=2000, bg_steps=5000,
        save_outputs=False, use_numba=True,
    )

    assert res["status"] == "success", f"Pipeline failed: {res.get('message', '')}"
    ps = np.asarray(res["P_S"])
    k = np.asarray(res["k_phys"])

    assert np.all(np.isfinite(ps)), "P_S contains NaN/Inf values"
    assert np.all(ps > 0), "P_S contains non-positive values"

    # Must have USR amplification: peak P_S >> CMB-scale P_S
    ps_amplified = ps[k > 1e8]
    if len(ps_amplified) > 0:
        ps_cmb = ps[k < 1]
        cmb_level = float(np.nanmedian(ps_cmb)) if len(ps_cmb) > 0 else 2.1e-9
        peak_ratio = float(np.nanmax(ps_amplified)) / max(cmb_level, 1e-30)
        assert peak_ratio > 100, (
            f"No USR peak: P_S amplification factor = {peak_ratio:.1e}"
        )


# ---------------------------------------------------------------------------
# Trajectory tests: the published ROUNDED primitives must trap the field.
#
# Locks the physics claim of ezquiaga/main.tex Sec. III: substituting the
# literature's 2-3 significant-figure couplings produces a local minimum at
# x~0.808 from which the inflaton cannot escape, so eps_H never reaches 1
# and inflation never ends.
# ---------------------------------------------------------------------------
PAPER_COUPLINGS = dict(lambda_0=2.23e-7, b_lambda=1.2e-6, xi_0=7.55,
                       b_xi=11.5, c=0.77)
X_BASIN_MIN = 0.808486    # local minimum of V(x) for the printed couplings
X_BARRIER_MAX = 0.759567  # adjacent local maximum (barrier top)


def _build_paper_model():
    """Model built from the LITERAL rounded couplings (no inflection match)."""
    m = EzquiagaCHIModel(**PAPER_COUPLINGS)
    m.x0, m.y0 = 8.0, -1e-4
    m.T_max, m.bg_steps = 1000.0, 5000
    return m


def _solve(model):
    T_span = np.linspace(0.0, model.T_max, model.bg_steps)
    sol = bg_solver.run_background_simulation(model, T_span)
    q = bg_solver.get_derived_quantities(sol, model)
    return sol, q


@pytest.mark.fast
def test_paper_primitives_stall():
    """The rounded couplings trap the field: eps_H never reaches 1."""
    m = _build_paper_model()
    sol, q = _solve(m)
    epsH = q["epsH"]

    assert np.all(np.isfinite(epsH)), "eps_H contains NaN/Inf"
    assert float(np.nanmax(epsH)) < 1.0, (
        f"printed couplings should never reach eps_H=1, "
        f"got max={np.nanmax(epsH):.3e}"
    )

    x_final = float(m._x_of_chi(sol[0][-1]))
    assert x_final == pytest.approx(X_BASIN_MIN, abs=1e-3), (
        f"field should settle at the basin bottom x~0.8085, got {x_final:.5f}"
    )


@pytest.mark.fast
def test_paper_primitives_have_local_minimum():
    """V(x) for the printed couplings has a min/max pair bracketing x_c."""
    m = _build_paper_model()

    vp_c = float(m._dVdx(0.784))
    vpp_c = float(m._d2Vdx2(0.784))
    assert vp_c < 0, f"V'(x_c) should be negative, got {vp_c:.3e}"
    assert vpp_c > 0, f"V''(x_c) should be positive (minimum), got {vpp_c:.3e}"

    # The two stationary points must straddle x_c.
    assert X_BARRIER_MAX < 0.784 < X_BASIN_MIN
    assert float(m._dVdx(X_BASIN_MIN)) == pytest.approx(0.0, abs=1e-6)
    assert float(m._dVdx(X_BARRIER_MAX)) == pytest.approx(0.0, abs=1e-6)


@pytest.mark.fast
def test_resolve_kgrid_honors_declared_bounds():
    """A declared grid wins; the PBH-weighted grid is only a fallback."""
    from scripts.full_pbh_pipeline import resolve_kgrid

    g = resolve_kgrid(1e-10, 1e28, 501)
    assert len(g) == 501
    assert g.min() == pytest.approx(1e-10, rel=1e-12)
    assert g.max() == pytest.approx(1e28, rel=1e-12)

    # Partial declaration fills the missing piece from the canonical default.
    g2 = resolve_kgrid(k_max=1e28)
    assert g2.max() == pytest.approx(1e28, rel=1e-12)
    assert g2.min() == pytest.approx(1e-10, rel=1e-12)

    # No declaration at all -> PBH-weighted fallback (CMB to PBH scales).
    g3 = resolve_kgrid()
    assert g3.min() < 1e-2 and g3.max() >= 1e22
    assert len(g3) != 501
