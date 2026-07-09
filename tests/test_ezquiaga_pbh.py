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
