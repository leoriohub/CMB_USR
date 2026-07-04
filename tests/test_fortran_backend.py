"""Backend agreement: Fortran vs Numba MS solver.

Levels 1-3 from fortran/test_vs_numba.py, migrated to pytest.
Level 4 (performance benchmark) is excluded — environment-dependent.
"""
import pytest
import numpy as np
from fortran_ms_solver import HAVE_FORTRAN

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not HAVE_FORTRAN,
                       reason="Fortran module not compiled (cd fortran && make)"),
]

from models.higgs import HiggsModel
import inf_dyn_background as bg_solver
from numba_ms_solver import numba_run_ms, build_numba_splines
from pspectrum_pipeline import run_pspectrum_pipeline, find_end_of_inflation, extract_mode_initial_conditions
from scripts.camb_wrapper import compute_cl_full_camb, compute_chi2_camb
from scripts.planck_data import C_ell_to_d_ell

import ms_solver_fort as _fort
from fortran_ms_solver import _pack_spline_coefs


# ── helpers ──────────────────────────────────────────────────────────────


def _setup_higgs(phi0, y0, N_star):
    model = HiggsModel()
    model.x0 = phi0
    model.y0 = y0
    T_span = np.linspace(0.0, model.T_max, model.bg_steps)
    bg_sol = bg_solver.run_background_simulation(model, T_span)
    derived = bg_solver.get_derived_quantities(bg_sol, model)
    end = find_end_of_inflation(derived["epsH"])
    if end == -1:
        end = len(T_span) - 1
    return model, bg_sol, T_span, derived, end


# ── Level 1: Single-mode trajectory ──────────────────────────────────────


@pytest.mark.fast
def test_level1_single_mode():
    """Single k-mode trajectory: Fortran and Numba must agree to 1e-5 in P_S."""
    model, bg_sol, T_span, derived, end_idx = _setup_higgs(5.70, -0.170, 52)

    k_phys = 0.002
    N_total = derived["N"][end_idx]
    N_pivot = N_total - 52.0
    pivot_idx = int(np.argmin(np.abs(derived["N"][:end_idx] - N_pivot)))
    k_pivot_code = np.exp(bg_sol[3][pivot_idx]) * bg_sol[2][pivot_idx]
    k_code = k_pivot_code * (k_phys / 0.002)

    xi, y0_m, zi, ni, t_start, t_end, _ = extract_mode_initial_conditions(
        bg_sol, T_span, end_idx, k_code, 100.0
    )

    ms_steps = 5000
    T_ms = np.linspace(t_start, t_end, ms_steps)
    k_rel = k_code * np.exp(-ni)

    bg_coefs = build_numba_splines(bg_sol, T_span)
    out_numba = numba_run_ms(bg_sol, T_span, T_ms, ni, k_code, model, bg_coefs=bg_coefs)

    bc_arr = _pack_spline_coefs(bg_coefs)
    yv = zi / k_rel
    vi = 1.0 / np.sqrt(2.0 * k_rel)
    y0_state = np.array([
        vi, k_rel/np.sqrt(2.0*k_rel)*yv, yv*vi, -k_rel/np.sqrt(2.0*k_rel)*(1-yv*yv),
        vi, k_rel/np.sqrt(2.0*k_rel)*yv, yv*vi, -k_rel/np.sqrt(2.0*k_rel)*(1-yv*yv),
    ], dtype=np.float64)
    out_fort = _fort.integrate_dp5_wrapper(
        y0_state, t_start, t_end, T_ms, bc_arr, k_rel, ni, model.S, model.v0, model.alpha, 0
    )

    abs_diff = np.abs(out_fort - out_numba)
    rel_err = abs_diff / np.maximum(np.abs(out_numba), 1e-30)
    max_rel_err = np.max(rel_err)
    assert max_rel_err < 0.1, f"Trajectory error {max_rel_err:.4e} >= 0.1"

    y_bg_end = bg_sol[1][end_idx]
    z_bg_end = bg_sol[2][end_idx]
    n_end_rel = bg_sol[3][end_idx] - ni
    epsH = max(y_bg_end**2 / (2.0 * z_bg_end**2), 1e-30)
    inv_A2 = np.exp(-2.0 * n_end_rel)

    for label, y_end in [("Numba", out_numba[:, -1]), ("Fortran", out_fort[:, -1])]:
        zeta2 = (y_end[0]**2 + y_end[2]**2) * inv_A2 * (model.S**2) / (2.0 * epsH)
        ps = (k_rel**3 * zeta2) / (2.0 * np.pi**2)
        h2 = (y_end[4]**2 + y_end[6]**2) * inv_A2 * (model.S**2)
        pt = 4.0 * (k_rel**3 * h2) / (np.pi**2)
        if label == "Numba":
            ps_nb, pt_nb = ps, pt
        else:
            ps_ft, pt_ft = ps, pt

    assert abs(ps_ft - ps_nb) / ps_nb < 1e-5, f"P_S diff {abs(ps_ft - ps_nb) / ps_nb:.4e}"
    assert abs(pt_ft - pt_nb) / pt_nb < 1e-5, f"P_T diff {abs(pt_ft - pt_nb) / pt_nb:.4e}"


# ── Level 2+3: Grid + CAMB agreement ─────────────────────────────────────


@pytest.mark.parametrize("phi0,y0,nstar,label", [
    (6.40, -0.475, 59, "Best chi2"),
    (5.75, -0.170, 55, "Best D2"),
    (5.70, -0.170, 52, "Best balance"),
])
def test_level2_grid_agreement(phi0, y0, nstar, label):
    """P_S(k) grid: Fortran and Numba must agree to 1e-4."""
    model = HiggsModel()
    model.x0 = phi0
    model.y0 = y0

    res_nb = run_pspectrum_pipeline(
        model=model, phi0=phi0, y0=y0, N_star=nstar,
        use_numba=True, backend='numba', save_outputs=False,
    )
    res_ft = run_pspectrum_pipeline(
        model=model, phi0=phi0, y0=y0, N_star=nstar,
        use_numba=False, backend='fortran', save_outputs=False,
    )

    ps_nb, ps_ft = res_nb["P_S"], res_ft["P_S"]
    pt_nb, pt_ft = res_nb["P_T"], res_ft["P_T"]

    rel_diff_ps = np.abs(ps_ft - ps_nb) / ps_nb
    assert np.nanmax(rel_diff_ps) < 1e-4, f"{label}: P_S max diff {np.nanmax(rel_diff_ps):.4e}"
    assert np.nanmean(rel_diff_ps) < 5e-5, f"{label}: P_S mean diff {np.nanmean(rel_diff_ps):.4e}"

    rel_diff_pt = np.abs(pt_ft - pt_nb) / pt_nb
    assert np.nanmax(rel_diff_pt) < 1e-4, f"{label}: P_T max diff {np.nanmax(rel_diff_pt):.4e}"


@pytest.mark.parametrize("phi0,y0,nstar", [
    (6.40, -0.475, 59),
    (5.75, -0.170, 55),
    (5.70, -0.170, 52),
])
def test_level3_camb_agreement(phi0, y0, nstar):
    """CAMB C_ell from Fortran vs Numba P_S(k): D₂ < 0.1 μK², χ² < 0.01."""
    model = HiggsModel()
    model.x0 = phi0
    model.y0 = y0

    res_nb = run_pspectrum_pipeline(
        model=model, phi0=phi0, y0=y0, N_star=nstar,
        use_numba=True, backend='numba', save_outputs=False,
    )
    res_ft = run_pspectrum_pipeline(
        model=model, phi0=phi0, y0=y0, N_star=nstar,
        use_numba=False, backend='fortran', save_outputs=False,
    )

    for label, ps_data in [("Numba", {"k_phys": res_nb["k_phys"], "P_S": res_nb["P_S"]}),
                           ("Fortran", {"k_phys": res_ft["k_phys"], "P_S": res_ft["P_S"]})]:
        ells, C_TT, _, _ = compute_cl_full_camb(ps_data, ell_max=2500)
        D_TT = C_ell_to_d_ell(ells, C_TT)
        D2 = D_TT[ells == 2][0]
        chi2, _, _ = compute_chi2_camb(ps_data, ell_max=29)
        if label == "Numba":
            D2_nb, chi2_nb = D2, chi2
        else:
            D2_ft, chi2_ft = D2, chi2

    assert abs(D2_ft - D2_nb) < 0.1, f"D₂ diff {abs(D2_ft - D2_nb):.4f} μK²"
    assert abs(chi2_ft - chi2_nb) < 0.01, f"χ² diff {abs(chi2_ft - chi2_nb):.4f}"
