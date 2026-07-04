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
from numba_ms_solver import numba_run_ms_grid, build_numba_splines
from pspectrum_pipeline import run_pspectrum_pipeline, find_end_of_inflation
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


# ── Level 1: Multi-mode grid comparison (direct backend) ────────────────


@pytest.mark.fast
def test_level1_grid_backend_agreement():
    """10 k-modes: Fortran and Numba raw P_S(k) must agree to 1e-4 before normalization."""
    model, bg_sol, T_span, derived, end_idx = _setup_higgs(5.70, -0.170, 52)

    N_total = derived["N"][end_idx]
    N_pivot = N_total - 52.0
    pivot_idx = int(np.argmin(np.abs(derived["N"][:end_idx] - N_pivot)))
    k_pivot_code = np.exp(bg_sol[3][pivot_idx]) * bg_sol[2][pivot_idx]

    k_phys = np.logspace(-5, 0, 10)
    k_codes = k_pivot_code * (k_phys / 0.002)

    bg_coefs = build_numba_splines(bg_sol, T_span)
    bc_arr = _pack_spline_coefs(bg_coefs)

    PS_nb, PT_nb, _ = numba_run_ms_grid(
        bg_sol, T_span, end_idx, k_codes, model, k_start_factor=100.0, bg_coefs=bg_coefs
    )
    PS_ft, PT_ft, _ = _fort.solve_ms_grid(
        k_codes, bc_arr, bg_sol[0], bg_sol[1], bg_sol[2], bg_sol[3], T_span,
        end_idx, 100.0, model.S, model.v0, model.alpha, 0
    )

    PS_nb, PS_ft = np.asarray(PS_nb), np.asarray(PS_ft)
    PT_nb, PT_ft = np.asarray(PT_nb), np.asarray(PT_ft)

    rel_ps = np.abs(PS_ft - PS_nb) / np.maximum(PS_nb, 1e-30)
    rel_pt = np.abs(PT_ft - PT_nb) / np.maximum(PT_nb, 1e-30)

    assert np.nanmax(rel_ps) < 1e-4, f"P_S max diff over 10 modes: {np.nanmax(rel_ps):.4e}"
    assert np.nanmax(rel_pt) < 1e-4, f"P_T max diff over 10 modes: {np.nanmax(rel_pt):.4e}"


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
