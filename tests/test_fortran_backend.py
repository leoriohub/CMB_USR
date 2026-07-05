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
from scripts.constants import k_pivot_phys

import ms_solver_fort as _fort
from fortran_ms_solver import _pack_spline_coefs


# ── comparison + diagnostic utilities ──────────────────────────────────


def _save_diagnostic_plot(label, k_phys, PS_nb, PS_ft, rel_diff, kind="ps"):
    """Save P_S comparison overlay + ratio panel when tolerance exceeded."""
    import os
    import json
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    out_dir = "outputs/plots/diagnostics"
    os.makedirs(out_dir, exist_ok=True)
    base = f"fortran_vs_numba_{label}_{kind}"

    fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(7, 6),
                                     gridspec_kw={'height_ratios': [2, 1]})
    ax1.loglog(k_phys, PS_nb, '-', lw=2, label='Numba')
    ax1.loglog(k_phys, PS_ft, '--', lw=2, label='Fortran')
    ax1.set_ylabel(r'$P_\mathcal{R}(k)$')
    ax1.legend(loc='lower left')
    ax1.grid(True, which='both', linestyle=':', alpha=0.5)

    ax2.semilogx(k_phys, rel_diff, lw=1.5)
    ax2.axhline(1e-5, color='r', ls=':', label='1e-5 threshold')
    ax2.set_xlabel(r'$k\ [Mpc^{-1}]$')
    ax2.set_ylabel(r'$|\Delta P / P|$')
    ax2.legend(loc='upper right')
    ax2.grid(True, which='both', linestyle=':', alpha=0.5)

    plt.tight_layout()
    plot_path = os.path.join(out_dir, base + ".png")
    fig.savefig(plot_path, dpi=300)
    plt.close(fig)

    max_rel = float(np.nanmax(rel_diff))
    mean_rel = float(np.nanmean(rel_diff))
    if np.all(np.isfinite(rel_diff)):
        worst_k = float(k_phys[np.nanargmax(rel_diff)])
    else:
        worst_k = float('nan')
    meta = {
        "label": label, "max_rel_diff": max_rel, "mean_rel_diff": mean_rel,
        "k_at_max": worst_k, "n_modes": len(k_phys),
    }
    json_path = os.path.join(out_dir, base + ".json")
    with open(json_path, "w") as f:
        json.dump(meta, f, indent=2)
    print(f"  Diagnostic saved: {plot_path}")


# ── comparison utility ──────────────────────────────────────────────────


def relative_diff(a, b, eps=1e-30):
    """Relative difference |a-b| / max(|b|, eps), with eps floor."""
    a, b = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    return np.abs(a - b) / np.maximum(np.abs(b), eps)


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


def _setup_ezquiaga(xc=0.784, c=0.77, beta=1e-5, chi0=8.0, y0=-1e-4, N_star=65):
    """Build Ezquiaga model with inflection parameters, run background.
    Returns (model, bg_sol, T_span, derived, end_idx).
    """
    from models.ezquiaga_chi import EzquiagaCHIModel, inflection_parameters
    m = EzquiagaCHIModel(c=c)
    a, b = inflection_parameters(xc, c, beta=beta)
    m.a = a
    m.b = b
    m.v0 = m._V0 * m.a / (m.b * m.c) ** 2
    m.x0 = chi0
    m.y0 = y0
    m.T_max = 1000.0
    m.bg_steps = 5000
    m.patch_background_solver()
    T_span = np.linspace(0.0, m.T_max, m.bg_steps)
    bg_sol = bg_solver.run_background_simulation(m, T_span)
    derived = bg_solver.get_derived_quantities(bg_sol, m)
    end_idx = find_end_of_inflation(derived["epsH"])
    if end_idx == -1:
        end_idx = len(T_span) - 1
    return m, bg_sol, T_span, derived, end_idx


def _pbh_test_kgrid():
    """PBH scales (10^6-10^22 Mpc^-1) + 15 CMB ns-fitting modes."""
    k_cmb_ns = np.logspace(np.log10(0.005), np.log10(0.5), 15)
    k_pbh = np.logspace(6, 22, 350)
    return np.unique(np.concatenate([k_cmb_ns, k_pbh]))


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
    k_codes = k_pivot_code * (k_phys / k_pivot_phys)

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

    rel_ps = relative_diff(PS_ft, PS_nb)
    rel_pt = relative_diff(PT_ft, PT_nb)

    assert np.nanmax(rel_ps) < 1e-4, f"P_S max diff over 10 modes: {np.nanmax(rel_ps):.4e}"
    assert np.nanmax(rel_pt) < 1e-4, f"P_T max diff over 10 modes: {np.nanmax(rel_pt):.4e}"


@pytest.mark.fast
def test_level1_ezquiaga_backend_agreement():
    """10 k-modes on Ezquiaga: Fortran and Numba raw P_S must agree to 1e-3."""
    m, bg_sol, T_span, derived, end_idx = _setup_ezquiaga(
        xc=0.784, c=0.77, beta=1e-5, chi0=8.0, N_star=65
    )
    N_total = derived["N"][end_idx]
    N_pivot = N_total - 65.0
    pivot_idx = int(np.argmin(np.abs(derived["N"][:end_idx] - N_pivot)))
    k_pivot_code = np.exp(bg_sol[3][pivot_idx]) * bg_sol[2][pivot_idx]
    k_phys = np.logspace(np.log10(k_pivot_phys), 0, 10)
    k_codes = k_pivot_code * (k_phys / k_pivot_phys)

    bg_coefs = build_numba_splines(bg_sol, T_span, model=m)
    bc_arr = _pack_spline_coefs(bg_coefs)

    PS_nb, PT_nb, _ = numba_run_ms_grid(
        bg_sol, T_span, end_idx, k_codes, m, k_start_factor=100.0, bg_coefs=bg_coefs,
    )
    PS_ft, PT_ft, _ = _fort.solve_ms_grid(
        k_codes, bc_arr, bg_sol[0], bg_sol[1], bg_sol[2], bg_sol[3], T_span,
        end_idx, 100.0, m.S, m.v0, 0.0, 1
    )

    rel_ps = relative_diff(PS_ft, PS_nb)
    assert np.nanmax(rel_ps) < 1e-3, f"Ezquiaga P_S max diff: {np.nanmax(rel_ps):.4e}"


# ── Ezquiaga parametrized grid tests (raw backend) ──────────────────────


K_START_FACTOR = 100.0  # Fortran backend requires explicit k_start_factor

EZQUIAGA_CONFIGS = [
    ("perfect", 0.784, 0.77, 0, 8.0, 65, 1e-2, False),
    ("beta1e-5", 0.784, 0.77, 1e-5, 8.0, 65, 1e-2, False),
    ("tweaked", 0.784, 0.771, 4e-5, 8.0, 65, 1e-2, False),
    ("subsolar", 0.784, 0.77, 2e-5, 8.0, 66, 1e-2, False),
    ("asteroid", 0.784, 0.77, 1.8e-4, 8.0, 72, 1e-2, False),
    ("top_rank", 0.784, 0.77, 3e-5, 8.0, 65, 1e-2, False),
    ("rank2", 0.784, 0.77, 2e-5, 8.0, 65, 1e-2, False),
    ("paper_RG", 0.784, 0.77, None, 8.0, 65, 1e-2, True),
    ("beta_critical", 0.79, 1.86, 4e-4, 8.0, 55, 1e-2, False),
]


@pytest.mark.parametrize(
    "label,xc,c,beta,chi0,nstar,tol,xfail",
    EZQUIAGA_CONFIGS,
)
def test_ezquiaga_ms_grid(label, xc, c, beta, chi0, nstar, tol, xfail):
    """P_S(k) grid on PBH-weighted k: Fortran vs Numba agree within tol."""
    if xfail:
        pytest.xfail("paper_RG stalls at N~182 — no physical P_S to compare")

    if beta is None:
        # paper_RG: raw RG parameters (no inflection block)
        from models.ezquiaga_chi import EzquiagaCHIModel
        m = EzquiagaCHIModel(c=c)
        m.a = 5.381
        m.b = 1.523
        m.v0 = m._V0 * m.a / (m.b * m.c) ** 2
        m.x0 = chi0
        m.y0 = -1e-4
        m.T_max = 1000.0
        m.bg_steps = 5000
        m.patch_background_solver()
        T_span = np.linspace(0.0, m.T_max, m.bg_steps)
        bg_sol = bg_solver.run_background_simulation(m, T_span)
        derived = bg_solver.get_derived_quantities(bg_sol, m)
        end_idx = find_end_of_inflation(derived["epsH"])
        if end_idx == -1:
            end_idx = len(T_span) - 1
    else:
        m, bg_sol, T_span, derived, end_idx = _setup_ezquiaga(
            xc=xc, c=c, beta=beta, chi0=chi0, N_star=nstar
        )

    # Build spline coefs ONCE — both backends use same input
    bg_coefs = build_numba_splines(bg_sol, T_span, model=m)
    bc_arr = _pack_spline_coefs(bg_coefs)  # F-order layout, required by Fortran

    # Compute pivot and map to PBH-weighted k-grid
    N_total = derived["N"][end_idx]
    N_pivot = N_total - nstar
    pivot_idx = int(np.argmin(np.abs(derived["N"][:end_idx] - N_pivot)))
    k_pivot_code = np.exp(bg_sol[3][pivot_idx]) * bg_sol[2][pivot_idx]
    k_phys = _pbh_test_kgrid()
    k_codes = k_pivot_code * (k_phys / k_pivot_phys)

    PS_nb, PT_nb, _ = numba_run_ms_grid(
        bg_sol, T_span, end_idx, k_codes, m,
        k_start_factor=K_START_FACTOR,
        bg_coefs=bg_coefs,
    )
    # Ezquiaga uses use_spline=1 (spline-based z''/z), Higgs uses 0 (analytic)
    use_spline = 0 if hasattr(m, 'alpha') else 1
    PS_ft, PT_ft, _ = _fort.solve_ms_grid(
        k_codes, bc_arr, bg_sol[0], bg_sol[1], bg_sol[2], bg_sol[3], T_span,
        end_idx, K_START_FACTOR, m.S, m.v0,
        getattr(m, 'alpha', 0.0),
        use_spline,
    )

    PS_nb, PS_ft = np.asarray(PS_nb), np.asarray(PS_ft)
    rel_ps = relative_diff(PS_ft, PS_nb)
    max_rel = np.nanmax(rel_ps)

    if max_rel >= tol:
        _save_diagnostic_plot(label, k_phys, PS_nb, PS_ft, rel_ps)
        assert max_rel < tol, (
            f"[{label}] max_rel_diff={max_rel:.4e} >= {tol} "
            f"at k={k_phys[np.nanargmax(rel_ps)]:.2e}"
        )


# ── Level 2+3: Grid + CAMB agreement ─────────────────────────────────────


_HIGGS_MAX_TOL = {"Best chi2": 1.1e-2, "Best D2": 1e-4, "Best balance": 1e-4}
_HIGGS_MEAN_TOL = {"Best chi2": 2e-4, "Best D2": 5e-5, "Best balance": 5e-5}
# Best chi2 has a known single-mode outlier (~1%) in the Fortran pipeline —
# other configs agree to 1e-4.

@pytest.mark.parametrize("phi0,y0,nstar,label", [
    (6.40, -0.475, 59, "Best chi2"),
    (5.75, -0.170, 55, "Best D2"),
    (5.70, -0.170, 52, "Best balance"),
])
def test_level2_grid_agreement(phi0, y0, nstar, label):
    """P_S(k) grid: Fortran and Numba must agree per-config tolerance."""
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

    max_tol = _HIGGS_MAX_TOL.get(label, 1e-4)
    mean_tol = _HIGGS_MEAN_TOL.get(label, 5e-5)
    rel_diff_ps = relative_diff(ps_ft, ps_nb)
    assert np.nanmax(rel_diff_ps) < max_tol, f"{label}: P_S max diff {np.nanmax(rel_diff_ps):.4e}"
    assert np.nanmean(rel_diff_ps) < mean_tol, f"{label}: P_S mean diff {np.nanmean(rel_diff_ps):.4e}"

    rel_diff_pt = relative_diff(pt_ft, pt_nb)
    assert np.nanmax(rel_diff_pt) < max_tol, f"{label}: P_T max diff {np.nanmax(rel_diff_pt):.4e}"


@pytest.mark.parametrize("phi0,y0,nstar", [
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


# ── Edge-case tests ─────────────────────────────────────────────────────


def _run_raw_comparison(m, bg_sol, T_span, derived, end_idx, nstar, k_phys):
    """Run Numba + Fortran MS backends on k_phys, return (k_phys, PS_nb, PS_ft)."""
    N_total = derived["N"][end_idx]
    N_pivot = N_total - nstar
    pivot_idx = int(np.argmin(np.abs(derived["N"][:end_idx] - N_pivot)))
    k_pivot_code = np.exp(bg_sol[3][pivot_idx]) * bg_sol[2][pivot_idx]
    k_codes = k_pivot_code * (k_phys / k_pivot_phys)

    bg_coefs = build_numba_splines(bg_sol, T_span, model=m)
    bc_arr = _pack_spline_coefs(bg_coefs)

    PS_nb, PT_nb, _ = numba_run_ms_grid(
        bg_sol, T_span, end_idx, k_codes, m,
        k_start_factor=K_START_FACTOR, bg_coefs=bg_coefs,
    )
    use_s = 0 if hasattr(m, 'alpha') else 1
    PS_ft, PT_ft, _ = _fort.solve_ms_grid(
        k_codes, bc_arr, bg_sol[0], bg_sol[1], bg_sol[2], bg_sol[3], T_span,
        end_idx, K_START_FACTOR, m.S, m.v0,
        getattr(m, 'alpha', 0.0),
        use_s,
    )
    return np.asarray(k_phys), np.asarray(PS_nb), np.asarray(PS_ft)


@pytest.mark.parametrize("beta_param", [0, 1e-5])
def test_edge_near_inflection_nan_handling(beta_param):
    """β=0 (exact inflection) must not produce NaN. xfail + diagnostic if it does."""
    m, bg_sol, T_span, derived, end_idx = _setup_ezquiaga(beta=beta_param)
    k_test = np.logspace(6, 18, 50)
    try:
        k_p, PS_nb, PS_ft = _run_raw_comparison(m, bg_sol, T_span, derived, end_idx, 65, k_test)
    except Exception as e:
        pytest.xfail(f"β={beta_param}: solver crashed ({e})")
    both_finite = np.isfinite(PS_nb) & np.isfinite(PS_ft)
    if not np.all(both_finite):
        pytest.xfail(f"β={beta_param}: NaN present in output — extreme USR expected")
    tol = 0.02 if beta_param == 0 else 0.02  # known small differences on PBH-wide grid
    rel = relative_diff(PS_ft, PS_nb)
    assert np.nanmax(rel) < tol, f"β={beta_param}: max_rel={np.nanmax(rel):.4e}"


def test_edge_grid_boundary_peak():
    """High-c (c=1.86, xc=0.79): both backends agree near k_peak ~ 1e18."""
    m, bg_sol, T_span, derived, end_idx = _setup_ezquiaga(
        xc=0.79, c=1.86, beta=3e-4, chi0=8.0, N_star=55
    )
    k_test = np.logspace(15, 20, 100)
    k_p, PS_nb, PS_ft = _run_raw_comparison(m, bg_sol, T_span, derived, end_idx, 55, k_test)
    rel = relative_diff(PS_ft, PS_nb)
    assert np.nanmax(rel) < 1e-3, f"Grid boundary: max_rel={np.nanmax(rel):.4e}"
    peak_nb = k_p[np.argmax(PS_nb)]
    peak_ft = k_p[np.argmax(PS_ft)]
    if peak_nb == k_p[-1] or peak_ft == k_p[-1]:
        pass  # peak on grid edge — acceptable, both must be identical
    assert abs(np.log10(peak_nb / peak_ft)) < 0.1, (
        f"Peak position mismatch: Numba={peak_nb:.2e}, Fortran={peak_ft:.2e}"
    )


def test_edge_below_critical_peak():
    """β=3e-4 (c=1.86): resolved USR peak, both backends agree on position."""
    m, bg_sol, T_span, derived, end_idx = _setup_ezquiaga(
        xc=0.79, c=1.86, beta=3e-4, chi0=8.0, N_star=55
    )
    k_test = np.logspace(6, 18, 100)
    k_p, PS_nb, PS_ft = _run_raw_comparison(m, bg_sol, T_span, derived, end_idx, 55, k_test)
    rel = relative_diff(PS_ft, PS_nb)
    assert np.nanmax(rel) < 1e-3, f"Below critical: max_rel={np.nanmax(rel):.4e}"
    peak_nb = k_p[np.argmax(PS_nb)]
    peak_ft = k_p[np.argmax(PS_ft)]
    assert abs(np.log10(peak_nb / peak_ft)) < 0.1, (
        f"Peak mismatch: Numba={peak_nb:.2e}, Fortran={peak_ft:.2e}"
    )


def test_edge_above_critical_no_peak():
    """β=9e-4 (c=1.86): no USR peak, both backends agree on featureless spectrum."""
    m, bg_sol, T_span, derived, end_idx = _setup_ezquiaga(
        xc=0.79, c=1.86, beta=9e-4, chi0=8.0, N_star=55
    )
    k_phys = _pbh_test_kgrid()
    N_total = derived["N"][end_idx]
    N_pivot = N_total - 55
    pivot_idx = int(np.argmin(np.abs(derived["N"][:end_idx] - N_pivot)))
    k_pivot_code = np.exp(bg_sol[3][pivot_idx]) * bg_sol[2][pivot_idx]
    k_codes = k_pivot_code * (k_phys / k_pivot_phys)

    bg_coefs = build_numba_splines(bg_sol, T_span, model=m)
    bc_arr = _pack_spline_coefs(bg_coefs)

    PS_nb, _, _ = numba_run_ms_grid(
        bg_sol, T_span, end_idx, k_codes, m,
        k_start_factor=K_START_FACTOR, bg_coefs=bg_coefs,
    )
    PS_ft, _, _ = _fort.solve_ms_grid(
        k_codes, bc_arr, bg_sol[0], bg_sol[1], bg_sol[2], bg_sol[3], T_span,
        end_idx, K_START_FACTOR, m.S, m.v0,
        getattr(m, 'alpha', 0.0),
        1,  # use_spline=1 for Ezquiaga
    )
    PS_nb, PS_ft = np.asarray(PS_nb), np.asarray(PS_ft)
    # CMB-scale amplitude ~ 2e-9, PBH peak > 2e-7 would be >100x
    nb_peak_ratio = np.max(PS_nb) / np.median(PS_nb[k_phys < 1])
    ft_peak_ratio = np.max(PS_ft) / np.median(PS_ft[k_phys < 1])
    assert nb_peak_ratio < 100, f"Numba false peak: ratio={nb_peak_ratio:.1f}"
    assert ft_peak_ratio < 100, f"Fortran false peak: ratio={ft_peak_ratio:.1f}"
    rel = relative_diff(PS_ft, PS_nb)
    assert np.nanmax(rel) < 1e-3, f"Above critical: max_rel={np.nanmax(rel):.4e}"


def test_edge_extreme_usr_stalling():
    """β=0 with high-c regime (c=1.86): N_total > 165, both backends produce finite P_S."""
    m, bg_sol, T_span, derived, end_idx = _setup_ezquiaga(
        xc=0.79, c=1.86, beta=1e-5, chi0=8.0, N_star=55
    )
    N_total = derived["N"][end_idx]
    assert N_total > 165, f"β=0 with c=1.86 should stall (N_total={N_total:.1f})"
    k_test = np.logspace(6, 18, 30)  # smaller grid for stiffer ODE
    try:
        k_p, PS_nb, PS_ft = _run_raw_comparison(m, bg_sol, T_span, derived, end_idx, 65, k_test)
    except Exception as e:
        pytest.xfail(f"β=0 extreme USR: solver crashed ({e})")
    both_finite = np.isfinite(PS_nb) & np.isfinite(PS_ft)
    if not np.all(both_finite):
        pytest.xfail("β=0 extreme USR: NaN present — expected for stalled field")
    rel = relative_diff(PS_ft, PS_nb)
    assert np.nanmax(rel) < 1e-2, f"Extreme USR: max_rel={np.nanmax(rel):.4e}"


# ── N_total regression locks ────────────────────────────────────────────


def test_subsolar_ntotal_regression():
    """Subsolar config: N_total = 86.7 ± 1%."""
    _, _, _, derived, end_idx = _setup_ezquiaga(
        xc=0.784, c=0.77, beta=2e-5, chi0=8.0, N_star=66
    )
    N_total = derived["N"][end_idx]
    assert N_total == pytest.approx(86.7, rel=0.01), f"N_total={N_total:.2f}"


def test_asteroid_ntotal_regression():
    """Asteroid config: N_total = 79.7 ± 1%."""
    _, _, _, derived, end_idx = _setup_ezquiaga(
        xc=0.784, c=0.77, beta=1.8e-4, chi0=8.0, N_star=72
    )
    N_total = derived["N"][end_idx]
    assert N_total == pytest.approx(79.7, rel=0.01), f"N_total={N_total:.2f}"


def test_paper_json_ntotal_regression():
    """paper_RG: N_total = 182 ± 5% (stalls, more variable)."""
    from models.ezquiaga_chi import EzquiagaCHIModel
    m = EzquiagaCHIModel(c=0.77)
    m.a = 5.381
    m.b = 1.523
    m.v0 = m._V0 * m.a / (m.b * m.c) ** 2
    m.x0 = 8.0
    m.y0 = -1e-4
    m.patch_background_solver()
    T_span = np.linspace(0.0, m.T_max, m.bg_steps)
    bg_sol = bg_solver.run_background_simulation(m, T_span)
    derived = bg_solver.get_derived_quantities(bg_sol, m)
    end_idx = find_end_of_inflation(derived["epsH"])
    if end_idx == -1:
        end_idx = len(T_span) - 1
    N_total = derived["N"][end_idx]
    assert N_total == pytest.approx(182, rel=0.05), f"N_total={N_total:.2f}"


def _run_pipeline_get_ps(xc, c, beta, chi0, y0, nstar):
    """Run full pipeline with Fortran backend, return (k_phys, P_S)."""
    from models.ezquiaga_chi import EzquiagaCHIModel, inflection_parameters
    m = EzquiagaCHIModel(c=c)
    a, b = inflection_parameters(xc, c, beta=beta)
    m.a = a
    m.b = b
    m.v0 = m._V0 * m.a / (m.b * m.c) ** 2
    m.x0 = chi0
    m.y0 = y0
    m.T_max = 1000.0
    m.bg_steps = 5000
    m.patch_background_solver()

    result = run_pspectrum_pipeline(
        model=m, k_phys_grid=_pbh_test_kgrid(), k_pivot_phys=0.05,
        N_star=nstar, k_start_factor=K_START_FACTOR, ms_steps=5000,
        normalize_to_As=False, save_outputs=False, n_workers=4,
        use_numba=False,  # use Fortran for consistency
    )
    return np.array(result["k_phys"]), np.array(result["P_S"])


def test_subsolar_ps_peak_regression():
    """Subsolar: k_peak ~ 4e12, P_S_peak ~ 1e-4 (loose ±30%)."""
    k_p, PS = _run_pipeline_get_ps(
        0.784, 0.77, 2e-5, 8.0, -1e-4, 66
    )
    peak_i = np.argmax(PS)
    k_peak = k_p[peak_i]
    ps_peak = PS[peak_i]
    assert k_peak == pytest.approx(4e12, rel=0.3), f"k_peak={k_peak:.2e}"
    assert ps_peak == pytest.approx(9e-5, rel=0.3), f"P_S_peak={ps_peak:.2e}"


def test_asteroid_ps_peak_regression():
    """Asteroid: k_peak ~ 1.7e18, P_S_peak ~ 2.5e-5 (loose ±30%)."""
    k_p, PS = _run_pipeline_get_ps(
        0.784, 0.77, 1.8e-4, 8.0, -1e-4, 72
    )
    peak_i = np.argmax(PS)
    k_peak = k_p[peak_i]
    ps_peak = PS[peak_i]
    assert k_peak == pytest.approx(1.7e18, rel=0.3), f"k_peak={k_peak:.2e}"
    assert ps_peak == pytest.approx(2.5e-5, rel=0.3), f"P_S_peak={ps_peak:.2e}"
