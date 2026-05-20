"""
Compare Python vs Numba per-mode time on lab.
"""
import time, numpy as np
from models import HiggsModel
import inf_dyn_background as bg_solver
import inf_dyn_MS_full as ms_solver
from pspectrum_pipeline import build_bg_interpolators_fast, build_weighted_kgrid, \
    find_end_of_inflation, extract_mode_initial_conditions, get_k_pivot_code, ensure_k_pivot
from numba_ms_solver import build_numba_splines, _extract_potential, make_rhs, \
    _get_integrator, _spline_eval
from scripts.constants import k_pivot_phys

m = HiggsModel(lam=0.13, xi=15000.0); m.x0 = 5.70; m.y0 = -0.170
T_bg = np.linspace(0, 2000, 10000)
bg = bg_solver.run_background_simulation(m, T_bg)
d = bg_solver.get_derived_quantities(bg, m)
ei = find_end_of_inflation(d["epsH"])
kpc, _, _ = get_k_pivot_code(bg, d, ei, 52.0)
kg = build_weighted_kgrid(1e-5, 1.0, k_pivot_phys, n_dense=40, n_outer=20)
kg, _ = ensure_k_pivot(kg, k_pivot_phys)
kcs = kpc * (kg / k_pivot_phys)
interp = build_bg_interpolators_fast(bg, T_bg)
n = len(kg)

# ── Python solver ──
t_py_s = 0.0
for kc in kcs:
    _, _, _, ni, ts, te, _ = extract_mode_initial_conditions(bg, T_bg, ei, kc, 100.0)
    Tms = np.linspace(ts, te, 5000)
    t0 = time.time()
    ms_solver.run_ms_simulation(interp, ni, Tms, kc, m)
    t_py_s += time.time() - t0
print(f"Python solver:  {t_py_s/n*1000:.2f}ms/mode  ({t_py_s:.1f}s total)")

# ── Numba solver ──
fn, df, d2f = _extract_potential(m)
rhs = make_rhs(fn, df, d2f, m.S, m.v0)
integrate = _get_integrator(m, m.S, m.v0)

t_nb_s = t_nb_bc = t_nb_ic = 0.0
for kc in kcs:
    _, _, _, ni, ts, te, _ = extract_mode_initial_conditions(bg, T_bg, ei, kc, 100.0)
    Tms = np.linspace(ts, te, 5000)
    kr = kc * np.exp(-ni)

    t0 = time.time(); bc = build_numba_splines(bg, T_bg); t_nb_bc += time.time() - t0

    t0 = time.time()
    zc = bc[2]; zi = _spline_eval(ts, *zc); yv = zi / kr
    vi = 1.0 / np.sqrt(2.0 * kr)
    y0 = np.array([vi, kr/np.sqrt(2*kr)*yv, yv*vi, -kr/np.sqrt(2*kr)*(1-yv*yv),
                   vi, kr/np.sqrt(2*kr)*yv, yv*vi, -kr/np.sqrt(2*kr)*(1-yv*yv)])
    t_nb_ic += time.time() - t0

    t0 = time.time(); ms = integrate(y0, ts, te, Tms, bc, kr, ni); t_nb_s += time.time() - t0

t_nb = t_nb_bc + t_nb_ic + t_nb_s
print(f"Numba splines:  {t_nb_bc/n*1000:.2f}ms/mode")
print(f"Numba ICs:      {t_nb_ic/n*1000:.2f}ms/mode")
print(f"Numba DP5:      {t_nb_s/n*1000:.2f}ms/mode")
print(f"Numba total:    {t_nb/n*1000:.2f}ms/mode  ({t_nb:.1f}s total)")
print(f"Speedup:        {t_py_s/t_nb:.1f}x")
