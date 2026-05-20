"""Check Numba cache hit rate and per-mode timing on lab."""
import time, numpy as np
from models import HiggsModel
import inf_dyn_background as bg_solver
from pspectrum_pipeline import build_weighted_kgrid, find_end_of_inflation, \
    extract_mode_initial_conditions, get_k_pivot_code, ensure_k_pivot
from numba_ms_solver import build_numba_splines, _extract_potential, make_rhs, \
    _get_integrator, _INTEGRATOR_CACHE, _spline_eval
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
bc = build_numba_splines(bg, T_bg)

integrate = _get_integrator(m, m.S, m.v0)
print(f"Cache pre-loop: {len(_INTEGRATOR_CACHE)} entries")

for i in range(min(10, len(kcs))):
    kc = kcs[i]
    _, _, _, ni, ts, te, _ = extract_mode_initial_conditions(bg, T_bg, ei, kc, 100.0)
    Tms = np.linspace(ts, te, 5000)
    kr = kc * np.exp(-ni)
    zc = bc[2]; zi = _spline_eval(ts, *zc); yv = zi / kr
    vi = 1.0 / np.sqrt(2.0 * kr)
    y0 = np.array([vi, kr/np.sqrt(2*kr)*yv, yv*vi, -kr/np.sqrt(2*kr)*(1-yv*yv),
                   vi, kr/np.sqrt(2*kr)*yv, yv*vi, -kr/np.sqrt(2*kr)*(1-yv*yv)])
    t0 = time.time()
    ms = integrate(y0, ts, te, Tms, bc, kr, ni)
    dt = time.time() - t0
    print(f"  mode {i}: {dt*1000:.1f}ms  cache={len(_INTEGRATOR_CACHE)}")
