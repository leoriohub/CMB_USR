"""
Profile pipeline overhead on lab machine.
Run: python profile_lab.py
"""
import time, numpy as np
from models import HiggsModel
import inf_dyn_background as bg_solver
import inf_dyn_MS_full as ms_solver
from pspectrum_pipeline import build_bg_interpolators_fast, build_weighted_kgrid, \
    find_end_of_inflation, extract_mode_initial_conditions, get_k_pivot_code, ensure_k_pivot
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

t_ex = t_ln = t_ru = t_ps = 0.0
for i, kc in enumerate(kcs):
    t0 = time.time(); _, _, _, ni, ts, te, _ = extract_mode_initial_conditions(bg, T_bg, ei, kc, 100.0); t_ex += time.time() - t0
    t0 = time.time(); Tms = np.linspace(ts, te, 5000); t_ln += time.time() - t0
    t0 = time.time(); ms = ms_solver.run_ms_simulation(interp, ni, Tms, kc, m); t_ru += time.time() - t0
    t0 = time.time(); ms_solver.get_ms_derived_quantities_with_bg(ms, interp, Tms, m, kc, ni); t_ps += time.time() - t0

tot = t_ex + t_ln + t_ru + t_ps
print(f"{n} modes, Python solver, 5000 ms_steps")
print(f"  extract ICs:  {t_ex/n*1000:.2f}ms/mode  ({(t_ex/n/tot)*100:.0f}%)")
print(f"  linspace:     {t_ln/n*1000:.2f}ms/mode  ({(t_ln/n/tot)*100:.0f}%)")
print(f"  MS solver:    {t_ru/n*1000:.2f}ms/mode  ({(t_ru/n/tot)*100:.0f}%)")
print(f"  P_S calc:     {t_ps/n*1000:.2f}ms/mode  ({(t_ps/n/tot)*100:.0f}%)")
print(f"  total:        {tot/n*1000:.2f}ms/mode  ({tot:.1f}s total)")
