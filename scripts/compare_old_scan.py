"""
Compare Python vs Numba vs old scan chi2/d2 for a sample of configs.
"""
import json, time, numpy as np
from models import HiggsModel
from pspectrum_pipeline import run_pspectrum_pipeline
from scripts.camb_wrapper import compute_cl_full_camb, compute_cl_camb_powerlaw
from scripts.planck_data import C_ell_to_d_ell
from scripts.chi2_analysis import chi2_unbinned
from scripts.constants import As, k_pivot_phys
from numba_ms_solver import _INTEGRATOR_CACHE, _POTENTIAL_CACHE

m = HiggsModel(lam=0.13, xi=15000.0)

with open("outputs/simulations/logs/camb_phase2_20260517_200905.jsonl") as f:
    header = json.loads(f.readline())
    k_phys = np.array(header["k_phys"])

diffs_chi2_py, diffs_chi2_nb = [], []
diffs_d2_py, diffs_d2_nb = [], []
count = 0

with open("outputs/simulations/logs/camb_phase2_20260517_200905.jsonl") as f:
    f.readline()
    for line in f:
        r = json.loads(line.strip())
        if r.get("status") != "ok" or count >= 15:
            continue
        phi0, y0, N_star = r["phi0"], r["y0"], r["N_star"]
        chi2_old, d2_old = r["chi2"], r["d2"]

        for label, nb in [("Python", False), ("Numba", True)]:
            _INTEGRATOR_CACHE.clear(); _POTENTIAL_CACHE.clear()
            res = run_pspectrum_pipeline(m, phi0=phi0, y0=y0, N_star=N_star,
                k_pivot_phys=k_pivot_phys, k_phys_grid=k_phys,
                normalize_to_As=True, As=As,
                n_workers=1, use_numba=nb, ms_steps=5000, save_outputs=False)
            if res["status"] != "success":
                continue
            pd = {"k_phys": res["k_phys"], "P_S": res["P_S"]}
            ec, CTT, _, _ = compute_cl_full_camb(pd, ell_max=2500)
            Dn = C_ell_to_d_ell(ec, CTT)
            d2n = float(Dn[0])
            el, Cl, _, _ = compute_cl_camb_powerlaw(ell_max=2500)
            Dl = C_ell_to_d_ell(el, Cl)
            chi2n, _, _ = chi2_unbinned(Dn, ec, Dl, el)

            dc = chi2n - chi2_old
            dd = d2n - d2_old
            if nb:
                diffs_chi2_nb.append(dc)
                diffs_d2_nb.append(dd)
            else:
                diffs_chi2_py.append(dc)
                diffs_d2_py.append(dd)
            print(f"{label:>8} {phi0:.2f} {y0:+.3f} {N_star:.1f}:  "
                  f"chi2={chi2n:.1f} (old={chi2_old:.1f}, d={dc:+.1f})  "
                  f"d2={d2n:.1f} (old={d2_old:.1f}, d={dd:+.2f})")
        count += 1

print()
if diffs_chi2_py:
    a = np.mean(np.abs(diffs_chi2_py))
    m = max(np.abs(diffs_chi2_py))
    print(f"Python vs old: mean|dchi2|={a:.1f} max|dchi2|={m:.1f}  "
          f"mean dchi2={np.mean(diffs_chi2_py):+.1f}")
if diffs_chi2_nb:
    a = np.mean(np.abs(diffs_chi2_nb))
    m = max(np.abs(diffs_chi2_nb))
    print(f"Numba vs old:  mean|dchi2|={a:.1f} max|dchi2|={m:.1f}  "
          f"mean dchi2={np.mean(diffs_chi2_nb):+.1f}")
if diffs_d2_py:
    print(f"Python vs old: mean|dd2|={np.mean(np.abs(diffs_d2_py)):.3f} "
          f"max|dd2|={max(np.abs(diffs_d2_py)):.3f}")
if diffs_d2_nb:
    print(f"Numba vs old:  mean|dd2|={np.mean(np.abs(diffs_d2_nb)):.3f} "
          f"max|dd2|={max(np.abs(diffs_d2_nb)):.3f}")
