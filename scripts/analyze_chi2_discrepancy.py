"""
Deep analysis: Numba chi2/d2 discrepancies vs old Python scan.

Runs 50 random configs with Python, Numba (rtol=1e-8), Numba (rtol=1e-12).
Reports distribution, outliers, and correlation.
"""
import json, time, numpy as np, sys
from models import HiggsModel
from pspectrum_pipeline import run_pspectrum_pipeline
from scripts.camb_wrapper import compute_cl_full_camb, compute_cl_camb_powerlaw
from scripts.planck_data import C_ell_to_d_ell
from scripts.chi2_analysis import chi2_unbinned
from scripts.constants import As, k_pivot_phys
from numba_ms_solver import _INTEGRATOR_CACHE, _POTENTIAL_CACHE

LAB_PATH = "outputs/simulations/logs/camb_phase2_20260517_200905.jsonl"
m = HiggsModel(lam=0.13, xi=15000.0)

with open(LAB_PATH) as f:
    header = json.loads(f.readline())
    k_phys = np.array(header["k_phys"])

# Collect all valid entries
all_entries = []
with open(LAB_PATH) as f:
    f.readline()
    for line in f:
        r = json.loads(line.strip())
        if r.get("status") == "ok":
            all_entries.append(r)

print(f"Total valid entries: {len(all_entries)}")
np.random.seed(42)
sample = np.random.choice(len(all_entries), min(50, len(all_entries)), replace=False)
sample_entries = [all_entries[i] for i in sample]

results = []

for idx, rec in enumerate(sample_entries):
    phi0, y0, N_star = rec["phi0"], rec["y0"], rec["N_star"]
    chi2_old, d2_old = rec["chi2"], rec["d2"]

    for label, nb, rtol in [
        ("Python", False, 1e-12),
        ("Numba_1e-8", True, 1e-8),
        ("Numba_1e-12", True, 1e-12),
    ]:
        _INTEGRATOR_CACHE.clear()
        _POTENTIAL_CACHE.clear()

        t0 = time.time()
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

        results.append({
            "label": label, "phi0": phi0, "y0": y0, "N_star": N_star,
            "chi2_new": chi2n, "chi2_old": chi2_old,
            "d2_new": d2n, "d2_old": d2_old,
            "dchi2": chi2n - chi2_old,
            "dd2": d2n - d2_old,
            "rel_dchi2": abs(chi2n - chi2_old) / max(abs(chi2_old), 1),
            "rel_dd2": abs(d2n - d2_old) / max(abs(d2_old), 1),
            "dt": time.time() - t0,
        })

        if (idx + 1) % 10 == 0 or (label == "Numba_1e-12" and idx == 0):
            print(f"  [{idx+1}/{len(sample_entries)}] {phi0:.2f} {y0:+.3f} {N_star:.1f} "
                  f"{label}: dchi2={chi2n-chi2_old:+.1f}", flush=True)

# Print summary per solver
print("\n=== Summary ===")
for label in ["Python", "Numba_1e-8", "Numba_1e-12"]:
    r = [x for x in results if x["label"] == label]
    dcs = [x["dchi2"] for x in r]
    dds = [x["dd2"] for x in r]
    print(f"\n{label} ({len(r)} configs):")
    print(f"  chi2:  mean={np.mean(dcs):+.2f}  std={np.std(dcs):.2f}  "
          f"mean|d|={np.mean(np.abs(dcs)):.2f}  max|d|={max(np.abs(dcs)):.2f}")
    print(f"  d2:    mean={np.mean(dds):+.3f}  std={np.std(dds):.3f}  "
          f"mean|d|={np.mean(np.abs(dds)):.3f}  max|d|={max(np.abs(dds)):.3f}")

# Print worst cases
print("\n=== Worst chi2 outliers (Numba rtol=1e-8) ===")
r8 = [x for x in results if x["label"] == "Numba_1e-8"]
r8.sort(key=lambda x: abs(x["dchi2"]), reverse=True)
for r in r8[:5]:
    print(f"  phi0={r['phi0']:.2f} y0={r['y0']:+.3f} N*={r['N_star']:.1f}:  "
          f"dchi2={r['dchi2']:+.1f}  dd2={r['dd2']:+.3f}  "
          f"rel_dchi2={r['rel_dchi2']:.4f}")

print("\n=== Worst d2 outliers (Numba rtol=1e-8) ===")
r8.sort(key=lambda x: abs(x["dd2"]), reverse=True)
for r in r8[:5]:
    print(f"  phi0={r['phi0']:.2f} y0={r['y0']:+.3f} N*={r['N_star']:.1f}:  "
          f"dchi2={r['dchi2']:+.1f}  dd2={r['dd2']:+.3f}  "
          f"rel_dchi2={r['rel_dchi2']:.4f}")

# Compare Numba 1e-8 vs Numba 1e-12 side by side
print("\n=== Numba rtol=1e-8 vs rtol=1e-12 (same configs) ===")
r12 = {f"{x['phi0']:.2f}_{x['y0']:+.3f}_{x['N_star']:.1f}": x for x in results if x["label"] == "Numba_1e-12"}
r8d = {f"{x['phi0']:.2f}_{x['y0']:+.3f}_{x['N_star']:.1f}": x for x in r8}

common = set(r8d.keys()) & set(r12.keys())
dc8 = [abs(r8d[k]["dchi2"]) for k in common]
dc12 = [abs(r12[k]["dchi2"]) for k in common]
dd8 = [abs(r8d[k]["dd2"]) for k in common]
dd12 = [abs(r12[k]["dd2"]) for k in common]

print(f"  Configs compared: {len(common)}")
print(f"  chi2 mean|d|:  rtol=1e-8: {np.mean(dc8):.2f}   rtol=1e-12: {np.mean(dc12):.2f}")
print(f"  chi2 max|d|:   rtol=1e-8: {max(dc8):.2f}   rtol=1e-12: {max(dc12):.2f}")
print(f"  d2 mean|d|:   rtol=1e-8: {np.mean(dd8):.4f}   rtol=1e-12: {np.mean(dd12):.4f}")
print(f"  d2 max|d|:    rtol=1e-8: {max(dd8):.4f}   rtol=1e-12: {max(dd12):.4f}")
