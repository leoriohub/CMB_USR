"""
Generate 2000 random Higgs configs, run Numba pipeline + CAMB, save JSONL.

Uses evaluate_config from camb_scan.py plus use_numba=True. Output compatible
with scan_stats.py and camb_scan_dashboard.py.
"""
import json, os, sys, time, random
import numpy as np
from models import HiggsModel
from pspectrum_pipeline import run_pspectrum_pipeline
from scripts.camb_wrapper import compute_cl_camb_powerlaw
from scripts.planck_data import C_ell_to_d_ell
from scripts.optimizer_utils import find_k_dip
from scripts.constants import As, k_pivot_phys
from scripts.camb_wrapper import compute_cl_full_camb as _camb

random.seed(42)
np.random.seed(42)

N = 2000
N_WORKERS = 4
LOG = os.path.expanduser("~/numba_2000_scan.jsonl")
D2_LCDM = 1028.7
CHI2_LCDM_FULL = 2573.04

k_phys = np.array([1e-5,2.154e-5,4.642e-5,0.0001,0.0001274,0.0001624,0.0002069,
    0.0002637,0.0003360,0.0004281,0.0005456,0.0006952,0.0008859,0.001129,
    0.001438,0.001833,0.002,0.002336,0.002976,0.003793,0.004833,0.006158,
    0.007848,0.01,0.04642,0.2154,1.0])

print(f"{N} configs, {len(k_phys)} k-modes, -> {LOG}")
sys.stdout.flush()

# Generate configs
configs = [(round(random.uniform(5.5,7.5),2),
            round(-random.uniform(0.05,0.8),3),
            round(random.uniform(49.0,62.0),1)) for _ in range(N)]

# Header
with open(LOG,"w") as f:
    f.write(json.dumps({"_type":"header","n":N,"k_phys":k_phys.tolist(),
        "xi":15000,"lam":0.13,"k_pivot_phys":k_pivot_phys,"As":As})+ "\n")

# LCDM baseline
ells_l, C_l, _, _ = compute_cl_camb_powerlaw(ell_max=2500)
D_lcdm = C_ell_to_d_ell(ells_l, C_l)

# Warmup Numba
print(" Warmup...", flush=True)
m = HiggsModel()
_ = run_pspectrum_pipeline(m, phi0=6.0, y0=-0.2, N_star=50.0,
    k_phys_grid=k_phys[:5], use_numba=True, n_workers=1, ms_steps=500,
    save_outputs=False)
print(" Go.", flush=True)

t0 = time.time()
with open(LOG,"a") as log:
    for i,(phi0,y0,N_star) in enumerate(configs):
        m2 = HiggsModel()
        r = run_pspectrum_pipeline(m2, phi0=phi0, y0=y0, N_star=N_star,
            k_phys_grid=k_phys, use_numba=True, n_workers=N_WORKERS,
            ms_steps=5000, save_outputs=False)

        e = {"_type":"data","eval":i+1,"total":N,"phi0":phi0,"y0":y0,"N_star":N_star}

        if r["status"]!="success":
            e["status"]="error"; e["error"]=r.get("message","?")
        else:
            PS = r["P_S"]
            kd = find_k_dip(r["k_phys"],PS) if np.any(np.isfinite(PS)) else -1.0
            e["k_dip"]=kd; e["N_total"]=round(r["metadata"].get("N_total",0),1)
            supp=0.0
            if kd>0:
                ps_dip=float(np.interp(kd,r["k_phys"],PS))
                ps_pv=float(np.interp(k_pivot_phys,r["k_phys"],PS))
                if ps_pv>0: supp=(1-ps_dip/ps_pv)*100
            e["suppression_pct"]=round(supp,1)

            try:
                ps_data={"k_phys":r["k_phys"],"P_S":PS}
                ec,CTT,_,_=_camb(ps_data,ell_max=2500)
                D=C_ell_to_d_ell(ec,CTT); d2=float(D[0])
                # Unbinned full chi2
                from scripts.chi2_analysis import chi2_unbinned
                chi2_m,chi2_l,_=chi2_unbinned(D,ec,D_lcdm,ells_l)
                e.update({"status":"ok","d2":round(d2,1),"chi2":round(chi2_m,2),
                    "chi2_lcdm":round(chi2_l,2),"dchi2":round(chi2_m-chi2_l,2),
                    "d2_lcdm":round(D2_LCDM,1)})
            except Exception as ex:
                e["status"]="camb_fail"; e["error"]=str(ex)

        log.write(json.dumps(e)+"\n"); log.flush()

        if (i+1)%100==0:
            elapsed=time.time()-t0; eta=elapsed/(i+1)*(N-i-1)
            ok=sum(1 for _ in open(LOG) if '"status":"ok"' in _)
            print(f"  [{i+1}/{N}] {elapsed:.0f}s ETA {eta:.0f}s ok={ok}",flush=True)

elapsed = time.time()-t0
ok=sum(1 for _ in open(LOG) if '"status":"ok"' in _)
print(f"Done: {N} in {elapsed:.0f}s ({elapsed/N:.2f}s/config) ok={ok}/{N}")
print(f"File: {LOG}")
