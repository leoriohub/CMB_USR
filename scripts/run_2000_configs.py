"""Generate 2000 random Higgs configs and run Numba pipeline + CAMB on each."""
import json, time, random, sys
import numpy as np
from models import HiggsModel
from pspectrum_pipeline import run_pspectrum_pipeline, build_weighted_kgrid
from scripts.camb_wrapper import compute_cl_full_camb
from scripts.planck_data import C_ell_to_d_ell
from scripts.constants import As, k_pivot_phys

random.seed(42)
np.random.seed(42)

# Use the same 27-mode k-grid as the original phase 2 scan for consistency
k_phys = np.array([
    1e-5, 2.154e-5, 4.642e-5, 0.0001, 0.0001274, 0.0001624, 0.0002069,
    0.0002637, 0.0003360, 0.0004281, 0.0005456, 0.0006952, 0.0008859,
    0.001129, 0.001438, 0.001833, 0.002, 0.002336, 0.002976, 0.003793,
    0.004833, 0.006158, 0.007848, 0.01, 0.04642, 0.2154, 1.0,
])

N_CONFIGS = 2000
N_WORKERS = 4

print(f"Generating {N_CONFIGS} random Higgs configs...")
print(f"k-grid: {len(k_phys)} modes")
print(f"Workers: {N_WORKERS}")
sys.stdout.flush()

configs = []
for _ in range(N_CONFIGS):
    phi0 = round(random.uniform(5.5, 7.5), 2)
    y0 = round(-random.uniform(0.05, 0.8), 3)
    N_star = round(random.uniform(49.0, 62.0), 1)
    configs.append((phi0, y0, N_star))

m = HiggsModel(lam=0.13, xi=15000.0)

t_start = time.time()
ok = 0
fail = 0
for i, (phi0, y0, N_star) in enumerate(configs):
    t0 = time.time()
    r = run_pspectrum_pipeline(
        m, phi0=phi0, y0=y0, N_star=N_star,
        k_pivot_phys=k_pivot_phys, k_phys_grid=k_phys,
        normalize_to_As=True, As=As,
        n_workers=N_WORKERS, use_numba=True,
        ms_steps=5000, save_outputs=False,
    )

    if r["status"] == "success":
        ok += 1
        ps_data = {"k_phys": r["k_phys"], "P_S": r["P_S"]}
        try:
            ells, CTT, _, _ = compute_cl_full_camb(ps_data, ell_max=2500)
            d2 = float(C_ell_to_d_ell(ells, CTT)[0])
        except Exception:
            d2 = -1
    else:
        fail += 1
        d2 = -1

    t = time.time() - t0
    elapsed = time.time() - t_start
    rate = (i + 1) / max(elapsed, 0.01)
    eta = (N_CONFIGS - i - 1) / rate if rate > 0 else 0

    if (i + 1) % 50 == 0:
        print(f"  [{i+1}/{N_CONFIGS}] {elapsed:.0f}s  ETA {eta:.0f}s  "
              f"ok={ok} fail={fail}  last: {t:.2f}s/config",
              flush=True)

elapsed = time.time() - t_start
print(f"\nDone: {N_CONFIGS} configs in {elapsed:.0f}s ({elapsed/N_CONFIGS:.2f}s/config)")
print(f"  OK: {ok}/{N_CONFIGS}  FAIL: {fail}/{N_CONFIGS}")
