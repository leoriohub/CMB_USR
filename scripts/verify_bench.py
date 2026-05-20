"""Quick verification benchmark."""
import time
from models import HiggsModel
from pspectrum_pipeline import run_pspectrum_pipeline, build_weighted_kgrid
from scripts.constants import As, k_pivot_phys

m = HiggsModel(lam=0.13, xi=15000.0)
k = build_weighted_kgrid(1e-5, 1.0, k_pivot_phys, n_dense=200, n_outer=100)
n = len(k)
print(f"Grid: {n} modes")

for label, nb in [("Python", False), ("Numba", True)]:
    for nw in [1, 4]:
        t0 = time.time()
        r = run_pspectrum_pipeline(
            m, phi0=5.70, y0=-0.170, N_star=52.0,
            k_phys_grid=k, use_numba=nb, n_workers=nw,
            save_outputs=False,
        )
        t = time.time() - t0
        ok = r["metadata"]["n_completed"]
        rate = ok / max(t, 0.01)
        print(f"  {label} {nw}w: {t:.2f}s  {rate:.0f} modes/s  ({ok}/{n} ok)")
