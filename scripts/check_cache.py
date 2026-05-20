"""Check cache sizes after a pipeline run."""
from models import HiggsModel
from pspectrum_pipeline import run_pspectrum_pipeline, build_weighted_kgrid
from scripts.constants import As, k_pivot_phys
from numba_ms_solver import _INTEGRATOR_CACHE, _POTENTIAL_CACHE

m = HiggsModel(lam=0.13, xi=15000.0)
k = build_weighted_kgrid(1e-5, 1.0, k_pivot_phys, n_dense=10, n_outer=5)

_POTENTIAL_CACHE.clear()
_INTEGRATOR_CACHE.clear()

r = run_pspectrum_pipeline(
    m, phi0=5.70, y0=-0.170, N_star=52.0,
    k_phys_grid=k, use_numba=True, n_workers=1, save_outputs=False,
)
ok = r["metadata"]["n_completed"]
print(f"modes: {ok}/{len(k)}  pot_cache={len(_POTENTIAL_CACHE)}  int_cache={len(_INTEGRATOR_CACHE)}")
