"""
2000 configs: Numba pipeline + CAMB, shared executor + model across all.
Output compatible with scan_stats.py and camb_scan_dashboard.py.
"""
import json, os, sys, time, random
import numpy as np
from concurrent.futures import ProcessPoolExecutor
from models import HiggsModel
from scripts.camb_scan import evaluate_config
from scripts.constants import As, k_pivot_phys

random.seed(42)
np.random.seed(42)

N = 2000
N_WORKERS = 4
LOG = os.path.expanduser("~/numba_2000_scan.jsonl")

k_phys = np.array([1e-5,2.154e-5,4.642e-5,0.0001,0.0001274,0.0001624,0.0002069,
    0.0002637,0.0003360,0.0004281,0.0005456,0.0006952,0.0008859,0.001129,
    0.001438,0.001833,0.002,0.002336,0.002976,0.003793,0.004833,0.006158,
    0.007848,0.01,0.04642,0.2154,1.0])

print(f"{N} configs, {len(k_phys)} k-modes, workers={N_WORKERS}, -> {LOG}")
sys.stdout.flush()

configs = [(round(random.uniform(5.5,7.5),2),
            round(-random.uniform(0.05,0.8),3),
            round(random.uniform(49.0,62.0),1)) for _ in range(N)]

class Args:
    workers = N_WORKERS
    lam = 0.13; xi = 15000.0
    ell_max = 2500; num_k = 80
    k_min = 1e-5; k_max = 1.0
args = Args()

# Header
with open(LOG,"w") as f:
    f.write(json.dumps({"_type":"header","n":N,"k_phys":k_phys.tolist(),
        "xi":args.xi,"lam":args.lam,"k_pivot_phys":k_pivot_phys,"As":As})+ "\n")

model = HiggsModel(lam=args.lam, xi=args.xi)
print(" Warmup...", flush=True)
_ = evaluate_config(6.0, -0.2, 50.0, args, k_phys_grid=k_phys[:3])
print(" Go.", flush=True)

t0 = time.time()
with open(LOG,"a") as log, ProcessPoolExecutor(N_WORKERS) as executor:
    for i,(phi0,y0,N_star) in enumerate(configs):
        e = evaluate_config(phi0, y0, N_star, args,
                            k_phys_grid=k_phys,
                            executor=executor, model=model)
        e["_type"] = "data"
        e["eval"] = i+1
        e["total"] = N
        log.write(json.dumps(e)+"\n"); log.flush()

        if (i+1)%100==0:
            elapsed=time.time()-t0; eta=elapsed/(i+1)*(N-i-1)
            ok=sum(1 for _ in open(LOG) if '"status":"ok"' in _)
            print(f"  [{i+1}/{N}] {elapsed:.0f}s ETA {eta:.0f}s ok={ok}",flush=True)

elapsed = time.time()-t0
ok=sum(1 for _ in open(LOG) if '"status":"ok"' in _)
print(f"Done: {N} in {elapsed:.0f}s ({elapsed/N:.2f}s/config) ok={ok}/{N}")
print(f"File: {LOG}")
