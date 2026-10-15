"""End-of-inflation diagnostic for Palatini Higgs -- unambiguous exit criterion.

`scripts.background_scan.find_end_of_inflation_simple` locates the first upward
crossing of eps_H = 1.0 *after* the first 5-step run with eps_H < 0.1. Under
kinetic dominance the initial eps_H spike and the subsequent freeze confuse that
heuristic: it can report a spurious early exit or fall through to the end of the
array, and the two branches are hard to tell apart from the returned N_total.

This script states the criterion directly:

  * integrate the background over the full requested horizon;
  * ignore the initial kinetic transient (N < N_skip);
  * report max(eps_H) over the remainder;
  * inflation ENDS  <=>  eps_H climbs back to >= 1 at some N > N_skip;
  * inflation NEVER ENDS  <=>  eps_H stays < 1 and keeps decaying to the horizon.

It also reports the eps_H minimum and its depth relative to the plateau value,
which is the quantity that controls whether a localized P_S(k) feature exists at
all (a *transient* dip is required; a monotone freeze produces none).

Usage:
    PYTHONPATH=. python outputs/archive/palatini_end_of_inflation.py \
        --xi 15000 --lam 0.13 --eps-init 0.0 0.5 2.0 --t-max 10000 --bg-steps 40000
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from models.registry import MODEL_REGISTRY, y0_for_initial_eps
from pspectrum_pipeline import model_from_config
from inf_dyn_background import run_background_simulation, get_derived_quantities


def analyse(model_name, params, eps_init, n_skip=1.0, x0=None):
    ics = {} if eps_init is None else {"eps_init": eps_init}
    if x0 is not None:
        ics["x0"] = x0
    # model_from_config applies model_params overrides (T_max, bg_steps) and
    # resolves ics.eps_init -> y0, so this cannot drift from the real pipeline.
    model = model_from_config({"model": model_name,
                               "model_params": dict(params),
                               "ics": ics})
    y0 = model.y0

    T = np.linspace(0.0, model.T_max, model.bg_steps)
    sol = run_background_simulation(model, T)
    d = get_derived_quantities(sol, model)

    eps = np.asarray(d["epsH"])
    N = np.asarray(d["N"])
    finite = np.isfinite(eps) & np.isfinite(N)
    eps, N = eps[finite], N[finite]
    if eps.size < 10:
        return {"eps_init": eps_init, "y0": y0, "error": "integration too short"}

    late = N > n_skip
    if not late.any():
        late = np.ones_like(N, dtype=bool)
    eps_late, N_late = eps[late], N[late]

    i_min = int(np.argmin(eps_late))
    eps_min = float(eps_late[i_min])
    N_min = float(N_late[i_min])
    eps_max_late = float(np.max(eps_late))
    eps_plateau = float(eps_late[-1])

    # Exit criterion: does eps_H reach 1 after the initial transient?
    exit_idx = np.where(eps_late >= 1.0)[0]
    ended = bool(exit_idx.size > 0)
    N_end = float(N_late[exit_idx[0]]) if ended else None

    # Transient dip: the minimum must be followed by a recovery of at least 10x.
    after = eps_late[i_min:]
    recovered = bool(after.size and np.max(after) >= 10.0 * eps_min)
    transient_dip = bool(ended and recovered and eps_min < 0.1 * max(eps_max_late, 1e-300))

    return {
        "model": model_name,
        "params": params,
        "eps_init": eps_init,
        "x0": float(model.x0),
        "y0": y0,
        "T_max": model.T_max,
        "bg_steps": model.bg_steps,
        "N_reached": float(N[-1]),
        "N_end": N_end,
        "inflation_ends": ended,
        "eps_min": eps_min,
        "N_min": N_min,
        "eps_max_late": eps_max_late,
        "eps_final": eps_plateau,
        "recovered_after_min": recovered,
        "transient_dip": transient_dip,
    }


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="PalatiniHiggsModel")
    p.add_argument("--xi", type=float, default=15000.0)
    p.add_argument("--lam", type=float, default=0.13)
    p.add_argument("--eps-init", nargs="+", type=float, default=[0.0, 0.5, 2.0])
    p.add_argument("--eps-none", action="store_true",
                   help="also run the model's own default anchor")
    p.add_argument("--x0-values", nargs="+", type=float, default=None,
                   help="scan initial field values instead of the SR anchor")
    p.add_argument("--t-max", type=float, default=10000.0)
    p.add_argument("--bg-steps", type=int, default=40000)
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--output", default=None)
    args = p.parse_args()

    params = {"lam": args.lam, "xi": args.xi,
              "T_max": args.t_max, "bg_steps": args.bg_steps}
    eps_vals = list(args.eps_init)
    if args.eps_none:
        eps_vals.append(None)
    x0_vals = list(args.x0_values) if args.x0_values else [None]

    jobs = [(args.model, params, e, 1.0, x) for x in x0_vals for e in eps_vals]
    results = []
    if args.n_workers > 1:
        with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
            futs = [pool.submit(analyse, *j) for j in jobs]
            for fut in as_completed(futs):
                results.append(fut.result())
    else:
        for j in jobs:
            results.append(analyse(*j))

    results.sort(key=lambda r: (r.get("x0") or 0.0,
                                r.get("eps_init") is None,
                                r.get("eps_init") or 0.0))
    for r in results:
        if "error" in r:
            print(f"x0={r.get('x0')} eps={r['eps_init']}: {r['error']}")
            continue
        verdict = ("ENDS" if r["inflation_ends"] else "NEVER ENDS")
        print(f"x0={r['x0']:<9.4g} eps0={str(r['eps_init']):<7} y0={r['y0']:<12.5g} "
              f"N_reached={r['N_reached']:<9.1f} "
              f"{verdict:<12} "
              f"eps_min={r['eps_min']:.3e} @N={r['N_min']:.2f} "
              f"recovered={r['recovered_after_min']} transient_dip={r['transient_dip']}")

    out = args.output or os.path.join("outputs", "simulations", "logs",
                                      "palatini_end_of_inflation.json")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as fh:
        json.dump({"args": vars(args), "results": results}, fh, indent=1)
    print(f"\nwrote {len(results)} records -> {out}")


if __name__ == "__main__":
    main()
