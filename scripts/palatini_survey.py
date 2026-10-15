"""Parameter survey over any registry model -- background observables, parallel.

Thin driver around `scripts.background_scan.analyze_background`, which is
model-agnostic: it takes an `InflationModel` plus (x0, y0, N_star) and returns
N_total, the pivot slow-roll quantities and the USR classification. This adds
the tensor-to-scalar ratio and an initial-condition parametrisation that is
comparable across models, then fans the configs out over processes.

Why initial eps_H rather than y0: the code-unit y0 scale differs by orders of
magnitude between frames (metric Higgs starts at y0 ~ -0.1, Palatini at
y0 ~ -1e-13), so sweeping y0 directly is not a like-for-like comparison. Fixing
eps_H(0) instead gives every config the same initial kinetic dominance.

Usage:
    PYTHONPATH=. python scripts/palatini_survey.py \
        --model PalatiniHiggsModel --xi 1e4 1e6 1e9 \
        --eps-init 0.5 1.0 2.0 --n-star 55 --n-workers 20

Outputs one JSON record per config to a JSONL log (crash-safe, append-only).
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

from models.registry import MODEL_REGISTRY, model_names, y0_for_initial_eps
from pspectrum_pipeline import model_from_config
from scripts.background_scan import analyze_background


def run_config(job):
    """Run one (model, IC, N_star) config and return its observables record."""
    (model_name, params, x0, y0, n_star, k_pivot) = job
    t0 = time.time()
    # Go through model_from_config so model_params overrides (T_max, bg_steps,
    # ...) and the ics contract behave exactly as they do in the pipeline.
    model = model_from_config({
        "model": model_name,
        "model_params": params,
        "ics": {"x0": x0, "y0": y0},
    })

    info = analyze_background(model, x0, y0, n_star, pivot_k=k_pivot)
    record = {
        "model": model_name,
        "model_params": params,
        "x0": x0,
        "y0": y0,
        "N_star": n_star,
        "k_pivot": k_pivot,
    }
    if info is None:
        record["ok"] = False
        record["reason"] = "N_total < N_star"
    else:
        record.update(info)
        record["ok"] = True
        # r = 16 eps_H evaluated at the pivot, same convention as
        # inf_dyn_background.get_derived_quantities.
        record["r_pivot"] = 16.0 * info["eps_pivot"]
    record["wall_s"] = round(time.time() - t0, 3)
    return record


def build_jobs(args):
    jobs = []
    params = {}
    if args.lam is not None:
        params["lam"] = args.lam
    if args.bg_steps is not None:
        params["bg_steps"] = args.bg_steps
    if args.t_max is not None:
        params["T_max"] = args.t_max
    xi_values = list(args.xi_list or [])
    if args.xi is not None:
        xi_values.append(args.xi)
    if not xi_values:
        xi_values = [None]
    for xi in xi_values:
        p = dict(params)
        if xi is not None:
            p["xi"] = xi
        factory = MODEL_REGISTRY[args.model]
        probe = factory(p, {})
        for eps in args.eps_init:
            try:
                y0 = y0_for_initial_eps(probe, eps)
            except ValueError:
                continue
            for n_star in args.n_star:
                jobs.append((args.model, p, probe.x0, y0, n_star, args.k_pivot))
    return jobs


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", default="PalatiniHiggsModel", choices=model_names())
    p.add_argument("--xi-list", nargs="+", type=float, default=None,
                    help="xi values to sweep (models with a non-minimal coupling)")
    p.add_argument("--xi", type=float, default=None, help="single xi")
    p.add_argument("--lam", type=float, default=None)
    p.add_argument("--eps-init", nargs="+", type=float, default=[0.5, 1.0, 2.0],
                   help="initial kinetic dominance eps_H(0) values")
    p.add_argument("--n-star", nargs="+", type=float, default=[55.0])
    p.add_argument("--k-pivot", type=float, default=0.002)
    p.add_argument("--bg-steps", type=int, default=None,
                   help="background grid resolution (convergence check)")
    p.add_argument("--t-max", type=float, default=None,
                   help="code-time integration horizon (convergence check)")
    p.add_argument("--n-workers", type=int, default=1)
    p.add_argument("--output", default=None)
    return p.parse_args()


def main():
    args = parse_args()
    jobs = build_jobs(args)
    if not jobs:
        raise SystemExit("no configs to run (check --xi-list / --eps-init)")
    print(f"{len(jobs)} configs, model={args.model}, n_workers={args.n_workers}")

    out = args.output or os.path.join("outputs", "simulations", "logs",
                                      f"survey_{args.model.lower()}.jsonl")
    os.makedirs(os.path.dirname(out), exist_ok=True)

    results = []
    with open(out, "a") as fh:
        if args.n_workers > 1:
            with ProcessPoolExecutor(max_workers=args.n_workers) as pool:
                futs = {pool.submit(run_config, j): j for j in jobs}
                for n, fut in enumerate(as_completed(futs), 1):
                    rec = fut.result()
                    results.append(rec)
                    fh.write(json.dumps(rec, cls=_NumpyEncoder) + "\n")
                    fh.flush()
                    print(f"[{n}/{len(jobs)}] {_line(rec)}", flush=True)
        else:
            for n, j in enumerate(jobs, 1):
                rec = run_config(j)
                results.append(rec)
                fh.write(json.dumps(rec, cls=_NumpyEncoder) + "\n")
                fh.flush()
                print(f"[{n}/{len(jobs)}] {_line(rec)}", flush=True)

    print(f"\nwrote {len(results)} records -> {out}")


def _line(rec):
    if not rec.get("ok"):
        return f"{rec['model']} x0={rec['x0']:.4g} y0={rec['y0']:.3g} -> {rec.get('reason')}"
    return (f"{rec['model']} x0={rec['x0']:.4g} y0={rec['y0']:.3g} "
            f"Nstar={rec['N_star']:.0f} Ntot={rec['N_total']:.1f} "
            f"ns={rec['n_s_sr_formula']:.5f} r={rec['r_pivot']:.3e} "
            f"eps={rec['eps_pivot']:.3e} {rec['usr_type']}")


class _NumpyEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, (np.integer,)):
            return int(o)
        if isinstance(o, (np.floating,)):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        return super().default(o)


if __name__ == "__main__":
    main()
