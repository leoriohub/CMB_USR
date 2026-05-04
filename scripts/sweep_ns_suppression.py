#!/usr/bin/env python3
"""
Sweep (phi0, y0) to find configurations matching both n_s^MS and CMB
low-ℓ power suppression from USR dynamics.

Output: JSON with per-point results sorted by score.
Designed for remote execution on a lab machine.
"""

import argparse
import json
import os
import sys
import datetime

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from models import HiggsModel
from numerical_observables_calculation import run_inflation_protocol
from scripts.pspectrum_pipeline import run_pspectrum_pipeline, ensure_k_pivot
from scripts.constants import As, k_pivot_phys


def worker_fn(task_tuple):
    phi0, y0, xi, lam, N_star, ns_target, ns_tol, k_grid = task_tuple
    try:
        model = HiggsModel(lam=lam, xi=xi)
        ns_result = run_inflation_protocol(
            model, phi0=phi0, y0=y0, N_star=N_star, save_to_file=False,
        )
        if ns_result["status"] != "success":
            return {
                "phi0": phi0,
                "y0": y0,
                "status": "error",
                "message": "ns: " + ns_result.get("message", ""),
            }

        ns_ms = ns_result["ns"]
        ns_sr = ns_result["ns_SR"]
        r_ms = ns_result["r"]
        N_total = ns_result["N_total"]

        model_ps = HiggsModel(lam=lam, xi=xi)
        ps_result = run_pspectrum_pipeline(
            model=model_ps,
            phi0=phi0,
            y0=y0,
            k_phys_grid=k_grid,
            N_star=N_star,
            normalize_to_As=True,
            save_outputs=False,
        )
        if ps_result["status"] != "success":
            return {
                "phi0": phi0,
                "y0": y0,
                "status": "error",
                "message": "P_S: " + ps_result.get("message", ""),
                "ns_MS": ns_ms,
                "ns_SR": ns_sr,
                "N_total": N_total,
            }

        k = ps_result["k_phys"]
        P = ps_result["P_S"]

        dip_sup = 0.0
        k_dip = -1.0
        P_dip = float(As)
        i_dip_local = None
        for i in range(2, len(k) - 2):
            if P[i] < P[i - 1] and P[i] < P[i + 1] and P[i] < As:
                if i_dip_local is None or P[i] < P[i_dip_local]:
                    i_dip_local = i
        if i_dip_local is not None and k[i_dip_local] < 0.1:
            k_dip = float(k[i_dip_local])
            P_dip = float(P[i_dip_local])
            dip_sup = float((1 - P_dip / As) * 100)

        mask_low = k < 0.01
        low_k_sup = (
            float((1 - np.mean(P[mask_low]) / As) * 100)
            if np.any(mask_low)
            else 0.0
        )

        ns_score = float(np.exp(-0.5 * ((ns_ms - ns_target) / ns_tol) ** 2))
        sup_score = float(min(dip_sup / 15.0, 1.0))
        k_score = 1.0 if 0 < k_dip < 0.01 else 0.0
        total_score = ns_score + sup_score + k_score

        return {
            "phi0": phi0,
            "y0": y0,
            "status": "ok",
            "ns_MS": ns_ms,
            "ns_SR": ns_sr,
            "r_MS": r_ms,
            "N_total": N_total,
            "k_dip": k_dip,
            "P_S_dip": P_dip,
            "dip_sup_pct": dip_sup,
            "low_k_sup_pct": low_k_sup,
            "ns_score": ns_score,
            "sup_score": sup_score,
            "k_score": k_score,
            "total_score": total_score,
        }
    except Exception as e:
        return {
            "phi0": phi0,
            "y0": y0,
            "status": "error",
            "message": f"{type(e).__name__}: {e}",
        }


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.ndarray,)):
            return obj.tolist()
        return super().default(obj)


def main():
    parser = argparse.ArgumentParser(
        description="Sweep (phi0, y0) for n_s + CMB power suppression match",
    )
    parser.add_argument("--phi0-min", type=float, default=5.70)
    parser.add_argument("--phi0-max", type=float, default=6.00)
    parser.add_argument("--phi0-step", type=float, default=0.01)
    parser.add_argument("--y0-min", type=float, default=-0.15)
    parser.add_argument("--y0-max", type=float, default=-0.05)
    parser.add_argument("--y0-step", type=float, default=0.01)
    parser.add_argument("--xi", type=float, default=15000.0)
    parser.add_argument("--lam", type=float, default=0.13)
    parser.add_argument("--N-star", type=float, default=60.0)
    parser.add_argument("--ns-target", type=float, default=0.975)
    parser.add_argument("--ns-tol", type=float, default=0.005)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--output-dir", type=str, default="outputs/sweeps")
    parser.add_argument("--num-k", type=int, default=80)
    parser.add_argument(
        "--force", action="store_true", help="Force recompute (ignore cache)",
    )
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    phi0_vals = np.arange(
        args.phi0_min, args.phi0_max + 0.5 * args.phi0_step, args.phi0_step
    )
    y0_vals = np.arange(
        args.y0_min, args.y0_max + 0.5 * args.y0_step, args.y0_step
    )
    grid = [(float(phi0), float(y0)) for phi0 in phi0_vals for y0 in y0_vals]

    k_grid = np.logspace(np.log10(1e-5), np.log10(1.0), args.num_k)
    k_grid, _ = ensure_k_pivot(k_grid, k_pivot_phys)

    n_total = len(grid)
    print(f"Sweeping {n_total} grid points")
    print(f"  phi0: {args.phi0_min:.2f} -> {args.phi0_max:.2f}  step {args.phi0_step:.2f}")
    print(f"  y0:   {args.y0_min:.3f} -> {args.y0_max:.3f}  step {args.y0_step:.3f}")
    print(f"  n_s target: {args.ns_target:.3f} +/- {args.ns_tol:.3f}")
    print(f"  n_workers:  {args.n_workers}")
    print(f"  k-modes:    {len(k_grid)}")
    print()

    results = []

    if args.n_workers > 1:
        from concurrent.futures import ProcessPoolExecutor, as_completed

        task_args = [
            (
                phi0,
                y0,
                args.xi,
                args.lam,
                args.N_star,
                args.ns_target,
                args.ns_tol,
                k_grid,
            )
            for phi0, y0 in grid
        ]
        with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
            futures = {
                executor.submit(worker_fn, ta): (ta[0], ta[1]) for ta in task_args
            }
            for i, future in enumerate(as_completed(futures), 1):
                r = future.result()
                results.append(r)
                label = f"phi0={r['phi0']:.2f}, y0={r['y0']:.3f}"
                if r["status"] == "ok":
                    print(
                        f"  [{i}/{n_total}] {label} -> "
                        f"n_s={r['ns_MS']:.4f}  dip={r['dip_sup_pct']:.1f}%  "
                        f"k_dip={r['k_dip']:.2e}"
                    )
                else:
                    print(
                        f"  [{i}/{n_total}] {label} -> "
                        f"ERROR: {r.get('message', '')[:60]}"
                    )
    else:
        for i, (phi0, y0) in enumerate(grid, 1):
            r = worker_fn(
                (
                    phi0,
                    y0,
                    args.xi,
                    args.lam,
                    args.N_star,
                    args.ns_target,
                    args.ns_tol,
                    k_grid,
                )
            )
            results.append(r)
            label = f"phi0={r['phi0']:.2f}, y0={r['y0']:.3f}"
            if r["status"] == "ok":
                print(
                    f"  [{i}/{n_total}] {label} -> "
                    f"n_s={r['ns_MS']:.4f}  dip={r['dip_sup_pct']:.1f}%  "
                    f"k_dip={r['k_dip']:.2e}"
                )
            else:
                print(
                    f"  [{i}/{n_total}] {label} -> "
                    f"ERROR: {r.get('message', '')[:60]}"
                )

    ok_results = [r for r in results if r["status"] == "ok"]
    ok_results.sort(key=lambda r: r["total_score"], reverse=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    config = {
        k: (str(v) if not isinstance(v, (int, float, bool, str)) else v)
        for k, v in vars(args).items()
    }

    output = {
        "metadata": {
            "timestamp": datetime.datetime.now().isoformat(),
            "config": config,
            "grid_size": n_total,
            "n_success": len(ok_results),
            "n_errors": len(results) - len(ok_results),
        },
        "results": ok_results,
        "errors": [r for r in results if r["status"] != "ok"],
    }

    filepath = os.path.join(
        args.output_dir, f"ns_suppression_sweep_{timestamp}.json"
    )
    with open(filepath, "w") as f:
        json.dump(output, f, indent=2, cls=NumpyEncoder)
    print(
        f"\nSaved {len(ok_results)} + {len(results) - len(ok_results)} errors -> {filepath}"
    )

    top = ok_results[:20]
    print(f"\n{'=' * 68}")
    print(f"{'TOP 20 CANDIDATES (ranked by total_score)':^68}")
    print(f"{'=' * 68}")
    print(
        f"  {'#':>3} {'phi0':>6} {'y0':>7} {'n_s^MS':>8} {'dip%':>6} "
        f"{'k_dip':>10} {'ns_scr':>7} {'sup_scr':>7} {'score':>6}"
    )
    print(f"  {'-' * 60}")
    for rank, r in enumerate(top, 1):
        print(
            f"  {rank:>3} {r['phi0']:>6.2f} {r['y0']:>7.3f} {r['ns_MS']:>8.4f} "
            f"{r['dip_sup_pct']:>5.1f}% {r['k_dip']:>10.2e} "
            f"{r['ns_score']:>7.3f} {r['sup_score']:>7.3f} {r['total_score']:>6.3f}"
        )
    print(f"{'=' * 68}")

    top_path = os.path.join(
        args.output_dir, f"top_candidates_{timestamp}.json"
    )
    with open(top_path, "w") as f:
        json.dump(
            {"candidates": top, "config": config}, f, indent=2, cls=NumpyEncoder
        )
    print(f"Top 20 -> {top_path}")


if __name__ == "__main__":
    main()
