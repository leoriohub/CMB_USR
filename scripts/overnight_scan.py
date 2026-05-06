"""
Overnight 3D scan over (phi0, y0, N_star) for Higgs USR.

Wide, coarse reconnaissance across a large (phi0, y0) window with
N_star scanned around an auto-solved center. Intended for iterative
coarse-to-fine search planning.
Saves EVERY evaluation instantly to JSONL with flush().
If power fails, the JSONL log is preserved and the scan can resume by
reading existing entries on restart.

Usage:
  python scripts/overnight_scan.py                              # full scan
  python scripts/overnight_scan.py --n-phi0 3 --n-y0 3          # quick test
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.pspectrum_pipeline import run_pspectrum_pipeline, find_end_of_inflation
from scripts.sachs_wolfe import compute_cl_sw
from scripts.planck_data import get_planck_data_asymmetric
from scripts.constants import As, k_pivot_phys, T_cmb
from models import HiggsModel
import inf_dyn_background as bg_solver


def _write_log(log_file, entry):
    log_file.write(json.dumps(entry) + "\n")
    log_file.flush()


def load_existing(log_path):
    completed = set()
    if os.path.exists(log_path):
        with open(log_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    phi0 = rec.get("phi0")
                    y0 = rec.get("y0")
                    ns = rec.get("N_star")
                    if phi0 is not None and y0 is not None and ns is not None:
                        completed.add((round(phi0, 4), round(y0, 4), round(ns, 4)))
                except json.JSONDecodeError:
                    pass
    return completed


def find_k_dip(k_phys, P_S):
    mask = k_phys < 0.01
    if not np.any(mask):
        return -1.0
    k_sub = k_phys[mask]
    ps_sub = P_S[mask]
    i_dip = int(np.argmin(ps_sub))
    if i_dip == 0 or i_dip == len(ps_sub) - 1:
        return -1.0
    return float(k_sub[i_dip])


def compute_chi2(pipeline_result, planck_data):
    ells_pl, D_pl, D_err_lower, D_err_upper = planck_data
    D_err = 0.5 * (D_err_lower + D_err_upper)
    mask = ells_pl <= 29
    ells_pl_m = ells_pl[mask]
    D_pl_m = D_pl[mask]
    D_err_m = D_err[mask]

    k_phys = pipeline_result["k_phys"]
    P_S = pipeline_result["P_S"]
    ells, C_ell = compute_cl_sw({"k_phys": k_phys, "P_S": P_S}, ell_max=30)
    D_ell = ells * (ells + 1) / (2 * np.pi) * C_ell * T_cmb**2 * 1e12

    D_ip = np.interp(ells_pl_m, ells, D_ell)
    return float(np.sum(((D_ip - D_pl_m) / D_err_m)**2))


def compute_dip_suppression(k_phys, P_S):
    mask = k_phys < 0.01
    if not np.any(mask):
        return 0.0, 0.0
    k_sub = k_phys[mask]
    ps_sub = P_S[mask]
    ps_min = ps_sub.min()
    ps_pivot = float(np.interp(k_pivot_phys, k_phys, P_S))
    suppression = (1 - ps_min / ps_pivot) * 100
    return float(suppression), float(ps_min)


def run_single(model, phi0, y0, N_star, args):
    try:
        result = run_pspectrum_pipeline(
            model=model, phi0=phi0, y0=y0,
            k_min=args.k_min, k_max=args.k_max,
            k_pivot_phys=k_pivot_phys,
            N_star=N_star,
            normalize_to_As=True, As=As,
            num_k=args.num_k,
            n_workers=args.workers,
            save_outputs=False,
        )
        if result["status"] != "success":
            return {"status": "pipe_error", "msg": result.get("message", "")}
        chi2_val = compute_chi2(result, args.planck_data)
        k_dip = find_k_dip(result["k_phys"], result["P_S"])
        supp, _ = compute_dip_suppression(result["k_phys"], result["P_S"])
        return {
            "status": "ok",
            "chi2": round(chi2_val, 2),
            "k_dip": k_dip,
            "suppression_pct": round(supp, 1),
        }
    except Exception as e:
        return {"status": "fail", "error": str(e)}


def scan_nstar(model, phi0, y0, N_total, dip_N, nstar_auto, args, log_file, done, total, t0):
    """Scan N_star around the auto-solved value."""
    nstar_min = max(nstar_auto - args.nstar_window, 10.0)
    nstar_max = min(nstar_auto + args.nstar_window, N_total - 1)
    nstar_vals = np.arange(nstar_min, nstar_max + 0.01, args.nstar_step)

    results = []
    for N_star in nstar_vals:
        N_star = round(N_star, 4)
        key = (round(phi0, 4), round(y0, 4), N_star)
        if key in args.completed:
            done[0] += 1
            continue

        done[0] += 1
        entry = {
            "eval": done[0], "total": total,
            "phi0": round(phi0, 4), "y0": round(y0, 4),
            "N_star": N_star,
            "N_total": round(N_total, 1),
            "dip_N": round(dip_N, 1),
            "nstar_auto": round(nstar_auto, 2),
            "timestamp": datetime.now().isoformat(),
        }

        res = run_single(model, phi0, y0, N_star, args)
        entry.update(res)

        _write_log(log_file, entry)
        results.append(entry)

        elapsed = time.time() - t0
        eta = elapsed / done[0] * (total - done[0]) if done[0] > 0 else 0
        chi2_str = f"chi2={res.get('chi2', '?'):.1f}" if "chi2" in res else f"SKIP ({res.get('status', '?')})"
        print(f"\r  [{done[0]:4d}/{total}] phi0={phi0:.2f} y0={y0:+.3f} "
              f"N*={N_star:.1f} {chi2_str} "
              f"k_dip={res.get('k_dip', 0):.2e} supp={res.get('suppression_pct', 0):+.1f}%  "
              f"ETA {eta/60:.0f}m", end="", flush=True)

    return results


def main(args):
    planck = get_planck_data_asymmetric()
    args.planck_data = planck

    phi0_vals = np.linspace(args.phi0_min, args.phi0_max, args.n_phi0)
    y0_vals = np.linspace(args.y0_min, args.y0_max, args.n_y0)

    # Estimate total: for each (phi0, y0), ~21 N_star values
    n_nstar_per = int(2 * args.nstar_window / args.nstar_step) + 1
    total_est = len(phi0_vals) * len(y0_vals) * n_nstar_per

    print(f"{'='*65}", flush=True)
    print(f"  Overnight Scan — Higgs USR", flush=True)
    print(f"  Parameters: (phi0, y0, N_star)", flush=True)
    print(f"  phi0: [{args.phi0_min:.2f}, {args.phi0_max:.2f}] x {args.n_phi0}", flush=True)
    print(f"  y0:   [{args.y0_min:.3f}, {args.y0_max:.3f}] x {args.n_y0}", flush=True)
    print(f"  N_star: auto_N +/- {args.nstar_window} step {args.nstar_step}", flush=True)
    print(f"  Estimated evaluations: ~{total_est}", flush=True)
    print(f"  Log: {args.log}", flush=True)
    print(f"{'='*65}", flush=True)

    os.makedirs(os.path.dirname(args.log), exist_ok=True)
    log_file = open(args.log, "a")
    args.completed = load_existing(args.log)
    print(f"  Found {len(args.completed)} completed entries, skipping those.",
          flush=True)

    done = [0]
    all_results = []
    t0 = time.time()
    k_dip_center = args.k_dip_target

    for phi0 in phi0_vals:
        for y0 in y0_vals:
            # Quick background check
            model_check = HiggsModel(lam=args.lam, xi=args.xi)
            model_check.phi0 = phi0
            model_check.y0 = y0
            T_bg = np.linspace(0, model_check.T_max, model_check.bg_steps)
            try:
                sol = bg_solver.run_background_simulation(model_check, T_bg)
                d = bg_solver.get_derived_quantities(sol, model_check)
            except Exception:
                done[0] += n_nstar_per
                print(f"\r  [{done[0]:4d}/{total_est}] phi0={phi0:.2f} y0={y0:+.3f} "
                      f"SKIP (bg_fail)", end="", flush=True)
                continue

            end_idx = find_end_of_inflation(d["epsH"])
            if end_idx == -1:
                end_idx = len(d["epsH"]) - 1
            N_total = float(d["N"][end_idx])

            eps_inf = d["epsH"][(d["N"] >= 0) & (d["N"] < N_total)]
            if len(eps_inf) < 10:
                done[0] += n_nstar_per
                print(f"\r  [{done[0]:4d}/{total_est}] phi0={phi0:.2f} y0={y0:+.3f} "
                      f"SKIP (no_usr)", end="", flush=True)
                continue
            eps_min = float(np.min(eps_inf))
            if eps_min > args.eps_min_threshold:
                done[0] += n_nstar_per
                print(f"\r  [{done[0]:4d}/{total_est}] phi0={phi0:.2f} y0={y0:+.3f} "
                      f"SKIP (sr_only)", end="", flush=True)
                continue
            dip_N = float(d["N"][np.argmin(eps_inf[10:]) + 10])
            N_after_dip = N_total - dip_N
            delta_N = np.log(k_pivot_phys / k_dip_center)
            nstar_auto = N_after_dip - delta_N

            if nstar_auto <= 0 or nstar_auto > N_total:
                done[0] += n_nstar_per
                print(f"\r  [{done[0]:4d}/{total_est}] phi0={phi0:.2f} y0={y0:+.3f} "
                      f"SKIP (bad_nstar)", end="", flush=True)
                continue

            model = HiggsModel(lam=args.lam, xi=args.xi)
            res = scan_nstar(model, phi0, y0, N_total, dip_N,
                             nstar_auto, args,
                             log_file, done, total_est, t0)
            all_results.extend(res)

    log_file.close()
    elapsed = time.time() - t0
    print(f"\n\n  Scan complete in {elapsed:.0f}s ({elapsed/3600:.1f}h)", flush=True)

    # Print best results
    ok = [r for r in all_results if r.get("status") == "ok" and "chi2" in r]
    ok.sort(key=lambda r: r["chi2"])
    print(f"\n  Best 20 configurations:", flush=True)
    print(f"    {'phi0':>6}  {'y0':>7}  {'N*':>6}  {'chi2':>6}  {'k_dip':>10}  {'supp%':>7}", flush=True)
    print(f"    {'-'*48}", flush=True)
    for r in ok[:20]:
        print(f"    {r['phi0']:6.2f}  {r['y0']:7.3f}  {r['N_star']:6.1f}  "
              f"{r['chi2']:6.2f}  {r['k_dip']:10.2e}  {r['suppression_pct']:7.1f}", flush=True)

    summary_path = os.path.join(
        ROOT_DIR, "outputs", "simulations", "scans",
        f"overnight_best_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(ok[:50], f, indent=2)
    print(f"\n  Best 50 saved -> {summary_path}", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(
        description="Overnight 3D scan: (phi0, y0, N_star) for Higgs USR")
    # Model
    p.add_argument("--xi", type=float, default=15000.0)
    p.add_argument("--lam", type=float, default=0.13)
    # Grid
    p.add_argument("--phi0-min", type=float, default=4.0)
    p.add_argument("--phi0-max", type=float, default=7.0)
    p.add_argument("--y0-min", type=float, default=-1.0)
    p.add_argument("--y0-max", type=float, default=-0.01)
    p.add_argument("--n-phi0", type=int, default=16)
    p.add_argument("--n-y0", type=int, default=16)
    # N_star
    p.add_argument("--nstar-window", type=float, default=5.0,
                   help="Scan N_star in [auto - window, auto + window]")
    p.add_argument("--nstar-step", type=float, default=0.5,
                   help="N_star step size")
    p.add_argument("--k-dip-target", type=float, default=1.8e-4,
                   help="Target k_dip for auto-N_star computation")
    p.add_argument("--eps-min-threshold", type=float, default=1e-2,
                   help="Skip if epsH min stays above this (SR-only)")
    # Pipeline
    p.add_argument("--num-k", type=int, default=60)
    p.add_argument("--k-min", type=float, default=5e-6)
    p.add_argument("--k-max", type=float, default=5.0)
    p.add_argument("--workers", type=int, default=8)
    # Logging
    p.add_argument("--log", type=str, default=None)
    args = p.parse_args()

    if args.log is None:
        ds = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log = os.path.join(
            ROOT_DIR, f"outputs/simulations/logs/overnight_scan_{ds}.jsonl")

    main(args)
