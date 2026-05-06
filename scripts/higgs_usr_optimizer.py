"""Higgs USR Grid-Scan Optimizer: find (phi0, y0) that puts the P_S(k) dip
at k ≈ 1-5e-4 Mpc^-1 and minimizes Planck low-ell chi^2.

Unlike usr_chi2_optimizer.py (differential_evolution), this script does a
deterministic grid scan because the Higgs USR dip is subtle (epsH ~ 2e-4)
and the valid parameter range is narrow and well-characterized.
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

from scripts.pspectrum_pipeline import run_pspectrum_pipeline, build_weighted_kgrid
from scripts.sachs_wolfe import compute_cl_sw, compute_cl_sw_powerlaw
from scripts.planck_data import get_planck_data, C_ell_to_d_ell
from scripts.constants import As, k_pivot_phys
from models import HiggsModel

# ── Helpers ──────────────────────────────────────────────────────────────────

def find_k_dip(k_phys, P_S):
    """Find the k value at the USR dip (local minimum for k < 0.01).

    Returns -1.0 if no clear dip is found.
    """
    mask = k_phys < 0.01
    if not np.any(mask):
        return -1.0
    k_sub = k_phys[mask]
    ps_sub = P_S[mask]
    i_dip = int(np.argmin(ps_sub))
    if i_dip == 0 or i_dip == len(ps_sub) - 1:
        return -1.0
    return float(k_sub[i_dip])


def compute_chi2(pipeline_result, ell_max=29):
    """Diagonal chi^2 vs Planck 2018 low-ell TT (Sachs-Wolfe approximation)."""
    data = {"k_phys": pipeline_result["k_phys"], "P_S": pipeline_result["P_S"]}
    ells_model, C_ell = compute_cl_sw(data, ell_max=ell_max)
    D_model = C_ell_to_d_ell(ells_model, C_ell)
    ells_pl, D_pl, D_err = get_planck_data()
    mask = ells_pl <= ell_max
    D_interp = np.interp(ells_pl[mask], ells_model, D_model)
    return float(np.sum(((D_interp - D_pl[mask]) / D_err[mask]) ** 2))


def compute_dip_suppression(k_phys, P_S):
    """Percentage suppression at the dip relative to the pivot amplitude."""
    k_dip = find_k_dip(k_phys, P_S)
    if k_dip <= 0:
        return 0.0, k_dip
    ps_dip = np.interp(k_dip, k_phys, P_S)
    ps_pivot = np.interp(k_pivot_phys, k_phys, P_S)
    suppression = float((1.0 - ps_dip / ps_pivot) * 100)
    return suppression, k_dip


def _write_log(log_file, entry):
    """Write a single evaluation result to the JSONL log."""
    log_file.write(json.dumps(entry) + "\n")
    log_file.flush()


def run_grid_scan(args):
    """Run a grid scan over (phi0, y0) with automatic N_star alignment.

    For each (phi0, y0) pair:
      1. Run a quick background-only solve to measure N_total and N_dip
      2. Compute N_star = N_total - N_dip - delta_N
         where delta_N = ln(k_pivot / k_dip_target) approximately equal to 4.8
      3. Run full pspectrum pipeline with computed N_star
      4. Compute chi^2, dip suppression, log to JSONL
    """
    import inf_dyn_background as bg_solver
    from scripts.pspectrum_pipeline import find_end_of_inflation

    phi0_vals = np.linspace(args.phi0_range[0], args.phi0_range[1],
                            args.n_phi0)
    y0_vals = np.linspace(args.y0_range[0], args.y0_range[1], args.n_y0)

    total = len(phi0_vals) * len(y0_vals)
    print(f"\n  Grid: {args.n_phi0} phi0 x {args.n_y0} y0 = {total} points",
          flush=True)
    print(f"  k_dip target: [{args.k_dip_range[0]:.1e}, "
          f"{args.k_dip_range[1]:.1e}] Mpc^-1", flush=True)

    log_path = args.log
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "a")

    results = []
    done = 0
    t0 = time.time()

    k_dip_center = np.sqrt(args.k_dip_range[0] * args.k_dip_range[1])
    delta_N = np.log(k_pivot_phys / k_dip_center)

    for phi0 in phi0_vals:
        for y0 in y0_vals:
            done += 1
            entry = {
                "eval": done, "phi0": float(phi0), "y0": float(y0),
                "timestamp": datetime.now().isoformat(),
            }

            # Quick background check
            model_check = HiggsModel(lam=args.lam, xi=args.xi)
            T_bg = np.linspace(0, model_check.T_max, model_check.bg_steps)
            try:
                sol = bg_solver.run_background_simulation(model_check, T_bg)
                d = bg_solver.get_derived_quantities(sol, model_check)
            except Exception:
                entry.update({"status": "bg_fail", "error": "background ODE failed"})
                _write_log(log_file, entry)
                results.append(entry)
                continue

            end_idx = find_end_of_inflation(d["epsH"])
            if end_idx == -1:
                end_idx = len(d["epsH"]) - 1
            N_total = float(d["N"][end_idx])

            # Find where epsH is minimum during inflation (the USR dip)
            eps_inf = d["epsH"][(d["N"] >= 0) & (d["N"] < N_total)]
            if len(eps_inf) < 10:
                entry.update({"status": "no_usl", "N_total": N_total})
                _write_log(log_file, entry)
                results.append(entry)
                continue
            dip_N = float(d["N"][np.argmin(eps_inf[10:]) + 10])
            N_after_dip = N_total - dip_N
            N_star = N_after_dip - delta_N

            if N_star <= 0 or N_star > N_total:
                entry.update({
                    "status": "bad_nstar",
                    "N_total": N_total, "dip_N": dip_N, "N_star": N_star,
                })
                _write_log(log_file, entry)
                results.append(entry)
                continue

            # Full pipeline
            model = HiggsModel(lam=args.lam, xi=args.xi)
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
            except Exception as exc:
                entry.update({"status": "pipe_fail", "N_star": N_star,
                              "N_total": N_total, "error": str(exc)})
                _write_log(log_file, entry)
                results.append(entry)
                continue

            if result["status"] != "success":
                entry.update({"status": "pipe_error", "N_star": N_star,
                              "N_total": N_total,
                              "msg": result.get("message", "")})
                _write_log(log_file, entry)
                results.append(entry)
                continue

            # Analysis
            try:
                chi2 = compute_chi2(result, ell_max=args.ell_max)
                k_dip = find_k_dip(result["k_phys"], result["P_S"])
                suppression, _ = compute_dip_suppression(
                    result["k_phys"], result["P_S"])
                ns_local = float(np.interp(
                    k_pivot_phys, result["k_phys"], result["P_S"]))
            except Exception as exc:
                entry.update({"status": "analysis_fail", "N_star": N_star,
                              "N_total": N_total, "error": str(exc)})
                _write_log(log_file, entry)
                results.append(entry)
                continue

            in_range = (args.k_dip_range[0] <= k_dip <= args.k_dip_range[1]
                        and k_dip > 0)
            entry.update({
                "status": "ok" if in_range else "dip_off_target",
                "N_star": round(N_star, 4),
                "N_total": round(N_total, 1),
                "dip_N": round(dip_N, 1),
                "chi2": round(chi2, 2),
                "k_dip": k_dip,
                "suppression_pct": round(suppression, 1),
                "P_S_pivot_raw": round(ns_local, 4e-9),
            })
            _write_log(log_file, entry)
            results.append(entry)

            elapsed = time.time() - t0
            eta = elapsed / done * (total - done) if done > 0 else 0
            print(f"\r  [{done:4d}/{total}] phi0={phi0:.2f} y0={y0:+.3f} "
                  f"chi2={chi2:.1f} N*={N_star:.1f} k_dip={k_dip:.2e} "
                  f"supp={suppression:+.1f}%  "
                  f"ETA {eta/60:.0f}m", end="", flush=True)

    log_file.close()
    elapsed = time.time() - t0
    print(f"\n\n  Grid scan complete in {elapsed:.0f}s ({elapsed/3600:.1f}h)",
          flush=True)
    return results
