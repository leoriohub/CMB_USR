"""Higgs USR optimizer: grid scan, differential evolution, or 3D scan.

Usage:
  python scripts/higgs_optimizer.py --strategy grid      # grid scan over (phi0, y0)
  python scripts/higgs_optimizer.py --strategy de        # differential evolution over (phi0, y0, N_star)
  python scripts/higgs_optimizer.py --strategy scan      # 3D scan with resume
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

from scripts.pspectrum_pipeline import (
    run_pspectrum_pipeline, find_end_of_inflation, build_weighted_kgrid,
)
from scripts.sachs_wolfe import compute_cl_sw, compute_cl_sw_powerlaw
from scripts.planck_data import get_planck_data, get_planck_data_asymmetric, C_ell_to_d_ell
from scripts.constants import As, k_pivot_phys, ROOT_DIR
from scripts.optimizer_utils import find_k_dip, compute_chi2, _write_log
from scripts.plotting import plot_scan_heatmap, plot_convergence, plot_best_dashboard
from models import HiggsModel
import inf_dyn_background as bg_solver


# ── Grid scan strategy (from higgs_usr_optimizer.py) ─────────────────────


def compute_dip_suppression(k_phys, P_S):
    """Percentage suppression at the dip relative to the pivot amplitude."""
    k_dip = find_k_dip(k_phys, P_S)
    if k_dip <= 0:
        return 0.0, k_dip
    ps_dip = np.interp(k_dip, k_phys, P_S)
    ps_pivot = np.interp(k_pivot_phys, k_phys, P_S)
    suppression = float((1.0 - ps_dip / ps_pivot) * 100)
    return suppression, k_dip


def run_grid_scan(args):
    """Run a grid scan over (phi0, y0) with automatic N_star alignment."""
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
            model_check.phi0 = phi0
            model_check.y0 = y0
            T_bg = np.linspace(0, model_check.T_max, model_check.bg_steps)
            try:
                sol = bg_solver.run_background_simulation(model_check, T_bg)
                d = bg_solver.get_derived_quantities(sol, model_check)
            except Exception:
                entry.update({"status": "bg_fail", "error": "background ODE failed"})
                _write_log(log_file, entry)
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done) if done > 0 else 0
                print(f"\r  [{done:4d}/{total}] phi0={phi0:.2f} y0={y0:+.3f} "
                      f"SKIP ({entry['status']}) ETA {eta/60:.0f}m", end="", flush=True)
                results.append(entry)
                continue

            end_idx = find_end_of_inflation(d["epsH"])
            if end_idx == -1:
                end_idx = len(d["epsH"]) - 1
            N_total = float(d["N"][end_idx])

            eps_inf = d["epsH"][(d["N"] >= 0) & (d["N"] < N_total)]
            if len(eps_inf) < 10:
                entry.update({"status": "no_usr", "N_total": N_total})
                _write_log(log_file, entry)
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done) if done > 0 else 0
                print(f"\r  [{done:4d}/{total}] phi0={phi0:.2f} y0={y0:+.3f} "
                      f"SKIP ({entry['status']}) ETA {eta/60:.0f}m", end="", flush=True)
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
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done) if done > 0 else 0
                print(f"\r  [{done:4d}/{total}] phi0={phi0:.2f} y0={y0:+.3f} "
                      f"SKIP ({entry['status']}) ETA {eta/60:.0f}m", end="", flush=True)
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
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done) if done > 0 else 0
                print(f"\r  [{done:4d}/{total}] phi0={phi0:.2f} y0={y0:+.3f} "
                      f"SKIP ({entry['status']}) ETA {eta/60:.0f}m", end="", flush=True)
                results.append(entry)
                continue

            if result["status"] != "success":
                entry.update({"status": "pipe_error", "N_star": N_star,
                              "N_total": N_total,
                              "msg": result.get("message", "")})
                _write_log(log_file, entry)
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done) if done > 0 else 0
                print(f"\r  [{done:4d}/{total}] phi0={phi0:.2f} y0={y0:+.3f} "
                      f"SKIP ({entry['status']}) ETA {eta/60:.0f}m", end="", flush=True)
                results.append(entry)
                continue

            # Analysis
            try:
                chi2 = compute_chi2(result, ell_max=args.ell_max)
                k_dip = find_k_dip(result["k_phys"], result["P_S"])
                suppression, _ = compute_dip_suppression(
                    result["k_phys"], result["P_S"])
                ps_pivot_raw = float(np.interp(
                    k_pivot_phys, result["k_phys"], result["P_S"]))
            except Exception as exc:
                entry.update({"status": "analysis_fail", "N_star": N_star,
                              "N_total": N_total, "error": str(exc)})
                _write_log(log_file, entry)
                elapsed = time.time() - t0
                eta = elapsed / done * (total - done) if done > 0 else 0
                print(f"\r  [{done:4d}/{total}] phi0={phi0:.2f} y0={y0:+.3f} "
                      f"SKIP ({entry['status']}) ETA {eta/60:.0f}m", end="", flush=True)
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
                "P_S_pivot_raw": float(ps_pivot_raw),
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


def print_best_results(results, args):
    """Print sorted table of best configurations and save a summary JSON."""
    ok = [r for r in results if r.get("status") == "ok"]
    if not ok:
        print("  No configurations with dip in target range.", flush=True)
        return None, None

    ok.sort(key=lambda r: r.get("chi2", 1e9))

    print(f"\n{'='*70}", flush=True)
    print(f"  Top configurations (k_dip in [{args.k_dip_range[0]:.1e}, "
          f"{args.k_dip_range[1]:.1e}]):", flush=True)
    print(f"  {'#':>3}  {'phi0':>6}  {'y0':>8}  {'N*':>7}  "
          f"{'chi2':>7}  {'k_dip':>10}  {'supp%':>7}", flush=True)
    print(f"  {'-'*57}", flush=True)
    for i, r in enumerate(ok[:20]):
        print(f"  {i+1:3d}  {r['phi0']:6.2f}  {r['y0']:+8.4f}  "
              f"{r['N_star']:7.1f}  {r['chi2']:7.1f}  "
              f"{r['k_dip']:10.2e}  {r['suppression_pct']:+6.1f}%",
              flush=True)

    sim_dir = os.path.join(ROOT_DIR, "outputs/simulations/scans")
    os.makedirs(sim_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary_path = os.path.join(sim_dir, f"higgs_scan_{date_str}.json")
    with open(summary_path, "w") as f:
        json.dump(ok, f, indent=2)
    print(f"\n  Full results saved -> {summary_path}", flush=True)
    return ok, date_str


# ── DE strategy (from usr_chi2_optimizer.py) ─────────────────────────────


def extract_ns_ms(k_phys, P_S, pivot_k):
    """Local spectral index at pivot via finite differences.

    ns(k_pivot) = 1 + d ln(P_S) / d ln(k) evaluated around pivot.
    Falls back to 0.975 if pivot is at the edge of the k-grid.
    """
    idx = int(np.argmin(np.abs(k_phys - pivot_k)))
    if idx == 0 or idx == len(k_phys) - 1:
        return 0.975
    dlnP = np.log(P_S[idx + 1]) - np.log(P_S[idx - 1])
    dlnk = np.log(k_phys[idx + 1]) - np.log(k_phys[idx - 1])
    return float(1.0 + dlnP / dlnk)


def build_objective(model_kwargs, num_kwargs, ns_target, ns_tol,
                    k_dip_range, log_path):
    """Return a closure: objective(params) -> loss for scipy.optimize.

    Loss = chi^2 + ns_penalty + k_penalty
      - ns_penalty = ((ns_MS - ns_target) / ns_tol)^2
      - k_penalty = 0 if dip in target zone, else 50
      - Pipeline failure returns 1000
    """
    log_file = open(log_path, "a") if log_path else None
    _eval_counter = [0]

    def objective(params):
        phi0 = float(params[0])
        y0 = float(params[1])
        N_star = float(params[2])
        _eval_counter[0] += 1
        eval_num = _eval_counter[0]

        model = HiggsModel(lam=model_kwargs["lam"], xi=model_kwargs["xi"])

        try:
            result = run_pspectrum_pipeline(
                model=model, phi0=phi0, y0=y0,
                k_min=num_kwargs["k_min"],
                k_max=num_kwargs["k_max"],
                k_pivot_phys=num_kwargs["k_pivot_phys"],
                N_star=N_star,
                normalize_to_As=True,
                As=num_kwargs["As"],
                n_workers=num_kwargs.get("workers", 1),
                num_k=num_kwargs.get("num_k", 80),
                save_outputs=False,
            )
        except Exception as exc:
            _log_eval(log_file, eval_num, phi0, y0, N_star,
                      loss=1000.0, status="exception", error=str(exc))
            return 1000.0

        if result["status"] != "success":
            _log_eval(log_file, eval_num, phi0, y0, N_star,
                      loss=1000.0, status="failed",
                      error=result.get("message", "unknown"))
            return 1000.0

        try:
            chi2 = compute_chi2(result, ell_max=num_kwargs.get("ell_max", 29))
            ns_MS = extract_ns_ms(
                result["k_phys"], result["P_S"],
                result["metadata"]["k_pivot_phys"],
            )
            k_dip = find_k_dip(result["k_phys"], result["P_S"])
        except Exception as exc:
            _log_eval(log_file, eval_num, phi0, y0, N_star,
                      loss=1000.0, status="analysis_error", error=str(exc))
            return 1000.0

        ns_penalty = ((ns_MS - ns_target) / ns_tol) ** 2
        if 0 < k_dip_range[0] <= k_dip <= k_dip_range[1]:
            k_penalty = 0.0
        else:
            k_penalty = 50.0
        loss = chi2 + ns_penalty + k_penalty

        _log_eval(log_file, eval_num, phi0, y0, N_star,
                  chi2=chi2, ns_MS=ns_MS, k_dip=k_dip,
                  loss=loss, status="ok")

        if eval_num % 10 == 0:
            s = (f"\r  [{eval_num:5d}] loss={loss:.2f}  "
                 f"chi2={chi2:.2f}  ns={ns_MS:.5f}  "
                 f"k_dip={k_dip:.3e}  "
                 f"phi0={phi0:.3f}  y0={y0:.4f}  N*={N_star:.1f}")
            print(s, end="", flush=True)

        return loss

    return objective


def _log_eval(log_file, eval_num, phi0, y0, N_star, **kw):
    if log_file is None:
        return
    entry = {
        "eval": eval_num,
        "phi0": round(phi0, 6),
        "y0": round(y0, 6),
        "N_star": round(N_star, 4),
        "timestamp": datetime.now().isoformat(),
    }
    entry.update(kw)
    log_file.write(json.dumps(entry) + "\n")
    log_file.flush()


def run_optimizer(args):
    """Run scipy.optimize.differential_evolution."""
    from scipy.optimize import differential_evolution

    model_kwargs = {"xi": args.xi, "lam": args.lam}
    num_kwargs = {
        "k_min": args.k_min,
        "k_max": args.k_max,
        "k_pivot_phys": k_pivot_phys,
        "As": As,
        "ell_max": args.ell_max,
        "workers": args.workers,
        "num_k": args.num_k,
    }

    log_path = os.path.abspath(args.log) if args.log else None
    if log_path:
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

    objective = build_objective(
        model_kwargs, num_kwargs,
        ns_target=args.ns_target,
        ns_tol=args.ns_tol,
        k_dip_range=args.k_dip_range,
        log_path=log_path,
    )

    bounds = [
        (args.phi0_range[0], args.phi0_range[1]),
        (args.y0_range[0], args.y0_range[1]),
        (args.nstar_range[0], args.nstar_range[1]),
    ]

    print(f"\n  Starting DE: popsize={args.popsize}, maxiter={args.maxiter}, "
          f"bounds={bounds}", flush=True)
    print(f"  Logging to: {log_path}", flush=True)
    t0 = time.time()

    result = differential_evolution(
        objective,
        bounds,
        maxiter=args.maxiter,
        popsize=args.popsize,
        seed=args.seed,
        updating="deferred",
        workers=1,
        polish=False,
        tol=0.001,
    )

    elapsed = time.time() - t0
    print(flush=True)
    print(f"\n  Optimization complete in {elapsed:.0f}s", flush=True)
    print(f"  Best: phi0={result.x[0]:.4f}, y0={result.x[1]:.4f}, "
          f"N_star={result.x[2]:.2f}, loss={result.fun:.3f}", flush=True)

    return result


def re_run_best(best_params, args):
    """Full pipeline at best config with diagnostics and plots."""
    phi0, y0, N_star = best_params
    print(f"\n{'='*60}", flush=True)
    print(f"  Re-running at best: phi0={phi0:.4f}, y0={y0:.4f}, "
          f"N_star={N_star:.2f}", flush=True)
    print(f"{'='*60}", flush=True)

    model = HiggsModel(lam=args.lam, xi=args.xi)

    output_dir = os.path.join(ROOT_DIR, "outputs/simulations/pspectra")
    os.makedirs(output_dir, exist_ok=True)

    k_grid = build_weighted_kgrid(args.k_min, args.k_max, k_pivot_phys)

    result = run_pspectrum_pipeline(
        model=model, phi0=phi0, y0=y0,
        k_min=args.k_min, k_max=args.k_max,
        k_pivot_phys=k_pivot_phys,
        N_star=N_star,
        normalize_to_As=True, As=As,
        output_dir=output_dir, save_outputs=True,
        k_phys_grid=k_grid,
        n_workers=args.workers,
    )

    if result["status"] != "success":
        print(f"  Pipeline failed: {result.get('message', 'unknown')}", flush=True)
        return

    print(f"  P_S(k) saved -> {result.get('output_file', 'N/A')}", flush=True)

    meta = result["metadata"]
    chi2 = compute_chi2(result, ell_max=args.ell_max)
    ns_MS = extract_ns_ms(result["k_phys"], result["P_S"],
                          meta["k_pivot_phys"])
    k_dip = find_k_dip(result["k_phys"], result["P_S"])
    N_total = meta["N_total"]

    ells_lcdm, C_lcdm, _ = compute_cl_sw_powerlaw(
        k_min=args.k_min, k_max=args.k_max,
        As=As, ns=0.965, ell_max=args.ell_max,
    )
    D_lcdm = C_ell_to_d_ell(ells_lcdm, C_lcdm)

    ells_pl, D_pl, D_err_l, D_err_u = get_planck_data_asymmetric()
    pl_mask = ells_pl <= args.ell_max
    ells_pl_m = ells_pl[pl_mask]
    D_pl_m = D_pl[pl_mask]
    D_lcdm_ip = np.interp(ells_pl_m, ells_lcdm, D_lcdm)
    residuals = D_lcdm_ip - D_pl_m
    err = np.where(residuals > 0, D_err_u[pl_mask], D_err_l[pl_mask])
    chi2_lcdm = float(np.sum((residuals / err) ** 2))
    dof = int(np.sum(pl_mask))

    dip_ps = np.interp(k_dip, result["k_phys"], result["P_S"]) if k_dip > 0 else As
    dip_sup = (dip_ps / As - 1) * 100 if k_dip > 0 else 0.0

    print(f"\n  chi2={chi2:.2f} (dof={dof})  LCDM chi2={chi2_lcdm:.2f}", flush=True)
    print(f"  dchi2 = {chi2 - chi2_lcdm:+.2f}", flush=True)
    print(f"  ns_MS={ns_MS:.5f}  k_dip={k_dip:.3e}  "
          f"dip_sup={dip_sup:.1f}%  N_total={N_total:.1f}", flush=True)

    # Save summary
    sim_dir = os.path.join(ROOT_DIR, "outputs/simulations/configs")
    os.makedirs(sim_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    summary = {
        "params": {"phi0": phi0, "y0": y0, "N_star": N_star},
        "chi2": chi2, "chi2_lcdm": chi2_lcdm,
        "dchi2_vs_lcdm": chi2 - chi2_lcdm, "dof": dof,
        "ns_MS": ns_MS, "k_dip": k_dip,
        "dip_suppression_pct": dip_sup,
        "N_total": N_total,
        "output_file": result.get("output_file"),
        "timestamp": datetime.now().isoformat(),
    }
    summary_path = os.path.join(sim_dir, f"best_config_{date_str}.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary saved -> {summary_path}", flush=True)

    # Plot dashboard
    try:
        ells_usr, C_usr = compute_cl_sw(result, ell_max=args.ell_max)
        D_usr = C_ell_to_d_ell(ells_usr, C_usr)
        D_err_sym = np.sqrt(D_err_l[pl_mask] * D_err_u[pl_mask])
        plot_best_dashboard(
            ells_usr, D_usr, ells_lcdm, D_lcdm,
            ells_pl_m, D_pl_m, D_err_sym,
            phi0, y0, N_star, k_dip, chi2, chi2_lcdm,
            filename=f"best_config_{date_str}", subdir="optimizer",
        )
        print(f"  Plot saved", flush=True)
    except Exception as exc:
        print(f"  Warning: plot failed: {exc}", flush=True)

    return summary


# ── Scan strategy (from overnight_scan.py) ───────────────────────────────


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
        chi2_val = compute_chi2(result, ell_max=args.ell_max)
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


def scan_nstar(model, phi0, y0, N_total, dip_N, nstar_auto, args,
               log_file, done, total, t0):
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


def run_nstar_scan(args):
    """3D scan over (phi0, y0, N_star) with resume capability."""
    phi0_vals = np.linspace(args.phi0_range[0], args.phi0_range[1], args.n_phi0)
    y0_vals = np.linspace(args.y0_range[0], args.y0_range[1], args.n_y0)

    n_nstar_per = int(2 * args.nstar_window / args.nstar_step) + 1
    total_est = len(phi0_vals) * len(y0_vals) * n_nstar_per
    viable_est = int(len(phi0_vals) * len(y0_vals) * args.viability_guess)
    total_realistic = viable_est * n_nstar_per

    print(f"{'='*65}", flush=True)
    print(f"  3D Scan — Higgs USR", flush=True)
    print(f"  Parameters: (phi0, y0, N_star)", flush=True)
    print(f"  phi0: [{args.phi0_range[0]:.2f}, {args.phi0_range[1]:.2f}] "
          f"x {args.n_phi0}  "
          f"(step {(args.phi0_range[1]-args.phi0_range[0])/(args.n_phi0-1):.3f})",
          flush=True)
    print(f"  y0:   [{args.y0_range[0]:.3f}, {args.y0_range[1]:.3f}] "
          f"x {args.n_y0}  "
          f"(step {(args.y0_range[1]-args.y0_range[0])/(args.n_y0-1):.3f})",
          flush=True)
    print(f"  N_star: auto_N +/- {args.nstar_window} step {args.nstar_step}  "
          f"-> {n_nstar_per} values/pair", flush=True)
    print(f"  Grid: {len(phi0_vals)} x {len(y0_vals)} = "
          f"{len(phi0_vals)*len(y0_vals)} pairs", flush=True)
    print(f"  Viability guess: ~{args.viability_guess*100:.0f}% "
          f"-> ~{total_realistic} evals", flush=True)
    print(f"  k-modes: {args.num_k}, workers: {args.workers}", flush=True)
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
            model_check = HiggsModel(lam=args.lam, xi=args.xi)
            model_check.phi0 = phi0
            model_check.y0 = y0
            T_bg = np.linspace(0, model_check.T_max, model_check.bg_steps)
            try:
                sol = bg_solver.run_background_simulation(model_check, T_bg)
                d = bg_solver.get_derived_quantities(sol, model_check)
            except Exception:
                done[0] += n_nstar_per
                print(f"\r  [{done[0]:4d}/{total_est}] phi0={phi0:.2f} "
                      f"y0={y0:+.3f} SKIP (bg_fail)", end="", flush=True)
                continue

            end_idx = find_end_of_inflation(d["epsH"])
            if end_idx == -1:
                end_idx = len(d["epsH"]) - 1
            N_total = float(d["N"][end_idx])

            eps_inf = d["epsH"][(d["N"] >= 0) & (d["N"] < N_total)]
            if len(eps_inf) < 10:
                done[0] += n_nstar_per
                print(f"\r  [{done[0]:4d}/{total_est}] phi0={phi0:.2f} "
                      f"y0={y0:+.3f} SKIP (no_usr)", end="", flush=True)
                continue
            eps_min = float(np.min(eps_inf))
            if eps_min > args.eps_min_threshold:
                done[0] += n_nstar_per
                print(f"\r  [{done[0]:4d}/{total_est}] phi0={phi0:.2f} "
                      f"y0={y0:+.3f} SKIP (sr_only)", end="", flush=True)
                continue
            dip_N = float(d["N"][np.argmin(eps_inf[10:]) + 10])
            if N_total - dip_N < 5:
                done[0] += n_nstar_per
                print(f"\r  [{done[0]:4d}/{total_est}] phi0={phi0:.2f} "
                      f"y0={y0:+.3f} SKIP (dip_at_end)", end="", flush=True)
                continue
            N_after_dip = N_total - dip_N
            delta_N = np.log(k_pivot_phys / k_dip_center)
            nstar_auto = N_after_dip - delta_N

            if nstar_auto <= 0 or nstar_auto > N_total:
                done[0] += n_nstar_per
                print(f"\r  [{done[0]:4d}/{total_est}] phi0={phi0:.2f} "
                      f"y0={y0:+.3f} SKIP (bad_nstar)", end="", flush=True)
                continue

            model = HiggsModel(lam=args.lam, xi=args.xi)
            res = scan_nstar(model, phi0, y0, N_total, dip_N,
                             nstar_auto, args,
                             log_file, done, total_est, t0)
            all_results.extend(res)

    log_file.close()
    elapsed = time.time() - t0
    print(f"\n\n  Scan complete in {elapsed:.0f}s ({elapsed/3600:.1f}h)",
          flush=True)

    # Print best results
    ok = [r for r in all_results if r.get("status") == "ok" and "chi2" in r]
    ok.sort(key=lambda r: r["chi2"])
    print(f"\n  Best 20 configurations:", flush=True)
    print(f"    {'phi0':>6}  {'y0':>7}  {'N*':>6}  {'chi2':>6}  "
          f"{'k_dip':>10}  {'supp%':>7}", flush=True)
    print(f"    {'-'*48}", flush=True)
    for r in ok[:20]:
        print(f"    {r['phi0']:6.2f}  {r['y0']:7.3f}  {r['N_star']:6.1f}  "
              f"{r['chi2']:6.2f}  {r['k_dip']:10.2e}  "
              f"{r['suppression_pct']:7.1f}", flush=True)

    summary_path = os.path.join(
        ROOT_DIR, "outputs", "simulations", "scans",
        f"nstar_scan_best_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    os.makedirs(os.path.dirname(summary_path), exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(ok[:50], f, indent=2)
    print(f"\n  Best 50 saved -> {summary_path}", flush=True)


# ── Unified CLI ──────────────────────────────────────────────────────────


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Higgs USR optimizer")
    p.add_argument("--strategy", default="grid", choices=["grid", "de", "scan"],
                   help="Optimization strategy")

    # Model
    p.add_argument("--xi", type=float, default=15000.0)
    p.add_argument("--lam", type=float, default=0.13)

    # Grid strategy params (also used for scan)
    p.add_argument("--phi0-range", nargs=2, type=float, default=[5.20, 5.80],
                   metavar=("MIN", "MAX"))
    p.add_argument("--y0-range", nargs=2, type=float, default=[-0.20, -0.03],
                   metavar=("MIN", "MAX"))
    p.add_argument("--n-phi0", type=int, default=13)
    p.add_argument("--n-y0", type=int, default=16)

    # DE strategy params
    p.add_argument("--nstar-range", nargs=2, type=float, default=[45.0, 62.0],
                   metavar=("MIN", "MAX"))
    p.add_argument("--ns-target", type=float, default=0.975)
    p.add_argument("--ns-tol", type=float, default=0.01)
    p.add_argument("--maxiter", type=int, default=200)
    p.add_argument("--popsize", type=int, default=15)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save-best", action="store_true")
    p.add_argument("--re-run-best", action="store_true")

    # Scan strategy params
    p.add_argument("--nstar-window", type=float, default=5.0)
    p.add_argument("--nstar-step", type=float, default=0.5)
    p.add_argument("--k-dip-target", type=float, default=1.8e-4)
    p.add_argument("--eps-min-threshold", type=float, default=1e-2)
    p.add_argument("--viability-guess", type=float, default=0.30)

    # Targets
    p.add_argument("--k-dip-range", nargs=2, type=float,
                   default=[1e-4, 5e-4], metavar=("MIN", "MAX"))
    p.add_argument("--ell-max", type=int, default=29)

    # Pipeline
    p.add_argument("--k-min", type=float, default=1e-5)
    p.add_argument("--k-max", type=float, default=1.0)
    p.add_argument("--num-k", type=int, default=80)
    p.add_argument("--workers", type=int, default=4)

    # Output
    p.add_argument("--log", type=str, default=None)
    p.add_argument("--plot-best", action="store_true",
                   help="re-run and plot the best configuration")

    return p.parse_args(argv)


# ── Main dispatch ────────────────────────────────────────────────────────


def main():
    args = parse_args()

    if args.log is None:
        ds = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log = os.path.join(
            ROOT_DIR, f"outputs/simulations/logs/{args.strategy}_scan_{ds}.jsonl")

    print(f"{'='*60}", flush=True)
    print(f"  Higgs USR Optimizer ({args.strategy})", flush=True)
    print(f"{'='*60}", flush=True)

    if args.strategy == "grid":
        results = run_grid_scan(args)
        ok, date_str = print_best_results(results, args)
        if ok is not None:
            phi0_vals = sorted(set(r["phi0"] for r in ok))
            y0_vals = sorted(set(r["y0"] for r in ok))
            chi2_map = np.full((len(y0_vals), len(phi0_vals)), np.nan)
            supp_map = np.full((len(y0_vals), len(phi0_vals)), np.nan)
            for r in ok:
                ip = phi0_vals.index(r["phi0"])
                iy = y0_vals.index(r["y0"])
                chi2_map[iy, ip] = r["chi2"]
                supp_map[iy, ip] = r["suppression_pct"]
            plot_scan_heatmap(phi0_vals, y0_vals, chi2_map, supp_map,
                              filename=f"higgs_scan_{date_str}",
                              subdir="optimizer")
        if ok is not None and args.plot_best:
            best = ok[0]
            re_run_best([best["phi0"], best["y0"], best["N_star"]], args)

    elif args.strategy == "de":
        opt_result = run_optimizer(args)
        if args.save_best or args.re_run_best:
            re_run_best(opt_result.x, args)
        if args.log and os.path.exists(args.log):
            records = []
            with open(args.log) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        records.append(json.loads(line))
            plot_convergence(
                records,
                filename=f"de_convergence_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                subdir="optimizer",
            )

    elif args.strategy == "scan":
        run_nstar_scan(args)

    else:
        print(f"  Unknown strategy: {args.strategy}", flush=True)
        sys.exit(1)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
