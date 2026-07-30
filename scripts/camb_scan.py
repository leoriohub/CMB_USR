"""
CAMB-based Higgs USR scan with automatic two-phase execution.

Usage:
  python scripts/camb_scan.py --test          # Quick test (3x3 grid)
  python scripts/camb_scan.py --auto          # Full automatic: broad → fine
  python scripts/camb_scan.py --phase broad   # Phase 1 only
  python scripts/camb_scan.py --phase fine    # Phase 2 only (needs Phase 1 results)
  python scripts/camb_scan.py --phase random --n-random 2000  # Random config scan
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

from functools import lru_cache
import numpy as np
from concurrent.futures import ProcessPoolExecutor

from pspectrum_pipeline import (
    run_pspectrum_pipeline, find_end_of_inflation,
    build_weighted_kgrid,
)
from scripts.camb_wrapper import compute_cl_camb_powerlaw, compute_cl_full_camb
from scripts.planck_data import get_planck_data_asymmetric, C_ell_to_d_ell
from scripts.optimizer_utils import find_k_dip, _write_log
from scripts.constants import As, k_pivot_phys, T_cmb
from scripts.plotting import get_path
from scripts.chi2_analysis import chi2_model_lcdm
from models import HiggsModel
import inf_dyn_background as bg_solver
K_DIP_MIN = 5e-5
K_DIP_MAX = 2.1e-3
CHI2_LCDM = 20.47
CHI2_LCDM_FULL = 2573.04
D2_LCDM = 1028.7

def _generate_grids(args):
    k_phys = np.logspace(np.log10(args.k_min), np.log10(args.k_max), args.num_k)
    ells = np.arange(args.ell_max + 1)
    return k_phys, ells

def compute_camb_curves(ps_data, ell_max):
    """Full CAMB: returns d2, ells, D_ell, C_TT, C_TE, C_EE."""
    ells, C_TT, C_TE, C_EE = compute_cl_full_camb(ps_data, ell_max=ell_max)
    D = C_ell_to_d_ell(ells, C_TT)
    return float(D[0]), ells, D, C_TT, C_TE, C_EE

def chi2_vs_planck(ells_model, D_model, ell_max_chi2=29):
    """Asymmetric diagonal chi2 vs Planck low-ell TT."""
    chi2, _ = chi2_model_lcdm(
        D_model, ells_model,
        ell_max=ell_max_chi2,
    )
    return chi2

@lru_cache(maxsize=4)
def lcdm_baseline(ell_max):
    """Cached LCDM D_ell at given ell_max."""
    ells, C, _, _ = compute_cl_camb_powerlaw(ell_max=ell_max)
    return ells, C

def passes_criteria(chi2, d2, n_star, k_dip, mode="chi2", max_chi2=None):
    if k_dip <= 0:
        return False
    if max_chi2 is None:
        max_chi2 = CHI2_LCDM if mode == "chi2" else 35.0
    return (chi2 <= max_chi2 and d2 < D2_LCDM
            and 50 <= n_star <= 65
            and K_DIP_MIN <= k_dip <= K_DIP_MAX)

def load_completed(log_path):
    completed = set()
    if not os.path.exists(log_path):
        return completed
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
                if rec.get("_type") == "header":
                    continue
                k = (round(rec.get("phi0", 0), 4),
                     round(rec.get("y0", 0), 4),
                     round(rec.get("N_star", 0), 2))
                completed.add(k)
            except (json.JSONDecodeError, KeyError):
                pass
    return completed

def evaluate_config(phi0, y0, N_star, args, k_phys_grid=None,
                    executor=None, model=None):
    """Run pipeline + CAMB. Pass k_phys_grid + executor + model for reuse."""
    if model is None:
        model = HiggsModel(lam=args.lam, xi=args.xi)
    try:
        result = run_pspectrum_pipeline(
            model=model, phi0=phi0, y0=y0,
            k_min=args.k_min, k_max=args.k_max,
            k_pivot_phys=k_pivot_phys,
            N_star=N_star,
            normalize_to_As=True, As=As,
            num_k=args.num_k if k_phys_grid is None else 0,
            k_phys_grid=k_phys_grid,
            n_workers=args.workers,
            executor=executor,
            save_outputs=False,
        )
    except Exception as e:
        return {"status": "pipe_exception", "error": str(e)}

    if result["status"] != "success":
        return {"status": "pipe_error", "error": result.get("message", "")}

    k_phys = result["k_phys"]
    P_S = result["P_S"]

    try:
        k_dip = find_k_dip(k_phys, P_S)
    except Exception:
        k_dip = -1.0

    ell_max_camb = getattr(args, "ell_max", 2500)

    try:
        ps_data = {"k_phys": k_phys, "P_S": P_S}
        d2_val, ells_c, D_c, C_TT, C_TE, C_EE = compute_camb_curves(
            ps_data, ell_max_camb)
        ells_l, C_l = lcdm_baseline(ell_max_camb)
        D_l = C_ell_to_d_ell(ells_l, C_l)

        low_ell_chi2 = chi2_vs_planck(ells_c, D_c, 29)
        low_ell_lcdm = chi2_vs_planck(ells_l, D_l, 29)

        if getattr(args, 'full_chi2', False):
            from scripts.chi2_analysis import chi2_binned, chi2_unbinned
            chi2_b, chi2_l_b, np_b = chi2_binned(D_c, ells_c, D_l, ells_l)
            chi2_u, chi2_l_u, np_u = chi2_unbinned(D_c, ells_c, D_l, ells_l)
            chi2_m = round(chi2_u, 2)
            chi2_l = round(chi2_l_u, 2)
            chi2_binned_model = round(chi2_b, 2)
            chi2_binned_lcdm = round(chi2_l_b, 2)
        else:
            chi2_m = round(low_ell_chi2, 2)
            chi2_l = round(low_ell_lcdm, 2)
            chi2_binned_model = None
            chi2_binned_lcdm = None

        dchi2 = chi2_m - chi2_l
    except Exception as e:
        return {"status": "camb_fail", "error": str(e), "k_dip": k_dip}

    supp = 0.0
    if k_dip > 0:
        ps_dip = float(np.interp(k_dip, k_phys, P_S))
        ps_pivot = float(np.interp(k_pivot_phys, k_phys, P_S))
        if ps_pivot > 0:
            supp = (1.0 - ps_dip / ps_pivot) * 100

    meta = result.get("metadata", {})
    entry = {
        "status": "ok",
        "chi2": round(chi2_m, 2),
        "chi2_lcdm": round(chi2_l, 2),
        "dchi2": round(dchi2, 2),
        "d2": round(d2_val, 1),
        "k_dip": k_dip,
        "suppression_pct": round(supp, 1),
        "N_total": round(meta.get("N_total", 0), 1),
        "N_star": round(N_star, 4),
        "d2_lcdm": round(D2_LCDM, 1),
        "D_ell": D_c.tolist(),
        "C_ell_TT": C_TT.tolist(),
        "C_ell_TE": C_TE.tolist(),
        "C_ell_EE": C_EE.tolist(),
        "P_S": P_S.tolist(),
    }
    if getattr(args, 'full_chi2', False):
        entry["chi2_low"] = round(low_ell_chi2, 2)
        entry["chi2_binned_model"] = chi2_binned_model
        entry["chi2_binned_lcdm"] = chi2_binned_lcdm
    return entry

def run_phase1(args, completed):
    phi0_vals = np.linspace(args.phi0_range[0], args.phi0_range[1],
                            args.n_phi0)
    y0_vals = np.linspace(args.y0_range[0], args.y0_range[1], args.n_y0)

    total = len(phi0_vals) * len(y0_vals)
    print(f"\n{'='*60}", flush=True)
    print(f"  PHASE 1: Broad scan")
    print(f"  phi0: [{args.phi0_range[0]:.1f}, {args.phi0_range[1]:.1f}] "
          f"x {args.n_phi0} = {phi0_vals}")
    print(f"  y0:   [{args.y0_range[0]:.2f}, {args.y0_range[1]:.2f}] "
          f"x {args.n_y0} = {y0_vals}")
    print(f"  Total pairs: {total}, 3 N_star scans per viable pair")
    print(f"  ~{int(total*0.3)} viable pairs x 3 = ~{int(total*0.3*3)} evals")
    print(f"  Est runtime: ~{int(total*0.3*3*35/60)} min")
    print(f"{'='*60}", flush=True)

    log_path = get_path("logs", f"camb_phase1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")

    if getattr(args, 'quick', False):
        k_phys = build_weighted_kgrid(
            args.k_min, args.k_max, k_pivot_phys,
            dense_zone=(1e-4, 1e-2), n_dense=20, n_outer=8,
        )
        ells_grid = np.arange(args.ell_max + 1)
    else:
        k_phys, ells_grid = _generate_grids(args)

    log_file = open(log_path, "a")
    try:
        _write_log(log_file, {
            "_type": "header",
            "k_phys": k_phys.tolist(),
            "ells": ells_grid.tolist(),
            "xi": args.xi,
            "lam": args.lam,
            "k_min": args.k_min,
            "k_max": args.k_max,
            "num_k": args.num_k,
            "ell_max": args.ell_max,
            "k_pivot_phys": k_pivot_phys,
            "As": As,
        })

        t0 = time.time()
        results = []
        done = 0
        N_star_vals = np.linspace(args.nstar_range[0], args.nstar_range[1], 3).tolist()

        for phi0 in phi0_vals:
            for y0 in y0_vals:
                done_phase = len(results) + sum(1 for r in results
                                                if r.get("status") != "bg_fail")

                # Quick background check + N_star auto-alignment
                model_check = HiggsModel(lam=args.lam, xi=args.xi)
                model_check.x0 = phi0
                model_check.y0 = y0
                T_bg = np.linspace(0, model_check.T_max, model_check.bg_steps)

                try:
                    sol = bg_solver.run_background_simulation(model_check, T_bg)
                    d = bg_solver.get_derived_quantities(sol, model_check)
                except Exception:
                    done += 1
                    eta = (time.time() - t0) / done * (total - done) if done else 0
                    print(f"\r  [{done:3d}/{total}] phi0={phi0:.2f} y0={y0:+.3f} "
                          f"SKIP (bg_fail) ETA {eta/60:.0f}m", end="", flush=True)
                    results.append({"phi0": phi0, "y0": y0, "status": "bg_fail"})
                    continue

                try:
                    end_idx = find_end_of_inflation(d["epsH"])
                    if end_idx < 1:
                        done += 1
                        eta = (time.time() - t0) / done * (total - done) if done else 0
                        print(f"\r  [{done:3d}/{total}] phi0={phi0:.2f} y0={y0:+.3f} "
                              f"SKIP (no end) ETA {eta/60:.0f}m", end="", flush=True)
                        results.append({"phi0": phi0, "y0": y0, "status": "no_end"})
                        continue

                    N_total = float(d["N"][end_idx])
                    max_ns = N_total - 1
                    ns_list = sorted(set([
                        round(min(N_star_vals[0], max_ns), 1),
                        round(min(N_star_vals[1], max_ns), 1),
                        round(min(N_star_vals[2], max_ns), 1),
                    ]))
                    for N_star in ns_list:
                        if (round(phi0, 4), round(y0, 4), round(N_star, 2)) in completed:
                            continue
                        _write_log(log_file, {
                            "_type": "bg_result", "phi0": round(phi0, 4),
                            "y0": round(y0, 4), "N_star": N_star,
                            "N_total": N_total, "end_idx": int(end_idx),
                        })
                        res = evaluate_config(
                            phi0, y0, N_star, args,
                            k_phys_grid=k_phys, model=model_check,
                        )
                        res["_type"] = "data"
                        results.append(res)

                        done += 1
                        eta = (time.time() - t0) / done * (total - done) if done else 0
                        chi2_str = (f"chi2={res.get('chi2', '?'):.1f}"
                                    if res.get("status") == "ok" else
                                    f"err={res.get('error', '?')[:20]}"
                                    if res.get("error") else "err")
                        d2_str = (f"d2={res.get('d2', 0):.0f}"
                                  if res.get("d2", 0) > 0 else "")
                        print(f"\r  [{done:3d}/{total}] phi0={phi0:.2f} y0={y0:+.3f} "
                              f"N*={N_star:.0f} {chi2_str} {d2_str} "
                              f"k_dip={res.get('k_dip', 0):.2e} "
                              f"ETA {eta/60:.0f}m", end="", flush=True)

                except Exception as exc:
                    done += 1
                    eta = (time.time() - t0) / done * (total - done) if done else 0
                    print(f"\r  [{done:3d}/{total}] phi0={phi0:.2f} y0={y0:+.3f} "
                          f"FAIL ({exc!s:.40s}) ETA {eta/60:.0f}m", end="", flush=True)
                    results.append({"phi0": phi0, "y0": y0, "status": "exception",
                                    "error": str(exc)})

            print()

    finally:
        log_file.close()
    elapsed = time.time() - t0
    print(f"\n  Phase 1 complete: {done} evals in {elapsed:.0f}s "
          f"({elapsed/60:.1f}m)", flush=True)

    ok = [r for r in results if r.get("status") == "ok"]
    print(f"  Passing configs: {len(ok)}", flush=True)
    return log_path

def find_promising_regions(log_path, args):
    """Read Phase 1 results, find regions passing all criteria."""
    regions = []
    if not os.path.exists(log_path):
        print(f"  No Phase 1 results at {log_path}", flush=True)
        return regions

    candidates = []
    with open(log_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("_type") == "header" or rec.get("status") != "ok":
                continue
            chi2 = rec.get("chi2", 999)
            d2 = rec.get("d2", 9999)
            n_star = rec.get("N_star", 0)
            k_dip = rec.get("k_dip", -1)
            if passes_criteria(chi2, d2, n_star, k_dip, mode=args.mode, max_chi2=getattr(args, "max_chi2", None)):
                candidates.append(rec)

    if not candidates:
        print("  No promising configs found in Phase 1 results", flush=True)
        return regions

    sort_by = getattr(args, "sort_by", None) or args.mode
    key_func = (lambda r: r.get("chi2", 999)) if sort_by == "chi2" else (lambda r: r.get("d2", 9999))
    candidates.sort(key=key_func)

    # Cluster nearby configs
    merged = []
    for c in candidates:
        added = False
        for cluster in merged:
            dc = abs(cluster["phi0"] - c["phi0"])
            dy = abs(cluster["y0"] - c["y0"])
            if dc <= args.phi0_step * 1.5 and dy <= args.y0_step * 1.5:
                if (sort_by == "d2" and c["d2"] < cluster["d2"]) or (sort_by != "d2" and c["chi2"] < cluster["chi2"]):
                    cluster.update(c)
                added = True
                break
        if not added:
            merged.append(dict(c))

    for m in merged[:args.max_regions]:
        regions.append((m["phi0"], m["y0"], max(m.get("N_star", 52), 50)))

    return regions

def run_phase2(args, completed, regions):
    if not regions:
        print("\n  No promising regions to scan in Phase 2", flush=True)
        return

    print(f"\n{'='*60}", flush=True)
    print(f"  PHASE 2: Fine scan")
    print(f"  Regions: {len(regions)}")
    for i, (phi0, y0, ns) in enumerate(regions):
        print(f"    {i+1}. phi0={phi0:.2f} y0={y0:.3f} N*={ns:.0f}")
    print(f"{'='*60}", flush=True)

    log_path = get_path("logs", f"camb_phase2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")

    if getattr(args, 'quick', False):
        k_phys = build_weighted_kgrid(
            args.k_min, args.k_max, k_pivot_phys,
            dense_zone=(1e-4, 1e-2), n_dense=20, n_outer=8,
        )
        ells_grid = np.arange(args.ell_max + 1)
    else:
        k_phys, ells_grid = _generate_grids(args)

    log_file = open(log_path, "a")
    try:
        _write_log(log_file, {
            "_type": "header",
            "k_phys": k_phys.tolist(),
            "ells": ells_grid.tolist(),
            "xi": args.xi,
            "lam": args.lam,
            "k_min": args.k_min,
            "k_max": args.k_max,
            "num_k": args.num_k,
            "ell_max": args.ell_max,
            "k_pivot_phys": k_pivot_phys,
            "As": As,
        })

        total_est = len(regions) * args.n_phi0_fine * args.n_y0_fine * args.n_nstar_fine
        t0 = time.time()
        done = [0]

        for reg_idx, (phi0_center, y0_center, ns_center) in enumerate(regions):
            phi0_vals = np.linspace(phi0_center - args.phi0_fine_window,
                                    phi0_center + args.phi0_fine_window,
                                    args.n_phi0_fine)
            y0_vals = np.linspace(y0_center - args.y0_fine_window,
                                  y0_center + args.y0_fine_window,
                                  args.n_y0_fine)
            ns_vals = np.linspace(max(ns_center - args.nstar_fine_window, 50),
                                  ns_center + args.nstar_fine_window,
                                  args.n_nstar_fine)

            print(f"\n  Region {reg_idx+1}: phi0~{phi0_center:.2f} "
                  f"y0~{y0_center:.3f} N*~{ns_center:.0f}", flush=True)
            print(f"    Grid: {len(phi0_vals)}x{len(y0_vals)}x{len(ns_vals)} "
                  f"= {len(phi0_vals)*len(y0_vals)*len(ns_vals)} evals", flush=True)

            for phi0 in phi0_vals:
                for y0 in y0_vals:
                    for N_star in ns_vals:
                        N_star = round(N_star, 1)
                        key = (round(phi0, 4), round(y0, 4), round(N_star, 2))
                        if key in completed:
                            done[0] += 1
                            continue

                        done[0] += 1
                        entry = {
                            "_type": "data",
                            "eval": done[0], "total": total_est,
                            "phi0": round(phi0, 4), "y0": round(y0, 4),
                            "N_star": N_star,
                            "phase": 2, "region": reg_idx + 1,
                            "timestamp": datetime.now().isoformat(),
                        }

                        res = evaluate_config(phi0, y0, N_star, args, k_phys_grid=k_phys)
                        entry.update(res)
                        entry.pop("k_phys", None)
                        entry.pop("ells", None)

                        _write_log(log_file, entry)

                        eta = (time.time() - t0) / done[0] * (total_est - done[0]) if done[0] else 0
                        chi2_str = (f"chi2={res.get('chi2', '?'):.1f}"
                                    if res.get("status") == "ok"
                                    else f"SKIP")
                        n_ok = sum(1 for r in open(log_path).readlines()
                                   if '"status": "ok"' in r) if os.path.exists(log_path) else 0
                        print(f"\r  [{done[0]:4d}/{total_est}] R{reg_idx+1} "
                              f"phi0={phi0:.2f} y0={y0:+.3f} N*={N_star:.0f} "
                              f"{chi2_str}  ok={n_ok}  ETA {eta/60:.0f}m",
                              end="", flush=True)

    finally:
        log_file.close()
    elapsed = time.time() - t0
    print(f"\n\n  Phase 2 complete: {done[0]} evals in "
          f"{elapsed:.0f}s ({elapsed/60:.1f}m)", flush=True)

def run_random_scan(args):
    """Run random config scan over phi0/y0/N_star ranges."""
    import random

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)

    log_path = get_path("logs", f"camb_random_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
    with open(log_path, "w") as f:
        f.write(json.dumps({
            "_type": "header",
            "n": args.n_random,
            "random_seed": args.random_seed,
            "phi0_range": args.phi0_range,
            "y0_range": args.y0_range,
            "nstar_range": args.nstar_range,
            "xi": args.xi,
            "lam": args.lam,
            "k_pivot_phys": k_pivot_phys,
            "As": As,
        }) + "\n")

    k_phys = build_weighted_kgrid(
        args.k_min, args.k_max, k_pivot_phys,
        dense_zone=(1e-4, 1e-2),
        n_dense=20 if args.quick else 140,
        n_outer=8 if args.quick else 70,
    )

    configs = [(round(random.uniform(args.phi0_range[0], args.phi0_range[1]), 2),
                round(-random.uniform(abs(args.y0_range[1]), abs(args.y0_range[0])), 3),
                round(random.uniform(args.nstar_range[0], args.nstar_range[1]), 1))
               for _ in range(args.n_random)]

    model = HiggsModel(lam=args.lam, xi=args.xi)
    done = 0

    print(f"\n{'='*60}", flush=True)
    print(f"  RANDOM SCAN: {args.n_random} configs", flush=True)
    print(f"  phi0: [{args.phi0_range[0]:.1f}, {args.phi0_range[1]:.1f}]", flush=True)
    print(f"  y0:   [{args.y0_range[0]:.2f}, {args.y0_range[1]:.2f}]", flush=True)
    print(f"  N*:   [{args.nstar_range[0]:.0f}, {args.nstar_range[1]:.0f}]", flush=True)
    print(f"  k-modes: {len(k_phys)}", flush=True)
    print(f"{'='*60}", flush=True)

    t0 = time.time()

    with open(log_path, "a") as log_file, \
         ProcessPoolExecutor(args.workers) as executor:
        for i, (phi0, y0, nstar) in enumerate(configs):
            res = evaluate_config(phi0, y0, nstar, args,
                                  k_phys_grid=k_phys,
                                  executor=executor, model=model)
            res["_type"] = "data"
            res["phi0"] = phi0
            res["y0"] = y0
            res["N_star"] = nstar
            log_file.write(json.dumps(res) + "\n")
            log_file.flush()

            if res.get("status") == "ok":
                done += 1

            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                eta = elapsed / (i + 1) * (args.n_random - i - 1)
                print(f"  [{i+1}/{args.n_random}] {elapsed:.0f}s "
                      f"ETA {eta:.0f}s  ok={done}", flush=True)

    elapsed = time.time() - t0
    print(f"\n  Random scan complete: {args.n_random} configs "
          f"in {elapsed:.0f}s ({elapsed/args.n_random:.2f}s/config) "
          f"ok={done}/{args.n_random}", flush=True)
    print(f"  Log: {log_path}", flush=True)

    summary_path = log_path.replace(".jsonl", "_summary.json")
    ok_records = []
    with open(log_path) as f:
        for line in f:
            r = json.loads(line.strip())
            if r.get("status") == "ok" and r.get("_type") != "header":
                ok_records.append(r)
    ok_records.sort(key=lambda r: r.get("chi2", 999))
    with open(summary_path, "w") as f:
        json.dump(ok_records[:50], f, indent=2)
    print(f"  Top-50 saved: {summary_path}", flush=True)
    return log_path

def print_summary(log_path_broad, log_path_fine=None, mode="chi2", max_chi2=None):
    print(f"\n{'='*60}", flush=True)
    print(f"  SCAN SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)

    for label, path in [("Phase 1", log_path_broad),
                        ("Phase 2", log_path_fine)]:
        if not path or not os.path.exists(path):
            continue
        ok = []
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("_type") == "header" or rec.get("status") != "ok":
                    continue
                chi2 = rec.get("chi2", 999)
                d2 = rec.get("d2", 9999)
                n_star = rec.get("N_star", 0)
                k_dip = rec.get("k_dip", -1)
                if passes_criteria(chi2, d2, n_star, k_dip, mode=mode, max_chi2=max_chi2):
                    ok.append(rec)

        if not ok:
            print(f"\n  {label}: 0 configs pass all criteria", flush=True)
            continue

        ok.sort(key=lambda r: r.get("chi2", 999))
        print(f"\n  {label}: {len(ok)} configs pass criteria", flush=True)
        print(f"  {'#':>3}  {'phi0':>6}  {'y0':>8}  {'N*':>6}  "
              f"{'chi2':>7}  {'d2':>7}  {'k_dip':>10}  {'supp%':>7}",
              flush=True)
        print(f"  {'-'*63}", flush=True)
        for i, r in enumerate(ok[:15]):
            print(f"  {i+1:3d}  {r['phi0']:6.2f}  {r['y0']:+8.3f}  "
                  f"{r['N_star']:6.1f}  {r['chi2']:7.2f}  "
                  f"{r.get('d2', 0):7.1f}  {r['k_dip']:10.2e}  "
                  f"{r.get('suppression_pct', 0):7.1f}%", flush=True)

def setup_args():
    p = argparse.ArgumentParser(description="CAMB-based Higgs USR scan")

    # Mode
    p.add_argument("--auto", action="store_true",
                   help="Automatic two-phase scan")
    p.add_argument("--test", action="store_true",
                   help="Quick test (3x3 grid, no Phase 2)")
    p.add_argument("--phase", choices=["broad", "fine", "random"], default=None)

    # Phase 1 params
    p.add_argument("--phi0-range", nargs=2, type=float, default=[5.5, 7.0])
    p.add_argument("--y0-range", nargs=2, type=float, default=[-1.0, -0.01])
    p.add_argument("--n-phi0", type=int, default=11)
    p.add_argument("--n-y0", type=int, default=10)
    p.add_argument("--phi0-step", type=float, default=0.15)

    # Phase 2 params
    p.add_argument("--n-phi0-fine", type=int, default=5)
    p.add_argument("--n-y0-fine", type=int, default=5)
    p.add_argument("--n-nstar-fine", type=int, default=9)
    p.add_argument("--phi0-fine-window", type=float, default=0.1)
    p.add_argument("--y0-fine-window", type=float, default=0.05)
    p.add_argument("--nstar-fine-window", type=float, default=2.0)
    p.add_argument("--max-regions", type=int, default=3)

    # Random scan params
    p.add_argument("--n-random", type=int, default=100,
                   help="Number of random configs for --phase random")
    p.add_argument("--random-seed", type=int, default=42,
                   help="RNG seed for --phase random")

    # Pivot selection
    p.add_argument(
        "--k-pivot", type=float, default=0.002,
        help="Pivot k_pivot_phys (Mpc^-1) — drives BOTH As normalization and "
             "n_s extraction (single-pivot invariant). Higgs default 0.002; "
             "Planck 0.05.",
    )
    p.add_argument("--nstar-range", nargs=2, type=float, default=[49.0, 62.0],
                   help="N_star range for --phase random")

    # Targets
    p.add_argument("--k-dip-target", type=float, default=1.8e-4)
    p.add_argument("--y0-step", type=float, default=0.1)

    # Pipeline
    p.add_argument("--k-min", type=float, default=1e-5)
    p.add_argument("--k-max", type=float, default=1.0)
    p.add_argument("--num-k", type=int, default=80)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--xi", type=float, default=15000.0)
    p.add_argument("--lam", type=float, default=0.13)
    p.add_argument("--ell-max", type=int, default=2500,
                   help="CAMB ell_max (default: 2500). Use 30 for fast scan.")

    # Mode control
    p.add_argument("--quick", action="store_true",
                   help="Fast screening: ~36 weighted k-modes, CAMB ell_max=30 (~2s/config)")
    p.add_argument("--full-chi2", action="store_true",
                   help="Full CAMB ell_max=2500 for chi2_full (~3x slower per config)")
    p.add_argument("--mode", choices=["chi2", "d2"], default="chi2",
                   help="Optimization mode (default: chi2)")
    p.add_argument("--max-chi2", type=float, default=None,
                   help="Maximum chi2 threshold (default: LCDM=20.47 for chi2 mode, 35 for d2 mode)")
    p.add_argument("--d2-target", type=float, default=225.9,
                   help="Target D2 value for ranking (default: 225.9 = Planck quadrupole)")
    p.add_argument("--sort-by", choices=["chi2", "d2"], default=None,
                   help="Sort metric for promising regions (default: same as --mode)")

    # Resume
    p.add_argument("--phase1-log", type=str, default=None,
                   help="Path to existing Phase 1 log for Phase 2")

    return p.parse_args()

def main():
    args = setup_args()

    # Apply the user-selected pivot globally so all bare `k_pivot_phys`
    # references in this module use it (single-pivot invariant).
    global k_pivot_phys, As
    from scripts.constants import As_planck, ns_sr_default
    k_pivot_phys = args.k_pivot
    As = As_planck * (args.k_pivot / 0.05) ** (ns_sr_default - 1.0)
    print(f"  k_pivot = {k_pivot_phys} Mpc^-1   A_s = {As:.3e}")

    if args.max_chi2 is None:
        if args.full_chi2:
            args.max_chi2 = CHI2_LCDM_FULL + 200  # allow 200 above LCDM full
        else:
            args.max_chi2 = CHI2_LCDM if args.mode == "chi2" else 35.0
    if args.sort_by is None:
        args.sort_by = args.mode

    if args.mode == "d2" and not any("--phi0-range" in a for a in sys.argv):
        args.phi0_range = [5.5, 7.5]
        args.n_phi0 = 16
    if args.mode == "d2" and not any("--y0-range" in a for a in sys.argv):
        args.y0_range = [-0.85, -0.10]
        args.n_y0 = 16

    print(f"{'='*60}", flush=True)
    print(f"  CAMB Higgs USR Scan", flush=True)
    print(f"  Mode: "
          f"{'test' if args.test else 'auto' if args.auto else args.phase}")
    print(f"{'='*60}", flush=True)

    if args.test:
        args.n_phi0 = 3
        args.n_y0 = 3
        args.phi0_range = [6.4, 6.8]
        args.y0_range = [-0.8, -0.4]
        args.auto = False
        args.phase = "broad"
        args.quick = True

    if args.quick and not args.full_chi2:
        args.ell_max = 30
    if args.full_chi2:
        if not args.quick and "--quick" not in " ".join(sys.argv):
            print("  --full-chi2 requires --quick. Enabling automatically.", flush=True)
            args.quick = True
        args.ell_max = 2500

    completed_phase1 = set()
    completed_phase2 = set()

    if args.phase == "broad" or args.auto:
        log_path_broad = run_phase1(args, completed_phase1)

        if args.test:
            print_summary(log_path_broad, mode=args.mode, max_chi2=args.max_chi2)
            print("\n  Test complete. Check results above.", flush=True)
            return

        print("\n  Analyzing Phase 1 results for promising regions...",
              flush=True)
        regions = find_promising_regions(log_path_broad, args)

        if not regions:
            print("\n  No promising regions found. Stopping.", flush=True)
            print_summary(log_path_broad, mode=args.mode, max_chi2=args.max_chi2)
            return

        if args.auto:
            print(f"\n  Found {len(regions)} promising region(s). "
                  f"Starting Phase 2...", flush=True)
            run_phase2(args, completed_phase2, regions)
            print_summary(log_path_broad, mode=args.mode, max_chi2=args.max_chi2)

    elif args.phase == "fine":
        if args.phase1_log is None:
            print("  ERROR: --phase1-log required for --phase fine",
                  file=sys.stderr, flush=True)
            sys.exit(1)

        regions = find_promising_regions(args.phase1_log, args)
        if not regions:
            print("  ERROR: No promising regions in Phase 1 log",
                  file=sys.stderr, flush=True)
            sys.exit(1)

        print(f"  Found {len(regions)} promising region(s)", flush=True)
        run_phase2(args, completed_phase2, regions)
        print_summary(args.phase1_log, mode=args.mode, max_chi2=args.max_chi2)

    elif args.phase == "random":
        log_path_random = run_random_scan(args)
        print_summary(log_path_random, mode=args.mode, max_chi2=args.max_chi2)

    if args.phase in ("broad", "fine", "random") or args.test:
        pass

    print("\nDone.", flush=True)

if __name__ == "__main__":
    main()
