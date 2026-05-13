"""
CAMB-based Higgs USR scan with automatic two-phase execution.

Usage:
  python scripts/camb_scan.py --test          # Quick test (3x3 grid)
  python scripts/camb_scan.py --auto          # Full automatic: broad → fine
  python scripts/camb_scan.py --phase broad   # Phase 1 only
  python scripts/camb_scan.py --phase fine    # Phase 2 only (needs Phase 1 results)
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime

import numpy as np

from scripts.pspectrum_pipeline import (
    run_pspectrum_pipeline, find_end_of_inflation,
)
from scripts.camb_wrapper import compute_chi2_camb, compute_cl_camb_powerlaw
from scripts.planck_data import get_planck_data_asymmetric, C_ell_to_d_ell
from scripts.optimizer_utils import find_k_dip, _write_log
from scripts.constants import As, k_pivot_phys, ROOT_DIR
from models import HiggsModel
import inf_dyn_background as bg_solver

T_cmb = 2.7255
K_DIP_MIN = 1.4e-4
K_DIP_MAX = 2.1e-3
CHI2_LCDM = 20.47
D2_LCDM = 1028.7


def dipole_power(ps_data):
    """D_ell at ℓ=2 via CAMB full."""
    from scripts.camb_wrapper import compute_cl_full_camb
    ells, C_TT, _, _ = compute_cl_full_camb(ps_data, ell_max=29)
    D = C_ell_to_d_ell(ells, C_TT)
    return float(D[0]), ells, D


def passes_criteria(chi2, d2, n_star, k_dip):
    if k_dip <= 0:
        return False
    return (chi2 <= CHI2_LCDM and d2 < D2_LCDM
            and n_star >= 50 and K_DIP_MIN <= k_dip <= K_DIP_MAX)


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
                k = (round(rec.get("phi0", 0), 4),
                     round(rec.get("y0", 0), 4),
                     round(rec.get("N_star", 0), 2))
                completed.add(k)
            except (json.JSONDecodeError, KeyError):
                pass
    return completed


def evaluate_config(phi0, y0, N_star, args):
    """Run full pipeline + CAMB chi2. Returns dict with results."""
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

    try:
        ps_data = {"k_phys": k_phys, "P_S": P_S}
        d2_val, ells_c, D_c = dipole_power(ps_data)
    except Exception as e:
        return {"status": "camb_fail", "error": str(e), "k_dip": k_dip}

    try:
        chi2_m, chi2_l, dchi2 = compute_chi2_camb(ps_data, ell_max=29)
    except Exception as e:
        return {"status": "chi2_fail", "error": str(e), "k_dip": k_dip,
                "d2": d2_val}

    supp = 0.0
    if k_dip > 0:
        ps_dip = float(np.interp(k_dip, k_phys, P_S))
        ps_pivot = float(np.interp(k_pivot_phys, k_phys, P_S))
        if ps_pivot > 0:
            supp = (1.0 - ps_dip / ps_pivot) * 100

    meta = result.get("metadata", {})
    return {
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
    }


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
    print(f"  Total pairs: {total}, viable est: ~{int(total*0.3)}")
    print(f"{'='*60}", flush=True)

    log_path = os.path.join(ROOT_DIR, "outputs/simulations/logs",
                            f"camb_phase1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "a")

    t0 = time.time()
    results = []
    done = 0

    for phi0 in phi0_vals:
        for y0 in y0_vals:
            done_phase = len(results) + sum(1 for r in results
                                            if r.get("status") != "bg_fail")

            # Quick background check + N_star auto-alignment
            model_check = HiggsModel(lam=args.lam, xi=args.xi)
            model_check.phi0 = phi0
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

            end_idx = find_end_of_inflation(d["epsH"])
            if end_idx == -1:
                end_idx = len(d["epsH"]) - 1
            N_total = float(d["N"][end_idx])

            eps_inf = d["epsH"][(d["N"] >= 0) & (d["N"] < N_total)]
            if len(eps_inf) < 10:
                done += 1
                eta = (time.time() - t0) / done * (total - done) if done else 0
                print(f"\r  [{done:3d}/{total}] phi0={phi0:.2f} y0={y0:+.3f} "
                      f"SKIP (no_usr) ETA {eta/60:.0f}m", end="", flush=True)
                results.append({"phi0": phi0, "y0": y0, "status": "no_usr"})
                continue

            dip_N = float(d["N"][np.argmin(eps_inf[10:]) + 10])
            N_after_dip = N_total - dip_N
            delta_N = np.log(k_pivot_phys / args.k_dip_target)
            nstar_auto = N_after_dip - delta_N

            if nstar_auto <= 0 or nstar_auto > N_total:
                done += 1
                eta = (time.time() - t0) / done * (total - done) if done else 0
                print(f"\r  [{done:3d}/{total}] phi0={phi0:.2f} y0={y0:+.3f} "
                      f"SKIP (bad_nstar) ETA {eta/60:.0f}m", end="", flush=True)
                results.append({"phi0": phi0, "y0": y0, "status": "bad_nstar"})
                continue

            if nstar_auto < 49 or nstar_auto > N_total - 1:
                done += 1
                eta = (time.time() - t0) / done * (total - done) if done else 0
                print(f"\r  [{done:3d}/{total}] phi0={phi0:.2f} y0={y0:+.3f} "
                      f"SKIP (nstar<49) ETA {eta/60:.0f}m", end="", flush=True)
                results.append({"phi0": phi0, "y0": y0, "status": "nstar_low"})
                continue

            N_star_candidates = [round(nstar_auto, 2)]
            if nstar_auto < 50:
                N_star_candidates.append(50.0)
            for N_star in N_star_candidates:
                key = (round(phi0, 4), round(y0, 4), round(N_star, 2))
                if key in completed:
                    done += 1
                    continue

                done += 1
                entry = {
                    "eval": done, "total": total,
                    "phi0": round(phi0, 4), "y0": round(y0, 4),
                    "N_star": N_star,
                    "N_total": round(N_total, 1),
                    "dip_N": round(dip_N, 1),
                    "nstar_auto": round(nstar_auto, 2),
                    "timestamp": datetime.now().isoformat(),
                }

                res = evaluate_config(phi0, y0, N_star, args)
                entry.update(res)

                _write_log(log_file, entry)

                eta = (time.time() - t0) / done * (total - done) if done else 0
                chi2_str = (f"chi2={res.get('chi2', '?'):.1f}"
                            if res.get("status") == "ok"
                            else f"SKIP ({res.get('status', '?')})")
                d2_str = (f"d2={res.get('d2', 0):.0f}"
                          if "d2" in res and res["d2"]
                          else "")
                print(f"\r  [{done:3d}/{total}] phi0={phi0:.2f} y0={y0:+.3f} "
                      f"N*={N_star:.0f} {chi2_str} {d2_str} "
                      f"k_dip={res.get('k_dip', 0):.2e} "
                      f"ETA {eta/60:.0f}m", end="", flush=True)

        print()

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
            if rec.get("status") != "ok":
                continue
            chi2 = rec.get("chi2", 999)
            d2 = rec.get("d2", 9999)
            n_star = rec.get("N_star", 0)
            k_dip = rec.get("k_dip", -1)
            if passes_criteria(chi2, d2, n_star, k_dip):
                candidates.append(rec)

    if not candidates:
        print("  No promising configs found in Phase 1 results", flush=True)
        return regions

    candidates.sort(key=lambda r: r.get("chi2", 999))

    # Cluster nearby configs
    merged = []
    for c in candidates:
        added = False
        for cluster in merged:
            dc = abs(cluster["phi0"] - c["phi0"])
            dy = abs(cluster["y0"] - c["y0"])
            if dc <= args.phi0_step * 1.5 and dy <= args.y0_step * 1.5:
                if c["chi2"] < cluster["chi2"]:
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

    log_path = os.path.join(ROOT_DIR, "outputs/simulations/logs",
                            f"camb_phase2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl")
    os.makedirs(os.path.dirname(log_path), exist_ok=True)
    log_file = open(log_path, "a")

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
                        "eval": done[0], "total": total_est,
                        "phi0": round(phi0, 4), "y0": round(y0, 4),
                        "N_star": N_star,
                        "phase": 2, "region": reg_idx + 1,
                        "timestamp": datetime.now().isoformat(),
                    }

                    res = evaluate_config(phi0, y0, N_star, args)
                    entry.update(res)

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

    log_file.close()
    elapsed = time.time() - t0
    print(f"\n\n  Phase 2 complete: {done[0]} evals in "
          f"{elapsed:.0f}s ({elapsed/60:.1f}m)", flush=True)


def print_summary(log_path_broad, log_path_fine=None):
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
                if rec.get("status") != "ok":
                    continue
                chi2 = rec.get("chi2", 999)
                d2 = rec.get("d2", 9999)
                n_star = rec.get("N_star", 0)
                k_dip = rec.get("k_dip", -1)
                if passes_criteria(chi2, d2, n_star, k_dip):
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
    p.add_argument("--phase", choices=["broad", "fine"], default=None)

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

    # Resume
    p.add_argument("--phase1-log", type=str, default=None,
                   help="Path to existing Phase 1 log for Phase 2")

    return p.parse_args()


def main():
    args = setup_args()

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

    completed_phase1 = set()
    completed_phase2 = set()

    if args.phase == "broad" or args.auto:
        log_path_broad = run_phase1(args, completed_phase1)

        if args.test:
            print_summary(log_path_broad)
            print("\n  Test complete. Check results above.", flush=True)
            return

        print("\n  Analyzing Phase 1 results for promising regions...",
              flush=True)
        regions = find_promising_regions(log_path_broad, args)

        if not regions:
            print("\n  No promising regions found. Stopping.", flush=True)
            print_summary(log_path_broad)
            return

        if args.auto:
            print(f"\n  Found {len(regions)} promising region(s). "
                  f"Starting Phase 2...", flush=True)
            run_phase2(args, completed_phase2, regions)
            print_summary(log_path_broad)

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
        print_summary(args.phase1_log)

    if args.phase in ("broad", "fine") or args.test:
        pass

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
