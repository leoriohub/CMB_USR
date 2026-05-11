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

from scripts.pspectrum_pipeline import (
    run_pspectrum_pipeline, find_end_of_inflation, build_weighted_kgrid,
)
from scripts.sachs_wolfe import compute_cl_sw, compute_cl_sw_powerlaw
from scripts.planck_data import get_planck_data, get_planck_data_asymmetric, C_ell_to_d_ell
from scripts.constants import As, k_pivot_phys, ROOT_DIR
from scripts.optimizer_utils import find_k_dip, compute_chi2, _write_log
from models import HiggsModel
import inf_dyn_background as bg_solver

# ── Helpers ──────────────────────────────────────────────────────────────────


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
    """Run a grid scan over (phi0, y0) with automatic N_star alignment.

    For each (phi0, y0) pair:
      1. Run a quick background-only solve to measure N_total and N_dip
      2. Compute N_star = N_total - N_dip - delta_N
         where delta_N = ln(k_pivot / k_dip_target) approximately equal to 4.8
      3. Run full pspectrum pipeline with computed N_star
      4. Compute chi^2, dip suppression, log to JSONL
    """
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

            # Find where epsH is minimum during inflation (the USR dip)
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


# ── Post-processing ────────────────────────────────────────────────────────────


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


def plot_scan_results(ok, args, date_str):
    """Generate chi^2 heatmap, suppression heatmap, and best P_S(k) overlay."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phi0_vals = sorted(set(r["phi0"] for r in ok))
    y0_vals = sorted(set(r["y0"] for r in ok))
    chi2_map = np.full((len(y0_vals), len(phi0_vals)), np.nan)
    supp_map = np.full((len(y0_vals), len(phi0_vals)), np.nan)

    for r in ok:
        ip = phi0_vals.index(r["phi0"])
        iy = y0_vals.index(r["y0"])
        chi2_map[iy, ip] = r["chi2"]
        supp_map[iy, ip] = r["suppression_pct"]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

    im1 = ax1.pcolormesh(phi0_vals, y0_vals, chi2_map, shading="auto",
                          cmap="viridis_r")
    ax1.set_xlabel("$\\phi_0$")
    ax1.set_ylabel("$y_0$")
    ax1.set_title(f"$\\chi^2$ vs Planck low-$\\ell$")
    plt.colorbar(im1, ax=ax1, label="$\\chi^2$")

    im2 = ax2.pcolormesh(phi0_vals, y0_vals, supp_map, shading="auto",
                          cmap="coolwarm", vmin=-50, vmax=50)
    ax2.set_xlabel("$\\phi_0$")
    ax2.set_ylabel("$y_0$")
    ax2.set_title("Dip Suppression [%]")
    plt.colorbar(im2, ax=ax2, label="suppression %")

    plt.suptitle("Higgs USR Grid Scan Results", fontsize=13, fontweight="bold")
    plt.tight_layout()

    plot_dir = os.path.join(ROOT_DIR, "outputs/plots/optimizer")
    os.makedirs(plot_dir, exist_ok=True)
    path = os.path.join(plot_dir, f"higgs_scan_{date_str}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Scan plot saved -> {path}", flush=True)


def plot_best_spectrum(best, args, date_str):
    """Re-run the pipeline at the best config with save and produce
    a dashboard plot (P_S(k) + C_ell comparison)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    phi0 = best["phi0"]
    y0 = best["y0"]
    N_star = best["N_star"]

    print(f"\n  Re-running best at phi0={phi0:.4f} y0={y0:.4f} "
          f"N_star={N_star:.2f}", flush=True)

    model = HiggsModel(lam=args.lam, xi=args.xi)
    k_grid = build_weighted_kgrid(args.k_min, args.k_max, k_pivot_phys)
    output_dir = os.path.join(ROOT_DIR, "outputs/simulations/pspectra")

    result = run_pspectrum_pipeline(
        model=model, phi0=phi0, y0=y0,
        k_min=args.k_min, k_max=args.k_max,
        k_pivot_phys=k_pivot_phys, N_star=N_star,
        normalize_to_As=True, As=As,
        output_dir=output_dir, save_outputs=True,
        k_phys_grid=k_grid, n_workers=args.workers,
    )

    if result["status"] != "success":
        print(f"  Pipeline failed: {result.get('message','')}", flush=True)
        return

    # P_S(k) plot
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.loglog(result["k_phys"], result["P_S"], "b-", lw=1.5)
    ax1.axvline(k_pivot_phys, color="k", ls="--", alpha=0.5,
                label=f"pivot ({k_pivot_phys} Mpc$^{{-1}}$)")
    kd = best.get("k_dip", -1.0)
    if kd > 0:
        ax1.axvline(kd, color="r", ls=":", alpha=0.7,
                    label=f"dip ({kd:.2e} Mpc$^{{-1}}$)")
    ax1.set_xlabel("$k$ [Mpc$^{-1}$]")
    ax1.set_ylabel("$\\mathcal{P}_\\mathcal{R}(k)$")
    ax1.set_title("Primordial Power Spectrum (best)")
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # C_ell comparison
    ells_usr, C_usr = compute_cl_sw(result, ell_max=args.ell_max)
    D_usr = C_ell_to_d_ell(ells_usr, C_usr)

    ells_lcdm, C_lcdm, _ = compute_cl_sw_powerlaw(
        k_min=args.k_min, k_max=args.k_max,
        As=As, ns=0.965, ell_max=args.ell_max,
    )
    D_lcdm = C_ell_to_d_ell(ells_lcdm, C_lcdm)

    ells_pl, D_pl, D_err = get_planck_data()
    mask = ells_pl <= args.ell_max

    ax2.errorbar(ells_pl[mask], D_pl[mask], yerr=D_err[mask],
                 fmt="ko", ms=4, capsize=2, label="Planck 2018")
    ax2.semilogx(ells_usr, D_usr, "r-", lw=1.5, label="USR Best")
    ax2.semilogx(ells_lcdm, D_lcdm, "gray", ls="--", lw=1,
                 label="$\\Lambda$CDM")
    ax2.set_xlabel("Multipole $\\ell$")
    ax2.set_ylabel("$D_\\ell$ [$\\mu$K$^2$]")
    ax2.set_title("CMB Low-$\\ell$ Power Spectrum")
    ax2.legend(fontsize=8)
    ax2.set_xlim(1.5, args.ell_max)
    ax2.grid(True, alpha=0.3)

    chi2 = compute_chi2(result, ell_max=args.ell_max)
    suppression = best.get("suppression_pct", 0)
    plt.suptitle(
        f"Best Higgs USR: $\\phi_0$={phi0:.2f}, $y_0$={y0:.3f}, "
        f"$N_{{*}}$={N_star:.0f}\n"
        f"$\\chi^2$={chi2:.1f}, dip at $k$={kd:.2e} Mpc$^{{-1}}$, "
        f"suppression={suppression:+.1f}%",
        fontsize=11, fontweight="bold",
    )

    plot_dir = os.path.join(ROOT_DIR, "outputs/plots/optimizer")
    os.makedirs(plot_dir, exist_ok=True)
    path = os.path.join(plot_dir, f"higgs_best_{date_str}.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Dashboard plot saved -> {path}", flush=True)


# ── CLI ─────────────────────────────────────────────────────────────────────────


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="Higgs USR Grid-Scan Optimizer: find (phi0, y0) for "
                    "the P_S(k) dip at k ~ 1-5e-4 Mpc^-1"
    )
    # Model
    p.add_argument("--xi", type=float, default=15000.0)
    p.add_argument("--lam", type=float, default=0.13)
    # Grid
    p.add_argument("--phi0-range", nargs=2, type=float,
                   default=[5.20, 5.80], metavar=("MIN", "MAX"))
    p.add_argument("--y0-range", nargs=2, type=float,
                   default=[-0.20, -0.05], metavar=("MIN", "MAX"))
    p.add_argument("--n-phi0", type=int, default=13,
                   help="phi0 grid points")
    p.add_argument("--n-y0", type=int, default=16,
                   help="y0 grid points")
    # Targets
    p.add_argument("--k-dip-range", nargs=2, type=float,
                   default=[1e-4, 5e-4], metavar=("MIN", "MAX"))
    p.add_argument("--ell-max", type=int, default=29)
    # Pipeline
    p.add_argument("--k-min", type=float, default=1e-5)
    p.add_argument("--k-max", type=float, default=1.0)
    p.add_argument("--num-k", type=int, default=80)
    p.add_argument("--workers", type=int, default=4,
                   help="k-mode parallel workers inside pipeline")
    # Output
    p.add_argument("--log", type=str, default=None,
                   help="JSONL log path (auto-generated if not given)")
    p.add_argument("--plot-best", action="store_true",
                   help="re-run and plot the best configuration")
    return p.parse_args(argv)


def main():
    args = parse_args()

    if args.log is None:
        ds = datetime.now().strftime("%Y%m%d_%H%M%S")
        args.log = os.path.join(
            ROOT_DIR, f"outputs/simulations/logs/higgs_scan_{ds}.jsonl")

    print(f"{'='*60}", flush=True)
    print(f"  Higgs USR Grid-Scan Optimizer", flush=True)
    print(f"  phi0=[{args.phi0_range[0]:.2f}, {args.phi0_range[1]:.2f}] "
          f"({args.n_phi0} pts)", flush=True)
    print(f"  y0=[{args.y0_range[0]:.3f}, {args.y0_range[1]:.3f}] "
          f"({args.n_y0} pts)", flush=True)
    print(f"  k_dip_target=[{args.k_dip_range[0]:.1e},"
          f"{args.k_dip_range[1]:.1e}] Mpc^-1", flush=True)
    print(f"  xi={args.xi}, lam={args.lam}, workers={args.workers}",
          flush=True)
    print(f"  Log: {args.log}", flush=True)
    print(f"{'='*60}", flush=True)

    results = run_grid_scan(args)

    ok, date_str = print_best_results(results, args)
    if ok is None:
        print("\nNo good configurations found.", flush=True)
        return

    plot_scan_results(ok, args, date_str)

    if args.plot_best and ok:
        plot_best_spectrum(ok[0], args, date_str)

    print(f"\nDone.", flush=True)


if __name__ == "__main__":
    main()
