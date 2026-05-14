"""
Self-contained full analysis pipeline for a single Higgs USR configuration.

Exercises all codebase capabilities:
  - Background integration + derived quantities
  - P_S(k) with high-resolution weighted k-grid
  - Full CAMB C_ell (TT/TE/EE, ell_max=2500) + Planck low-ell comparison
  - Publication-ready plots (PS, D_ell, CAMB comparison, full sky)
  - All outputs in a single run-specific subfolder
"""

import argparse
import json
import os
import shutil
import sys
import time

import numpy as np

import matplotlib.pyplot as plt
from scripts.plotting import (
    plot_background, plot_ps, plot_dell,
    plot_camb_comparison, plot_camb_fullsky,
)

import inf_dyn_background as bg_solver
from models import HiggsModel
from scripts.constants import As, k_pivot_phys, ROOT_DIR
from pspectrum_pipeline import (
    run_pspectrum_pipeline,
    build_weighted_kgrid,
    ensure_k_pivot,
)
from scripts.camb_wrapper import compute_cl_full_camb, compute_cl_camb_powerlaw
from scripts.planck_data import get_planck_data_asymmetric, C_ell_to_d_ell


def save_background(model, T_span_bg, output_dir, run_label):
    """Run background integration and save trajectory + derived quantities."""
    bg_sol = bg_solver.run_background_simulation(model, T_span_bg)
    derived = bg_solver.get_derived_quantities(bg_sol, model)

    x, y, z, n = bg_sol
    record = {
        "_type": "result",
        "format_version": 2,
        "metadata": {
            "model": model.name,
            "x0": float(model.x0),
            "y0": float(model.y0),
            "xi": getattr(model, "xi_val", None),
            "lam": getattr(model, "lam", None),
            "bg_steps": int(len(T_span_bg)),
            "T_max": float(T_span_bg[-1]),
        },
        "trajectory": {
            "T": T_span_bg.tolist(),
            "phi": x.tolist(),
            "y": y.tolist(),
            "z": z.tolist(),
            "n": n.tolist(),
        },
        "derived": {
            "N": derived["N"].tolist(),
            "epsH": derived["epsH"].tolist(),
            "etaH": derived["etaH"].tolist(),
            "ns": derived["ns"].tolist(),
            "r": derived["r"].tolist(),
            "Ps": derived["Ps"].tolist(),
            "Pt": derived["Pt"].tolist(),
        },
    }

    path = os.path.join(output_dir, f"background_{run_label}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"  Saved: {path}")
    return bg_sol, derived


def compute_camb(result, output_dir, run_label, ell_max=2500):
    """Compute full CAMB C_ell and chi^2, save to JSON, return data dict."""
    ells_camb, C_camb, C_TE, C_EE = compute_cl_full_camb(result, ell_max=ell_max)
    D_camb = C_ell_to_d_ell(ells_camb, C_camb)

    ells_pl, C_pl, _, _ = compute_cl_camb_powerlaw(ell_max=ell_max)
    D_pl = C_ell_to_d_ell(ells_pl, C_pl)

    planck_ells, D_planck, D_err_lower, D_err_upper = get_planck_data_asymmetric()

    chi2_model = 0.0
    chi2_lcdm = 0.0
    for i, ell_val in enumerate(planck_ells):
        if ell_val > 29:
            continue
        idx = int(np.argmin(np.abs(ells_camb - ell_val)))
        r_model = D_camb[idx] - D_planck[i]
        r_lcdm = D_pl[idx] - D_planck[i]
        sigma = D_err_upper[i] if r_model > 0 else D_err_lower[i]
        chi2_model += (r_model / sigma) ** 2
        sigma = D_err_upper[i] if r_lcdm > 0 else D_err_lower[i]
        chi2_lcdm += (r_lcdm / sigma) ** 2

    record = {
        "_type": "result",
        "format_version": 2,
        "metadata": {"ell_max": ell_max, "computation": "CAMB_full"},
        "c_ell": {
            "model": {
                "ells": ells_camb.tolist(),
                "C_ell_TT": C_camb.tolist(),
                "D_ell": D_camb.tolist(),
                "chi2": chi2_model,
            },
            "lcdm": {
                "ells": ells_pl.tolist(),
                "C_ell_TT": C_pl.tolist(),
                "D_ell": D_pl.tolist(),
                "chi2": chi2_lcdm,
            },
            "planck_data": {
                "ells": planck_ells.tolist(),
                "D_ell": D_planck.tolist(),
                "D_ell_err_lower": D_err_lower.tolist(),
                "D_ell_err_upper": D_err_upper.tolist(),
            },
        },
    }

    path = os.path.join(output_dir, f"camb_{run_label}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"  Saved: {path}")

    return {
        "ells": ells_camb,
        "D_camb": D_camb,
        "D_pl": D_pl,
        "planck_ells": planck_ells,
        "D_planck": D_planck,
        "D_err_lower": D_err_lower,
        "D_err_upper": D_err_upper,
        "chi2_model": chi2_model,
        "chi2_lcdm": chi2_lcdm,
    }





def collect_run_outputs(run_label, result, configs_dir, cell_dir, pspectra_dir,
                        diag_plots_dir, powerloss_plots_dir, chi2_model_val,
                        chi2_lcdm_val):
    """Copy all outputs for this run into a single convenience directory."""
    run_dir = os.path.join(ROOT_DIR, "outputs", "simulations", "runs", run_label)
    subdirs = {
        "configs": os.path.join(run_dir, "configs"),
        "c_ell": os.path.join(run_dir, "c_ell"),
        "pspectra": os.path.join(run_dir, "pspectra"),
        "plots/diagnostics": os.path.join(run_dir, "plots", "diagnostics"),
        "plots/powerloss": os.path.join(run_dir, "plots", "powerloss"),
    }
    # Wipe stale files before copying fresh ones
    for d in subdirs.values():
        if os.path.exists(d):
            for f in os.listdir(d):
                fp = os.path.join(d, f)
                if os.path.isfile(fp):
                    os.remove(fp)
        os.makedirs(d, exist_ok=True)

    files = [
        (os.path.join(configs_dir, f"background_{run_label}.json"),
         subdirs["configs"]),
        (os.path.join(cell_dir, f"camb_{run_label}.json"),
         subdirs["c_ell"]),
    ]

    pspectrum_path = result.get("output_file")
    if pspectrum_path and os.path.exists(pspectrum_path):
        files.append((pspectrum_path, subdirs["pspectra"]))

    files.append(
        (os.path.join(diag_plots_dir, f"background_{run_label}.png"),
         subdirs["plots/diagnostics"]))
    files.append(
        (os.path.join(powerloss_plots_dir, f"ps_{run_label}.png"),
         subdirs["plots/powerloss"]))
    files.append(
        (os.path.join(powerloss_plots_dir, f"dell_{run_label}.png"),
         subdirs["plots/powerloss"]))

    copied = []
    for src, dst_dir in files:
        if os.path.exists(src):
            shutil.copy2(src, dst_dir)
            copied.append(os.path.join(dst_dir, os.path.basename(src)))

    meta = result["metadata"]
    k = result["k_phys"]
    ps = result["P_S"]
    k_dip = k[np.argmin(ps)]
    p_dip = np.min(ps)
    suppression = (1 - p_dip / As) * 100

    readme_path = os.path.join(run_dir, "README.txt")
    with open(readme_path, "w") as f:
        f.write(f"Higgs USR Analysis — {run_label}\n")
        f.write(f"{'=' * 50}\n\n")
        f.write(f"phi0 / x0:  {meta.get('x0', '?'):.2f}\n")
        f.write(f"y0:         {meta.get('y0', '?'):.3f}\n")
        f.write(f"N_star:     {meta.get('N_star', '?'):.2f}\n")
        f.write(f"N_total:    {meta.get('N_total', '?'):.2f}\n")
        f.write(f"N_pivot:    {meta.get('N_pivot', '?'):.1f}\n")
        f.write(f"chi2 (USR): {chi2_model_val:.2f}\n")
        f.write(f"chi2 (LCDM): {chi2_lcdm_val:.2f}\n")
        f.write(f"k_dip:      {k_dip:.2e} Mpc^-1\n")
        f.write(f"P_S(k_dip): {p_dip:.3e}\n")
        f.write(f"Suppress:   {suppression:.1f}%\n")
        f.write(f"\nFiles:\n")
        for c in copied:
            f.write(f"  {c}\n")

    print(f"\n  Run directory: {run_dir}/")
    return run_dir


def print_summary(result, bg_sol, derived, chi2_model_val, chi2_camb_model, camb_ell_max, output_dir):
    """Print a structured summary of the analysis."""
    meta = result["metadata"]
    k = result["k_phys"]
    ps = result["P_S"]

    k_dip = k[np.argmin(ps)]
    p_dip = np.min(ps)
    suppression = (1 - p_dip / As) * 100

    N_total = meta.get("N_total", "?")
    N_pivot = meta.get("N_pivot", "?")
    n_modes_ok = meta.get("n_completed", "?")
    n_modes_total = meta.get("num_k", "?")

    print()
    print("=" * 44)
    print("  ANALYSIS SUMMARY")
    print("=" * 44)
    print(f"  Model:          {meta.get('model', '?')}")
    print(f"  phi0:           {meta.get('x0', '?'):.2f}")
    print(f"  y0:             {meta.get('y0', '?'):.3f}")
    xi_v = meta.get("xi", None)
    if xi_v is not None:
        print(f"  xi:             {xi_v}")
    lam_v = meta.get("lam", None)
    if lam_v is not None:
        print(f"  lambda:         {lam_v}")
    print(f"  N_star:         {meta.get('N_star', '?'):.2f}")
    print(f"  N_total:        {N_total}")
    if isinstance(N_pivot, (int, float)):
        print(f"  N_pivot:        {N_pivot:.1f}")
    print(f"  k_dip:          {k_dip:.2e} Mpc^-1")
    print(f"  P_S(k_dip):     {p_dip:.4e}")
    print(f"  Suppression:    {suppression:.1f}%")
    print(f"  chi2 CAMB (ℓ≤29):  {chi2_camb_model:.2f}")
    print(f"  CAMB ℓ_max:     {camb_ell_max}")
    print(f"  k-modes:        {n_modes_ok}/{n_modes_total} completed")
    print(f"  Output dir:     {output_dir}/")
    print("=" * 44)
    print()


def parse_args():
    p = argparse.ArgumentParser(
        description="Self-contained full analysis for a single Higgs USR configuration"
    )
    p.add_argument("--phi0", type=float, required=True, help="Field initial value")
    p.add_argument("--y0", type=float, required=True, help="Velocity initial value")
    p.add_argument("--nstar", type=float, required=True, help="N_star for pivot alignment")
    p.add_argument("--xi", type=float, default=15000.0, help="Higgs non-minimal coupling")
    p.add_argument("--lam", type=float, default=0.13, help="Self-coupling")
    p.add_argument("--ms-steps", type=int, default=5000, help="MS integration steps")
    p.add_argument("--bg-steps", type=int, default=1000, help="Background integration steps")
    p.add_argument("--T-max", type=float, default=500.0, help="Max conformal time")
    p.add_argument("--workers", type=int, default=os.cpu_count(),
                   help="Parallel workers for MS integration (default: all cores)")
    p.add_argument("--no-interp", action="store_true",
                   help="Disable plot interpolation (raw data points)")
    p.add_argument("--camb-ell-max", type=int, default=2500,
                   help="ell_max for CAMB computation (default: 2500)")
    return p.parse_args()


def main():
    args = parse_args()

    run_label = f"phi{args.phi0:.2f}_y0{args.y0:.3f}_nstar{args.nstar:.1f}"

    # AGENTS.md output hierarchy
    configs_dir = os.path.join(ROOT_DIR, "outputs", "simulations", "configs")
    pspectra_dir = os.path.join(ROOT_DIR, "outputs", "simulations", "pspectra")
    cell_dir = os.path.join(ROOT_DIR, "outputs", "simulations", "c_ell")
    diag_plots_dir = os.path.join(ROOT_DIR, "outputs", "plots", "diagnostics")
    powerloss_plots_dir = os.path.join(ROOT_DIR, "outputs", "plots", "powerloss")
    for d in [configs_dir, pspectra_dir, cell_dir, diag_plots_dir, powerloss_plots_dir]:
        os.makedirs(d, exist_ok=True)

    plt.rcParams.update({
        "font.size": 8,
        "axes.labelsize": 8,
        "axes.titlesize": 9,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 6,
        "figure.dpi": 300,
        "font.family": "serif",
    })

    print("\n  1. Background integration...")
    t0 = time.time()
    model = HiggsModel(lam=args.lam, xi=args.xi)
    model.x0 = args.phi0
    model.y0 = args.y0
    model.T_max = args.T_max
    model.bg_steps = args.bg_steps
    T_span_bg = np.linspace(0.0, model.T_max, model.bg_steps)
    bg_sol, derived = save_background(model, T_span_bg, configs_dir, run_label)
    print(f"     Done in {time.time() - t0:.1f}s")

    print("\n  2. P_S(k) pipeline (high-resolution weighted grid)...")
    t0 = time.time()
    k_grid = build_weighted_kgrid(
        k_min=1e-5, k_max=1.0, k_pivot_phys=k_pivot_phys,
        n_dense=200, n_outer=100,
    )
    result = run_pspectrum_pipeline(
        model=model,
        phi0=args.phi0,
        y0=args.y0,
        N_star=args.nstar,
        k_phys_grid=k_grid,
        ms_steps=args.ms_steps,
        T_span_bg=T_span_bg,
        normalize_to_As=True,
        As=As,
        save_outputs=True,
        output_dir=pspectra_dir,
        n_workers=args.workers,
    )
    if result["status"] != "success":
        print(f"  FAILED: {result['message']}")
        sys.exit(1)
    print(f"     Done in {time.time() - t0:.1f}s")

    print(f"\n  3. Full CAMB C_ell computation (ell_max={args.camb_ell_max})...")
    t0 = time.time()
    camb_data = compute_camb(result, cell_dir, run_label, ell_max=args.camb_ell_max)
    print(f"     Done in {time.time() - t0:.1f}s")
    print(f"     chi2 (CAMB model) = {camb_data['chi2_model']:.2f},  chi2 (CAMB LCDM) = {camb_data['chi2_lcdm']:.2f},  Δ = {camb_data['chi2_model'] - camb_data['chi2_lcdm']:+.2f}")

    print("\n  4. Plotting...")
    t0 = time.time()
    plot_background(bg_sol, derived, filename=f"background_{run_label}")
    plot_ps(result["k_phys"], result["P_S"], label="Higgs USR",
            filename=f"ps_{run_label}")
    plot_dell(camb_data["ells"], camb_data["D_camb"],
              planck_ells=camb_data["planck_ells"],
              D_planck=camb_data["D_planck"],
              D_err_lower=camb_data["D_err_lower"],
              D_err_upper=camb_data["D_err_upper"],
              D_ell_lcdm=camb_data["D_pl"],
              ells_lcdm=camb_data["ells"],
              model_label="Higgs USR",
              filename=f"dell_{run_label}")
    plot_camb_comparison(camb_data, filename=f"camb_dell_{run_label}")
    plot_camb_fullsky(camb_data, filename=f"camb_fullsky_{run_label}")
    print(f"     Done in {time.time() - t0:.1f}s")

    print("\n  5. Collecting outputs...")
    t0 = time.time()
    chi2_model_val = camb_data["chi2_model"]
    chi2_lcdm_val = camb_data["chi2_lcdm"]
    collect_run_outputs(run_label, result, configs_dir, cell_dir, pspectra_dir,
                        diag_plots_dir, powerloss_plots_dir, chi2_model_val,
                        chi2_lcdm_val)
    print(f"     Done in {time.time() - t0:.1f}s")

    print_summary(result, bg_sol, derived, chi2_model_val,
                  chi2_model_val, args.camb_ell_max, configs_dir)


if __name__ == "__main__":
    main()
