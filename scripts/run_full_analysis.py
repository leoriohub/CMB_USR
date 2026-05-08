"""
Self-contained full analysis pipeline for a single Higgs USR configuration.

Exercises all codebase capabilities:
  - Background integration + derived quantities
  - P_S(k) with high-resolution weighted k-grid
  - CMB D_ell via Sachs-Wolfe + LCDM baseline + Planck data
  - Publication-ready plots (PS, D_ell, background dashboard)
  - All outputs in a single run-specific subfolder

Usage:
  python scripts/run_full_analysis.py --phi0 6.6 --y0 -0.67 --nstar 55.76
  python scripts/run_full_analysis.py --phi0 6.0 --y0 -0.538 --nstar 37.3  # deepest dip
"""

import argparse
import json
import os
import sys
import time

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import inf_dyn_background as bg_solver
from models import HiggsModel
from scripts.constants import As, k_pivot_phys, r_ls, T_cmb
from scripts.pspectrum_pipeline import (
    run_pspectrum_pipeline,
    build_weighted_kgrid,
    ensure_k_pivot,
)
from scripts.sachs_wolfe import compute_cl_sw, compute_cl_sw_powerlaw
from scripts.planck_data import get_planck_data_asymmetric


def save_background(model, T_span_bg, output_dir):
    """Run background integration and save trajectory + derived quantities."""
    bg_sol = bg_solver.run_background_simulation(model, T_span_bg)
    derived = bg_solver.get_derived_quantities(bg_sol, model)

    x, y, z, n = bg_sol
    record = {
        "metadata": {
            "model": model.name,
            "phi0": float(model.phi0),
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

    path = os.path.join(output_dir, "background.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"  Saved: {path}")
    return bg_sol, derived


def save_cell(result, output_dir):
    """Compute C_ell and D_ell, compare with Planck and LCDM baseline."""
    ell_max = 29
    ells, C_ell = compute_cl_sw(result, ell_max=ell_max, r_ls=r_ls)
    D_ell = ells * (ells + 1) / (2 * np.pi) * C_ell * T_cmb ** 2 * 1e12

    _, C_ell_pl, P_S_lcdm_dense = compute_cl_sw_powerlaw(
        As=As, ns=0.965, k_pivot=k_pivot_phys, ell_max=ell_max, r_ls=r_ls
    )
    D_ell_pl = ells * (ells + 1) / (2 * np.pi) * C_ell_pl * T_cmb ** 2 * 1e12

    planck_ells, D_planck, D_err_lower, D_err_upper = get_planck_data_asymmetric()

    def chi2_model(D_model):
        chi2 = 0.0
        for i, ell_val in enumerate(planck_ells):
            idx = np.argmin(np.abs(ells - ell_val))
            residual = D_model[idx] - D_planck[i]
            sigma = D_err_upper[i] if residual > 0 else D_err_lower[i]
            chi2 += (residual / sigma) ** 2
        return chi2

    chi2_model_val = chi2_model(D_ell)
    chi2_lcdm_val = chi2_model(D_ell_pl)

    record = {
        "metadata": {"ell_max": ell_max, "r_ls": r_ls, "T_cmb": T_cmb},
        "model": {
            "ells": ells.tolist(),
            "C_ell": C_ell.tolist(),
            "D_ell": D_ell.tolist(),
            "chi2": chi2_model_val,
        },
        "lcdm": {
            "ells": ells.tolist(),
            "C_ell": C_ell_pl.tolist(),
            "D_ell": D_ell_pl.tolist(),
            "ns": 0.965,
            "chi2": chi2_lcdm_val,
        },
        "planck_data": {
            "ells": planck_ells.tolist(),
            "D_ell": D_planck.tolist(),
            "D_ell_err_lower": D_err_lower.tolist(),
            "D_ell_err_upper": D_err_upper.tolist(),
        },
    }

    path = os.path.join(output_dir, "cell.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2)
    print(f"  Saved: {path}")
    return ells, D_ell, D_ell_pl, planck_ells, D_planck, D_err_lower, D_err_upper, chi2_model_val, chi2_lcdm_val


def plot_background(bg_sol, derived, output_dir):
    """4-panel background dashboard."""
    x, y, z, n = bg_sol
    N = derived["N"]
    epsH = derived["epsH"]
    etaH = derived["etaH"]

    COL_PHI = "#1f77b4"
    COL_EPS = "#d62728"
    COL_ETA = "#2ca02c"

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    ax.plot(N, x, color=COL_PHI, lw=2)
    ax.set_xlabel(r"$N$ (e-folds)", fontsize=20)
    ax.set_ylabel(r"$\phi$", fontsize=20)
    ax.grid(True, alpha=0.25, which="both")

    ax = axes[0, 1]
    ax.semilogy(N, epsH, color=COL_EPS, lw=2)
    ax.axhline(1.0, color="gray", ls="--", lw=1.5, alpha=0.5)
    ax.text(0.98, 0.95, r"$\epsilon_H = 1$", transform=ax.transAxes,
            fontsize=14, color="gray", ha="right", va="top")
    ax.set_xlabel(r"$N$ (e-folds)", fontsize=20)
    ax.set_ylabel(r"$\epsilon_H$", fontsize=20)
    ax.grid(True, alpha=0.25, which="both")

    ax = axes[1, 0]
    ax.plot(N, etaH, color=COL_ETA, lw=2)
    ax.axhline(0.0, color="gray", ls="--", lw=1.5, alpha=0.5)
    ax.set_xlabel(r"$N$ (e-folds)", fontsize=20)
    ax.set_ylabel(r"$\eta_H$", fontsize=20)
    ax.grid(True, alpha=0.25, which="both")

    ax = axes[1, 1]
    ax.plot(x, y, color="#8c564b", lw=2)
    ax.set_xlabel(r"$\phi$", fontsize=20)
    ax.set_ylabel(r"$d\phi/dT$", fontsize=20)
    ax.grid(True, alpha=0.25, which="both")

    fig.tight_layout()
    for ext in ["png", "pdf"]:
        path = os.path.join(output_dir, f"background.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def plot_ps(result, output_dir):
    """P_S(k) plot with power-law baseline and dip annotation."""
    k = result["k_phys"]
    ps = result["P_S"]
    meta = result["metadata"]

    ns_lcdm = 0.965
    ps_lcdm = As * (k / k_pivot_phys) ** (ns_lcdm - 1.0)

    k_dip = k[np.argmin(ps)]
    p_dip = np.min(ps)
    suppression = (1 - p_dip / As) * 100

    COL_USR = "#d62728"
    COL_LCDM = "#2c3e50"
    COL_DIP = "darkred"

    fig, ax = plt.subplots(figsize=(8, 6))

    ax.loglog(k, ps_lcdm, "-", color=COL_LCDM, lw=2, alpha=0.7,
              label=r"Power-law ($n_s = 0.965$)")
    ax.loglog(k, ps, "-", color=COL_USR, lw=2,
              label=rf"Higgs USR ($\phi_0={meta['phi0']},\ y_0 = {meta['y0']}$)")

    ax.axvline(k_dip, color=COL_DIP, ls=":", lw=1.5, alpha=0.5)
    ax.annotate(
        rf"$k_{{\rm dip}} = {k_dip:.2e}$",
        xy=(k_dip, p_dip),
        xytext=(0.55, 0.7), textcoords="axes fraction",
        arrowprops=dict(arrowstyle="->", color=COL_DIP, lw=1.5),
        fontsize=15, color=COL_DIP,
    )

    ax.axvline(k_pivot_phys, color="gray", ls="--", lw=1.5, alpha=0.4)
    ax.annotate(
        r"$k_* = 0.05$",
        xy=(k_pivot_phys, As * 0.6),
        fontsize=14, color="gray", ha="center",
    )

    ax.axhline(As, color="gray", ls="--", lw=1.5, alpha=0.3)
    ax.text(0.98, 0.95, r"$A_s$", transform=ax.transAxes,
            fontsize=14, color="gray", ha="right", va="top")

    ax.text(0.98, 0.05, rf"$S = {suppression:.0f}\%$",
            transform=ax.transAxes, fontsize=16, color=COL_DIP,
            ha="right", va="bottom",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    ax.set_xlabel(r"$k\ [{\rm Mpc}^{-1}]$", fontsize=20)
    ax.set_ylabel(r"$\mathcal{P}_{\mathcal{R}}(k)$", fontsize=20)
    ax.legend(fontsize=17)
    ax.grid(True, alpha=0.25, which="both")

    fig.tight_layout()
    for ext in ["png", "pdf"]:
        path = os.path.join(output_dir, f"ps.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def plot_dell(ells, D_ell_model, D_ell_pl, planck_ells, D_planck, D_err_lower, D_err_upper, chi2_model, chi2_lcdm, phi0, y0, output_dir):
    """D_ell plot with Planck data and LCDM baseline."""
    COL_USR = "#d62728"
    COL_LCDM = "#2c3e50"

    fig, ax = plt.subplots(figsize=(8.5, 6))

    ax.errorbar(
        planck_ells, D_planck, yerr=[D_err_lower, D_err_upper],
        fmt="o", color="#2c3e50", capsize=4, capthick=1.5,
        markersize=7, elinewidth=1.5,
        label=r"Planck 2018 low-$\ell$ TT (Commander)",
    )

    ax.semilogy(ells, D_ell_model, "-", color=COL_USR, lw=2.5,
                label=rf"Higgs USR ($\phi_0={phi0},\ y_0 = {y0}$)")
    ax.semilogy(ells, D_ell_pl, "--", color=COL_LCDM, lw=2, alpha=0.7,
                label=r"Power-law ($n_s = 0.965$, SW approx.)")

    ax.set_xlabel(r"$\ell$", fontsize=20)
    ax.set_ylabel(r"$D_\ell^{\,TT}\ [\mu{\rm K}^2]$", fontsize=20)
    ax.legend(fontsize=14, loc="lower right")
    ax.grid(True, alpha=0.25, which="both")

    chi2_str = f"$\\chi^2_{{\\rm USR}} = {chi2_model:.1f}$\n$\\chi^2_{{\\rm LCDM}} = {chi2_lcdm:.1f}$"
    ax.text(0.97, 0.97, chi2_str, transform=ax.transAxes,
            fontsize=15, va="top", ha="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8))

    fig.tight_layout()
    for ext in ["png", "pdf"]:
        path = os.path.join(output_dir, f"dell.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        print(f"  Saved: {path}")
    plt.close(fig)


def print_summary(result, bg_sol, derived, chi2_model_val, output_dir):
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
    print(f"  phi0:           {meta.get('phi0', '?'):.2f}")
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
    print(f"  chi2 (vs Planck): {chi2_model_val:.2f}")
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
    p.add_argument("--bg-steps", type=int, default=10000, help="Background integration steps")
    p.add_argument("--T-max", type=float, default=5000.0, help="Max conformal time")
    p.add_argument("--workers", type=int, default=os.cpu_count(),
                   help="Parallel workers for MS integration (default: all cores)")
    p.add_argument("--output-dir", type=str, default=None,
                   help="Output directory (default: auto-generated)")
    return p.parse_args()


def main():
    args = parse_args()

    output_dir = args.output_dir
    if output_dir is None:
        safe = f"run_phi{args.phi0:.2f}_y0{args.y0:.3f}"
        output_dir = os.path.join(ROOT_DIR, "outputs", "simulations", safe)
    os.makedirs(output_dir, exist_ok=True)
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)

    plt.rcParams.update({
        "font.size": 16,
        "axes.labelsize": 20,
        "axes.titlesize": 22,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 17,
        "figure.dpi": 150,
        "font.family": "serif",
    })

    print("\n  1. Background integration...")
    t0 = time.time()
    model = HiggsModel(lam=args.lam, xi=args.xi)
    T_span_bg = np.linspace(0.0, args.T_max, args.bg_steps)
    bg_sol, derived = save_background(model, T_span_bg, output_dir)
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
        output_dir=output_dir,
        n_workers=args.workers,
    )
    if result["status"] != "success":
        print(f"  FAILED: {result['message']}")
        sys.exit(1)
    print(f"     Done in {time.time() - t0:.1f}s")

    print("\n  3. C_ell / D_ell computation...")
    t0 = time.time()
    ells, D_ell_model, D_ell_pl, planck_ells, D_planck, D_err_lower, D_err_upper, chi2_model_val, chi2_lcdm_val = save_cell(result, output_dir)
    print(f"     Done in {time.time() - t0:.1f}s")
    print(f"     chi2 (model) = {chi2_model_val:.2f},  chi2 (LCDM) = {chi2_lcdm_val:.2f}")

    print("\n  4. Plotting...")
    t0 = time.time()
    plot_background(bg_sol, derived, plots_dir)
    plot_ps(result, plots_dir)
    plot_dell(ells, D_ell_model, D_ell_pl, planck_ells, D_planck, D_err_lower, D_err_upper,
              chi2_model_val, chi2_lcdm_val, args.phi0, args.y0, plots_dir)
    print(f"     Done in {time.time() - t0:.1f}s")

    print_summary(result, bg_sol, derived, chi2_model_val, output_dir)


if __name__ == "__main__":
    main()
