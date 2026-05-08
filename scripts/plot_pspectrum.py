"""
Plot P_S(k) and D_ell from a cached spectrum JSON.

Usage:
  python scripts/plot_pspectrum.py outputs/simulations/pspectra/Punctuated_*.json
  python scripts/plot_pspectrum.py outputs/simulations/pspectra/Punctuated_*.json --save plots/
"""

import argparse
import glob
import os, sys

import numpy as np
import matplotlib.pyplot as plt
from scripts.pspectrum_pipeline import load_pspectrum
from scripts.sachs_wolfe import compute_cl_sw, compute_cl_sw_powerlaw
from scripts.constants import T_cmb


def plot_pspectrum(path, save_dir=None):
    data = load_pspectrum(path)
    k = data["k_phys"]
    ps = data["P_S"]
    meta = data["metadata"]

    ells, C_ell = compute_cl_sw(data, ell_max=30)
    D_ell = ells * (ells + 1) / (2 * np.pi) * C_ell * T_cmb**2 * 1e12

    ns = meta.get("ns", 0.965)
    _, C_ell_pl, _ = compute_cl_sw_powerlaw(ns=ns)
    D_ell_pl = ells * (ells + 1) / (2 * np.pi) * C_ell_pl * T_cmb**2 * 1e12

    from scripts.planck_data import get_planck_data_asymmetric
    ells_pl, D_pl, D_err_lower, D_err_upper = get_planck_data_asymmetric()

    # Apply styling matching the golden publication figure
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.labelsize': 16,
        'legend.fontsize': 12,
    })

    model_name = "USR Higgs" if "Higgs" in meta.get("model", "") else "PI"
    phi0_str = meta.get('phi0', '?')
    y0_str = meta.get('y0', '?')
    model = meta.get("model", "?")
    Ntot = meta.get("N_total", "?")
    Npiv = meta.get("N_pivot", "?")
    
    title_str = (f"{model}    " +
                 f"$\\varphi_0={phi0_str}$    $y_0={y0_str}$    " +
                 f"$N_{{\\mathrm{{tot}}}}={Ntot}$    $N_{{\\mathrm{{pivot}}}}={Npiv:.1f}$")

    # ==========================================
    # FIGURE 1: P_S(k)
    # ==========================================
    fig1, ax1 = plt.subplots(figsize=(8, 5))

    ax1.loglog(k, ps, "b-", linewidth=2.0, label=r"$\mathcal{P}_\mathcal{S}(k)$ (PI)")
    ax1.axvline(meta.get("k_pivot_phys", 0.05), color="gray", ls="--", alpha=0.6,
                label=f"pivot $k_*={meta.get('k_pivot_phys', 0.05)}$ Mpc$^{{-1}}$")
    
    ax1.set_xlabel(r"$k$ [Mpc$^{-1}$]")
    ax1.set_ylabel(r"$\mathcal{P}_\mathcal{S}(k)$")
    ax1.legend()
    ax1.grid(True, which='major', color='#e0e0e0', lw=0.8)
    ax1.grid(True, which='minor', color='#f0f0f0', lw=0.4)
    fig1.suptitle(title_str, fontsize=12)
    fig1.tight_layout()

    # ==========================================
    # FIGURE 2: D_ell
    # ==========================================
    fig2, ax2 = plt.subplots(figsize=(8, 6))

    # Shaded region for l <= 5
    ax2.axvspan(1.5, 5.5, color='#e6e6fa', alpha=0.5)
    ax2.text(3.5, 200, r"$\ell \leq 5$", rotation=90, color='#7b68ee', 
             verticalalignment='center', horizontalalignment='center', fontsize=14, alpha=0.8)

    # Power-law
    ax2.plot(ells, D_ell_pl, color='#7f8c8d', lw=2.5, ls='--', 
             label=r"Power-law ($n_s = 0.965$, SW approx.)")
    
    # USR Higgs/PI model
    ax2.plot(ells, D_ell, color='#d62728', lw=2.5, 
             label=fr"{model_name} ($\phi_0 = {phi0_str},\ y_0 = {y0_str}$)")

    # Planck data
    ax2.errorbar(ells_pl, D_pl, yerr=[D_err_lower, D_err_upper],
                 fmt='o', color='#34495e', capsize=3, elinewidth=1.5, markersize=7, 
                 label=r"Planck 2018 low-$\ell$ TT (Commander)")

    ax2.set_yscale('log')
    ax2.set_xlim(1.5, 30)
    ax2.set_ylim(80, 2500)
    
    ax2.set_xlabel(r"$\ell$")
    ax2.set_ylabel(r"$D_\ell^{\,TT}$ [$\mu$K$^2$]")

    ax2.grid(True, which='major', color='#e0e0e0', lw=0.8)
    ax2.grid(True, which='minor', color='#f0f0f0', lw=0.4)
    ax2.legend(loc='lower right', framealpha=1.0)
    
    fig2.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(path))[0]
        
        out1 = os.path.join(save_dir, f"ps_{base}.png")
        fig1.savefig(out1, dpi=150, bbox_inches="tight")
        print(f"Saved: {out1}")
        
        out2 = os.path.join(save_dir, f"dell_{base}.png")
        fig2.savefig(out2, dpi=150, bbox_inches="tight")
        print(f"Saved: {out2}")
    else:
        plt.show()


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Plot P_S(k) + D_ell from cached JSON(s)")
    p.add_argument("pattern", help="Glob pattern for spectrum JSON file(s)")
    p.add_argument("--save", "-s", default=None,
                   help="Directory to save plots (default: show interactively)")
    args = p.parse_args()

    files = sorted(glob.glob(args.pattern))
    if not files:
        print(f"No files matching: {args.pattern}")
        sys.exit(1)

    for f in files:
        print(f"Plotting: {f}")
        plot_pspectrum(f, args.save)
