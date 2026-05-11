"""
Plot P_S(k) and D_ell from a cached spectrum JSON.

Follows output conventions of run_full_analysis.py (Tol colorblind-safe palette,
column-width figures, serif fonts, 300 DPI PNG + PDF).

Usage:
  python scripts/plot_pspectrum.py <spectrum.json> [--save <dir>]
  python scripts/plot_pspectrum.py "outputs/simulations/pspectra/Punctuated_*.json"
"""

import argparse
import glob
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from scripts.pspectrum_pipeline import load_pspectrum
from scripts.sachs_wolfe import compute_cl_sw, compute_cl_sw_powerlaw
from scripts.constants import T_cmb, As, k_pivot_phys
from scripts.planck_data import get_planck_data_asymmetric

# ── Publication-ready style (matches run_full_analysis.py) ──────────────────
plt.rcParams.update({
    "font.size": 8,
    "axes.labelsize": 8,
    "axes.titlesize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "legend.fontsize": 5,
    "figure.dpi": 300,
    "font.family": "serif",
})

# Tol colorblind-safe palette + user overrides
TOL = {
    "blue": "#4477AA",
    "red": "#CC3311",
    "green": "#228833",
    "yellow": "#EE8866",
    "teal": "#44BB99",
    "purple": "#AA3377",
    "grey": "#666666",
    "dark": "#222222",
}
C_USR = "#2A9D8F"   # user teal for USR model
C_LCDM = "#D73959"   # user red for power-law baseline
C_PLANCK = "#000000" # black for Planck data


def plot_pspectrum(path, save_dir=None):
    data = load_pspectrum(path)
    k = data["k_phys"]
    ps = data["P_S"]
    meta = data["metadata"]

    # ── C_ell / D_ell ────────────────────────────────────────────────────
    ells, C_ell = compute_cl_sw(data, ell_max=30)
    D_ell = ells * (ells + 1) / (2 * np.pi) * C_ell * T_cmb**2 * 1e12

    # ── LCDM power-law reference ──────────────────────────────────────────
    ns_lcdm = 0.965
    k_pl = np.logspace(-5, 0, 500)
    P_S_pl = As * (k_pl / k_pivot_phys) ** (ns_lcdm - 1.0)

    _, C_ell_pl_sw, _ = compute_cl_sw_powerlaw(ns=ns_lcdm)
    D_ell_pl = ells * (ells + 1) / (2 * np.pi) * C_ell_pl_sw * T_cmb**2 * 1e12

    # ── Planck low-ell data ──────────────────────────────────────────────
    ells_pl, D_pl, D_err_lower, D_err_upper = get_planck_data_asymmetric()

    # ── Metadata ─────────────────────────────────────────────────────────
    phi0_str = meta.get("phi0", "?")
    y0_str = meta.get("y0", "?")
    model_name = meta.get("model", "?")
    Ntot = meta.get("N_total", "?")
    Npiv = meta.get("N_pivot", "?")

    # =====================================================================
    # FIGURE 1:  P_S(k)
    # =====================================================================
    fig1, ax1 = plt.subplots(figsize=(3.35, 2.6))

    ax1.loglog(k_pl, P_S_pl, "-", color=C_LCDM, lw=1.2, alpha=0.7,
               label=r"Power-law ($n_s=0.965$)")
    ax1.loglog(k, ps, "-", color=C_USR, lw=1.5,
               label=rf"Higgs USR ($x_0={phi0_str},\ y_0={y0_str}$)")

    ax1.set_xlabel(r"$k\ [{\rm Mpc}^{-1}]$")
    ax1.set_ylabel(r"$\mathcal{P}_{\mathcal{R}}(k)$")
    ax1.legend(loc="lower right")
    ax1.grid(True, alpha=0.25, which="both")

    fig1.tight_layout()

    # =====================================================================
    # FIGURE 2:  D_ell  (Sachs-Wolfe, ℓ ≤ 30)
    # =====================================================================
    fig2, ax2 = plt.subplots(figsize=(3.7, 2.6))

    ax2.errorbar(ells_pl, D_pl, yerr=[D_err_upper, D_err_lower],
                 fmt="o", color=C_PLANCK, capsize=3, capthick=1,
                 markersize=4, elinewidth=1,
                 label=r"Planck 2018 low-$\ell$ TT")
    ax2.plot(ells, D_ell_pl, "--", color=C_LCDM, lw=1.2, alpha=0.7,
                 label=r"Power-law ($n_s=0.965$)")
    ax2.plot(ells, D_ell, "-", color=C_USR, lw=1.5,
                 label=rf"Higgs USR ($x_0={phi0_str},\ y_0={y0_str}$)")

    ax2.set_xlabel(r"$\ell$")
    ax2.set_ylabel(r"$D_\ell^{\,TT}\ [\mu{\rm K}^2]$")
    ax2.set_ylim(bottom=0)
    ax2.legend(loc="lower right")
    ax2.grid(True, alpha=0.25, which="both")

    fig2.tight_layout()

    # ── Save ─────────────────────────────────────────────────────────────
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(path))[0]

        for stem, fig in [(f"ps_{base}", fig1), (f"dell_{base}", fig2)]:
            for ext in ["png", "pdf"]:
                out = os.path.join(save_dir, f"{stem}.{ext}")
                fig.savefig(out, dpi=300, bbox_inches="tight")
                print(f"Saved: {out}")
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
