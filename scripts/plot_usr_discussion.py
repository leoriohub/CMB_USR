"""
Generate the two key figures for the USR physics discussion:
  Fig 1: P_S(k) — USR Higgs vs power-law spectrum with Planck n_s=0.965
  Fig 2: CMB D_ell — Sachs-Wolfe approximation vs Planck low-ell data

Usage:
    python scripts/plot_usr_discussion.py            # load cache if available
    python scripts/plot_usr_discussion.py --recompute # force fresh pipeline run

Outputs: outputs/plots/powerloss/
    fig1_ps_usr_vs_powerlaw.png
    fig2_dell_usr_vs_planck.png
"""

import glob
import os
import sys

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(ROOT_DIR)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.pspectrum_pipeline import (
    run_pspectrum_pipeline,
    build_weighted_kgrid,
    load_pspectrum,
)
from scripts.sachs_wolfe import compute_cl_sw, compute_cl_sw_powerlaw
from scripts.planck_data import get_planck_data
from scripts.constants import As, k_pivot_phys, r_ls, T_cmb
from models import HiggsModel

OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs/plots/powerloss")
CACHE_DIR = os.path.join(ROOT_DIR, "outputs/simulations/pspectra")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Configuration ──────────────────────────────────────────────────────────
PHI0 = 5.76
Y0 = -0.12
XI = 15000.0
LAM = 0.13
N_WORKERS = max(1, os.cpu_count() // 2)

RECOMPUTE = "--recompute" in sys.argv

model = HiggsModel(lam=LAM, xi=XI)

# ── Load or compute P_S(k) ────────────────────────────────────────────────
pattern = f"*_phi{PHI0:.2f}_y0{Y0:.3f}_run_*.json"
matches = sorted(glob.glob(os.path.join(CACHE_DIR, pattern)),
                 key=os.path.getmtime, reverse=True)

if matches and not RECOMPUTE:
    cache_path = matches[0]
    print(f"Loading cached P_S(k) from {cache_path}...")
    cached = load_pspectrum(cache_path)
    k_phys = cached["k_phys"]
    P_S = cached["P_S"]
    meta = cached["metadata"]
    print(f"  {len(k_phys)} k-modes loaded")
    if len(k_phys) < 200:
        print("  Low-resolution cache — adding --recompute for denser grid")
else:
    if RECOMPUTE:
        print("Forcing recompute...")
    n_dense = 200
    n_outer = 100
    k_grid = build_weighted_kgrid(
        k_min=1e-5, k_max=1.0, k_pivot_phys=k_pivot_phys,
        n_dense=n_dense, n_outer=n_outer,
    )
    print(f"Running P_S(k) pipeline for phi0={PHI0}, y0={Y0} "
          f"({len(k_grid)} modes)...")
    result = run_pspectrum_pipeline(
        model=model, phi0=PHI0, y0=Y0,
        k_phys_grid=k_grid,
        normalize_to_As=True, As=As,
        n_workers=N_WORKERS,
    )
    if result["status"] != "success":
        print(f"FAILED: {result['message']}")
        sys.exit(1)
    k_phys = result["k_phys"]
    P_S = result["P_S"]
    meta = result["metadata"]
print(f"  N_total = {meta['N_total']:.2f},  N_pivot = {meta['N_pivot']:.2f}")
print(f"  {len(k_phys)} k-modes")

# ── Power-law LCDM spectrum ────────────────────────────────────────────────
ns_lcdm = 0.965
P_S_lcdm = As * (k_phys / k_pivot_phys) ** (ns_lcdm - 1.0)

# ── Sachs-Wolfe C_ell ──────────────────────────────────────────────────────
ell_max = 29
print("Computing C_ell via Sachs-Wolfe...")
ells, C_ell_usr = compute_cl_sw(
    {"k_phys": k_phys, "P_S": P_S}, ell_max=ell_max, r_ls=r_ls
)
_, C_ell_lcdm, P_S_lcdm_dense = compute_cl_sw_powerlaw(
    As=As, ns=ns_lcdm, k_pivot=k_pivot_phys, ell_max=ell_max, r_ls=r_ls
)

# Convert dimensionless C_ell to D_ell [muK^2]
T_factor = (T_cmb * 1e6) ** 2
D_ell_usr = C_ell_usr * ells * (ells + 1.0) * T_factor / (2.0 * np.pi)
D_ell_lcdm = C_ell_lcdm * ells * (ells + 1.0) * T_factor / (2.0 * np.pi)

# Planck low-ell data
planck_ells, planck_D, planck_err = get_planck_data()

# ── Style ───────────────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 16,
    "axes.labelsize": 20,
    "axes.titlesize": 22,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 17,
    "figure.dpi": 150,
})

COL_USR = "#d62728"
COL_LCDM = "#2c3e50"
COL_DIP = "darkred"

# ── Figure 1: P_S(k) ───────────────────────────────────────────────────────
print("Plotting Figure 1: P_S(k)...")
fig1, ax1 = plt.subplots(figsize=(8, 6))

ax1.loglog(k_phys, P_S_lcdm, "-", color=COL_LCDM, lw=2, alpha=0.7,
           label=r"Power-law ($n_s = 0.965$)")
ax1.loglog(k_phys, P_S, "-", color=COL_USR, lw=2,
             label=rf"USR Higgs ($\phi_0={PHI0},\ y_0 = {Y0})$")

k_dip = k_phys[np.argmin(P_S[k_phys < 1.0])]
P_dip = P_S[k_phys < 1.0][np.argmin(P_S[k_phys < 1.0])]
ax1.axvline(k_dip, color=COL_DIP, ls=":", lw=1.5, alpha=0.5)
ax1.annotate(
    rf"$k_{{\rm dip}} = {k_dip:.2e}$",
    xy=(k_dip, P_dip),
    xytext=(0.55, 0.7), textcoords="axes fraction",
    arrowprops=dict(arrowstyle="->", color=COL_DIP, lw=1.5),
    fontsize=15, color=COL_DIP,
)

ax1.axvline(k_pivot_phys, color="gray", ls="--", lw=1.5, alpha=0.4)
ax1.annotate(
    r"$k_* = 0.05$",
    xy=(k_pivot_phys, As * 0.6),
    fontsize=14, color="gray", ha="center",
)

ax1.axhline(As, color="gray", ls="--", lw=1.5, alpha=0.3)
ax1.text(0.98, 0.95, r"$A_s$", transform=ax1.transAxes,
         fontsize=14, color="gray", ha="right", va="top")

ax1.set_xlabel(r"$k\ [{\rm Mpc}^{-1}]$")
ax1.set_ylabel(r"$\mathcal{P}_{\mathcal{R}}(k)$")
ax1.legend()
ax1.grid(True, alpha=0.25, which="both")

fig1.savefig(os.path.join(OUTPUT_DIR, "fig1_ps_usr_vs_powerlaw.png"),
            dpi=300, bbox_inches="tight")
print(f"  Saved: {os.path.join(OUTPUT_DIR, 'fig1_ps_usr_vs_powerlaw.png')}")

# ── Figure 2: D_ell ──────────────────────────────────────────────────────
print("Plotting Figure 2: D_ell...")
fig2, ax2 = plt.subplots(figsize=(8.5, 6))

ax2.errorbar(
    planck_ells, planck_D, yerr=planck_err,
    fmt="o", color="#2c3e50", capsize=4, capthick=1.5,
    markersize=7, elinewidth=1.5,
    label=r"Planck 2018 low-$\ell$ TT (Commander)",
)

ax2.semilogy(ells, D_ell_usr, "-", color=COL_USR, lw=2.5,
           label=rf"USR Higgs ($\phi_0={PHI0},\ y_0 = {Y0})$")
ax2.semilogy(ells, D_ell_lcdm, "--", color=COL_LCDM, lw=2, alpha=0.7,
             label=r"Power-law ($n_s = 0.965$, SW approx.)")

ax2.set_xlabel(r"$\ell$")
ax2.set_ylabel(r"$D_\ell^{\,TT}\ [\mu{\rm K}^2]$")
ax2.legend()
ax2.grid(True, alpha=0.25, which="both")

fig2.savefig(os.path.join(OUTPUT_DIR, "fig2_dell_usr_vs_planck.png"),
             dpi=300, bbox_inches="tight")
print(f"  Saved: {os.path.join(OUTPUT_DIR, 'fig2_dell_usr_vs_planck.png')}")

plt.close("all")
print("\nDone. Both figures saved to outputs/plots/powerloss/")
