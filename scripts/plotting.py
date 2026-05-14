"""Plotting utilities for CMB anomaly analysis.

All functions follow publication-ready conventions:
- Two-column format (~3.25-3.5in wide single, ~7in full)
- 300 DPI minimum
- Colorblind-friendly palette (Tol 2012)
- Big fonts: axis >= 14pt, ticks >= 12pt, legend >= 11pt
- Export PNG only
"""
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.constants import As, k_pivot_phys, T_cmb, ROOT_DIR
from scipy.interpolate import interp1d

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

OUTPUT_DIRS = {
    "diagnostics": os.path.join(ROOT_DIR, "outputs/plots/diagnostics"),
    "powerloss": os.path.join(ROOT_DIR, "outputs/plots/powerloss"),
}


def _ensure_dir(subdir):
    os.makedirs(OUTPUT_DIRS.get(subdir, subdir), exist_ok=True)
    return OUTPUT_DIRS.get(subdir, subdir)


def _save_fig(fig, path_base, subdir="diagnostics"):
    out_dir = _ensure_dir(subdir)
    path = os.path.join(out_dir, f"{path_base}.png")
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


def plot_ps(k_phys, P_S, label="Model", filename="ps", subdir="powerloss",
            show_lcdm=True, k_dip=None, k_pivot=True):
    """Plot primordial power spectrum with optional LCDM baseline."""
    mask = np.isfinite(P_S)
    if np.sum(mask) > 5:
        logk_interp = interp1d(np.log(k_phys[mask]), P_S[mask], kind="cubic",
                               bounds_error=False, fill_value="extrapolate")
        k_dense = np.logspace(np.log10(k_phys[mask].min()), np.log10(k_phys[mask].max()), 1000)
        ps_dense = np.clip(logk_interp(np.log(k_dense)), 0, None)
    else:
        k_dense, ps_dense = k_phys, P_S

    fig, ax = plt.subplots(figsize=(3.35, 2.6))

    ax.loglog(k_dense, ps_dense, "-", color=TOL["red"], lw=1.5, label=label)

    if show_lcdm:
        ns_lcdm = 0.965
        ps_lcdm = As * (k_dense / k_pivot_phys) ** (ns_lcdm - 1.0)
        ax.loglog(k_dense, ps_lcdm, "-", color=TOL["dark"], lw=1.2, alpha=0.6,
                  label=r"$\Lambda$CDM")

    if k_dip is not None and k_dip > 0:
        ps_dip = float(np.interp(k_dip, k_phys, P_S))
        ax.axvline(k_dip, color=TOL["red"], ls=":", lw=1.5, alpha=0.5)
        ax.annotate(f"$k_{{dip}}={k_dip:.2e}$",
                    xy=(k_dip, ps_dip),
                    xytext=(0.55, 0.7), textcoords="axes fraction",
                    arrowprops=dict(arrowstyle="->", color=TOL["red"], lw=1.5),
                    fontsize=10, color=TOL["red"])

    if k_pivot:
        ax.axvline(k_pivot_phys, color=TOL["grey"], ls="--", lw=1, alpha=0.4)

    ax.set_xlabel(r"$k\ [{\rm Mpc}^{-1}]$", fontsize=10)
    ax.set_ylabel(r"$\mathcal{P}_{\mathcal{R}}(k)$", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, which="both")

    fig.tight_layout()
    _save_fig(fig, filename, subdir)


def plot_dell(ells, D_ell_model, planck_ells=None, D_planck=None,
              D_err_lower=None, D_err_upper=None, D_ell_lcdm=None,
              ells_lcdm=None, model_label="Model", filename="dell",
              subdir="powerloss", ell_max=30):
    """Plot D_ell with Planck data and optional LCDM baseline."""
    fig, ax = plt.subplots(figsize=(3.7, 2.6))

    if planck_ells is not None and D_planck is not None:
        mask = planck_ells <= ell_max
        yerr = [D_err_lower[mask], D_err_upper[mask]] if D_err_upper is not None else None
        ax.errorbar(planck_ells[mask], D_planck[mask], yerr=yerr,
                    fmt="o", color=TOL["dark"], capsize=3, capthick=1,
                    markersize=4, elinewidth=1,
                    label=r"Planck 2018 low-$\ell$ TT")

    ell_dense = np.linspace(ells.min(), min(ells.max(), ell_max), 200)
    D_interp = interp1d(ells, D_ell_model, kind="cubic")(ell_dense)
    ax.plot(ell_dense, D_interp, "-", color=TOL["red"], lw=1.5, label=model_label)

    if ells_lcdm is not None and D_ell_lcdm is not None:
        mask = ells_lcdm <= ell_max
        D_lcdm_interp = interp1d(ells_lcdm[mask], D_ell_lcdm[mask], kind="cubic")(ell_dense)
        ax.plot(ell_dense, D_lcdm_interp, "--", color=TOL["grey"], lw=1.2,
                alpha=0.6, label=r"$\Lambda$CDM")

    ax.set_xlabel(r"$\ell$", fontsize=10)
    ax.set_ylabel(r"$D_\ell^{TT}\ [\mu{\rm K}^2]$", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, which="both")
    ax.set_xlim(1.5, ell_max + 0.5)

    fig.tight_layout()
    _save_fig(fig, filename, subdir)


def plot_background(bg_sol, derived, filename="background", subdir="diagnostics"):
    """4-panel background trajectory dashboard."""
    x, y, z, n = bg_sol
    N = derived["N"]
    epsH = derived["epsH"]
    etaH = derived["etaH"]

    fig, axes = plt.subplots(2, 2, figsize=(7, 5.5))

    ax = axes[0, 0]
    ax.plot(N, x, color=TOL["blue"], lw=1.5)
    ax.set_xlabel(r"$N$ (e-folds)", fontsize=10)
    ax.set_ylabel(r"$\phi$ [$M_P$]", fontsize=10)
    ax.grid(True, alpha=0.25, which="both")

    ax = axes[0, 1]
    ax.semilogy(N, epsH, color=TOL["red"], lw=1.5)
    ax.axhline(1.0, color=TOL["grey"], ls="--", lw=1, alpha=0.5)
    ax.text(0.98, 0.95, r"$\epsilon_H = 1$", transform=ax.transAxes,
            color=TOL["grey"], ha="right", va="top", fontsize=9)
    ax.set_xlabel(r"$N$ (e-folds)", fontsize=10)
    ax.set_ylabel(r"$\epsilon_H$", fontsize=10)
    ax.grid(True, alpha=0.25, which="both")

    ax = axes[1, 0]
    ax.plot(N, etaH, color=TOL["green"], lw=1.5)
    ax.axhline(0.0, color=TOL["grey"], ls="--", lw=1, alpha=0.5)
    ax.set_xlabel(r"$N$ (e-folds)", fontsize=10)
    ax.set_ylabel(r"$\eta_H$", fontsize=10)
    ax.grid(True, alpha=0.25, which="both")

    ax = axes[1, 1]
    ax.plot(x, y, color=TOL["yellow"], lw=1.5)
    ax.set_xlabel(r"$\phi$ [$M_P$]", fontsize=10)
    ax.set_ylabel(r"$d\phi/dT$ [code units]", fontsize=10)
    ax.grid(True, alpha=0.25, which="both")

    fig.tight_layout()
    _save_fig(fig, filename, subdir)


def plot_camb_comparison(camb_data, filename="camb_dell",
                         subdir="powerloss"):
    """CAMB D_ell vs LCDM + Planck at low ell."""
    ells = camb_data["ells"]
    D_camb = camb_data["D_camb"]
    D_pl = camb_data["D_pl"]
    p_ells = camb_data["planck_ells"]
    D_p = camb_data["D_planck"]
    D_err_lo = camb_data["D_err_lower"]
    D_err_hi = camb_data["D_err_upper"]

    low = ells <= 30
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    ax.errorbar(p_ells, D_p, yerr=[D_err_lo, D_err_hi],
                fmt="o", color=TOL["dark"], capsize=3, capthick=1,
                markersize=4, elinewidth=1, label="Planck 2018", zorder=5)
    ax.plot(ells[low], D_camb[low], "-", color=TOL["blue"], lw=1.5,
            label="CAMB (full)", zorder=4)
    ax.plot(ells[low], D_pl[low], "--", color=TOL["grey"], lw=1.2,
            label=r"$\Lambda$CDM (CAMB)", zorder=2)

    ax.set_xlabel(r"$\ell$", fontsize=10)
    ax.set_ylabel(r"$D_\ell^{TT}\ [\mu{\rm K}^2]$", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, which="both")
    ax.set_xlim(1.5, 31)

    fig.tight_layout()
    _save_fig(fig, filename, subdir)


def plot_camb_fullsky(camb_data, filename="camb_fullsky", subdir="powerloss"):
    """Full-sky CAMB D_ell plot."""
    ells = camb_data["ells"]
    D_camb = camb_data["D_camb"]
    D_pl = camb_data["D_pl"]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(ells, D_camb, "-", color=TOL["red"], lw=1.2, label="Model (CAMB)")
    ax.semilogy(ells, D_pl, "--", color=TOL["dark"], lw=1.2, alpha=0.6,
                label=r"$\Lambda$CDM")
    ax.set_xlabel(r"$\ell$", fontsize=14)
    ax.set_ylabel(r"$D_\ell^{TT}\ [\mu{\rm K}^2]$", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.25, which="both")
    ax.set_xlim(1.5, ells.max())

    fig.tight_layout()
    _save_fig(fig, filename, subdir)



