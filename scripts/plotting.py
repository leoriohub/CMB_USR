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
    "optimizer": os.path.join(ROOT_DIR, "outputs/plots/optimizer"),
    "paper": os.path.join(ROOT_DIR, "outputs/plots/paper"),
    "pspectra": os.path.join(ROOT_DIR, "outputs/simulations/pspectra"),
    "c_ell": os.path.join(ROOT_DIR, "outputs/simulations/c_ell"),
    "configs": os.path.join(ROOT_DIR, "outputs/simulations/configs"),
    "logs": os.path.join(ROOT_DIR, "outputs/simulations/logs"),
    "scans": os.path.join(ROOT_DIR, "outputs/simulations/scans"),
}


def get_path(category, filename):
    """Return full path for an output file. Creates directory if needed."""
    out_dir = OUTPUT_DIRS.get(category)
    if out_dir is None:
        raise ValueError(
            f"Unknown output category: {category}. "
            f"Available: {list(OUTPUT_DIRS.keys())}"
        )
    os.makedirs(out_dir, exist_ok=True)
    return os.path.join(out_dir, filename)


def save_fig(fig, filename, category="diagnostics", dpi=300):
    """Save a matplotlib figure to the correct output directory."""
    if not filename.endswith(".png"):
        filename += ".png"
    path = get_path(category, filename)
    fig.savefig(path, dpi=dpi, bbox_inches="tight")
    print(f"  Saved: {path}")
    plt.close(fig)


def make_filename(name, phi0=None, y0=None, nstar=None, ext=".json"):
    """Generate standardized output filename.

    Pattern: {name}_phi{phi0:.2f}_y0{y0:+.3f}_nstar{nstar:.1f}{ext}
    If phi0 is None, returns {name}{ext} (for special files like camb_lcdm).

    Examples:
        make_filename("ps", 6.60, -0.736, 52.6)         → "ps_phi6.60_y0-0.736_nstar52.6.json"
        make_filename("camb", 6.60, -0.736, 52.6)        → "camb_phi6.60_y0-0.736_nstar52.6.json"
        make_filename("camb_lcdm")                        → "camb_lcdm.json"
        make_filename("planck", 6.60, -0.736, 52.6, ".png") → "planck_phi6.60_y0-0.736_nstar52.6.png"
    """
    if phi0 is not None:
        return f"{name}_phi{phi0:.2f}_y0{y0:+.3f}_nstar{nstar:.1f}{ext}"
    return f"{name}{ext}"


def find_ps(phi0, y0, nstar, tolerance=3.0):
    """Find cached P_S(k) JSON matching config params.

    Returns (path, metadata) or (None, None) if not found.
    Matches by phi0, y0, and N_star within tolerance.
    Tries new convention first, then legacy patterns.
    """
    import glob
    import json

    ps_dir = get_path("pspectra", "")

    # New convention
    pattern = os.path.join(ps_dir, f"ps_phi{phi0:.2f}_y0{y0:+.3f}_nstar*.json")
    matches = sorted(glob.glob(pattern))

    # Legacy patterns for backward compat during transition
    if not matches:
        for pat in [
            os.path.join(ps_dir, f"PS_Higgs*phi{phi0:.2f}_y0{y0:.3f}_*.json"),
            os.path.join(ps_dir, f"Higgs_Inflation*phi{phi0:.2f}_y0{y0:.3f}_*.json"),
        ]:
            matches = sorted(glob.glob(pat))
            if matches:
                break

    if not matches:
        return None, None

    # Score by N_star proximity
    scored = []
    for m in matches:
        try:
            with open(m) as f:
                rec = json.load(f)
            md = rec.get("metadata", {})
            ns = md.get("N_star", 0)
            if abs(ns - nstar) <= tolerance:
                scored.append((abs(ns - nstar), m, md))
        except Exception:
            continue

    if not scored:
        return None, None
    scored.sort(key=lambda x: x[0])
    return scored[0][1], scored[0][2]


def plot_ps(k_phys, P_S, label="Model", filename="ps", category="powerloss",
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
    save_fig(fig, filename, category)


def plot_dell(ells, D_ell_model, planck_ells=None, D_planck=None,
              D_err_lower=None, D_err_upper=None, D_ell_lcdm=None,
              ells_lcdm=None, model_label="Model", filename="dell",
              category="powerloss", ell_max=30):
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
    save_fig(fig, filename, category)


def plot_background(bg_sol, derived, filename="background", category="diagnostics"):
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
    save_fig(fig, filename, category)


def plot_camb_comparison(camb_data, filename="camb_dell",
                         category="powerloss"):
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
    save_fig(fig, filename, category)


def plot_camb_fullsky(camb_data, filename="camb_fullsky", category="powerloss"):
    """
    Broken-axis full-sky D_ell plot with Planck 2018 data.

    Left panel: log x-scale (ell=2-30) for low-ell Commander data.
    Right panel: linear x-scale (ell=32-2500) for binned TT.
    """
    ells = camb_data["ells"]
    D_camb = camb_data["D_camb"]
    D_pl = camb_data["D_pl"]
    ells_lcdm = camb_data["ells_lcdm"]
    p_ells = camb_data["planck_ells"]
    D_p = camb_data["D_planck"]
    D_lo = camb_data["D_err_lower"]
    D_hi = camb_data["D_err_upper"]

    fig = plt.figure(figsize=(7, 3.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 4], wspace=0)
    ax_left = fig.add_subplot(gs[0])
    ax_right = fig.add_subplot(gs[1], sharey=ax_left)

    ax_left.spines["right"].set_visible(False)
    ax_right.spines["left"].set_visible(False)
    ax_right.tick_params(left=False)

    ax_left.set_xscale("log")
    ax_left.set_xlim(1.8, 32)
    ax_left.set_xticks([2, 10, 30])
    ax_left.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax_left.tick_params(axis="x", which="minor", bottom=False)

    ax_right.set_xlim(32, ells.max())
    ax_right.tick_params(labelleft=False)

    ax_left.set_ylabel(r"$D_\ell^{TT}$ [$\mu$K$^2$]", fontsize=11)
    ax_left.set_ylim(-100, 6500)

    for ax in [ax_left, ax_right]:
        ax.plot(ells, D_camb, "-", color=TOL["red"], lw=1.2,
                label="Higgs USR", zorder=4)
        ax.plot(ells_lcdm, D_pl, "--", color=TOL["dark"], lw=1.2,
                alpha=0.6, label=r"$\Lambda$CDM", zorder=3)

    ax_left.errorbar(p_ells, D_p, yerr=[D_lo, D_hi], fmt="o",
                     color=TOL["dark"], capsize=1.5, markersize=2,
                     elinewidth=0.4, label="Planck 2018", zorder=5)

    ax_left.set_xlabel(r"$\ell$", fontsize=11)
    ax_right.set_xlabel(r"$\ell$", fontsize=11)

    ax_left.grid(True, alpha=0.15, which="both")
    ax_right.grid(True, alpha=0.15, which="both")

    ax_right.axvline(x=32, color=TOL["grey"], ls="--", lw=1.5, zorder=0)
    ax_right.legend(loc="upper right", fontsize=9)

    fig.tight_layout()
    save_fig(fig, filename, category)
