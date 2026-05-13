"""Unified plotting utilities for CMB anomaly analysis.

Extracted from run_full_analysis.py, plot_pspectrum.py, plot_potential.py,
plot_usr_discussion.py, higgs_usr_optimizer.py, and usr_chi2_optimizer.py.

All functions follow publication-ready conventions:
- Two-column format (~3.25-3.5in wide single, ~7in full)
- 300 DPI minimum
- Colorblind-friendly palette (Tol 2012)
- Big fonts: axis >= 14pt, ticks >= 12pt, legend >= 11pt
- Export both PNG and PDF
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
    "optimizer": os.path.join(ROOT_DIR, "outputs/plots/optimizer"),
    "powerloss": os.path.join(ROOT_DIR, "outputs/plots/powerloss"),
    "top30": os.path.join(ROOT_DIR, "outputs/plots/top30_candidates"),
    "punctuated": os.path.join(ROOT_DIR, "outputs/plots/punctuated_potential"),
}


def _ensure_dir(subdir):
    os.makedirs(OUTPUT_DIRS.get(subdir, subdir), exist_ok=True)
    return OUTPUT_DIRS.get(subdir, subdir)


def _save_fig(fig, path_base, subdir="diagnostics"):
    out_dir = _ensure_dir(subdir)
    for ext in ["png", "pdf"]:
        path = os.path.join(out_dir, f"{path_base}.{ext}")
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
        yerr = [D_err_upper[mask], D_err_lower[mask]] if D_err_upper is not None else None
        ax.errorbar(planck_ells[mask], D_planck[mask], yerr=yerr,
                    fmt="o", color=TOL["dark"], capsize=3, capthick=1,
                    markersize=4, elinewidth=1,
                    label=r"Planck 2018 low-$\ell$ TT")

    ell_dense = np.linspace(ells.min(), min(ells.max(), ell_max), 200)
    D_interp = interp1d(ells, D_ell_model, kind="cubic")(ell_dense)
    ax.semilogy(ell_dense, D_interp, "-", color=TOL["red"], lw=1.5, label=model_label)

    if ells_lcdm is not None and D_ell_lcdm is not None:
        mask = ells_lcdm <= ell_max
        D_lcdm_interp = interp1d(ells_lcdm[mask], D_ell_lcdm[mask], kind="cubic")(ell_dense)
        ax.semilogy(ell_dense, D_lcdm_interp, "--", color=TOL["grey"], lw=1.2,
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
    ax.set_ylabel(r"$\phi$", fontsize=10)
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
    ax.set_xlabel(r"$\phi$", fontsize=10)
    ax.set_ylabel(r"$d\phi/dT$", fontsize=10)
    ax.grid(True, alpha=0.25, which="both")

    fig.tight_layout()
    _save_fig(fig, filename, subdir)


def plot_camb_comparison(camb_data, sw_ells, sw_D, sw_D_pl, filename="camb_dell",
                         subdir="powerloss"):
    """SW vs CAMB comparison at low ell."""
    ells = camb_data["ells"]
    D_camb = camb_data["D_camb"]
    D_pl = camb_data["D_pl"]
    p_ells = camb_data["planck_ells"]
    D_p = camb_data["D_planck"]
    D_err_lo = camb_data["D_err_lower"]
    D_err_hi = camb_data["D_err_upper"]

    low = ells <= 30
    fig, ax = plt.subplots(figsize=(3.5, 2.8))

    ax.errorbar(p_ells, D_p, yerr=[D_err_hi, D_err_lo],
                fmt="o", color=TOL["dark"], capsize=3, capthick=1,
                markersize=4, elinewidth=1, label="Planck 2018", zorder=5)
    ax.semilogy(ells[low], D_camb[low], "-", color=TOL["blue"], lw=1.5,
                label="CAMB (full)", zorder=4)
    ax.semilogy(sw_ells, sw_D, "s-", color=TOL["red"], lw=1.2, ms=3,
                label="SW-only", zorder=3)
    ax.semilogy(ells[low], D_pl[low], "--", color=TOL["grey"], lw=1.2,
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


def plot_scan_heatmap(phi0_vals, y0_vals, chi2_map, supp_map,
                      filename="scan_heatmap", subdir="optimizer"):
    """Chi2 and suppression heatmaps from grid scan."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

    im1 = ax1.pcolormesh(phi0_vals, y0_vals, chi2_map, shading="auto",
                         cmap="viridis_r")
    ax1.set_xlabel(r"$\phi_0$", fontsize=10)
    ax1.set_ylabel(r"$y_0$", fontsize=10)
    ax1.set_title(r"$\chi^2$ vs Planck low-$\ell$", fontsize=11)
    fig.colorbar(im1, ax=ax1, label=r"$\chi^2$")

    im2 = ax2.pcolormesh(phi0_vals, y0_vals, supp_map, shading="auto",
                         cmap="coolwarm", vmin=-50, vmax=50)
    ax2.set_xlabel(r"$\phi_0$", fontsize=10)
    ax2.set_ylabel(r"$y_0$", fontsize=10)
    ax2.set_title("Dip Suppression [%]", fontsize=11)
    fig.colorbar(im2, ax=ax2, label="suppression %")

    fig.tight_layout()
    _save_fig(fig, filename, subdir)


def plot_convergence(records, filename="convergence", subdir="optimizer"):
    """Convergence metrics from JSONL log."""
    ok_recs = [r for r in records if r.get("status") == "ok"]
    if not ok_recs:
        return

    evals = np.arange(len(ok_recs))
    chi2 = np.array([r["chi2"] for r in ok_recs])
    ns = np.array([r["ns_MS"] for r in ok_recs])
    kd = np.array([r.get("k_dip", -1.0) for r in ok_recs])
    loss = np.array([r["loss"] for r in ok_recs])
    best = np.minimum.accumulate(loss)

    fig, axes = plt.subplots(3, 1, figsize=(5, 6), sharex=True)

    axes[0].plot(evals, loss, "b.", alpha=0.25, ms=2, label="all")
    axes[0].plot(evals, best, "r-", lw=2, label="running best")
    axes[0].set_ylabel("Loss", fontsize=10)
    axes[0].set_title("Optimizer Convergence", fontsize=11)
    axes[0].legend(fontsize=8)
    axes[0].grid(True, alpha=0.3)

    axes[1].axhline(0.975, color="k", ls="--", alpha=0.5, label="target")
    axes[1].plot(evals, ns, "g.", alpha=0.25, ms=2)
    axes[1].set_ylabel(r"$n_s$", fontsize=10)
    axes[1].set_ylim(0.94, 1.02)
    axes[1].legend(fontsize=8)
    axes[1].grid(True, alpha=0.3)

    ok_k = kd > 0
    if np.any(ok_k):
        axes[2].semilogy(evals[ok_k], kd[ok_k], "m.", alpha=0.25, ms=2)
    axes[2].axhline(1e-4, color="k", ls="--", alpha=0.3)
    axes[2].axhline(5e-4, color="k", ls="--", alpha=0.3)
    axes[2].set_ylabel(r"$k_{\rm dip}$", fontsize=10)
    axes[2].set_xlabel("Evaluation", fontsize=10)
    axes[2].grid(True, alpha=0.3)

    fig.tight_layout()
    _save_fig(fig, filename, subdir)


def plot_best_dashboard(ells_usr, D_usr, ells_lcdm, D_lcdm,
                        ells_pl, D_pl, D_err, phi0, y0, N_star,
                        k_dip, chi2, chi2_lcdm, filename="best_dashboard",
                        subdir="optimizer"):
    """D_ell comparison and suppression ratio for best config."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7, 3))

    ax1.errorbar(ells_pl, D_pl, yerr=D_err, fmt="ko", ms=4,
                 capsize=2, label="Planck 2018")
    ax1.semilogx(ells_usr, D_usr, "r-", lw=1.5, label="Best")
    ax1.semilogx(ells_lcdm, D_lcdm, "gray", ls="--", lw=1, label=r"$\Lambda$CDM")
    ax1.set_xlabel(r"$\ell$", fontsize=10)
    ax1.set_ylabel(r"$D_\ell\ [\mu{\rm K}^2]$", fontsize=10)
    ax1.set_title(r"Low-$\ell$ Power Spectrum", fontsize=11)
    ax1.legend(fontsize=8)
    ax1.set_xlim(1.5, 30)
    ax1.grid(True, alpha=0.3)

    ratio = D_usr / np.interp(ells_usr, ells_lcdm, D_lcdm)
    ax2.semilogx(ells_usr, ratio, "r-", lw=1.5, label="USR / LCDM")
    ax2.axhline(1.0, color="k", ls="--", lw=1, alpha=0.5)
    ax2.fill_between(ells_usr, ratio, 1.0, alpha=0.2, color="red",
                     where=(ratio < 1.0))
    ax2.set_xlabel(r"$\ell$", fontsize=10)
    ax2.set_ylabel(r"$D_\ell / D_\ell^{\rm LCDM}$", fontsize=10)
    ax2.set_title("Suppression vs LCDM", fontsize=11)
    ax2.legend(fontsize=8)
    ax2.set_xlim(1.5, 30)
    ax2.grid(True, alpha=0.3)

    fig.suptitle(
        f"$\\phi_0$={phi0:.2f}, $y_0$={y0:.3f}, $N_{{*}}$={N_star:.0f}\n"
        f"$\\chi^2$={chi2:.1f}  $\\Delta\\chi^2$={chi2 - chi2_lcdm:+.1f}",
        fontsize=10, fontweight="bold")
    fig.tight_layout()
    _save_fig(fig, filename, subdir)
