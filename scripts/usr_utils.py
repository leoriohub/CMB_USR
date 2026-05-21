"""
Shared utilities for reproducing paper figures and tables.

Provides:
  compute_epsilon2          ε₂ = 2ε₁ - 2η_H
  measure_usr_duration      integrate dN where ε₂ < threshold
  compute_xcmb              field value at pivot exit N* e-folds before end
  plot_defaults             publication-ready rcParams
"""
import numpy as np
from models import HiggsModel
import inf_dyn_background as bg_solver
from pspectrum_pipeline import find_end_of_inflation
from scripts.constants import lam_default, xi_default


def compute_epsilon2(epsH, etaH):
    """ε₂ = d ln ε₁ / d N = 2 ε₁ - 2 η_H  (paper Eq. 40)."""
    return 2.0 * epsH - 2.0 * etaH


def measure_usr_duration(N, epsH, eps2, threshold=-5.5):
    """
    Number of e-folds where ε₂ < threshold (USR phase).

    Algorithm from USR_Search.ipynb:
      1. Find onset of inflation (first epsH < 1).
      2. Find end of inflation (first epsH >= 1 after onset).
      3. Truncate to inflationary segment.
      4. Sum dN where eps2 < threshold.
    """
    if np.all(epsH >= 1.0):
        return 0.0
    start = int(np.where(epsH < 1.0)[0][0])
    post = epsH[start:] >= 1.0
    if np.any(post):
        cutoff = start + int(np.where(post)[0][0])
    else:
        cutoff = len(epsH)
    N_inf = N[start:cutoff]
    eps2_inf = eps2[start:cutoff]
    mask = eps2_inf < threshold
    if not np.any(mask):
        return 0.0
    dN = np.diff(N_inf, prepend=N_inf[0])
    return float(np.sum(dN[mask]))


def run_bg_for_usr(phi0, y0, xi=xi_default, lam=lam_default,
                   T_max=None, bg_steps=10000):
    """
    Run background simulation for USR analysis.

    T_max scales with xi per notebook convention: max(100, xi/5).
    Returns (N, epsH, eps2) arrays.
    """
    if T_max is None:
        T_max = max(100.0, xi / 5.0)
    model = HiggsModel(lam=lam, xi=xi)
    model.x0 = phi0
    model.y0 = y0
    T_span = np.linspace(0.0, T_max, bg_steps)
    bg_sol = bg_solver.run_background_simulation(model, T_span)
    derived = bg_solver.get_derived_quantities(bg_sol, model)
    eps2 = compute_epsilon2(derived["epsH"], derived["etaH"])
    return np.asarray(derived["N"]), np.asarray(derived["epsH"]), np.asarray(eps2)


def compute_xcmb(x0=5.7, y0_sr=-1e-6, N_star=60, lam=lam_default, xi=xi_default,
                 T_max=2000.0, bg_steps=100000, model_cls=None):
    """
    Compute field value at CMB pivot exit (N_star e-folds before end).

    Uses a slow-roll trajectory to find x_cmb.
    Returns (x_cmb, N_total).
    model_cls: model class (default: HiggsModel). Pass FullHiggsModel for full potential.
    """
    if model_cls is None:
        model_cls = HiggsModel
    model = model_cls(lam=lam, xi=xi)
    model.x0 = x0
    model.y0 = y0_sr
    T_span = np.linspace(0.0, T_max, bg_steps)
    bg_sol = bg_solver.run_background_simulation(model, T_span)
    derived = bg_solver.get_derived_quantities(bg_sol, model)
    end_idx = find_end_of_inflation(derived["epsH"])
    N_total = float(derived["N"][end_idx])
    N_pivot = N_total - N_star
    if N_pivot < derived["N"][0]:
        return None, N_total
    pivot_idx = int(np.argmin(np.abs(derived["N"][:end_idx] - N_pivot)))
    x_cmb = float(bg_sol[0][pivot_idx])
    return x_cmb, N_total


def set_paper_style(use_tex=False):
    """Apply publication-ready matplotlib rcParams."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.size": 14,
        "axes.labelsize": 16,
        "axes.titlesize": 18,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "legend.fontsize": 12,
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "font.family": "serif",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": ":",
    })
    if use_tex:
        plt.rcParams.update({
            "text.usetex": True,
            "font.family": "serif",
        })
