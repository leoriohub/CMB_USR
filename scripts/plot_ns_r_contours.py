"""Plot n_s-r contours for Starobinsky model vs Planck/BK18 data."""
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from scripts.plotting import TOL, PAPER_RCPARAMS, save_fig, get_path

# ── 1. Starobinsky analytic prediction (parametrized by N*) ──
n_star_vals = np.linspace(50, 70, 21)
ns_pred = 1.0 - 2.0 / n_star_vals
r_pred = 12.0 / n_star_vals**2

# ── 2. Plot ──
with plt.rc_context(PAPER_RCPARAMS):
    fig, ax = plt.subplots(figsize=(3.5, 3.0))

    # Starobinsky curve
    ax.plot(ns_pred, r_pred, "-", color=TOL["red"], lw=1.5, label="Starobinsky R$^2$")
    # Annotate N* values
    for n_star in [50, 60, 70]:
        idx = int(np.argmin(np.abs(n_star_vals - n_star)))
        ax.annotate(f"N$_*$={n_star}",
                    (ns_pred[idx], r_pred[idx]),
                    (ns_pred[idx] + 0.002, r_pred[idx] + 0.0005),
                    fontsize=6, ha="left",
                    arrowprops=dict(arrowstyle="->", lw=0.5, color=TOL["grey"]))

    # Planck 2018 + BK18 contours (approximate ellipses)
    center_ns = 0.966
    ell_68 = Ellipse((center_ns, 0.0), width=2*0.006, height=2*0.012,
                     angle=-30, facecolor="none", edgecolor=TOL["blue"], lw=0.8, alpha=0.5,
                     label="Planck+BK18 68%")
    ell_95 = Ellipse((center_ns, 0.0), width=2*0.014, height=2*0.028,
                     angle=-30, facecolor=TOL["blue"], edgecolor=TOL["blue"],
                     lw=0.5, alpha=0.12,
                     label="Planck+BK18 95%")
    ax.add_patch(ell_95)
    ax.add_patch(ell_68)

    # Mark the numerical result
    ax.plot(0.9649, 0.0036, "D", color=TOL["red"], ms=4, zorder=5,
            label="This work (N*=55)")

    ax.set_xlim(0.94, 1.00)
    ax.set_ylim(0.0, 0.035)
    ax.set_xlabel(r"$n_s$")
    ax.set_ylabel(r"$r$ (tensor-to-scalar ratio)")
    ax.legend(fontsize=6, loc="upper right")
    ax.grid(True, alpha=0.2)

    save_fig(fig, "ns_r_starobinsky", "paper")
    print(f"Saved: {get_path('paper', 'ns_r_starobinsky.png')}")
