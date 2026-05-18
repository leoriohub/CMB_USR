"""
Reproduce Fig. 4: ε₂ evolution with USR and transition phases.

Config: x₀ = 5.43, y₀ = -0.07, ξ = 15000, λ = 0.13.
Shows two shaded regions (contiguous, same baseline y=-6):
  - Pure USR:    ε₂ < -5.5  (red)
  - Transition:  -5.5 < ε₂ < -1  (yellow)
"""
import numpy as np
import matplotlib.pyplot as plt
from scripts.usr_utils import run_bg_for_usr, measure_usr_duration
from scripts.plotting import get_path, TOL


def _set_fonts():
    plt.rcParams.update({
        "font.size": 14, "axes.labelsize": 16, "axes.titlesize": 18,
        "xtick.labelsize": 14, "ytick.labelsize": 14, "legend.fontsize": 12,
        "figure.dpi": 300, "font.family": "serif",
        "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": ":",
    })


def main():
    xi = 15000.0
    lam = 0.13
    phi0 = 5.43
    y0 = -0.07

    _set_fonts()

    T_max = max(100.0, xi / 5.0)
    N, epsH, eps2 = run_bg_for_usr(phi0, y0, xi=xi, lam=lam,
                                   T_max=T_max, bg_steps=10000)

    # Truncate at end of inflation for clean plot
    if np.any(epsH >= 1.0):
        end = int(np.argmax(epsH >= 1.0))
        cutoff = min(len(N), end + 20)
    else:
        cutoff = len(N)

    N_plot = N[:cutoff]
    eps2_plot = eps2[:cutoff]

    dur_usr = measure_usr_duration(N, epsH, eps2, threshold=-5.5)
    transition_mask = (eps2_plot > -5.5) & (eps2_plot < -1)
    dN = np.diff(N_plot, prepend=N_plot[0])
    dur_trans = float(np.sum(dN[transition_mask]))
    print(f"  Pure USR duration (ε₂ < -5.5):  {dur_usr:.4f} e-folds")
    print(f"  Transition duration (-5.5 < ε₂ < -1): {dur_trans:.4f} e-folds")

    fig, ax = plt.subplots(figsize=(7, 5))

    ax.plot(N_plot, eps2_plot, linewidth=2, color=TOL["blue"],
            label=r"$\epsilon_2 \equiv d\ln\epsilon_1/dN$")

    ax.axhline(0, color="k", linewidth=0.8, alpha=0.3)
    ax.axhline(-6, color=TOL["red"], linestyle="--", linewidth=1.5, alpha=0.8)

    # Two-layer fill for guaranteed contiguity:
    # 1. Base layer: all ε₂ < -1 (transition + USR) in yellow
    ax.fill_between(N_plot, eps2_plot, -6,
                    where=(eps2_plot < -1) & (eps2_plot > -8),
                    color="#DDCC33", alpha=0.3, label="Transition")

    # 2. Overlay: pure USR (ε₂ < -5.5) in red — same x-grid, no gaps possible
    ax.fill_between(N_plot, eps2_plot, -6,
                    where=(eps2_plot < -5.5) & (eps2_plot > -8),
                    color=TOL["red"], alpha=0.4, label="Pure USR")

    ax.text(0.1, -6.5, "USR", color=TOL["red"],
            fontweight="bold", alpha=0.8, fontsize=13)

    # Find last index where ε₂ < -1 (end of transition regime)
    below_minus1 = eps2_plot < -1
    if np.any(below_minus1):
        last_below = int(np.where(below_minus1)[0][-1])
        x_max = float(N_plot[last_below]) + 0.5
    else:
        x_max = float(N_plot[-1])

    ax.set_xlabel("e-folds ($N$)")
    ax.set_ylabel(r"$\epsilon_2$")
    ax.set_xlim(0, x_max)
    ax.set_ylim(-8, 2)
    ax.legend(loc="upper right")

    out = get_path("paper", "paper_fig4_eps2_evolution.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
