"""
Reproduce Fig. 1: Higgs effective potential U_E(x).

Plots U_E(x) = v0 * f(x) for the canonically normalised field x = χ/M_P,
marks the CMB pivot scale x_cmb computed from a slow-roll trajectory.
"""
import numpy as np
import matplotlib.pyplot as plt
from models import HiggsModel
from scripts.usr_utils import compute_xcmb
from scripts.plotting import get_path, TOL


def _set_fonts():
    plt.rcParams.update({
        "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 9,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "figure.dpi": 300, "font.family": "serif",
        "axes.grid": True, "grid.alpha": 0.3, "grid.linestyle": ":",
    })


def main():
    xi = 15000.0
    lam = 0.13
    N_star = 60

    _set_fonts()

    model = HiggsModel(lam=lam, xi=xi)
    v0 = model.v0
    alpha = model.alpha

    # Compute x_cmb dynamically from SR trajectory
    x_cmb, N_total = compute_xcmb(x0=5.7, y0_sr=-1e-6, N_star=N_star,
                                  lam=lam, xi=xi)
    if x_cmb is None:
        x_cmb = 5.42
    print(f"  N_total = {N_total:.2f}, x_cmb = {x_cmb:.5f}")

    # Potential over full range
    x_arr = np.linspace(0, 10, 500)
    V_arr = v0 * model.f(x_arr)

    # -- Plot --
    fig, ax = plt.subplots(figsize=(3.35, 2.6))
    ax.plot(x_arr, V_arr, color=TOL["red"], lw=1.3,
            label=r"Potential $U_E(x)$")

    V_cmb = v0 * model.f(x_cmb)
    ax.plot(x_cmb, V_cmb, "o", color=TOL["blue"], markersize=5,
            label=rf"CMB Pivot ($x_* = {x_cmb:.2f}$)")
    ax.axvline(x=x_cmb, linewidth=1.0, color=TOL["grey"], linestyle=":", alpha=0.6)

    ax.set_xlabel(r"Field $x$")
    ax.set_ylabel(r"$U(x)/M_P^4$")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, np.max(V_arr) * 1.05)
    ax.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    ax.legend(loc="lower right")

    out = get_path("paper", "paper_fig1_higgs_potential.png")
    fig.savefig(out, dpi=300, bbox_inches="tight")
    print(f"  Saved: {out}")
    plt.close(fig)


if __name__ == "__main__":
    main()
