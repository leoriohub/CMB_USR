"""
Plot P_S(k) and D_ell from a cached spectrum JSON.

Usage:
  python scripts/plot_pspectrum.py outputs/cmb_results/pspectra/Punctuated_*.json
  python scripts/plot_pspectrum.py outputs/cmb_results/pspectra/Punctuated_*.json --save plots/
"""

import argparse
import glob
import os, sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

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

    ell_center = np.arange(2, 30) + 0.5
    planck_Dl = np.array([574.7, 1058.4, 1189.3, 1048.7, 1006.0, 1039.2,
                          941.3, 746.0, 720.2, 685.6, 607.0, 530.8,
                          490.6, 518.1, 501.2, 464.9, 438.4, 474.8,
                          461.5, 391.6, 328.5, 320.5, 358.2, 380.3,
                          318.6, 335.3, 314.2, 328.3])
    planck_err = np.array([140.0, 160.0, 150.0, 130.0, 120.0, 110.0,
                           100.0, 90.0, 80.0, 70.0, 65.0, 60.0,
                           55.0, 50.0, 50.0, 45.0, 45.0, 45.0,
                           45.0, 45.0, 40.0, 40.0, 40.0, 40.0,
                           40.0, 40.0, 40.0, 40.0])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(9, 8), sharex=False,
                                    gridspec_kw={"height_ratios": [1, 1]})

    ax1.loglog(k, ps, "b-", linewidth=1.5, label=r"$\mathcal{P}_\mathcal{S}(k)$ (PI)")
    ax1.axvline(meta.get("k_pivot_phys", 0.05), color="k", ls="--", alpha=0.3,
                label=f"pivot $k_*={meta.get('k_pivot_phys', 0.05)}$ Mpc$^{{-1}}$")
    ax1.set_ylabel(r"$\mathcal{P}_\mathcal{S}(k)$")
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    ax2.plot(ells, D_ell, "b-", linewidth=1.5, label=f"PI (SW), {meta.get('model','?')}")
    ax2.plot(ells, D_ell_pl, "k--", linewidth=1, alpha=0.6, label=r"$\Lambda$CDM ($n_s=%.3f$)" % ns)
    ax2.errorbar(ell_center, planck_Dl, yerr=planck_err,
                 fmt="ro", markersize=3, capsize=2, alpha=0.7, label="Planck 2018 (Commander)")
    ax2.set_xlabel(r"$\ell$")
    ax2.set_ylabel(r"$D_\ell^{\,TT}$ ($\mu$K$^2$)")
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    model = meta.get("model", "?")
    phi0 = meta.get("phi0", "?")
    y0 = meta.get("y0", "?")
    Ntot = meta.get("N_total", "?")
    Npiv = meta.get("N_pivot", "?")
    fig.suptitle(
        f"{model}  "
        + "  ".join([f"$\\varphi_0={phi0}$", f"$y_0={y0}$",
                     f"$N_{{\\mathrm{{tot}}}}={Ntot}$", f"$N_{{\\mathrm{{pivot}}}}={Npiv:.1f}$"]),
        fontsize=11,
    )

    fig.tight_layout()

    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        base = os.path.splitext(os.path.basename(path))[0]
        out = os.path.join(save_dir, f"plot_{base}.png")
        fig.savefig(out, dpi=150, bbox_inches="tight")
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
