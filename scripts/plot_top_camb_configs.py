"""
Plot D_ell and P_S comparison for multiple Higgs USR configs.

Usage:
  python scripts/plot_top_camb_configs.py \
    --phi0 6.70,6.30,7.10 \
    --y0 -0.070,-0.095,-0.170 \
    --nstar 65.2,64.7,62.7 \
    --labels "best,mild+,low-ell_dip" \
    --output-suffix my_comparison

Default: shows the 6 diverse configs from the CAMB scan.
"""
import argparse
import glob
import json
import os

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scripts.constants import As, k_pivot_phys, T_cmb, ROOT_DIR
from scripts.pspectrum_pipeline import run_pspectrum_pipeline
from scripts.camb_wrapper import (
    compute_cl_full_camb, compute_cl_camb_powerlaw, compute_chi2_camb,
)
from scripts.planck_data import C_ell_to_d_ell, get_planck_data_asymmetric
from models import HiggsModel

OUT_DIR = os.path.join(ROOT_DIR, "outputs/plots/diagnostics")
PSPECTRA_DIR = os.path.join(ROOT_DIR, "outputs/simulations/pspectra")
os.makedirs(OUT_DIR, exist_ok=True)

TOL = {"blue": "#4477AA", "red": "#CC3311", "green": "#228833",
       "yellow": "#EE8866", "teal": "#44BB99", "purple": "#AA3377",
       "grey": "#666666", "dark": "#222222"}

COLORS = ["#CC3311", "#EE8866", "#44BB99", "#AA3377",
          "#4477AA", "#228833", "#DDCC77", "#88CCEE"]

plt.rcParams.update({"font.size": 11, "axes.labelsize": 13,
                     "xtick.labelsize": 10, "ytick.labelsize": 10,
                     "legend.fontsize": 8, "figure.dpi": 150})

DEFAULT_CONFIGS = [
    {"label": "best mild", "phi0": 6.70, "y0": -0.070, "N_star": 65.2},
    {"label": "mild+", "phi0": 6.30, "y0": -0.095, "N_star": 64.7},
    {"label": "low-ell dip", "phi0": 7.10, "y0": -0.170, "N_star": 62.7},
    {"label": "deep dip", "phi0": 6.55, "y0": -0.230, "N_star": 63.4},
    {"label": "golden ref", "phi0": 6.60, "y0": -0.736, "N_star": 52.6},
    {"label": "lowest D2", "phi0": 6.55, "y0": -0.340, "N_star": 63.2},
]


def find_ps_file(phi0, y0, n_star):
    pat = "*phi{:.2f}_y0{:.3f}_*".format(phi0, y0)
    matches = sorted(glob.glob(os.path.join(PSPECTRA_DIR, "PS_Higgs*" + pat)))
    scored = []
    for m in matches:
        try:
            with open(m) as f:
                md = json.load(f)["metadata"]
            ns = md.get("N_star", 0)
        except Exception:
            ns = 0
        scored.append((abs(ns - n_star), m))
    if not scored:
        return None
    scored.sort(key=lambda x: x[0])
    return scored[0][1] if scored[0][0] <= 3 else None


def load_or_run(cfg):
    path = find_ps_file(cfg["phi0"], cfg["y0"], cfg["N_star"])
    if path is not None:
        with open(path) as f:
            rec = json.load(f)
        spec = rec["spectrum"]
        return {"k_phys": np.array(spec["k_phys"]),
                "P_S": np.array(spec["P_S"])}

    print("  Running pipeline phi0={} y0={} N*={}".format(
        cfg["phi0"], cfg["y0"], cfg["N_star"]))
    model = HiggsModel(lam=0.13, xi=15000.0)
    result = run_pspectrum_pipeline(
        model=model, phi0=cfg["phi0"], y0=cfg["y0"],
        k_min=1e-5, k_max=1.0, k_pivot_phys=k_pivot_phys,
        N_star=cfg["N_star"], normalize_to_As=True, As=As,
        num_k=80, n_workers=4, save_outputs=False,
    )
    if result["status"] != "success":
        print("  FAILED: {}".format(result.get("message", "")))
        return None
    return {"k_phys": result["k_phys"], "P_S": result["P_S"]}


def compute_camb(ps_data):
    ells, C_TT, _, _ = compute_cl_full_camb(ps_data, ell_max=2500)
    D = C_ell_to_d_ell(ells, C_TT)
    chi2, _, _ = compute_chi2_camb(ps_data, ell_max=29)
    return ells, D, chi2


def build_configs(args):
    if args.phi0 is not None:
        phi0s = [float(x) for x in args.phi0.split(",")]
        y0s = [float(x) for x in args.y0.split(",")]
        nstars = [float(x) for x in args.nstar.split(",")]
        if args.labels is not None:
            labels = [x.strip() for x in args.labels.split(",")]
        else:
            labels = ["phi{:.2f}_y0{:.3f}_N*{:.1f}".format(p, y, n)
                      for p, y, n in zip(phi0s, y0s, nstars)]
        if not (len(phi0s) == len(y0s) == len(nstars) == len(labels)):
            raise ValueError("phi0, y0, nstar, labels must have same length")
        return [{"label": l, "phi0": p, "y0": y, "N_star": n}
                for l, p, y, n in zip(labels, phi0s, y0s, nstars)]
    return list(DEFAULT_CONFIGS)


def parse_args():
    p = argparse.ArgumentParser(description="Plot D_ell and P_S for multiple Higgs USR configs")
    p.add_argument("--phi0", type=str, default=None,
                   help="Comma-separated phi0 values")
    p.add_argument("--y0", type=str, default=None,
                   help="Comma-separated y0 values")
    p.add_argument("--nstar", type=str, default=None,
                   help="Comma-separated N_star values")
    p.add_argument("--labels", type=str, default=None,
                   help="Comma-separated labels (optional)")
    p.add_argument("--output-suffix", type=str, default="camb_top_configs",
                   help="Suffix for output filenames")
    return p.parse_args()


def main():
    args = parse_args()
    configs = build_configs(args)
    suffix = args.output_suffix

    print("LCDM baseline...")
    ells_l, C_l, _, _ = compute_cl_camb_powerlaw(ell_max=2500)
    D_lcdm = C_ell_to_d_ell(ells_l, C_l)
    p_ells, D_p, D_lo, D_hi = get_planck_data_asymmetric()

    results = []
    for cfg in configs:
        print("\n" + cfg["label"] + "...")
        ps = load_or_run(cfg)
        if ps is None:
            continue
        ells, D, chi2 = compute_camb(ps)
        results.append({**cfg, "ells": ells, "D": D, "chi2": round(chi2, 2),
                        "k_phys": ps["k_phys"], "P_S": ps["P_S"]})

    if not results:
        print("No configs successfully processed. Exiting.")
        return

    print("\nGenerating plots...")

    # P_S(k) comparison
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    k_plot = np.logspace(-5, 0, 500)
    ps_lcdm = As * (k_plot / k_pivot_phys) ** (0.965 - 1.0)
    ax.loglog(k_plot, ps_lcdm, "--", color=TOL["grey"], lw=1.5, alpha=0.6,
              label=r"$\Lambda$CDM")
    for i, r in enumerate(results):
        c = COLORS[i % len(COLORS)]
        lab = r"{} $\chi^2$={:.1f}".format(r["label"], r["chi2"])
        ax.loglog(r["k_phys"], r["P_S"], "-", color=c, lw=1.2, label=lab)
    ax.axvline(k_pivot_phys, color=TOL["grey"], ls=":", lw=0.8, alpha=0.4)
    ax.set_xlabel(r"$k$ [Mpc$^{-1}$]")
    ax.set_ylabel(r"$\mathcal{P}_{\mathcal{R}}(k)$")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.25, which="both")
    ax.set_xlim(1e-5, 1)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, suffix + "_ps.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: {}_ps.png".format(suffix))

    # Low-ell D_ell comparison
    fig, ax = plt.subplots(figsize=(5.5, 3.8))
    ax.errorbar(p_ells, D_p, yerr=[D_hi, D_lo], fmt="o", color=TOL["dark"],
                capsize=3, markersize=4, elinewidth=1, label="Planck 2018",
                zorder=5)
    low = ells_l <= 30
    ax.semilogy(ells_l[low], D_lcdm[low], "--", color=TOL["grey"], lw=1.5,
                alpha=0.6, label=r"$\Lambda$CDM", zorder=2)
    for i, r in enumerate(results):
        c = COLORS[i % len(COLORS)]
        lab = r"{} $\chi^2$={:.1f}".format(r["label"], r["chi2"])
        e = r["ells"]
        m = e <= 30
        ax.semilogy(e[m], r["D"][m], "-", color=c, lw=1.5, label=lab, zorder=3)
    ax.set_xlabel(r"$\ell$", fontsize=13)
    ax.set_ylabel(r"$D_\ell^{TT}$ [$\mu$K$^2$]", fontsize=13)
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(True, alpha=0.25, which="both")
    ax.set_xlim(1.5, 31)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, suffix + "_dell_low.png"),
                dpi=300, bbox_inches="tight")
    plt.close(fig)
    print("  Saved: {}_dell_low.png".format(suffix))


if __name__ == "__main__":
    main()
