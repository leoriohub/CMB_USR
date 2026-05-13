"""
Plot top Higgs best candidates: PS, D_ell (SW), SW vs CAMB, full-sky CAMB.

Loads top N configs from top30_candidates.jsonl, finds cached P_S(k),
computes C_ell via SW and CAMB, saves everything to
outputs/simulations/best_candidates/.

Usage:
  python scripts/plot_best_candidates.py [N_configs]
"""
import glob
import json
import os
import re
import sys

import numpy as np

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.interpolate import interp1d

from scripts.constants import As, k_pivot_phys, T_cmb, ROOT_DIR
from scripts.sachs_wolfe import compute_cl_sw
from scripts.camb_wrapper import compute_cl_full_camb, compute_cl_camb_powerlaw
from scripts.planck_data import get_planck_data_asymmetric, C_ell_to_d_ell, d_ell_to_C_ell

CONFIGS_FILE = os.path.join(ROOT_DIR, "outputs/simulations/logs/top30_candidates.jsonl")
PSPECTRA_DIR = os.path.join(ROOT_DIR, "outputs/simulations/pspectra")
OUTPUT_DIR = os.path.join(ROOT_DIR, "outputs/simulations/best_candidates")
C_ELL_DIR = os.path.join(OUTPUT_DIR, "c_ell")
PLOTS_DIR = os.path.join(OUTPUT_DIR, "plots")

TOL = {"blue": "#4477AA", "red": "#CC3311", "green": "#228833", "yellow": "#EE8866",
       "teal": "#44BB99", "purple": "#AA3377", "grey": "#666666", "dark": "#222222"}

PALETTE = ["#4477AA", "#CC3311", "#228833", "#EE8866", "#AA3377",
           "#44BB99", "#DDCC77", "#88CCEE", "#BBBBBB", "#333333"]

plt.rcParams.update({"font.size": 11, "axes.labelsize": 12, "axes.titlesize": 13,
                     "xtick.labelsize": 10, "ytick.labelsize": 10, "legend.fontsize": 9,
                     "figure.dpi": 150})


def load_top_configs(path, n=5):
    configs = []
    with open(path) as f:
        for line in f:
            if line.strip():
                configs.append(json.loads(line))
    configs.sort(key=lambda c: c.get("chi2", 999))
    return configs[:n]


def find_cached_ps(phi0, y0, n_star):
    n_int = round(n_star)
    fn_pat = f"*phi{phi0:.2f}_y0{y0:.3f}_*"
    matches = sorted(glob.glob(os.path.join(PSPECTRA_DIR, f"PS_Higgs*{fn_pat}")))
    if not matches:
        return None
    scored = [(abs(int(m.split("Nstar")[1].split("_")[0]) - n_int) if "Nstar" in m else 999, m)
              for m in matches if "Nstar" in m]
    if not scored:
        return matches[0]
    scored.sort(key=lambda x: x[0])
    return scored[0][1] if scored[0][0] <= 3 else scored[0][1]


def chi2_model(D_model, p_ells, D_planck, D_lo, D_hi, model_ells):
    chi2 = 0.0
    for i, ell_val in enumerate(p_ells):
        if ell_val > 29:
            continue
        idx = int(np.argmin(np.abs(model_ells - ell_val)))
        residual = D_model[idx] - D_planck[i]
        sigma = D_hi[i] if residual > 0 else D_lo[i]
        chi2 += (residual / sigma) ** 2
    return chi2


def process_config(cfg, lcdm_cache, planck_data, results_list):
    phi0, y0, n_star = cfg["phi0"], cfg["y0"], cfg["N_star"]
    chi2_val = cfg.get("chi2", 0)
    slug = f"phi{phi0:.2f}_y0{y0:.3f}_Nstar{n_star:.0f}"
    short_label = f"$\\phi_0$={phi0:.2f} $y_0$={y0:.3f} $N_*$={n_star:.0f}"
    full_label = f"{short_label} $\\chi^2$={chi2_val:.1f}"

    ps_path = find_cached_ps(phi0, y0, n_star)
    if ps_path is None:
        print(f"  SKIP {slug}: no cached P_S")
        return None
    print(f"  {slug}  chi2={chi2_val:.2f}")

    data = json.load(open(ps_path))
    spec = data["spectrum"]
    k_phys = np.array(spec["k_phys"])
    P_S = np.array(spec["P_S"])

    ps_dict = {"k_phys": k_phys, "P_S": P_S}

    ells_sw, C_sw = compute_cl_sw(ps_dict, ell_max=30)
    D_sw = C_ell_to_d_ell(ells_sw, C_sw)

    ells_c, C_c, _, _ = compute_cl_full_camb(ps_dict, ell_max=2500)
    D_camb = C_ell_to_d_ell(ells_c, C_c)

    ells_l, D_lcdm, D_lcdm_full = lcdm_cache
    D_lcdm_low = D_lcdm[ells_l <= 30]

    p_ells, D_p, D_lo, D_hi = planck_data
    chi2_model_val = chi2_model(D_camb, p_ells, D_p, D_lo, D_hi, ells_c)
    chi2_lcdm_val = chi2_model(D_lcdm_full, p_ells, D_p, D_lo, D_hi, ells_l)

    camb_data = {
        "ells": ells_c, "D_camb": D_camb, "D_pl": D_lcdm_full,
        "planck_ells": p_ells, "D_planck": D_p,
        "D_err_lower": D_lo, "D_err_upper": D_hi,
    }

    plot_individual(slug, full_label, k_phys, P_S, ells_sw, D_sw,
                    ells_c, D_camb, D_lcdm_low, ells_l, D_lcdm_full,
                    camb_data, planck_data)

    out = {"slug": slug, "config": cfg, "k_phys": k_phys, "P_S": P_S,
           "ells_sw": ells_sw, "D_sw": D_sw, "ells_camb": ells_c, "D_camb": D_camb,
           "chi2_model": chi2_model_val, "chi2_lcdm": chi2_lcdm_val}
    results_list.append(out)

    ce_out = os.path.join(C_ELL_DIR, f"camb_{slug}.json")
    json.dump({"slug": slug, "config": cfg, "ells": ells_c.tolist(),
               "D_camb": D_camb.tolist(), "D_sw": D_sw.tolist(),
               "chi2_model": chi2_model_val, "chi2_lcdm": chi2_lcdm_val,
               "D_planck": D_p.tolist(), "planck_ells": p_ells.tolist()},
              open(ce_out, "w"), indent=2)

    return out


def plot_individual(slug, label, k_phys, P_S, ells_sw, D_sw,
                    ells_c, D_camb, D_lcdm_low, ells_l, D_lcdm_full,
                    camb_data, planck_data):
    p_ells, D_p, D_lo, D_hi = planck_data
    d = os.path.join(PLOTS_DIR, slug)
    os.makedirs(d, exist_ok=True)

    # 1. PS plot
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.loglog(k_phys, P_S, "-", color=TOL["red"], lw=1.5, label=label)
    ns = 0.965
    k_plot = np.logspace(np.log10(max(k_phys.min(), 1e-6)), np.log10(k_phys.max()), 500)
    ps_lcdm = As * (k_plot / k_pivot_phys) ** (ns - 1.0)
    ax.loglog(k_plot, ps_lcdm, "--", color=TOL["grey"], lw=1.2, alpha=0.6, label=r"$\Lambda$CDM")
    ax.axvline(k_pivot_phys, color=TOL["grey"], ls=":", lw=0.8, alpha=0.4)
    ax.set_xlabel(r"$k$ [Mpc$^{-1}$]")
    ax.set_ylabel(r"$\mathcal{P}_{\mathcal{R}}(k)$")
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(d, "ps.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2. Low-ell D_ell (SW + CAMB + LCDM + Planck)
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.errorbar(p_ells, D_p, yerr=[D_hi, D_lo], fmt="o", color=TOL["dark"],
                capsize=3, markersize=4, elinewidth=1, label="Planck 2018", zorder=5)
    low = ells_c <= 30
    ax.semilogy(ells_c[low], D_camb[low], "-", color=TOL["blue"], lw=1.5,
                label="CAMB (full)", zorder=4)
    ax.semilogy(ells_sw, D_sw, "s-", color=TOL["red"], lw=1.2, ms=3,
                label="SW-only", zorder=3)
    low_l = ells_l <= 30
    ax.semilogy(ells_l[low_l], D_lcdm_full[low_l], "--", color=TOL["grey"], lw=1.2,
                label=r"$\Lambda$CDM (CAMB)", zorder=2)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$D_\ell^{TT}$ [$\mu$K$^2$]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, which="both")
    ax.set_xlim(1.5, 31)
    fig.tight_layout()
    fig.savefig(os.path.join(d, "dell_low_ell.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 3. Full-sky CAMB D_ell
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.semilogy(ells_c, D_camb, "-", color=TOL["red"], lw=1.2, label=label)
    ax.semilogy(ells_l, D_lcdm_full, "--", color=TOL["dark"], lw=1.2, alpha=0.6,
                label=r"$\Lambda$CDM")
    ax.set_xlabel(r"$\ell$", fontsize=14)
    ax.set_ylabel(r"$D_\ell^{TT}$ [$\mu$K$^2$]", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.25, which="both")
    ax.set_xlim(1.5, ells_c.max())
    fig.tight_layout()
    fig.savefig(os.path.join(d, "dell_fullsky.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 4. ISW fraction
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    isw_pct = (D_camb[low] - D_sw) / D_sw * 100
    ax.plot(ells_c[low], isw_pct, "-", color=TOL["purple"], lw=1.5)
    ax.axhline(0, color=TOL["grey"], ls="--", lw=0.8)
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"(CAMB $-$ SW) / SW [%]")
    ax.set_title("ISW Fraction", fontsize=11)
    ax.grid(True, alpha=0.25, which="both")
    ax.set_xlim(1.5, 31)
    fig.tight_layout()
    fig.savefig(os.path.join(d, "isw_fraction.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_comparison(results, planck_data, ells_l, D_lcdm_full):
    p_ells, D_p, D_lo, D_hi = planck_data
    comp_dir = os.path.join(PLOTS_DIR, "comparison")
    os.makedirs(comp_dir, exist_ok=True)

    # 1. PS overlay
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    for i, r in enumerate(results):
        c = PALETTE[i % len(PALETTE)]
        ax.loglog(r["k_phys"], r["P_S"], "-", color=c, lw=1.2,
                  label=rf"$\chi^2$={r['config']['chi2']:.1f}")
    k_plot = np.logspace(-5, 0, 500)
    ps_lcdm = As * (k_plot / k_pivot_phys) ** (0.965 - 1.0)
    ax.loglog(k_plot, ps_lcdm, "--", color=TOL["grey"], lw=1.5, alpha=0.6, label=r"$\Lambda$CDM")
    ax.axvline(k_pivot_phys, color=TOL["grey"], ls=":", lw=0.8, alpha=0.4)
    ax.set_xlabel(r"$k$ [Mpc$^{-1}$]")
    ax.set_ylabel(r"$\mathcal{P}_{\mathcal{R}}(k)$")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(os.path.join(comp_dir, "ps_overlay.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 2. Low-ell D_ell overlay (CAMB only)
    fig, ax = plt.subplots(figsize=(3.5, 2.8))
    ax.errorbar(p_ells, D_p, yerr=[D_hi, D_lo], fmt="o", color=TOL["dark"],
                capsize=3, markersize=4, elinewidth=1, label="Planck 2018", zorder=5)
    for i, rc in enumerate(results):
        c = PALETTE[i % len(PALETTE)]
        low = rc["ells_camb"] <= 30
        chi = rc["config"]["chi2"]
        ax.semilogy(rc["ells_camb"][low], rc["D_camb"][low], "-", color=c, lw=1.2,
                    label=rf"$\chi^2$={chi:.1f}")
    ax.set_xlabel(r"$\ell$")
    ax.set_ylabel(r"$D_\ell^{TT}$ [$\mu$K$^2$]")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.25, which="both")
    ax.set_xlim(1.5, 31)
    fig.tight_layout()
    fig.savefig(os.path.join(comp_dir, "dell_overlay.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # 3. Full-sky D_ell overlay
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for i, rc in enumerate(results):
        c = PALETTE[i % len(PALETTE)]
        chi = rc["config"]["chi2"]
        ax.semilogy(rc["ells_camb"], rc["D_camb"], "-", color=c, lw=1.2,
                    label=rf"$\chi^2$={chi:.1f}")
    ax.semilogy(ells_l, D_lcdm_full, "--", color=TOL["grey"], lw=1.5, alpha=0.6,
                label=r"$\Lambda$CDM")
    ax.set_xlabel(r"$\ell$", fontsize=14)
    ax.set_ylabel(r"$D_\ell^{TT}$ [$\mu$K$^2$]", fontsize=14)
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.25, which="both")
    ax.set_xlim(1.5, 2500)
    fig.tight_layout()
    fig.savefig(os.path.join(comp_dir, "dell_fullsky_overlay.png"), dpi=300, bbox_inches="tight")
    plt.close(fig)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    os.makedirs(PLOTS_DIR, exist_ok=True)
    os.makedirs(C_ELL_DIR, exist_ok=True)

    configs = load_top_configs(CONFIGS_FILE, n=n)
    print(f"Loaded {len(configs)} top configs")

    planck_data = get_planck_data_asymmetric()

    print("\nLCDM baseline...")
    ells_l, C_l, _, _ = compute_cl_camb_powerlaw(ell_max=2500)
    D_lcdm_full = C_ell_to_d_ell(ells_l, C_l)
    lcdm_cache = (ells_l, D_lcdm_full, D_lcdm_full)

    results = []
    for i, cfg in enumerate(configs):
        print(f"\n[{i+1}/{len(configs)}] ", end="")
        process_config(cfg, lcdm_cache, planck_data, results)

    print(f"\nComparison plots...")
    plot_comparison(results, planck_data, ells_l, D_lcdm_full)

    with open(os.path.join(OUTPUT_DIR, "configs.json"), "w") as f:
        json.dump([r["config"] for r in results], f, indent=2)
    with open(os.path.join(OUTPUT_DIR, "summary.json"), "w") as f:
        json.dump([{"slug": r["slug"], "chi2_model": r["chi2_model"],
                     "chi2_lcdm": r["chi2_lcdm"], "Delta": r["chi2_model"] - r["chi2_lcdm"]}
                    for r in results], f, indent=2)

    print(f"\nDone. Output in {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
