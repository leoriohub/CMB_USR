#!/usr/bin/env python3
"""
Compaction PBH diagnostics — why Ezquiaga P_S spectra give zero
abundance under compaction formalism but work with Press-Schechter.
Barriers: (1) σ₀² suppression via Gaussian windowing, (2) C_c/σ₀ ≳ 14.
"""

from __future__ import annotations

import json

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.special import erfc

from scripts.compaction import beta_f_compaction
from scripts.full_pbh_pipeline import beta_f_press_schechter
from scripts.plotting import TOL, PAPER_RCPARAMS, get_path, save_fig


# ── Data loading ──────────────────────────────────────────────────────────


def compute_compaction_data(ps_path: str) -> dict:
    """Load P_S(k), run compaction, return diagnostics."""
    with open(ps_path) as f:
        d = json.load(f)
    if isinstance(d, dict) and "spectrum" in d and isinstance(d["spectrum"], dict):
        k_phys = np.array(d["spectrum"]["k_phys"])
        P_S = np.array(d["spectrum"]["P_S"])
    else:
        k_phys = np.array(d["k_phys"])
        P_S = np.array(d["P_S"])

    ss = slice(None, None, 5)
    k_sub, ps_sub = k_phys[ss], P_S[ss]
    print(f"  P_S: {len(k_phys)} modes, sub: {len(k_sub)}", flush=True)

    print("  Compaction (sigma0 method)...", flush=True)
    beta_f, M_pbh, meta = beta_f_compaction(k_sub, ps_sub, beta_f_method="sigma0")

    print("  σ₀² from compaction meta...", flush=True)
    sigma0_sub = np.sqrt(np.maximum(meta.get("sigma0_arr", np.zeros_like(beta_f)), 1e-300))
    sigma0_all = np.interp(k_phys, k_sub, sigma0_sub)
    s2_all = sigma0_all ** 2

    # Interpolate C_c to full k grid
    C_c_sub = meta["C_c_arr"]
    C_c_all = np.interp(k_phys, k_sub, C_c_sub)

    # Formation fractions
    ZETA_C = 0.077
    beta_ps = beta_f_press_schechter(P_S, zeta_c=ZETA_C)
    beta_comp = erfc(C_c_all / (np.sqrt(2.0) * sigma0_all))

    return {
        "k": k_phys, "ps": P_S, "k_sub": k_sub, "ps_sub": ps_sub,
        "s2_all": s2_all, "beta_f": beta_f, "M_pbh": M_pbh,
        "C_max": meta["C_max_arr"], "C_c": meta["C_c_arr"],
        "M_H": meta["M_H_arr"],
        "sigma0": np.sqrt(np.maximum(meta.get("sigma0_arr", np.zeros_like(beta_f)), 1e-300)),
        "sigma0_all": sigma0_all,
        "C_c_all": C_c_all,
        "beta_ps": beta_ps,
        "beta_comp": beta_comp,
    }


# ── Plot 1: β_PS vs β_comp comparison ────────────────────────────────────


def plot_formation_comparison(data: dict, save_path: str) -> None:
    """Semi-log: β_PS(k) vs β_comp(k) showing why compaction gives ≈0."""
    k = data["k"]
    beta_ps = data["beta_ps"]
    beta_comp = data["beta_comp"]

    # Find the peak of β_PS and the corresponding β_comp
    peak_idx = int(np.argmax(beta_ps))
    k_peak = k[peak_idx]
    ps_val = beta_ps[peak_idx]
    comp_val = beta_comp[peak_idx]
    ratio_val = ps_val / max(comp_val, 1e-300)
    log10_ratio = np.log10(ratio_val)

    with plt.rc_context(PAPER_RCPARAMS):
        fig, ax = plt.subplots(figsize=(3.5, 2.8))

        ax.semilogy(k, beta_ps, "-", color=TOL["blue"], lw=1.0,
                    label=r"$\beta_{\mathrm{PS}}(\zeta_c=0.077)$", zorder=3)
        ax.semilogy(k, beta_comp, "--", color=TOL["red"], lw=1.0,
                    label=r"$\beta_{\mathrm{comp}}(C_{\mathrm{c}}\approx 0.44)$", zorder=4)

        # Observable threshold
        ax.axhline(1e-10, color=TOL["grey"], ls="--", lw=0.8, zorder=2,
                   label="Observable PBH threshold")

        # CMB band
        ax.axvspan(3e-4, 0.1, alpha=0.06, color=TOL["grey"], zorder=0)

        # Annotation: ratio arrow from β_PS peak down to β_comp
        ax.annotate(
            rf"$\beta_{{\mathrm{{PS}}}}/\beta_{{\mathrm{{comp}}}} \sim 10^{{{log10_ratio:.0f}}}$",
            xy=(k_peak, comp_val),
            xytext=(k_peak * 5, ps_val * 0.1),
            fontsize=7, color=TOL["dark"],
            arrowprops=dict(arrowstyle="->", color=TOL["dark"], lw=0.8),
            zorder=10,
        )

        ax.text(0.5, 0.05, r"$\beta_{\mathrm{comp}} \approx 0$ (astronomically rare)",
                transform=ax.transAxes, fontsize=6, color=TOL["dark"], alpha=0.6,
                ha="center", va="bottom")
        ax.set_xlabel(r"$k$ [Mpc$^{-1}$]")
        ax.set_ylabel(r"$\beta$ (formation fraction)")
        ax.set_xscale("log")
        ax.set_ylim(1e-30, 1)
        ax.set_xlim(k.min(), k.max())
        ax.legend(loc="upper right", fontsize=6.5)
        save_fig(fig, save_path)


# ── Plot 2: Collapse barriers ────────────────────────────────────────────


def plot_collapse_barriers(data: dict, save_path: str) -> None:
    """Two panels: collapse criterion + fluctuation rarity."""
    ok = data["M_pbh"] > 0
    if ok.sum() == 0:
        print("  WARNING: no collapsing modes — showing all")
        ok = np.ones(len(data["M_pbh"]), dtype=bool)

    M_H, ratio = data["M_H"][ok], data["C_max"][ok] / data["C_c"][ok]
    rarity = data["C_c"][ok] / data["sigma0"][ok]
    mr = float(np.nanmin(rarity))
    mi = int(np.nanargmin(rarity))

    with plt.rc_context(PAPER_RCPARAMS):
        fig, axes = plt.subplots(1, 2, figsize=(7, 3), gridspec_kw={"wspace": 0.3})

        ax = axes[0]
        ax.axhline(1.0, color=TOL["grey"], ls="--", lw=1.0,
                   label=r"$C_{\max}=C_{\mathrm{c}}$")
        ax.scatter(M_H, ratio, s=4, c=TOL["blue"], alpha=0.7, zorder=5)
        if np.all(ratio > 1.0):
            ax.text(0.05, 0.95, r"$\checkmark$ All $C_{\max} \gg C_{\mathrm{c}}$",
                    transform=ax.transAxes, fontsize=7, color=TOL["dark"], va="top")
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$M_H$ [$M_\odot$]")
        ax.set_ylabel(r"$C_{\max} / C_{\mathrm{c}}(\alpha)$")
        ax.set_xlim(M_H.min() * 0.9, M_H.max() * 1.1)
        ax.set_ylim(0.5, max(np.nanmax(ratio) * 1.15, 10.0))
        ax.legend(loc="lower right", fontsize=6.5)
        ax.text(M_H.min() * 1.1, max(np.nanmax(ratio) * 0.95, 8.0), "(a)",
                fontsize=8, fontweight="bold", color=TOL["dark"], va="top")

        ax = axes[1]
        ax.axhline(4.42, color=TOL["yellow"], ls="--", lw=1.0, label=r"$\beta \approx 10^{-5}$")
        ax.axhline(6.47, color=TOL["purple"], ls="--", lw=1.0, label=r"$\beta \approx 10^{-10}$")
        ax.axhspan(0, 4.0, color=TOL["green"], alpha=0.08, label="Observable PBH")
        ax.scatter(M_H, rarity, s=4, c=TOL["red"], alpha=0.7, zorder=5)
        ax.annotate(rf"min $C_{{\mathrm{{c}}}}/\sigma_0 \approx {mr:.0f}$",
                    xy=(M_H[mi], rarity[mi]), xytext=(0.15, 0.35),
                    textcoords="axes fraction", fontsize=6.5, color=TOL["dark"],
                    arrowprops=dict(arrowstyle="->", color=TOL["dark"], lw=0.8))
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_xlabel(r"$M_H$ [$M_\odot$]")
        ax.set_ylabel(r"$C_{\mathrm{c}} / \sigma_0$")
        ax.set_xlim(M_H.min() * 0.9, M_H.max() * 1.1)
        ax.set_ylim(1.0, max(np.nanmax(rarity) * 2.0, 50.0))
        ax.legend(loc="upper right", fontsize=6)
        ax.text(M_H.min() * 1.1, max(np.nanmax(rarity) * 1.8, 40.0), "(b)",
                fontsize=8, fontweight="bold", color=TOL["dark"], va="top")
        save_fig(fig, save_path)


# ── Main ──────────────────────────────────────────────────────────────────


def main() -> None:
    data = compute_compaction_data("outputs/simulations/pspectra/ps_Ezquiaga_dense700.json")
    plot_formation_comparison(data, get_path("diagnostics", "compaction_formation_comparison.png"))
    plot_collapse_barriers(data, get_path("diagnostics", "compaction_collapse_barriers.png"))


if __name__ == "__main__":
    main()
