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

from scripts.compaction import (
    _compute_zeta_profile_vectorized,
    _simpson_weights,
    beta_f_compaction,
)
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

    print("  σ₀² for full k-grid...", flush=True)
    ln_k = np.log(k_phys)
    w = _simpson_weights(ln_k)
    rp = np.concatenate([[0.0], np.logspace(-3.0, 1.5, 499)])
    s2_all: np.ndarray = np.empty(len(k_phys))
    for i in range(0, len(k_phys), 5):
        _, _, s2 = _compute_zeta_profile_vectorized(ln_k, w, k_phys, P_S, rp, 1.0 / k_phys[i], 1.0)
        s2_all[i] = s2
    mask = s2_all > 0
    if mask.sum() > 1:
        s2_all = np.where(mask, s2_all, np.interp(k_phys, k_phys[mask], s2_all[mask]))

    return {
        "k": k_phys, "ps": P_S, "k_sub": k_sub, "ps_sub": ps_sub,
        "s2_all": s2_all, "beta_f": beta_f, "M_pbh": M_pbh,
        "C_max": meta["C_max_arr"], "C_c": meta["C_c_arr"],
        "M_H": meta["M_H_arr"],
        "sigma0": np.sqrt(np.maximum(meta.get("sigma0_arr", np.zeros_like(beta_f)), 1e-300)),
    }


# ── Plot 1: Smoothing penalty ────────────────────────────────────────────


def plot_smoothing_penalty(data: dict, save_path: str) -> None:
    """Log-log: P_S(k) vs σ₀²(R) showing smoothing suppression."""
    k, ps = data["k"], data["ps"]
    s2 = data["s2_all"]
    sub_idx = np.arange(0, len(k), 5)
    valid = (s2[sub_idx] > 0) & (ps[sub_idx] > 0)
    ratio = np.where(valid, ps[sub_idx] / s2[sub_idx], 1.0)
    mr = float(np.nanmax(ratio))
    mr_k = float(k[sub_idx][np.argmax(ratio)])

    with plt.rc_context(PAPER_RCPARAMS):
        fig, ax = plt.subplots(figsize=(3.5, 2.8))
        ax.loglog(k, ps, "-", color=TOL["blue"], lw=0.8, alpha=0.7,
                  label=r"$\mathcal{P}_{\mathcal{S}}(k)$", zorder=3)
        ax.loglog(k, s2, "--", color=TOL["red"], lw=1.5,
                  label=r"$\sigma_0^2(R)$", zorder=4)
        ax.scatter(k[sub_idx], s2[sub_idx], s=6, color=TOL["red"], alpha=0.5, zorder=5)
        ax.axvspan(3e-4, 0.1, alpha=0.06, color=TOL["grey"], zorder=0)
        ax.annotate(rf"$\sigma_0^2$ suppression $\sim 10^{{{int(np.log10(mr))}}}\times$",
                    xy=(mr_k, s2[sub_idx][np.argmax(ratio)]),
                    xytext=(0.45, 0.65), textcoords="axes fraction",
                    fontsize=6.5, color=TOL["dark"],
                    arrowprops=dict(arrowstyle="->", color=TOL["dark"], lw=0.8))
        ax.set_xlabel(r"$k$ [Mpc$^{-1}$]")
        ax.set_ylabel(r"$\mathcal{P}_{\mathcal{S}}(k)$ and $\sigma_0^2(R=1/k)$")
        ax.legend(loc="lower left", fontsize=6.5)
        ax.set_xlim(k.min(), k.max())
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
        ax.set_xlabel(r"$M_H$ [$M_\odot$]")
        ax.set_ylabel(r"$C_{\max} / C_{\mathrm{c}}(\alpha)$")
        ym = max(np.nanmax(ratio) * 1.15, 2.0)
        ax.set_xlim(M_H.min() * 0.9, M_H.max() * 1.1)
        ax.set_ylim(0, ym)
        ax.legend(loc="lower right", fontsize=6.5)
        ax.text(M_H.min() * 1.1, ym * 0.95, "(a)", fontsize=8, fontweight="bold",
                color=TOL["dark"], va="top")

        ax = axes[1]
        ax.axhline(4.42, color=TOL["yellow"], ls="--", lw=1.0, label=r"$\beta \approx 10^{-5}$")
        ax.axhline(6.47, color=TOL["purple"], ls="--", lw=1.0, label=r"$\beta \approx 10^{-10}$")
        ax.axhspan(0, 4.0, color=TOL["green"], alpha=0.08, label="Observable PBH")
        ax.scatter(M_H, rarity, s=4, c=TOL["red"], alpha=0.7, zorder=5)
        ax.annotate(rf"min $C_{{\mathrm{{c}}}}/\sigma_0 \approx {mr:.0f}$",
                    xy=(M_H[mi], rarity[mi]), xytext=(0.45, 0.55),
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
    plot_smoothing_penalty(data, get_path("diagnostics", "compaction_smoothing_penalty.png"))
    plot_collapse_barriers(data, get_path("diagnostics", "compaction_collapse_barriers.png"))


if __name__ == "__main__":
    main()
