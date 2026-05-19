"""
CAMB Scan Results Dashboard

Generates a standalone HTML dashboard with all key info from camb_scan JSONL logs.
Collapsible sections, configurable filtering by chi2/D2/suppression/dip position/N_star.

Usage:
  python scripts/camb_scan_dashboard.py                                                       # latest logs auto-detect
  python scripts/camb_scan_dashboard.py --phase1 <path> --phase2 <path>
  python scripts/camb_scan_dashboard.py --chi2-max-delta 50 --supp-min 10 --top-n 10           # relaxed filter, more plots
  python scripts/camb_scan_dashboard.py --no-golden --no-correlations --open                  # minimal dashboard
"""
import argparse
import base64
import json
import os
import subprocess
import sys
from datetime import datetime
from io import BytesIO

import numpy as np
from dataclasses import dataclass

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize

from scripts.plotting import TOL, get_path, make_filename
from scripts.planck_data import get_planck_data_asymmetric, C_ell_to_d_ell
from scripts.camb_wrapper import compute_cl_camb_powerlaw
from scripts.constants import As, k_pivot_phys, ROOT_DIR


@dataclass
class FilterConfig:
    chi2_max_delta: float = 5.0
    d2_max: float | None = None
    supp_min: float = 0.0
    nstar_min: float = 0.0
    dip_min: float | None = None
    dip_max: float | None = None
    top_n: int = 5
    table_top_n: int = 15
    hide_golden: bool = False
    hide_correlations: bool = False


CHI2_LCDM_LOW = 20.47
CHI2_LCDM_FULL = 2573.0
D2_LCDM = 1028.7
GOLDEN = {"phi0": 6.60, "y0": -0.736, "N_star": 52.59, "chi2": 20.23, "d2": 985.0}
OUTPUT_HTML = "camb_scan_dashboard.html"

NPC_LOW = Normalize(vmin=17, vmax=35)
NPC_FULL = Normalize(vmin=2570, vmax=2600)
CMAP = "viridis_r"


def load_jsonls(phase1_path, phase2_path):
    headers = {"phase1": None, "phase2": None}
    records = []
    for phase, path in [("phase1", phase1_path), ("phase2", phase2_path)]:
        if not path or not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("_type") == "header":
                    headers[phase] = rec
                else:
                    rec["_phase"] = phase
                    records.append(rec)
    return headers, records


def get_ells_from_headers(headers):
    for h in [headers["phase2"], headers["phase1"]]:
        if h and "ells" in h:
            ells = np.array(h["ells"])
            return ells[ells >= 2]
    return np.arange(2, 31)


def get_kphys_from_headers(headers):
    for h in [headers["phase2"], headers["phase1"]]:
        if h and "k_phys" in h:
            return np.array(h["k_phys"])
    return np.logspace(-5, 0, 80)


def get_planck():
    p_ells, D_p, D_lo, D_hi = get_planck_data_asymmetric()
    return p_ells, D_p, D_lo, D_hi


def get_lcdm(ell_max=30):
    ells, C, _, _ = compute_cl_camb_powerlaw(ell_max=ell_max)
    D = C_ell_to_d_ell(ells, C)
    return ells, C, D


def records_to_ok(records):
    return [r for r in records if r.get("status") == "ok"]


def build_ps_lookup(phase1_path, phase2_path):
    lookup = {}
    for path in [phase1_path, phase2_path]:
        if not path or not os.path.exists(path):
            continue
        with open(path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                    if r.get("status") == "ok" and "P_S" in r:
                        key = (round(r["phi0"], 4), round(r["y0"], 4), round(r["N_star"], 2))
                        lookup[key] = np.array(r["P_S"])
                except Exception:
                    pass
    return lookup


def attach_ps_from_lookup(records, lookup):
    for r in records:
        key = (round(r["phi0"], 4), round(r["y0"], 4), round(r["N_star"], 2))
        if key in lookup:
            r["P_S"] = lookup[key].tolist()
    return records


def prepare_records(records, use_full_chi2):
    for r in records:
        if use_full_chi2:
            r["_chi2"] = r.get("chi2_unbinned", r.get("chi2", 9999))
            r["_chi2_lcdm"] = r.get("chi2_lcdm_unbinned", r.get("chi2_lcdm", CHI2_LCDM_FULL))
            r["_D_ell"] = r.get("D_ell_full", r.get("D_ell", []))
            r["_chi2_bin"] = r.get("chi2_binned", r.get("chi2_binned_model", None))
            r["_mode"] = "full"
            if "d2" not in r and len(r["_D_ell"]) > 0:
                r["d2"] = round(r["_D_ell"][0], 1)
        else:
            r["_chi2"] = r.get("chi2", 9999)
            r["_chi2_lcdm"] = CHI2_LCDM_LOW
            r["_D_ell"] = r.get("D_ell", [])
            r["_chi2_bin"] = None
            r["_mode"] = "low"
    return records


def apply_filters(records, cfg, chi2_lcdm):
    def _passes(r):
        chi2_ok = r["_chi2"] <= chi2_lcdm + cfg.chi2_max_delta
        d2_ok = cfg.d2_max is None or r.get("d2", 9999) <= cfg.d2_max
        supp_ok = r.get("suppression_pct", 0) >= cfg.supp_min
        nstar_ok = r["N_star"] >= cfg.nstar_min
        dip = r.get("k_dip", -1)
        dip_ok = (cfg.dip_min is None or dip >= cfg.dip_min) and (cfg.dip_max is None or dip <= cfg.dip_max)
        return chi2_ok and d2_ok and supp_ok and nstar_ok and dip_ok
    return [r for r in records if _passes(r)]


def load_full_chi2(path):
    records = []
    if not path or not os.path.exists(path):
        return records
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if rec.get("status") == "ok":
                records.append(rec)
    return records


def img_to_b64(fig):
    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=200, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return base64.b64encode(buf.read()).decode("utf-8")

# ---------------------------------------------------------------------------
# SECTION 1: Executive Summary (HTML table, no plot)
# ---------------------------------------------------------------------------

def build_summary_html(ok_p1, ok_p2, all_ok, cfg):
    html = []

    # Metrics
    best_chi2 = min(all_ok, key=lambda r: r["_chi2"])
    best_supp = max(all_ok, key=lambda r: r.get("suppression_pct", 0))
    best_d2 = min(all_ok, key=lambda r: r["d2"])

    use_full = all_ok[0].get("_mode") == "full" if all_ok else False
    chi2_lcdm = CHI2_LCDM_FULL if use_full else CHI2_LCDM_LOW
    metric_label = "χ²_full" if use_full else "χ²_low"

    html.append('<div class="summary-grid">')
    html.append(f'<div class="summary-card"><span class="stat">{len(all_ok)}</span><span class="label">OK configs</span></div>')
    html.append(f'<div class="summary-card"><span class="stat">{best_chi2["_chi2"]:.1f}</span><span class="label">Best {metric_label}</span></div>')
    html.append(f'<div class="summary-card"><span class="stat">{best_supp.get("suppression_pct", 0):.1f}%</span><span class="label">Max suppression</span></div>')
    html.append(f'<div class="summary-card"><span class="stat">{best_d2["d2"]:.0f}</span><span class="label">Lowest D₂</span></div>')
    html.append('</div>')

    # Passing criteria
    passing = apply_filters(all_ok, cfg, chi2_lcdm)
    html.append(f'<p>Configs passing filter (Δχ² ≤ {cfg.chi2_max_delta:.0f}): <strong>{len(passing)}</strong></p>')
    html.append(f'<p>LCDM reference: {metric_label} = <strong>{chi2_lcdm:.1f}</strong></p>')

    # Best config detail
    bc = min(passing, key=lambda r: r["_chi2"]) if passing else best_chi2
    dchi2 = bc["_chi2"] - bc.get("_chi2_lcdm", chi2_lcdm)
    html.append('<div class="best-config">')
    html.append('<h3>Best Config</h3>')
    html.append(f'<table class="metric-table">')
    html.append(f'<tr><td>φ₀</td><td>{bc["phi0"]:.2f}</td></tr>')
    html.append(f'<tr><td>y₀</td><td>{bc["y0"]:+.3f}</td></tr>')
    html.append(f'<tr><td>N<sub>*</sub></td><td>{bc["N_star"]:.1f}</td></tr>')
    html.append(f'<tr><td>N<sub>total</sub></td><td>{bc.get("N_total", 0):.1f}</td></tr>')
    html.append(f'<tr><td>{metric_label}</td><td>{bc["_chi2"]:.1f}</td></tr>')
    html.append(f'<tr><td>Δ{metric_label} vs LCDM</td><td>{dchi2:+.1f}</td></tr>')
    if use_full and bc.get("_chi2_bin") is not None:
        html.append(f'<tr><td>χ²_binned</td><td>{bc["_chi2_bin"]:.1f}</td></tr>')
    html.append(f'<tr><td>D₂</td><td>{bc["d2"]:.1f} μK²</td></tr>')
    html.append(f'<tr><td>k<sub>dip</sub></td><td>{bc.get("k_dip", 0):.2e} Mpc⁻¹</td></tr>')
    html.append(f'<tr><td>Suppression</td><td>{bc.get("suppression_pct", 0):.1f}%</td></tr>')
    html.append('</table>')
    html.append('</div>')

    # Top 15 table
    passing_sorted = sorted(passing, key=lambda r: r["_chi2"])[:cfg.table_top_n]
    html.append(f'<h3>Top {cfg.table_top_n} by {metric_label}</h3>')
    html.append('<div class="table-wrap"><table class="data-table">')
    html.append('<tr><th>#</th><th>Phase</th><th>φ₀</th><th>y₀</th><th>N<sub>*</sub></th>'
                f'<th>{metric_label}</th><th>Δ</th>'
                f'{"<th>χ²_bin</th>" if use_full else ""}'
                '<th>D₂</th><th>k<sub>dip</sub></th><th>Supp%</th><th>N<sub>tot</sub></th></tr>')
    for i, r in enumerate(passing_sorted):
        phase_label = "Fine" if r.get("_phase") == "phase2" else "Broad"
        dchi2 = r["_chi2"] - r.get("_chi2_lcdm", chi2_lcdm)
        html.append(f'<tr><td>{i+1}</td><td>{phase_label}</td>'
                    f'<td>{r["phi0"]:.2f}</td><td>{r["y0"]:+.3f}</td><td>{r["N_star"]:.1f}</td>'
                    f'<td>{r["_chi2"]:.1f}</td><td>{dchi2:+.1f}</td>'
                    f'{"<td>" + str(r.get("_chi2_bin", "")) + "</td>" if use_full else ""}'
                    f'<td>{r["d2"]:.0f}</td><td>{r.get("k_dip", 0):.2e}</td>'
                    f'<td>{r.get("suppression_pct", 0):.1f}</td><td>{r.get("N_total", 0):.1f}</td></tr>')
    html.append('</table></div>')

    return "\n".join(html)

# ---------------------------------------------------------------------------
# SECTION 2: Parameter Space Maps
# ---------------------------------------------------------------------------

def plot_param_space(ok_records):
    phi0s = np.array([r["phi0"] for r in ok_records])
    y0s = np.array([r["y0"] for r in ok_records])
    chi2s = np.array([r["_chi2"] for r in ok_records])
    supps = np.array([r.get("suppression_pct", 0) for r in ok_records])

    use_full = ok_records[0].get("_mode") == "full" if ok_records else False
    npc = NPC_FULL if use_full else NPC_LOW

    fig, axes = plt.subplots(1, 2, figsize=(8, 3.5))

    sc1 = axes[0].scatter(phi0s, y0s, c=chi2s, cmap=CMAP, norm=npc, s=12, alpha=0.7, edgecolors="none")
    axes[0].set_xlabel(r"$\phi_0$ [$M_P$]", fontsize=11)
    axes[0].set_ylabel(r"$y_0$ [code units]", fontsize=11)
    axes[0].set_title(r"Colored by $\chi^2$", fontsize=12)
    cbar1 = plt.colorbar(sc1, ax=axes[0], shrink=0.75)
    cbar1.set_label(r"$\chi^2$", fontsize=10)

    sc2 = axes[1].scatter(phi0s, y0s, c=supps, cmap="plasma_r", s=12, alpha=0.7, edgecolors="none", vmin=0, vmax=75)
    axes[1].set_xlabel(r"$\phi_0$ [$M_P$]", fontsize=11)
    axes[1].set_ylabel(r"$y_0$ [code units]", fontsize=11)
    axes[1].set_title("Colored by Suppression %", fontsize=12)
    cbar2 = plt.colorbar(sc2, ax=axes[1], shrink=0.75)
    cbar2.set_label("Suppression %", fontsize=10)

    for ax in axes:
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    return img_to_b64(fig)

# ---------------------------------------------------------------------------
# SECTION 3: D_ell Overlay
# ---------------------------------------------------------------------------

def plot_dell_overlay(ok_records, ells, planck_data, lcdm_d, cfg):
    p_ells, D_p, D_lo, D_hi = planck_data
    ells_l, _, D_l = lcdm_d

    use_full = ok_records[0].get("_mode") == "full" if ok_records else False
    chi2_lcdm = ok_records[0].get("_chi2_lcdm", CHI2_LCDM_FULL if use_full else CHI2_LCDM_LOW)
    filtered = apply_filters(ok_records, cfg, chi2_lcdm)
    top = sorted(filtered, key=lambda r: r["_chi2"])[:cfg.top_n]
    if not top:
        top = sorted(ok_records, key=lambda r: r["_chi2"])[:5]

    if use_full:
        from scripts.chi2_analysis import load_planck_binned

    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2),
                             gridspec_kw={"width_ratios": [1, 2.2]})

    # ---- LEFT: low-ell (ℓ=2-30) with Commander data ----
    ax = axes[0]
    mask_p = p_ells <= 30
    ax.errorbar(p_ells[mask_p], D_p[mask_p], yerr=[D_lo[mask_p], D_hi[mask_p]],
                fmt="o", color=TOL["dark"], capsize=2, markersize=3, elinewidth=0.8,
                label="Planck", zorder=5, alpha=0.7)
    mask_l = ells_l <= 30
    ax.plot(ells_l[mask_l], D_l[mask_l], "--", color=TOL["grey"], lw=1.5, alpha=0.7, label=r"$\Lambda$CDM")

    colors = [TOL["red"], TOL["blue"], TOL["green"], TOL["purple"], TOL["teal"]]
    for i, r in enumerate(top):
        D = np.array(r["_D_ell"])
        c = colors[i % len(colors)]
        ell_plot = np.arange(2, 2 + len(D))
        label = rf"$\phi_0$={r['phi0']:.1f}, $y_0$={r['y0']:+.2f}"
        ax.plot(ell_plot[:29], D[:29], "-", color=c, lw=1.2, label=label, alpha=0.85)

    ax.set_xlabel(r"$\ell$", fontsize=11)
    ax.set_ylabel(r"$D_\ell^{TT}$ [$\mu$K$^2$]", fontsize=11)
    ax.legend(fontsize=6.5, loc="upper right", ncol=1, framealpha=0.8)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(1.5, 30.5)

    # ---- RIGHT: full-sky (ℓ=2-2500) with binned Planck data ----
    ax = axes[1]
    lcdm_plot = ells_l
    ax.plot(lcdm_plot, D_l, "--", color=TOL["grey"], lw=1.2, alpha=0.6, label=r"$\Lambda$CDM")

    if use_full:
        try:
            b_ells, b_D, b_lo, b_hi = load_planck_binned()
            b_sigma = (b_lo + b_hi) / 2
            ax.errorbar(b_ells, b_D, yerr=b_sigma, fmt="o", color=TOL["dark"],
                        capsize=1.5, markersize=2, elinewidth=0.6, alpha=0.5, label="Planck binned")
        except Exception:
            pass

    for i, r in enumerate(top):
        D = np.array(r["_D_ell"])
        c = colors[i % len(colors)]
        ell_plot = np.arange(2, 2 + len(D))
        ch = r["_chi2"]
        ax.plot(ell_plot, D, "-", color=c, lw=0.8, alpha=0.7,
                label=rf"$\chi^2$={ch:.0f}")

    ax.set_xlabel(r"$\ell$", fontsize=11)
    ax.set_ylabel(r"$D_\ell^{TT}$ [$\mu$K$^2$]", fontsize=11)
    ax.legend(fontsize=6.5, loc="upper right", ncol=1, framealpha=0.8)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(1.5, 2500)

    fig.tight_layout()
    return img_to_b64(fig)

# ---------------------------------------------------------------------------
# SECTION 4: P_S(k) Overlay
# ---------------------------------------------------------------------------

def plot_ps_overlay(ok_records, k_phys, planck_data, cfg):
    chi2_lcdm = ok_records[0]["_chi2_lcdm"] if ok_records else 2573
    filtered = apply_filters(ok_records, cfg, chi2_lcdm)
    top = sorted([r for r in filtered if "P_S" in r], key=lambda r: r["_chi2"])[:cfg.top_n]
    if not top:
        top = sorted([r for r in ok_records if "P_S" in r], key=lambda r: r["_chi2"])[:5]

    if not top:
        fig, ax = plt.subplots(figsize=(5, 3.2))
        ax.text(0.5, 0.5, "P_S(k) data not available\n(not stored in these logs)", transform=ax.transAxes,
                ha="center", va="center", fontsize=12, color=TOL["grey"])
        ax.set_xlabel(r"$k$ [Mpc$^{-1}$]", fontsize=11)
        fig.tight_layout()
        return img_to_b64(fig)

    ns_lcdm = 0.965
    ps_lcdm = As * (k_phys / k_pivot_phys) ** (ns_lcdm - 1.0)

    fig, ax = plt.subplots(figsize=(5, 3.2))

    ax.loglog(k_phys, ps_lcdm, "--", color=TOL["grey"], lw=1.5, alpha=0.6, label=r"$\Lambda$CDM")

    colors = [TOL["red"], TOL["blue"], TOL["green"], TOL["purple"], TOL["teal"]]
    for i, r in enumerate(top):
        P = np.array(r["P_S"])[:len(k_phys)]
        c = colors[i % len(colors)]
        label = rf"$\phi_0$={r['phi0']:.1f}, $y_0$={r['y0']:+.2f}, $\chi^2$={r['_chi2']:.0f}"
        ax.loglog(k_phys, P, "-", color=c, lw=1.2, label=label, alpha=0.85)

    ax.axvline(k_pivot_phys, color=TOL["grey"], ls=":", lw=1, alpha=0.4, label=f"$k_{{\\rm pivot}}$={k_pivot_phys:.3f}")
    ax.axvspan(1.4e-4, 2.1e-3, color=TOL["yellow"], alpha=0.08, label=r"CMB low-$\ell$ window")

    ax.set_xlabel(r"$k$ [Mpc$^{-1}$]", fontsize=11)
    ax.set_ylabel(r"$\mathcal{P}_{\mathcal{R}}(k)$", fontsize=11)
    ax.legend(fontsize=7, loc="best", framealpha=0.8)
    ax.grid(True, alpha=0.2, which="both")

    fig.tight_layout()
    return img_to_b64(fig)

# ---------------------------------------------------------------------------
# SECTION 5: Correlation Grid
# ---------------------------------------------------------------------------

def plot_correlations(ok_records):
    chi2s = np.array([r["_chi2"] for r in ok_records])
    supps = np.array([r.get("suppression_pct", 0) for r in ok_records])
    d2s = np.array([r["d2"] for r in ok_records])
    nstars = np.array([r["N_star"] for r in ok_records])
    kdips = np.array([r.get("k_dip", -1) for r in ok_records])
    ntotals = np.array([r.get("N_total", 0) for r in ok_records])

    valid = (kdips > 0) & (np.isfinite(chi2s)) & (chi2s < 200)

    fig, axes = plt.subplots(2, 3, figsize=(9, 5.5))
    pairs = [
        (chi2s[valid], supps[valid], r"$\chi^2$", "Suppression %", TOL["red"]),
        (chi2s[valid], d2s[valid], r"$\chi^2$", r"$D_2$ [$\mu$K$^2$]", TOL["blue"]),
        (nstars[valid], chi2s[valid], r"$N_*$", r"$\chi^2$", TOL["green"]),
        (kdips[valid], chi2s[valid], r"$k_{\rm dip}$ [Mpc$^{-1}$]", r"$\chi^2$", TOL["purple"]),
        (supps[valid], d2s[valid], "Suppression %", r"$D_2$ [$\mu$K$^2$]", TOL["teal"]),
        (ntotals[valid], supps[valid], r"$N_{\rm total}$", "Suppression %", TOL["yellow"]),
    ]

    for ax, (x, y, xl, yl, c) in zip(axes.flat, pairs):
        ax.scatter(x, y, s=8, c=c, alpha=0.4, edgecolors="none")
        ax.set_xlabel(xl, fontsize=10)
        ax.set_ylabel(yl, fontsize=10)
        ax.grid(True, alpha=0.2)

    fig.tight_layout()
    return img_to_b64(fig)

# ---------------------------------------------------------------------------
# SECTION 6: Golden Comparison
# ---------------------------------------------------------------------------

def plot_golden_comparison(ok_records, ells, k_phys, planck_data, lcdm_d, cfg):
    p_ells, D_p, D_lo, D_hi = planck_data
    ells_l, _, D_l = lcdm_d

    use_full = ok_records[0].get("_mode") == "full" if ok_records else False
    chi2_lcdm = ok_records[0]["_chi2_lcdm"] if ok_records else 2573

    passing = apply_filters(ok_records, cfg, chi2_lcdm)
    best = min(passing, key=lambda r: r["_chi2"]) if passing else min(ok_records, key=lambda r: r["_chi2"])

    if use_full:
        fig, axes = plt.subplots(1, 3, figsize=(11.5, 3.2),
                                 gridspec_kw={"width_ratios": [1.2, 2.5, 2]})
    else:
        fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.2))

    # ---- D_ell panel: low-ℓ ----
    ax = axes[0]
    mask_p = p_ells <= 30
    ax.errorbar(p_ells[mask_p], D_p[mask_p], yerr=[D_lo[mask_p], D_hi[mask_p]],
                fmt="o", color=TOL["dark"], capsize=2, markersize=3, elinewidth=0.8,
                label="Planck 2018", zorder=5, alpha=0.6)
    mask_l = ells_l <= 30
    ax.plot(ells_l[mask_l], D_l[mask_l], "--", color=TOL["grey"], lw=1.5, alpha=0.6, label=r"$\Lambda$CDM")

    D_new = np.array(best["_D_ell"])
    ax.plot(np.arange(2, 2+len(D_new))[:29], D_new[:29], "-", color=TOL["red"], lw=1.8,
            label=rf"New best ($\chi^2$={best['_chi2']:.1f})")

    golden_near = min(ok_records, key=lambda r: abs(r["phi0"]-GOLDEN["phi0"]) + abs(r["y0"]-GOLDEN["y0"])*0.5 + abs(r["N_star"]-GOLDEN["N_star"])*0.05)
    D_golden = np.array(golden_near["_D_ell"])
    ax.plot(np.arange(2, 2+len(D_golden))[:29], D_golden[:29], "-.", color=TOL["purple"], lw=1.5, alpha=0.7,
            label=rf"Golden-like ($\chi^2$={golden_near['_chi2']:.1f})")

    ax.set_xlabel(r"$\ell$", fontsize=11)
    ax.set_ylabel(r"$D_\ell^{TT}$ [$\mu$K$^2$]", fontsize=11)
    ax.legend(fontsize=8, loc="upper right", framealpha=0.8)
    ax.grid(True, alpha=0.2)
    ax.set_xlim(1.5, 30.5)

    # ---- Full-sky D_ell panel (only if full-chi2 mode) ----
    if use_full:
        ax = axes[1]
        from scripts.chi2_analysis import load_planck_binned
        ax.plot(ells_l, D_l, "--", color=TOL["grey"], lw=1.2, alpha=0.6, label=r"$\Lambda$CDM")
        try:
            b_ells, b_D, b_lo, b_hi = load_planck_binned()
            b_sigma = (b_lo + b_hi) / 2
            ax.errorbar(b_ells, b_D, yerr=b_sigma, fmt="o", color=TOL["dark"],
                        capsize=1.5, markersize=2, elinewidth=0.6, alpha=0.4, label="Planck")
        except Exception:
            pass

        ell_plot = np.arange(2, 2 + len(D_new))
        ax.plot(ell_plot, D_new, "-", color=TOL["red"], lw=1.2, alpha=0.8, label="New best")
        ax.plot(ell_plot[:len(D_golden)], D_golden, "-.", color=TOL["purple"], lw=1.2, alpha=0.7, label="Golden-like")
        ax.set_xlabel(r"$\ell$", fontsize=11)
        ax.set_ylabel(r"$D_\ell^{TT}$ [$\mu$K$^2$]", fontsize=11)
        ax.legend(fontsize=8, loc="upper right", framealpha=0.8)
        ax.grid(True, alpha=0.2)
        ax.set_xlim(1.5, 2500)
        ax2 = axes[2]
    else:
        ax2 = axes[1]

    # ---- P_S panel ----
    ax = ax2
    has_ps_best = "P_S" in best
    has_ps_golden = "P_S" in golden_near
    ns_lcdm = 0.965
    ps_lcdm = As * (k_phys / k_pivot_phys) ** (ns_lcdm - 1.0)
    ax.loglog(k_phys, ps_lcdm, "--", color=TOL["grey"], lw=1.5, alpha=0.5, label=r"$\Lambda$CDM")

    if has_ps_best:
        P_new = np.array(best["P_S"])[:len(k_phys)]
        ax.loglog(k_phys, P_new, "-", color=TOL["red"], lw=1.8, label=rf"New best")

    if has_ps_golden:
        P_golden = np.array(golden_near["P_S"])[:len(k_phys)]
        ax.loglog(k_phys, P_golden, "-.", color=TOL["purple"], lw=1.5, alpha=0.7, label=rf"Golden-like")

    if not has_ps_best and not has_ps_golden:
        ax.text(0.5, 0.5, "P_S(k) data not available", transform=ax.transAxes,
                ha="center", va="center", fontsize=11, color=TOL["grey"])

    ax.axvspan(1.4e-4, 2.1e-3, color=TOL["yellow"], alpha=0.08, label=r"CMB low-$\ell$")
    ax.axvline(k_pivot_phys, color=TOL["grey"], ls=":", lw=1, alpha=0.4)

    ax.set_xlabel(r"$k$ [Mpc$^{-1}$]", fontsize=11)
    ax.set_ylabel(r"$\mathcal{P}_{\mathcal{R}}(k)$", fontsize=11)
    ax.legend(fontsize=8, loc="best", framealpha=0.8)
    ax.grid(True, alpha=0.2, which="both")

    fig.tight_layout()

    # Also save the individual PNG
    path = get_path("diagnostics", make_filename("golden_comparison", best["phi0"], best["y0"], best["N_star"], ext=".png"))
    fig.savefig(path, dpi=300, bbox_inches="tight")
    print(f"  Saved: {path}")

    html = []
    html.append(f'<img src="data:image/png;base64,{img_to_b64(fig)}" class="dashboard-img">')

    # Metrics comparison table
    chi2_lcdm_label = best["_chi2_lcdm"] if "full" in str(best.get("_mode", "")) else CHI2_LCDM_LOW
    metric_label = "χ²_full" if best.get("_mode") == "full" else "χ²_low"
    golden_chi2_use = GOLDEN["chi2"] if best.get("_mode") != "full" else GOLDEN["chi2"]

    html.append('<h3>Metrics Comparison</h3>')
    html.append('<table class="metric-table" style="margin:0 auto; min-width:450px;">')
    html.append(f'<tr><th>Metric</th><th>Golden (φ₀=6.60, y₀=-0.736, N*=52.6)</th><th>New Best</th></tr>')
    html.append(f'<tr><td>{metric_label}</td><td>{golden_chi2_use:.1f}</td><td>{best["_chi2"]:.1f}</td></tr>')
    dchi2 = best["_chi2"] - chi2_lcdm_label
    dchi2_g = golden_chi2_use - chi2_lcdm_label
    html.append(f'<tr><td>Δ{metric_label} vs LCDM</td><td>{dchi2_g:+.1f}</td><td>{dchi2:+.1f}</td></tr>')
    if best.get("_mode") == "full" and best.get("_chi2_bin") is not None:
        html.append(f'<tr><td>χ²_binned</td><td>—</td><td>{best["_chi2_bin"]:.1f}</td></tr>')
    html.append(f'<tr><td>D₂ [μK²]</td><td>{GOLDEN["d2"]:.0f}</td><td>{best["d2"]:.0f}</td></tr>')
    html.append(f'<tr><td>φ₀</td><td>{GOLDEN["phi0"]:.2f}</td><td>{best["phi0"]:.2f}</td></tr>')
    html.append(f'<tr><td>y₀</td><td>{GOLDEN["y0"]:+.3f}</td><td>{best["y0"]:+.3f}</td></tr>')
    html.append(f'<tr><td>N<sub>*</sub></td><td>{GOLDEN["N_star"]:.1f}</td><td>{best["N_star"]:.1f}</td></tr>')
    html.append(f'<tr><td>Suppression</td><td>~63%</td><td>{best.get("suppression_pct", 0):.1f}%</td></tr>')
    improvement = golden_chi2_use - best["_chi2"]
    html.append(f'<tr><td><strong>{metric_label} improvement</strong></td><td></td><td><strong>{improvement:+.1f}</strong></td></tr>')
    html.append('</table>')

    return "\n".join(html)

# ---------------------------------------------------------------------------
# HTML Assembly
# ---------------------------------------------------------------------------

HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>CAMB Scan Results Dashboard</title>
<style>
  *, *::before, *::after {{ box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
         margin: 0; padding: 0; background: #f5f5f5; color: #222; line-height: 1.5; }}
  .sidebar {{ position: fixed; left: 0; top: 0; width: 220px; height: 100vh;
             background: #1a1a2e; color: #eee; padding: 20px 0; overflow-y: auto; }}
  .sidebar h2 {{ font-size: 14px; padding: 8px 20px; margin: 0; color: #888; text-transform: uppercase; letter-spacing: 1px; }}
  .sidebar a {{ display: block; padding: 8px 20px; color: #ccc; text-decoration: none; font-size: 14px; }}
  .sidebar a:hover {{ background: #16213e; color: #fff; }}
  .content {{ margin-left: 220px; padding: 30px 40px; max-width: 1100px; }}
  h1 {{ font-size: 24px; margin-bottom: 5px; color: #1a1a2e; }}
  .subtitle {{ color: #666; font-size: 14px; margin-bottom: 30px; }}
  h2 {{ font-size: 20px; color: #1a1a2e; border-bottom: 2px solid #e0e0e0; padding-bottom: 8px; margin-top: 40px; }}
  h3 {{ font-size: 16px; color: #333; margin-top: 20px; }}
  .summary-grid {{ display: flex; gap: 15px; flex-wrap: wrap; margin: 20px 0; }}
  .summary-card {{ background: #fff; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.08);
                  padding: 15px 20px; flex: 1; min-width: 120px; text-align: center; }}
  .summary-card .stat {{ display: block; font-size: 28px; font-weight: 700; color: #1a1a2e; }}
  .summary-card .label {{ display: block; font-size: 12px; color: #888; margin-top: 4px; }}
  .table-wrap {{ overflow-x: auto; }}
  .data-table {{ border-collapse: collapse; width: 100%; font-size: 13px; background: #fff;
                box-shadow: 0 1px 4px rgba(0,0,0,0.06); border-radius: 6px; }}
  .data-table th {{ background: #1a1a2e; color: #fff; padding: 8px 10px; text-align: center;
                    font-weight: 600; font-size: 12px; white-space: nowrap; }}
  .data-table td {{ padding: 6px 10px; text-align: center; border-bottom: 1px solid #eee; }}
  .data-table tr:hover td {{ background: #f0f0fa; }}
  .metric-table {{ border-collapse: collapse; font-size: 14px; background: #fff;
                  box-shadow: 0 1px 4px rgba(0,0,0,0.06); border-radius: 6px; width: auto; }}
  .metric-table td, .metric-table th {{ padding: 6px 14px; border-bottom: 1px solid #eee; text-align: left; }}
  .metric-table th {{ background: #1a1a2e; color: #fff; font-size: 12px; }}
  .best-config {{ background: #fff; border-radius: 8px; box-shadow: 0 1px 4px rgba(0,0,0,0.08);
                 padding: 15px 20px; display: inline-block; margin: 10px 0; }}
  .best-config h3 {{ margin-top: 0; }}
  .dashboard-img {{ max-width: 100%; height: auto; display: block; margin: 15px 0;
                   border-radius: 6px; box-shadow: 0 1px 6px rgba(0,0,0,0.1); }}
  .footer {{ margin-top: 40px; padding: 15px 0; border-top: 1px solid #ddd;
             font-size: 12px; color: #888; text-align: center; }}
  @media print {{ .sidebar {{ display: none; }} .content {{ margin-left: 0; padding: 20px; }} }}
</style>
</head>
<body>
<div class="sidebar">
  <h2>Navigation</h2>
  {sidebar}
</div>
<div class="content">
  <h1>CAMB Scan Results Dashboard</h1>
  <div class="subtitle">Generated: {date} | Scan: {phase1_name}, {phase2_name}</div>

  <details open>
  <summary><h2 style="display:inline; font-size:20px; color:#1a1a2e; border:none; padding:0; margin:0;">1. Executive Summary</h2></summary>
  <section id="summary">
    {summary}
  </section>
  </details>

  <details>
  <summary><h2 style="display:inline; font-size:20px; color:#1a1a2e; border:none; padding:0; margin:0;">2. Parameter Space Maps</h2></summary>
  <section id="param-space">
    <img src="data:image/png;base64,{param_space}" class="dashboard-img">
  </section>
  </details>

  <details>
  <summary><h2 style="display:inline; font-size:20px; color:#1a1a2e; border:none; padding:0; margin:0;">3. D<sub>ℓ</sub> Overlay</h2></summary>
  <section id="dell-overlay">
    <img src="data:image/png;base64,{dell_overlay}" class="dashboard-img">
  </section>
  </details>

  <details>
  <summary><h2 style="display:inline; font-size:20px; color:#1a1a2e; border:none; padding:0; margin:0;">4. P<sub>S</sub>(k) Overlay</h2></summary>
  <section id="ps-overlay">
    <img src="data:image/png;base64,{ps_overlay}" class="dashboard-img">
  </section>
  </details>

  <details>
  <summary><h2 style="display:inline; font-size:20px; color:#1a1a2e; border:none; padding:0; margin:0;">5. Correlation Plots</h2></summary>
  <section id="correlations">
    {corr_section}
  </section>
  </details>

  <details>
  <summary><h2 style="display:inline; font-size:20px; color:#1a1a2e; border:none; padding:0; margin:0;">6. Golden Comparison</h2></summary>
  <section id="golden">
    {golden_section}
  </section>
  </details>

  <script>
  document.querySelectorAll('.sidebar a').forEach(function(a) {{
    a.addEventListener('click', function(e) {{
      var target = document.querySelector(a.getAttribute('href'));
      if (target) {{
        var details = target.closest('details');
        if (details) details.open = true;
      }}
    }});
  }});
  </script>

  <div class="footer">
    CMB Anomaly Project &mdash; Higgs USR Inflation (ξ=15000, λ=0.13)
  </div>
</div>
</body>
</html>"""


def build_html(summary_html, param_space_b64, dell_b64, ps_b64, corr_b64, golden_html,
               phase1_path, phase2_path, cfg):
    p1_name = os.path.basename(phase1_path) if phase1_path else "N/A"
    p2_name = os.path.basename(phase2_path) if phase2_path else "N/A"
    date_str = datetime.now().strftime("%Y-%m-%d %H:%M")

    sidebar_links = []
    sidebar_links.append('<a href="#summary">1. Executive Summary</a>')
    sidebar_links.append('<a href="#param-space">2. Parameter Space</a>')
    sidebar_links.append('<a href="#dell-overlay">3. D<sub>ℓ</sub> Overlay</a>')
    sidebar_links.append('<a href="#ps-overlay">4. P<sub>S</sub>(k) Overlay</a>')
    if not cfg.hide_correlations:
        sidebar_links.append('<a href="#correlations">5. Correlations</a>')
    if not cfg.hide_golden:
        sidebar_links.append('<a href="#golden">6. Golden Comparison</a>')
    sidebar = "\n    ".join(sidebar_links)

    if not cfg.hide_correlations and corr_b64:
        corr_section = f'<p>Pairwise relationships between key metrics across successful configs.</p>\n    <img src="data:image/png;base64,{corr_b64}" class="dashboard-img">'
    else:
        corr_section = ""

    if not cfg.hide_golden and golden_html:
        golden_section = golden_html
    else:
        golden_section = ""

    return HTML_TEMPLATE.format(
        sidebar=sidebar,
        date=date_str,
        phase1_name=p1_name,
        phase2_name=p2_name,
        summary=summary_html,
        param_space=param_space_b64,
        dell_overlay=dell_b64,
        ps_overlay=ps_b64,
        corr_section=corr_section,
        golden_section=golden_section,
    )


def auto_detect_latest():
    log_dir = get_path("logs", "")
    candidates = [f for f in os.listdir(log_dir) if f.startswith("camb_phase") and f.endswith(".jsonl")]
    phase1 = sorted([f for f in candidates if "phase1" in f]) or [None]
    phase2 = sorted([f for f in candidates if "phase2" in f]) or [None]
    return (
        os.path.join(log_dir, phase1[-1]) if phase1[0] else None,
        os.path.join(log_dir, phase2[-1]) if phase2[0] else None,
    )


def auto_detect_full_chi2():
    log_dir = get_path("logs", "")
    candidates = sorted([f for f in os.listdir(log_dir) if f.startswith("camb_full_chi2") and f.endswith(".jsonl")])
    return os.path.join(log_dir, candidates[-1]) if candidates else None


def main():
    p = argparse.ArgumentParser(description="CAMB Scan Dashboard")
    p.add_argument("--phase1", type=str, default=None, help="Phase 1 JSONL path")
    p.add_argument("--phase2", type=str, default=None, help="Phase 2 JSONL path")
    p.add_argument("--full-chi2", type=str, default=None, help="Full chi2 JSONL path")
    p.add_argument("--open", action="store_true", help="Open in browser after generation")
    p.add_argument("--chi2-max-delta", type=float, default=5.0,
                   help="Max Δχ² vs LCDM for passing filter (default: 5)")
    p.add_argument("--d2-max", type=float, default=None,
                   help="Max D₂ [μK²] for passing filter")
    p.add_argument("--supp-min", type=float, default=0.0,
                   help="Min suppression %% for passing filter (default: 0)")
    p.add_argument("--nstar-min", type=float, default=0.0,
                   help="Min N_star for passing filter (default: 0)")
    p.add_argument("--dip-min", type=float, default=None,
                   help="Min k_dip [Mpc⁻¹] for passing filter")
    p.add_argument("--dip-max", type=float, default=None,
                   help="Max k_dip [Mpc⁻¹] for passing filter")
    p.add_argument("--top-n", type=int, default=5,
                   help="Number of configs in overlay plots (default: 5)")
    p.add_argument("--table-top-n", type=int, default=15,
                   help="Number of configs in summary table (default: 15)")
    p.add_argument("--no-golden", action="store_true",
                   help="Skip golden comparison section")
    p.add_argument("--no-correlations", action="store_true",
                   help="Skip correlation plots section")
    args = p.parse_args()

    filter_config = FilterConfig(
        chi2_max_delta=args.chi2_max_delta,
        d2_max=args.d2_max,
        supp_min=args.supp_min,
        nstar_min=args.nstar_min,
        dip_min=args.dip_min,
        dip_max=args.dip_max,
        top_n=args.top_n,
        table_top_n=args.table_top_n,
        hide_golden=args.no_golden,
        hide_correlations=args.no_correlations,
    )

    phase1_path = args.phase1
    phase2_path = args.phase2
    full_chi2_path = args.full_chi2

    # Auto-detect full-chi2 if not specified
    if not full_chi2_path:
        full_chi2_path = auto_detect_full_chi2()
        if full_chi2_path:
            print(f"Auto-detected full-chi2: {full_chi2_path}")

    use_full_chi2 = full_chi2_path is not None and os.path.exists(full_chi2_path)

    if not use_full_chi2:
        # Legacy mode — load original phase logs
        if not phase1_path and not phase2_path:
            phase1_path, phase2_path = auto_detect_latest()
            print(f"Auto-detected:")
            print(f"  Phase 1: {phase1_path}")
            print(f"  Phase 2: {phase2_path}")

        if not phase1_path and not phase2_path:
            print("ERROR: No log files found. Specify --phase1 and/or --phase2.")
            sys.exit(1)

        print("Loading data (low-ℓ mode)...")
        headers, all_records = load_jsonls(phase1_path, phase2_path)
        ok_records = records_to_ok(all_records)
        print(f"  Total records: {len(all_records)}, OK: {len(ok_records)}")

        print("Loading LCDM baseline and Planck data...")
        ells = get_ells_from_headers(headers)
        k_phys = get_kphys_from_headers(headers)
        planck_data = get_planck()
        lcdm_data = get_lcdm(ell_max=30)

        # Auto-detect inline full-chi2 in phase logs
        if ok_records:
            median_chi2 = np.median([r.get("chi2", 0) for r in ok_records])
            if median_chi2 > 2000:
                print("  Auto-detected inline full-chi2 in phase logs")
                use_full_chi2 = True
                lcdm_data = get_lcdm(ell_max=2500)

    else:
        # Full-chi2 mode
        print(f"Loading data (full-chi² mode)...")
        ok_records = load_full_chi2(full_chi2_path)
        print(f"  OK records: {len(ok_records)}")

        # Get k_phys from phase headers for P_S plots
        if not phase1_path and not phase2_path:
            phase1_path, phase2_path = auto_detect_latest()
        headers, _ = load_jsonls(phase1_path, phase2_path)
        ells = get_ells_from_headers(headers) if headers else np.arange(2, 31)
        k_phys = get_kphys_from_headers(headers) if headers else np.logspace(-5, 0, 80)

        # Attach P_S from original logs
        ps_lookup = build_ps_lookup(phase1_path, phase2_path)
        ok_records = attach_ps_from_lookup(ok_records, ps_lookup)
        found_ps = sum(1 for r in ok_records if "P_S" in r)
        print(f"  P_S attached: {found_ps}/{len(ok_records)}")

        planck_data = get_planck()
        lcdm_data = get_lcdm(ell_max=2500)

    # Normalize fields
    ok_records = prepare_records(ok_records, use_full_chi2)

    chi2_lcdm = CHI2_LCDM_FULL if use_full_chi2 else CHI2_LCDM_LOW
    filtered_records = apply_filters(ok_records, filter_config, chi2_lcdm)

    print("Building dashboard...")
    print(f"  Mode: {'full-ℓ' if use_full_chi2 else 'low-ℓ'} (LCDM χ² = {chi2_lcdm})")
    print(f"  Filter: Δχ² ≤ {filter_config.chi2_max_delta:.0f}, D₂ {f'≤ {filter_config.d2_max}' if filter_config.d2_max else 'any'}, supp ≥ {filter_config.supp_min}%, N* ≥ {filter_config.nstar_min}")
    print(f"  Records: {len(filtered_records)}/{len(ok_records)} pass")

    # Split by phase for stats
    ok_p1 = [r for r in ok_records if r.get("_phase") == "phase1" or r.get("_source_phase") == "phase1"]
    ok_p2 = [r for r in ok_records if r.get("_phase") == "phase2" or r.get("_source_phase") == "phase2"]

    section_count = 6
    section_i = 0

    # Section 1: Summary HTML
    section_i += 1
    print(f"  {section_i}/{section_count} Executive summary...")
    summary_html = build_summary_html(ok_p1, ok_p2, ok_records, filter_config)

    # Section 2: Parameter space
    section_i += 1
    print(f"  {section_i}/{section_count} Parameter space maps...")
    param_b64 = plot_param_space(filtered_records)

    # Section 3: D_ell overlay
    section_i += 1
    print(f"  {section_i}/{section_count} D_ell overlay...")
    dell_b64 = plot_dell_overlay(filtered_records, ells, planck_data, lcdm_data, filter_config)

    # Section 4: P_S(k) overlay
    section_i += 1
    print(f"  {section_i}/{section_count} P_S(k) overlay...")
    ps_b64 = plot_ps_overlay(filtered_records, k_phys, planck_data, filter_config)

    # Section 5: Correlations
    section_i += 1
    if not filter_config.hide_correlations:
        print(f"  {section_i}/{section_count} Correlation plots...")
        corr_b64 = plot_correlations(filtered_records)
    else:
        print(f"  {section_i}/{section_count} Correlation plots... skipped")
        corr_b64 = None

    # Section 6: Golden comparison
    section_i += 1
    if not filter_config.hide_golden:
        print(f"  {section_i}/{section_count} Golden comparison...")
        golden_html = plot_golden_comparison(filtered_records, ells, k_phys, planck_data, lcdm_data, filter_config)
    else:
        print(f"  {section_i}/{section_count} Golden comparison... skipped")
        golden_html = None

    # Assemble HTML
    print("Assembling HTML...")
    html = build_html(summary_html, param_b64, dell_b64, ps_b64, corr_b64, golden_html,
                      phase1_path, phase2_path, filter_config)

    out_path = get_path("diagnostics", OUTPUT_HTML)
    with open(out_path, "w") as f:
        f.write(html)
    print(f"\nDashboard saved: {out_path}")
    print(f"  File size: {os.path.getsize(out_path) / 1024:.0f} KB")

    if args.open:
        try:
            subprocess.run(["xdg-open", out_path], check=False)
        except Exception:
            try:
                subprocess.run(["open", out_path], check=False)
            except Exception:
                print(f"  Open manually: {out_path}")


if __name__ == "__main__":
    main()
