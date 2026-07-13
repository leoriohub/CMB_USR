#!/usr/bin/env python3
"""
PBH Abundance Comparison — 2x4 publication grid.
Rows: Press-Schechter, Compaction formation.
Columns: Chisholm, BHL, Eddington, PR accretion.
Uses cached P_S(k) — no MS solver re-run.
"""

import json, os, sys, time
import numpy as np

from scripts.constants import k_eq_default, M_eq_default, gamma_default

GAMMA = gamma_default
K_EQ = k_eq_default
M_EQ = M_eq_default
ZETA_C = 0.077

# ── Load cached P_S(k) ──────────────────────────────────────────────────
candidates = [
    "/home/diego/Projects/CMB_Anomaly/outputs/simulations/pspectra/"
    "ps_Ezquiaga_dense700.json",
]
for p in candidates:
    if os.path.exists(p):
        ps_path = p; break
else:
    sys.exit("No cached P_S(k) found")

with open(ps_path) as f:
    d = json.load(f)
if 'spectrum' in d and isinstance(d['spectrum'], dict):
    k_phys = np.array(d['spectrum']['k_phys'])
    P_S = np.array(d['spectrum']['P_S'])
else:
    k_phys = np.array(d['k_phys'])
    P_S = np.array(d['P_S'])

# ── Imports ─────────────────────────────────────────────────────────────
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scripts.accretion import ChisholmAccretion, EddingtonAccretion, PBHAccretion, PR_Accretion
from scripts.full_pbh_pipeline import beta_equality, beta_f_press_schechter, mass_from_k
from scripts.compaction import compute_C_profile, compute_zeta_r_profile, C_critical
from scripts.constraint_mapping import (
    BOUND_REGISTRY, load_bound_dataset, map_constraint_to_present,
)

# ── Accretion models ───────────────────────────────────────────────────
MODELS = [("Chisholm",    ChisholmAccretion()),
          ("BHL",         PBHAccretion(model="BHL", lambda_acc=0.1)),
          ("Eddington",   EddingtonAccretion(f_boost=1e4, lambda_acc=0.1)),
          ("PR",          PR_Accretion(lambda_acc=0.1))]

# ── Helpers ────────────────────────────────────────────────────────────
def f_pbh_from_beta(bf, k): return np.asarray(bf) * np.asarray(k) / K_EQ
def f_total(fp, M):
    if len(fp) < 2: return 0.0
    s = np.argsort(M)
    return float(np.trapezoid(fp[s], np.log(np.maximum(M[s], 1e-300))))

# ── Pre-compute PS formation ──────────────────────────────────────────
Mf_ps = mass_from_k(k_phys, GAMMA, K_EQ, M_EQ)
bf_ps = beta_f_press_schechter(P_S, zeta_c=ZETA_C)
fp_ps = f_pbh_from_beta(bf_ps, k_phys)
ok_ps = np.isfinite(Mf_ps) & (Mf_ps > 0) & np.isfinite(fp_ps) & (fp_ps > 0)
Mf_ps, fp_ps, k_ps = Mf_ps[ok_ps], fp_ps[ok_ps], k_phys[ok_ps]

logM = np.log(Mf_ps)
median_logM = np.median(logM)
t_k = np.abs(logM - median_logM) < 4.0 * np.std(logM)
Mf_ps, fp_ps, k_ps = Mf_ps[t_k], fp_ps[t_k], k_ps[t_k]
print(f"PS: {len(Mf_ps)} modes, M ∈ [{Mf_ps.min():.2e}, {Mf_ps.max():.2e}]")

# ── Pre-compute Compaction formation ──────────────────────────────────
print("Compaction metrics ...", flush=True)
t0 = time.time()
_K, _GC, r_pf = 4.0, 0.36, np.logspace(-3, 1.5, 200)
Mf_cp, fp_cp = np.zeros(len(k_phys)), np.zeros(len(k_phys))
for i in range(0, len(k_phys), 5):
    ki = k_phys[i]
    _, zr = compute_zeta_r_profile(k_phys, P_S, r_pf, R_smooth=1.0/ki)
    cr = compute_C_profile(r_pf, zr)
    if cr["C_max"] > C_critical(cr["alpha"]):
        MH = GAMMA * M_EQ * (K_EQ / ki) ** 2
        Mf_cp[i] = _K * MH * (cr["C_max"] - C_critical(cr["alpha"])) ** _GC
    fp_cp[i] = beta_f_press_schechter(np.array([P_S[i]]), zeta_c=ZETA_C)[0]
ok_cp = (Mf_cp > 0) & (fp_cp > 0) & (fp_cp <= 1)
Mf_cp, fp_cp, k_cp = Mf_cp[ok_cp], fp_cp[ok_cp], k_phys[ok_cp]
print(f"Compaction: {len(Mf_cp)} modes, M ∈ [{Mf_cp.min():.2e}, {Mf_cp.max():.2e}] "
      f"({time.time()-t0:.1f}s)")

# ── Plot ──────────────────────────────────────────────────────────────
from scripts.plotting import TOL, PAPER_RCPARAMS, get_path, save_fig
from matplotlib.patches import Patch

KEY_CONSTRAINTS = ['cmb_agius2024', 'microlensing_hsc', 'egrb', 'lvk_subsolar', 'eridanus_ii']
CONSTRAINT_COLORS = {
    'cmb_agius2024':  TOL["blue"],
    'microlensing_hsc': TOL["teal"],
    'egrb':           TOL["yellow"],
    'lvk_subsolar':   TOL["red"],
    'eridanus_ii':    TOL["green"],
}
CONSTRAINT_LABELS = {
    'cmb_agius2024':  r'CMB (Agius+2024)',
    'microlensing_hsc': r'HSC microlensing',
    'egrb':           r'EGRB (Fermi)',
    'lvk_subsolar':   r'LIGO subsolar',
    'eridanus_ii':    r'Eri II',
}
FORMATION_NAMES = [r'Press--Schechter', 'Compaction']


def _plot_accretion_comparison(Mf_ps, fp_ps, Mf_cp, fp_cp, MODELS):
    """Publication-quality 2x4 PBH abundance comparison grid.

    Rows: formation models (Press-Schechter, Compaction).
    Columns: accretion models (Chisholm, BHL, Eddington, PR).

    Uses PAPER_RCPARAMS, TOL colors, shared constraint legend below grid.
    """
    with plt.rc_context(PAPER_RCPARAMS):
        fig, axes = plt.subplots(
            2, 4, figsize=(7, 5.2),
            sharex='col',
            gridspec_kw={'wspace': 0.04, 'hspace': 0.25},
            constrained_layout=False,
        )

        constraint_handles = []
        for cname in KEY_CONSTRAINTS:
            color = CONSTRAINT_COLORS[cname]
            constraint_handles.append(
                Patch(facecolor=color, alpha=0.25, edgecolor=color,
                      linewidth=0.5, label=CONSTRAINT_LABELS[cname])
            )

        for col, (acc_name, acc) in enumerate(MODELS):
            for row in range(2):
                ax = axes[row, col]

                if row == 0:
                    M_f, fp = Mf_ps, fp_ps
                else:
                    if len(Mf_cp) == 0:
                        ax.text(0.5, 0.5, 'No collapse', transform=ax.transAxes,
                                ha='center', va='center', fontsize=7,
                                color=TOL["grey"], style='italic')
                        continue
                    M_f, fp = Mf_cp, fp_cp

                M_p = np.array([
                    acc.M_of_redshift(float(m), 0.0) if m > 1e-300 else 0.0
                    for m in M_f
                ])
                ok = np.isfinite(M_p) & (M_p > 0) & np.isfinite(fp) & (fp > 0)
                M_p, fp = M_p[ok], fp[ok]
                if len(M_p) < 2:
                    ax.text(0.5, 0.5, 'No collapse', transform=ax.transAxes,
                            ha='center', va='center', fontsize=7,
                            color=TOL["grey"], style='italic')
                    continue

                for cname in KEY_CONSTRAINTS:
                    try:
                        M_b, f_b, meta = load_bound_dataset(cname)
                        z_probe = meta['z_probe']
                        if z_probe > 0:
                            M_b, f_b = map_constraint_to_present(M_b, f_b, acc, z_probe)
                        ok_b = (M_b > 0) & (f_b > 0)
                        M_b, f_b = M_b[ok_b], f_b[ok_b]
                        color = CONSTRAINT_COLORS[cname]
                        ax.fill_between(M_b, f_b, 10.0, color=color, alpha=0.12, lw=0)
                    except Exception:
                        pass

                ft = f_total(fp, M_p)
                ax.loglog(M_p, fp, '-', color=TOL["red"], lw=1.0, zorder=10)

                ft_ypos = 0.04 if row == 1 else 0.95
                ft_va = 'bottom' if row == 1 else 'top'
                ax.text(0.96, ft_ypos, f'$f_{{\\mathrm{{tot}}}}$ = {ft:.1e}',
                        transform=ax.transAxes, ha='right', va=ft_va,
                        fontsize=5.5, color=TOL["dark"], alpha=0.65)

                panel_idx = col + row * 4
                label = chr(ord('a') + panel_idx)
                ax.text(0.03, 0.96, f'({label})', transform=ax.transAxes,
                        fontsize=7, fontweight='bold', va='top', ha='left',
                        color=TOL["dark"])

                ax.set_xscale('log')
                ax.set_yscale('log')
                ax.set_xlim(1e-12, 1e14)
                y_lo = 1e-12 if row == 1 else 1e-6
                y_hi = 1e-4 if row == 1 else 1
                ax.set_ylim(y_lo, y_hi)
                ax.tick_params(labelsize=6.5, which='both', direction='in',
                               top=True, right=True, length=2.5, width=0.4)
                ax.tick_params(which='minor', length=1.5, width=0.3)
                ax.grid(True, alpha=0.12, which='major', ls='-', color=TOL["grey"])

        for row, name in enumerate(FORMATION_NAMES):
            axes[row, 0].set_ylabel(
                f'{name}\n' + r'$f_\mathrm{PBH}$',
                fontsize=7.5, labelpad=4,
            )

        for col, (acc_name, _) in enumerate(MODELS):
            axes[0, col].set_title(acc_name, fontsize=7.5, fontweight='bold',
                                   pad=3, color=TOL["dark"])

        for col in range(4):
            axes[1, col].set_xlabel(r'$M\ [M_\odot]$', fontsize=7.5)

        fig.legend(
            handles=constraint_handles, loc='lower center',
            ncol=5, fontsize=5.5, frameon=False,
            bbox_to_anchor=(0.52, -0.01),
            handletextpad=0.4, columnspacing=1.2, labelspacing=0.3,
        )

        fig.text(
            0.5, 1.0,
            r'Ezquiaga CHI ($\beta=10^{-5}$, $\chi_0=8.0$, $N_*=65$)',
            ha='center', va='bottom', fontsize=7.5, color=TOL["dark"],
        )

        fig.subplots_adjust(left=0.10, right=0.98, top=0.93, bottom=0.13,
                            wspace=0.04, hspace=0.30)

        _OUT_NAME = "comparison_accretion_models.png"
        out_path = get_path("pbh", _OUT_NAME)
        fig.savefig(out_path, dpi=300, bbox_inches='tight')
        print(f"\nSaved: {out_path}")
        plt.close(fig)


_plot_accretion_comparison(Mf_ps, fp_ps, Mf_cp, fp_cp, MODELS)
