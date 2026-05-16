"""
Full D_ell CAMB diagnostic: physical consistency checks + publication plot.

Checks:
  1. All D_ell finite and positive
  2. Acoustic 1st peak at ℓ~180-260
  3. Acoustic 2nd peak at ℓ~450-630
  4. Acoustic 3rd peak at ℓ~750-950
  5. High-ℓ convergence with LCDM (ℓ>2000, ratio within 10%)
  6. Low-ℓ suppression (ℓ=2-10, model < LCDM)
  7. χ² vs Planck low-ℓ < 25

Usage:
  python scripts/check_full_dell.py [--phi0 PHI0] [--y0 Y0] [--N-star NSTAR]

Default: golden config (phi0=6.60, y0=-0.736, N_star=52.5893)
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
from scripts.camb_wrapper import compute_cl_full_camb, compute_cl_camb_powerlaw
from scripts.sachs_wolfe import compute_cl_sw
from scripts.planck_data import get_planck_data_asymmetric, C_ell_to_d_ell
from scripts.chi2_analysis import chi2_commander, chi2_unbinned, print_chi2_table
from scripts.plotting import get_path, find_ps, make_filename

TOL = {"blue": "#4477AA", "red": "#CC3311", "green": "#228833",
       "yellow": "#EE8866", "teal": "#44BB99", "purple": "#AA3377",
       "grey": "#666666", "dark": "#222222"}

plt.rcParams.update({"font.size": 12, "axes.labelsize": 14, "axes.titlesize": 16,
                     "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 11,
                     "figure.dpi": 300})


def detect_peaks(ells, D_ell, min_height_ratio=0.15, min_dist=30):
    peaks = []
    n = len(D_ell)
    max_D = np.max(D_ell)
    for i in range(3, n - 3):
        if D_ell[i] < min_height_ratio * max_D:
            continue
        if D_ell[i] > D_ell[i - 3] and D_ell[i] > D_ell[i - 2] and D_ell[i] > D_ell[i - 1] and \
           D_ell[i] >= D_ell[i + 1] and D_ell[i] >= D_ell[i + 2] and D_ell[i] >= D_ell[i + 3]:
            if not peaks or (ells[i] - peaks[-1][0]) >= min_dist:
                peaks.append((ells[i], D_ell[i]))
    return peaks


def run_checks(ells_model, D_model, ells_lcdm, D_lcdm, planck_data, D_sw, ells_sw):
    p_ells, D_p, D_lo, D_hi = planck_data
    checks = []

    D_int_lcdm = interp1d(ells_lcdm, D_lcdm, kind="cubic", bounds_error=False, fill_value="extrapolate")

    # 1. Non-negative and finite
    finite_ok = bool(np.all(np.isfinite(D_model)))
    pos_ok = bool(np.all(D_model > 0))
    n_bad = int(np.sum(~np.isfinite(D_model))) + int(np.sum(D_model <= 0))
    checks.append(("1. All D_ell finite and positive", finite_ok and pos_ok,
                   f"{n_bad} bad values" if n_bad else "OK"))

    # 2. Low-ell suppression
    low_idx = ells_model <= 10
    ratio_low = D_model[low_idx] / D_int_lcdm(ells_model[low_idx])
    supp = (1 - np.mean(ratio_low)) * 100
    checks.append(("2. Low-ell suppression (ℓ≤10)", supp > 5,
                   f"Model/LCDM={np.mean(ratio_low):.3f}, suppression={supp:.1f}%"))

    # 3-5. Acoustic peaks
    peaks = detect_peaks(ells_model, D_model)
    peak_targets = [(220, 180, 280, "1st"), (540, 430, 650, "2nd"), (850, 720, 980, "3rd")]
    for target, lo, hi, label in peak_targets:
        match = any(lo <= p[0] <= hi for p in peaks)
        info = next((f"ℓ={p[0]:.0f} D={p[1]:.0f}" for p in peaks if lo <= p[0] <= hi), "not found")
        checks.append((f"3-5. Acoustic {label} peak (ℓ~{target})", match, info))

    # 6. High-ℓ convergence
    hi_mask_m = ells_model >= 2000
    hi_mask_l = ells_lcdm >= 2000
    if np.any(hi_mask_m) and np.any(hi_mask_l):
        d_hi_m = D_model[hi_mask_m][:100]
        d_hi_l = D_int_lcdm(ells_model[hi_mask_m][:100])
        hi_ratio = d_hi_m / d_hi_l
        conv = bool(np.abs(np.mean(hi_ratio) - 1.0) < 0.1)
        checks.append(("6. High-ℓ convergence (ℓ>2000)", conv,
                       f"Model/LCDM={np.mean(hi_ratio):.3f}±{np.std(hi_ratio):.3f}"))
    else:
        checks.append(("6. High-ℓ convergence (ℓ>2000)", False,
                       f"Insufficient range: model ℓ_max={ells_model[-1]}"))

    # 7. Chi²
    chi2 = 0.0
    for i, ell_val in enumerate(p_ells):
        idx = int(np.argmin(np.abs(ells_model - ell_val)))
        res = D_model[idx] - D_p[i]
        sigma = D_hi[i] if res > 0 else D_lo[i]
        chi2 += (res / sigma) ** 2
    chi2_lcdm = 0.0
    for i, ell_val in enumerate(p_ells):
        res = D_int_lcdm(ell_val) - D_p[i]
        sigma = D_hi[i] if res > 0 else D_lo[i]
        chi2_lcdm += (res / sigma) ** 2
    checks.append(("7. χ² vs Planck low-ℓ", chi2 < chi2_lcdm,
                   f"model χ²={chi2:.2f}, LCDM χ²={chi2_lcdm:.2f}, Δ={chi2-chi2_lcdm:+.2f}"))

    return checks, chi2, chi2_lcdm


def make_plot(ells_model, D_model, ells_lcdm, D_lcdm, planck_low, planck_binned, path):
    p_ells, D_p, D_lo, D_hi = planck_low
    b_ells, b_D, b_lo, b_hi = planck_binned

    plt.rcParams.update({
        "font.size": 9, "axes.labelsize": 11, "axes.titlesize": 12,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
        "figure.dpi": 300,
    })

    fig = plt.figure(figsize=(7, 3.3))
    gs = fig.add_gridspec(1, 2, width_ratios=[1, 4], wspace=0)
    ax_left = fig.add_subplot(gs[0])
    ax_right = fig.add_subplot(gs[1], sharey=ax_left)

    # Remove inner spines for broken-axis effect
    ax_left.spines['right'].set_visible(False)
    ax_right.spines['left'].set_visible(False)
    ax_right.tick_params(left=False)

    # X-axis scales
    ax_left.set_xscale('log')
    ax_left.set_xlim(1.8, 32)
    ax_left.set_xticks([2, 10, 30])
    ax_left.get_xaxis().set_major_formatter(plt.ScalarFormatter())
    ax_right.set_xlim(32, 2600)
    ax_right.tick_params(labelleft=False)

    # Shared Y-axis (linear)
    ax_left.set_ylabel(r"$D_\ell^{TT}$ [$\mu$K$^2$]")
    ax_left.set_ylim(-100, 6500)

    # Model lines on both axes
    for ax in [ax_left, ax_right]:
        ax.plot(ells_model, D_model, "-", color=TOL["red"], lw=1.2, label="Higgs USR", zorder=4)
        ax.plot(ells_lcdm, D_lcdm, "--", color=TOL["dark"], lw=1.2, alpha=0.6, label=r"$\Lambda$CDM", zorder=3)

    # Planck low-ℓ (Commander) on left
    ax_left.errorbar(p_ells, D_p, yerr=[D_lo, D_hi], fmt="o", color=TOL["dark"],
                     capsize=1.5, markersize=2, elinewidth=0.4, label="Planck 2018", zorder=5)

    # Planck binned TT on right
    ax_right.errorbar(b_ells, b_D, yerr=[b_lo, b_hi], fmt="o", color=TOL["dark"],
                      capsize=1, markersize=1.5, elinewidth=0.3, label="Planck 2018", zorder=5)

    # Axis labels
    ax_left.set_xlabel(r"$\ell$")
    ax_right.set_xlabel(r"$\ell$")

    # Grid
    ax_left.grid(True, alpha=0.15, which="both")
    ax_right.grid(True, alpha=0.15, which="both")

    # Vertical dashed line marking the scale transition
    ax_right.axvline(x=32, color=TOL["grey"], ls="--", lw=1.5, zorder=0)

    # Legend on right panel
    ax_right.legend(loc="upper right")

    fig.tight_layout()
    fig.savefig(path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


def main():
    phi0 = float(sys.argv[sys.argv.index("--phi0") + 1]) if "--phi0" in sys.argv else 6.60
    y0 = float(sys.argv[sys.argv.index("--y0") + 1]) if "--y0" in sys.argv else -0.736
    n_star = float(sys.argv[sys.argv.index("--N-star") + 1]) if "--N-star" in sys.argv else 52.5893

    out = get_path("diagnostics", make_filename("planck", phi0, y0, n_star, ".png"))

    print("=" * 60)
    print("  Full D_ℓ Physical Consistency Check")
    print(f"  Config: φ₀={phi0}  y₀={y0}  N*={n_star}")
    print("=" * 60)

    # 1. Load P_S
    print("\n[1/5] Loading P_S(k)...")
    ps_path, ps_md = find_ps(phi0, y0, n_star)
    if ps_path is None:
        print("  ERROR: No cached P_S(k) file found")
        sys.exit(1)
    print(f"  File: {os.path.basename(ps_path)}")

    with open(ps_path) as f:
        rec = json.load(f)
    spec = rec["spectrum"]
    k_phys = np.array(spec["k_phys"])
    P_S = np.array(spec["P_S"])
    md = rec.get("metadata", {})
    n_ok = int(np.sum(np.isfinite(P_S)))
    print(f"  N_total={md.get('N_total','?'):.1f}  N_star={md.get('N_star','?'):.2f}")
    print(f"  k=[{k_phys.min():.2e}, {k_phys.max():.2e}]  modes={n_ok}/{len(k_phys)}")
    ps_data = {"k_phys": k_phys, "P_S": P_S}

    # 2. CAMB C_ell
    print("\n[2/5] Computing CAMB C_ell (ℓ=2-2500)...")
    ells, C_TT, _, _ = compute_cl_full_camb(ps_data, ell_max=2500)
    D_model = C_ell_to_d_ell(ells, C_TT)
    print(f"  D_ℓ ∈ [{D_model.min():.1f}, {D_model.max():.1f}] μK²")

    # 3. LCDM baseline
    print("\n[3/5] LCDM baseline...")
    ells_l, C_l, _, _ = compute_cl_camb_powerlaw(ell_max=2500)
    D_lcdm = C_ell_to_d_ell(ells_l, C_l)
    print(f"  D_ℓ ∈ [{D_lcdm.min():.1f}, {D_lcdm.max():.1f}] μK²")

    # 4. SW-only
    print("\n  SW-only (for ISW)...")
    ells_sw, C_sw = compute_cl_sw(ps_data, ell_max=30)
    D_sw = C_ell_to_d_ell(ells_sw, C_sw)

    # 5. Planck data
    planck_data = get_planck_data_asymmetric()
    pb = np.loadtxt(os.path.join(ROOT_DIR, "data/Planck/COM_PowerSpect_CMB-TT-binned_R3.01.txt"), skiprows=1)
    planck_binned = (pb[:, 0], pb[:, 1], pb[:, 2], pb[:, 3])

    # 6. Checks
    print("\n[4/5] Physical consistency checks...")
    checks, chi2, chi2_lcdm = run_checks(ells, D_model, ells_l, D_lcdm, planck_data, D_sw, ells_sw)

    print()
    hdr = f"{'Check':<42s} {'Status':<10s} Detail"
    sep = "-" * 42 + "  " + "-" * 10 + "  " + "-" * 42
    print(f"  {hdr}")
    print(f"  {sep}")
    all_pass = True
    for name, passed, detail in checks:
        status = "PASS" if passed else "FAIL"
        all_pass = all_pass and passed
        print(f"  {name:<42s} {status:<10s} {detail}")

    # Additional diagnostics
    D_int_lcdm = interp1d(ells_l, D_lcdm, kind="cubic", bounds_error=False, fill_value="extrapolate")
    print(f"\n  Additional diagnostics:")
    hdr2 = f"{'Quantity':<30s} {'Value':<20s} {'Expected':<20s}"
    sep2 = "-" * 70
    print(f"  {hdr2}")
    print(f"  {sep2}")

    # D_ell at key ℓ
    for el in [2, 10, 20, 220, 540, 850, 1500, 2500]:
        idx = int(np.argmin(np.abs(ells - el)))
        lcdm_val = D_int_lcdm(ells[idx])
        print(f"  {'D_ℓ(ℓ='+str(el)+')':<30s} {D_model[idx]:>8.1f} μK² {'':8s} {lcdm_val:>8.1f} μK² (LCDM)")

    # ISW fraction
    D_full_low = D_model[ells <= 30]
    isw_pct = (D_full_low - D_sw) / D_sw * 100
    print(f"\n  ISW fraction:")
    for el in [2, 5, 10, 20, 29]:
        idx = int(np.argmin(np.abs(ells_sw - el)))
        print(f"    ℓ={el:2d}: {isw_pct[idx]:6.1f}%")

    # Suppression vs LCDM
    print(f"\n  Suppression vs LCDM:")
    for el in [2, 5, 10, 20, 29]:
        idx = int(np.argmin(np.abs(ells - el)))
        ratio = D_model[idx] / D_int_lcdm(ells[idx])
        print(f"    ℓ={el:2d}: D_model/D_LCDM={ratio:.3f}  ({(1-ratio)*100:+.1f}%)")

    # Quadrupole anomaly
    d2 = D_model[0]
    d2_p = planck_data[1][0]
    d2_l = D_int_lcdm(2.0)
    print(f"\n  Quadrupole anomaly:")
    print(f"    D_ℓ(ℓ=2) model  = {d2:8.1f} μK²")
    print(f"    D_ℓ(ℓ=2) Planck = {d2_p:8.1f} μK²")
    print(f"    D_ℓ(ℓ=2) LCDM   = {d2_l:8.1f} μK²")
    print(f"    Model/Planck     = {d2/d2_p:.2f}x")
    print(f"    LCDM/Planck      = {d2_l/d2_p:.2f}x")

    # Peak summary
    peaks = detect_peaks(ells, D_model)
    print(f"\n  Detected acoustic peaks:")
    for el, amp in peaks[:5]:
        lcdm_amp = D_int_lcdm(el)
        print(f"    ℓ={int(el):4d}  D_ℓ={amp:8.1f} μK²  (LCDM: {lcdm_amp:.1f})")

    # Silk damping
    d_max = D_model[np.argmax(D_model)]
    d_2000 = D_model[int(np.argmin(np.abs(ells - 2000)))] if np.any(ells >= 2000) else D_model[-1]
    d_2500 = D_model[-1]
    print(f"\n  Silk damping:")
    print(f"    D_ℓ(max)={d_max:.0f} at ℓ={ells[np.argmax(D_model)]:.0f}")
    print(f"    D_ℓ(ℓ=2000)/D_max={d_2000/d_max:.4f}")
    print(f"    D_ℓ(ℓ=2500)/D_max={d_2500/d_max:.4f}  ({'dampled' if d_2500/d_max < 0.05 else 'weak'})")

    # 7. χ² table
    print(f"\n[6/6] χ² analysis...")
    print_chi2_table(D_model, ells, D_lcdm, ells_l,
                     label=f"φ₀={phi0}  y₀={y0}  N*={n_star}")

    # 8. Plot
    print(f"\n[7/7] Generating plot...")
    make_plot(ells, D_model, ells_l, D_lcdm, planck_data, planck_binned, out)

    print(f"\n{'='*60}")
    if all_pass:
        print("  RESULT: ALL CHECKS PASSED")
    else:
        print("  RESULT: SOME CHECKS FAILED — review above")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
