"""
Compare HiggsModel vs FullHiggsModel for the paper config.
phi0=5.98, y0=-0.300, N_star=55.0, xi=15000, lam=0.13

Checks if the approximate plateau potential gives different
P_S(k) and D_ell than the exact conformal inversion.
"""
import sys
import json
import time
import numpy as np
import warnings

sys.path.insert(0, ".")

from models import HiggsModel, FullHiggsModel
import inf_dyn_background as bg_solver
from pspectrum_pipeline import (
    run_pspectrum_pipeline, find_end_of_inflation,
    build_weighted_kgrid, get_k_pivot_code,
)
from scripts.camb_wrapper import compute_cl_full_camb, compute_cl_camb_powerlaw
from scripts.planck_data import get_planck_data_asymmetric, C_ell_to_d_ell
from scripts.constants import As, k_pivot_phys
from scripts.plotting import get_path, make_filename

PHI0 = 5.98
Y0 = -0.300
N_STAR = 55.0
XI = 15000.0
LAM = 0.13
T_MAX = 2000.0
BG_STEPS = 10000
MS_STEPS = 5000


def run_background(ModelClass, label):
    """Run background integration, return (bg_sol, derived, end_idx, model)."""
    print(f"\n  --- Background: {label} ---")
    model = ModelClass(lam=LAM, xi=XI)
    model.x0 = PHI0
    model.y0 = Y0
    model.T_max = T_MAX
    model.bg_steps = BG_STEPS
    T_span = np.linspace(0, T_MAX, BG_STEPS)

    bg = bg_solver.run_background_simulation(model, T_span)
    derived = bg_solver.get_derived_quantities(bg, model)
    end_idx = find_end_of_inflation(derived["epsH"])

    if end_idx < 0:
        print(f"  WARNING: No end of inflation found for {label}")
        return bg, derived, None, model

    k_pc, pivot_idx, N_total = get_k_pivot_code(bg, derived, end_idx, N_STAR)
    x_cmb = bg[0][pivot_idx] if pivot_idx is not None else None

    print(f"    N_total = {derived['N'][end_idx]:.2f}")
    print(f"    x_cmb   = {x_cmb:.5f}" if x_cmb else "    x_cmb   = N/A")
    print(f"    end_idx = {end_idx} / {len(derived['N'])}")

    return bg, derived, end_idx, model


def run_ps_pipeline(model, label, T_span_bg):
    """Run full P_S(k) pipeline."""
    print(f"\n  --- P_S(k): {label} ---")
    k_grid = build_weighted_kgrid(
        1e-5, 1.0, k_pivot_phys,
        dense_zone=(1e-4, 1e-2), n_dense=200, n_outer=100,
    )
    print(f"    k-modes: {len(k_grid)} (weighted grid)")

    result = run_pspectrum_pipeline(
        model=model, phi0=PHI0, y0=Y0, N_star=N_STAR,
        k_pivot_phys=k_pivot_phys,
        k_phys_grid=k_grid,
        T_span_bg=T_span_bg,
        ms_steps=MS_STEPS,
        normalize_to_As=True, As=As,
        n_workers=1,
        save_outputs=True,
    )

    if result["status"] != "success":
        print(f"    FAILED: {result.get('message', 'unknown')}")
        return None

    k_phys = result["k_phys"]
    P_S = result["P_S"]
    meta = result["metadata"]

    k_dip = k_phys[np.nanargmin(P_S)]
    p_dip = np.nanmin(P_S)
    p_pivot = float(np.interp(k_pivot_phys, k_phys, P_S))
    supp = (1 - p_dip / p_pivot) * 100

    print(f"    k_dip       = {k_dip:.4e} Mpc^-1")
    print(f"    P_S(k_dip)  = {p_dip:.4e}")
    print(f"    Suppression = {supp:.1f}%")
    print(f"    N_total     = {meta.get('N_total', '?'):.2f}")
    print(f"    norm_scale  = {meta.get('scale_factor', '?'):.4e}")

    return {"k_phys": k_phys, "P_S": P_S, "meta": meta, "k_dip": k_dip, "supp": supp}


def run_camb(ps_data, label):
    """Run CAMB and compute D_ell + chi2."""
    print(f"\n  --- CAMB D_ell: {label} ---")
    ells, C_TT, C_TE, C_EE = compute_cl_full_camb(ps_data, ell_max=2500)
    D = C_ell_to_d_ell(ells, C_TT)

    ells_l, C_l, _, _ = compute_cl_camb_powerlaw(ell_max=2500)
    D_l = C_ell_to_d_ell(ells_l, C_l)

    pl_ells, pl_D, pl_err_l, pl_err_u = get_planck_data_asymmetric()

    chi2_m = 0.0
    chi2_l = 0.0
    for i, el in enumerate(pl_ells):
        if el > 29:
            continue
        idx = int(np.argmin(np.abs(ells - el)))
        r_m = D[idx] - pl_D[i]
        r_l = D_l[idx] - pl_D[i]
        s_m = pl_err_u[i] if r_m > 0 else pl_err_l[i]
        s_l = pl_err_u[i] if r_l > 0 else pl_err_l[i]
        chi2_m += (r_m / s_m) ** 2
        chi2_l += (r_l / s_l) ** 2

    dchi2 = chi2_m - chi2_l

    d2 = D[2]
    d220 = D[np.argmin(np.abs(ells - 220))] if 220 <= ells[-1] else None
    d540 = D[np.argmin(np.abs(ells - 540))] if 540 <= ells[-1] else None
    d850 = D[np.argmin(np.abs(ells - 850))] if 850 <= ells[-1] else None

    # high-ell ratio (ell > 2000)
    high_mask = ells > 2000
    d_l_high = np.interp(ells[high_mask], ells_l, D_l)
    high_ratio = np.mean(D[high_mask] / d_l_high) if np.any(high_mask) else None

    print(f"    D_2        = {d2:.0f} uK^2")
    print(f"    D_220      = {d220:.0f} uK^2" if d220 else "    D_220 = N/A")
    print(f"    chi2_model = {chi2_m:.2f}")
    print(f"    chi2_lcdm  = {chi2_l:.2f}")
    print(f"    Delta_chi2 = {dchi2:+.2f}")
    print(f"    high-ratio = {high_ratio:.4f}" if high_ratio else "    high-ratio = N/A")

    return {
        "ells": ells, "D": D, "D_lcdm": D_l, "ells_lcdm": ells_l,
        "d2": d2, "d220": d220, "d540": d540, "d850": d850,
        "chi2_model": chi2_m, "chi2_lcdm": chi2_l, "dchi2": dchi2,
        "high_ratio": high_ratio,
        "planck": {"ells": pl_ells, "D": pl_D, "err_l": pl_err_l, "err_u": pl_err_u},
    }


# ── Main ─────────────────────────────────────────────────────────────
print("=" * 72)
print("  HiggsModel vs FullHiggsModel Comparison")
print(f"  Config: phi0={PHI0}, y0={Y0}, N*={N_STAR}, xi={XI}, lam={LAM}")
print("=" * 72)

results = {}

for ModelClass, label in [(HiggsModel, "Approx (HiggsModel)"), (FullHiggsModel, "Full (FullHiggsModel)")]:
    t0 = time.time()

    # Step 1: Background
    bg, derived, end_idx, model = run_background(ModelClass, label)
    if end_idx is None:
        print(f"  SKIPPING {label} (no end of inflation)")
        continue
    T_span = np.linspace(0, model.T_max, model.bg_steps)

    # Step 2: P_S(k) pipeline
    ps_result = run_ps_pipeline(model, label, T_span)
    if ps_result is None:
        print(f"  SKIPPING CAMB for {label}")
        continue

    # Step 3: CAMB
    ps_data = {"k_phys": ps_result["k_phys"], "P_S": ps_result["P_S"]}
    camb_result = run_camb(ps_data, label)

    results[label] = {
        "bg": {"N_total": float(derived["N"][end_idx]), "end_idx": int(end_idx)},
        "ps": ps_result,
        "camb": camb_result,
    }

    dt = time.time() - t0
    print(f"\n  [{label}] Total: {dt:.0f}s")

# ── Side-by-side comparison ──────────────────────────────────────────
print("\n" + "=" * 72)
print("  COMPARISON TABLE")
print("=" * 72)

if not results:
    print("  No results to compare.")
    sys.exit(1)

keys = list(results.keys())
ref_key = keys[0]
cmp_key = keys[1] if len(keys) > 1 else None

print(f"\n  {'Quantity':<25} {keys[0]:>15}", end="")
if cmp_key:
    print(f" {cmp_key:>15} {'Ratio':>10}")
else:
    print()
print("  " + "-" * (65 if cmp_key else 25))

r = results[ref_key]
print(f"  {'N_total':<25} {r['bg']['N_total']:>15.2f}", end="")
if cmp_key:
    print(f" {results[cmp_key]['bg']['N_total']:>15.2f} {'—':>10}")
else:
    print()

r_ps = r["ps"]
print(f"  {'k_dip [Mpc^-1]':<25} {r_ps['k_dip']:>15.4e}", end="")
if cmp_key:
    c_ps = results[cmp_key]["ps"]
    print(f" {c_ps['k_dip']:>15.4e} {r_ps['k_dip']/c_ps['k_dip']:>10.6f}")
else:
    print()

print(f"  {'Suppression [%]':<25} {r_ps['supp']:>15.2f}", end="")
if cmp_key:
    print(f" {results[cmp_key]['ps']['supp']:>15.2f} {'—':>10}")
else:
    print()

if r.get("camb"):
    rc = r["camb"]
    print(f"  {'D_2 [uK^2]':<25} {rc['d2']:>15.0f}", end="")
    if cmp_key and results[cmp_key].get("camb"):
        cc = results[cmp_key]["camb"]
        print(f" {cc['d2']:>15.0f} {rc['d2']/cc['d2']:>10.4f}")
    else:
        print()

    print(f"  {'D_220 [uK^2]':<25} {rc['d220']:>15.0f}", end="")
    if cmp_key and results[cmp_key].get("camb"):
        cc = results[cmp_key]["camb"]
        print(f" {cc['d220']:>15.0f} {rc['d220']/cc['d220']:>10.4f}")
    else:
        print()

    print(f"  {'chi2_model (ell<30)':<25} {rc['chi2_model']:>15.2f}", end="")
    if cmp_key and results[cmp_key].get("camb"):
        cc = results[cmp_key]["camb"]
        print(f" {cc['chi2_model']:>15.2f} {rc['chi2_model']/cc['chi2_model']:>10.4f}")
    else:
        print()

    print(f"  {'chi2_lcdm':<25} {rc['chi2_lcdm']:>15.2f}", end="")
    if cmp_key and results[cmp_key].get("camb"):
        cc = results[cmp_key]["camb"]
        print(f" {cc['chi2_lcdm']:>15.2f} {rc['chi2_lcdm']/cc['chi2_lcdm']:>10.4f}")
    else:
        print()

    print(f"  {'Delta chi2':<25} {rc['dchi2']:>+15.2f}", end="")
    if cmp_key and results[cmp_key].get("camb"):
        print(f" {results[cmp_key]['camb']['dchi2']:>+15.2f} {'—':>10}")
    else:
        print()

    hr = rc.get("high_ratio")
    if hr:
        print(f"  {'High-ell ratio (>2000)':<25} {hr:>15.4f}", end="")
        if cmp_key and results[cmp_key].get("camb"):
            chr_ = results[cmp_key]["camb"].get("high_ratio")
            print(f" {chr_:>15.4f}" if chr_ else f" {'N/A':>15} {'—':>10}")
        else:
            print()

# P_S element-wise comparison
if cmp_key:
    print("\n" + "-" * 72)
    print("  P_S(k) element-wise comparison:")
    r_ps = results[ref_key]["ps"]
    c_ps = results[cmp_key]["ps"]
    k_interp = np.geomspace(
        max(r_ps["k_phys"].min(), c_ps["k_phys"].min()),
        min(r_ps["k_phys"].max(), c_ps["k_phys"].max()),
        200,
    )
    ps_r_interp = np.interp(np.log(k_interp), np.log(r_ps["k_phys"]), r_ps["P_S"])
    ps_c_interp = np.interp(np.log(k_interp), np.log(c_ps["k_phys"]), c_ps["P_S"])
    ratio = ps_r_interp / ps_c_interp
    print(f"    Max |ratio - 1| = {np.max(np.abs(ratio - 1)):.6e}")
    print(f"    Mean |ratio - 1| = {np.mean(np.abs(ratio - 1)):.6e}")

    # P_S near pivot
    pv_idx_r = np.argmin(np.abs(r_ps["k_phys"] - k_pivot_phys))
    pv_idx_c = np.argmin(np.abs(c_ps["k_phys"] - k_pivot_phys))
    ps_p_r = r_ps["P_S"][pv_idx_r]
    ps_p_c = c_ps["P_S"][pv_idx_c]
    print(f"    P_S(k_pivot) ratio = {ps_p_r/ps_p_c:.8f}")
    print(f"    (approx={ps_p_r:.6e}, full={ps_p_c:.6e})")

    # D_ell element-wise comparison
    if results[ref_key].get("camb") and results[cmp_key].get("camb"):
        r_c = results[ref_key]["camb"]
        c_c = results[cmp_key]["camb"]
        ell_interp = np.arange(
            max(r_c["ells"].min(), c_c["ells"].min()),
            min(r_c["ells"].max(), c_c["ells"].max()) + 1,
        )
        d_r = np.interp(ell_interp, r_c["ells"], r_c["D"])
        d_c = np.interp(ell_interp, c_c["ells"], c_c["D"])
        d_ratio = d_r / d_c
        print(f"    Max |D_ratio - 1| = {np.max(np.abs(d_ratio - 1)):.6e}")
        print(f"    Mean |D_ratio - 1| = {np.mean(np.abs(d_ratio - 1)):.6e}")

print("=" * 72)
print("  Done.")
