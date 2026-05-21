"""
Modular χ² analysis vs Planck 2018 TT data.
Supports Commander low-ℓ, unbinned PLIK, and binned PLIK.
"""
import os

import numpy as np

from scripts.constants import ROOT_DIR
from scripts.planck_data import C_ell_to_d_ell


def load_planck_low():
    data = np.loadtxt(
        os.path.join(ROOT_DIR, "data/Planck/planck_2018_low_ell_tt.csv"),
        skiprows=1, delimiter=",",
    )
    return (data[:, 0].astype(int), data[:, 1], data[:, 2], data[:, 3])


def load_planck_unbinned():
    data = np.loadtxt(
        os.path.join(ROOT_DIR, "data/Planck/COM_PowerSpect_CMB-TT-full_R3.01.txt"),
        skiprows=1,
    )
    return (data[:, 0].astype(int), data[:, 1], data[:, 2], data[:, 3])


def load_planck_binned():
    data = np.loadtxt(
        os.path.join(ROOT_DIR, "data/Planck/COM_PowerSpect_CMB-TT-binned_R3.01.txt"),
        skiprows=1,
    )
    return (data[:, 0], data[:, 1], data[:, 2], data[:, 3])


def chi2_model_lcdm(D_model, ells_model, planck_ells=None, D_planck=None,
                    D_err_lower=None, D_err_upper=None,
                    D_lcdm=None, ells_lcdm=None, ell_max=29):
    """Compute χ² for model vs Planck low-ℓ, optionally also for LCDM.

    If planck_* is None, loads from planck_data.get_planck_data_asymmetric().
    Uses np.interp (smooth) for interpolation.
    Returns (chi2_model, chi2_lcdm).
    """
    if planck_ells is None:
        from scripts.planck_data import get_planck_data_asymmetric
        planck_ells, D_planck, D_err_lower, D_err_upper = get_planck_data_asymmetric()

    chi2_m = 0.0
    for i, ell_val in enumerate(planck_ells):
        if ell_val > ell_max:
            continue
        dm = np.interp(ell_val, ells_model, D_model)
        rm = dm - D_planck[i]
        sigma = D_err_upper[i] if rm > 0 else D_err_lower[i]
        chi2_m += (rm / sigma) ** 2

    chi2_l = 0.0
    if D_lcdm is not None and ells_lcdm is not None:
        for i, ell_val in enumerate(planck_ells):
            if ell_val > ell_max:
                continue
            dl = np.interp(ell_val, ells_lcdm, D_lcdm)
            rl = dl - D_planck[i]
            sigma = D_err_upper[i] if rl > 0 else D_err_lower[i]
            chi2_l += (rl / sigma) ** 2

    return chi2_m, chi2_l


def _chi2_model_lcdm(D_model, ells_model, D_lcdm, ells_lcdm,
                     planck_ells, planck_D, planck_lo, planck_hi):
    """Legacy private wrapper — used by chi2_commander/unbinned/binned."""
    chi2_m, chi2_l = chi2_model_lcdm(
        D_model, ells_model,
        planck_ells=planck_ells, D_planck=planck_D,
        D_err_lower=planck_lo, D_err_upper=planck_hi,
        D_lcdm=D_lcdm, ells_lcdm=ells_lcdm, ell_max=9999,
    )
    return chi2_m, chi2_l, len(planck_ells)


def chi2_commander(D_model, ells_model, D_lcdm, ells_lcdm):
    """Low-ℓ χ² from Commander (ℓ=2-29), asymmetric errors."""
    planck_low = load_planck_low()
    return _chi2_model_lcdm(D_model, ells_model, D_lcdm, ells_lcdm, *planck_low)


def chi2_unbinned(D_model, ells_model, D_lcdm, ells_lcdm):
    """Full-ℓ χ² from unbinned PLIK (ℓ=2-2508), asymmetric errors."""
    planck_full = load_planck_unbinned()
    return _chi2_model_lcdm(D_model, ells_model, D_lcdm, ells_lcdm, *planck_full)


def chi2_binned(D_model, ells_model, D_lcdm, ells_lcdm):
    """Full-ℓ χ² from binned PLIK (ℓ≈47-2500), symmetric errors.

    Returns (chi2_model, chi2_lcdm, n_bins).
    """
    b_ells, b_D, b_lo, b_hi = load_planck_binned()
    chi2_m = 0.0
    chi2_l = 0.0
    n = 0
    for ell, dp, dlo, dhi in zip(b_ells, b_D, b_lo, b_hi):
        dm = np.interp(ell, ells_model, D_model)
        dl = np.interp(ell, ells_lcdm, D_lcdm)
        sigma = (dlo + dhi) / 2
        chi2_m += ((dm - dp) / sigma) ** 2
        chi2_l += ((dl - dp) / sigma) ** 2
        n += 1
    return chi2_m, chi2_l, n


def print_chi2_table(D_model, ells_model, D_lcdm, ells_lcdm, label="Config"):
    print(f"  χ² vs Planck 2018 TT — {label}")
    print(f"  {'Dataset':<25} {'N':<6} {'Model':<10} {'LCDM':<10} {'Δχ²':<8} {'χ²/dof':<8}")
    print(f"  {'-'*67}")

    datasets = [
        ("Commander ℓ=2-29", chi2_commander),
        ("Unbinned ℓ=2-2508", chi2_unbinned),
        ("Binned ℓ=47-2500", chi2_binned),
    ]
    for name, fn in datasets:
        cm, cl, n = fn(D_model, ells_model, D_lcdm, ells_lcdm)
        dof = max(n - 1, 1)
        print(f"  {name:<25} {n:<6} {cm:<10.2f} {cl:<10.2f} {cm-cl:<+8.2f} {cm/dof:<8.3f}")
    print(f"  {'-'*67}")


def run_camb_from_config(phi0, y0, n_star):
    import json

    from scripts.camb_wrapper import compute_cl_full_camb, compute_cl_camb_powerlaw
    from scripts.plotting import find_ps

    ps_path, ps_md = find_ps(phi0, y0, n_star)
    if ps_path is None:
        raise FileNotFoundError(f"No PS file for phi0={phi0}, y0={y0}, Nstar={n_star}")

    with open(ps_path) as f:
        raw = json.load(f)
    spec = raw["spectrum"]
    ps_data = {"k_phys": spec["k_phys"], "P_S": spec["P_S"]}

    ells, C_tt, _, _ = compute_cl_full_camb(ps_data, ell_max=2500)
    D_model = C_ell_to_d_ell(ells, C_tt)
    ells_l, C_tt_l, _, _ = compute_cl_camb_powerlaw(ell_max=2500)
    D_lcdm = C_ell_to_d_ell(ells_l, C_tt_l)

    return D_model, ells, D_lcdm, ells_l


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--phi0", type=float, default=6.60)
    parser.add_argument("--y0", type=float, default=-0.736)
    parser.add_argument("--nstar", type=float, default=52.59)
    args = parser.parse_args()

    D_model, ells, D_lcdm, ells_l = run_camb_from_config(args.phi0, args.y0, args.nstar)
    print_chi2_table(D_model, ells, D_lcdm, ells_l,
                     label=f"φ₀={args.phi0} y₀={args.y0} N*={args.nstar}")
