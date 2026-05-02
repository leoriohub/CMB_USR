"""
Chi-squared analysis for Higgs USR vs Planck 2018 low-ℓ data.

Computes:
- chi^2 for the Higgs USR model
- chi^2 for the LCDM baseline (power-law PS)
- Delta chi^2 per ell bin
- Net improvement/deterioration signalled by USR ringing
"""
import json
import os
import sys

import numpy as np

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

from scripts.camb_wrapper import compute_cl_sw, compute_cl_sw_powerlaw
from scripts.pspectrum_pipeline import load_pspectrum
from scripts.planck_data import get_planck_data, C_ell_to_d_ell


def compute_chi2(ells_model, C_ell_model, ells_data, D_ell_data, D_ell_err,
                 Tcmb=2.7255):
    """Diagonal chi^2 comparing model C_ell to Planck binned D_ell."""
    D_ell_model = C_ell_to_d_ell(ells_model, C_ell_model, Tcmb=Tcmb)
    chi2 = np.sum(((D_ell_model - D_ell_data) / D_ell_err) ** 2)
    return chi2, D_ell_model


def analyse(pspectrum_path, ell_max=30, r_ls=14000.0, verbose=True):
    """Full chi^2 analysis for a given P_S(k) file vs Planck low-ell."""
    data = load_pspectrum(pspectrum_path)
    meta = data["metadata"]
    k_phys = data["k_phys"]
    P_S = data["P_S"]

    ells_planck, D_planck, D_err_planck = get_planck_data()
    ells_exp = min(ell_max, ells_planck[-1])
    mask = ells_planck <= ells_exp
    ells_p = ells_planck[mask]
    D_p = D_planck[mask]
    D_err_p = D_err_planck[mask]

    ells_all, C_ell_all = compute_cl_sw(data, ell_max=ells_exp, r_ls=r_ls)
    ells_USR = ells_all[ells_all >= 2]
    C_ell_USR = C_ell_all[ells_all >= 2]
    _, D_ell_usr = compute_chi2(ells_USR, C_ell_USR, ells_p, D_p, D_err_p)
    chi2_usr = float(np.sum(((D_ell_usr - D_p) / D_err_p) ** 2))

    _, C_ell_lcdm_all, _ = compute_cl_sw_powerlaw(
        k_min=k_phys[0], k_max=k_phys[-1], ell_max=ells_exp, r_ls=r_ls
    )
    ells_LCDM = ells_all[ells_all >= 2]
    C_ell_LCDM = C_ell_lcdm_all[ells_all >= 2]
    _, D_ell_lcdm = compute_chi2(ells_LCDM, C_ell_LCDM, ells_p, D_p, D_err_p)
    chi2_lcdm = float(np.sum(((D_ell_lcdm - D_p) / D_err_p) ** 2))

    delta_chi2 = chi2_usr - chi2_lcdm
    dof = len(ells_p)
    chi2_reduced_usr = chi2_usr / dof
    chi2_reduced_lcdm = chi2_lcdm / dof

    if verbose:
        print(f"\n{'='*60}")
        print(f"Chi2 Analysis: {meta['model']}, "
              f"phi0={meta['phi0']}, yi={meta['yi']}, xi={meta.get('xi','N/A')}")
        print(f"N_total={meta['N_total']:.2f}, N_pivot={meta['N_pivot']:.0f}")
        print(f"{'='*60}")
        print(f"{'ell':>4s}  {'D_Planck':>10s}  {'D_USR':>10s}  {'D_LCDM':>10s}  {'Pull_USR':>8s}")
        for j in range(len(ells_p)):
            pull = (D_ell_usr[j] - D_p[j]) / D_err_p[j]
            print(f"{ells_p[j]:4d}  {D_p[j]:10.1f}  {D_ell_usr[j]:10.1f}  "
                  f"{D_ell_lcdm[j]:10.1f}  {pull:8.2f}")
        print(f"{'='*60}")
        print(f"chi2_USR   = {chi2_usr:.3f}  (reduced = {chi2_reduced_usr:.3f})")
        print(f"chi2_LCDM  = {chi2_lcdm:.3f}  (reduced = {chi2_reduced_lcdm:.3f})")
        print(f"Delta chi2 = {delta_chi2:+.3f}")
        sign = "worse" if delta_chi2 > 0 else "better"
        print(f"Higgs USR is {sign} than LCDM by dchi2 = {delta_chi2:+.3f}")
        print(f"{'='*60}\n")

    return dict(
        model=meta["model"],
        phi0=meta["phi0"],
        yi=meta["yi"],
        xi=meta.get("xi"),
        N_total=meta["N_total"],
        N_pivot=meta["N_pivot"],
        ell_max=ell_max,
        dof=dof,
        chi2_usr=chi2_usr,
        chi2_lcdm=chi2_lcdm,
        delta_chi2=delta_chi2,
        chi2_reduced_usr=chi2_reduced_usr,
        chi2_reduced_lcdm=chi2_reduced_lcdm,
        ells=ells_p.tolist(),
        D_planck=D_p.tolist(),
        D_err=D_err_p.tolist(),
        D_ell_usr=D_ell_usr.tolist(),
        D_ell_lcdm=D_ell_lcdm.tolist(),
    )


def save_chi2_results(result, output_dir="outputs/cmb_results/chi2"):
    os.makedirs(output_dir, exist_ok=True)
    safe = f"Higgs_Inflation_phi{result['phi0']:.2f}_yi{result['yi']:.3f}"
    path = os.path.join(output_dir, f"chi2_{safe}.json")
    with open(path, "w") as f:
        json.dump(result, f, indent=2)
    return path


if __name__ == "__main__":
    import argparse
    import glob

    parser = argparse.ArgumentParser()
    parser.add_argument("pspectrum_path",
                        help="Path to P_S(k) JSON file or directory")
    parser.add_argument("--ell-max", type=int, default=29)
    parser.add_argument("--r-ls", type=float, default=14000.0)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()

    if os.path.isdir(args.pspectrum_path):
        files = sorted(glob.glob(os.path.join(args.pspectrum_path, "*.json")))
    else:
        files = [args.pspectrum_path]

    for fp in files:
        res = analyse(fp, ell_max=args.ell_max, r_ls=args.r_ls)
        if args.save:
            out = save_chi2_results(res)
            print(f"Saved: {out}")
