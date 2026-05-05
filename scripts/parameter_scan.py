"""
Grid scan over (phi0, y0) for Higgs USR model, evaluating chi^2 vs Planck low-ell.

For each point on the grid:
    1. Run the P_S(k) pipeline
    2. Compute Sachs-Wolfe C_ell
    3. Compute chi^2 against Planck 2018 binned low-ell data
    4. Compare to LCDM power-law baseline chi^2

Results are saved as JSON for analysis in notebooks.
"""

import argparse
import json
import os
import sys
import time

ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed

from scripts.constants import As, k_pivot_phys, N_star_default, ell_max_default, r_ls
from scripts.pspectrum_pipeline import run_pspectrum_pipeline
from scripts.sachs_wolfe import compute_cl_sw, compute_cl_sw_powerlaw
from scripts.planck_data import get_planck_data, C_ell_to_d_ell
from models import HiggsModel


def run_single_point(phi0, y0, xi, lam, num_k, k_pivot_phys, N_star, As, ell_max, r_ls):
    """Run the full pipeline for a single (phi0, y0) and return chi^2 results."""
    model = HiggsModel(lam=lam, xi=xi)
    result = run_pspectrum_pipeline(
        model=model, phi0=phi0, y0=y0,
        k_min=1e-5, k_max=1.0, num_k=num_k,
        k_pivot_phys=k_pivot_phys, N_star=N_star,
        normalize_to_As=True, As=As, save_outputs=False,
    )
    if result["status"] != "success":
        return {
            "phi0": phi0, "y0": y0, "status": "error",
            "message": result.get("message", "unknown"),
        }

    k_phys = result["k_phys"]
    meta = result["metadata"]

    ells_planck, D_planck, D_err = get_planck_data()
    mask = ells_planck <= ell_max
    ells_p = ells_planck[mask]
    D_p = D_planck[mask]
    D_err_p = D_err[mask]

    ells_all, C_ell_all = compute_cl_sw(result, ell_max=ell_max, r_ls=r_ls)
    ells_usr = ells_all[ells_all >= 2]
    C_ell_usr = C_ell_all[ells_all >= 2]
    D_ell_usr = C_ell_to_d_ell(ells_usr, C_ell_usr)

    _, C_ell_lcdm_all, _ = compute_cl_sw_powerlaw(
        k_min=k_phys[0], k_max=k_phys[-1], ell_max=ell_max, r_ls=r_ls,
    )
    ells_lcdm = ells_all[ells_all >= 2]
    C_ell_lcdm = C_ell_lcdm_all[ells_all >= 2]
    D_ell_lcdm = C_ell_to_d_ell(ells_lcdm, C_ell_lcdm)

    D_usr_interp = np.interp(ells_p, ells_usr, D_ell_usr)
    D_lcdm_interp = np.interp(ells_p, ells_lcdm, D_ell_lcdm)

    chi2_usr = float(np.sum(((D_usr_interp - D_p) / D_err_p) ** 2))
    chi2_lcdm = float(np.sum(((D_lcdm_interp - D_p) / D_err_p) ** 2))
    delta_chi2 = chi2_usr - chi2_lcdm
    dof = len(ells_p)

    return {
        "phi0": phi0,
        "y0": y0,
        "status": "success",
        "N_total": meta["N_total"],
        "chi2_usr": chi2_usr,
        "chi2_lcdm": chi2_lcdm,
        "delta_chi2": delta_chi2,
        "chi2_reduced_usr": chi2_usr / dof,
        "chi2_reduced_lcdm": chi2_lcdm / dof,
        "dof": dof,
    }


def main():
    parser = argparse.ArgumentParser(description="Grid scan over (phi0, y0) for Higgs USR model.")
    parser.add_argument("--xi", type=float, default=15000.0)
    parser.add_argument("--lam", type=float, default=0.13)
    parser.add_argument("--phi0-min", type=float, default=5.4)
    parser.add_argument("--phi0-max", type=float, default=6.4)
    parser.add_argument("--phi0-step", type=float, default=0.1)
    parser.add_argument("--y0", type=float, nargs="+",
                        default=[-0.001, -0.01, -0.03, -0.05, -0.07, -0.10, -0.12, -0.15],
                        help="y0 values (e.g. --y0 -0.001 -0.10 -0.03)")
    parser.add_argument("--num-k", type=int, default=100)
    parser.add_argument("--k-pivot-phys", type=float, default=k_pivot_phys)
    parser.add_argument("--N-star", type=float, default=N_star_default)
    parser.add_argument("--As", type=float, default=As)
    parser.add_argument("--ell-max", type=int, default=ell_max_default)
    parser.add_argument("--r-ls", type=float, default=r_ls)
    parser.add_argument("--n-workers", type=int, default=1)
    parser.add_argument("--output-dir", default="outputs/cmb_results/scans")
    args = parser.parse_args()

    y0_vals = args.y0
    phi0_vals = np.round(np.arange(args.phi0_min, args.phi0_max + args.phi0_step / 2, args.phi0_step), 2)

    grid = [(float(p), float(y)) for p in phi0_vals for y in y0_vals]
    total = len(grid)
    results = []

    print(f"Scanning {total} points ({len(phi0_vals)}x phi0 × {len(y0_vals)} y0) on {args.n_workers} workers...")

    t0 = time.time()
    if args.n_workers > 1:
        with ProcessPoolExecutor(max_workers=args.n_workers) as executor:
            futures = {
                executor.submit(run_single_point, p, y, args.xi, args.lam, args.num_k,
                                args.k_pivot_phys, args.N_star, args.As,
                                args.ell_max, args.r_ls): (p, y)
                for p, y in grid
            }
            for i, future in enumerate(as_completed(futures), 1):
                r = future.result()
                results.append(r)
                p, y = futures[future]
                if r["status"] == "success":
                    print(f"[{i}/{total}] {p:.2f} {y:.3f} -> chi2={r['chi2_usr']:.1f} delta={r['delta_chi2']:+.1f}")
                else:
                    print(f"[{i}/{total}] {p:.2f} {y:.3f} -> error: {r.get('message', '?')}")
    else:
        for i, (p, y) in enumerate(grid, 1):
            r = run_single_point(p, y, args.xi, args.lam, args.num_k,
                                 args.k_pivot_phys, args.N_star, args.As,
                                 args.ell_max, args.r_ls)
            results.append(r)
            if r["status"] == "success":
                print(f"[{i}/{total}] {p:.2f} {y:.3f} -> chi2={r['chi2_usr']:.1f} delta={r['delta_chi2']:+.1f}")
            else:
                print(f"[{i}/{total}] {p:.2f} {y:.3f} -> error: {r.get('message', '?')}")

    elapsed = time.time() - t0
    n_success = sum(1 for r in results if r["status"] == "success")
    n_error = total - n_success
    print(f"\nDone in {elapsed:.0f}s — {n_success} succeeded, {n_error} failed")

    phi0_vals_list = [float(v) for v in phi0_vals]
    output = {
        "config": {
            "xi": args.xi,
            "lam": args.lam,
            "N_star": args.N_star,
            "As": args.As,
            "num_k": args.num_k,
            "ell_max": args.ell_max,
            "r_ls": args.r_ls,
            "phi0_vals": phi0_vals_list,
            "y0_vals": y0_vals,
        },
        "results": results,
    }

    os.makedirs(args.output_dir, exist_ok=True)
    out_path = os.path.join(args.output_dir, "parameter_scan.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"Saved -> {out_path}")


if __name__ == "__main__":
    main()
