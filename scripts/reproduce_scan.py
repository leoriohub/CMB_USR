"""
Reproduce a camb_scan phase 2 run with the Numba solver and compare chi2/d2.

Usage:
    python -m scripts.reproduce_scan <path_to_old_scan.jsonl> [--max N]

Requires the old JSONL from the lab machine (e.g., via scp or direct path).
Only compares entries with status="ok".
"""
import argparse
import json
import sys
import time

import numpy as np

from models import HiggsModel
from pspectrum_pipeline import run_pspectrum_pipeline, build_weighted_kgrid
from scripts.camb_wrapper import compute_cl_full_camb, compute_cl_camb_powerlaw
from scripts.planck_data import C_ell_to_d_ell, get_planck_data_asymmetric
from scripts.constants import As, k_pivot_phys


def compute_d2(ps_data):
    """Compute D_ell[2] from P_S data via CAMB."""
    ells, C_TT, _, _ = compute_cl_full_camb(ps_data, ell_max=2500)
    D = C_ell_to_d_ell(ells, C_TT)
    return float(D[0]), ells, D


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("jsonl_path", help="Path to old camb_scan JSONL")
    parser.add_argument("--max", type=int, default=None, help="Max configs to check")
    parser.add_argument("--workers", type=int, default=4, help="Pipeline workers")
    args = parser.parse_args()

    # Read header
    with open(args.jsonl_path) as f:
        header = json.loads(f.readline())
    if header.get("_type") != "header":
        print("ERROR: First line must be a header")
        sys.exit(1)

    k_phys = np.array(header["k_phys"])
    print(f"Header: {len(k_phys)} k-modes, xi={header.get('xi')}, lam={header.get('lam')}")
    print(f"  k_min={header.get('k_min')}, k_max={header.get('k_max')}")
    print(f"  ell_max={header.get('ell_max')}")

    # Read entries
    entries = []
    with open(args.jsonl_path) as f:
        f.readline()  # skip header
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("status") == "ok":
                entries.append(rec)

    if args.max:
        entries = entries[:args.max]

    print(f"\nComparing {len(entries)} configs...")
    print(f"{'#':>4} {'phi0':>6} {'y0':>8} {'N*':>6} {'chi2_old':>10} {'chi2_new':>10} "
          f"{'dchi2':>8} {'d2_old':>7} {'d2_new':>7} {'dd2':>7} {'match':>8}")
    print("-" * 85)

    max_dchi2 = 0.0
    max_dd2 = 0.0
    m = HiggsModel(lam=header.get("lam", 0.13), xi=header.get("xi", 15000.0))

    # Warmup Numba
    print(" Warming up...", end=" ", flush=True)
    _ = run_pspectrum_pipeline(
        m, phi0=6.0, y0=-0.2, N_star=50.0,
        k_phys_grid=k_phys[:5], use_numba=True,
        n_workers=1, save_outputs=False,
    )
    print("done.")

    t_start = time.time()
    for i, rec in enumerate(entries):
        phi0 = rec["phi0"]
        y0 = rec["y0"]
        N_star = rec["N_star"]
        chi2_old = rec.get("chi2", 0)
        d2_old = rec.get("d2", 0)

        t0 = time.time()
        result = run_pspectrum_pipeline(
            model=m, phi0=phi0, y0=y0, N_star=N_star,
            k_pivot_phys=k_pivot_phys, k_phys_grid=k_phys,
            normalize_to_As=True, As=As,
            n_workers=args.workers, use_numba=True,
            ms_steps=5000, save_outputs=False,
        )
        if result["status"] != "success":
            print(f"  {i:4d} {phi0:6.2f} {y0:8.3f} {N_star:6.1f}  "
                  f"PIPE FAIL: {result.get('message','?')}")
            continue

        ps_data = {"k_phys": result["k_phys"], "P_S": result["P_S"]}
        try:
            ells_c, C_TT, _, _ = compute_cl_full_camb(ps_data, ell_max=2500)
            D_new = C_ell_to_d_ell(ells_c, C_TT)
            d2_new = float(D_new[0])

            # LCDM baseline for full chi2
            ells_l, C_l, _, _ = compute_cl_camb_powerlaw(ell_max=2500)
            D_l = C_ell_to_d_ell(ells_l, C_l)

            # Full chi2 (same method as original scan)
            from scripts.chi2_analysis import chi2_unbinned
            chi2_new, chi2_lcdm_new, _ = chi2_unbinned(D_new, ells_c, D_l, ells_l)
        except Exception as e:
            print(f"  {i:4d} {phi0:6.2f} {y0:8.3f} {N_star:6.1f}  "
                  f"CAMB FAIL: {e}")
            continue

        chi2_diff = abs(chi2_old - chi2_new)
        dd2 = abs(d2_old - d2_new)

        # Track max deviations
        if chi2_diff > max_dchi2:
            max_dchi2 = chi2_diff
        if dd2 > max_dd2:
            max_dd2 = dd2

        if (i + 1) % 25 == 0 or i == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / max(elapsed, 0.01)
            eta = (len(entries) - i - 1) / rate
            print(f"  [{i+1}/{len(entries)}] {elapsed:.0f}s elapsed, ETA {eta:.0f}s",
                  flush=True)

    t_total = time.time() - t_start
    print(f"\nTotal: {len(entries)} configs in {t_total:.0f}s ({t_total/len(entries):.2f}s/config)")
    print(f"Max |chi2_diff| = {max_dchi2:.4f}")
    print(f"Max |d2_diff|   = {max_dd2:.4f} uK^2")

    if max_dchi2 < 200.0 and max_dd2 < 2.0:
        print(f"PASS: chi2 diff < 200, d2 diff < 2 uK^2.")
        sys.exit(0)
    else:
        print("FAIL: Deviations exceed tolerance.")
        sys.exit(1)


if __name__ == "__main__":
    main()
