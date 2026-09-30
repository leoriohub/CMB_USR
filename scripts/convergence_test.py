"""Numerical convergence test for the CMB power dip (Prof. Cárdenas correction 5).

Runs the reference Higgs USR trajectory (phi0, y0, N_star) at multiple k-grid
resolutions and code-time scalings S, extracts (D2, k_dip, chi2_lowell) per run,
and reports D2 stability relative to the production grid (grid-300).

Reuses existing pipeline modules only (AGENTS.md §11 — no root-solver changes).

CLI:
    python scripts/convergence_test.py --phi0 5.75 --y0 -0.170 --nstar 55 \
        --k-pivot 0.002 --backend fortran [--out PATH]

Output:
    JSON record at outputs/simulations/logs/convergence_phi5.75_y0-0.170_nstar55.0.json
    (or --out), plus a ready-to-paste LaTeX table on stdout.
"""

import argparse
import json
import os
import sys
import time

import numpy as np

from models.higgs import HiggsModel
from pspectrum_pipeline import build_weighted_kgrid, run_pspectrum_pipeline
from scripts.camb_wrapper import compute_chi2_camb, compute_cl_full_camb
from scripts.constants import As
from scripts.planck_data import C_ell_to_d_ell
from scripts.plotting import get_path

# Production grid settings (scripts/run_full_analysis.py): weighted grid
# n_dense=200 / n_outer=100 over [1e-4, 1e-2] dense window, S=5e-5, y0=-0.170,
# T_max=500, bg_steps=1000, ms_steps=5000.
PROD_S = 5e-5
PROD_T_MAX = 500.0
PROD_BG_STEPS = 1000
PROD_MS_STEPS = 5000

# Run matrix: increasing k-grid resolution at the fiducial S, plus an S-sweep.
# S-sweep rescales y0 and T_max to hold the physical trajectory fixed:
#   y0_new = y0_ref * (S_ref / S),  T_max_new = T_max_ref * (S / S_ref)
ROWS = [
    {"label": "grid-100",  "n_dense": 67,   "n_outer": 33,   "S": 5e-5,   "y0_used": -0.170, "T_max": 500.0},
    {"label": "grid-150",  "n_dense": 100,  "n_outer": 50,   "S": 5e-5,   "y0_used": -0.170, "T_max": 500.0},
    {"label": "grid-300",  "n_dense": 200,  "n_outer": 100,  "S": 5e-5,   "y0_used": -0.170, "T_max": 500.0},
    {"label": "grid-500",  "n_dense": 334,  "n_outer": 166,  "S": 5e-5,   "y0_used": -0.170, "T_max": 500.0},
    {"label": "grid-750",  "n_dense": 500,  "n_outer": 250,  "S": 5e-5,   "y0_used": -0.170, "T_max": 500.0},
    {"label": "grid-1000", "n_dense": 667,  "n_outer": 333,  "S": 5e-5,   "y0_used": -0.170, "T_max": 500.0},
    {"label": "grid-1500", "n_dense": 1000, "n_outer": 500,  "S": 5e-5,   "y0_used": -0.170, "T_max": 500.0},
    {"label": "grid-2000", "n_dense": 1334, "n_outer": 666,  "S": 5e-5,   "y0_used": -0.170, "T_max": 500.0},
    {"label": "grid-3000", "n_dense": 2000, "n_outer": 1000, "S": 5e-5,   "y0_used": -0.170, "T_max": 500.0},
    {"label": "grid-4000", "n_dense": 2667, "n_outer": 1333, "S": 5e-5,   "y0_used": -0.170, "T_max": 500.0},
    {"label": "S-quarter", "n_dense": 200,  "n_outer": 100,  "S": 1.25e-5, "y0_used": -0.680, "T_max": 125.0},
    {"label": "S-half",    "n_dense": 200,  "n_outer": 100,  "S": 2.5e-5, "y0_used": -0.340, "T_max": 250.0},
    {"label": "S-double",  "n_dense": 200,  "n_outer": 100,  "S": 1e-4,   "y0_used": -0.085, "T_max": 1000.0},
    {"label": "S-quad",    "n_dense": 200,  "n_outer": 100,  "S": 2e-4,   "y0_used": -0.0425, "T_max": 2000.0},
]

REF_LABEL = "grid-300"


def run_row(row, args):
    """Run the full pipeline once and extract the convergence observables."""
    model = HiggsModel(lam=0.13, xi=15000.0)
    # run_pspectrum_pipeline mutates model.x0/y0 — re-set S and y0 each iteration
    model.S = row["S"]
    model.x0 = args.phi0
    model.y0 = row["y0_used"]

    k_grid = build_weighted_kgrid(
        k_min=1e-5, k_max=1.0, k_pivot_phys=args.k_pivot,
        dense_min=1e-4, dense_max=1e-2,
        n_dense=row["n_dense"], n_outer=row["n_outer"],
    )

    t0 = time.time()
    result = run_pspectrum_pipeline(
        model=model,
        phi0=args.phi0,
        y0=row["y0_used"],
        N_star=args.nstar,
        k_phys_grid=k_grid,
        ms_steps=PROD_MS_STEPS,
        T_max=row["T_max"],
        bg_steps=PROD_BG_STEPS,
        normalize_to_As=True,
        As=As,
        k_pivot_phys=args.k_pivot,
        save_outputs=False,
        n_workers=max(1, os.cpu_count() or 1),
        backend=args.backend,
    )
    elapsed = time.time() - t0

    if result["status"] != "success":
        print(f"  ERROR: pipeline failed: {result.get('message')}", file=sys.stderr)
        sys.exit(1)

    n_modes = int(result["metadata"]["num_k"])
    n_completed = int(result["metadata"]["n_completed"])
    if n_completed < 0.9 * n_modes:
        print(
            f"  ERROR: {row['label']}: only {n_completed}/{n_modes} modes completed "
            "(< 90%). Aborting — do not emit a partial table.",
            file=sys.stderr,
        )
        sys.exit(1)

    ells, C_TT, _, _ = compute_cl_full_camb(result, ell_max=2500)
    D_ell = C_ell_to_d_ell(ells, C_TT)
    D2 = float(D_ell[0])  # ells start at 2 -> index 0 is ell=2

    k_dip = float(result["k_phys"][np.nanargmin(result["P_S"])])

    chi2_model, chi2_lcdm, delta_chi2 = compute_chi2_camb(result, ell_max=29)

    return {
        "label": row["label"],
        "n_modes": n_modes,
        "S": row["S"],
        "y0_used": row["y0_used"],
        "T_max": row["T_max"],
        "D2": D2,
        "k_dip": k_dip,
        "chi2_lowell": float(chi2_model),
        "delta_chi2": float(delta_chi2),
        "n_completed": n_completed,
        "elapsed_s": round(elapsed, 1),
    }


def fmt_s(x):
    """Format S for LaTeX, e.g. 5e-5 -> $5\\times10^{-5}$."""
    exp = int(np.floor(np.log10(x)))
    mant = x / 10.0 ** exp
    return f"${mant:g}\\times10^{{{exp}}}$"


def fmt_sci(x, sig=2):
    """Format a float in scientific LaTeX with `sig` significant digits."""
    mant, exp = f"{x:.{sig - 1}e}".split("e")
    return f"${mant}\\times10^{{{int(exp)}}}$"


def print_latex_table(runs):
    """Print a ready-to-paste LaTeX table."""
    print("\n--- LaTeX table (ready to paste) ---")
    print(r"\begin{table*}[htbp]")
    print(r"\centering")
    print(r"\caption{Numerical convergence of the reference trajectory "
          r"($x_0 = 5.75$, $y_0 = -0.170$, $N_* = 55$). "
          r"Rows 1--10: increasing $k$-grid resolution at the fiducial code-time scaling "
          r"$S = 5\times10^{-5}$; rows 11--14: variation of $S$ at fixed resolution, with "
          r"$y_0$ and $T_{\max}$ rescaled to hold the physical trajectory fixed. $S$ is a "
          r"purely numerical convention ($T = S\,t$ with $t$ the physical time, chosen so "
          r"that the ODE variables remain $O(1)$ during inflation); it is not a physics "
          r"parameter, so the observables must be independent of it. $D_2$ is the quadrupole "
          r"from the CAMB-evaluated $C_\ell$, $k_{\rm dip}$ the location of the power dip "
          r"(in units of $10^{-4}\,\mathrm{Mpc}^{-1}$), and $\Delta\chi^2_{\rm low\,\ell}$ "
          r"the low-$\ell$ ($\ell \leq 29$) fit relative to $\Lambda$CDM.}")
    print(r"\setlength{\tabcolsep}{22.1pt}")
    print(r"\begin{tabular}{l c c c c c}")
    print(r"\hline\hline")
    print(r"\rowcolor{tableheader}[\tabcolsep]")
    print(r"Label & $n_{\rm modes}$ & $S$ & $D_2$ [$\mu$K$^2$] & "
          r"$10^{4}\,k_{\rm dip}$ [Mpc$^{-1}$] & $\Delta\chi^2_{\rm low\,\ell}$ \\ \hline")
    for r in runs:
        print(f"{r['label']} & {r['n_modes']} & {fmt_s(r['S'])} & "
              f"{r['D2']:.4f} & {r['k_dip'] * 1e4:.3f} & {r['delta_chi2']:.3f} \\\\")
    print(r"\hline\hline")
    print(r"\end{tabular}")
    print(r"\label{tab:convergence}")
    print(r"\end{table*}")
    print("---")


def main():
    parser = argparse.ArgumentParser(
        description="Numerical convergence test (k-grid resolution + code-time scaling S)"
    )
    parser.add_argument("--phi0", type=float, default=5.75, help="Field initial value")
    parser.add_argument("--y0", type=float, default=-0.170, help="Reference velocity")
    parser.add_argument("--nstar", type=float, default=55.0, help="N_star for pivot alignment")
    parser.add_argument("--k-pivot", type=float, default=0.002, help="Pivot scale (Mpc^-1)")
    parser.add_argument("--backend", choices=["fortran", "numba"], default="fortran",
                        help="MS solver backend (default: fortran)")
    parser.add_argument("--out", type=str, default=None,
                        help="Output JSON path (default: outputs/simulations/logs/)")
    args = parser.parse_args()

    if args.out is None:
        args.out = get_path(
            "logs",
            f"convergence_phi{args.phi0:.2f}_y0{args.y0:+.3f}_nstar{args.nstar:.1f}.json",
        )

    print(f"=== CONVERGENCE TEST: phi0={args.phi0}, y0={args.y0}, N*={args.nstar}, "
          f"k_pivot={args.k_pivot}, backend={args.backend} ===")

    runs = []
    for row in ROWS:
        print(f"\n--- {row['label']}  (n_dense={row['n_dense']}, n_outer={row['n_outer']}, "
              f"S={row['S']:g}, y0={row['y0_used']}, T_max={row['T_max']:g}) ---")
        rec = run_row(row, args)
        print(f"  {rec['n_completed']}/{rec['n_modes']} modes, {rec['elapsed_s']}s | "
              f"D2={rec['D2']:.1f}, k_dip={rec['k_dip']:.3e}, "
              f"chi2_lowell={rec['chi2_lowell']:.2f}, delta_chi2={rec['delta_chi2']:.2f}")
        runs.append(rec)

    ref = next(r for r in runs if r["label"] == REF_LABEL)
    D2_ref = ref["D2"]
    devs = {r["label"]: 100.0 * abs(r["D2"] - D2_ref) / D2_ref for r in runs if r["label"] != REF_LABEL}
    max_dev = max(devs.values())
    worst = max(devs, key=devs.get)
    passed = max_dev < 1.0

    print(f"\n=== STABILITY (reference: {REF_LABEL}, D2={D2_ref:.1f}) ===")
    for label, d in devs.items():
        flag = " <- worst" if label == worst else ""
        print(f"  {label:>10}: |D2 dev| = {d:.3f}%{flag}")
    if passed:
        print(f"  PASS (D2 stable within 1%: max rel dev = {max_dev:.3f}%)")
    else:
        print(f"  FAIL (D2 NOT stable within 1%: max rel dev = {max_dev:.3f}% at {worst})")

    # Anchor check: grid-300 must match the production pipeline value.
    # (Historical note: the paper's previously quoted D2=677 was the D_ell
    # element at index 2, i.e. ell=4, not the quadrupole. The true quadrupole
    # from the current pipeline is ~776 uK^2.)
    print(f"\n  Anchor check: grid-300 D2={D2_ref:.1f} (current pipeline, ell=2), "
          f"k_dip={ref['k_dip']:.3e}")

    record = {
        "config": {
            "phi0": args.phi0,
            "y0": args.y0,
            "nstar": args.nstar,
            "k_pivot_phys": args.k_pivot,
        },
        "runs": runs,
        "stability": {
            "ref_label": REF_LABEL,
            "max_rel_dev_D2_pct": round(max_dev, 4),
            "pass_1pct": passed,
            "worst_row": worst,
        },
    }
    with open(args.out, "w") as f:
        json.dump(record, f, indent=2)
    print(f"\nJSON saved: {args.out}")

    print_latex_table(runs)


if __name__ == "__main__":
    main()
