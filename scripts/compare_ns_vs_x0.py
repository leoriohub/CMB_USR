#!/usr/bin/env python3
"""
n_s extraction comparison: LSQ vs log-derivative across ALL cached P_S(k) files.

Reads every existing P_S(k) JSON from outputs/simulations/pspectra/,
computes n_s both ways (LSQ and derivative) at k_pivot, and produces:
  - Table summary (stdout)
  - Scatter plot: outputs/plots/paper/ns_method_comparison.png

Usage:
    python scripts/compare_ns_vs_x0.py
    python scripts/compare_ns_vs_x0.py --k-pivot 0.05 --ns-window 3.0
    python scripts/compare_ns_vs_x0.py --dir /custom/path --output my_plot.png
"""

import argparse
import glob
import json
import os
import sys
import warnings

import numpy as np

warnings.filterwarnings("ignore", category=RuntimeWarning)
os.environ["OMP_NUM_THREADS"] = "1"

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.interpolate import CubicSpline

from scripts.plotting import TOL, PAPER_RCPARAMS, OUTPUT_DIRS, save_fig, get_path
from scripts.constants import k_pivot_phys, As
from scripts.observables import extract_ns


def load_ps_json(path):
    """Load a P_S(k) JSON file, return (k_phys, P_S, metadata) or (None, None, None)."""
    try:
        with open(path) as f:
            rec = json.load(f)
    except (json.JSONDecodeError, OSError) as e:
        print(f"  WARNING: can't read {path}: {e}")
        return None, None, None

    spec = rec.get("spectrum", {})
    k_phys = np.array(spec.get("k_phys", []), dtype=float)
    P_S = np.array(spec.get("P_S", []), dtype=float)
    meta = rec.get("metadata", {})

    # Handle partial checkpoints
    if spec.get("partial", False):
        n_comp = spec.get("n_completed", 0)
        if n_comp < 10:
            return None, None, None

    if len(k_phys) < 5 or len(P_S) < 5:
        return None, None, None

    # Filter to finite values only
    ok = np.isfinite(P_S) & (P_S > 1e-30) & np.isfinite(k_phys)
    if ok.sum() < 5:
        return None, None, None
    return k_phys[ok], P_S[ok], meta


def main():
    parser = argparse.ArgumentParser(
        description="n_s extraction comparison: LSQ vs log-derivative across all cached P_S(k)"
    )
    parser.add_argument("--k-pivot", type=float, default=k_pivot_phys,
                        help="Pivot wavenumber (default: %(default)s)")
    parser.add_argument("--ns-window", type=float, default=4.0,
                        help="LSQ fit half-width [k_p/w, k_p*w] (default: %(default)s)")
    parser.add_argument("--dir", type=str, default=None,
                        help="P_S(k) JSON directory (default: outputs/simulations/pspectra/)")
    parser.add_argument("--output", type=str, default="ns_method_comparison",
                        help="Output filename (no extension, default: %(default)s)")
    args = parser.parse_args()

    # Resolve P_S directory
    if args.dir is not None:
        ps_dir = args.dir
    else:
        ps_dir = get_path("pspectra", "")
    if not os.path.isdir(ps_dir):
        print(f"ERROR: directory not found: {ps_dir}")
        sys.exit(1)

    # Find all JSON files (both new and legacy naming conventions)
    json_files = sorted(glob.glob(os.path.join(ps_dir, "ps_*.json")))
    json_files += sorted(glob.glob(os.path.join(ps_dir, "PS_Higgs_*.json")))
    json_files += sorted(glob.glob(os.path.join(ps_dir, "Higgs_Inflation_*.json")))
    json_files = sorted(set(json_files))  # deduplicate

    if not json_files:
        print(f"No P_S(k) JSON files found in {ps_dir}")
        sys.exit(1)

    print(f"Found {len(json_files)} P_S(k) file(s) in {ps_dir}")
    print(f"k_pivot = {args.k_pivot}, ns_window = {args.ns_window}")
    print()

    # Process each file
    rows = []
    for fpath in json_files:
        k_phys, P_S, meta = load_ps_json(fpath)
        if k_phys is None:
            continue

        fname = os.path.basename(fpath)

        # Extract model parameters from metadata
        x0 = meta.get("x0", meta.get("phi0", None))
        y0 = meta.get("y0", None)
        N_star = meta.get("N_star", None)
        N_total = meta.get("N_total", None)

        # Derive model label from filename or metadata
        label = meta.get("model", "")
        if x0 is not None and y0 is not None:
            label += f" x₀={x0:.3g} y₀={y0:.3g}"
            if N_star is not None:
                label += f" N*={N_star:.1f}"

        # n_s via LSQ
        ns_lsq, meta_lsq = extract_ns(k_phys, P_S, args.k_pivot,
                                      ns_window=args.ns_window, method="lsq")

        # n_s via log derivative
        ns_der, meta_der = extract_ns(k_phys, P_S, args.k_pivot,
                                      method="derivative")

        if ns_lsq is None and ns_der is None:
            continue

        delta = (ns_der - ns_lsq) if (ns_der is not None and ns_lsq is not None) else None
        k_range_str = meta_der.get("k_range", ["?", "?"])
        k_range_str = f"[{k_range_str[0]:.2e}, {k_range_str[1]:.2e}]"

        rows.append({
            "file": fname,
            "label": label,
            "x0": x0,
            "y0": y0,
            "N_star": N_star,
            "N_total": N_total,
            "ns_lsq": ns_lsq,
            "ns_der": ns_der,
            "delta": delta,
            "n_modes_lsq": meta_lsq.get("n_modes", 0) if meta_lsq else 0,
            "n_modes_der": meta_der.get("n_modes", 0) if meta_der else 0,
            "k_range": k_range_str,
        })

    if not rows:
        print("No valid P_S(k) files could be processed (k_pivot may be outside grid).")
        sys.exit(1)

    # ── Print summary table ──
    print(f"{'File':<50} {'n_s(LSQ)':>10} {'n_s(Der)':>10} {'Δ':>10} {'n_modes':>8} {'N*':>7}")
    print("-" * 100)
    for r in rows:
        ns_lsq_str = f"{r['ns_lsq']:.4f}" if r['ns_lsq'] is not None else "FAIL"
        ns_der_str = f"{r['ns_der']:.4f}" if r['ns_der'] is not None else "FAIL"
        delta_str = f"{r['delta']:+.5f}" if r['delta'] is not None else "N/A"
        n_modes_str = f"{r['n_modes_der']}"
        nstar_str = f"{r['N_star']:.1f}" if r['N_star'] is not None else "?"
        print(f"{r['file']:<50} {ns_lsq_str:>10} {ns_der_str:>10} {delta_str:>10} "
              f"{n_modes_str:>8} {nstar_str:>7}")

    # ── Summary stats ──
    valid_deltas = [r['delta'] for r in rows if r['delta'] is not None]
    if valid_deltas:
        deltas = np.array(valid_deltas)
        print(f"\nSummary: {len(valid_deltas)} valid comparisons")
        print(f"  Δ (der - lsq):  mean={np.mean(deltas):+.6f}  "
              f"std={np.std(deltas):.6f}  "
              f"max|Δ|={np.max(np.abs(deltas)):.6f}")
        n_above_1pct = int(np.sum(np.abs(deltas) > 0.01))
        print(f"  |Δ| > 0.01: {n_above_1pct}/{len(deltas)} "
              f"({100*n_above_1pct/len(deltas):.1f}%)")

    # ── Plot: LSQ vs derivative scatter ──
    valid = [r for r in rows if r['ns_lsq'] is not None and r['ns_der'] is not None]
    if valid:
        ns_lsqs = np.array([r['ns_lsq'] for r in valid])
        ns_ders = np.array([r['ns_der'] for r in valid])
        deltas = ns_ders - ns_lsqs

        with plt.rc_context(PAPER_RCPARAMS):
            fig, (ax_sc, ax_hist) = plt.subplots(
                1, 2, figsize=(5.5, 2.8),
                gridspec_kw={"width_ratios": [2, 1], "wspace": 0.35},
            )

            # Left: scatter
            lims = [min(ns_lsqs.min(), ns_ders.min()) * 0.998,
                    max(ns_lsqs.max(), ns_ders.max()) * 1.002]
            ax_sc.plot(lims, lims, "--", color=TOL["grey"], lw=0.6, alpha=0.5,
                       label="1:1")
            ax_sc.scatter(ns_lsqs, ns_ders, c=TOL["red"], s=10, alpha=0.7,
                          edgecolors="none", zorder=3)
            ax_sc.set_xlabel(r"$n_s$ (LSQ fit)", fontsize=9)
            ax_sc.set_ylabel(r"$n_s$ (derivative)", fontsize=9)
            ax_sc.set_xlim(lims)
            ax_sc.set_ylim(lims)
            ax_sc.set_aspect("equal")
            ax_sc.grid(True, alpha=0.2, which="both")

            # Right: histogram of differences
            ax_hist.hist(deltas, bins=max(10, len(deltas) // 3),
                         color=TOL["blue"], alpha=0.7, edgecolor="none")
            ax_hist.axvline(0, color=TOL["grey"], ls="--", lw=0.6, alpha=0.5)
            ax_hist.set_xlabel(r"$\Delta n_s$ (der $-$ lsq)", fontsize=9)
            ax_hist.set_ylabel("Count", fontsize=9)
            ax_hist.grid(True, alpha=0.2, which="both")

            # Annotation
            ax_hist.text(
                0.95, 0.95,
                f"N={len(deltas)}\n"
                f"μ={np.mean(deltas):+.5f}\n"
                f"σ={np.std(deltas):.5f}",
                transform=ax_hist.transAxes, fontsize=6, ha="right", va="top",
                bbox=dict(boxstyle="round,pad=0.2", fc="w", ec="none", alpha=0.7),
            )

            save_fig(fig, args.output, "paper")

    print(f"\nPlot: {get_path('paper', args.output + '.png')}")


if __name__ == "__main__":
    main()
